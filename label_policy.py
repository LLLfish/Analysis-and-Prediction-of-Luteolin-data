"""Deterministic grouping and measured-label decisions for ChEMBL evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional

from evidence_builder import ActivityEvidence, load_entity_bundle, resolve_target_components


_ACCESSION = re.compile(r"^[A-Z0-9]{6}(?:[A-Z0-9]{4})?$")
_CHEMBL_ID = re.compile(r"^CHEMBL[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEDGER_CATEGORIES = (
    "retained", "duplicate", "excluded", "unverified", "conflict_candidate",
)
_POLICY_PATH = Path(__file__).resolve().parent / "policies" / "label_policy.json"
_POLICY_KEYS = {
    "schema_version", "policy_version", "entity_bundle_schema_version",
    "confirmed_source_kinds", "test_only_source_kinds", "entity_eligibility",
    "endpoints", "exact_relation", "minimum_confidence", "units_to_nanomolar",
    "thresholds", "assay_compatibility_fields",
}
_ENTITY_POLICY = {
    "target_type": "SINGLE PROTEIN",
    "organism": "Homo sapiens",
    "taxonomy_id": 9606,
    "component_type": "PROTEIN",
    "component_count": 1,
    "accession_pattern": _ACCESSION.pattern,
}
_ENDPOINT_POLICY = {"labeled": ["Ki", "Kd"], "independent_only": ["IC50"]}
_UNIT_POLICY = {
    "pM": "0.001", "nM": "1", "uM": "1000", "µM": "1000",
    "μM": "1000", "mM": "1000000", "M": "1000000000",
}
_ASSAY_FIELDS = {
    "assay_type", "assay_category", "assay_test_type", "assay_organism",
    "assay_tax_id", "assay_cell_type", "assay_subcellular_fraction",
    "bao_format", "bao_label",
}


class LabelPolicyInputError(ValueError):
    """Raised when an input cannot be represented or interpreted safely."""


@dataclass(frozen=True)
class LedgerEntry:
    evidence_id: str
    record_sha256: str
    disposition: str
    reason: str
    pair_key: Optional[str]
    occurrence: int
    record_json: str


@dataclass(frozen=True)
class FilteringLedger:
    """Immutable, input-order-independent disposition record for every row."""

    entries: tuple[LedgerEntry, ...] = ()

    @property
    def input_count(self) -> int:
        return len(self.entries)

    @property
    def counts(self) -> dict[str, int]:
        result = {category: 0 for category in _LEDGER_CATEGORIES}
        for entry in self.entries:
            result[entry.disposition] += 1
        return result

    @property
    def sha256(self) -> str:
        return _sha256([asdict(entry) for entry in self.entries])


@dataclass(frozen=True)
class PairLabelGroup:
    """All source evidence for one native molecule and protein identity."""

    pair_key: str
    split_group_id: str
    molecule_chembl_id: str
    protein_accession: str
    taxonomy_id: int
    evidence_ids: tuple[str, ...]
    records_json: tuple[str, ...]
    group_sha256: str


@dataclass(frozen=True)
class LabelDecision:
    pair_key: str
    split_group_id: str
    endpoint: str
    assay_condition_sha256: str
    label: str
    evidence_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    provenance_partition: str


@dataclass(frozen=True)
class LabelPolicyResult:
    policy_version: str
    policy_sha256: str
    confirmed_labeled_pairs: tuple[LabelDecision, ...]
    test_only_labeled_pairs: tuple[LabelDecision, ...]
    gray: tuple[LabelDecision, ...]
    unknown: tuple[LabelDecision, ...]
    conflicts: tuple[LabelDecision, ...]
    excluded: tuple[LabelDecision, ...]
    unverified: tuple[LabelDecision, ...]
    counts: Mapping[str, int]
    ledger: FilteringLedger
    training_ready: bool = False


@dataclass(frozen=True)
class _PreparedEvidence:
    evidence_id: str
    record_sha256: str
    record_json: str
    record: Optional[dict[str, Any]]
    pair_key: Optional[str]
    pair_identity: Optional[tuple[str, int, str]]
    disposition: str
    reason: str
    duplicate_key: str
    measurement_sha256: str


def _json_value(value: Any, *, depth: int = 0) -> Any:
    """Convert supported values to stable JSON while preserving invalid numbers."""
    if depth > 64:
        raise LabelPolicyInputError("evidence exceeds the maximum JSON nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        token = "nan" if math.isnan(value) else ("+infinity" if value > 0 else "-infinity")
        return {"__invalid_float__": token}
    if isinstance(value, Decimal):
        token = str(value)
        if not value.is_finite():
            return {"__invalid_decimal__": token}
        return {"__decimal__": token}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise LabelPolicyInputError("evidence object keys must be strings")
        return {
            key: _json_value(value[key], depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, depth=depth + 1) for item in value]
    raise LabelPolicyInputError(
        f"unsupported evidence value type: {type(value).__module__}.{type(value).__qualname__}"
    )


def _json_text(value: Any) -> str:
    return json.dumps(
        _json_value(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _same_raw_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    if isinstance(left, Decimal) and not left.is_finite():
        return not right.is_finite() and str(left) == str(right)
    return left == right


def _evidence_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, ActivityEvidence):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _pair_identity(record: Mapping[str, Any]) -> tuple[
    Optional[tuple[str, int, str]], str, str
]:
    """Return a strict native pair identity or an explicit quarantine reason."""
    molecule_id = record.get("molecule_chembl_id")
    activity_id = record.get("activity_id")
    target_id = record.get("target_chembl_id")
    assay_id = record.get("assay_chembl_id")
    document_id = record.get("document_chembl_id")
    for name, value in (("molecule_chembl_id", molecule_id), ("target_chembl_id", target_id),
                        ("assay_chembl_id", assay_id), ("document_chembl_id", document_id)):
        if not isinstance(value, str) or not _CHEMBL_ID.fullmatch(value):
            return None, "unverified", f"missing_or_invalid_{name}"
    if not ((type(activity_id) is int and activity_id > 0)
            or (isinstance(activity_id, str) and activity_id.strip())):
        return None, "unverified", "missing_or_invalid_activity_id"

    if record.get("target_type") != "SINGLE PROTEIN":
        if record.get("target_type") is None:
            return None, "unverified", "missing_target_type"
        return None, "excluded", "target_type_not_single_protein"
    organism = record.get("target_organism")
    taxonomy = record.get("target_taxonomy_id")
    if organism != "Homo sapiens" or type(taxonomy) is not int:
        if organism is None or taxonomy is None:
            return None, "unverified", "missing_target_species"
        return None, "excluded", "target_species_not_human"
    if taxonomy != 9606:
        return None, "excluded", "target_taxonomy_not_human"

    components = record.get("target_components")
    if not isinstance(components, list):
        return None, "unverified", "missing_target_components"
    if len(components) == 0:
        return None, "unverified", "missing_target_component"
    if len(components) != 1:
        return None, "excluded", "target_component_count_not_one"
    component = components[0]
    if not isinstance(component, Mapping):
        return None, "unverified", "invalid_target_component"
    component = dict(component)
    # ChEMBL stores species and taxonomy on the parent target. Derive absent
    # component facts for eligibility while preserving the raw component as-is
    # for _linked_metadata_error's exact source snapshot comparison.
    if component.get("organism") is None:
        component["organism"] = organism
    if taxonomy is not None:
        for tax_key in ("tax_id", "taxonomy_id", "target_taxonomy_id"):
            if component.get(tax_key) is None:
                component[tax_key] = taxonomy
    if component.get("component_type") != "PROTEIN":
        return None, "excluded", "component_type_not_protein"
    if component.get("organism") != "Homo sapiens":
        return None, "excluded", "component_species_not_human"
    for tax_key in ("tax_id", "taxonomy_id", "target_taxonomy_id"):
        if tax_key in component:
            component_taxonomy = component[tax_key]
            if type(component_taxonomy) is not int:
                return None, "unverified", "invalid_component_taxonomy"
            if component_taxonomy != taxonomy:
                return None, "excluded", "component_taxonomy_mismatch"
    accession = component.get("accession")
    if not isinstance(accession, str) or not _ACCESSION.fullmatch(accession):
        return None, "unverified", "missing_or_invalid_protein_accession"

    # Use the producer's canonical identity resolver for every species/taxonomy
    # alias, so an omitted primary field cannot hide an explicit conflicting alias.
    native_target = record.get("target_record")
    resolution_target = dict(native_target) if isinstance(native_target, Mapping) else {}
    for key, value in (("target_chembl_id", target_id), ("target_type", record["target_type"]),
                       ("organism", organism), ("tax_id", taxonomy)):
        resolution_target.setdefault(key, value)
    resolution_target["target_components"] = components
    try:
        resolved = resolve_target_components(
            resolution_target, source_kind=record.get("source_kind", "legacy_unverified"),
        )
    except (TypeError, ValueError):
        return None, "unverified", "invalid_target_component_evidence"
    if len(resolved) != 1 or resolved[0].get("resolution_status") != "verified":
        failed = resolved[0] if resolved else {}
        disposition = "excluded" if failed.get("resolution_status") == "excluded" else "unverified"
        return None, disposition, failed.get("reason_code") or "unresolved_target_component"
    if resolved[0].get("accession") != accession or resolved[0].get("taxonomy_id") != taxonomy:
        return None, "excluded", "component_identity_mismatch"

    raw = record.get("raw_activity_record")
    if not isinstance(raw, Mapping):
        return None, "unverified", "missing_raw_activity_record"
    for field in (
        "activity_id", "molecule_chembl_id", "target_chembl_id",
        "assay_chembl_id", "document_chembl_id", "standard_type",
        "standard_relation", "standard_value", "standard_units", "pchembl_value",
    ):
        if field not in raw or not _same_raw_value(record.get(field), raw.get(field)):
            return None, "unverified", f"raw_activity_mismatch:{field}"

    identity = (molecule_id, taxonomy, accession)
    pair_key = f"{molecule_id}|{taxonomy}|{accession}"
    return identity, "retained", pair_key


def _prepare_evidence(value: Any) -> _PreparedEvidence:
    record = _evidence_mapping(value)
    if record is None:
        record = {"__invalid_evidence_type__": f"{type(value).__module__}.{type(value).__qualname__}",
                  "raw": value}
        disposition, reason, pair_identity, pair_key = "unverified", "activity_evidence_not_object", None, None
    else:
        disposition = "unverified"
        reason = "incomplete_source_provenance"
        identity, pair_disposition, pair_reason = _pair_identity(record)
        pair_identity = identity
        pair_key = None
        if identity is not None:
            pair_key = f"{identity[0]}|{identity[1]}|{identity[2]}"
            kind = record.get("source_kind")
            source_path = record.get("source_path")
            source_hash = record.get("source_sha256")
            response_hash = record.get("response_sha256")
            receipts_complete = (
                isinstance(source_path, str) and bool(source_path)
                and isinstance(source_hash, str) and _SHA256.fullmatch(source_hash) is not None
                and isinstance(response_hash, str) and _SHA256.fullmatch(response_hash) is not None
                and all(
                    isinstance(record.get(f"{name}_record_source_path"), str)
                    and isinstance(record.get(f"{name}_record_source_sha256"), str)
                    and _SHA256.fullmatch(record[f"{name}_record_source_sha256"]) is not None
                    for name in ("target", "assay", "document")
                )
            )
            if (pair_disposition == "retained" and kind == "online_snapshot" and receipts_complete
                    and record.get("metadata_status") == "complete"
                    and record.get("evidence_status") == "observed"
                    and record.get("page_status") == "complete"
                    and record.get("pagination_status") == "complete"):
                disposition, reason = "retained", "source_receipts_present_pending_bundle_verification"
            elif pair_disposition == "excluded":
                disposition, reason = "excluded", pair_reason
            elif kind == "synthetic_test":
                disposition, reason = "unverified", "synthetic_test_only"
            elif kind != "online_snapshot":
                disposition, reason = "unverified", "source_kind_not_confirmed"
            else:
                disposition, reason = "unverified", "incomplete_source_provenance"
        else:
            disposition, reason = pair_disposition, pair_reason

    record_json = _json_text(record)
    record_sha = hashlib.sha256(record_json.encode("utf-8")).hexdigest()
    measurement = record.get("raw_activity_record") if isinstance(record, Mapping) else None
    if not isinstance(measurement, Mapping):
        measurement = record
    measurement_sha = _sha256(measurement)
    activity_id = record.get("activity_id")
    source_id = record.get("source_id")
    source_kind = record.get("source_kind")
    source_hash = record.get("source_sha256")
    response_hash = record.get("response_sha256")
    snapshot = source_hash if isinstance(source_hash, str) and _SHA256.fullmatch(source_hash) else response_hash
    if not isinstance(snapshot, str) or _SHA256.fullmatch(snapshot) is None:
        snapshot = record_sha
    duplicate_payload = {
        "source_id": source_id,
        "source_kind": source_kind,
        "activity_id": activity_id,
        "snapshot_sha256": snapshot,
        "source_context_sha256": _sha256({
            "source_path": record.get("source_path"),
            "source_version": record.get("source_version"),
            "target_type": record.get("target_type"),
            "target_organism": record.get("target_organism"),
            "target_taxonomy_id": record.get("target_taxonomy_id"),
            "target_components": record.get("target_components"),
            "target_record": record.get("target_record"),
            "target_record_source_path": record.get("target_record_source_path"),
            "target_record_source_sha256": record.get("target_record_source_sha256"),
            "assay_record": record.get("assay_record"),
            "assay_record_source_path": record.get("assay_record_source_path"),
            "assay_record_source_sha256": record.get("assay_record_source_sha256"),
            "document_record": record.get("document_record"),
            "document_record_source_path": record.get("document_record_source_path"),
            "document_record_source_sha256": record.get("document_record_source_sha256"),
            "confidence_score": record.get("confidence_score"),
            "confidence_source": record.get("confidence_source"),
            "assay_confidence_score": record.get("assay_confidence_score"),
            "metadata_status": record.get("metadata_status"),
            "entity_bundle_version": record.get("entity_bundle_version"),
            "entity_bundle_hash": record.get("entity_bundle_hash"),
        }),
    }
    duplicate_key = _json_text(duplicate_payload)
    evidence_payload = {**duplicate_payload, "measurement_sha256": measurement_sha}
    evidence_id = "evidence-v1-" + _sha256(evidence_payload)[:24]
    return _PreparedEvidence(
        evidence_id=evidence_id,
        record_sha256=record_sha,
        record_json=record_json,
        record=record if isinstance(record, dict) else None,
        pair_key=pair_key,
        pair_identity=pair_identity,
        disposition=disposition,
        reason=reason,
        duplicate_key=duplicate_key,
        measurement_sha256=measurement_sha,
    )


def group_evidence(
    activity_rows: Iterable[ActivityEvidence | Mapping[str, Any]],
) -> tuple[tuple[PairLabelGroup, ...], FilteringLedger]:
    """Group complete producer evidence by molecule ID plus protein identity.

    Grouping is pure: inputs are copied, sorted, and never mutated. Synthetic and
    unverified evidence still receives a split group when identity is resolvable,
    but its ledger disposition keeps it out of confirmed label decisions.
    """
    if isinstance(activity_rows, (str, bytes, bytearray, Mapping)):
        raise LabelPolicyInputError("activity_rows must be an iterable of evidence records")
    try:
        values = list(activity_rows)
    except TypeError as exc:
        raise LabelPolicyInputError("activity_rows must be iterable") from exc

    prepared = [_prepare_evidence(value) for value in values]
    by_duplicate_key: dict[str, list[_PreparedEvidence]] = {}
    for item in prepared:
        by_duplicate_key.setdefault(item.duplicate_key, []).append(item)

    retained: list[_PreparedEvidence] = []
    entries: list[LedgerEntry] = []
    for duplicate_key in sorted(by_duplicate_key):
        candidates = by_duplicate_key[duplicate_key]
        by_measurement: dict[str, list[_PreparedEvidence]] = {}
        for item in candidates:
            by_measurement.setdefault(item.measurement_sha256, []).append(item)
        if len(by_measurement) > 1:
            # Same native ID and source hash with different raw rows is not a duplicate.
            for item in sorted(candidates, key=lambda entry: (entry.record_sha256, entry.evidence_id)):
                entries.append(LedgerEntry(
                    evidence_id=item.evidence_id,
                    record_sha256=item.record_sha256,
                    disposition="unverified",
                    reason="source_identity_has_conflicting_raw_activity",
                    pair_key=item.pair_key,
                    occurrence=1,
                    record_json=item.record_json,
                ))
                if item.pair_identity is not None:
                    retained.append(item)
            continue
        copies = next(iter(by_measurement.values()))
        ordered = sorted(copies, key=lambda entry: (entry.record_sha256, entry.evidence_id))
        representative = ordered[0]
        retained.append(representative)
        entries.append(LedgerEntry(
            evidence_id=representative.evidence_id,
            record_sha256=representative.record_sha256,
            disposition=representative.disposition,
            reason=representative.reason,
            pair_key=representative.pair_key,
            occurrence=1,
            record_json=representative.record_json,
        ))
        for occurrence, duplicate in enumerate(ordered[1:], start=2):
            entries.append(LedgerEntry(
                evidence_id=duplicate.evidence_id,
                record_sha256=duplicate.record_sha256,
                disposition="duplicate",
                reason="same_activity_id_and_source_snapshot",
                pair_key=duplicate.pair_key,
                occurrence=occurrence,
                record_json=duplicate.record_json,
            ))

    grouped: dict[tuple[str, int, str], list[_PreparedEvidence]] = {}
    for item in retained:
        if item.pair_identity is not None and item.record is not None:
            grouped.setdefault(item.pair_identity, []).append(item)

    groups: list[PairLabelGroup] = []
    for identity in sorted(grouped):
        molecule_id, taxonomy_id, accession = identity
        pair_key = f"{molecule_id}|{taxonomy_id}|{accession}"
        pair_hash = _sha256({
            "schema": "pair-label-group-v1",
            "molecule_chembl_id": molecule_id,
            "taxonomy_id": taxonomy_id,
            "accession": accession,
        })
        split_group_id = f"pair-v1-{pair_hash}"
        ordered = sorted(grouped[identity], key=lambda entry: (entry.evidence_id, entry.record_sha256))
        evidence_ids = tuple(item.evidence_id for item in ordered)
        records_json = tuple(item.record_json for item in ordered)
        group_hash = _sha256({
            "split_group_id": split_group_id,
            "evidence_ids": list(evidence_ids),
            "record_sha256": [item.record_sha256 for item in ordered],
        })
        groups.append(PairLabelGroup(
            pair_key=pair_key,
            split_group_id=split_group_id,
            molecule_chembl_id=molecule_id,
            protein_accession=accession,
            taxonomy_id=taxonomy_id,
            evidence_ids=evidence_ids,
            records_json=records_json,
            group_sha256=group_hash,
        ))

    entries.sort(key=lambda entry: (
        entry.evidence_id, entry.occurrence, entry.record_sha256,
        entry.disposition, entry.reason,
    ))
    return tuple(groups), FilteringLedger(entries=tuple(entries))


def _reject_json_constant(token: str) -> None:
    raise LabelPolicyInputError(f"non-finite JSON constant is forbidden: {token}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LabelPolicyInputError(f"duplicate policy key: {key}")
        result[key] = value
    return result


def _load_policy(policy_path: Any = None) -> tuple[dict[str, Any], str, dict[str, Decimal]]:
    path = _POLICY_PATH if policy_path is None else policy_path
    if not isinstance(path, (str, os.PathLike)):
        raise LabelPolicyInputError("policy_path must be a filesystem path")
    try:
        policy_bytes = Path(path).read_bytes()
        policy = json.loads(
            policy_bytes.decode("utf-8-sig"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except LabelPolicyInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LabelPolicyInputError(f"label policy could not be read as strict UTF-8 JSON: {exc}") from exc
    if not isinstance(policy, dict) or set(policy) != _POLICY_KEYS:
        raise LabelPolicyInputError("label policy has an unsupported or incomplete schema")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise LabelPolicyInputError("unsupported label policy schema_version")
    if type(policy["entity_bundle_schema_version"]) is not int or policy["entity_bundle_schema_version"] != 2:
        raise LabelPolicyInputError("label policy requires entity bundle schema version 2")
    if not isinstance(policy["policy_version"], str) or not policy["policy_version"].strip():
        raise LabelPolicyInputError("policy_version must be a non-empty string")
    if policy["confirmed_source_kinds"] != ["online_snapshot"]:
        raise LabelPolicyInputError("only verified online_snapshot sources may be confirmed")
    if policy["test_only_source_kinds"] != ["synthetic_test"]:
        raise LabelPolicyInputError("synthetic_test must remain in its own partition")
    if (not isinstance(policy["entity_eligibility"], dict)
            or _json_text(policy["entity_eligibility"]) != _json_text(_ENTITY_POLICY)):
        raise LabelPolicyInputError("policy entity eligibility must remain human SINGLE PROTEIN")
    endpoints = policy["endpoints"]
    if not isinstance(endpoints, dict) or set(endpoints) != {"labeled", "independent_only"}:
        raise LabelPolicyInputError("endpoint policy requires labeled and independent_only lists")
    labeled = endpoints.get("labeled")
    independent = endpoints.get("independent_only")
    allowed = {"Ki", "Kd", "IC50"}
    if (not isinstance(labeled, list) or not labeled
            or any(not isinstance(item, str) for item in labeled)
            or len(labeled) != len(set(labeled))
            or not set(labeled) <= {"Ki", "Kd"}):
        raise LabelPolicyInputError("labeled endpoints must be a non-empty unique Ki/Kd subset")
    if (not isinstance(independent, list) or not independent
            or any(not isinstance(item, str) for item in independent)
            or len(independent) != len(set(independent))
            or "IC50" not in independent):
        raise LabelPolicyInputError("independent_only must uniquely include IC50 and unlabeled endpoints")
    if set(labeled) & set(independent) or set(labeled) | set(independent) != allowed:
        raise LabelPolicyInputError("labeled and independent endpoints must be disjoint and cover Ki/Kd/IC50")
    if policy["exact_relation"] != "=":
        raise LabelPolicyInputError("only exact relation '=' is supported")
    if type(policy["minimum_confidence"]) is not int or policy["minimum_confidence"] != 8:
        raise LabelPolicyInputError("this policy version requires integer confidence >= 8")
    thresholds = policy["thresholds"]
    if (not isinstance(thresholds, dict)
            or set(thresholds) != {"active_min_p", "inactive_max_p"}
            or type(thresholds.get("active_min_p")) is not int
            or type(thresholds.get("inactive_max_p")) is not int
            or thresholds != {"active_min_p": 6, "inactive_max_p": 5}):
        raise LabelPolicyInputError("this policy version requires inclusive p thresholds 6 and 5")
    if policy["units_to_nanomolar"] != _UNIT_POLICY:
        raise LabelPolicyInputError("unit table differs from the reviewed exact unit allowlist")
    fields = policy["assay_compatibility_fields"]
    if (not isinstance(fields, list) or not fields
            or any(not isinstance(field, str) or field not in _ASSAY_FIELDS for field in fields)
            or len(fields) != len(set(fields))):
        raise LabelPolicyInputError("assay_compatibility_fields is invalid")

    units: dict[str, Decimal] = {}
    for unit, raw_factor in policy["units_to_nanomolar"].items():
        if not isinstance(unit, str) or not isinstance(raw_factor, str):
            raise LabelPolicyInputError("unit names and conversion factors must be strings")
        try:
            factor = Decimal(raw_factor)
        except InvalidOperation as exc:
            raise LabelPolicyInputError(f"invalid conversion factor for {unit}") from exc
        if not factor.is_finite() or factor <= 0:
            raise LabelPolicyInputError(f"conversion factor for {unit} must be finite and positive")
        units[unit] = factor
    return policy, hashlib.sha256(policy_bytes).hexdigest(), units


def _load_source_bundle(path: Any) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    if not isinstance(path, (str, os.PathLike)):
        raise LabelPolicyInputError("entity_bundle_path must be a filesystem path")
    try:
        bundle = load_entity_bundle(path, verify_sources=True)
    except Exception as exc:
        raise LabelPolicyInputError(f"entity bundle failed source verification: {exc}") from exc
    if bundle.get("schema_version") != 2:
        raise LabelPolicyInputError("entity bundle schema does not match the policy")
    return bundle


def _mapping_matches(left: Any, right: Any) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    try:
        return _json_text(left) == _json_text(right)
    except LabelPolicyInputError:
        return False


def _verified_bundle_membership(
    record_json: str,
    record: Mapping[str, Any],
    bundle: Optional[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    allow_test_only: bool,
) -> tuple[str, str]:
    source_kind = record.get("source_kind")
    if source_kind in policy["test_only_source_kinds"]:
        if not allow_test_only:
            return "unverified", "synthetic_test_only"
        if bundle is not None:
            bundle_activities = bundle.get("activities", [])
            if not any(_mapping_matches(record, item) for item in bundle_activities):
                return "unverified", "activity_not_member_of_verified_entity_bundle"
        return "test_only", "synthetic_test_only"
    if source_kind not in policy["confirmed_source_kinds"]:
        return "unverified", "source_kind_not_confirmable"
    if bundle is None:
        return "unverified", "verified_entity_bundle_required"
    if not any(_mapping_matches(record, item) for item in bundle.get("activities", [])):
        return "unverified", "activity_not_member_of_verified_entity_bundle"
    if (record.get("entity_bundle_version") != bundle.get("entity_version")
            or record.get("entity_bundle_hash") != bundle.get("content_sha256")):
        return "unverified", "activity_entity_bundle_link_mismatch"
    if (record.get("metadata_status") != "complete"
            or record.get("evidence_status") != "observed"
            or record.get("page_status") != "complete"
            or record.get("pagination_status") != "complete"):
        return "unverified", "source_activity_capture_incomplete"

    identity, disposition, reason_or_key = _pair_identity(record)
    if identity is None or disposition != "retained":
        return "unverified", reason_or_key
    molecule_id, taxonomy_id, accession = identity
    molecule = next((item for item in bundle.get("molecules", [])
                     if isinstance(item, Mapping)
                     and item.get("molecule_chembl_id") == molecule_id), None)
    if (molecule is None or molecule.get("structure_status") != "verified"
            or molecule.get("source_kind") != "online_snapshot"):
        return "unverified", "molecule_entity_not_source_verified"
    protein = next((item for item in bundle.get("proteins", [])
                    if isinstance(item, Mapping)
                    and item.get("accession") == accession
                    and item.get("taxonomy_id") == taxonomy_id), None)
    if (protein is None or protein.get("identity_status") != "verified"
            or protein.get("sequence_status") != "verified"
            or protein.get("source_kind") != "online_snapshot"):
        return "unverified", "protein_entity_not_source_verified"
    return "confirmed", ""


def _linked_metadata_error(record: Mapping[str, Any]) -> Optional[str]:
    if (record.get("metadata_status") != "complete"
            or record.get("evidence_status") != "observed"
            or record.get("page_status") != "complete"
            or record.get("pagination_status") != "complete"):
        return "source_activity_capture_incomplete"
    target = record.get("target_record")
    assay = record.get("assay_record")
    document = record.get("document_record")
    if not isinstance(target, Mapping) or target.get("target_chembl_id") != record.get("target_chembl_id"):
        return "target_snapshot_link_mismatch"
    if (target.get("target_type") != "SINGLE PROTEIN"
            or target.get("organism") != "Homo sapiens"
            or target.get("tax_id") != 9606
            or not _mapping_matches(
                {"components": target.get("target_components")},
                {"components": record.get("target_components")},
            )):
        return "target_snapshot_entity_mismatch"
    if (not isinstance(assay, Mapping)
            or assay.get("assay_chembl_id") != record.get("assay_chembl_id")
            or assay.get("target_chembl_id") != record.get("target_chembl_id")):
        return "assay_snapshot_link_mismatch"
    if (not isinstance(document, Mapping)
            or document.get("document_chembl_id") != record.get("document_chembl_id")):
        return "document_snapshot_link_mismatch"
    return None


def _assay_condition(record: Mapping[str, Any], fields: list[str]) -> tuple[Optional[str], Optional[str]]:
    assay = record.get("assay_record")
    if (not isinstance(assay, Mapping)
            or assay.get("assay_chembl_id") != record.get("assay_chembl_id")
            or assay.get("target_chembl_id") != record.get("target_chembl_id")):
        return None, "assay_snapshot_link_mismatch"
    condition: dict[str, Any] = {}
    for field in fields:
        value = assay.get(field)
        if value is None:
            condition[field] = None
        elif field == "assay_tax_id" and type(value) is int and value > 0:
            condition[field] = value
        elif field != "assay_tax_id" and isinstance(value, str):
            condition[field] = value
        else:
            return None, f"invalid_assay_condition:{field}"
    return _sha256(condition), None


def _confidence_error(record: Mapping[str, Any], minimum: int) -> Optional[str]:
    raw = record.get("raw_activity_record")
    assay = record.get("assay_record")
    source = record.get("confidence_source")
    if source == "activity" and isinstance(raw, Mapping):
        source_value = raw.get("confidence_score")
    elif source == "assay" and isinstance(assay, Mapping):
        source_value = assay.get("confidence_score")
        if record.get("assay_confidence_score") != source_value:
            return "assay_confidence_source_mismatch"
    else:
        return "confidence_source_missing"
    effective = record.get("confidence_score")
    if type(source_value) is not int or type(effective) is not int:
        return "confidence_invalid_type"
    if effective != source_value:
        return "confidence_source_mismatch"
    if effective < minimum:
        return "confidence_below_minimum"
    return None


def _decimal_measurement(value: Any) -> tuple[Optional[Decimal], Optional[str]]:
    if type(value) is int:
        text = str(value)
    elif type(value) is float:
        if not math.isfinite(value):
            return None, "measurement_value_not_finite_positive"
        text = str(value)
    elif isinstance(value, str) and value and value == value.strip() and len(value) <= 64:
        text = value
    else:
        return None, "measurement_value_invalid_type"
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return None, "measurement_value_invalid_number"
    if not parsed.is_finite() or parsed <= 0 or abs(parsed.adjusted()) > 300:
        return None, "measurement_value_not_finite_positive"
    if len(parsed.as_tuple().digits) > 128:
        return None, "measurement_value_precision_limit"
    return parsed, None


def _evaluate_measurement(
    record: Mapping[str, Any],
    policy: Mapping[str, Any],
    unit_factors: Mapping[str, Decimal],
) -> tuple[str, str, str, Optional[str]]:
    endpoint = record.get("standard_type")
    endpoint_value = endpoint if isinstance(endpoint, str) else "unknown"
    condition_hash, condition_error = _assay_condition(
        record, policy["assay_compatibility_fields"],
    )
    if condition_error:
        return "excluded", condition_error, endpoint_value, None
    link_error = _linked_metadata_error(record)
    if link_error:
        return "unverified", link_error, endpoint_value, condition_hash
    if endpoint_value == "IC50" or endpoint_value not in policy["endpoints"]["labeled"]:
        return "excluded", "endpoint_not_label_eligible", endpoint_value, condition_hash
    if record.get("standard_relation") != policy["exact_relation"]:
        return "excluded", "relation_not_exact", endpoint_value, condition_hash
    confidence_error = _confidence_error(record, policy["minimum_confidence"])
    if confidence_error:
        return "excluded", confidence_error, endpoint_value, condition_hash
    value = record.get("standard_value")
    if value is None:
        return "unknown", "measurement_value_missing", endpoint_value, condition_hash
    unit = record.get("standard_units")
    if not isinstance(unit, str) or unit not in unit_factors:
        return "excluded", "unit_not_allowed", endpoint_value, condition_hash
    decimal_value, value_error = _decimal_measurement(value)
    if value_error:
        return "excluded", value_error, endpoint_value, condition_hash
    assert decimal_value is not None
    with localcontext() as context:
        context.prec = 256
        value_nm = decimal_value * unit_factors[unit]
        active_boundary = Decimal(10) ** (9 - policy["thresholds"]["active_min_p"])
        inactive_boundary = Decimal(10) ** (9 - policy["thresholds"]["inactive_max_p"])
    if value_nm <= active_boundary:
        return "active", "measured_value_meets_inclusive_p6_boundary", endpoint_value, condition_hash
    if value_nm >= inactive_boundary:
        return "inactive", "measured_value_meets_inclusive_p5_boundary", endpoint_value, condition_hash
    return "gray", "measured_value_inside_gray_band", endpoint_value, condition_hash


def _make_decision(
    group: Optional[PairLabelGroup],
    *,
    pair_key: str,
    split_group_id: str,
    endpoint: str,
    condition_hash: str,
    label: str,
    evidence_ids: Iterable[str],
    reasons: Iterable[str],
    partition: str,
) -> LabelDecision:
    return LabelDecision(
        pair_key=pair_key,
        split_group_id=split_group_id,
        endpoint=endpoint,
        assay_condition_sha256=condition_hash,
        label=label,
        evidence_ids=tuple(sorted(set(evidence_ids))),
        reasons=tuple(sorted(set(reasons))),
        provenance_partition=partition,
    )


def _validate_filtering_ledger(ledger: FilteringLedger) -> dict[str, LedgerEntry]:
    if not isinstance(ledger.entries, tuple):
        raise LabelPolicyInputError("filtering ledger entries must be an immutable tuple")
    primary: dict[str, LedgerEntry] = {}
    occurrences: dict[str, set[int]] = {}
    entry_keys: set[tuple[str, int]] = set()
    for entry in ledger.entries:
        if not isinstance(entry, LedgerEntry):
            raise LabelPolicyInputError("filtering ledger contains an invalid entry type")
        if (not isinstance(entry.evidence_id, str) or not entry.evidence_id
                or type(entry.occurrence) is not int or entry.occurrence < 1
                or entry.disposition not in _LEDGER_CATEGORIES
                or not isinstance(entry.reason, str)
                or not isinstance(entry.record_json, str)
                or not isinstance(entry.record_sha256, str)
                or _SHA256.fullmatch(entry.record_sha256) is None):
            raise LabelPolicyInputError("filtering ledger entry has invalid fields")
        key = (entry.evidence_id, entry.occurrence)
        if key in entry_keys:
            raise LabelPolicyInputError("filtering ledger assigns an input occurrence more than once")
        entry_keys.add(key)
        occurrences.setdefault(entry.evidence_id, set()).add(entry.occurrence)
        if hashlib.sha256(entry.record_json.encode("utf-8")).hexdigest() != entry.record_sha256:
            raise LabelPolicyInputError("filtering ledger record hash does not match its raw record")
        if (entry.occurrence == 1 and entry.disposition == "duplicate") or (
            entry.occurrence > 1 and entry.disposition != "duplicate"
        ):
            raise LabelPolicyInputError("filtering ledger duplicate occurrences are inconsistent")
        try:
            record = json.loads(entry.record_json, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, LabelPolicyInputError) as exc:
            raise LabelPolicyInputError(f"filtering ledger contains invalid raw JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise LabelPolicyInputError("filtering ledger raw evidence must be a JSON object")
        if "__invalid_evidence_type__" in record:
            if entry.pair_key is not None:
                raise LabelPolicyInputError("invalid raw evidence cannot claim a pair identity")
        else:
            prepared = _prepare_evidence(record)
            if prepared.record_json != entry.record_json or prepared.evidence_id != entry.evidence_id:
                raise LabelPolicyInputError("filtering ledger evidence ID does not match its raw record")
            if prepared.pair_key != entry.pair_key:
                raise LabelPolicyInputError("filtering ledger pair key does not match its raw record")
        if entry.occurrence == 1:
            primary[entry.evidence_id] = entry
    for evidence_id, values in occurrences.items():
        if values != set(range(1, len(values) + 1)):
            raise LabelPolicyInputError(f"filtering ledger occurrence sequence is incomplete for {evidence_id}")
    return primary


def apply_label_policy(
    groups: Iterable[PairLabelGroup],
    *,
    ledger: Optional[FilteringLedger] = None,
    policy_path: Any = None,
    entity_bundle_path: Any = None,
    allow_test_only: bool = False,
) -> LabelPolicyResult:
    """Apply measured Ki/Kd rules after independent source/entity verification.

    A real confirmed label must be an exact activity row in a source-verified
    schema-v2 entity bundle. Synthetic evidence may exercise the algorithm only
    when explicitly requested, and is returned in a disjoint test-only table.
    """
    if type(allow_test_only) is not bool:
        raise LabelPolicyInputError("allow_test_only must be a boolean")
    if not isinstance(ledger, FilteringLedger):
        raise LabelPolicyInputError("a FilteringLedger from group_evidence is required")
    policy, policy_hash, unit_factors = _load_policy(policy_path)
    bundle = _load_source_bundle(entity_bundle_path)
    try:
        group_values = list(groups)
    except TypeError as exc:
        raise LabelPolicyInputError("groups must be iterable") from exc
    if any(not isinstance(group, PairLabelGroup) for group in group_values):
        raise LabelPolicyInputError("groups must contain PairLabelGroup values")
    ledger_primary = _validate_filtering_ledger(ledger)

    grouped_evidence_ids: set[str] = set()
    seen_pair_identities: set[tuple[str, int, str]] = set()
    seen_split_group_ids: set[str] = set()
    outcomes: list[tuple[PairLabelGroup, str, dict[str, Any], str, str, str, str]] = []
    for group in group_values:
        if (not isinstance(group.molecule_chembl_id, str)
                or not _CHEMBL_ID.fullmatch(group.molecule_chembl_id)
                or type(group.taxonomy_id) is not int
                or not isinstance(group.protein_accession, str)
                or not _ACCESSION.fullmatch(group.protein_accession)
                or not isinstance(group.split_group_id, str)
                or not isinstance(group.group_sha256, str)):
            raise LabelPolicyInputError("pair group has an invalid canonical pair identity")
        canonical_identity = (group.molecule_chembl_id, group.taxonomy_id, group.protein_accession)
        expected_pair_key = f"{group.molecule_chembl_id}|{group.taxonomy_id}|{group.protein_accession}"
        if group.pair_key != expected_pair_key:
            raise LabelPolicyInputError("pair_key does not match canonical molecule/protein identity")
        if canonical_identity in seen_pair_identities or group.split_group_id in seen_split_group_ids:
            raise LabelPolicyInputError("duplicate or forked canonical pair groups are not allowed")
        seen_pair_identities.add(canonical_identity)
        seen_split_group_ids.add(group.split_group_id)
        if (not isinstance(group.evidence_ids, tuple)
                or not isinstance(group.records_json, tuple)
                or len(group.evidence_ids) != len(group.records_json)
                or any(not isinstance(value, str) for value in group.evidence_ids)
                or any(not isinstance(value, str) for value in group.records_json)
                or len(set(group.evidence_ids)) != len(group.evidence_ids)):
            raise LabelPolicyInputError("pair group evidence arrays are malformed")
        parsed: list[tuple[str, dict[str, Any], str]] = []
        for evidence_id, record_json in zip(group.evidence_ids, group.records_json):
            if evidence_id in grouped_evidence_ids:
                raise LabelPolicyInputError("one evidence ID is assigned to multiple pair groups")
            try:
                record = json.loads(record_json, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, LabelPolicyInputError) as exc:
                raise LabelPolicyInputError(f"pair group contains invalid evidence JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise LabelPolicyInputError("pair group activity evidence must be an object")
            prepared = _prepare_evidence(record)
            if prepared.evidence_id != evidence_id or prepared.record_json != record_json:
                raise LabelPolicyInputError("pair group evidence identity or canonical content mismatch")
            if prepared.pair_identity != (
                group.molecule_chembl_id, group.taxonomy_id, group.protein_accession,
            ):
                raise LabelPolicyInputError("pair group identity does not match its activity evidence")
            ledger_entry = ledger_primary.get(evidence_id)
            if (ledger_entry is None or ledger_entry.record_sha256 != prepared.record_sha256
                    or ledger_entry.record_json != record_json
                    or ledger_entry.pair_key != prepared.pair_key):
                raise LabelPolicyInputError("pair group evidence does not match its filtering ledger")
            parsed.append((evidence_id, record, record_json))
            grouped_evidence_ids.add(evidence_id)
        split_hash = _sha256({
            "schema": "pair-label-group-v1",
            "molecule_chembl_id": group.molecule_chembl_id,
            "taxonomy_id": group.taxonomy_id,
            "accession": group.protein_accession,
        })
        expected_group_hash = _sha256({
            "split_group_id": f"pair-v1-{split_hash}",
            "evidence_ids": list(group.evidence_ids),
            "record_sha256": [hashlib.sha256(value.encode("utf-8")).hexdigest()
                              for value in group.records_json],
        })
        if (group.split_group_id != f"pair-v1-{split_hash}"
                or group.group_sha256 != expected_group_hash):
            raise LabelPolicyInputError("pair group split identity or content hash mismatch")

        for evidence_id, record, record_json in parsed:
            partition, trust_reason = _verified_bundle_membership(
                record_json, record, bundle, policy,
                allow_test_only=allow_test_only,
            )
            if partition == "unverified":
                outcomes.append((group, evidence_id, record, partition, "unverified",
                                 trust_reason, ""))
                continue
            status, reason, endpoint, condition_hash = _evaluate_measurement(
                record, policy, unit_factors,
            )
            outcomes.append((group, evidence_id, record, partition, status,
                             reason, condition_hash or ""))

    expected_grouped_ids = {
        evidence_id for evidence_id, entry in ledger_primary.items()
        if entry.pair_key is not None
    }
    if grouped_evidence_ids != expected_grouped_ids:
        raise LabelPolicyInputError("pair groups and ledger do not assign every canonical evidence row exactly once")

    decisions_by_class: dict[str, list[LabelDecision]] = {
        "confirmed_labeled_pairs": [], "test_only_labeled_pairs": [],
        "gray": [], "unknown": [], "conflicts": [], "excluded": [], "unverified": [],
    }
    ledger_updates: dict[str, tuple[str, str]] = {}
    buckets: dict[tuple[str, str, str, str], list[tuple[PairLabelGroup, str, dict[str, Any], str, str, str]]] = {}
    for group, evidence_id, record, partition, status, reason, condition_hash in outcomes:
        endpoint = record.get("standard_type") if isinstance(record.get("standard_type"), str) else "unknown"
        bucket_key = (group.pair_key, endpoint, condition_hash, partition)
        buckets.setdefault(bucket_key, []).append((group, evidence_id, record, status, reason, condition_hash))

    directly_conflicted_pairs: set[tuple[str, str]] = set()
    for bucket_key, rows in buckets.items():
        labels = {row[3] for row in rows if row[3] in {"active", "inactive"}}
        if labels == {"active", "inactive"}:
            directly_conflicted_pairs.add((bucket_key[0], bucket_key[3]))

    for bucket_key in sorted(buckets):
        rows = buckets[bucket_key]
        group = rows[0][0]
        pair_key, endpoint, condition_hash, partition = bucket_key
        candidate_rows = [row for row in rows if row[3] in {"active", "inactive"}]
        candidate_labels = {row[3] for row in candidate_rows}
        pair_conflicted = (pair_key, partition) in directly_conflicted_pairs
        if pair_conflicted and candidate_rows:
            evidence_ids = [row[1] for row in candidate_rows]
            is_direct_conflict = candidate_labels == {"active", "inactive"}
            decision = _make_decision(
                group, pair_key=pair_key, split_group_id=group.split_group_id,
                endpoint=endpoint, condition_hash=condition_hash, label="conflict",
                evidence_ids=evidence_ids,
                reasons=("active_inactive_same_endpoint_assay_condition" if is_direct_conflict
                         else "pair_quarantined_by_conflicting_endpoint",),
                partition=partition,
            )
            decisions_by_class["conflicts"].append(decision)
            if partition == "confirmed":
                for evidence_id in evidence_ids:
                    ledger_updates[evidence_id] = (
                        "conflict_candidate",
                        "active_inactive_conflict" if is_direct_conflict
                        else "pair_quarantined_by_conflicting_endpoint",
                    )
        elif candidate_labels:
            label = next(iter(candidate_labels))
            evidence_ids = [row[1] for row in candidate_rows]
            reasons = [row[4] for row in candidate_rows]
            decision = _make_decision(
                group, pair_key=pair_key, split_group_id=group.split_group_id,
                endpoint=endpoint, condition_hash=condition_hash, label=label,
                evidence_ids=evidence_ids, reasons=reasons, partition=partition,
            )
            result_key = "confirmed_labeled_pairs" if partition == "confirmed" else "test_only_labeled_pairs"
            decisions_by_class[result_key].append(decision)
            if partition == "confirmed":
                for evidence_id in evidence_ids:
                    ledger_updates[evidence_id] = ("retained", f"label_{label}")

        for status in ("gray", "unknown", "excluded", "unverified"):
            status_rows = [row for row in rows if row[3] == status]
            if not status_rows:
                continue
            # Keep distinct reason classes independently auditable.
            reason_groups: dict[str, list[tuple[PairLabelGroup, str, dict[str, Any], str, str, str]]] = {}
            for row in status_rows:
                reason_groups.setdefault(row[4], []).append(row)
            result_key = {"gray": "gray", "unknown": "unknown", "excluded": "excluded",
                          "unverified": "unverified"}[status]
            for reason in sorted(reason_groups):
                reason_rows = reason_groups[reason]
                decision = _make_decision(
                    group, pair_key=pair_key, split_group_id=group.split_group_id,
                    endpoint=endpoint, condition_hash=condition_hash, label=status,
                    evidence_ids=[row[1] for row in reason_rows], reasons=(reason,), partition=partition,
                )
                decisions_by_class[result_key].append(decision)
                if partition == "confirmed":
                    disposition = "retained" if status == "gray" else status
                    if disposition not in _LEDGER_CATEGORIES:
                        disposition = "unverified"
                    for row in reason_rows:
                        ledger_updates[row[1]] = (disposition, reason)

    # Preserve invalid-entity and malformed evidence that could not form a pair group.
    for entry in ledger.entries:
        if entry.evidence_id in grouped_evidence_ids or entry.disposition == "duplicate":
            continue
        label = "excluded" if entry.disposition == "excluded" else "unverified"
        decision = _make_decision(
            None, pair_key=entry.pair_key or "", split_group_id="", endpoint="unknown",
            condition_hash="", label=label, evidence_ids=(entry.evidence_id,),
            reasons=(entry.reason,), partition="unverified",
        )
        decisions_by_class[label].append(decision)

    updated_entries: list[LedgerEntry] = []
    for entry in ledger.entries:
        update = ledger_updates.get(entry.evidence_id)
        if update is not None and entry.disposition != "duplicate" and entry.occurrence == 1:
            updated_entries.append(replace(entry, disposition=update[0], reason=update[1]))
        else:
            updated_entries.append(entry)
    updated_entries.sort(key=lambda entry: (
        entry.evidence_id, entry.occurrence, entry.record_sha256,
        entry.disposition, entry.reason,
    ))
    result_ledger = FilteringLedger(entries=tuple(updated_entries))

    for values in decisions_by_class.values():
        values.sort(key=lambda item: (
            item.pair_key, item.endpoint, item.assay_condition_sha256,
            item.provenance_partition, item.label, item.evidence_ids,
        ))
    counts = {
        "confirmed_active": 0, "confirmed_inactive": 0,
        "test_only_active": 0, "test_only_inactive": 0,
        "confirmed_gray": 0, "test_only_gray": 0,
        "confirmed_unknown": 0, "test_only_unknown": 0,
        "confirmed_conflict": 0, "test_only_conflict": 0,
        "confirmed_excluded": 0, "test_only_excluded": 0,
        "unverified": 0,
    }
    for result_key, rows in decisions_by_class.items():
        for decision in rows:
            if decision.provenance_partition == "unverified":
                counts["unverified"] += len(decision.evidence_ids)
                continue
            prefix = "confirmed" if decision.provenance_partition == "confirmed" else "test_only"
            suffix = decision.label
            if suffix in {"active", "inactive", "gray", "unknown", "conflict", "excluded"}:
                counts[f"{prefix}_{suffix}"] += len(decision.evidence_ids)

    return LabelPolicyResult(
        policy_version=policy["policy_version"],
        policy_sha256=policy_hash,
        confirmed_labeled_pairs=tuple(decisions_by_class["confirmed_labeled_pairs"]),
        test_only_labeled_pairs=tuple(decisions_by_class["test_only_labeled_pairs"]),
        gray=tuple(decisions_by_class["gray"]),
        unknown=tuple(decisions_by_class["unknown"]),
        conflicts=tuple(decisions_by_class["conflicts"]),
        excluded=tuple(decisions_by_class["excluded"]),
        unverified=tuple(decisions_by_class["unverified"]),
        counts=counts,
        ledger=result_ledger,
        training_ready=False,
    )

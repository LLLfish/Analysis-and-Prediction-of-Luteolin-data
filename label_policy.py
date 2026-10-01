"""Deterministic grouping and measured-label decisions for ChEMBL evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Optional

from evidence_builder import ActivityEvidence


_ACCESSION = re.compile(r"^[A-Z0-9]{6}(?:[A-Z0-9]{4})?$")
_CHEMBL_ID = re.compile(r"^CHEMBL[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEDGER_CATEGORIES = (
    "retained", "duplicate", "excluded", "unverified", "conflict_candidate",
)


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

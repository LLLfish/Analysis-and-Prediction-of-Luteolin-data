"""Read-only provenance and readiness reports for the three legacy data roots."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

from data_contracts import canonical_json_sha256, semantic_manifest_hash


_SOURCE_SPECS = (
    ("apl_data", "data", "legacy_unverified"),
    ("framework_output_data", "framework_output/data", "legacy_unverified"),
    ("offline_data", "offline_data.json", "static_reference"),
)
_ACTIVITY_REQUIRED_FIELDS = (
    "activity_id", "assay_chembl_id", "document_chembl_id", "target_chembl_id",
    "target_components", "standard_type", "standard_relation", "standard_value", "standard_units",
)
_PROTEIN_REQUIRED_FIELDS = ("accession", "taxonomy_id", "sequence", "sequence_sha256")
_ID_FIELDS = frozenset({
    "activity_id", "assay_chembl_id", "document_chembl_id", "target_chembl_id",
    "molecule_chembl_id", "chembl_id", "target_id", "accession", "primaryaccession",
    "uniprot_id", "protein_id", "stringid_a", "stringid_b", "protein_a", "protein_b",
})
_COLLECTION_KEYS = frozenset({
    "activities", "targets", "proteins", "ppi", "edges", "interactions", "molecules",
    "compound", "compounds", "records", "results", "data",
})
_SEQUENCE_KEYS = frozenset({"sequence", "full_sequence", "amino_acid_sequence", "aa_sequence"})
_PAGE_KEYS = frozenset({"pagination_status", "page_status", "page_meta", "page_count", "next_page", "next"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceAuditError(ValueError):
    """The requested source root or report destination is unsafe or unavailable."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _frozen_receipt_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    entities = manifest.get("entities")
    provenance = manifest.get("source_provenance")
    policies = manifest.get("policies")
    activities = manifest.get("activities")
    pairs = manifest.get("pairs")
    split = manifest.get("split")
    if not isinstance(entities, Mapping) or not isinstance(provenance, (list, tuple)):
        raise SourceAuditError("canonical corpus receipt requires entity and raw-provenance bindings")
    if not isinstance(policies, Mapping) or not isinstance(activities, Mapping):
        raise SourceAuditError("canonical corpus receipt requires frozen policies and activity evidence")
    if not isinstance(pairs, Mapping) or not isinstance(split, Mapping):
        raise SourceAuditError("canonical corpus receipt requires pair and split claims")
    entity_fields = ("path", "schema_version", "entity_version", "content_sha256", "file_sha256")
    policy_fields = ("path", "file_sha256", "semantic_sha256", "schema_version", "policy_version")
    if any(field not in entities for field in entity_fields):
        raise SourceAuditError("canonical corpus receipt entity reference is incomplete")
    if any(not isinstance(policies.get(kind), Mapping) or any(field not in policies[kind] for field in policy_fields)
           for kind in ("split", "label")):
        raise SourceAuditError("canonical corpus receipt policy references are incomplete")
    if not isinstance(activities.get("semantic_sha256"), str):
        raise SourceAuditError("canonical corpus receipt activity semantic hash is missing")
    return {
        "entities": {field: entities[field] for field in entity_fields},
        "raw_provenance_sha256": canonical_json_sha256(provenance),
        "policies": {
            kind: {field: policies[kind][field] for field in policy_fields}
            for kind in ("split", "label")
        },
        "activity_evidence_semantic_sha256": activities["semantic_sha256"],
        "split_claim_sha256": canonical_json_sha256({"split": split, "pairs": pairs}),
    }


def _receipt_count_map(value: Any, keys: Sequence[str], label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise SourceAuditError(f"canonical corpus receipt {label} metrics are missing")
    result: dict[str, int] = {}
    for key in keys:
        count = value.get(key)
        if type(count) is not int or count < 0:
            raise SourceAuditError(f"canonical corpus receipt {label}.{key} must be an evaluated nonnegative integer")
        result[key] = count
    return result


def build_frozen_corpus_readiness_receipt(manifest: Mapping[str, Any], audit_report: Any) -> dict[str, Any]:
    """Create an immutable readiness receipt only from a clean independent corpus audit."""
    if not isinstance(manifest, Mapping):
        raise SourceAuditError("canonical corpus receipt manifest must be an object")
    if hasattr(audit_report, "to_dict") and callable(audit_report.to_dict):
        audit_report = audit_report.to_dict()
    if not isinstance(audit_report, Mapping):
        raise SourceAuditError("canonical corpus receipt requires an independent audit report")
    if (audit_report.get("status") != "clean" or audit_report.get("exit_code") != 0
            or audit_report.get("findings") not in ([], ())):
        raise SourceAuditError("canonical corpus readiness receipt requires a clean, complete independent audit")
    if type(audit_report.get("training_ready")) is not bool:
        raise SourceAuditError("canonical corpus audit did not derive a boolean training_ready value")

    verified_entities = _receipt_count_map(
        audit_report.get("verified_entity_counts"), ("molecules", "proteins"), "verified_entity_counts",
    )
    component_counts = _receipt_count_map(
        audit_report.get("component_counts"), ("molecule_clusters", "protein_clusters"), "component_counts",
    )
    label_dispositions = audit_report.get("label_disposition_counts")
    if not isinstance(label_dispositions, Mapping) or any(type(count) is not int or count < 0 for count in label_dispositions.values()):
        raise SourceAuditError("canonical corpus label dispositions must be independently evaluated nonnegative counts")
    ledger_dispositions = audit_report.get("ledger_disposition_counts")
    if not isinstance(ledger_dispositions, Mapping) or any(type(count) is not int or count < 0 for count in ledger_dispositions.values()):
        raise SourceAuditError("canonical corpus ledger dispositions must be independently evaluated nonnegative counts")
    support = audit_report.get("class_support")
    if not isinstance(support, Mapping):
        raise SourceAuditError("canonical corpus audit omitted independently derived double-cold support")
    double_cold_support: dict[str, dict[str, int]] = {}
    for partition in ("train", "val", "test"):
        double_cold_support[partition] = _receipt_count_map(support.get(partition), ("active", "inactive"), f"class_support.{partition}")

    bindings = _frozen_receipt_bindings(manifest)
    excluded_labels = {
        "confirmed": label_dispositions.get("confirmed_excluded", 0),
        "test_only": label_dispositions.get("test_only_excluded", 0),
        "ledger": ledger_dispositions.get("excluded", 0),
    }
    receipt: dict[str, Any] = {
        "receipt_schema_version": 1,
        "scope_kind": "canonical_frozen_corpus",
        "scope_sha256": canonical_json_sha256(bindings),
        "bindings": bindings,
        "metrics": {
            "verified_entity_counts": verified_entities,
            "component_counts": component_counts,
            "label_disposition_counts": dict(label_dispositions),
            "ledger_disposition_counts": dict(ledger_dispositions),
            "excluded_label_dispositions": excluded_labels,
            "double_cold_split_support": double_cold_support,
            "audit_status": audit_report["status"],
            "audit_exit_code": audit_report["exit_code"],
            "training_ready": audit_report["training_ready"],
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {token}")))


def _git_state(code_root: Path) -> dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(code_root), *args], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    git_root = run("rev-parse", "--show-toplevel")
    if not git_root:
        return {"status": "not_initialized", "git_root": None, "branch": None, "commit": None,
                "dirty": None, "dirty_paths": []}
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=all")
    if not branch or not commit or status is None:
        return {"status": "git_unavailable", "git_root": git_root, "branch": branch, "commit": commit,
                "dirty": None, "dirty_paths": []}
    dirty_paths = [line[2:].lstrip() for line in status.splitlines() if len(line) >= 3]
    return {"status": "git", "git_root": git_root, "branch": branch, "commit": commit,
            "dirty": bool(dirty_paths), "dirty_paths": dirty_paths}


def _path_intersects(left: Path, right: Path) -> bool:
    left_key = os.path.normcase(str(left.resolve(strict=False)))
    right_key = os.path.normcase(str(right.resolve(strict=False)))
    try:
        return os.path.commonpath((left_key, right_key)) in {left_key, right_key}
    except ValueError:
        return False


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    junction = getattr(path, "is_junction", None)
    if callable(junction) and junction():
        return True
    try:
        import stat
        return bool(getattr(path.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except (OSError, AttributeError):
        return False


def _assert_safe_source_root(root: Path) -> Path:
    candidate = root.expanduser()
    if not candidate.is_absolute():
        raise SourceAuditError("APL root must be an absolute path")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir() or _is_reparse(candidate):
        raise SourceAuditError("APL root must be an existing regular directory")
    return resolved


def _list_regular_files(source_root: Path) -> tuple[list[Path], list[str]]:
    if not source_root.exists():
        return [], ["source_path_missing"]
    if not source_root.is_dir() or _is_reparse(source_root):
        return [], ["source_path_not_regular_directory"]
    files: list[Path] = []
    issues: list[str] = []
    for parent_text, directories, names in os.walk(source_root, topdown=True, followlinks=False):
        parent = Path(parent_text)
        safe_directories: list[str] = []
        for name in sorted(directories):
            child = parent / name
            if _is_reparse(child):
                issues.append(f"reparse_directory_skipped:{child.relative_to(source_root).as_posix()}")
            else:
                safe_directories.append(name)
        directories[:] = safe_directories
        for name in sorted(names):
            path = parent / name
            if _is_reparse(path):
                issues.append(f"reparse_file_skipped:{path.relative_to(source_root).as_posix()}")
                continue
            if path.is_file():
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(source_root).as_posix()), issues


def _walk_json(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_json(item, f"{path}[{index}]")


def _collections(payload: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(payload, list):
        counts["$"] = len(payload)
    for path, value in _walk_json(payload):
        if not isinstance(value, Mapping):
            continue
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _COLLECTION_KEYS and isinstance(item, list):
                counts[path + "." + str(key)] = len(item)
            elif normalized == "compound" and isinstance(item, Mapping):
                counts[path + "." + str(key)] = 1
            elif normalized == "proteins" and isinstance(item, Mapping):
                counts[path + "." + str(key)] = len(item)
    return dict(sorted(counts.items()))


def _iter_record_mappings(payload: Any) -> Iterable[Mapping[str, Any]]:
    for _path, value in _walk_json(payload):
        if isinstance(value, Mapping) and any(not isinstance(item, (Mapping, list)) for item in value.values()):
            yield value


def _flatten_ids(payload: Any) -> dict[str, list[str]]:
    found: dict[str, set[str]] = {}
    for _path, value in _walk_json(payload):
        if not isinstance(value, Mapping):
            continue
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _ID_FIELDS and isinstance(item, (str, int)) and not isinstance(item, bool):
                text = str(item).strip()
                if text:
                    found.setdefault(normalized, set()).add(text)
    return {key: sorted(values) for key, values in sorted(found.items())}


def _row_field_stats(payload: Any) -> tuple[dict[str, int], int, int, bool, bool]:
    records = list(_iter_record_mappings(payload))
    counts: dict[str, int] = {}
    for row in records:
        for key in row:
            counts[str(key)] = counts.get(str(key), 0) + 1
    sequence_rows = 0
    sequence_hash_rows = 0
    taxonomy_rows = 0
    page_proof = False
    for row in records:
        normalized = {str(key).casefold(): value for key, value in row.items()}
        if any(isinstance(normalized.get(key), str) and normalized.get(key).strip() for key in _SEQUENCE_KEYS):
            sequence_rows += 1
        if any(isinstance(value, str) and _SHA256.fullmatch(value) for key, value in normalized.items()
               if "sequence" in key and "sha256" in key):
            sequence_hash_rows += 1
        if normalized.get("taxonomy_id") is not None or normalized.get("ncbitaxonomy") is not None:
            taxonomy_rows += 1
        if any(key in _PAGE_KEYS for key in normalized):
            page_proof = True
    return counts, len(records), sequence_rows, sequence_hash_rows > 0, page_proof


def _source_kind(source_id: str, relative_path: str, payload: Any) -> str:
    lower = relative_path.casefold()
    if source_id == "offline_data" or "tcmsp" in lower:
        return "static_reference"
    return "legacy_unverified"


def _audit_file(source_id: str, source_root: Path, path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    relative = path.relative_to(source_root).as_posix()
    suffix = path.suffix.casefold()
    base: dict[str, Any] = {
        "relative_path": relative,
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "format": suffix.lstrip(".") or "binary",
        "parse_status": "not_applicable",
        "record_count": None,
        "collection_counts": {},
        "fields_present": [],
        "field_counts": {},
        "id_values": {},
        "source_kind": "legacy_unverified",
        "declared_source": None,
        "sequence_record_count": 0,
        "sequence_hash_present": False,
        "page_proof_present": False,
    }
    payload: Any = None
    if suffix == ".json":
        try:
            payload = _decode_json(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            base.update(parse_status="invalid_json", parse_error=str(exc), source_kind="legacy_unverified")
            return base
        base["parse_status"] = "parsed"
        collection_counts = _collections(payload)
        record_count = len(payload) if isinstance(payload, list) else next(iter(collection_counts.values())) if len(collection_counts) == 1 else None
        fields, _row_count, seq_count, seq_hash, page_proof = _row_field_stats(payload)
        base.update(
            record_count=record_count,
            collection_counts=collection_counts,
            fields_present=sorted(fields),
            field_counts=fields,
            id_values=_flatten_ids(payload),
            sequence_record_count=seq_count,
            sequence_hash_present=seq_hash,
            page_proof_present=page_proof,
            source_kind=_source_kind(source_id, relative, payload),
        )
        if source_id == "offline_data" and isinstance(payload, Mapping):
            core_payload = {key: value for key, value in payload.items() if str(key).casefold() != "tcmsp"}
            base["id_values"] = _flatten_ids(core_payload)
            static_reference = payload.get("tcmsp")
            if isinstance(static_reference, Mapping):
                base["embedded_static_reference"] = {
                    "source_kind": "static_reference",
                    "declared_source": static_reference.get("source"),
                    "collection_counts": _collections(static_reference),
                    "id_values": _flatten_ids(static_reference),
                    "label_statistics_status": "not_evaluated_static_reference",
                    "qualified_label_counts": None,
                }
        if isinstance(payload, Mapping) and isinstance(payload.get("source"), str):
            base["declared_source"] = payload.get("source")
    elif suffix in {".csv", ".tsv"}:
        try:
            text = raw.decode("utf-8-sig")
            delimiter = "\t" if suffix == ".tsv" else ","
            rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
        except (UnicodeDecodeError, csv.Error) as exc:
            base.update(parse_status="invalid_table", parse_error=str(exc))
            return base
        fields = set(rows[0]) if rows else set()
        counts = {field: sum(1 for row in rows if row.get(field) not in (None, "")) for field in sorted(fields)}
        ids: dict[str, set[str]] = {}
        for row in rows:
            for key, value in row.items():
                if key and key.casefold() in _ID_FIELDS and value:
                    ids.setdefault(key.casefold(), set()).add(str(value).strip())
        base.update(
            parse_status="parsed",
            record_count=len(rows),
            collection_counts={"$rows": len(rows)},
            fields_present=sorted(field for field in fields if field),
            field_counts=counts,
            id_values={key: sorted(values) for key, values in sorted(ids.items())},
        )
    return base


def _root_metrics(source_id: str, files: Sequence[Mapping[str, Any]], source_kind: str) -> dict[str, Any]:
    ids: dict[str, set[str]] = {}
    for file in files:
        if source_id != "offline_data" and file.get("source_kind") == "static_reference":
            continue
        for key, values in file.get("id_values", {}).items():
            ids.setdefault(key, set()).update(str(value) for value in values)
    def id_values(*keys: str) -> set[str]:
        return set().union(*(ids.get(key, set()) for key in keys))

    chembl_files = [row for row in files if Path(str(row.get("relative_path", ""))).name == "chembl_luteolin_targets.json"]
    uniprot_files = [row for row in files if Path(str(row.get("relative_path", ""))).name == "uniprot_proteins.json"]
    string_files = [row for row in files if Path(str(row.get("relative_path", ""))).name == "string_ppi_network.json"]
    compound_files = [row for row in files if Path(str(row.get("relative_path", ""))).name == "compound_info.csv"]
    def best_count(rows: Sequence[Mapping[str, Any]]) -> int:
        return max((int(row.get("record_count") or 0) for row in rows), default=0)

    activity_count = best_count(chembl_files)
    protein_count = best_count(uniprot_files) or len(id_values("accession", "primaryaccession", "uniprot_id"))
    ppi_count = best_count(string_files)
    molecule_count = best_count(compound_files) or len(id_values("molecule_chembl_id", "chembl_id"))
    target_count = len(id_values("target_chembl_id", "target_id"))
    if source_id == "offline_data":
        offline = next((row for row in files if Path(str(row.get("relative_path", ""))).name == "offline_data.json"), {})
        collections = offline.get("collection_counts", {})
        activity_count = 0
        molecule_count = collections.get("$.compound", 0)
        target_count = collections.get("$.targets", target_count)
        protein_count = collections.get("$.proteins", protein_count)
        ppi_count = collections.get("$.interactions", collections.get("$.ppi", collections.get("$.edges", 0)))
    sequence_count = sum(int(row.get("sequence_record_count") or 0) for row in files)
    missing_activity_fields = list(_ACTIVITY_REQUIRED_FIELDS)
    if chembl_files:
        present = set().union(*(set(row.get("fields_present", [])) for row in chembl_files))
        missing_activity_fields = [field for field in _ACTIVITY_REQUIRED_FIELDS if field not in present]
    missing_protein_fields = list(_PROTEIN_REQUIRED_FIELDS)
    if uniprot_files:
        present = set().union(*(set(row.get("fields_present", [])) for row in uniprot_files))
        missing_protein_fields = [field for field in _PROTEIN_REQUIRED_FIELDS if field not in present]
    label_statistics_status = "not_evaluated" if activity_count else "not_evaluated_no_activity_records"
    no_source_activity_rows = activity_count == 0
    entity_reason = (
        f"{source_id}: legacy source rows do not carry a source-verified canonical entity bundle; "
        "canonical molecule/protein identity and sequence verification were not run."
    )
    cluster_reason = (
        f"{source_id}: source-verified entities are unavailable, so molecule/protein clustering was not evaluated."
    )
    split_reason = (
        f"{source_id}: canonical grouping, approved label policy, cluster construction, and double-cold splitting were not run."
    )
    excluded_entity_reason = (
        f"{source_id}: canonical entity eligibility and exclusion rules were not evaluated for this legacy source."
    )
    excluded_label_reason = (
        f"{source_id}: canonical evidence grouping and label-policy exclusion dispositions were not evaluated."
    )

    def unevaluated_count(reason: str) -> dict[str, Any]:
        return {"count": None, "status": "not_evaluated", "reason": reason}

    split_support: dict[str, Any] = {
        "status": "not_evaluated",
        "reason": split_reason,
        "splits": {
            split_name: {
                label: unevaluated_count(split_reason)
                for label in ("active", "inactive")
            }
            for split_name in ("train", "validation", "test")
        },
    }
    return {
        "scope_kind": "legacy_source_root",
        "label_statistics_status": label_statistics_status,
        "activity_row_audit": {
            "status": label_statistics_status,
            "observed_source_rows": activity_count,
            "unverified_activity_records": activity_count,
            "qualified_label_counts": None,
            "reason": "Source audit counts legacy source rows but does not run canonical grouping or the approved label policy.",
        },
        "observed_counts": {
            "compound_records": molecule_count,
            "distinct_molecule_ids": len(id_values("molecule_chembl_id", "chembl_id")),
            "activity_records": activity_count,
            "activity_records_status": "no_source_activity_rows_observed" if no_source_activity_rows else "unverified_legacy_source_rows",
            "unverified_activity_records": activity_count,
            "distinct_target_ids": target_count,
            "protein_records": protein_count,
            "distinct_accessions": len(id_values("accession", "primaryaccession", "uniprot_id")),
            "records_with_sequence": sequence_count,
            "ppi_edges": ppi_count,
            "confirmed_active": 0 if no_source_activity_rows else None,
            "confirmed_inactive": 0 if no_source_activity_rows else None,
            "gray": None,
            "unknown": None,
            "quarantine": None,
        },
        "entity_coverage": {
            "molecule_records": {
                "observed_count": molecule_count,
                "verified_count": None,
                "status": "not_evaluated",
                "reason": entity_reason,
            },
            "protein_records": {
                "observed_count": protein_count,
                "verified_count": None,
                "status": "not_evaluated",
                "reason": entity_reason,
            },
        },
        "cluster_coverage": {
            "molecule_cluster_count": unevaluated_count(cluster_reason),
            "protein_cluster_count": unevaluated_count(cluster_reason),
        },
        "double_cold_split_support": split_support,
        "excluded": {
            "entities": unevaluated_count(excluded_entity_reason),
            "labels": unevaluated_count(excluded_label_reason),
        },
        "ids": {key: sorted(values) for key, values in sorted(ids.items())},
        "missing_activity_fields": missing_activity_fields,
        "missing_protein_fields": missing_protein_fields,
        "sequence_hash_present": any(bool(row.get("sequence_hash_present")) for row in files),
        "page_proof_present": any(bool(row.get("page_proof_present")) for row in files),
        "source_kind": source_kind,
    }


def compute_readiness(source_report: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source_report, Mapping):
        raise SourceAuditError("source report must be an object")
    counts = source_report.get("observed_counts", {})
    counts = counts if isinstance(counts, Mapping) else {}
    reasons: list[str] = []
    if source_report.get("source_kind") != "online_snapshot":
        reasons.append("source_is_legacy_or_static_not_verified_online_snapshot")
    if int(counts.get("compound_records") or 0) < 2:
        reasons.append("fewer_than_two_distinct_real_compounds")
    label_status = source_report.get("label_statistics_status")
    if label_status == "not_evaluated":
        reasons.append("source_label_statistics_not_evaluated")
    elif label_status == "not_evaluated_no_activity_records":
        reasons.append("no_activity_records_available_for_source_label_classification")
    elif label_status == "evaluated":
        if type(counts.get("confirmed_active")) is not int or counts["confirmed_active"] == 0:
            reasons.append("no_source_verified_confirmed_active_measurements")
        if type(counts.get("confirmed_inactive")) is not int or counts["confirmed_inactive"] == 0:
            reasons.append("no_source_verified_confirmed_inactive_measurements")
    missing_activity = source_report.get("missing_activity_fields", [])
    if missing_activity:
        reasons.append("activity_evidence_missing_fields:" + ",".join(sorted(str(item) for item in missing_activity)))
    missing_protein = source_report.get("missing_protein_fields", [])
    if missing_protein:
        reasons.append("protein_sequence_evidence_missing_fields:" + ",".join(sorted(str(item) for item in missing_protein)))
    if int(counts.get("records_with_sequence") or 0) == 0 or not source_report.get("sequence_hash_present"):
        reasons.append("no_verified_full_protein_sequence_and_sequence_hash")
    if not source_report.get("page_proof_present"):
        reasons.append("no_complete_source_page_chain_proof")
    if source_report.get("source_id") == "offline_data":
        reasons.append("offline_reference_is_static_not_measurement_evidence")
    reasons.append("no_independent_double_cold_split_support_was_established_by_this_legacy_source_audit")
    return {"training_ready": False, "reasons": sorted(set(reasons))}


def _source_report(apl_root: Path, source_id: str, relative: str, default_kind: str) -> dict[str, Any]:
    source_root = apl_root / Path(relative)
    if source_root.is_file() and not _is_reparse(source_root):
        files = [_audit_file(source_id, source_root.parent, source_root)]
        source_path_exists = True
        source_issues: list[str] = []
        root_bytes = source_root.read_bytes()
        if source_id == "offline_data":
            try:
                payload = _decode_json(root_bytes)
                collections = _collections(payload)
                for row in files:
                    row["collection_counts"] = collections
                    row["record_count"] = None
                    row["source_kind"] = "static_reference"
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                pass
    else:
        files, source_issues = _list_regular_files(source_root)
        source_path_exists = source_root.exists()
        files = [_audit_file(source_id, source_root, path) for path in files]
    source_kind = "static_reference" if source_id == "offline_data" else default_kind
    if any(file.get("source_kind") == "static_reference" for file in files):
        source_kind = "static_reference" if source_id == "offline_data" else "legacy_unverified"
    file_receipts = [{"relative_path": row["relative_path"], "bytes": row["bytes"], "sha256": row["sha256"]} for row in files]
    base = {
        "source_id": source_id,
        "source_path": relative,
        "source_exists": source_path_exists,
        "source_kind": source_kind,
        "source_sha256": hashlib.sha256(_canonical_bytes(file_receipts)).hexdigest(),
        "file_count": len(files),
        "files": files,
        "source_issues": sorted(source_issues),
    }
    base.update(_root_metrics(source_id, files, source_kind))
    readiness = compute_readiness(base)
    base.update(readiness)
    if not source_path_exists:
        base["reasons"] = sorted(set(base["reasons"] + ["source_path_missing"]))
    if source_issues:
        base["reasons"] = sorted(set(base["reasons"] + ["source_tree_contains_skipped_reparse_paths"]))
    return base


def _validate_report_output(apl_root: Path, output_dir: Any) -> Path:
    if output_dir is None or not os.fspath(output_dir).strip():
        raise SourceAuditError("an explicit new --out directory is required")
    candidate = Path(output_dir).expanduser()
    if not candidate.is_absolute():
        raise SourceAuditError("report output path must be absolute")
    cursor = candidate
    while True:
        if cursor.exists() and _is_reparse(cursor):
            raise SourceAuditError(f"report output crosses a symlink or junction: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    resolved = candidate.resolve(strict=False)
    protected = [
        apl_root / relative for relative in (
            "data", "framework_output", "offline_data.json", ".model_cache", "features", "output_dashboard.html",
        )
    ]
    if any(_path_intersects(resolved, path) for path in protected):
        raise SourceAuditError("report output intersects a protected historical source or output path")
    for source_id, relative, _kind in _SOURCE_SPECS:
        source_path = apl_root / Path(relative)
        if _path_intersects(resolved, source_path):
            raise SourceAuditError(f"report output intersects source {source_id}")
    if resolved.exists():
        raise SourceAuditError("report output directory already exists; refusing to overwrite")
    return resolved


def audit_legacy_sources(apl_root: os.PathLike[str] | str, *, output_dir: Any = None) -> dict[str, Any]:
    """Audit the three legacy roots separately; never add their counts together."""
    root = _assert_safe_source_root(Path(apl_root))
    sources = [_source_report(root, source_id, relative, kind) for source_id, relative, kind in _SOURCE_SPECS]
    code_version_control = _git_state(root)
    collection: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "apl_root": str(root),
        "code_version_control": code_version_control,
        "sources": sources,
        "training_ready": False,
        "note": "Each source is audited independently; file and record counts are not summed across roots.",
        "report_paths": {},
    }
    collection["semantic_sha256"] = semantic_manifest_hash(collection)
    if output_dir is not None:
        destination = _validate_report_output(root, output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        report_paths: dict[str, str] = {}
        for source in sources:
            path = destination / f"{source['source_id']}.json"
            report = dict(source)
            report["code_version_control"] = code_version_control
            raw = _canonical_bytes(report)
            temp = path.with_name(path.name + ".tmp")
            with temp.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
            report_paths[source["source_id"]] = str(path)
        collection["report_paths"] = report_paths
        collection["semantic_sha256"] = semantic_manifest_hash(collection)
    return collection


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit legacy APL data roots and write separate provenance reports")
    parser.add_argument("--apl-root", required=True, help="absolute APL code/data root")
    parser.add_argument("--out", required=True, help="new independent report output directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_argument_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        result = audit_legacy_sources(args.apl_root, output_dir=args.out)
    except (SourceAuditError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "incomplete", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

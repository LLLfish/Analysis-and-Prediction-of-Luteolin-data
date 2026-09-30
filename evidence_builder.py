"""Build versioned, provenance-preserving activity evidence bundles.

The module deliberately separates measured evidence from the legacy display
projection. Local manifests are the default input; network retrieval is an
explicit, bounded CLI operation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests

from data_contracts import canonical_json_sha256


ENTITY_SCHEMA_VERSION = 2
ACTIVITY_SCHEMA_VERSION = 1
CHEMBL_ORIGIN = "https://www.ebi.ac.uk"
CHEMBL_API_ROOT = f"{CHEMBL_ORIGIN}/chembl/api/data"
ALLOWED_SOURCE_KINDS = {
    "online_snapshot",
    "legacy_unverified",
    "static_reference",
    "synthetic_test",
}
CHEMBL_ID = re.compile(r"^CHEMBL[0-9]+$")
_MAX_JSON_DEPTH = 64


class EvidenceInputError(ValueError):
    """Raised when source input cannot be interpreted without guessing."""


class BoundedFetchError(RuntimeError):
    """A bounded request stopped before all source data was captured."""


@dataclass(frozen=True)
class ActivityEvidence:
    """One ChEMBL activity row plus its unchanged source evidence."""

    schema_version: int
    source_id: str
    source_kind: str
    source_version: Optional[str]
    source_path: Optional[str]
    source_sha256: Optional[str]
    molecule_chembl_id: Optional[str]
    activity_id: Any
    assay_chembl_id: Optional[str]
    document_chembl_id: Optional[str]
    target_chembl_id: Optional[str]
    target_type: Optional[str]
    target_organism: Optional[str]
    target_taxonomy_id: Optional[int]
    target_components: list[dict[str, Any]]
    confidence_score: Any
    standard_type: Any
    standard_relation: Any
    standard_value: Any
    standard_units: Any
    pchembl_value: Any
    raw_activity_record: dict[str, Any]
    response_sha256: str
    page_index: int
    page_status: str
    pagination_status: str
    evidence_status: str
    missing_fields: list[str] = field(default_factory=list)
    entity_bundle_version: Optional[str] = None
    entity_bundle_hash: Optional[str] = None


@dataclass(frozen=True)
class ActivityPage:
    """A raw API page with a stable response fingerprint and receipt."""

    molecule_chembl_id: str
    page_index: int
    payload: dict[str, Any]
    response_sha256: str
    source_kind: str
    source_version: Optional[str]
    source_path: Optional[str]
    source_sha256: Optional[str]
    page_status: str
    next_url: Optional[str]
    activity_count: int


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_kind(value: Any, default: Optional[str] = None) -> str:
    if value is None:
        value = default
    if not isinstance(value, str) or value not in ALLOWED_SOURCE_KINDS:
        raise EvidenceInputError(f"unsupported or missing source_kind: {value!r}")
    return value


def _taxonomy_adapter(value: Any) -> Optional[int]:
    """Accept official JSON integer or decimal-string taxids at the boundary."""
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _copy_json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceInputError(f"{label} must be a JSON object")
    try:
        _json_bytes(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise EvidenceInputError(f"{label} is not finite JSON data: {exc}") from exc
    return deepcopy(value)


def _validate_activity_page(page: Any, expected_offset: Optional[int] = None) -> tuple[dict[str, Any], list[dict[str, Any]], Optional[str]]:
    page_obj = _copy_json_object(page, "activity page")
    rows = page_obj.get("activities")
    meta = page_obj.get("page_meta")
    if not isinstance(rows, list) or not isinstance(meta, dict):
        raise EvidenceInputError("activity page requires list activities and object page_meta")
    if any(not isinstance(row, dict) for row in rows):
        raise EvidenceInputError("every activity item must be an object")
    if "next" not in meta or (meta["next"] is not None and not isinstance(meta["next"], str)):
        raise EvidenceInputError("page_meta.next must be null or a URL string")
    offset = meta.get("offset")
    if expected_offset is not None and (type(offset) is not int or offset != expected_offset):
        raise EvidenceInputError(
            f"activity page offset mismatch: expected {expected_offset}, got {offset!r}"
        )
    total = meta.get("total_count")
    if total is not None and (type(total) is not int or total < 0):
        raise EvidenceInputError("page_meta.total_count must be a non-negative integer or null")
    return page_obj, rows, meta["next"]


def _page_chain_status(pages: Sequence[Mapping[str, Any]], row_count: int) -> str:
    if not pages:
        return "incomplete"
    try:
        first_meta = pages[0]["page_meta"]
        if not isinstance(first_meta, dict) or first_meta.get("offset") != 0:
            return "incomplete"
        expected_offset = 0
        totals: set[int] = set()
        for index, page in enumerate(pages):
            _, rows, next_url = _validate_activity_page(page, expected_offset)
            meta = page["page_meta"]
            expected_offset += len(rows)
            if type(meta.get("total_count")) is int:
                totals.add(meta["total_count"])
            if index < len(pages) - 1 and next_url is None:
                return "incomplete"
            if index == len(pages) - 1 and next_url is not None:
                return "incomplete"
        if len(totals) > 1:
            return "incomplete"
        if totals and next(iter(totals)) != row_count:
            return "incomplete"
    except (EvidenceInputError, KeyError, TypeError):
        return "incomplete"
    return "complete"


def _target_fields(row: Mapping[str, Any], target: Optional[Mapping[str, Any]]) -> tuple[
    Optional[str], Optional[str], Optional[int], list[dict[str, Any]]
]:
    target_obj = target if isinstance(target, Mapping) else {}
    target_type = target_obj.get("target_type", row.get("target_type"))
    organism = target_obj.get("organism", row.get("target_organism"))
    taxonomy = _taxonomy_adapter(target_obj.get("tax_id", row.get("target_tax_id")))
    components = target_obj.get("target_components", row.get("target_components", []))
    if not isinstance(components, list):
        components = []
    copied_components = [deepcopy(item) for item in components if isinstance(item, dict)]
    return (
        target_type if isinstance(target_type, str) else None,
        organism if isinstance(organism, str) else None,
        taxonomy,
        copied_components,
    )


def activity_evidence_from_pages(
    molecule_chembl_id: str,
    pages: Sequence[Mapping[str, Any]],
    *,
    source_kind: str,
    source_version: Optional[str] = None,
    page_receipts: Optional[Sequence[Mapping[str, Any]]] = None,
    targets: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[ActivityEvidence]:
    """Convert raw pages into evidence without dropping source fields."""
    if not isinstance(molecule_chembl_id, str) or not CHEMBL_ID.fullmatch(molecule_chembl_id):
        raise EvidenceInputError("molecule_chembl_id must be a ChEMBL identifier")
    source_kind = _source_kind(source_kind)
    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
        raise EvidenceInputError("pages must be a sequence of raw page objects")
    receipts = list(page_receipts or [])
    target_map = targets or {}
    copied_pages: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    page_rows: list[tuple[dict[str, Any], list[dict[str, Any]], str]] = []
    expected_offset = 0
    for index, raw_page in enumerate(pages):
        page_obj, rows, _ = _validate_activity_page(raw_page, expected_offset)
        source_page = page_obj.get("raw_response", page_obj)
        if not isinstance(source_page, dict):
            raise EvidenceInputError("raw_response must be an object when present")
        receipt = receipts[index] if index < len(receipts) else {}
        response_hash = receipt.get("response_sha256") if isinstance(receipt, Mapping) else None
        if response_hash is None:
            response_hash = canonical_json_sha256(source_page)
        elif not _is_sha256(response_hash):
            raise EvidenceInputError(f"page {index + 1} response hash is invalid")
        source_hash = receipt.get("source_sha256") if isinstance(receipt, Mapping) else None
        if receipt.get("source_path") and _is_sha256(source_hash) and source_hash != response_hash:
            raise EvidenceInputError(f"page {index + 1} raw source hash differs from response hash")
        copied_pages.append(page_obj)
        page_rows.append((page_obj, rows, response_hash))
        all_rows.extend(rows)
        expected_offset += len(rows)
    pagination_status = _page_chain_status(copied_pages, len(all_rows))

    evidence: list[ActivityEvidence] = []
    for index, (page_obj, rows, response_hash) in enumerate(page_rows):
        receipt = receipts[index] if index < len(receipts) else {}
        page_status = "complete" if isinstance(page_obj.get("page_meta"), dict) else "invalid"
        raw_path = receipt.get("source_path") if isinstance(receipt, Mapping) else None
        raw_hash = receipt.get("source_sha256") if isinstance(receipt, Mapping) else None
        page_meta = page_obj["page_meta"]
        for row in rows:
            raw = _copy_json_object(row, "activity record")
            row_molecule_id = raw.get("molecule_chembl_id")
            molecule_id = row_molecule_id if isinstance(row_molecule_id, str) else molecule_chembl_id
            # A row's molecule id is not synthesized; the query id remains a separately named scope.
            evidence_molecule_id = row_molecule_id if isinstance(row_molecule_id, str) else None
            target_id = raw.get("target_chembl_id")
            target_id_key = target_id if isinstance(target_id, str) else ""
            target = target_map.get(target_id_key)
            target_type, organism, taxonomy, components = _target_fields(raw, target)
            missing = [
                name for name, value in (
                    ("activity_id", raw.get("activity_id")),
                    ("assay_chembl_id", raw.get("assay_chembl_id")),
                    ("document_chembl_id", raw.get("document_chembl_id")),
                    ("molecule_chembl_id", evidence_molecule_id),
                    ("target_chembl_id", target_id if isinstance(target_id, str) else None),
                    ("target_type", target_type),
                    ("target_components", components if components else None),
                ) if value is None
            ]
            evidence_status = "observed" if not missing and source_kind in {
                "online_snapshot", "synthetic_test"
            } else "unverified"
            evidence.append(ActivityEvidence(
                schema_version=ACTIVITY_SCHEMA_VERSION,
                source_id="chembl_activity",
                source_kind=source_kind,
                source_version=source_version,
                source_path=raw_path if isinstance(raw_path, str) else None,
                source_sha256=raw_hash if _is_sha256(raw_hash) else None,
                molecule_chembl_id=evidence_molecule_id,
                activity_id=raw.get("activity_id"),
                assay_chembl_id=(raw.get("assay_chembl_id") if isinstance(raw.get("assay_chembl_id"), str) else None),
                document_chembl_id=(raw.get("document_chembl_id") if isinstance(raw.get("document_chembl_id"), str) else None),
                target_chembl_id=target_id if isinstance(target_id, str) else None,
                target_type=target_type,
                target_organism=organism,
                target_taxonomy_id=taxonomy,
                target_components=components,
                confidence_score=raw.get("confidence_score", raw.get("target_confidence")),
                standard_type=raw.get("standard_type"),
                standard_relation=raw.get("standard_relation"),
                standard_value=raw.get("standard_value"),
                standard_units=raw.get("standard_units"),
                pchembl_value=raw.get("pchembl_value"),
                raw_activity_record=raw,
                response_sha256=response_hash,
                page_index=index + 1,
                page_status=page_status,
                pagination_status=pagination_status,
                evidence_status=evidence_status,
                missing_fields=missing,
            ))
    return evidence


def activity_evidence_from_collector_payload(payload: Mapping[str, Any]) -> list[ActivityEvidence]:
    """Normalize collector output; old truncated rows stay unverified."""
    if not isinstance(payload, Mapping):
        raise EvidenceInputError("collector ChEMBL payload must be an object")
    declared_kind = payload.get("source_kind")
    if declared_kind not in ALLOWED_SOURCE_KINDS:
        declared_kind = "legacy_unverified"
    molecule_id = payload.get("chembl_id")
    if not isinstance(molecule_id, str) or not CHEMBL_ID.fullmatch(molecule_id):
        molecule_id = None
    if isinstance(payload.get("activity_pages"), list):
        if molecule_id is None:
            raise EvidenceInputError("collector payload is missing its queried ChEMBL molecule id")
        return activity_evidence_from_pages(
            molecule_id,
            payload["activity_pages"],
            source_kind=declared_kind,
            source_version=payload.get("source_version"),
            page_receipts=payload.get("page_receipts"),
            targets=payload.get("target_records", {}),
        )
    raw_activities = payload.get("activities")
    if isinstance(raw_activities, list):
        pages = [{
            "page_meta": {"limit": len(raw_activities), "next": None, "offset": 0,
                          "previous": None, "total_count": len(raw_activities)},
            "activities": raw_activities,
        }]
        kind = declared_kind if declared_kind == "online_snapshot" else "legacy_unverified"
        return activity_evidence_from_pages(
            molecule_id, pages, source_kind=kind,
            source_version=payload.get("source_version"),
            targets=payload.get("target_records", {}),
        )
    legacy_rows = payload.get("targets", [])
    if not isinstance(legacy_rows, list):
        return []
    evidence: list[ActivityEvidence] = []
    for index, value in enumerate(legacy_rows):
        if not isinstance(value, dict):
            continue
        raw = deepcopy(value)
        response_hash = canonical_json_sha256(raw)
        missing = [name for name in (
            "activity_id", "assay_chembl_id", "document_chembl_id",
            "molecule_chembl_id", "target_components",
        )]
        evidence.append(ActivityEvidence(
            schema_version=ACTIVITY_SCHEMA_VERSION,
            source_id="chembl_legacy_row",
            source_kind="legacy_unverified",
            source_version=None,
            source_path=None,
            source_sha256=None,
            molecule_chembl_id=None,
            activity_id=None,
            assay_chembl_id=None,
            document_chembl_id=None,
            target_chembl_id=(raw.get("target_chembl_id") if isinstance(raw.get("target_chembl_id"), str) else None),
            target_type=None,
            target_organism=(raw.get("organism") if isinstance(raw.get("organism"), str) else None),
            target_taxonomy_id=None,
            target_components=[],
            confidence_score=None,
            standard_type=raw.get("activity_type"),
            standard_relation=None,
            standard_value=raw.get("activity_value"),
            standard_units=raw.get("activity_unit"),
            pchembl_value=raw.get("pchembl_value"),
            raw_activity_record=raw,
            response_sha256=response_hash,
            page_index=index + 1,
            page_status="legacy_snapshot",
            pagination_status="unknown",
            evidence_status="unverified",
            missing_fields=missing,
        ))
    return evidence


class LegacyDisplayAdapter:
    """One-way projection for the historical single-compound report schema."""

    @staticmethod
    def targets(activity_rows: Iterable[ActivityEvidence | Mapping[str, Any]]) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        for item in activity_rows:
            row = asdict(item) if isinstance(item, ActivityEvidence) else dict(item)
            raw = row.get("raw_activity_record")
            raw = raw if isinstance(raw, Mapping) else {}
            projected.append({
                "target_chembl_id": row.get("target_chembl_id"),
                "target_name": raw.get("target_pref_name") or raw.get("pref_name") or raw.get("target_name"),
                "gene_symbol": raw.get("gene_symbol"),
                "organism": row.get("target_organism"),
                "activity_type": row.get("standard_type"),
                "activity_relation": row.get("standard_relation"),
                "activity_value": row.get("standard_value"),
                "activity_unit": row.get("standard_units"),
                "pchembl_value": row.get("pchembl_value"),
                "evidence_status": row.get("evidence_status", "unverified"),
                "source_kind": row.get("source_kind", "legacy_unverified"),
                "activity_id": row.get("activity_id"),
                "assay_chembl_id": row.get("assay_chembl_id"),
                "document_chembl_id": row.get("document_chembl_id"),
            })
        return projected


def _bundle_hash_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(bundle))
    payload.pop("content_sha256", None)
    payload.pop("entity_version", None)
    for activity in payload.get("activities", []):
        if isinstance(activity, dict):
            activity.pop("entity_bundle_hash", None)
            activity.pop("entity_bundle_version", None)
    return payload


def _safe_relative_file(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise EvidenceInputError("source_path must be a non-empty relative path")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvidenceInputError(f"source_path escapes bundle root: {relative_path!r}")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath((str(root_resolved), str(resolved))) != str(root_resolved):
        raise EvidenceInputError(f"source_path escapes bundle root: {relative_path!r}")
    return resolved


def load_entity_bundle(path: os.PathLike[str] | str, *, verify_sources: bool = True) -> dict[str, Any]:
    """Load and verify canonical bundle, bundle hash, activity links and raw files."""
    bundle_path = Path(path).resolve()
    with bundle_path.open("r", encoding="utf-8-sig") as stream:
        bundle = json.load(stream)
    if not isinstance(bundle, dict) or type(bundle.get("schema_version")) is not int:
        raise EvidenceInputError("entity bundle requires an integer schema_version")
    if bundle["schema_version"] != ENTITY_SCHEMA_VERSION:
        raise EvidenceInputError("unsupported entity bundle schema_version")
    expected_hash = bundle.get("content_sha256")
    if not _is_sha256(expected_hash):
        raise EvidenceInputError("entity bundle content_sha256 is missing or invalid")
    computed_hash = canonical_json_sha256(_bundle_hash_payload(bundle))
    if computed_hash != expected_hash:
        raise EvidenceInputError("entity bundle content hash mismatch")
    expected_version = f"entities-v{ENTITY_SCHEMA_VERSION}-{expected_hash[:16]}"
    if bundle.get("entity_version") != expected_version:
        raise EvidenceInputError("entity bundle entity_version mismatch")
    if not isinstance(bundle.get("activities"), list):
        raise EvidenceInputError("entity bundle activities must be a list")
    for activity in bundle["activities"]:
        if not isinstance(activity, dict):
            raise EvidenceInputError("entity bundle activity must be an object")
        if activity.get("entity_bundle_hash") != expected_hash or activity.get("entity_bundle_version") != expected_version:
            raise EvidenceInputError("activity references a different entity bundle")
    provenance = bundle.get("source_provenance")
    if not isinstance(provenance, list):
        raise EvidenceInputError("entity bundle source_provenance must be a list")
    if verify_sources:
        for source in provenance:
            if not isinstance(source, dict) or not _is_sha256(source.get("raw_sha256")):
                raise EvidenceInputError("invalid raw source provenance entry")
            source_path = _safe_relative_file(bundle_path.parent, source.get("source_path"))
            if not source_path.is_file():
                raise EvidenceInputError(f"raw source file is missing: {source.get('source_path')}")
            if _sha256_bytes(source_path.read_bytes()) != source["raw_sha256"]:
                raise EvidenceInputError(f"raw source hash mismatch: {source.get('source_path')}")
    return bundle


def write_entity_bundle(
    output_dir: os.PathLike[str] | str,
    *,
    activities: Sequence[ActivityEvidence | Mapping[str, Any]],
    source_provenance: Sequence[Mapping[str, Any]],
    molecules: Optional[Sequence[Mapping[str, Any]]] = None,
    proteins: Optional[Sequence[Mapping[str, Any]]] = None,
    exclusions: Optional[Sequence[Mapping[str, Any]]] = None,
    source_kind: Optional[str] = None,
) -> dict[str, Any]:
    """Write deterministic entities.json in a newly created staging directory."""
    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise EvidenceInputError("output staging directory must already be created")
    activity_rows = [asdict(row) if isinstance(row, ActivityEvidence) else deepcopy(dict(row)) for row in activities]
    bundle: dict[str, Any] = {
        "schema_version": ENTITY_SCHEMA_VERSION,
        "entity_version": None,
        "source_kind": _source_kind(source_kind, "legacy_unverified") if source_kind is not None else "mixed",
        "molecules": sorted((deepcopy(dict(item)) for item in (molecules or [])), key=lambda item: str(item.get("molecule_chembl_id") or "")),
        "proteins": sorted((deepcopy(dict(item)) for item in (proteins or [])), key=lambda item: (str(item.get("accession") or ""), str(item.get("taxonomy_id") or ""))),
        "activities": activity_rows,
        "exclusions": [deepcopy(dict(item)) for item in (exclusions or [])],
        "source_provenance": sorted(
            (deepcopy(dict(item)) for item in source_provenance),
            key=lambda item: (str(item.get("source_id") or ""), str(item.get("source_path") or "")),
        ),
    }
    content_hash = canonical_json_sha256(_bundle_hash_payload(bundle))
    entity_version = f"entities-v{ENTITY_SCHEMA_VERSION}-{content_hash[:16]}"
    bundle["entity_version"] = entity_version
    bundle["content_sha256"] = content_hash
    for activity in bundle["activities"]:
        activity["entity_bundle_version"] = entity_version
        activity["entity_bundle_hash"] = content_hash
    path = root / "entities.json"
    with path.open("xb") as stream:
        stream.write(_json_bytes(bundle))
    return load_entity_bundle(path)


def _protected_output_path(path: Path) -> bool:
    apl_root = Path(__file__).resolve().parent
    protected = (
        apl_root / "data",
        apl_root / "framework_output",
        apl_root / "offline_data.json",
        apl_root / ".model_cache",
    )
    resolved = path.resolve()
    for protected_path in protected:
        target = protected_path.resolve()
        try:
            common = Path(os.path.commonpath((str(target), str(resolved))))
        except ValueError:
            continue
        if common == target or common == resolved:
            return True
    return False


def _create_output_root(output_dir: os.PathLike[str] | str) -> Path:
    if output_dir is None or not os.fspath(output_dir).strip():
        raise EvidenceInputError("an explicit --out staging directory is required")
    root = Path(output_dir).expanduser().resolve()
    if _protected_output_path(root):
        raise EvidenceInputError("output root intersects protected historical data or model paths")
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {root}")
    root.mkdir(parents=True, exist_ok=False)
    return root


def _write_raw_bytes(root: Path, relative_path: str, content: bytes) -> tuple[str, str]:
    output = _safe_relative_file(root, relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.parent.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite raw snapshot: {relative_path}")
    output.write_bytes(content)
    return relative_path.replace("\\", "/"), _sha256_bytes(content)


def _manifest_source_kind(manifest: Mapping[str, Any]) -> str:
    raw = manifest.get("source_kind", manifest.get("fixture_kind"))
    return _source_kind(raw, "legacy_unverified")


def _local_page_records(
    manifest: Mapping[str, Any],
    raw_root: Path,
    root: Path,
    manifest_path: Path,
    source_kind: str,
) -> tuple[list[ActivityEvidence], list[dict[str, Any]], str]:
    chembl = manifest.get("chembl")
    if not isinstance(chembl, dict):
        raise EvidenceInputError("local manifest requires a chembl object")
    per_molecule = chembl.get("activities_by_molecule")
    if not isinstance(per_molecule, dict):
        raise EvidenceInputError("local manifest requires activities_by_molecule")
    target_rows = chembl.get("targets", [])
    if not isinstance(target_rows, list) or any(not isinstance(item, dict) for item in target_rows):
        raise EvidenceInputError("chembl.targets must be a list of objects")
    target_map = {
        item.get("target_chembl_id"): item
        for item in target_rows
        if isinstance(item.get("target_chembl_id"), str)
    }
    all_activities: list[ActivityEvidence] = []
    receipts: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    manifest_bytes = manifest_path.read_bytes()
    manifest_relative, manifest_hash = _write_raw_bytes(root, "raw/source_manifest.json", manifest_bytes)
    provenance.append({
        "source_id": "local_manifest",
        "source_kind": source_kind,
        "source_path": manifest_relative,
        "raw_sha256": manifest_hash,
        "source_format": "json",
    })
    overall = "complete"
    for molecule_id in sorted(per_molecule):
        if not isinstance(molecule_id, str) or not CHEMBL_ID.fullmatch(molecule_id):
            raise EvidenceInputError(f"invalid molecule id in local manifest: {molecule_id!r}")
        molecule_entry = per_molecule[molecule_id]
        if not isinstance(molecule_entry, dict) or not isinstance(molecule_entry.get("pages"), list):
            raise EvidenceInputError(f"activities_by_molecule.{molecule_id}.pages must be a list")
        page_payloads = molecule_entry["pages"]
        page_receipts: list[dict[str, Any]] = []
        expected_offset = 0
        for index, raw_page in enumerate(page_payloads):
            page_obj, rows, _ = _validate_activity_page(raw_page, expected_offset)
            expected_offset += len(rows)
            raw_page_bytes = _json_bytes(page_obj)
            rel = f"raw/chembl/activities/{molecule_id}/page-{index + 1:04d}.json"
            saved_path, saved_hash = _write_raw_bytes(root, rel, raw_page_bytes)
            response_hash = canonical_json_sha256(page_obj)
            page_receipt = {
                "source_id": f"chembl_activity:{molecule_id}:page:{index + 1}",
                "source_kind": source_kind,
                "source_path": saved_path,
                "source_sha256": saved_hash,
                "response_sha256": response_hash,
            }
            page_receipts.append(page_receipt)
            provenance.append({
                "source_id": page_receipt["source_id"],
                "source_kind": source_kind,
                "source_path": saved_path,
                "raw_sha256": saved_hash,
                "response_sha256": response_hash,
            })
        status = _page_chain_status(page_payloads, expected_offset)
        if status != "complete":
            overall = "incomplete"
        all_activities.extend(activity_evidence_from_pages(
            molecule_id,
            page_payloads,
            source_kind=source_kind,
            source_version=manifest.get("chembl_release"),
            page_receipts=page_receipts,
            targets=target_map,
        ))
        receipts.extend(page_receipts)
    return all_activities, provenance, overall


def ingest_local_manifest(
    manifest_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Build a complete offline bundle from a local source manifest."""
    source_path = Path(manifest_path).resolve()
    with source_path.open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise EvidenceInputError("local manifest root must be an object")
    source_kind = _manifest_source_kind(manifest)
    root = _create_output_root(output_dir)
    try:
        activities, provenance, pagination_status = _local_page_records(
            manifest, root / "raw", root, source_path, source_kind
        )
        if pagination_status != "complete":
            partial = {
                "status": "incomplete",
                "source_kind": source_kind,
                "pagination_status": pagination_status,
                "activity_count": len(activities),
                "source_provenance": provenance,
            }
            (root / "partial_ingest.json").write_bytes(_json_bytes(partial))
            return partial
        bundle = write_entity_bundle(
            root, activities=activities, source_provenance=provenance,
            source_kind=source_kind,
        )
        return {
            "status": "complete",
            "source_kind": source_kind,
            "pagination_status": pagination_status,
            "activity_count": len(activities),
            "content_sha256": bundle["content_sha256"],
            "entity_version": bundle["entity_version"],
            "bundle_path": str(root / "entities.json"),
            "source_provenance": bundle["source_provenance"],
        }
    except Exception:
        # Raw snapshots remain for diagnosis, but no complete bundle is left behind.
        raise


def _validate_chembl_next(next_url: Any, current_url: str) -> Optional[str]:
    if next_url is None:
        return None
    if not isinstance(next_url, str) or not next_url.strip():
        raise EvidenceInputError("page_meta.next must be null or a non-empty URL")
    if next_url.startswith("//"):
        raise EvidenceInputError("protocol-relative page_meta.next is not allowed")
    parsed = urlparse(next_url)
    if parsed.scheme:
        if parsed.scheme != "https" or parsed.hostname != "www.ebi.ac.uk" or parsed.username or parsed.password:
            raise EvidenceInputError("page_meta.next host must remain https://www.ebi.ac.uk")
        absolute = next_url
    else:
        if not next_url.startswith("/"):
            raise EvidenceInputError("relative page_meta.next must start with '/'")
        absolute = urljoin(CHEMBL_ORIGIN, next_url)
    resolved = urlparse(absolute)
    if resolved.hostname != "www.ebi.ac.uk" or resolved.scheme != "https":
        raise EvidenceInputError("page_meta.next host must remain https://www.ebi.ac.uk")
    if resolved.path not in {
        "/chembl/api/data/activity",
        "/chembl/api/data/activity.json",
    }:
        raise EvidenceInputError("page_meta.next must continue the ChEMBL activity resource")
    if resolved.fragment:
        raise EvidenceInputError("page_meta.next fragment is not allowed")
    return absolute


def _bounded_get_json(
    url: str,
    *,
    timeout: float,
    max_response_bytes: int,
    http_get: Optional[Callable[..., Any]] = None,
) -> tuple[dict[str, Any], bytes, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.ebi.ac.uk" or parsed.username or parsed.password:
        raise EvidenceInputError("ChEMBL request host must be https://www.ebi.ac.uk")
    getter = http_get or requests.get
    response = getter(
        url,
        headers={"Accept": "application/json"},
        timeout=timeout,
        stream=True,
        allow_redirects=False,
    )
    response_url = getattr(response, "url", url)
    actual = urlparse(response_url)
    if actual.scheme != "https" or actual.hostname != "www.ebi.ac.uk":
        raise EvidenceInputError("response URL escaped the ChEMBL host allowlist")
    status_code = getattr(response, "status_code", 200)
    if type(status_code) is not int or 300 <= status_code < 400:
        raise BoundedFetchError(f"redirect or invalid HTTP status from ChEMBL: {status_code!r}")
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    headers = getattr(response, "headers", {})
    content_length = headers.get("Content-Length") if isinstance(headers, Mapping) else None
    if content_length is not None:
        try:
            if int(content_length) > max_response_bytes:
                raise BoundedFetchError("ChEMBL response exceeds max_response_bytes")
        except ValueError as exc:
            raise BoundedFetchError("invalid ChEMBL Content-Length") from exc
    chunks: list[bytes] = []
    total = 0
    iterator = response.iter_content(chunk_size=min(max_response_bytes + 1, 65536))
    for chunk in iterator:
        if not chunk:
            continue
        if not isinstance(chunk, bytes):
            raise BoundedFetchError("HTTP body stream returned non-byte content")
        total += len(chunk)
        if total > max_response_bytes:
            raise BoundedFetchError("ChEMBL response exceeds max_response_bytes")
        chunks.append(chunk)
    body = b"".join(chunks)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundedFetchError(f"ChEMBL response is not UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BoundedFetchError("ChEMBL response JSON root must be an object")
    return payload, body, _sha256_bytes(body)


def _validate_budget(name: str, value: Any, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _collect_chembl(
    molecule_ids: Sequence[str],
    *,
    max_pages: int,
    max_records: int,
    timeout: float,
    max_requests: int,
    max_response_bytes: int,
    http_get: Optional[Callable[..., Any]] = None,
    interruption: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    if not isinstance(molecule_ids, Sequence) or isinstance(molecule_ids, (str, bytes)) or not molecule_ids:
        raise ValueError("molecule_ids must be a non-empty sequence")
    unique_ids: list[str] = []
    for value in molecule_ids:
        if not isinstance(value, str) or not CHEMBL_ID.fullmatch(value):
            raise ValueError(f"invalid ChEMBL molecule id: {value!r}")
        if value not in unique_ids:
            unique_ids.append(value)
    max_pages = _validate_budget("max_pages", max_pages)
    max_records = _validate_budget("max_records", max_records)
    max_requests = _validate_budget("max_requests", max_requests)
    max_response_bytes = _validate_budget("max_response_bytes", max_response_bytes)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")
    deadline = time.monotonic() + float(timeout)
    request_count = 0
    page_count = 0
    activity_count = 0
    activity_pages: list[dict[str, Any]] = []
    page_receipts: list[dict[str, Any]] = []
    molecule_records: dict[str, dict[str, Any]] = {}
    target_records: dict[str, dict[str, Any]] = {}
    page_statuses: dict[str, str] = {value: "complete" for value in unique_ids}
    incomplete_reasons: list[str] = []

    def request(url: str) -> tuple[dict[str, Any], bytes, str]:
        nonlocal request_count
        if interruption is not None and interruption():
            raise BoundedFetchError("interrupted")
        if time.monotonic() >= deadline:
            raise BoundedFetchError("timeout budget exhausted")
        if request_count >= max_requests:
            raise BoundedFetchError("max_requests reached")
        request_count += 1
        remaining = deadline - time.monotonic()
        return _bounded_get_json(
            url,
            timeout=max(0.001, min(float(timeout), remaining)),
            max_response_bytes=max_response_bytes,
            http_get=http_get,
        )

    for molecule_id in unique_ids:
        try:
            molecule_url = f"{CHEMBL_API_ROOT}/molecule/{molecule_id}.json"
            payload, body, body_hash = request(molecule_url)
            returned_id = payload.get("molecule_chembl_id")
            if returned_id != molecule_id:
                raise BoundedFetchError("molecule response identity mismatch")
            molecule_records[molecule_id] = payload
            # The full molecule response hash is saved with its request receipt by the caller.
            page_receipts.append({
                "source_id": f"chembl_molecule:{molecule_id}",
                "source_kind": "online_snapshot",
                "source_sha256": body_hash,
                "response_sha256": body_hash,
                "payload": payload,
                "raw_response_text": body.decode("utf-8"),
            })
        except Exception as exc:
            incomplete_reasons.append(f"molecule:{molecule_id}:{exc}")
            page_statuses[molecule_id] = "incomplete"
            continue

        next_url: Optional[str] = f"{CHEMBL_API_ROOT}/activity.json?{urlencode({'molecule_chembl_id': molecule_id, 'limit': 100, 'offset': 0})}"
        expected_offset = 0
        local_pages: list[dict[str, Any]] = []
        local_row_count = 0
        while next_url is not None:
            if page_count >= max_pages:
                incomplete_reasons.append(f"max_pages reached for {molecule_id}")
                page_statuses[molecule_id] = "incomplete"
                break
            try:
                payload, body, response_hash = request(next_url)
                normalized_page, rows, advertised_next = _validate_activity_page(payload, expected_offset)
                request_counted = len(rows)
                remaining_records = max_records - activity_count
                if remaining_records <= 0:
                    incomplete_reasons.append("max_records reached")
                    page_statuses[molecule_id] = "incomplete"
                    break
                page_file_hash = response_hash
                page_index = page_count + 1
                receipt = {
                    "source_id": f"chembl_activity:{molecule_id}:page:{page_index}",
                    "source_kind": "online_snapshot",
                    "source_sha256": page_file_hash,
                    "response_sha256": response_hash,
                }
                page = {
                    "molecule_chembl_id": molecule_id,
                    "source_kind": "online_snapshot",
                    "response_sha256": response_hash,
                    "page_meta": deepcopy(normalized_page["page_meta"]),
                    "activities": deepcopy(rows[:remaining_records]),
                    "raw_response_text": body.decode("utf-8"),
                }
                # Preserve the exact response object in the snapshot, even if the record cap
                # makes only a prefix eligible for canonical evidence rows.
                page["raw_response"] = normalized_page
                local_pages.append(page)
                activity_pages.append(page)
                page_receipts.append(receipt)
                page_count += 1
                page_rows = len(rows[:remaining_records])
                activity_count += page_rows
                local_row_count += page_rows
                expected_offset += len(rows)
                if len(rows) > remaining_records:
                    incomplete_reasons.append("max_records split a page; raw page retained")
                    page_statuses[molecule_id] = "incomplete"
                    break
                if advertised_next is None:
                    next_url = None
                else:
                    try:
                        next_url = _validate_chembl_next(advertised_next, next_url)
                    except EvidenceInputError as exc:
                        incomplete_reasons.append(f"invalid next URL for {molecule_id}: {exc}")
                        page_statuses[molecule_id] = "incomplete"
                        next_url = None
                        break
                    if page_count >= max_pages:
                        incomplete_reasons.append(f"max_pages reached for {molecule_id}")
                        page_statuses[molecule_id] = "incomplete"
                        next_url = None
                        break
                if request_counted == 0 and advertised_next is not None:
                    incomplete_reasons.append(f"empty non-terminal page for {molecule_id}")
                    page_statuses[molecule_id] = "incomplete"
                    next_url = None
                    break
            except Exception as exc:
                incomplete_reasons.append(f"activity_page:{molecule_id}:{exc}")
                page_statuses[molecule_id] = "incomplete"
                break
        if page_statuses[molecule_id] == "complete" and local_pages:
            if _page_chain_status(local_pages, local_row_count) != "complete":
                page_statuses[molecule_id] = "incomplete"
                incomplete_reasons.append(f"page metadata/count mismatch for {molecule_id}")
        elif not local_pages:
            page_statuses[molecule_id] = "incomplete"

    # Fetch only target records needed to resolve component identity. Missing target
    # metadata remains explicit and does not turn a successfully captured page into a
    # protein identity.
    target_ids = sorted({
        str(row.get("target_chembl_id"))
        for page in activity_pages
        for row in page["activities"]
        if isinstance(row.get("target_chembl_id"), str)
    })
    for target_id in target_ids:
        try:
            if not CHEMBL_ID.fullmatch(target_id):
                raise EvidenceInputError("target id is not a ChEMBL identifier")
            payload, body, response_hash = request(f"{CHEMBL_API_ROOT}/target/{target_id}.json")
            if payload.get("target_chembl_id") != target_id:
                raise BoundedFetchError("target response identity mismatch")
            target_records[target_id] = payload
            page_receipts.append({
                "source_id": f"chembl_target:{target_id}",
                "source_kind": "online_snapshot",
                "source_sha256": response_hash,
                "response_sha256": response_hash,
                "payload": payload,
                "raw_response_text": body.decode("utf-8"),
            })
        except Exception as exc:
            incomplete_reasons.append(f"target:{target_id}:{exc}")

    pagination_status = "complete" if all(value == "complete" for value in page_statuses.values()) else "incomplete"
    return {
        "source_kind": "online_snapshot",
        "source_version": None,
        "molecule_ids": unique_ids,
        "molecule_records": molecule_records,
        "target_records": target_records,
        "activity_pages": activity_pages,
        "page_receipts": page_receipts,
        "page_statuses": page_statuses,
        "pagination_status": pagination_status,
        "incomplete_reasons": incomplete_reasons,
        "request_count": request_count,
        "page_count": page_count,
        "activity_count": activity_count,
        "budgets": {
            "max_pages": max_pages,
            "max_records": max_records,
            "max_requests": max_requests,
            "max_response_bytes": max_response_bytes,
            "timeout_seconds": float(timeout),
        },
    }


def _persist_fetch_result(result: Mapping[str, Any], root: Path) -> tuple[list[ActivityEvidence], list[dict[str, Any]]]:
    provenance: list[dict[str, Any]] = []
    target_records = result.get("target_records", {})
    evidence_receipts: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    for index, page in enumerate(result.get("activity_pages", [])):
        raw = page.get("raw_response", page)
        molecule_id = next((
            str(row.get("molecule_chembl_id"))
            for row in page.get("activities", [])
            if isinstance(row.get("molecule_chembl_id"), str)
        ), "CHEMBL_UNKNOWN")
        page_index = len([item for item in raw_pages if item.get("_molecule_id") == molecule_id]) + 1
        raw_text = page.get("raw_response_text")
        content = raw_text.encode("utf-8") if isinstance(raw_text, str) else _json_bytes(raw)
        relative = f"raw/chembl/activities/{molecule_id}/page-{page_index:04d}.json"
        saved_path, saved_hash = _write_raw_bytes(root, relative, content)
        source = {
            "source_id": f"chembl_activity:{molecule_id}:page:{page_index}",
            "source_kind": "online_snapshot",
            "source_path": saved_path,
            "raw_sha256": saved_hash,
        }
        provenance.append(source)
        evidence_receipts.append({
            "source_id": source["source_id"],
            "source_path": saved_path,
            "source_sha256": saved_hash,
            "response_sha256": page.get("response_sha256") or _sha256_bytes(content),
        })
        raw_pages.append({**deepcopy(dict(page)), "_molecule_id": molecule_id})
    for receipt in result.get("page_receipts", []):
        if not isinstance(receipt, dict) or not str(receipt.get("source_id", "")).startswith(("chembl_molecule:", "chembl_target:")):
            continue
        payload = receipt.get("payload")
        if not isinstance(payload, dict):
            continue
        category, identifier = receipt["source_id"].split(":", 1)
        relative = f"raw/chembl/{category.removeprefix('chembl_')}/{identifier}.json"
        raw_text = receipt.get("raw_response_text")
        content = raw_text.encode("utf-8") if isinstance(raw_text, str) else _json_bytes(payload)
        if _is_sha256(receipt.get("response_sha256")) and _sha256_bytes(content) != receipt["response_sha256"]:
            raise EvidenceInputError(f"raw ChEMBL response hash mismatch: {receipt['source_id']}")
        saved_path, saved_hash = _write_raw_bytes(root, relative, content)
        provenance.append({
            "source_id": receipt["source_id"],
            "source_kind": "online_snapshot",
            "source_path": saved_path,
            "raw_sha256": saved_hash,
            "response_sha256": receipt.get("response_sha256"),
        })
    pages_by_molecule: dict[str, list[dict[str, Any]]] = {}
    receipts_by_molecule: dict[str, list[dict[str, Any]]] = {}
    for index, page in enumerate(raw_pages):
        molecule_id = page.pop("_molecule_id")
        source_page = page.get("raw_response")
        if not isinstance(source_page, dict):
            source_page = {"page_meta": page.get("page_meta"), "activities": page.get("activities", [])}
        page_obj = {
            "page_meta": deepcopy(page.get("page_meta", source_page.get("page_meta"))),
            "activities": deepcopy(page.get("activities", [])),
            "raw_response": deepcopy(source_page),
        }
        pages_by_molecule.setdefault(molecule_id, []).append({
            **page_obj,
        })
        receipts_by_molecule.setdefault(molecule_id, []).append(evidence_receipts[index])
    activities: list[ActivityEvidence] = []
    for molecule_id, pages in sorted(pages_by_molecule.items()):
        activities.extend(activity_evidence_from_pages(
            molecule_id, pages, source_kind="online_snapshot",
            page_receipts=receipts_by_molecule[molecule_id],
            targets=target_records,
        ))
    return activities, provenance


def fetch_chembl_bounded(
    molecule_ids: Sequence[str],
    output_dir: os.PathLike[str] | str,
    *,
    max_pages: int = 10,
    max_records: int = 1000,
    timeout: float = 30,
    max_requests: int = 128,
    max_response_bytes: int = 8_000_000,
    http_get: Optional[Callable[..., Any]] = None,
    interruption: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """Fetch a bounded, multi-molecule ChEMBL evidence set into a new root."""
    root = _create_output_root(output_dir)
    result = _collect_chembl(
        molecule_ids,
        max_pages=max_pages,
        max_records=max_records,
        timeout=timeout,
        max_requests=max_requests,
        max_response_bytes=max_response_bytes,
        http_get=http_get,
        interruption=interruption,
    )
    activities, provenance = _persist_fetch_result(result, root)
    if result["pagination_status"] == "complete":
        bundle = write_entity_bundle(
            root, activities=activities, source_provenance=provenance,
            source_kind="online_snapshot",
        )
        result["content_sha256"] = bundle["content_sha256"]
        result["entity_version"] = bundle["entity_version"]
        result["bundle_path"] = str(root / "entities.json")
    else:
        partial = {
            "status": "incomplete",
            "source_kind": "online_snapshot",
            "pagination_status": "incomplete",
            "activity_count": len(activities),
            "request_count": result["request_count"],
            "page_count": result["page_count"],
            "budgets": result["budgets"],
            "incomplete_reasons": result["incomplete_reasons"],
            "source_provenance": provenance,
        }
        (root / "partial_ingest.json").write_bytes(_json_bytes(partial))
        result.update(partial)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build traceable molecule-target activity evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("build-local", help="build from a local JSON manifest")
    local.add_argument("--manifest", required=True, type=Path)
    local.add_argument("--out", required=True, type=Path)
    fetch = subparsers.add_parser("fetch-chembl", help="fetch bounded activity pages by ChEMBL molecule ID")
    fetch.add_argument("--molecule-id", action="append", required=True)
    fetch.add_argument("--out", required=True, type=Path)
    fetch.add_argument("--max-pages", type=int, default=10)
    fetch.add_argument("--max-records", type=int, default=1000)
    fetch.add_argument("--timeout", type=float, default=30)
    fetch.add_argument("--max-requests", type=int, default=128)
    fetch.add_argument("--max-response-bytes", type=int, default=8_000_000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.command == "build-local":
            result = ingest_local_manifest(args.manifest, args.out)
        elif args.command == "fetch-chembl":
            result = fetch_chembl_bounded(
                args.molecule_id,
                args.out,
                max_pages=args.max_pages,
                max_records=args.max_records,
                timeout=args.timeout,
                max_requests=args.max_requests,
                max_response_bytes=args.max_response_bytes,
            )
        else:
            raise EvidenceInputError(f"unsupported command: {args.command}")
    except Exception as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status", result.get("pagination_status")) == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

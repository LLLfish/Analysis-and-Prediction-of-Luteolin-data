"""Build versioned, provenance-preserving activity evidence bundles.

The module deliberately separates measured evidence from the legacy display
projection. Local manifests are the default input; network retrieval is an
explicit, bounded CLI operation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass, replace
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

from data_contracts import ProteinRecord, canonical_json_sha256, validate_protein_record


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
    target_components: list[Any]
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
    source_url: Optional[str] = None
    query: Optional[dict[str, Any]] = None
    page_offset: Optional[int] = None
    pagination_next: Optional[str] = None
    target_record: Optional[dict[str, Any]] = None
    target_record_source_path: Optional[str] = None
    target_record_source_sha256: Optional[str] = None
    assay_record: Optional[dict[str, Any]] = None
    assay_record_source_path: Optional[str] = None
    assay_record_source_sha256: Optional[str] = None
    document_record: Optional[dict[str, Any]] = None
    document_record_source_path: Optional[str] = None
    document_record_source_sha256: Optional[str] = None
    confidence_source: Optional[str] = None
    assay_confidence_score: Any = None
    metadata_status: str = "unknown"


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


@dataclass(frozen=True)
class MoleculeRecord:
    """Molecule identity plus raw and RDKit-canonical structure evidence."""

    molecule_chembl_id: Optional[str]
    pref_name: Optional[str]
    molecule_type: Optional[str]
    raw_structure: Optional[dict[str, Any]]
    canonical_smiles: Optional[str]
    structure_sha256: Optional[str]
    structure_status: str
    structure_status_reason: Optional[str]
    source_kind: str
    source_provenance: list[dict[str, Any]] = field(default_factory=list)


def _source_provenance_entry(
    *, source_id: str, source_kind: str, source_path: Optional[str],
    source_sha256: Optional[str], source_url: Optional[str] = None,
) -> dict[str, Any]:
    if source_kind not in ALLOWED_SOURCE_KINDS:
        source_kind = "legacy_unverified"
    raw_hash = source_sha256 if _is_sha256(source_sha256) else None
    entry: dict[str, Any] = {
        "source_id": source_id,
        "source_kind": source_kind,
        "raw_sha256": raw_hash,
    }
    if isinstance(source_path, str) and source_path:
        entry["source_path"] = source_path.replace("\\", "/")
    if isinstance(source_url, str) and source_url:
        entry["source_url"] = source_url
    return entry


def molecule_record_from_source(
    raw_record: Mapping[str, Any],
    *,
    source_kind: str,
    source_path: Optional[str] = None,
    source_sha256: Optional[str] = None,
    source_url: Optional[str] = None,
) -> MoleculeRecord:
    """Parse and canonicalize one ChEMBL molecule without inventing its identity."""
    raw = _copy_json_object(dict(raw_record), "ChEMBL molecule record")
    source_kind = _source_kind(source_kind)
    molecule_id = raw.get("molecule_chembl_id")
    if not isinstance(molecule_id, str) or not CHEMBL_ID.fullmatch(molecule_id):
        molecule_id = None
    name = raw.get("pref_name")
    molecule_type = raw.get("molecule_type")
    structures = raw.get("molecule_structures")
    raw_structure = deepcopy(structures) if isinstance(structures, dict) else None
    raw_hash = source_sha256 if _is_sha256(source_sha256) else canonical_json_sha256(raw)
    provenance = [_source_provenance_entry(
        source_id=f"chembl_molecule:{molecule_id or 'unresolved'}",
        source_kind=source_kind,
        source_path=source_path,
        source_sha256=raw_hash,
        source_url=source_url,
    )]
    canonical_smiles = None
    structure_hash = None
    if molecule_id is None:
        return MoleculeRecord(
            molecule_chembl_id=None,
            pref_name=name if isinstance(name, str) else None,
            molecule_type=molecule_type if isinstance(molecule_type, str) else None,
            raw_structure=raw_structure,
            canonical_smiles=None,
            structure_sha256=None,
            structure_status="unverified",
            structure_status_reason="missing_molecule_chembl_id",
            source_kind=source_kind,
            source_provenance=provenance,
        )
    if not raw_structure:
        return MoleculeRecord(
            molecule_chembl_id=molecule_id,
            pref_name=name if isinstance(name, str) else None,
            molecule_type=molecule_type if isinstance(molecule_type, str) else None,
            raw_structure=raw_structure,
            canonical_smiles=None,
            structure_sha256=None,
            structure_status="missing",
            structure_status_reason="molecule_structure_missing",
            source_kind=source_kind,
            source_provenance=provenance,
        )
    smiles = raw_structure.get("canonical_smiles")
    inchi = raw_structure.get("standard_inchi") or raw_structure.get("standard_inchi_key")
    try:
        from rdkit import Chem
    except Exception:
        return MoleculeRecord(
            molecule_chembl_id=molecule_id,
            pref_name=name if isinstance(name, str) else None,
            molecule_type=molecule_type if isinstance(molecule_type, str) else None,
            raw_structure=raw_structure,
            canonical_smiles=None,
            structure_sha256=None,
            structure_status="error",
            structure_status_reason="rdkit_unavailable",
            source_kind=source_kind,
            source_provenance=provenance,
        )
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None and isinstance(inchi, str) and inchi.startswith("InChI="):
        try:
            mol = Chem.MolFromInchi(inchi)
        except Exception:
            mol = None
    if mol is None:
        return MoleculeRecord(
            molecule_chembl_id=molecule_id,
            pref_name=name if isinstance(name, str) else None,
            molecule_type=molecule_type if isinstance(molecule_type, str) else None,
            raw_structure=raw_structure,
            canonical_smiles=None,
            structure_sha256=None,
            structure_status="invalid",
            structure_status_reason="structure_parse_failed",
            source_kind=source_kind,
            source_provenance=provenance,
        )
    try:
        fragment_count = len(Chem.GetMolFrags(mol))
        if fragment_count != 1:
            return MoleculeRecord(
                molecule_chembl_id=molecule_id,
                pref_name=name if isinstance(name, str) else None,
                molecule_type=molecule_type if isinstance(molecule_type, str) else None,
                raw_structure=raw_structure,
                canonical_smiles=None,
                structure_sha256=None,
                structure_status="excluded",
                structure_status_reason="multiple_fragments",
                source_kind=source_kind,
                source_provenance=provenance,
            )
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        structure_hash = _sha256_bytes(canonical_smiles.encode("utf-8"))
    except Exception as exc:
        return MoleculeRecord(
            molecule_chembl_id=molecule_id,
            pref_name=name if isinstance(name, str) else None,
            molecule_type=molecule_type if isinstance(molecule_type, str) else None,
            raw_structure=raw_structure,
            canonical_smiles=None,
            structure_sha256=None,
            structure_status="invalid",
            structure_status_reason=f"structure_normalization_failed:{type(exc).__name__}",
            source_kind=source_kind,
            source_provenance=provenance,
        )
    return MoleculeRecord(
        molecule_chembl_id=molecule_id,
        pref_name=name if isinstance(name, str) else None,
        molecule_type=molecule_type if isinstance(molecule_type, str) else None,
        raw_structure=raw_structure,
        canonical_smiles=canonical_smiles,
        structure_sha256=structure_hash,
        structure_status="verified",
        structure_status_reason=None,
        source_kind=source_kind,
        source_provenance=provenance,
    )


def resolve_target_components(
    target_record: Mapping[str, Any],
    *,
    source_kind: str,
    source_path: Optional[str] = None,
    source_sha256: Optional[str] = None,
    source_url: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Resolve every component; only a confirmed human SINGLE PROTEIN is eligible."""
    target = _copy_json_object(dict(target_record), "ChEMBL target record")
    source_kind = _source_kind(source_kind)
    target_id = target.get("target_chembl_id")
    target_type = target.get("target_type")
    organism = target.get("organism")
    target_taxonomy = _taxonomy_adapter(target.get("tax_id", target.get("target_tax_id")))
    components = target.get("target_components")
    target_raw_hash = source_sha256 if _is_sha256(source_sha256) else canonical_json_sha256(target)
    identity_source = _source_provenance_entry(
        source_id=f"chembl_target:{target_id if isinstance(target_id, str) else 'unresolved'}",
        source_kind=source_kind,
        source_path=source_path,
        source_sha256=target_raw_hash,
        source_url=source_url,
    )
    base = {
        "target_chembl_id": target_id if isinstance(target_id, str) else None,
        "target_type": target_type if isinstance(target_type, str) else None,
        "organism": organism if isinstance(organism, str) else None,
        "target_taxonomy_id": target_taxonomy,
        "target_record": target,
        "target_record_sha256": target_raw_hash,
        "target_record_source": identity_source,
        "source_kind": source_kind,
    }
    if not isinstance(components, list) or not components:
        return [{
            **base,
            "component_index": None,
            "component": None,
            "accession": None,
            "taxonomy_id": None,
            "taxonomy_evidence_path": None,
            "identity_key": None,
            "resolution_status": "unverified",
            "reason_code": "missing_target_components",
        }]
    target_is_single = target_type == "SINGLE PROTEIN"
    single_component = len(components) == 1
    results: list[dict[str, Any]] = []
    for index, value in enumerate(components):
        component = deepcopy(value)
        if not isinstance(component, dict):
            results.append({
                **base,
                "component_index": index,
                "component": component,
                "accession": None,
                "taxonomy_id": None,
                "taxonomy_evidence_path": None,
                "identity_key": None,
                "resolution_status": "excluded",
                "reason_code": "malformed_target_component",
            })
            continue
        accession = component.get("accession")
        if not isinstance(accession, str) or not accession.strip():
            status, reason, accession_value = "unverified", "missing_component_accession", None
        else:
            accession_candidate = accession.strip()
            if _UNIPROT_ACCESSION.fullmatch(accession_candidate):
                accession_value = accession_candidate
                status, reason = "verified", None
            else:
                accession_value = None
                status, reason = "unverified", "invalid_component_accession"
        if not isinstance(accession, str) or not accession.strip():
            accession_candidate = None
        component_type = component.get("component_type")
        if not target_is_single:
            status, reason = "excluded", "target_type_not_single_protein"
        elif not single_component:
            status, reason = "excluded", "single_protein_component_count_not_one"
        elif organism != "Homo sapiens" or target_taxonomy != 9606:
            status, reason = "excluded", "non_human_taxonomy"
        elif component_type != "PROTEIN":
            status, reason = "excluded", "component_type_not_protein"
        elif accession_value is None and accession_candidate is None:
            status, reason = "unverified", "missing_component_accession"
        component_tax = _taxonomy_adapter(
            component.get("component_tax_id", component.get("tax_id", component.get("taxonomy_id")))
        )
        if component_tax is not None and target_taxonomy is not None and component_tax != target_taxonomy:
            status, reason = "excluded", "component_taxonomy_mismatch"
        taxonomy = component_tax if component_tax is not None else target_taxonomy
        taxonomy_path = (
            f"target_components[{index}].component_tax_id"
            if component_tax is not None and "component_tax_id" in component
            else f"target_components[{index}].tax_id"
            if component_tax is not None and "tax_id" in component
            else "target.tax_id"
            if taxonomy is not None
            else None
        )
        if status == "verified" and taxonomy is None:
            status, reason = "unverified", "missing_component_taxonomy"
        identity_key = [taxonomy, accession_value] if taxonomy is not None and accession_value is not None else None
        results.append({
            **base,
            "component_index": index,
            "component": component,
            "accession": accession_value,
            "accession_candidate": accession_candidate,
            "taxonomy_id": taxonomy,
            "taxonomy_evidence_path": taxonomy_path,
            "identity_key": identity_key,
            "resolution_status": status,
            "reason_code": reason,
        })
    return results


_UNIPROT_ORIGIN = "https://rest.uniprot.org"
_UNIPROT_ACCESSION = re.compile(r"^[A-Z0-9]{6}(?:[A-Z0-9]{4})?$")
_STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _uniprot_source_error_record(
    component: Mapping[str, Any], *, source_kind: str = "legacy_unverified",
    reason: str, sequence_status: str = "error",
    raw_sequence: Optional[str] = None,
    full_sequence: Optional[str] = None,
    sequence_length: Optional[int] = None,
    sequence_sha256: Optional[str] = None,
    source_provenance: Optional[list[dict[str, Any]]] = None,
    returned_accession: Optional[str] = None,
    returned_taxonomy_id: Optional[int] = None,
    identity_status: Optional[str] = None,
) -> ProteinRecord:
    raw_accession = component.get("accession")
    accession = raw_accession if (
        isinstance(raw_accession, str)
        and _UNIPROT_ACCESSION.fullmatch(raw_accession) is not None
    ) else None
    taxonomy = component.get("taxonomy_id")
    taxonomy_id = taxonomy if type(taxonomy) is int and taxonomy > 0 else None
    status = component.get("resolution_status")
    resolved_identity = identity_status or ("verified" if status == "verified" else "unverified")
    record = ProteinRecord(
        accession=accession,
        taxonomy_id=taxonomy_id,
        raw_sequence=raw_sequence,
        full_sequence=full_sequence,
        sequence_length=sequence_length,
        sequence_sha256=sequence_sha256,
        sequence_status=sequence_status,
        sequence_status_reason=reason,
        identity_status=resolved_identity,
        source_kind=source_kind if source_kind in {"online_snapshot", "legacy_unverified", "synthetic_test"} else "legacy_unverified",
        source_provenance=source_provenance or [],
        component_evidence=deepcopy(dict(component)),
        returned_accession=returned_accession,
        returned_taxonomy_id=returned_taxonomy_id,
    )
    validate_protein_record(record)
    return record


def _normalise_sequence(raw_sequence: Any) -> tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    if raw_sequence is None:
        return None, None, None, "sequence_value_missing"
    if not isinstance(raw_sequence, str):
        return None, None, None, "sequence_value_not_text"
    normalized = "".join(char for char in raw_sequence if not char.isspace()).upper()
    if not normalized:
        return "", 0, _sha256_bytes(b""), "sequence_value_empty"
    return normalized, len(normalized), _sha256_bytes(normalized.encode("utf-8")), None


def _hydrate_from_uniprot_payload(
    component: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    raw_bytes: bytes,
    source_kind: str,
    source_path: Optional[str],
    source_url: Optional[str] = None,
    source_id: Optional[str] = None,
) -> ProteinRecord:
    if not isinstance(payload, Mapping):
        return _uniprot_source_error_record(component, source_kind=source_kind, reason="uniprot_json_root_not_object")
    requested_accession = component.get("accession")
    requested_taxonomy = component.get("taxonomy_id")
    source_hash = _sha256_bytes(raw_bytes)
    provenance = [_source_provenance_entry(
        source_id=source_id or f"uniprot:{requested_accession}",
        source_kind=source_kind,
        source_path=source_path,
        source_sha256=source_hash,
        source_url=source_url,
    )]
    returned_accession = payload.get("primaryAccession")
    organism = payload.get("organism")
    organism = organism if isinstance(organism, Mapping) else {}
    returned_taxonomy = _taxonomy_adapter(organism.get("taxonId"))
    identity_status = "verified"
    mismatch_reason = None
    if not isinstance(requested_accession, str) or _UNIPROT_ACCESSION.fullmatch(requested_accession) is None:
        identity_status, mismatch_reason = "unverified", "component_accession_missing_or_invalid"
    elif returned_accession != requested_accession:
        identity_status, mismatch_reason = "mismatch", "accession_mismatch"
    elif type(requested_taxonomy) is not int or requested_taxonomy <= 0:
        identity_status, mismatch_reason = "unverified", "component_taxonomy_missing_or_invalid"
    elif returned_taxonomy != requested_taxonomy:
        identity_status, mismatch_reason = "mismatch", "taxonomy_mismatch"

    sequence = payload.get("sequence")
    sequence = sequence if isinstance(sequence, Mapping) else {}
    raw_sequence = sequence.get("value")
    full_sequence, sequence_length, sequence_hash, sequence_error = _normalise_sequence(raw_sequence)
    declared_length = sequence.get("length")
    if sequence_error == "sequence_value_missing":
        sequence_status, reason = "missing", sequence_error
    elif sequence_error:
        sequence_status, reason = "invalid", sequence_error
    elif type(declared_length) is not int:
        sequence_status, reason = "invalid", "declared_length_missing_or_invalid"
    elif declared_length != sequence_length:
        sequence_status, reason = "invalid", "declared_length_mismatch"
    elif not full_sequence or any(residue not in _STANDARD_RESIDUES for residue in full_sequence):
        sequence_status, reason = "invalid", "nonstandard_residue"
    else:
        sequence_status, reason = "verified", None
    if sequence_status == "verified" and source_kind == "legacy_unverified":
        sequence_status, reason = "unverified", "source_kind_legacy_unverified"
    if mismatch_reason is not None:
        sequence_status, reason = "invalid", mismatch_reason

    record = ProteinRecord(
        accession=requested_accession if isinstance(requested_accession, str) else None,
        taxonomy_id=requested_taxonomy if type(requested_taxonomy) is int else None,
        raw_sequence=raw_sequence if isinstance(raw_sequence, str) else None,
        full_sequence=full_sequence,
        sequence_length=sequence_length,
        sequence_sha256=sequence_hash,
        sequence_status=sequence_status,
        sequence_status_reason=reason,
        identity_status=identity_status,
        source_kind=source_kind,
        source_provenance=provenance,
        component_evidence=deepcopy(dict(component)),
        returned_accession=returned_accession if isinstance(returned_accession, str) else None,
        returned_taxonomy_id=returned_taxonomy,
    )
    validate_protein_record(record)
    return record


def _bounded_get_uniprot_json(
    accession: str,
    *,
    timeout: float,
    max_response_bytes: int,
    http_get: Optional[Callable[..., Any]] = None,
) -> tuple[dict[str, Any], bytes, str, str]:
    if _UNIPROT_ACCESSION.fullmatch(accession) is None:
        raise EvidenceInputError("UniProt accession must be an exact 6- or 10-character identifier")
    deadline = time.monotonic() + timeout
    if time.monotonic() >= deadline:
        raise BoundedFetchError("UniProt timeout budget exhausted")
    url = f"{_UNIPROT_ORIGIN}/uniprotkb/{accession}.json"
    getter = http_get or requests.get
    response = getter(
        url,
        headers={"Accept": "application/json"},
        timeout=timeout,
        stream=True,
        allow_redirects=False,
    )
    response_url = getattr(response, "url", url)
    parsed = urlparse(response_url)
    if parsed.scheme != "https" or parsed.hostname != "rest.uniprot.org" or parsed.path != f"/uniprotkb/{accession}.json":
        raise BoundedFetchError("UniProt response escaped the official accession endpoint")
    status = getattr(response, "status_code", 200)
    if type(status) is not int or 300 <= status < 400:
        raise BoundedFetchError(f"UniProt redirect or invalid HTTP status: {status!r}")
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    headers = getattr(response, "headers", {})
    length = headers.get("Content-Length") if isinstance(headers, Mapping) else None
    if length is not None:
        try:
            if int(length) < 0:
                raise BoundedFetchError("negative UniProt Content-Length")
            if int(length) > max_response_bytes:
                raise BoundedFetchError("UniProt response exceeds max_response_bytes")
        except ValueError as exc:
            raise BoundedFetchError("invalid UniProt Content-Length") from exc
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=min(max_response_bytes + 1, 65536)):
        if time.monotonic() >= deadline:
            raise BoundedFetchError("UniProt timeout budget exhausted while streaming")
        if not chunk:
            continue
        if not isinstance(chunk, bytes):
            raise BoundedFetchError("UniProt HTTP body stream returned non-byte content")
        total += len(chunk)
        if total > max_response_bytes:
            raise BoundedFetchError("UniProt response exceeds max_response_bytes")
        chunks.append(chunk)
    body = b"".join(chunks)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundedFetchError(f"UniProt response is not UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BoundedFetchError("UniProt response JSON root must be an object")
    return payload, body, _sha256_bytes(body), url


def _fasta_sequence(path: Path, accession: str) -> tuple[Optional[str], Optional[str]]:
    text = path.read_text(encoding="utf-8-sig")
    records: list[tuple[str, str]] = []
    header: Optional[str] = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "\n".join(chunks)))
            header = line[1:].strip()
            chunks = []
        elif line.strip():
            if header is None:
                return None, "fasta_header_missing"
            chunks.append(line)
    if header is not None:
        records.append((header, "\n".join(chunks)))
    if len(records) != 1:
        return None, "fasta_record_count_not_one"
    header, sequence = records[0]
    tokens = header.split()
    token = tokens[0] if tokens else ""
    if "|" in token:
        parts = token.split("|")
        header_accession = parts[1] if len(parts) >= 2 else ""
    else:
        header_accession = token
    if header_accession != accession:
        return sequence, "fasta_accession_mismatch"
    return sequence, None


def hydrate_protein_record(
    component: Mapping[str, Any],
    *,
    source_kind: Optional[str] = None,
    source_path: Optional[str] = None,
    uniprot_json: Optional[Mapping[str, Any]] = None,
    raw_response_bytes: Optional[bytes] = None,
    fasta_path: Optional[os.PathLike[str] | str] = None,
    taxonomy_metadata_path: Optional[os.PathLike[str] | str] = None,
    fasta_source_path: Optional[str] = None,
    taxonomy_metadata_source_path: Optional[str] = None,
    online: bool = False,
    output_root: Optional[os.PathLike[str] | str] = None,
    timeout: float = 30,
    max_requests: int = 1,
    max_response_bytes: int = 8_000_000,
    http_get: Optional[Callable[..., Any]] = None,
) -> ProteinRecord:
    """Hydrate one proven accession from local UniProt JSON/FASTA or explicit bounded GET."""
    if not isinstance(component, Mapping):
        raise EvidenceInputError("protein component identity must be an object")
    kind = source_kind or "legacy_unverified"
    if kind not in {"online_snapshot", "legacy_unverified", "synthetic_test"}:
        kind = "legacy_unverified"
    if component.get("resolution_status") != "verified":
        return _uniprot_source_error_record(
            component, source_kind=kind, sequence_status="unverified",
            reason=str(component.get("reason_code") or "component_identity_not_verified"),
        )
    accession = component.get("accession")
    taxonomy = component.get("taxonomy_id")
    if not isinstance(accession, str) or _UNIPROT_ACCESSION.fullmatch(accession) is None:
        return _uniprot_source_error_record(
            component, source_kind=kind, sequence_status="unverified",
            reason="component_accession_missing_or_invalid", identity_status="unverified",
        )
    if type(taxonomy) is not int or taxonomy <= 0:
        return _uniprot_source_error_record(
            component, source_kind=kind, sequence_status="unverified",
            reason="component_taxonomy_missing_or_invalid", identity_status="unverified",
        )

    if online:
        if type(max_requests) is not int or max_requests < 1:
            return _uniprot_source_error_record(component, source_kind="online_snapshot", reason="max_requests_exhausted")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            return _uniprot_source_error_record(component, source_kind="online_snapshot", reason="invalid_timeout")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            return _uniprot_source_error_record(component, source_kind="online_snapshot", reason="invalid_max_response_bytes")
        if output_root is None:
            return _uniprot_source_error_record(component, source_kind="online_snapshot", reason="explicit_output_root_required")
        root = _create_output_root(output_root)
        try:
            payload, body, raw_hash, url = _bounded_get_uniprot_json(
                accession,
                timeout=float(timeout),
                max_response_bytes=max_response_bytes,
                http_get=http_get,
            )
            relative = f"raw/uniprot/{accession}.json"
            saved_path, saved_hash = _write_raw_bytes(root, relative, body)
            if saved_hash != raw_hash:
                raise EvidenceInputError("saved UniProt source hash differs from response hash")
            record = _hydrate_from_uniprot_payload(
                component, payload, raw_bytes=body, source_kind="online_snapshot",
                source_path=saved_path, source_url=url, source_id=f"uniprot:{accession}",
            )
            (root / "protein_record.json").write_bytes(_json_bytes(asdict(record)))
            return record
        except Exception as exc:
            partial = {"status": "incomplete", "accession": accession, "reason": str(exc)}
            (root / "partial_uniprot.json").write_bytes(_json_bytes(partial))
            return _uniprot_source_error_record(
                component, source_kind="online_snapshot", sequence_status="error",
                reason=f"online_retrieval_failed:{type(exc).__name__}",
            )

    if uniprot_json is not None:
        if not isinstance(uniprot_json, Mapping):
            return _uniprot_source_error_record(component, source_kind=kind, reason="uniprot_json_root_not_object")
        if raw_response_bytes is None:
            raw_response_bytes = _json_bytes(dict(uniprot_json))
        try:
            source_payload = json.loads(raw_response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _uniprot_source_error_record(component, source_kind=kind, reason="uniprot_raw_json_invalid")
        if source_payload != dict(uniprot_json):
            return _uniprot_source_error_record(component, source_kind=kind, reason="uniprot_raw_payload_mismatch")
        path_value = os.fspath(source_path) if source_path is not None else None
        record = _hydrate_from_uniprot_payload(
            component, uniprot_json, raw_bytes=raw_response_bytes,
            source_kind=kind, source_path=path_value,
            source_id=f"uniprot:{accession}",
        )
        if not path_value and record.sequence_status == "verified":
            record = _uniprot_source_error_record(
                component, source_kind=kind, sequence_status="unverified",
                reason="raw_source_path_missing",
                raw_sequence=record.raw_sequence,
                full_sequence=record.full_sequence,
                sequence_length=record.sequence_length,
                sequence_sha256=record.sequence_sha256,
                source_provenance=[],
            )
        return record

    if fasta_path is not None or taxonomy_metadata_path is not None:
        if fasta_path is None or taxonomy_metadata_path is None:
            return _uniprot_source_error_record(component, source_kind=kind, reason="fasta_taxonomy_metadata_pair_required")
        fasta_file = Path(fasta_path)
        metadata_file = Path(taxonomy_metadata_path)
        try:
            metadata_bytes = metadata_file.read_bytes()
            metadata = json.loads(metadata_bytes.decode("utf-8-sig"))
            fasta_bytes = fasta_file.read_bytes()
            fasta_sequence, fasta_error = _fasta_sequence(fasta_file, accession)
        except Exception as exc:
            return _uniprot_source_error_record(component, source_kind=kind, sequence_status="error", reason=f"local_sequence_source_error:{type(exc).__name__}")
        metadata = metadata if isinstance(metadata, dict) else {}
        returned_accession = metadata.get("primaryAccession", metadata.get("accession"))
        organism = metadata.get("organism")
        organism = organism if isinstance(organism, Mapping) else {}
        returned_tax = _taxonomy_adapter(organism.get("taxonId", metadata.get("taxonomy_id")))
        if returned_accession != accession:
            return _uniprot_source_error_record(
                component, source_kind=kind, reason="accession_mismatch",
                sequence_status="invalid", returned_accession=returned_accession,
                returned_taxonomy_id=returned_tax, identity_status="mismatch",
                raw_sequence=fasta_sequence,
            )
        if returned_tax != taxonomy:
            return _uniprot_source_error_record(
                component, source_kind=kind, reason="taxonomy_mismatch",
                sequence_status="invalid", returned_accession=returned_accession,
                returned_taxonomy_id=returned_tax, identity_status="mismatch",
                raw_sequence=fasta_sequence,
            )
        seq_obj = metadata.get("sequence")
        declared = seq_obj.get("length") if isinstance(seq_obj, Mapping) else metadata.get("sequence_length")
        source_path_fasta = fasta_source_path or f"raw/uniprot/{accession}.fasta"
        source_path_meta = taxonomy_metadata_source_path or f"raw/uniprot/{accession}-metadata.json"
        provenance = [
            _source_provenance_entry(
                source_id=f"uniprot_fasta:{accession}", source_kind=kind,
                source_path=source_path_fasta, source_sha256=_sha256_bytes(fasta_bytes),
            ),
            _source_provenance_entry(
                source_id=f"uniprot_taxonomy_metadata:{accession}", source_kind=kind,
                source_path=source_path_meta, source_sha256=_sha256_bytes(metadata_bytes),
            ),
        ]
        full, length, seq_hash, seq_error = _normalise_sequence(fasta_sequence)
        if fasta_error:
            status, reason = "invalid", fasta_error
        elif seq_error:
            status, reason = ("missing" if seq_error == "sequence_value_missing" else "invalid"), seq_error
        elif type(declared) is int and declared != length:
            status, reason = "invalid", "declared_length_mismatch"
        elif declared is not None and type(declared) is not int:
            status, reason = "invalid", "declared_length_missing_or_invalid"
        elif not full or any(residue not in _STANDARD_RESIDUES for residue in full):
            status, reason = "invalid", "nonstandard_residue"
        else:
            status, reason = "verified", None
        if status == "verified" and kind == "legacy_unverified":
            status, reason = "unverified", "source_kind_legacy_unverified"
        record = ProteinRecord(
            accession=accession, taxonomy_id=taxonomy,
            raw_sequence=fasta_sequence, full_sequence=full,
            sequence_length=length, sequence_sha256=seq_hash,
            sequence_status=status, sequence_status_reason=reason,
            identity_status="verified", source_kind=kind,
            source_provenance=provenance,
            component_evidence=deepcopy(dict(component)),
            returned_accession=returned_accession,
            returned_taxonomy_id=returned_tax,
        )
        validate_protein_record(record)
        return record

    return _uniprot_source_error_record(
        component, source_kind=kind, sequence_status="missing",
        reason="local_sequence_source_missing",
    )


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
            if next_url is not None:
                try:
                    _validate_chembl_next(next_url, f"{CHEMBL_API_ROOT}/activity.json")
                except EvidenceInputError:
                    return "incomplete"
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
    assays: Optional[Mapping[str, Mapping[str, Any]]] = None,
    documents: Optional[Mapping[str, Mapping[str, Any]]] = None,
    metadata_receipts: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[ActivityEvidence]:
    """Convert raw pages into evidence without dropping source fields."""
    if not isinstance(molecule_chembl_id, str) or not CHEMBL_ID.fullmatch(molecule_chembl_id):
        raise EvidenceInputError("molecule_chembl_id must be a ChEMBL identifier")
    source_kind = _source_kind(source_kind)
    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
        raise EvidenceInputError("pages must be a sequence of raw page objects")
    receipts = list(page_receipts or [])
    target_map = targets or {}
    assay_map = assays or {}
    document_map = documents or {}
    metadata_receipts = metadata_receipts or {}
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
            if isinstance(target, Mapping) and target.get("target_chembl_id") != target_id_key:
                target = None
            target_type, organism, taxonomy, components = _target_fields(raw, target)
            assay_id = raw.get("assay_chembl_id")
            assay_id_key = assay_id if isinstance(assay_id, str) else ""
            assay = assay_map.get(assay_id_key)
            if isinstance(assay, Mapping) and assay.get("assay_chembl_id") != assay_id_key:
                assay = None
            document_id = raw.get("document_chembl_id")
            document_id_key = document_id if isinstance(document_id, str) else ""
            document = document_map.get(document_id_key)
            if isinstance(document, Mapping) and document.get("document_chembl_id") != document_id_key:
                document = None
            raw_confidence = raw.get("confidence_score")
            if raw_confidence is not None:
                confidence_score = raw_confidence
                confidence_source = "activity"
            elif isinstance(assay, Mapping) and assay.get("confidence_score") is not None:
                confidence_score = assay.get("confidence_score")
                confidence_source = "assay"
            else:
                confidence_score = None
                confidence_source = None
            target_source = metadata_receipts.get(f"target:{target_id_key}", {})
            assay_source = metadata_receipts.get(f"assay:{assay_id_key}", {})
            document_source = metadata_receipts.get(f"document:{document_id_key}", {})
            missing = [
                name for name, value in (
                    ("activity_id", raw.get("activity_id")),
                    ("assay_chembl_id", raw.get("assay_chembl_id")),
                    ("document_chembl_id", raw.get("document_chembl_id")),
                    ("molecule_chembl_id", evidence_molecule_id),
                    ("target_chembl_id", target_id if isinstance(target_id, str) else None),
                    ("target_type", target_type),
                    ("target_components", components if components else None),
                    ("target_metadata", target if target_id_key else True),
                    ("assay_metadata", assay if assay_id_key else True),
                    ("document_metadata", document if document_id_key else True),
                    ("confidence_score", confidence_score),
                ) if value is None
            ]
            evidence_status = "observed" if not missing and source_kind in {
                "online_snapshot", "synthetic_test"
            } else "unverified"
            metadata_status = "complete" if (
                evidence_molecule_id == molecule_chembl_id
                and target is not None and assay is not None and document is not None
                and confidence_score is not None
            ) else "incomplete"
            source_url = page_obj.get("source_url")
            query = {
                "molecule_chembl_id": molecule_chembl_id,
                "limit": page_meta.get("limit"),
                "offset": page_meta.get("offset"),
            }
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
                confidence_score=confidence_score,
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
                source_url=source_url if isinstance(source_url, str) else None,
                query=query,
                page_offset=page_meta.get("offset") if type(page_meta.get("offset")) is int else None,
                pagination_next=page_meta.get("next") if isinstance(page_meta.get("next"), str) else None,
                target_record=deepcopy(dict(target)) if isinstance(target, Mapping) else None,
                target_record_source_path=target_source.get("source_path") if isinstance(target_source.get("source_path"), str) else None,
                target_record_source_sha256=target_source.get("source_sha256") if _is_sha256(target_source.get("source_sha256")) else None,
                assay_record=deepcopy(dict(assay)) if isinstance(assay, Mapping) else None,
                assay_record_source_path=assay_source.get("source_path") if isinstance(assay_source.get("source_path"), str) else None,
                assay_record_source_sha256=assay_source.get("source_sha256") if _is_sha256(assay_source.get("source_sha256")) else None,
                document_record=deepcopy(dict(document)) if isinstance(document, Mapping) else None,
                document_record_source_path=document_source.get("source_path") if isinstance(document_source.get("source_path"), str) else None,
                document_record_source_sha256=document_source.get("source_sha256") if _is_sha256(document_source.get("source_sha256")) else None,
                confidence_source=confidence_source,
                assay_confidence_score=assay.get("confidence_score") if isinstance(assay, Mapping) else None,
                metadata_status=metadata_status,
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
        grouped_pages: dict[str, list[Mapping[str, Any]]] = {}
        for page in payload["activity_pages"]:
            if not isinstance(page, Mapping):
                raise EvidenceInputError("collector activity page must be an object")
            queried_id = page.get("molecule_chembl_id")
            if not isinstance(queried_id, str) or not CHEMBL_ID.fullmatch(queried_id):
                queried_id = molecule_id
            if queried_id is None:
                row_ids = {
                    row.get("molecule_chembl_id")
                    for row in page.get("activities", [])
                    if isinstance(row, Mapping)
                    and isinstance(row.get("molecule_chembl_id"), str)
                }
                if len(row_ids) == 1:
                    queried_id = next(iter(row_ids))
            if not isinstance(queried_id, str) or not CHEMBL_ID.fullmatch(queried_id):
                raise EvidenceInputError("collector page lacks an explicit queried molecule identity")
            grouped_pages.setdefault(queried_id, []).append(page)

        all_receipts = [
            item for item in payload.get("page_receipts", [])
            if isinstance(item, Mapping)
        ]
        metadata_receipts: dict[str, Mapping[str, Any]] = {}
        for receipt in all_receipts:
            source_id = receipt.get("source_id")
            if not isinstance(source_id, str):
                continue
            for resource in ("target", "assay", "document", "molecule"):
                prefix = f"chembl_{resource}:"
                if source_id.startswith(prefix):
                    identifier = source_id[len(prefix):]
                    metadata_receipts[f"{resource}:{identifier}"] = receipt
                    break

        output: list[ActivityEvidence] = []
        used_page_receipts: set[int] = set()
        for queried_id, pages in grouped_pages.items():
            group_receipts: list[Mapping[str, Any]] = []
            for page in pages:
                wanted_hash = page.get("response_sha256")
                match_index = next((
                    index for index, receipt in enumerate(all_receipts)
                    if index not in used_page_receipts
                    and str(receipt.get("source_id", "")).startswith(f"chembl_activity:{queried_id}:")
                    and receipt.get("response_sha256") == wanted_hash
                ), None)
                if match_index is None:
                    group_receipts.append({})
                else:
                    used_page_receipts.add(match_index)
                    group_receipts.append(all_receipts[match_index])
            output.extend(activity_evidence_from_pages(
                queried_id,
                pages,
                source_kind=declared_kind,
                source_version=payload.get("source_version"),
                page_receipts=group_receipts,
                targets=payload.get("target_records", {}),
                assays=payload.get("assay_records", {}),
                documents=payload.get("document_records", {}),
                metadata_receipts=metadata_receipts,
            ))
        return output
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
            target_metadata = row.get("target_record")
            target_metadata = target_metadata if isinstance(target_metadata, Mapping) else {}
            projected.append({
                "target_chembl_id": row.get("target_chembl_id"),
                "target_name": (
                    raw.get("target_pref_name") or raw.get("pref_name") or raw.get("target_name")
                    or target_metadata.get("pref_name")
                ),
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


def _record_mapping(value: Any, label: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    raise TypeError(f"{label} must be a record object")


def _validate_molecule_record_dict(record: Mapping[str, Any]) -> None:
    molecule_id = record.get("molecule_chembl_id")
    if molecule_id is not None and (not isinstance(molecule_id, str) or not CHEMBL_ID.fullmatch(molecule_id)):
        raise EvidenceInputError("molecule entity has an invalid ChEMBL identity")
    status = record.get("structure_status")
    if status not in {"verified", "missing", "invalid", "excluded", "error", "unverified"}:
        raise EvidenceInputError("molecule entity has an unsupported structure_status")
    source_kind = record.get("source_kind")
    if source_kind not in ALLOWED_SOURCE_KINDS:
        raise EvidenceInputError("molecule entity has an unsupported source_kind")
    raw_structure = record.get("raw_structure")
    if raw_structure is not None and not isinstance(raw_structure, dict):
        raise EvidenceInputError("molecule raw_structure must be an object or null")
    provenance = record.get("source_provenance")
    if not isinstance(provenance, list):
        raise EvidenceInputError("molecule source_provenance must be a list")
    for source in provenance:
        if not isinstance(source, dict) or not _is_sha256(source.get("raw_sha256")):
            raise EvidenceInputError("molecule source provenance requires a raw SHA-256")
    smiles = record.get("canonical_smiles")
    structure_hash = record.get("structure_sha256")
    if smiles is None:
        if structure_hash is not None:
            raise EvidenceInputError("molecule structure hash requires canonical_smiles")
    else:
        if not isinstance(smiles, str) or not smiles:
            raise EvidenceInputError("canonical_smiles must be non-empty text")
        if not _is_sha256(structure_hash) or _sha256_bytes(smiles.encode("utf-8")) != structure_hash:
            raise EvidenceInputError("molecule structure_sha256 mismatch")
    if status == "verified":
        if molecule_id is None or smiles is None or not provenance:
            raise EvidenceInputError("verified molecule requires identity, structure, and provenance")
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                raise EvidenceInputError("verified molecule has invalid or multi-fragment canonical structure")
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except ImportError as exc:
            raise EvidenceInputError("RDKit is required to verify canonical molecule structures") from exc
        if canonical != smiles:
            raise EvidenceInputError("molecule canonical_smiles is not RDKit-canonical")
    elif not record.get("structure_status_reason"):
        raise EvidenceInputError("non-verified molecule structure requires a reason")


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
    if not isinstance(bundle.get("molecules"), list) or not isinstance(bundle.get("proteins"), list):
        raise EvidenceInputError("entity bundle molecules and proteins must be lists")
    for molecule in bundle["molecules"]:
        if not isinstance(molecule, dict):
            raise EvidenceInputError("entity bundle molecule must be an object")
        _validate_molecule_record_dict(molecule)
    for protein in bundle["proteins"]:
        if not isinstance(protein, dict):
            raise EvidenceInputError("entity bundle protein must be an object")
        try:
            validate_protein_record(ProteinRecord(**protein))
        except (TypeError, ValueError) as exc:
            raise EvidenceInputError(f"invalid shared ProteinRecord: {exc}") from exc
    for activity in bundle["activities"]:
        if not isinstance(activity, dict):
            raise EvidenceInputError("entity bundle activity must be an object")
        if activity.get("entity_bundle_hash") != expected_hash or activity.get("entity_bundle_version") != expected_version:
            raise EvidenceInputError("activity references a different entity bundle")
    provenance = bundle.get("source_provenance")
    if not isinstance(provenance, list):
        raise EvidenceInputError("entity bundle source_provenance must be a list")
    listed_sources = {
        (item.get("source_path"), item.get("raw_sha256"))
        for item in provenance
        if isinstance(item, dict) and isinstance(item.get("source_path"), str) and _is_sha256(item.get("raw_sha256"))
    }

    def require_source_link(source_path: Any, source_hash: Any, label: str, *, required: bool) -> None:
        if source_path is None and source_hash is None:
            if required:
                raise EvidenceInputError(f"{label} is missing its raw source path/hash")
            return
        if not isinstance(source_path, str) or not _is_sha256(source_hash):
            raise EvidenceInputError(f"{label} has an incomplete raw source reference")
        if Path(source_path).is_absolute() or ".." in Path(source_path).parts:
            raise EvidenceInputError(f"{label} raw source path is not bundle-relative")
        if (source_path, source_hash) not in listed_sources:
            raise EvidenceInputError(f"{label} raw source is not listed by the bundle manifest")

    for molecule in bundle["molecules"]:
        for source in molecule.get("source_provenance", []):
            require_source_link(
                source.get("source_path"), source.get("raw_sha256"),
                f"molecule {molecule.get('molecule_chembl_id')}", required=True,
            )
    for protein in bundle["proteins"]:
        for source in protein.get("source_provenance", []):
            require_source_link(
                source.get("source_path"), source.get("raw_sha256"),
                f"protein {protein.get('accession')}", required=True,
            )
    for activity in bundle["activities"]:
        complete_metadata = activity.get("metadata_status") == "complete"
        require_source_link(
            activity.get("source_path"), activity.get("source_sha256"),
            f"activity {activity.get('activity_id')} page", required=activity.get("pagination_status") == "complete",
        )
        for field_prefix, raw_field in (
            ("target_record", "target_record"),
            ("assay_record", "assay_record"),
            ("document_record", "document_record"),
        ):
            if activity.get(raw_field) is not None:
                require_source_link(
                    activity.get(f"{field_prefix}_source_path"),
                    activity.get(f"{field_prefix}_source_sha256"),
                    f"activity {activity.get('activity_id')} {field_prefix}",
                    required=complete_metadata,
                )
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
    activity_rows = [_record_mapping(row, "activity") for row in activities]
    bundle: dict[str, Any] = {
        "schema_version": ENTITY_SCHEMA_VERSION,
        "entity_version": None,
        "source_kind": _source_kind(source_kind, "legacy_unverified") if source_kind is not None else "mixed",
        "molecules": sorted((_record_mapping(item, "molecule") for item in (molecules or [])), key=lambda item: str(item.get("molecule_chembl_id") or "")),
        "proteins": sorted((_record_mapping(item, "protein") for item in (proteins or [])), key=lambda item: (str(item.get("taxonomy_id") or ""), str(item.get("accession") or ""))),
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


def _snapshot_local_collection(
    *,
    rows: Any,
    resource: str,
    id_field: str,
    root: Path,
    source_kind: str,
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise EvidenceInputError(f"chembl.{resource}s must be a list")
    records: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    row_snapshots: list[dict[str, Any]] = []
    ambiguous: set[str] = set()
    for index, value in enumerate(rows):
        record = _copy_json_object(value, f"chembl.{resource}s[{index}]")
        identifier = record.get(id_field)
        if not isinstance(identifier, str) or not CHEMBL_ID.fullmatch(identifier):
            identifier = None
        file_id = identifier or f"unresolved-{index + 1:04d}"
        relative = f"raw/chembl/{resource}/{file_id}.json"
        raw_bytes = _json_bytes(record)
        saved_path, saved_hash = _write_raw_bytes(root, relative, raw_bytes)
        source_id = f"chembl_{resource}:{identifier or f'unresolved-{index + 1:04d}'}"
        source = {
            "source_id": source_id,
            "source_kind": source_kind,
            "source_path": saved_path,
            "raw_sha256": saved_hash,
            "response_sha256": canonical_json_sha256(record),
        }
        provenance.append(source)
        row_snapshots.append({
            "record": record,
            "source_path": saved_path,
            "source_sha256": saved_hash,
            "source_id": source_id,
            "identifier": identifier,
        })
        if identifier is None:
            continue
        receipt_key = f"{resource}:{identifier}"
        if identifier in records:
            if canonical_json_sha256(records[identifier]) != canonical_json_sha256(record):
                records.pop(identifier, None)
                ambiguous.add(identifier)
            continue
        if identifier not in ambiguous:
            records[identifier] = record
            receipts[receipt_key] = {
                "source_id": source_id,
                "source_path": saved_path,
                "source_sha256": saved_hash,
                "response_sha256": canonical_json_sha256(record),
            }
    return {
        "records": records,
        "receipts": receipts,
        "rows": row_snapshots,
        "ambiguous_ids": sorted(ambiguous),
    }


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
    all_activities: list[ActivityEvidence] = []
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
    source_index: dict[str, Any] = {}
    metadata_receipts: dict[str, dict[str, Any]] = {}
    for resource, field_name, id_field in (
        ("molecule", "molecules", "molecule_chembl_id"),
        ("target", "targets", "target_chembl_id"),
        ("assay", "assays", "assay_chembl_id"),
        ("document", "documents", "document_chembl_id"),
    ):
        snapshot = _snapshot_local_collection(
            rows=chembl.get(field_name, []),
            resource=resource,
            id_field=id_field,
            root=root,
            source_kind=source_kind,
            provenance=provenance,
        )
        source_index[resource] = snapshot["records"]
        source_index[f"{resource}_rows"] = snapshot["rows"]
        source_index[f"{resource}_ambiguous_ids"] = snapshot["ambiguous_ids"]
        metadata_receipts.update(snapshot["receipts"])
    target_map = source_index["target"]
    assay_map = source_index["assay"]
    document_map = source_index["document"]
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
            assays=assay_map,
            documents=document_map,
            metadata_receipts=metadata_receipts,
        ))
    metadata_status = "complete"
    known_molecule_ids = set(source_index["molecule"])
    for activity in all_activities:
        if activity.metadata_status != "complete" or activity.molecule_chembl_id not in known_molecule_ids:
            metadata_status = "incomplete"
            break
    source_index["metadata_receipts"] = metadata_receipts
    return all_activities, provenance, overall, metadata_status, source_index


def _local_entity_records(
    manifest: Mapping[str, Any],
    root: Path,
    manifest_dir: Path,
    source_kind: str,
    provenance: list[dict[str, Any]],
    source_index: Mapping[str, Any],
    activities: Sequence[ActivityEvidence],
) -> tuple[list[MoleculeRecord], list[ProteinRecord], list[dict[str, Any]]]:
    molecules: list[MoleculeRecord] = []
    for snapshot in source_index.get("molecule_rows", []):
        molecules.append(molecule_record_from_source(
            snapshot["record"],
            source_kind=source_kind,
            source_path=snapshot["source_path"],
            source_sha256=snapshot["source_sha256"],
        ))

    chembl = manifest.get("chembl", {})
    target_records = source_index.get("target", {})
    metadata_receipts = source_index.get("metadata_receipts", {})
    target_ids = sorted({
        row.target_chembl_id for row in activities
        if isinstance(row.target_chembl_id, str)
    })
    exclusion_rows: list[dict[str, Any]] = []
    component_rows: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for target_id in target_ids:
        target = target_records.get(target_id)
        target_source = metadata_receipts.get(f"target:{target_id}", {})
        if not isinstance(target, Mapping):
            exclusion_rows.append({
                "target_chembl_id": target_id,
                "reason_code": "target_metadata_missing",
                "resolution_status": "unverified",
            })
            continue
        resolutions = resolve_target_components(
            target,
            source_kind=source_kind,
            source_path=target_source.get("source_path"),
            source_sha256=target_source.get("source_sha256"),
        )
        for resolution in resolutions:
            if resolution.get("resolution_status") != "verified":
                exclusion_rows.append({
                    "target_chembl_id": target_id,
                    "component_index": resolution.get("component_index"),
                    "component": deepcopy(resolution.get("component")),
                    "reason_code": resolution.get("reason_code"),
                    "resolution_status": resolution.get("resolution_status"),
                    "target_record_sha256": resolution.get("target_record_sha256"),
                    "target_record_source": deepcopy(resolution.get("target_record_source")),
                })
                continue
            key = (resolution["taxonomy_id"], resolution["accession"])
            component_rows.setdefault(key, []).append(resolution)

    uniprot_records = manifest.get("uniprot", {})
    if not isinstance(uniprot_records, Mapping):
        uniprot_records = {}
    proteins: list[ProteinRecord] = []
    for (taxonomy_id, accession), references in sorted(component_rows.items()):
        component = references[0]
        raw_uniprot = uniprot_records.get(accession)
        uniprot_kind = source_kind
        if isinstance(raw_uniprot, Mapping) and isinstance(raw_uniprot.get("source_kind"), str):
            uniprot_kind = _source_kind(raw_uniprot.get("source_kind"))

        def resolve_input_path(value: Any) -> Path:
            if not isinstance(value, str) or not value.strip():
                raise EvidenceInputError("UniProt manifest source path must be non-empty text")
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = manifest_dir / candidate
            return candidate.resolve()

        def copy_input_source(value: Any, relative_path: str, source_id: str) -> tuple[Path, str]:
            actual_path = resolve_input_path(value)
            content = actual_path.read_bytes()
            saved_path, saved_hash = _write_raw_bytes(root, relative_path, content)
            provenance.append({
                "source_id": source_id,
                "source_kind": uniprot_kind,
                "source_path": saved_path,
                "source_origin_name": actual_path.name,
                "raw_sha256": saved_hash,
            })
            return root / saved_path, saved_hash

        if isinstance(raw_uniprot, str):
            raw_uniprot = {"json_path": raw_uniprot}
        json_reference = None
        fasta_reference = None
        metadata_reference = None
        if isinstance(raw_uniprot, Mapping):
            if "primaryAccession" in raw_uniprot:
                json_payload = _copy_json_object(dict(raw_uniprot), f"UniProt {accession}")
                raw_bytes = _json_bytes(json_payload)
                relative = f"raw/uniprot/{accession}.json"
                saved_path, saved_hash = _write_raw_bytes(root, relative, raw_bytes)
                provenance.append({
                    "source_id": f"uniprot:{accession}",
                    "source_kind": uniprot_kind,
                    "source_path": saved_path,
                    "raw_sha256": saved_hash,
                    "response_sha256": canonical_json_sha256(json_payload),
                })
                protein = hydrate_protein_record(
                    component,
                    source_kind=uniprot_kind,
                    source_path=saved_path,
                    uniprot_json=json_payload,
                    raw_response_bytes=raw_bytes,
                )
            else:
                json_reference = raw_uniprot.get("json_path", raw_uniprot.get("raw_json_path"))
                fasta_reference = raw_uniprot.get("fasta_path")
                metadata_reference = raw_uniprot.get("taxonomy_metadata_path", raw_uniprot.get("metadata_path"))
        else:
            protein = _uniprot_source_error_record(
                component, source_kind=uniprot_kind, sequence_status="missing",
                reason="local_sequence_source_missing",
            )

        if isinstance(raw_uniprot, Mapping) and "primaryAccession" not in raw_uniprot:
            if isinstance(json_reference, str):
                try:
                    actual_path = resolve_input_path(json_reference)
                    raw_bytes = actual_path.read_bytes()
                    json_payload = json.loads(raw_bytes.decode("utf-8-sig"))
                    if not isinstance(json_payload, dict):
                        raise EvidenceInputError("UniProt JSON source root must be an object")
                    relative = f"raw/uniprot/{accession}.json"
                    saved_path, saved_hash = _write_raw_bytes(root, relative, raw_bytes)
                    provenance.append({
                        "source_id": f"uniprot:{accession}",
                        "source_kind": uniprot_kind,
                        "source_path": saved_path,
                        "source_origin_name": actual_path.name,
                        "raw_sha256": saved_hash,
                    })
                    protein = hydrate_protein_record(
                        component, source_kind=uniprot_kind, source_path=saved_path,
                        uniprot_json=json_payload, raw_response_bytes=raw_bytes,
                    )
                except Exception as exc:
                    protein = _uniprot_source_error_record(
                        component, source_kind=uniprot_kind, sequence_status="error",
                        reason=f"local_uniprot_json_error:{type(exc).__name__}",
                    )
            elif isinstance(fasta_reference, str) and isinstance(metadata_reference, str):
                try:
                    fasta_saved, _ = copy_input_source(
                        fasta_reference, f"raw/uniprot/{accession}.fasta", f"uniprot_fasta:{accession}"
                    )
                    metadata_saved, _ = copy_input_source(
                        metadata_reference,
                        f"raw/uniprot/{accession}-metadata.json",
                        f"uniprot_taxonomy_metadata:{accession}",
                    )
                    protein = hydrate_protein_record(
                        component,
                        source_kind=uniprot_kind,
                        fasta_path=fasta_saved,
                        taxonomy_metadata_path=metadata_saved,
                        fasta_source_path=f"raw/uniprot/{accession}.fasta",
                        taxonomy_metadata_source_path=f"raw/uniprot/{accession}-metadata.json",
                    )
                except Exception as exc:
                    protein = _uniprot_source_error_record(
                        component, source_kind=uniprot_kind, sequence_status="error",
                        reason=f"local_fasta_metadata_error:{type(exc).__name__}",
                    )
            else:
                protein = hydrate_protein_record(component, source_kind=uniprot_kind)
        reference_records = [
            {
                "target_chembl_id": reference.get("target_chembl_id"),
                "target_type": reference.get("target_type"),
                "organism": reference.get("organism"),
                "taxonomy_id": taxonomy_id,
                "taxonomy_evidence_path": reference.get("taxonomy_evidence_path"),
                "component_index": reference.get("component_index"),
                "component": deepcopy(reference.get("component")),
                "target_record_sha256": reference.get("target_record_sha256"),
                "target_record_source": deepcopy(reference.get("target_record_source")),
            }
            for reference in references
        ]
        protein = replace(protein, component_evidence={"references": reference_records})
        validate_protein_record(protein)
        proteins.append(protein)
    return molecules, proteins, exclusion_rows


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
        activities, provenance, pagination_status, metadata_status, source_index = _local_page_records(
            manifest, root / "raw", root, source_path, source_kind
        )
        molecules, proteins, exclusions = _local_entity_records(
            manifest, root, source_path.parent, source_kind, provenance, source_index, activities
        )
        if pagination_status != "complete" or metadata_status != "complete":
            partial = {
                "status": "incomplete",
                "source_kind": source_kind,
                "pagination_status": pagination_status,
                "metadata_status": metadata_status,
                "activity_count": len(activities),
                "molecule_count": len(molecules),
                "protein_count": len(proteins),
                "exclusion_count": len(exclusions),
                "source_provenance": provenance,
            }
            (root / "partial_ingest.json").write_bytes(_json_bytes(partial))
            return partial
        bundle = write_entity_bundle(
            root, activities=activities, source_provenance=provenance,
            molecules=molecules,
            proteins=proteins,
            exclusions=exclusions,
            source_kind=source_kind,
        )
        return {
            "status": "complete",
            "source_kind": source_kind,
            "pagination_status": pagination_status,
            "metadata_status": metadata_status,
            "activity_count": len(activities),
            "molecule_count": len(molecules),
            "protein_count": len(proteins),
            "exclusion_count": len(exclusions),
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
    deadline: Optional[float] = None,
) -> tuple[dict[str, Any], bytes, str]:
    if deadline is None:
        deadline = time.monotonic() + timeout
    if time.monotonic() >= deadline:
        raise BoundedFetchError("ChEMBL total timeout budget exhausted")
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
            declared_length = int(content_length)
            if declared_length < 0:
                raise BoundedFetchError("negative ChEMBL Content-Length")
            if declared_length > max_response_bytes:
                raise BoundedFetchError("ChEMBL response exceeds max_response_bytes")
        except ValueError as exc:
            raise BoundedFetchError("invalid ChEMBL Content-Length") from exc
    chunks: list[bytes] = []
    total = 0
    iterator = response.iter_content(chunk_size=min(max_response_bytes + 1, 65536))
    for chunk in iterator:
        if time.monotonic() >= deadline:
            raise BoundedFetchError("ChEMBL total timeout budget exhausted while streaming")
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
    assay_records: dict[str, dict[str, Any]] = {}
    document_records: dict[str, dict[str, Any]] = {}
    page_statuses: dict[str, str] = {value: "complete" for value in unique_ids}
    incomplete_reasons: list[str] = []
    metadata_failures: list[str] = []

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
            deadline=deadline,
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
                for row in rows:
                    if row.get("molecule_chembl_id") != molecule_id:
                        reason = f"activity:{row.get('activity_id')}:molecule_identity_mismatch"
                        metadata_failures.append(reason)
                        incomplete_reasons.append(reason)
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
                    "source_url": next_url,
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
            reason = f"target:{target_id}:{exc}"
            incomplete_reasons.append(reason)
            metadata_failures.append(reason)

    activity_rows = [
        row
        for page in activity_pages
        for row in page.get("activities", [])
        if isinstance(row, dict)
    ]

    def fetch_linked_metadata(resource: str, identifier_field: str, collection: dict[str, dict[str, Any]]) -> None:
        requested_ids = sorted({
            row.get(identifier_field)
            for row in activity_rows
            if isinstance(row.get(identifier_field), str)
        })
        for identifier in requested_ids:
            if not CHEMBL_ID.fullmatch(identifier):
                reason = f"{resource}:{identifier}:invalid_chembl_id"
                incomplete_reasons.append(reason)
                metadata_failures.append(reason)
                continue
            try:
                payload, body, response_hash = request(f"{CHEMBL_API_ROOT}/{resource}/{identifier}.json")
                response_id = payload.get(identifier_field)
                if response_id != identifier:
                    raise BoundedFetchError(f"{resource} response identity mismatch")
                collection[identifier] = payload
                page_receipts.append({
                    "source_id": f"chembl_{resource}:{identifier}",
                    "source_kind": "online_snapshot",
                    "source_sha256": response_hash,
                    "response_sha256": response_hash,
                    "payload": payload,
                    "raw_response_text": body.decode("utf-8"),
                })
            except Exception as exc:
                reason = f"{resource}:{identifier}:{exc}"
                incomplete_reasons.append(reason)
                metadata_failures.append(reason)

    fetch_linked_metadata("assay", "assay_chembl_id", assay_records)
    fetch_linked_metadata("document", "document_chembl_id", document_records)

    for index, row in enumerate(activity_rows):
        for resource, identifier_field, collection in (
            ("target", "target_chembl_id", target_records),
            ("assay", "assay_chembl_id", assay_records),
            ("document", "document_chembl_id", document_records),
        ):
            identifier = row.get(identifier_field)
            if not isinstance(identifier, str):
                reason = f"activity:{row.get('activity_id')}:{resource}_id_missing"
                incomplete_reasons.append(reason)
                metadata_failures.append(reason)
            elif identifier not in collection:
                reason = f"activity:{row.get('activity_id')}:{resource}_metadata_missing"
                if reason not in incomplete_reasons:
                    incomplete_reasons.append(reason)
                if reason not in metadata_failures:
                    metadata_failures.append(reason)
        assay_id = row.get("assay_chembl_id")
        assay = assay_records.get(assay_id) if isinstance(assay_id, str) else None
        if not isinstance(assay, dict) or assay.get("confidence_score") is None:
            if row.get("confidence_score") is None:
                reason = f"activity:{row.get('activity_id')}:assay_confidence_missing"
                incomplete_reasons.append(reason)
                metadata_failures.append(reason)

    metadata_status = "incomplete" if metadata_failures else "complete"

    pagination_status = "complete" if all(value == "complete" for value in page_statuses.values()) else "incomplete"
    return {
        "status": "complete" if pagination_status == "complete" and metadata_status == "complete" else "incomplete",
        "source_kind": "online_snapshot",
        "source_version": None,
        "molecule_ids": unique_ids,
        "molecule_records": molecule_records,
        "target_records": target_records,
        "assay_records": assay_records,
        "document_records": document_records,
        "activity_pages": activity_pages,
        "page_receipts": page_receipts,
        "page_statuses": page_statuses,
        "pagination_status": pagination_status,
        "metadata_status": metadata_status,
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
    assay_records = result.get("assay_records", {})
    document_records = result.get("document_records", {})
    evidence_receipts: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    for index, page in enumerate(result.get("activity_pages", [])):
        raw = page.get("raw_response", page)
        molecule_id = next((
            str(row.get("molecule_chembl_id"))
            for row in page.get("activities", [])
            if isinstance(row.get("molecule_chembl_id"), str)
        ), page.get("molecule_chembl_id", "CHEMBL_UNKNOWN"))
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
        if not isinstance(receipt, dict) or not str(receipt.get("source_id", "")).startswith((
            "chembl_molecule:", "chembl_target:", "chembl_assay:", "chembl_document:",
        )):
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
            "source_url": page.get("source_url"),
        }
        pages_by_molecule.setdefault(molecule_id, []).append({
            **page_obj,
        })
        receipts_by_molecule.setdefault(molecule_id, []).append(evidence_receipts[index])
    metadata_receipts: dict[str, dict[str, Any]] = {}
    for source in provenance:
        source_id = source.get("source_id")
        if not isinstance(source_id, str):
            continue
        for resource in ("target", "assay", "document"):
            prefix = f"chembl_{resource}:"
            if source_id.startswith(prefix):
                metadata_receipts[f"{resource}:{source_id[len(prefix):]}"] = {
                    "source_path": source.get("source_path"),
                    "source_sha256": source.get("raw_sha256"),
                }
                break
    activities: list[ActivityEvidence] = []
    for molecule_id, pages in sorted(pages_by_molecule.items()):
        activities.extend(activity_evidence_from_pages(
            molecule_id, pages, source_kind="online_snapshot",
            page_receipts=receipts_by_molecule[molecule_id],
            targets=target_records,
            assays=assay_records,
            documents=document_records,
            metadata_receipts=metadata_receipts,
        ))
    return activities, provenance


def _entities_from_chembl_capture(
    result: Mapping[str, Any],
    provenance: Sequence[Mapping[str, Any]],
) -> tuple[list[MoleculeRecord], list[ProteinRecord], list[dict[str, Any]]]:
    source_by_id = {
        item.get("source_id"): item
        for item in provenance
        if isinstance(item, Mapping) and isinstance(item.get("source_id"), str)
    }
    molecules: list[MoleculeRecord] = []
    molecule_records = result.get("molecule_records", {})
    if isinstance(molecule_records, Mapping):
        for molecule_id, raw in sorted(molecule_records.items()):
            source = source_by_id.get(f"chembl_molecule:{molecule_id}", {})
            molecules.append(molecule_record_from_source(
                raw,
                source_kind="online_snapshot",
                source_path=source.get("source_path"),
                source_sha256=source.get("raw_sha256"),
                source_url=f"{CHEMBL_API_ROOT}/molecule/{molecule_id}.json",
            ))

    target_records = result.get("target_records", {})
    if not isinstance(target_records, Mapping):
        target_records = {}
    activity_pages = result.get("activity_pages", [])
    target_ids = sorted({
        row.get("target_chembl_id")
        for page in activity_pages if isinstance(page, Mapping)
        for row in page.get("activities", []) if isinstance(row, Mapping)
        and isinstance(row.get("target_chembl_id"), str)
    })
    exclusions: list[dict[str, Any]] = []
    component_refs: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for target_id in target_ids:
        target = target_records.get(target_id)
        source = source_by_id.get(f"chembl_target:{target_id}", {})
        if not isinstance(target, Mapping):
            exclusions.append({"target_chembl_id": target_id, "reason_code": "target_metadata_missing"})
            continue
        for resolution in resolve_target_components(
            target,
            source_kind="online_snapshot",
            source_path=source.get("source_path"),
            source_sha256=source.get("raw_sha256"),
            source_url=f"{CHEMBL_API_ROOT}/target/{target_id}.json",
        ):
            if resolution.get("resolution_status") != "verified":
                exclusions.append({
                    "target_chembl_id": target_id,
                    "component_index": resolution.get("component_index"),
                    "component": deepcopy(resolution.get("component")),
                    "reason_code": resolution.get("reason_code"),
                    "resolution_status": resolution.get("resolution_status"),
                    "target_record_sha256": resolution.get("target_record_sha256"),
                    "target_record_source": deepcopy(resolution.get("target_record_source")),
                })
                continue
            key = (resolution["taxonomy_id"], resolution["accession"])
            component_refs.setdefault(key, []).append(resolution)

    proteins: list[ProteinRecord] = []
    for (taxonomy_id, accession), references in sorted(component_refs.items()):
        protein = hydrate_protein_record(references[0], source_kind="online_snapshot")
        protein = replace(protein, component_evidence={
            "references": [
                {
                    "target_chembl_id": reference.get("target_chembl_id"),
                    "target_record_sha256": reference.get("target_record_sha256"),
                    "target_record_source": deepcopy(reference.get("target_record_source")),
                    "taxonomy_id": taxonomy_id,
                    "accession": accession,
                    "component_index": reference.get("component_index"),
                    "component": deepcopy(reference.get("component")),
                }
                for reference in references
            ]
        })
        validate_protein_record(protein)
        proteins.append(protein)
    return molecules, proteins, exclusions


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
    if result["pagination_status"] == "complete" and result.get("metadata_status") == "complete":
        molecules, proteins, exclusions = _entities_from_chembl_capture(result, provenance)
        bundle = write_entity_bundle(
            root, activities=activities, source_provenance=provenance,
            molecules=molecules,
            proteins=proteins,
            exclusions=exclusions,
            source_kind="online_snapshot",
        )
        result["content_sha256"] = bundle["content_sha256"]
        result["entity_version"] = bundle["entity_version"]
        result["bundle_path"] = str(root / "entities.json")
        result["molecule_count"] = len(molecules)
        result["protein_count"] = len(proteins)
        result["exclusion_count"] = len(exclusions)
    else:
        partial = {
            "status": "incomplete",
            "source_kind": "online_snapshot",
            "pagination_status": result["pagination_status"],
            "metadata_status": result.get("metadata_status", "incomplete"),
            "overall_status": "incomplete",
            "activity_count": len(activities),
            "molecule_count": 0,
            "protein_count": 0,
            "exclusion_count": 0,
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
    verify = subparsers.add_parser("verify-bundle", help="verify entity bundle hash and raw source files")
    verify.add_argument("bundle", type=Path)
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
        elif args.command == "verify-bundle":
            bundle = load_entity_bundle(args.bundle)
            result = {
                "status": "verified",
                "entity_version": bundle["entity_version"],
                "content_sha256": bundle["content_sha256"],
                "molecule_count": len(bundle["molecules"]),
                "protein_count": len(bundle["proteins"]),
                "activity_count": len(bundle["activities"]),
                "exclusion_count": len(bundle.get("exclusions", [])),
            }
        else:
            raise EvidenceInputError(f"unsupported command: {args.command}")
    except Exception as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"complete", "verified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

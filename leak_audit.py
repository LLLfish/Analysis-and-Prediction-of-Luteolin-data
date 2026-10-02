"""Independent, source-bound leakage audit for frozen double-cold splits.

This module deliberately does not import ``dataset_builder``. It reloads the
producer's verified schema-v2 entity bundle, validates every byte-bound source
receipt, and recomputes molecular/protein neighbor edges and pair-level support
from canonical entities and evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

from data_contracts import ProteinRecord, canonical_json_sha256, validate_protein_record
from evidence_builder import EvidenceInputError, load_entity_bundle
from label_policy import apply_label_policy, group_evidence


_POLICY_PATH = Path(__file__).resolve().parent / "policies" / "split_policy.json"
_LABEL_POLICY_PATH = Path(__file__).resolve().parent / "policies" / "label_policy.json"
_POLICY_RELATIVE_PATHS = {
    "split": "policies/split_policy.json",
    "label": "policies/label_policy.json",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHEMBL_ID = re.compile(r"^CHEMBL[0-9]+$")
_SPLITS = ("train", "val", "test")
_CLASSES = ("active", "inactive")
_BUNDLE_SCHEMA_VERSION = 2


class AuditInputError(ValueError):
    """The caller supplied an incomplete or malformed audit request."""


class _AuditStop(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class NeighborHit:
    kind: str
    entity_ids: tuple[str, str]
    metric: float
    threshold: float
    details: Mapping[str, Any] = field(default_factory=dict)
    source_evidence: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "entity_ids": list(self.entity_ids),
            "metric": self.metric,
            "threshold": self.threshold,
            "details": _json_safe(self.details),
            "source_evidence": [_json_safe(item) for item in self.source_evidence],
        }


@dataclass(frozen=True)
class NeighborRecomputeResult:
    status: str
    molecule_neighbors: tuple[NeighborHit, ...] = ()
    protein_neighbors: tuple[NeighborHit, ...] = ()
    error: Optional[str] = None

    @property
    def exit_code(self) -> int:
        if self.status == "ok":
            return 0
        if self.status in {"size_limit", "dependency_missing"}:
            return 4
        return 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "molecule_neighbors": [row.to_dict() for row in self.molecule_neighbors],
            "protein_neighbors": [row.to_dict() for row in self.protein_neighbors],
            "error": self.error,
        }


@dataclass(frozen=True)
class AuditFinding:
    code: str
    message: str
    entity_ids: tuple[str, ...] = ()
    split: Optional[str] = None
    metric: Optional[float] = None
    threshold: Optional[float] = None
    source_evidence: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "entity_ids": list(self.entity_ids),
            "split": self.split,
            "metric": self.metric,
            "threshold": self.threshold,
            "source_evidence": [_json_safe(item) for item in self.source_evidence],
        }


@dataclass(frozen=True)
class AuditReport:
    status: str
    exit_code: int
    findings: tuple[AuditFinding, ...]
    class_support: Mapping[str, Mapping[str, int]]
    training_ready: bool
    entity_version: Optional[str] = None
    entity_bundle_hash: Optional[str] = None
    split_policy_sha256: Optional[str] = None
    source_provenance: tuple[Mapping[str, Any], ...] = ()
    error: Optional[str] = None
    split_policy_file_sha256: Optional[str] = None
    label_policy_sha256: Optional[str] = None
    label_policy_semantic_sha256: Optional[str] = None
    verified_entity_counts: Optional[Mapping[str, int]] = None
    component_counts: Optional[Mapping[str, int]] = None
    label_disposition_counts: Optional[Mapping[str, int]] = None
    ledger_disposition_counts: Optional[Mapping[str, int]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "exit_code": self.exit_code,
            "findings": [item.to_dict() for item in self.findings],
            "class_support": {key: dict(value) for key, value in self.class_support.items()},
            "training_ready": self.training_ready,
            "entity_version": self.entity_version,
            "entity_bundle_hash": self.entity_bundle_hash,
            "split_policy_sha256": self.split_policy_sha256,
            "split_policy_file_sha256": self.split_policy_file_sha256,
            "label_policy_sha256": self.label_policy_sha256,
            "label_policy_semantic_sha256": self.label_policy_semantic_sha256,
            "verified_entity_counts": dict(self.verified_entity_counts) if self.verified_entity_counts is not None else None,
            "component_counts": dict(self.component_counts) if self.component_counts is not None else None,
            "label_disposition_counts": dict(self.label_disposition_counts) if self.label_disposition_counts is not None else None,
            "ledger_disposition_counts": dict(self.ledger_disposition_counts) if self.ledger_disposition_counts is not None else None,
            "source_provenance": [_json_safe(item) for item in self.source_provenance],
            "error": self.error,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def _reject_json_constant(token: str) -> None:
    raise AuditInputError(f"non-finite JSON number is not allowed: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditInputError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_json_bytes(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8-sig"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, AuditInputError) as exc:
        raise AuditInputError(f"{label} is not valid unique-key UTF-8 JSON: {exc}") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(value: Any) -> str:
    return canonical_json_sha256(value)


def _bundle_hash_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(bundle))
    payload.pop("content_sha256", None)
    payload.pop("entity_version", None)
    activities = payload.get("activities", [])
    if isinstance(activities, list):
        for activity in activities:
            if isinstance(activity, dict):
                activity.pop("entity_bundle_hash", None)
                activity.pop("entity_bundle_version", None)
    return payload


def _strict_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise AuditInputError(f"{label} must be a positive integer")
    return value


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AuditInputError(f"{label} must be a nonnegative integer")
    return value


def _load_policy(policy_path: Any = None, budget_overrides: Optional[Mapping[str, Any]] = None) -> tuple[dict[str, Any], str, dict[str, int]]:
    path = _POLICY_PATH if policy_path is None else Path(policy_path)
    if not isinstance(path, Path) or not path.is_file():
        raise AuditInputError("split policy must identify an existing JSON file")
    raw = path.read_bytes()
    policy = _read_json_bytes(raw, "split policy")
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version", "policy_version", "entity_bundle_schema_version", "molecule",
        "protein", "connected_components", "split", "budgets",
    }:
        raise AuditInputError("split policy has unknown or missing top-level fields")
    if (type(policy.get("schema_version")) is not int or policy.get("schema_version") != 1
            or type(policy.get("entity_bundle_schema_version")) is not int
            or policy.get("entity_bundle_schema_version") != _BUNDLE_SCHEMA_VERSION
            or policy.get("policy_version") != "double-cold-neighbor-split-v1"):
        raise AuditInputError("unsupported split policy version")
    molecule = policy.get("molecule")
    fingerprint = molecule.get("fingerprint") if isinstance(molecule, dict) else None
    scaffold = molecule.get("scaffold") if isinstance(molecule, dict) else None
    if (not isinstance(molecule, dict) or set(molecule) != {"identity_fields", "scaffold", "fingerprint"}
            or molecule.get("identity_fields") != ["molecule_chembl_id"]
            or not isinstance(scaffold, dict)
            or scaffold.get("method") != "RDKit Murcko scaffold"
            or scaffold.get("empty_scaffold_links") is not False
            or scaffold.get("edge_when") != "both nonempty canonical scaffold strings are equal"
            or not isinstance(fingerprint, dict)
            or fingerprint.get("method") != "Morgan"
            or type(fingerprint.get("radius")) is not int or fingerprint.get("radius") != 2
            or type(fingerprint.get("bits")) is not int or fingerprint.get("bits") != 2048
            or fingerprint.get("similarity") != "Tanimoto"
            or fingerprint.get("operator") != ">="
            or fingerprint.get("inclusive") is not True
            or fingerprint.get("threshold") != 0.8):
        raise AuditInputError("split policy molecule identity/scaffold/fingerprint contract is unsupported")
    protein = policy.get("protein")
    alignment = protein.get("alignment") if isinstance(protein, dict) else None
    identity = protein.get("identity") if isinstance(protein, dict) else None
    coverage = protein.get("coverage") if isinstance(protein, dict) else None
    if (not isinstance(protein, dict) or set(protein) != {"identity_fields", "alignment", "identity", "coverage"}
            or protein.get("identity_fields") != ["taxonomy_id", "accession"]
            or not isinstance(alignment, dict)
            or alignment.get("api") != "Bio.Align.PairwiseAligner"
            or alignment.get("mode") != "global"
            or alignment.get("substitution_matrix") != "BLOSUM62"
            or alignment.get("open_gap_score") != -10.0
            or alignment.get("extend_gap_score") != -0.5
            or alignment.get("epsilon") != 0.0
            or alignment.get("tie_policy") != "first optimal alignment in PairwiseAligner iteration order after stable sequence identity ordering"
            or alignment.get("aligned_non_gap_columns") != "columns where both complete-sequence indices advance together"
            or alignment.get("gap_columns_in_identity") is not False
            or not isinstance(identity, dict)
            or identity.get("formula") != "exact_residue_matches / aligned_non_gap_columns"
            or identity.get("operator") != ">="
            or identity.get("inclusive") is not True
            or identity.get("threshold") != 0.3
            or not isinstance(coverage, dict)
            or coverage.get("formula") != "aligned_non_gap_columns / each full_sequence_length"
            or coverage.get("operator") != ">="
            or coverage.get("inclusive") is not True
            or coverage.get("threshold") != 0.8):
        raise AuditInputError("split policy protein alignment contract is unsupported")
    split_policy = policy.get("split")
    if (not isinstance(split_policy, dict)
            or split_policy.get("names") != ["train", "val", "test"]
            or split_policy.get("ratios") != {"train": 0.8, "val": 0.1, "test": 0.1}
            or type(split_policy.get("seed")) is not int or split_policy.get("seed") != 42
            or split_policy.get("cluster_order") != "sort stable cluster IDs, then random.Random(seed).shuffle"
            or split_policy.get("allocation") != "largest_remainder"
            or split_policy.get("rounding_tie_order") != ["train", "val", "test"]
            or policy.get("connected_components") != "undirected transitive connected components"):
        raise AuditInputError("split policy allocation/component contract is unsupported")
    budgets = policy.get("budgets")
    if (not isinstance(budgets, dict)
            or set(budgets) != {"molecule_pairs", "protein_pairs", "alignment_cells", "alignment_cells_measure"}
            or budgets.get("alignment_cells_measure") != "sum of len(sequence_a) * len(sequence_b) for every alignment attempted in one build"):
        raise AuditInputError("split policy budgets are missing")
    limits = {
        key: _strict_nonnegative_int(budgets.get(key), f"budget {key}")
        for key in ("molecule_pairs", "protein_pairs", "alignment_cells")
    }
    if budget_overrides is not None:
        if not isinstance(budget_overrides, Mapping):
            raise AuditInputError("budget_overrides must be an object")
        for key, value in budget_overrides.items():
            if key not in limits:
                raise AuditInputError(f"unsupported audit budget: {key}")
            limits[key] = _strict_nonnegative_int(value, f"budget override {key}")
    return policy, _canonical_hash(policy), limits


def _dependency_error(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return isinstance(exc, (ImportError, ModuleNotFoundError)) or any(
        marker in message for marker in ("rdkit is required", "biopython", "bio.align is required", "no module named 'bio'")
    )


def _safe_relative_file(root: Path, relative_path: Any, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise AuditInputError(f"{label} path must be non-empty relative text")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AuditInputError(f"{label} path escapes its frozen root: {relative_path!r}")
    root_resolved = root.resolve(strict=True)
    unresolved = root / candidate
    resolved = unresolved.resolve(strict=True)
    try:
        common = os.path.normcase(os.path.commonpath((str(root_resolved), str(resolved))))
    except ValueError as exc:
        raise AuditInputError(f"{label} path is on a different volume") from exc
    if common != os.path.normcase(str(root_resolved)):
        raise AuditInputError(f"{label} path resolves outside its frozen root")
    cursor = unresolved
    while cursor != root and root in cursor.parents:
        is_junction = getattr(cursor, "is_junction", None)
        try:
            import stat
            reparse = bool(getattr(cursor.lstat(), "st_file_attributes", 0)
                           & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        except (OSError, AttributeError):
            reparse = False
        if cursor.is_symlink() or (callable(is_junction) and is_junction()) or reparse:
            raise AuditInputError(f"{label} path crosses a symbolic link or junction")
        cursor = cursor.parent
    if not resolved.is_file():
        raise AuditInputError(f"{label} path is not a file")
    return resolved


def _load_policy_snapshot(
    manifest_root: Path,
    policy_bindings: Any,
    kind: str,
    override_path: Any = None,
) -> tuple[Optional[Path], Optional[bytes], Optional[dict[str, Any]], Optional[AuditFinding]]:
    if not isinstance(policy_bindings, Mapping) or set(policy_bindings) != {"split", "label"}:
        return None, None, None, AuditFinding(
            code="policy_bindings_invalid", message="manifest policies must bind exactly split and label snapshots",
        )
    reference = policy_bindings.get(kind)
    required = {"path", "file_sha256", "semantic_sha256", "schema_version", "policy_version"}
    if not isinstance(reference, Mapping) or set(reference) != required:
        return None, None, None, AuditFinding(
            code=f"{kind}_policy_binding_invalid", message=f"frozen {kind} policy binding is incomplete or has unknown fields",
        )
    expected_path = _POLICY_RELATIVE_PATHS[kind]
    if reference.get("path") != expected_path:
        return None, None, None, AuditFinding(
            code=f"{kind}_policy_path_mismatch",
            message=f"frozen {kind} policy path must be the canonical version-relative snapshot path",
            entity_ids=(str(reference.get("path")), expected_path),
        )
    for field_name in ("file_sha256", "semantic_sha256"):
        if not isinstance(reference.get(field_name), str) or not _SHA256.fullmatch(reference[field_name]):
            return None, None, None, AuditFinding(
                code=f"{kind}_policy_binding_invalid",
                message=f"frozen {kind} policy {field_name} is missing or malformed",
            )
    if type(reference.get("schema_version")) is not int or not isinstance(reference.get("policy_version"), str):
        return None, None, None, AuditFinding(
            code=f"{kind}_policy_binding_invalid", message=f"frozen {kind} policy version metadata is malformed",
        )
    try:
        snapshot_path = _safe_relative_file(manifest_root, reference.get("path"), f"{kind} policy snapshot")
        raw = snapshot_path.read_bytes()
    except (AuditInputError, OSError) as exc:
        return None, None, None, AuditFinding(
            code=f"{kind}_policy_snapshot_missing", message=str(exc),
            entity_ids=(expected_path,),
        )
    actual_file_sha256 = _sha256(raw)
    if actual_file_sha256 != reference.get("file_sha256"):
        return None, None, None, AuditFinding(
            code=f"{kind}_policy_snapshot_hash_mismatch",
            message=f"frozen {kind} policy bytes differ from their manifest file hash",
            entity_ids=(actual_file_sha256, str(reference.get("file_sha256"))),
        )
    try:
        policy = _read_json_bytes(raw, f"frozen {kind} policy")
    except AuditInputError as exc:
        return None, None, None, AuditFinding(
            code=f"{kind}_policy_snapshot_invalid", message=str(exc), entity_ids=(expected_path,),
        )
    if not isinstance(policy, Mapping):
        return None, None, None, AuditFinding(
            code=f"{kind}_policy_snapshot_invalid", message=f"frozen {kind} policy root must be an object",
        )
    actual_semantic_sha256 = _canonical_hash(policy)
    if actual_semantic_sha256 != reference.get("semantic_sha256"):
        return None, None, None, AuditFinding(
            code=f"{kind}_policy_semantic_mismatch",
            message=f"frozen {kind} policy meaning differs from its manifest semantic hash",
            entity_ids=(actual_semantic_sha256, str(reference.get("semantic_sha256"))),
        )
    if (type(policy.get("schema_version")) is not int
            or policy.get("schema_version") != reference.get("schema_version")
            or policy.get("policy_version") != reference.get("policy_version")):
        return None, None, None, AuditFinding(
            code=f"{kind}_policy_version_mismatch",
            message=f"frozen {kind} policy schema/version differs from its manifest reference",
            entity_ids=(str(policy.get("schema_version")), str(policy.get("policy_version")),
                        str(reference.get("schema_version")), str(reference.get("policy_version"))),
        )
    if override_path is not None:
        try:
            override = Path(override_path).expanduser()
            if override.is_symlink() or not override.is_file():
                raise OSError("override must be a regular non-symlink policy file")
            override_bytes = override.read_bytes()
        except (OSError, TypeError, ValueError) as exc:
            return None, None, None, AuditFinding(
                code=f"{kind}_policy_override_mismatch",
                message=f"explicit {kind} policy override cannot be verified against the frozen bytes: {exc}",
            )
        if override_bytes != raw:
            return None, None, None, AuditFinding(
                code=f"{kind}_policy_override_mismatch",
                message=f"explicit {kind} policy override bytes differ from the frozen policy snapshot",
                entity_ids=(_sha256(override_bytes), actual_file_sha256),
            )
    return snapshot_path, raw, dict(policy), None


def _load_entity_for_audit(entity_path: Path) -> tuple[dict[str, Any], bytes, list[AuditFinding]]:
    findings: list[AuditFinding] = []
    raw = entity_path.read_bytes()
    parsed = _read_json_bytes(raw, "entities.json")
    if not isinstance(parsed, dict):
        raise AuditInputError("entities.json root must be an object")
    if type(parsed.get("schema_version")) is not int or parsed.get("schema_version") != _BUNDLE_SCHEMA_VERSION:
        findings.append(AuditFinding(
            code="entity_schema_unsupported",
            message="frozen entities.json uses an unsupported schema version",
            entity_ids=(str(parsed.get("schema_version")),),
        ))
    expected_hash = parsed.get("content_sha256")
    if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
        findings.append(AuditFinding(
            code="entity_content_hash_invalid", message="entities.json content_sha256 is missing or malformed",
        ))
    else:
        actual_hash = _canonical_hash(_bundle_hash_payload(parsed))
        if actual_hash != expected_hash:
            findings.append(AuditFinding(
                code="entity_content_hash_mismatch",
                message="entities.json semantic content does not match content_sha256",
                entity_ids=(actual_hash, expected_hash),
            ))
        expected_version = f"entities-v{_BUNDLE_SCHEMA_VERSION}-{expected_hash[:16]}"
        if parsed.get("entity_version") != expected_version:
            findings.append(AuditFinding(
                code="entity_version_mismatch",
                message="entity_version is not derived from the frozen content hash",
                entity_ids=(str(parsed.get("entity_version")), expected_version),
            ))

    provenance = parsed.get("source_provenance")
    if not isinstance(provenance, list):
        findings.append(AuditFinding(
            code="source_provenance_invalid", message="entities.json source_provenance must be an array",
        ))
    else:
        for source in provenance:
            if not isinstance(source, Mapping):
                findings.append(AuditFinding(
                    code="source_provenance_invalid", message="source provenance entry is not an object",
                ))
                continue
            relative = source.get("source_path")
            expected_raw_hash = source.get("raw_sha256")
            try:
                source_path = _safe_relative_file(entity_path.parent, relative, "raw source")
                actual_raw_hash = _sha256(source_path.read_bytes())
            except (AuditInputError, OSError) as exc:
                findings.append(AuditFinding(
                    code="source_snapshot_invalid",
                    message=str(exc),
                    entity_ids=(str(source.get("source_id") or "unknown"),),
                    source_evidence=(_json_safe(source),),
                ))
                continue
            if not isinstance(expected_raw_hash, str) or actual_raw_hash != expected_raw_hash:
                evidence = dict(source)
                evidence.update(expected_sha256=expected_raw_hash, actual_sha256=actual_raw_hash)
                findings.append(AuditFinding(
                    code="source_hash_mismatch",
                    message="raw source or linked metadata bytes differ from the frozen receipt",
                    entity_ids=(str(source.get("source_id") or "unknown"),),
                    source_evidence=(evidence,),
                ))

    try:
        loaded = load_entity_bundle(entity_path, verify_sources=True)
    except (EvidenceInputError, TypeError, ValueError, OSError) as exc:
        if _dependency_error(exc):
            raise _AuditStop("dependency_missing", str(exc)) from exc
        message = str(exc)
        if "raw source hash mismatch" in message or "response hash differs" in message:
            code = "source_hash_mismatch"
        elif "unsupported schema_version" in message:
            code = "entity_schema_unsupported"
        elif "content hash" in message:
            code = "entity_content_hash_mismatch"
        elif "entity_version" in message:
            code = "entity_version_mismatch"
        elif "source" in message or "provenance" in message:
            code = "source_provenance_invalid"
        else:
            code = "entity_bundle_invalid"
        if not any(item.code == code for item in findings):
            findings.append(AuditFinding(code=code, message=f"producer verification rejected frozen bundle: {message}"))
        loaded = parsed
    return loaded, raw, findings


def _verified_molecules(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    molecules = bundle.get("molecules")
    if not isinstance(molecules, list):
        raise AuditInputError("verified bundle molecules must be an array")
    result: list[dict[str, Any]] = []
    for row in molecules:
        if not isinstance(row, Mapping) or row.get("structure_status") != "verified":
            continue
        identifier = row.get("molecule_chembl_id")
        smiles = row.get("canonical_smiles")
        if not isinstance(identifier, str) or not _CHEMBL_ID.fullmatch(identifier):
            continue
        if not isinstance(smiles, str) or not smiles:
            continue
        result.append({
            "entity_id": identifier,
            "canonical_smiles": smiles,
            "structure_sha256": row.get("structure_sha256"),
            "source_provenance": tuple(row.get("source_provenance", ())),
        })
    return sorted(result, key=lambda item: item["entity_id"])


def _verified_proteins(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    proteins = bundle.get("proteins")
    if not isinstance(proteins, list):
        raise AuditInputError("verified bundle proteins must be an array")
    result: list[dict[str, Any]] = []
    for row in proteins:
        if not isinstance(row, Mapping):
            continue
        try:
            record = ProteinRecord(**dict(row))
            validate_protein_record(record)
        except (TypeError, ValueError):
            continue
        if record.identity_status != "verified" or record.sequence_status != "verified":
            continue
        if record.accession is None or record.taxonomy_id is None or record.full_sequence is None:
            continue
        entity_id = f"{record.taxonomy_id}|{record.accession}"
        result.append({
            "entity_id": entity_id,
            "taxonomy_id": record.taxonomy_id,
            "accession": record.accession,
            "full_sequence": record.full_sequence,
            "sequence_sha256": record.sequence_sha256,
            "source_provenance": tuple(record.source_provenance),
        })
    return sorted(result, key=lambda item: (item["taxonomy_id"], item["accession"]))


def _pair_count(size: int) -> int:
    return size * (size - 1) // 2


def _molecule_hits(entities: Sequence[Mapping[str, Any]], policy: Mapping[str, Any], limit: int) -> tuple[NeighborHit, ...]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise _AuditStop("dependency_missing", f"RDKit is required for independent molecule audit: {exc}") from exc
    if _pair_count(len(entities)) > limit:
        raise _AuditStop("size_limit", f"molecule pair budget exceeded: {_pair_count(len(entities))} > {limit}")
    fingerprint_policy = policy["molecule"]["fingerprint"]
    threshold = float(fingerprint_policy["threshold"])
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fingerprint_policy["radius"]), fpSize=int(fingerprint_policy["bits"]),
    )
    prepared: list[tuple[Mapping[str, Any], Any, str, Any]] = []
    for entity in entities:
        mol = Chem.MolFromSmiles(entity["canonical_smiles"])
        if mol is None:
            raise AuditInputError(f"RDKit could not parse frozen molecule {entity['entity_id']}")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if canonical != entity["canonical_smiles"]:
            raise AuditInputError(f"frozen molecule structure is not canonical: {entity['entity_id']}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        fingerprint = generator.GetFingerprint(mol)
        prepared.append((entity, mol, scaffold, fingerprint))

    hits: list[NeighborHit] = []
    for index, (left, _left_mol, left_scaffold, left_fp) in enumerate(prepared):
        for right, _right_mol, right_scaffold, right_fp in prepared[index + 1:]:
            left_id = str(left["entity_id"])
            right_id = str(right["entity_id"])
            ids = (left_id, right_id)
            evidence = tuple(left.get("source_provenance", ())) + tuple(right.get("source_provenance", ()))
            if left_id == right_id:
                hits.append(NeighborHit(
                    "molecule_identity", ids, 1.0, 1.0,
                    {"rule": "same ChEMBL identity appears more than once"}, evidence,
                ))
            if left_scaffold and right_scaffold and left_scaffold == right_scaffold:
                hits.append(NeighborHit(
                    "molecule_scaffold", ids, 1.0, 1.0,
                    {"scaffold": left_scaffold, "rule": "same nonempty Murcko scaffold"}, evidence,
                ))
            similarity = float(DataStructs.TanimotoSimilarity(left_fp, right_fp))
            if not math.isfinite(similarity):
                raise AuditInputError(f"non-finite Tanimoto score for {left_id} and {right_id}")
            if similarity >= threshold:
                hits.append(NeighborHit(
                    "molecule_morgan", ids, similarity, threshold,
                    {"radius": int(fingerprint_policy["radius"]), "bits": int(fingerprint_policy["bits"]),
                     "operator": ">="}, evidence,
                ))
    return tuple(hits)


def _aligned_metrics(sequence_a: str, sequence_b: str, alignment: Any) -> tuple[int, int]:
    coordinates = alignment.coordinates
    matches = 0
    aligned_non_gap = 0
    for index in range(coordinates.shape[1] - 1):
        start_a = int(coordinates[0, index])
        end_a = int(coordinates[0, index + 1])
        start_b = int(coordinates[1, index])
        end_b = int(coordinates[1, index + 1])
        delta_a = end_a - start_a
        delta_b = end_b - start_b
        if delta_a <= 0 or delta_b <= 0:
            continue
        if delta_a != delta_b:
            raise AuditInputError("global alignment produced a non-diagonal non-gap segment")
        aligned_non_gap += delta_a
        for offset in range(delta_a):
            if sequence_a[start_a + offset] == sequence_b[start_b + offset]:
                matches += 1
    return matches, aligned_non_gap


def _protein_hits(entities: Sequence[Mapping[str, Any]], policy: Mapping[str, Any], limits: Mapping[str, int]) -> tuple[NeighborHit, ...]:
    try:
        from Bio import Align
        from Bio.Align import substitution_matrices
    except ImportError as exc:
        raise _AuditStop("dependency_missing", f"Bio.Align is required for independent protein audit: {exc}") from exc
    pair_total = _pair_count(len(entities))
    if pair_total > limits["protein_pairs"]:
        raise _AuditStop("size_limit", f"protein pair budget exceeded: {pair_total} > {limits['protein_pairs']}")
    alignment_policy = policy["protein"]["alignment"]
    identity_policy = policy["protein"]["identity"]
    coverage_policy = policy["protein"]["coverage"]
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.epsilon = 0.0

    hits: list[NeighborHit] = []
    alignment_cells = 0
    for index, left in enumerate(entities):
        for right in entities[index + 1:]:
            sequence_a = str(left["full_sequence"])
            sequence_b = str(right["full_sequence"])
            cells = len(sequence_a) * len(sequence_b)
            if alignment_cells + cells > limits["alignment_cells"]:
                raise _AuditStop(
                    "size_limit",
                    f"protein alignment cell budget exceeded: {alignment_cells + cells} > {limits['alignment_cells']}",
                )
            alignment_cells += cells
            alignment = aligner.align(sequence_a, sequence_b)[0]
            matches, non_gap = _aligned_metrics(sequence_a, sequence_b, alignment)
            if non_gap == 0:
                continue
            identity = matches / non_gap
            coverage_a = non_gap / len(sequence_a)
            coverage_b = non_gap / len(sequence_b)
            identity_threshold = float(identity_policy["threshold"])
            coverage_threshold = float(coverage_policy["threshold"])
            if (identity >= identity_threshold
                    and coverage_a >= coverage_threshold
                    and coverage_b >= coverage_threshold):
                ids = (str(left["entity_id"]), str(right["entity_id"]))
                evidence = tuple(left.get("source_provenance", ())) + tuple(right.get("source_provenance", ()))
                hits.append(NeighborHit(
                    "protein_global_identity_coverage", ids, identity, identity_threshold,
                    {
                        "matches": matches,
                        "aligned_non_gap_columns": non_gap,
                        "coverage_a": coverage_a,
                        "coverage_b": coverage_b,
                        "coverage_threshold": coverage_threshold,
                        "matrix": "BLOSUM62",
                        "open_gap_score": -10.0,
                        "extend_gap_score": -0.5,
                        "epsilon": 0.0,
                        "tie_policy": alignment_policy.get("tie_policy"),
                        "sequence_sha256_a": left.get("sequence_sha256"),
                        "sequence_sha256_b": right.get("sequence_sha256"),
                    },
                    evidence,
                ))
    return tuple(hits)


def _recompute_neighbors(bundle: Mapping[str, Any], policy: Mapping[str, Any], limits: Mapping[str, int]) -> NeighborRecomputeResult:
    try:
        molecules = _verified_molecules(bundle)
        proteins = _verified_proteins(bundle)
        molecule_neighbors = _molecule_hits(molecules, policy, limits["molecule_pairs"])
        protein_neighbors = _protein_hits(proteins, policy, limits)
        return NeighborRecomputeResult("ok", molecule_neighbors, protein_neighbors)
    except _AuditStop as exc:
        return NeighborRecomputeResult(exc.status, error=str(exc))
    except (AuditInputError, EvidenceInputError, TypeError, ValueError, OSError) as exc:
        if _dependency_error(exc):
            return NeighborRecomputeResult("dependency_missing", error=str(exc))
        return NeighborRecomputeResult("invalid_input", error=str(exc))


def independently_recompute_neighbors(
    entity_bundle_path: os.PathLike[str] | str,
    *,
    policy_path: Any = None,
    budget_overrides: Optional[Mapping[str, Any]] = None,
) -> NeighborRecomputeResult:
    """Recompute all molecule/protein neighbor edges from one verified bundle."""
    try:
        policy, _policy_sha256, limits = _load_policy(policy_path, budget_overrides)
        bundle = load_entity_bundle(entity_bundle_path, verify_sources=True)
        return _recompute_neighbors(bundle, policy, limits)
    except _AuditStop as exc:
        return NeighborRecomputeResult(exc.status, error=str(exc))
    except (AuditInputError, EvidenceInputError, TypeError, ValueError, OSError) as exc:
        if _dependency_error(exc):
            return NeighborRecomputeResult("dependency_missing", error=str(exc))
        return NeighborRecomputeResult("invalid_input", error=str(exc))


def _component_memberships(
    entity_ids: Iterable[str], hits: Iterable[NeighborHit], *, prefix: str,
) -> dict[str, str]:
    parent = {identifier: identifier for identifier in entity_ids}

    def find(identifier: str) -> str:
        root = identifier
        while parent[root] != root:
            root = parent[root]
        while parent[identifier] != identifier:
            following = parent[identifier]
            parent[identifier] = root
            identifier = following
        return root

    for hit in hits:
        left, right = hit.entity_ids
        if left not in parent or right not in parent:
            continue
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            lexical_root = min(root_left, root_right)
            parent[root_left] = lexical_root
            parent[root_right] = lexical_root
    groups: dict[str, list[str]] = defaultdict(list)
    for identifier in sorted(parent):
        groups[find(identifier)].append(identifier)
    result: dict[str, str] = {}
    for members in groups.values():
        component_id = f"{prefix}:{min(members)}"
        result.update({identifier: component_id for identifier in members})
    return result


def _source_evidence_for_group(group: Any, bundle: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    evidence_ids = set(getattr(group, "evidence_ids", ()))
    rows: list[Mapping[str, Any]] = []
    for activity in bundle.get("activities", []):
        if isinstance(activity, Mapping) and str(activity.get("activity_id")) in evidence_ids:
            rows.append({
                "activity_id": activity.get("activity_id"),
                "source_path": activity.get("source_path"),
                "source_sha256": activity.get("source_sha256"),
                "target_record_source_path": activity.get("target_record_source_path"),
                "target_record_source_sha256": activity.get("target_record_source_sha256"),
                "assay_record_source_path": activity.get("assay_record_source_path"),
                "assay_record_source_sha256": activity.get("assay_record_source_sha256"),
                "document_record_source_path": activity.get("document_record_source_path"),
                "document_record_source_sha256": activity.get("document_record_source_sha256"),
            })
    return tuple(rows)


def _finding_from_hit(hit: NeighborHit, split_a: str, split_b: str) -> AuditFinding:
    code = "molecule_neighbor_cross_split" if hit.kind.startswith("molecule_") else "protein_neighbor_cross_split"
    return AuditFinding(
        code=code,
        message=f"independently recomputed {hit.kind} connects entities in different splits ({split_a}, {split_b})",
        entity_ids=hit.entity_ids,
        split=f"{split_a}|{split_b}",
        metric=hit.metric,
        threshold=hit.threshold,
        source_evidence=hit.source_evidence,
    )


def _empty_support() -> dict[str, dict[str, int]]:
    return {split: {label: 0 for label in _CLASSES} for split in _SPLITS}


def _support_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("split")
        partition = row.get("provenance_partition")
        labels = set(row.get("labels", ())) if isinstance(row.get("labels"), (list, tuple)) else set()
        group_id = row.get("split_group_id")
        pair_key = row.get("pair_key")
        if split not in _SPLITS or partition != "confirmed" or not isinstance(group_id, str) or not isinstance(pair_key, str):
            continue
        if len(labels) != 1 or not labels.issubset(set(_CLASSES)):
            continue
        groups[(split, next(iter(labels)))].add(group_id)
        pair_by_group[group_id].add(pair_key)
    support = _empty_support()
    invalid_group_ids = {group_id for group_id, pairs in pair_by_group.items() if len(pairs) != 1}
    for (split, label), group_ids in groups.items():
        support[split][label] = len(group_ids - invalid_group_ids)
    return support


def _label_maps(label_result: Any, groups: Sequence[Any]) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    group_by_pair = {group.pair_key: group for group in groups}
    if len(group_by_pair) != len(groups):
        raise AuditInputError("source evidence produced duplicate canonical pair groups")
    confirmed: dict[str, set[str]] = defaultdict(set)
    test_only: dict[str, set[str]] = defaultdict(set)
    diagnostics: dict[str, set[str]] = defaultdict(set)
    for field_name, destination in (
        ("confirmed_labeled_pairs", confirmed),
        ("test_only_labeled_pairs", test_only),
    ):
        for decision in getattr(label_result, field_name, ()):
            group = group_by_pair.get(decision.pair_key)
            if (group is None or decision.split_group_id != group.split_group_id
                    or not set(decision.evidence_ids).issubset(set(group.evidence_ids))
                    or decision.label not in set(_CLASSES)):
                raise AuditInputError(f"label policy emitted detached or unsupported decision: {decision.pair_key}")
            destination[decision.pair_key].add(decision.label)
    for field_name in ("gray", "unknown", "conflicts", "excluded", "unverified"):
        for decision in getattr(label_result, field_name, ()):
            if decision.pair_key in group_by_pair:
                diagnostics[decision.pair_key].update(
                    f"{field_name}:{reason}" for reason in decision.reasons
                )
    return dict(confirmed), dict(test_only), dict(diagnostics)


def _pair_row(
    group: Any,
    *,
    split: Optional[str],
    labels: Iterable[str],
    provenance_partition: str,
    reasons: Iterable[str],
    molecule_cluster_id: Optional[str],
    protein_cluster_id: Optional[str],
) -> dict[str, Any]:
    return {
        "split_group_id": group.split_group_id,
        "pair_key": group.pair_key,
        "molecule_cluster_id": molecule_cluster_id,
        "protein_cluster_id": protein_cluster_id,
        "split": split,
        "labels": sorted(set(labels)),
        "provenance_partition": provenance_partition,
        "reasons": sorted(set(reasons)),
        "evidence_ids": sorted(group.evidence_ids),
        "group_sha256": group.group_sha256,
        "evidence_records_json": list(group.records_json),
    }


def _row_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("split_group_id"), row.get("pair_key"), row.get("split"),
        tuple(sorted(row.get("labels", ()))) if isinstance(row.get("labels"), (list, tuple)) else (),
        row.get("provenance_partition"), tuple(sorted(row.get("reasons", ()))) if isinstance(row.get("reasons"), (list, tuple)) else (),
        tuple(sorted(row.get("evidence_ids", ()))) if isinstance(row.get("evidence_ids"), (list, tuple)) else (),
        row.get("group_sha256"), tuple(row.get("evidence_records_json", ())) if isinstance(row.get("evidence_records_json"), (list, tuple)) else (),
        row.get("molecule_cluster_id"), row.get("protein_cluster_id"),
    )


def _normalize_memberships(
    value: Any,
    *,
    kind: str,
    known_entities: set[str],
    verified_entities: set[str],
    findings: list[AuditFinding],
    evidence_by_entity: Mapping[str, tuple[Mapping[str, Any], ...]],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        findings.append(AuditFinding(
            code="membership_map_invalid", message=f"{kind} memberships must be an object",
        ))
        return {}
    result: dict[str, str] = {}
    for raw_id, partition in value.items():
        entity_id = str(raw_id)
        if entity_id not in known_entities:
            findings.append(AuditFinding(
                code="unknown_entity_mapping", message=f"{kind} membership references an entity absent from frozen entities.json",
                entity_ids=(entity_id,), split=str(partition),
            ))
            continue
        if entity_id not in verified_entities:
            findings.append(AuditFinding(
                code="entity_not_clusterable", message=f"{kind} membership assigns an entity without verified source identity/structure/sequence",
                entity_ids=(entity_id,), split=str(partition),
                source_evidence=evidence_by_entity.get(entity_id, ()),
            ))
            continue
        if partition not in _SPLITS:
            findings.append(AuditFinding(
                code="membership_split_invalid", message=f"{kind} membership uses an unsupported split name",
                entity_ids=(entity_id,), split=str(partition),
                source_evidence=evidence_by_entity.get(entity_id, ()),
            ))
            continue
        result[entity_id] = str(partition)
    for entity_id in sorted(verified_entities - set(result)):
        findings.append(AuditFinding(
            code="missing_entity_assignment", message=f"verified {kind} entity is absent from split membership map",
            entity_ids=(entity_id,), source_evidence=evidence_by_entity.get(entity_id, ()),
        ))
    return result


def _source_rows_by_pair(bundle: Mapping[str, Any], groups: Sequence[Any]) -> dict[str, tuple[Mapping[str, Any], ...]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    wanted = {group.pair_key: set(group.evidence_ids) for group in groups}
    for activity in bundle.get("activities", []):
        if not isinstance(activity, Mapping):
            continue
        activity_id = str(activity.get("activity_id"))
        for pair_key, evidence_ids in wanted.items():
            if activity_id in evidence_ids:
                result[pair_key].append({
                    "activity_id": activity.get("activity_id"),
                    "source_path": activity.get("source_path"),
                    "source_sha256": activity.get("source_sha256"),
                    "target_record_source_path": activity.get("target_record_source_path"),
                    "target_record_source_sha256": activity.get("target_record_source_sha256"),
                    "assay_record_source_path": activity.get("assay_record_source_path"),
                    "assay_record_source_sha256": activity.get("assay_record_source_sha256"),
                    "document_record_source_path": activity.get("document_record_source_path"),
                    "document_record_source_sha256": activity.get("document_record_source_sha256"),
                })
    return {key: tuple(values) for key, values in result.items()}


def _manifest_failure(message: str, code: str = "manifest_invalid") -> AuditReport:
    finding = AuditFinding(code=code, message=message)
    return AuditReport("invalid_input", 2, (finding,), _empty_support(), False, error=message)


def _receipt_bindings_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    entities = manifest.get("entities")
    provenance = manifest.get("source_provenance")
    policies = manifest.get("policies")
    activities = manifest.get("activities")
    pairs = manifest.get("pairs")
    split = manifest.get("split")
    if not isinstance(entities, Mapping) or not isinstance(provenance, (list, tuple)):
        raise AuditInputError("source readiness receipt requires entity and raw provenance bindings")
    if not isinstance(policies, Mapping) or not isinstance(activities, Mapping):
        raise AuditInputError("source readiness receipt requires frozen policies and activity evidence")
    if not isinstance(pairs, Mapping) or not isinstance(split, Mapping):
        raise AuditInputError("source readiness receipt requires pair and split claims")
    entity_fields = ("path", "schema_version", "entity_version", "content_sha256", "file_sha256")
    policy_fields = ("path", "file_sha256", "semantic_sha256", "schema_version", "policy_version")
    if any(field not in entities for field in entity_fields):
        raise AuditInputError("source readiness receipt entity reference is incomplete")
    if any(not isinstance(policies.get(kind), Mapping) or any(field not in policies[kind] for field in policy_fields)
           for kind in ("split", "label")):
        raise AuditInputError("source readiness receipt policy references are incomplete")
    if not isinstance(activities.get("semantic_sha256"), str):
        raise AuditInputError("source readiness receipt activity semantic hash is missing")
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


def _receipt_violation(code: str, message: str) -> AuditReport:
    finding = AuditFinding(code=code, message=message)
    return AuditReport("violation", 3, (finding,), _empty_support(), False, error=message)


def _not_auditable(
    status: str,
    message: str,
    *,
    findings: Sequence[AuditFinding] = (),
    entity_version: Optional[str] = None,
    entity_bundle_hash: Optional[str] = None,
    split_policy_sha256: Optional[str] = None,
    source_provenance: tuple[Mapping[str, Any], ...] = (),
    split_policy_file_sha256: Optional[str] = None,
    label_policy_sha256: Optional[str] = None,
    label_policy_semantic_sha256: Optional[str] = None,
) -> AuditReport:
    return AuditReport(
        status, 4, tuple(findings), _empty_support(), False,
        entity_version=entity_version,
        entity_bundle_hash=entity_bundle_hash,
        split_policy_sha256=split_policy_sha256,
        source_provenance=source_provenance,
        error=message,
        split_policy_file_sha256=split_policy_file_sha256,
        label_policy_sha256=label_policy_sha256,
        label_policy_semantic_sha256=label_policy_semantic_sha256,
    )


def _integrity_code_for_bundle_error(message: str) -> str:
    text = message.casefold()
    if "unsupported schema_version" in text:
        return "entity_schema_unsupported"
    if "raw source hash mismatch" in text or "response hash differs" in text:
        return "source_hash_mismatch"
    if "content hash" in text:
        return "entity_content_hash_mismatch"
    if "entity_version" in text:
        return "entity_version_mismatch"
    if "source" in text or "provenance" in text:
        return "source_provenance_invalid"
    return "entity_bundle_invalid"


def _audit_split_core(
    manifest_path: os.PathLike[str] | str,
    *,
    policy_path: Any = None,
    label_policy_path: Any = None,
    budget_overrides: Optional[Mapping[str, Any]] = None,
    _allow_unreceipted_candidate: bool = False,
) -> AuditReport:
    """Core implementation; the candidate flag is reachable only through the writer-private wrapper."""
    findings: list[AuditFinding] = []
    policy_report_context: dict[str, Any] = {}
    try:
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file() or manifest_file.is_symlink():
            return _manifest_failure("manifest path must be a regular, non-symlink file")
        manifest_root = manifest_file.resolve(strict=True).parent
        manifest = _read_json_bytes(manifest_file.read_bytes(), "split manifest")
        if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 2:
            return _manifest_failure("split manifest requires supported integer schema_version 2 with frozen policy bindings")

        receipt_sidecar = manifest_root / "source_readiness_receipt.json"
        sidecar_signal = receipt_sidecar.exists() or receipt_sidecar.is_symlink()
        contract_signal = "source_readiness_contract_version" in manifest
        reference_signal = "source_readiness_receipt_ref" in manifest
        receipt_signal = contract_signal or reference_signal or sidecar_signal
        source_receipt: Optional[Mapping[str, Any]] = None
        if not receipt_signal and not _allow_unreceipted_candidate:
            message = "schema-2 manifest has no source-readiness receipt and is legacy_unreceipted, read-only, and not ready"
            finding = AuditFinding(code="legacy_unreceipted", message=message)
            return AuditReport("legacy_unreceipted", 4, (finding,), _empty_support(), False, error=message)
        if receipt_signal:
            contract_version = manifest.get("source_readiness_contract_version")
            if not contract_signal:
                return _receipt_violation(
                    "source_readiness_contract_marker_missing",
                    "source readiness receipt signals require source_readiness_contract_version=1",
                )
            if type(contract_version) is not int or contract_version != 1:
                return _receipt_violation(
                    "source_readiness_contract_unsupported",
                    "source_readiness_contract_version must be integer 1",
                )
            receipt_ref = manifest.get("source_readiness_receipt_ref")
            if not isinstance(receipt_ref, Mapping):
                return _receipt_violation(
                    "source_readiness_receipt_reference_missing",
                    "source readiness contract requires a receipt reference",
                )
            if not sidecar_signal:
                return _receipt_violation(
                    "source_readiness_receipt_missing",
                    "source readiness contract receipt sidecar is missing",
                )
            if receipt_ref.get("path") != "source_readiness_receipt.json":
                return _receipt_violation(
                    "source_readiness_receipt_path_invalid",
                    "source readiness receipt reference must point to source_readiness_receipt.json beside the manifest",
                )
            try:
                receipt_path = _safe_relative_file(manifest_root, receipt_ref.get("path"), "source readiness receipt")
                receipt_bytes = receipt_path.read_bytes()
                source_receipt_value = _read_json_bytes(receipt_bytes, "source readiness receipt")
            except (AuditInputError, OSError, TypeError, ValueError) as exc:
                return _receipt_violation("source_readiness_receipt_invalid", str(exc))
            if not isinstance(source_receipt_value, Mapping):
                return _receipt_violation("source_readiness_receipt_invalid", "source readiness receipt must be an object")
            if source_receipt_value.get("receipt_schema_version") != 1 or type(source_receipt_value.get("receipt_schema_version")) is not int:
                return _receipt_violation(
                    "source_readiness_receipt_unsupported",
                    f"unsupported source readiness receipt schema version: {source_receipt_value.get('receipt_schema_version')}",
                )
            required_receipt_fields = {
                "receipt_schema_version", "scope_kind", "receipt_sha256", "scope_sha256", "bindings", "metrics",
            }
            if set(source_receipt_value) != required_receipt_fields:
                return _receipt_violation(
                    "source_readiness_receipt_invalid",
                    "source readiness receipt fields do not match receipt_schema_version=1",
                )
            if source_receipt_value.get("scope_kind") != "canonical_frozen_corpus":
                return _receipt_violation(
                    "source_readiness_receipt_scope_invalid",
                    "source readiness receipt scope_kind must be canonical_frozen_corpus",
                )
            actual_receipt_file_hash = _sha256(receipt_bytes)
            if receipt_ref.get("file_sha256") != actual_receipt_file_hash:
                return _receipt_violation(
                    "source_readiness_receipt_file_hash_mismatch",
                    "receipt sidecar bytes differ from the file hash in the manifest reference",
                )
            actual_receipt_semantic_hash = canonical_json_sha256(source_receipt_value)
            if receipt_ref.get("semantic_sha256") != actual_receipt_semantic_hash:
                return _receipt_violation(
                    "source_readiness_receipt_semantic_hash_mismatch",
                    "receipt payload differs from the semantic hash in the manifest reference",
                )
            receipt_body = {key: value for key, value in source_receipt_value.items() if key != "receipt_sha256"}
            if source_receipt_value.get("receipt_sha256") != canonical_json_sha256(receipt_body):
                return _receipt_violation(
                    "source_readiness_receipt_hash_mismatch",
                    "receipt_sha256 does not match the canonical receipt payload",
                )
            source_receipt = source_receipt_value

        entity_ref = manifest.get("entities")
        if not isinstance(entity_ref, Mapping):
            return _manifest_failure("split manifest entities reference is missing")
        required_entity_fields = ("path", "schema_version", "entity_version", "content_sha256", "file_sha256")
        if any(field_name not in entity_ref for field_name in required_entity_fields):
            return _manifest_failure("split manifest entities reference is incomplete")
        try:
            entity_path = _safe_relative_file(manifest_root, entity_ref.get("path"), "entities")
        except (AuditInputError, OSError) as exc:
            findings.append(AuditFinding(
                code="entity_path_invalid", message=str(exc),
                entity_ids=(str(entity_ref.get("path")),),
            ))
            entity_path = None
        if entity_path is None:
            return AuditReport("violation", 3, tuple(findings), _empty_support(), False, error="unsafe or invalid entity path")

        raw_entity_bytes = entity_path.read_bytes()
        actual_file_hash = _sha256(raw_entity_bytes)
        if actual_file_hash != entity_ref.get("file_sha256"):
            findings.append(AuditFinding(
                code="entity_file_hash_mismatch",
                message="frozen entities.json bytes differ from the file hash recorded by the manifest",
                entity_ids=(str(entity_ref.get("file_sha256")), actual_file_hash),
                source_evidence=({"entity_path": entity_ref.get("path"), "expected_sha256": entity_ref.get("file_sha256"),
                                  "actual_sha256": actual_file_hash},),
            ))
        try:
            bundle, _same_bytes, bundle_findings = _load_entity_for_audit(entity_path)
            findings.extend(bundle_findings)
        except _AuditStop as exc:
            return _not_auditable(exc.status, str(exc), findings=findings)
        except (AuditInputError, OSError) as exc:
            findings.append(AuditFinding(
                code="entity_bundle_invalid", message=str(exc),
                entity_ids=(str(entity_ref.get("path")),),
            ))
            return AuditReport("violation", 3, tuple(findings), _empty_support(), False, error=str(exc))

        if entity_ref.get("schema_version") != bundle.get("schema_version"):
            findings.append(AuditFinding(
                code="entity_schema_reference_mismatch", message="manifest schema_version does not match frozen entities.json",
                entity_ids=(str(entity_ref.get("schema_version")), str(bundle.get("schema_version"))),
            ))
        if entity_ref.get("entity_version") != bundle.get("entity_version"):
            findings.append(AuditFinding(
                code="entity_version_reference_mismatch", message="manifest entity_version does not match frozen entities.json",
                entity_ids=(str(entity_ref.get("entity_version")), str(bundle.get("entity_version"))),
            ))
        if entity_ref.get("content_sha256") != bundle.get("content_sha256"):
            findings.append(AuditFinding(
                code="entity_content_reference_mismatch", message="manifest content_sha256 does not match frozen entities.json",
                entity_ids=(str(entity_ref.get("content_sha256")), str(bundle.get("content_sha256"))),
            ))
        if _canonical_hash(manifest.get("source_provenance")) != _canonical_hash(bundle.get("source_provenance")):
            findings.append(AuditFinding(
                code="manifest_source_provenance_mismatch",
                message="manifest source provenance differs from the same frozen entities.json receipts",
                source_evidence=tuple(_json_safe(item) for item in bundle.get("source_provenance", []) if isinstance(item, Mapping)),
            ))

        split = manifest.get("split")
        if not isinstance(split, Mapping):
            return _manifest_failure("split manifest split object is missing")
        policy_bindings = manifest.get("policies")
        split_policy_path, split_policy_bytes, split_policy_value, split_policy_error = _load_policy_snapshot(
            manifest_root, policy_bindings, "split", policy_path,
        )
        label_policy_path_frozen, label_policy_bytes, label_policy_value, label_policy_error = _load_policy_snapshot(
            manifest_root, policy_bindings, "label", label_policy_path,
        )
        policy_findings = [row for row in (split_policy_error, label_policy_error) if row is not None]
        if policy_findings:
            return AuditReport(
                "violation", 3, tuple(policy_findings), _empty_support(), False,
                entity_version=str(bundle.get("entity_version")) if bundle.get("entity_version") is not None else None,
                entity_bundle_hash=str(bundle.get("content_sha256")) if bundle.get("content_sha256") is not None else None,
                error="frozen policy snapshot or explicit override does not match its manifest binding",
            )
        if (split_policy_path is None or split_policy_value is None
                or label_policy_path_frozen is None or label_policy_value is None):
            return _manifest_failure("policy snapshot loader did not produce both frozen policies", "policy_binding_invalid")
        try:
            policy, policy_sha256, limits = _load_policy(split_policy_path)
        except AuditInputError as exc:
            finding = AuditFinding(
                code="split_policy_snapshot_unsupported",
                message=f"frozen split policy does not satisfy the independently supported contract: {exc}",
                entity_ids=(str(manifest.get("policies", {}).get("split", {}).get("path")),),
            )
            return AuditReport(
                "violation", 3, (finding,), _empty_support(), False,
                entity_version=str(bundle.get("entity_version")) if bundle.get("entity_version") is not None else None,
                entity_bundle_hash=str(bundle.get("content_sha256")) if bundle.get("content_sha256") is not None else None,
                error=str(exc),
            )
        if budget_overrides is not None:
            if not isinstance(budget_overrides, Mapping):
                return _manifest_failure("budget_overrides must be an object", "budget_override_invalid")
            try:
                for key, value in budget_overrides.items():
                    if key not in limits:
                        raise AuditInputError(f"unsupported audit budget: {key}")
                    limits[key] = _strict_nonnegative_int(value, f"budget override {key}")
            except AuditInputError as exc:
                return _manifest_failure(str(exc), "budget_override_invalid")
        label_policy_semantic_sha256 = _canonical_hash(label_policy_value)
        split_reference = policy_bindings["split"]
        label_reference = policy_bindings["label"]
        policy_report_context = {
            "entity_version": str(bundle.get("entity_version")) if bundle.get("entity_version") is not None else None,
            "entity_bundle_hash": str(bundle.get("content_sha256")) if bundle.get("content_sha256") is not None else None,
            "split_policy_sha256": policy_sha256,
            "source_provenance": tuple(bundle.get("source_provenance", ())),
            "split_policy_file_sha256": split_reference.get("file_sha256"),
            "label_policy_sha256": label_reference.get("file_sha256"),
            "label_policy_semantic_sha256": label_reference.get("semantic_sha256"),
        }
        audit_marker = manifest.get("audit")
        audit_is_final = (
            isinstance(audit_marker, Mapping)
            and audit_marker.get("status") == "clean"
            and audit_marker.get("exit_code") == 0
            and audit_marker.get("finding_count") == 0
        )
        if "audit" in manifest and not audit_is_final:
            findings.append(AuditFinding(
                code="manifest_audit_marker_invalid",
                message="a committed audit marker must record clean status, exit code 0, and zero findings",
            ))
        activity_binding = manifest.get("activities")
        pair_binding = manifest.get("pairs")
        if not isinstance(activity_binding, Mapping) or not isinstance(pair_binding, Mapping):
            return _manifest_failure("manifest must bind activity and pair ledgers to the frozen entity version")
        required_split_fields = (
            "split_policy_sha256", "split_policy_file_sha256", "split_policy_semantic_sha256",
            "split_policy_schema_version", "split_policy_version", "label_policy_sha256",
            "label_policy_file_sha256", "label_policy_semantic_sha256", "label_policy_schema_version",
            "label_policy_version", "entity_version", "entity_bundle_hash", "molecule_memberships",
            "protein_memberships", "pair_assignments",
            "crosspair_quarantine", "unassigned_pairs", "test_only_pair_assignments",
            "ambiguous_label_quarantine", "class_support", "training_ready",
        )
        if any(field_name not in split for field_name in required_split_fields):
            return _manifest_failure("split manifest omits one or more assignment/quarantine/support ledgers")
        if split.get("strict_split_status") in {"size_limit", "dependency_missing"}:
            return _not_auditable(
                str(split.get("strict_split_status")),
                str(split.get("error") or "split was not fully built"),
                findings=findings,
                **policy_report_context,
            )
        claimed_policy_hash = split.get("split_policy_sha256")
        if claimed_policy_hash != policy_sha256:
            findings.append(AuditFinding(
                code="split_policy_hash_mismatch", message="split policy hash does not identify the policy used by this independent audit",
                entity_ids=(str(claimed_policy_hash), policy_sha256),
            ))
        if (split.get("split_policy_file_sha256") != split_reference["file_sha256"]
                or split.get("split_policy_semantic_sha256") != split_reference["semantic_sha256"]
                or split.get("split_policy_schema_version") != split_reference["schema_version"]
                or split.get("split_policy_version") != split_reference["policy_version"]):
            findings.append(AuditFinding(
                code="split_policy_claim_mismatch",
                message="split claim policy byte/semantic hash and version do not match the frozen split-policy snapshot",
                entity_ids=(str(split.get("split_policy_file_sha256")), str(split_reference["file_sha256"])),
            ))
        if (split.get("label_policy_sha256") != label_reference["file_sha256"]
                or split.get("label_policy_file_sha256") != label_reference["file_sha256"]
                or split.get("label_policy_semantic_sha256") != label_reference["semantic_sha256"]
                or split.get("label_policy_schema_version") != label_reference["schema_version"]
                or split.get("label_policy_version") != label_reference["policy_version"]):
            findings.append(AuditFinding(
                code="label_policy_claim_mismatch",
                message="split claim label-policy byte/semantic hash and version do not match the frozen label-policy snapshot",
                entity_ids=(str(split.get("label_policy_sha256")), str(label_reference["file_sha256"])),
            ))
        expected_entity_version = bundle.get("entity_version")
        expected_entity_hash = bundle.get("content_sha256")
        if split.get("entity_version") != expected_entity_version or split.get("entity_bundle_hash") != expected_entity_hash:
            findings.append(AuditFinding(
                code="split_entity_binding_mismatch",
                message="split ledger is not bound to the same frozen entity version/content hash",
                entity_ids=(str(split.get("entity_version")), str(split.get("entity_bundle_hash"))),
            ))
        expected_activity_hash = _canonical_hash(bundle.get("activities", []))
        if (activity_binding.get("entity_version") != expected_entity_version
                or activity_binding.get("entity_bundle_hash") != expected_entity_hash
                or activity_binding.get("record_count") != len(bundle.get("activities", []))
                or activity_binding.get("semantic_sha256") != expected_activity_hash):
            findings.append(AuditFinding(
                code="activity_entity_binding_mismatch",
                message="activity record count/hash/version does not bind to the frozen entities.json bytes",
                entity_ids=(str(activity_binding.get("entity_version")), str(activity_binding.get("entity_bundle_hash"))),
            ))
        pair_fields = (
            "pair_assignments", "crosspair_quarantine", "unassigned_pairs",
            "test_only_pair_assignments", "ambiguous_label_quarantine",
        )
        claimed_pair_payload = {field_name: split.get(field_name) for field_name in pair_fields}
        claimed_ledger_counts = {
            field_name: len(split[field_name]) if isinstance(split.get(field_name), list) else -1
            for field_name in pair_fields
        }
        if (pair_binding.get("entity_version") != expected_entity_version
                or pair_binding.get("entity_bundle_hash") != expected_entity_hash
                or pair_binding.get("semantic_sha256") != _canonical_hash(claimed_pair_payload)
                or pair_binding.get("ledger_counts") != claimed_ledger_counts):
            findings.append(AuditFinding(
                code="pair_entity_binding_mismatch",
                message="pair/quarantine ledger counts or hash are not bound to the same frozen entity version",
                entity_ids=(str(pair_binding.get("entity_version")), str(pair_binding.get("entity_bundle_hash"))),
            ))

        neighbor_result = _recompute_neighbors(bundle, policy, limits)
        if neighbor_result.status in {"size_limit", "dependency_missing"}:
            return _not_auditable(
                neighbor_result.status,
                neighbor_result.error or neighbor_result.status,
                findings=findings,
                **policy_report_context,
            )
        if neighbor_result.status != "ok":
            findings.append(AuditFinding(
                code="neighbor_recompute_invalid", message=neighbor_result.error or "neighbor recomputation failed",
            ))
            return AuditReport("violation", 3, tuple(findings), _empty_support(), False,
                               str(bundle.get("entity_version")), str(bundle.get("content_sha256")), policy_sha256,
                               tuple(bundle.get("source_provenance", ())), neighbor_result.error,
                               split_policy_file_sha256=split_reference.get("file_sha256"),
                               label_policy_sha256=label_reference.get("file_sha256"),
                               label_policy_semantic_sha256=label_reference.get("semantic_sha256"))

        molecules = _verified_molecules(bundle)
        proteins = _verified_proteins(bundle)
        molecule_occurrences = Counter(row["entity_id"] for row in molecules)
        protein_occurrences = Counter(row["entity_id"] for row in proteins)
        for identifier, count in sorted(molecule_occurrences.items()):
            if count > 1:
                findings.append(AuditFinding(
                    code="duplicate_entity_identity",
                    message="the same canonical molecule identity appears more than once in the frozen entity bundle",
                    entity_ids=(identifier, str(count)),
                    source_evidence=tuple(
                        source for row in molecules if row["entity_id"] == identifier
                        for source in row.get("source_provenance", ())
                    ),
                ))
        for identifier, count in sorted(protein_occurrences.items()):
            if count > 1:
                findings.append(AuditFinding(
                    code="duplicate_entity_identity",
                    message="the same taxonomy/accession identity appears more than once in the frozen entity bundle",
                    entity_ids=(identifier, str(count)),
                    source_evidence=tuple(
                        source for row in proteins if row["entity_id"] == identifier
                        for source in row.get("source_provenance", ())
                    ),
                ))
        known_molecules = {
            str(row.get("molecule_chembl_id")) for row in bundle.get("molecules", [])
            if isinstance(row, Mapping) and isinstance(row.get("molecule_chembl_id"), str)
        }
        known_proteins = {
            f"{row.get('taxonomy_id')}|{row.get('accession')}" for row in bundle.get("proteins", [])
            if isinstance(row, Mapping) and row.get("taxonomy_id") is not None and row.get("accession") is not None
        }
        verified_molecule_ids = {row["entity_id"] for row in molecules}
        verified_protein_ids = {row["entity_id"] for row in proteins}
        molecule_evidence = {row["entity_id"]: tuple(row.get("source_provenance", ())) for row in molecules}
        protein_evidence = {row["entity_id"]: tuple(row.get("source_provenance", ())) for row in proteins}
        molecule_memberships = _normalize_memberships(
            split.get("molecule_memberships"), kind="molecule",
            known_entities=known_molecules, verified_entities=verified_molecule_ids,
            findings=findings, evidence_by_entity=molecule_evidence,
        )
        protein_memberships = _normalize_memberships(
            split.get("protein_memberships"), kind="protein",
            known_entities=known_proteins, verified_entities=verified_protein_ids,
            findings=findings, evidence_by_entity=protein_evidence,
        )

        molecule_components = _component_memberships(
            verified_molecule_ids, neighbor_result.molecule_neighbors, prefix="molecule-cluster-v1",
        )
        protein_components = _component_memberships(
            verified_protein_ids, neighbor_result.protein_neighbors, prefix="protein-cluster-v1",
        )
        for hit in neighbor_result.molecule_neighbors + neighbor_result.protein_neighbors:
            memberships = molecule_memberships if hit.kind.startswith("molecule_") else protein_memberships
            split_a = memberships.get(hit.entity_ids[0])
            split_b = memberships.get(hit.entity_ids[1])
            if split_a is None or split_b is None:
                findings.append(AuditFinding(
                    code="neighbor_assignment_missing",
                    message=f"independent {hit.kind} neighbor has an entity without a split assignment",
                    entity_ids=hit.entity_ids,
                    split=f"{split_a or 'unassigned'}|{split_b or 'unassigned'}",
                    metric=hit.metric, threshold=hit.threshold, source_evidence=hit.source_evidence,
                ))
            elif split_a != split_b:
                findings.append(_finding_from_hit(hit, split_a, split_b))

        try:
            groups, filtering_ledger = group_evidence(bundle.get("activities", []))
            label_result = apply_label_policy(
                groups, ledger=filtering_ledger, policy_path=label_policy_path_frozen,
                entity_bundle_path=entity_path, allow_test_only=True,
            )
            confirmed, test_only, diagnostics = _label_maps(label_result, groups)
        except Exception as exc:
            if _dependency_error(exc):
                return _not_auditable("dependency_missing", str(exc), findings=findings, **policy_report_context)
            findings.append(AuditFinding(
                code="evidence_policy_rejected", message=f"frozen evidence could not be independently regrouped and labeled: {exc}",
            ))
            return AuditReport("violation", 3, tuple(findings), _empty_support(), False,
                               str(bundle.get("entity_version")), str(bundle.get("content_sha256")), policy_sha256,
                               tuple(bundle.get("source_provenance", ())), str(exc),
                               split_policy_file_sha256=split_reference.get("file_sha256"),
                               label_policy_sha256=label_reference.get("file_sha256"),
                               label_policy_semantic_sha256=label_reference.get("semantic_sha256"))

        if (label_result.policy_sha256 != label_reference.get("file_sha256")
                or _canonical_hash(label_policy_value) != label_reference.get("semantic_sha256")
                or label_result.policy_version != label_reference.get("policy_version")):
            findings.append(AuditFinding(
                code="label_policy_recompute_mismatch",
                message="label recomputation did not use the exact frozen label-policy bytes/version",
                entity_ids=(str(label_result.policy_sha256), str(label_reference.get("file_sha256"))),
            ))

        evidence_by_pair = _source_rows_by_pair(bundle, groups)
        for entry in filtering_ledger.entries:
            if entry.disposition == "duplicate":
                findings.append(AuditFinding(
                    code="duplicate_activity_record",
                    message="same source activity identity appears more than once in the frozen evidence bundle",
                    entity_ids=(str(entry.evidence_id), str(entry.occurrence)),
                    source_evidence=({"pair_key": entry.pair_key, "record_sha256": entry.record_sha256,
                                      "reason": entry.reason},),
                ))
            if entry.pair_key is None and entry.disposition != "duplicate":
                findings.append(AuditFinding(
                    code="unmapped_activity_evidence",
                    message="activity evidence cannot be mapped to one canonical molecule/protein pair",
                    entity_ids=(str(entry.evidence_id),),
                    source_evidence=({"record_sha256": entry.record_sha256, "reason": entry.reason},),
                ))

        expected: dict[str, dict[str, dict[str, Any]]] = {
            "pair_assignments": {},
            "crosspair_quarantine": {},
            "unassigned_pairs": {},
            "test_only_pair_assignments": {},
            "ambiguous_label_quarantine": {},
        }
        ambiguous_pairs: set[str] = set()
        groups_by_pair = {group.pair_key: group for group in groups}
        if len(groups_by_pair) != len(groups):
            findings.append(AuditFinding(code="duplicate_pair_identity", message="canonical evidence produced duplicate pair groups"))

        for group in sorted(groups, key=lambda item: (item.pair_key, item.split_group_id)):
            pair_parts = group.pair_key.split("|")
            if len(pair_parts) != 3:
                findings.append(AuditFinding(
                    code="pair_identity_invalid", message="canonical pair key does not contain molecule, taxonomy, accession",
                    entity_ids=(group.pair_key,), source_evidence=evidence_by_pair.get(group.pair_key, ()),
                ))
                continue
            molecule_id = group.molecule_chembl_id
            protein_id = f"{group.taxonomy_id}|{group.protein_accession}"
            if molecule_id not in known_molecules or protein_id not in known_proteins:
                findings.append(AuditFinding(
                    code="unknown_entity_mapping", message="activity pair references an entity absent from frozen entity records",
                    entity_ids=(molecule_id, protein_id), source_evidence=evidence_by_pair.get(group.pair_key, ()),
                ))
            molecule_cluster_id = molecule_components.get(molecule_id)
            protein_cluster_id = protein_components.get(protein_id)
            molecule_split = molecule_memberships.get(molecule_id)
            protein_split = protein_memberships.get(protein_id)
            confirmed_labels = confirmed.get(group.pair_key, set())
            test_labels = test_only.get(group.pair_key, set())
            labels = confirmed_labels or test_labels
            partition = "confirmed" if confirmed_labels else "test_only" if test_labels else "unlabeled"
            diagnostic_reasons = diagnostics.get(group.pair_key, set())
            ambiguous = len(confirmed_labels) > 1 or (not confirmed_labels and len(test_labels) > 1)
            if ambiguous:
                ambiguous_pairs.add(group.pair_key)
                reasons = diagnostic_reasons | {"ambiguous_multi_label"}
                expected["ambiguous_label_quarantine"][group.pair_key] = _pair_row(
                    group, split=None, labels=labels, provenance_partition=partition, reasons=reasons,
                    molecule_cluster_id=molecule_cluster_id, protein_cluster_id=protein_cluster_id,
                )
                if molecule_cluster_id is None or protein_cluster_id is None or molecule_split is None or protein_split is None:
                    expected["unassigned_pairs"][group.pair_key] = _pair_row(
                        group, split=None, labels=labels, provenance_partition=partition,
                        reasons=reasons | {"entity_not_clusterable"},
                        molecule_cluster_id=molecule_cluster_id, protein_cluster_id=protein_cluster_id,
                    )
                elif molecule_split != protein_split:
                    expected["crosspair_quarantine"][group.pair_key] = _pair_row(
                        group, split=None, labels=labels, provenance_partition=partition,
                        reasons=reasons | {"crosspair_cluster_partition_mismatch"},
                        molecule_cluster_id=molecule_cluster_id, protein_cluster_id=protein_cluster_id,
                    )
                continue

            if molecule_cluster_id is None or protein_cluster_id is None or molecule_split is None or protein_split is None:
                expected["unassigned_pairs"][group.pair_key] = _pair_row(
                    group, split=None, labels=labels, provenance_partition=partition,
                    reasons=diagnostic_reasons | {"entity_not_clusterable"},
                    molecule_cluster_id=molecule_cluster_id, protein_cluster_id=protein_cluster_id,
                )
                continue
            if molecule_split != protein_split:
                expected["crosspair_quarantine"][group.pair_key] = _pair_row(
                    group, split=None, labels=labels, provenance_partition=partition,
                    reasons=diagnostic_reasons | {"crosspair_cluster_partition_mismatch"},
                    molecule_cluster_id=molecule_cluster_id, protein_cluster_id=protein_cluster_id,
                )
                continue
            if confirmed_labels:
                expected["pair_assignments"][group.pair_key] = _pair_row(
                    group, split=molecule_split, labels=confirmed_labels, provenance_partition="confirmed",
                    reasons=(), molecule_cluster_id=molecule_cluster_id, protein_cluster_id=protein_cluster_id,
                )
            elif test_labels:
                expected["test_only_pair_assignments"][group.pair_key] = _pair_row(
                    group, split=molecule_split, labels=test_labels, provenance_partition="test_only",
                    reasons=(), molecule_cluster_id=molecule_cluster_id, protein_cluster_id=protein_cluster_id,
                )
            else:
                expected["unassigned_pairs"][group.pair_key] = _pair_row(
                    group, split=None, labels=(), provenance_partition="unlabeled",
                    reasons=diagnostic_reasons or {"no_confirmed_label"},
                    molecule_cluster_id=molecule_cluster_id, protein_cluster_id=protein_cluster_id,
                )

        actual_bucket_names = {
            "pair_assignments": "pair_assignments",
            "crosspair_quarantine": "crosspair_quarantine",
            "unassigned_pairs": "unassigned_pairs",
            "test_only_pair_assignments": "test_only_pair_assignments",
            "ambiguous_label_quarantine": "ambiguous_label_quarantine",
        }
        actual_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
        for category, field_name in actual_bucket_names.items():
            values = split.get(field_name)
            if not isinstance(values, list):
                findings.append(AuditFinding(
                    code="split_ledger_invalid", message=f"{field_name} must be an array",
                ))
                actual_rows[category] = {}
                continue
            bucket: dict[str, Mapping[str, Any]] = {}
            for row in values:
                if not isinstance(row, Mapping) or not isinstance(row.get("pair_key"), str):
                    findings.append(AuditFinding(
                        code="split_ledger_invalid", message=f"{field_name} contains a malformed row",
                    ))
                    continue
                pair_key = row["pair_key"]
                if pair_key in bucket:
                    findings.append(AuditFinding(
                        code="duplicate_pair_assignment", message=f"pair is repeated within {field_name}",
                        entity_ids=(pair_key,), source_evidence=evidence_by_pair.get(pair_key, ()),
                    ))
                    continue
                bucket[pair_key] = row
                ids = row.get("evidence_ids")
                if not isinstance(ids, (list, tuple)) or len(ids) != len(set(str(item) for item in ids)):
                    findings.append(AuditFinding(
                        code="duplicate_activity_assignment", message=f"{field_name} repeats or malforms an activity evidence ID",
                        entity_ids=(pair_key,), source_evidence=evidence_by_pair.get(pair_key, ()),
                    ))
            actual_rows[category] = bucket

        for category, expected_rows in expected.items():
            actual = actual_rows.get(category, {})
            for pair_key in sorted(set(expected_rows) - set(actual)):
                findings.append(AuditFinding(
                    code="split_ledger_missing_pair", message=f"{category} omits a source-derived pair record",
                    entity_ids=(pair_key,), source_evidence=evidence_by_pair.get(pair_key, ()),
                ))
            for pair_key in sorted(set(actual) - set(expected_rows)):
                findings.append(AuditFinding(
                    code="split_ledger_unexpected_pair", message=f"{category} contains a pair not justified by frozen source evidence",
                    entity_ids=(pair_key,), source_evidence=evidence_by_pair.get(pair_key, ()),
                ))
            for pair_key in sorted(set(expected_rows) & set(actual)):
                expected_row = expected_rows[pair_key]
                actual_row = actual[pair_key]
                if _row_signature(expected_row) != _row_signature(actual_row):
                    findings.append(AuditFinding(
                        code="split_ledger_mapping_mismatch", message=f"{category} row differs from independently derived pair/evidence mapping",
                        entity_ids=(pair_key,), split=str(actual_row.get("split")) if actual_row.get("split") is not None else None,
                        source_evidence=evidence_by_pair.get(pair_key, ()),
                    ))
                if category == "ambiguous_label_quarantine" and "ambiguous_multi_label" not in set(actual_row.get("reasons", ())):
                    findings.append(AuditFinding(
                        code="ambiguous_multi_label_marker_missing",
                        message="ambiguous pair quarantine dropped its ambiguous_multi_label reason",
                        entity_ids=(pair_key,), source_evidence=evidence_by_pair.get(pair_key, ()),
                    ))

        all_category_pairs: dict[str, set[str]] = defaultdict(set)
        for category, rows in actual_rows.items():
            for pair_key, row in rows.items():
                all_category_pairs[pair_key].add(category)
                ids = row.get("evidence_ids", ())
                if isinstance(ids, (list, tuple)):
                    if any(str(evidence_id) not in set(group.evidence_ids)
                           for group in groups if group.pair_key == pair_key for evidence_id in ids):
                        findings.append(AuditFinding(
                            code="unknown_activity_mapping", message="split ledger references an activity ID outside its canonical pair group",
                            entity_ids=(pair_key,), source_evidence=evidence_by_pair.get(pair_key, ()),
                        ))
        for pair_key, categories in all_category_pairs.items():
            permitted = {category for category, rows in expected.items() if pair_key in rows}
            if not categories.issubset(permitted):
                findings.append(AuditFinding(
                    code="split_ledger_category_mismatch", message="pair appears in a ledger category not derived from source evidence",
                    entity_ids=(pair_key,), source_evidence=evidence_by_pair.get(pair_key, ()),
                ))

        independently_derived_support = _support_from_rows(expected["pair_assignments"].values())
        claimed_support = split.get("class_support")
        if not isinstance(claimed_support, Mapping):
            findings.append(AuditFinding(code="class_support_invalid", message="class_support must be an object"))
        else:
            normalized_claimed: dict[str, dict[str, int]] = {}
            for partition in _SPLITS:
                values = claimed_support.get(partition)
                if not isinstance(values, Mapping):
                    normalized_claimed[partition] = {label: -1 for label in _CLASSES}
                    continue
                normalized_claimed[partition] = {
                    label: values.get(label) if type(values.get(label)) is int else -1
                    for label in _CLASSES
                }
            if normalized_claimed != independently_derived_support:
                findings.append(AuditFinding(
                    code="class_support_mismatch",
                    message="caller class support differs from unique pair-level counts independently derived from frozen labels and memberships",
                    source_evidence=tuple(_json_safe(item) for item in bundle.get("source_provenance", []) if isinstance(item, Mapping)),
                ))

        support_ready = all(independently_derived_support[partition][label] > 0 for partition in _SPLITS for label in _CLASSES)
        expected_split_status = "ready" if support_ready else "insufficient"
        if split.get("strict_split_status") != expected_split_status:
            findings.append(AuditFinding(
                code="strict_split_status_mismatch",
                message="caller strict_split_status differs from independently recomputed retained class support",
                entity_ids=(str(split.get("strict_split_status")), expected_split_status),
            ))
        computed_ready = support_ready and not findings
        if type(split.get("training_ready")) is not bool:
            findings.append(AuditFinding(code="training_ready_invalid", message="training_ready must be boolean"))
        elif ((split.get("training_ready") is True and not computed_ready)
              or (audit_is_final and split.get("training_ready") != computed_ready)):
            findings.append(AuditFinding(
                code="training_ready_mismatch",
                message="final/affirmative training_ready claim does not match independent retained support and audit results",
                source_evidence=tuple(_json_safe(item) for item in bundle.get("source_provenance", []) if isinstance(item, Mapping)),
            ))
        verified_entity_counts = {"molecules": len(molecules), "proteins": len(proteins)}
        component_counts = {
            "molecule_clusters": len(set(molecule_components.values())),
            "protein_clusters": len(set(protein_components.values())),
        }
        label_disposition_counts = {str(key): int(value) for key, value in label_result.counts.items()}
        ledger_disposition_counts = {str(key): int(value) for key, value in label_result.ledger.counts.items()}
        if source_receipt is not None:
            try:
                expected_receipt_bindings = _receipt_bindings_from_manifest(manifest)
            except AuditInputError as exc:
                findings.append(AuditFinding(
                    code="source_readiness_receipt_binding_invalid",
                    message=f"manifest cannot satisfy the source readiness receipt bindings: {exc}",
                ))
            else:
                if _canonical_hash(source_receipt.get("bindings")) != _canonical_hash(expected_receipt_bindings):
                    findings.append(AuditFinding(
                        code="source_readiness_receipt_binding_mismatch",
                        message="receipt bindings do not describe this exact entity, source, policy, evidence, and split corpus",
                    ))
                expected_scope_hash = canonical_json_sha256(expected_receipt_bindings)
                if source_receipt.get("scope_sha256") != expected_scope_hash:
                    findings.append(AuditFinding(
                        code="source_readiness_receipt_scope_mismatch",
                        message="receipt scope_sha256 does not match independently reconstructed corpus bindings",
                    ))
            expected_receipt_metrics = {
                "verified_entity_counts": verified_entity_counts,
                "component_counts": component_counts,
                "label_disposition_counts": label_disposition_counts,
                "ledger_disposition_counts": ledger_disposition_counts,
                "excluded_label_dispositions": {
                    "confirmed": label_disposition_counts.get("confirmed_excluded", 0),
                    "test_only": label_disposition_counts.get("test_only_excluded", 0),
                    "ledger": ledger_disposition_counts.get("excluded", 0),
                },
                "double_cold_split_support": independently_derived_support,
                "audit_status": "clean",
                "audit_exit_code": 0,
                "training_ready": computed_ready,
            }
            if _canonical_hash(source_receipt.get("metrics")) != _canonical_hash(expected_receipt_metrics):
                findings.append(AuditFinding(
                    code="source_readiness_receipt_metrics_mismatch",
                    message="receipt metrics differ from independently recomputed entity, cluster, label, split, or readiness values",
                ))
        computed_ready = computed_ready and not findings

        exit_code = 3 if findings else 0
        status = "violation" if findings else "clean"
        return AuditReport(
            status=status,
            exit_code=exit_code,
            findings=tuple(findings),
            class_support=independently_derived_support,
            training_ready=computed_ready,
            entity_version=str(bundle.get("entity_version")) if bundle.get("entity_version") is not None else None,
            entity_bundle_hash=str(bundle.get("content_sha256")) if bundle.get("content_sha256") is not None else None,
            split_policy_sha256=policy_sha256,
            source_provenance=tuple(bundle.get("source_provenance", ())),
            split_policy_file_sha256=str(split_reference.get("file_sha256")),
            label_policy_sha256=str(label_reference.get("file_sha256")),
            label_policy_semantic_sha256=label_policy_semantic_sha256,
            verified_entity_counts=verified_entity_counts,
            component_counts=component_counts,
            label_disposition_counts=label_disposition_counts,
            ledger_disposition_counts=ledger_disposition_counts,
        )
    except _AuditStop as exc:
        return _not_auditable(exc.status, str(exc), findings=findings, **policy_report_context)
    except (AuditInputError, EvidenceInputError, TypeError, ValueError, OSError) as exc:
        if _dependency_error(exc):
            return _not_auditable("dependency_missing", str(exc), findings=findings, **policy_report_context)
        if isinstance(exc, AuditInputError):
            findings.append(AuditFinding(code="audit_input_invalid", message=str(exc)))
            return AuditReport("violation", 3, tuple(findings), _empty_support(), False, error=str(exc))
        findings.append(AuditFinding(
            code=_integrity_code_for_bundle_error(str(exc)),
            message=f"audit could not validate a frozen contract: {exc}",
        ))
        return AuditReport("violation", 3, tuple(findings), _empty_support(), False, error=str(exc))


def _audit_candidate_split(manifest_path: os.PathLike[str] | str) -> AuditReport:
    """Audit only an unpublished writer candidate before its receipt is written."""
    candidate = Path(manifest_path)
    if not candidate.name.startswith(".manifest.audit.tmp."):
        return _manifest_failure("private candidate audit accepts only the frozen writer's provisional manifest")
    return _audit_split_core(candidate, _allow_unreceipted_candidate=True)


def audit_split(
    manifest_path: os.PathLike[str] | str,
    *,
    policy_path: Any = None,
    label_policy_path: Any = None,
    budget_overrides: Optional[Mapping[str, Any]] = None,
) -> AuditReport:
    """Audit a frozen split manifest without trusting its cluster IDs or counts."""
    return _audit_split_core(
        manifest_path,
        policy_path=policy_path,
        label_policy_path=label_policy_path,
        budget_overrides=budget_overrides,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently audit a frozen split manifest")
    parser.add_argument("--manifest", required=True, help="path to a frozen manifest.json")
    parser.add_argument("--policy", help="optional split policy file override; its bytes must match the frozen snapshot")
    parser.add_argument("--label-policy", help="optional label policy file override; its bytes must match the frozen snapshot")
    parser.add_argument("--max-molecule-pairs", type=int, help="runtime-only upper bound for this audit")
    parser.add_argument("--max-protein-pairs", type=int, help="runtime-only upper bound for this audit")
    parser.add_argument("--max-alignment-cells", type=int, help="runtime-only upper bound for this audit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_argument_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    budget_overrides = {
        key: value for key, value in (
            ("molecule_pairs", args.max_molecule_pairs),
            ("protein_pairs", args.max_protein_pairs),
            ("alignment_cells", args.max_alignment_cells),
        ) if value is not None
    }
    report = audit_split(
        args.manifest,
        policy_path=args.policy,
        label_policy_path=args.label_policy,
        budget_overrides=budget_overrides or None,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

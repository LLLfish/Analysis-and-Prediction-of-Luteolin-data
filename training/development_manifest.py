"""Train-only engineering development manifest for Phase 03 (T3-01).

The development manifest is an independent, source-bound audit ledger.  It is
derived *only* from the frozen corpus manifest's split.pair_assignments rows
whose split is 'train'; quarantine, validation, test and unassigned ledgers are
never enumerated.  Every train candidate receives exactly one disposition:

* included with a stable reason code when the pair has a unique binary label,
  verified molecule structure / protein identity and sequence, and a ready,
  finite, source-bound 320-d ESM feature row;
* excluded with a stable reason code otherwise.

No label is invented or rewritten, no old artifact is overwritten (output only
creates a brand-new file inside a brand-new P1_03_Artifacts_<32 hex> run root),
and the frozen source files are read-only inputs.

CPU-only module: standard library only.  The registered CUDA interpreter never
imports this module and it performs no training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ESM_MODEL_ID = "facebook/esm2_t6_8M_UR50D"
ESM_EMBEDDING_DIM = 320
LABEL_MAPPING = {"inactive": 0, "active": 1}
PURPOSE = "engineering_development_manifest"
PURPOSE_STATEMENT = (
    "train-only engineering development manifest for in-sample fit diagnostics; "
    "no independent holdout, no AUC/probability calibration/ranking or "
    "generalisation claim"
)
ELIGIBILITY_RULES_VERSION = "phase03-development-eligibility-v1"
CANDIDATE_RULE = (
    "candidates are only frozen split.pair_assignments rows with split == "
    "'train'; no other split ledger is enumerated"
)
SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "development_manifest.json"

EXPECTED_FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "P1_02_04_Artifacts_847e7b27ed5942e19af64e2bd65b8d2f"
    / "01d118a1293346d38a58f116f7c86e2a"
    / "feature_manifest.json"
)
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "46d13b05db30c15e39cc4b455a5ab3ec1a0faf6265017b32301b08212da0c1c5"
)
RUN_ROOT_PATTERN = re.compile(r"P1_03_Artifacts_[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError("invalid_sha256:" + name)
    return value


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(name + "_missing:" + str(path))
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(name + "_not_mapping:" + str(path))
    return value


def _absolute_input_file(path: Any, code: str) -> Path:
    candidate = Path(os.path.expanduser(os.fspath(path)))
    if not candidate.is_absolute():
        raise ValueError(code + "_path_must_be_absolute")
    if not candidate.is_file():
        raise ValueError(code + "_missing:" + str(candidate))
    return candidate


def parse_pair_key(pair_key: Any) -> tuple[str, int, str]:
    """Split and validate molecule_chembl_id|taxonomy_id|accession."""
    if not isinstance(pair_key, str) or not pair_key:
        raise ValueError("pair_key_not_string")
    parts = pair_key.split("|")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("pair_key_malformed:" + pair_key)
    molecule_id, taxonomy_text, accession = parts
    if not taxonomy_text.isdigit():
        raise ValueError("pair_key_taxonomy_invalid:" + pair_key)
    return molecule_id, int(taxonomy_text), accession


def _unique_index(
    records: Any, key: str, label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(label + "_records_not_list")
    index: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(label + "_record_not_mapping:" + str(position))
        value = record.get(key)
        if not _is_text(value):
            raise ValueError(label + "_record_missing_" + key + ":" + str(position))
        if value in index:
            raise ValueError(label + "_duplicate_" + key + ":" + value)
        index[value] = record
    return index


def _evidence_ids(value: Any, pair_key: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("pair_evidence_ids_invalid:" + pair_key)
    identifiers: list[str] = []
    for item in value:
        if not _is_text(item):
            raise ValueError("pair_evidence_ids_invalid:" + pair_key)
        identifiers.append(item)
    return identifiers


# ---------------------------------------------------------------------------
# Source loading and hash/bundle verification
# ---------------------------------------------------------------------------

def load_verified_sources(
    manifest_path: os.PathLike[str] | str,
    entities_path: os.PathLike[str] | str,
    feature_manifest_path: os.PathLike[str] | str,
    *,
    expected_feature_manifest_sha256: Any = None,
    expected_feature_manifest_path: Any = None,
) -> dict[str, Any]:
    """Load the frozen manifest/entities and selected feature snapshot.

    Cross-checks every recorded source/bundle/version binding.  Identity or
    provenance contradictions raise stable error codes; availability problems
    are left to per-row adjudication.
    """
    frozen_file = _absolute_input_file(manifest_path, "frozen_manifest")
    entities_file = _absolute_input_file(entities_path, "entities")
    feature_file = _absolute_input_file(feature_manifest_path, "feature_manifest")

    expected_feature_sha = _require_sha256(
        EXPECTED_FEATURE_MANIFEST_SHA256
        if expected_feature_manifest_sha256 is None
        else expected_feature_manifest_sha256,
        "expected_feature_manifest_sha256",
    )
    expected_feature_path = Path(
        os.path.expanduser(os.fspath(
            EXPECTED_FEATURE_MANIFEST_PATH
            if expected_feature_manifest_path is None
            else expected_feature_manifest_path
        ))
    )

    manifest = _load_json_object(frozen_file, "frozen_manifest")
    manifest_sha256 = sha256_file(frozen_file)
    if type(manifest.get("schema_version")) is not int:
        raise ValueError("frozen_manifest_schema_version_invalid")

    entity_meta = manifest.get("entities")
    if not isinstance(entity_meta, Mapping):
        raise ValueError("frozen_manifest_entities_missing")
    entity_relative = entity_meta.get("path")
    if not _is_text(entity_relative):
        raise ValueError("frozen_manifest_entities_path_invalid")
    declared_entities = Path(entity_relative)
    if declared_entities.is_absolute():
        declared_entities_path = declared_entities
    else:
        if ".." in declared_entities.parts:
            raise ValueError("frozen_manifest_entities_path_not_relative")
        declared_entities_path = frozen_file.parent / declared_entities
    if declared_entities_path.resolve(strict=False) != entities_file.resolve(strict=False):
        raise ValueError("entities_path_not_frozen_entities_path")

    entities_sha256 = sha256_file(entities_file)
    expected_entities_sha = _require_sha256(
        entity_meta.get("file_sha256"), "entities_file_sha256",
    )
    if expected_entities_sha != entities_sha256:
        raise ValueError("entities_file_sha256_mismatch")
    content_sha256 = _require_sha256(
        entity_meta.get("content_sha256"), "entities_content_sha256",
    )
    entity_version = entity_meta.get("entity_version")
    if not _is_text(entity_version):
        raise ValueError("frozen_manifest_entity_version_missing")

    entities = _load_json_object(entities_file, "frozen_entities")
    if entities.get("content_sha256") != content_sha256:
        raise ValueError("entities_content_sha256_mismatch")
    if entities.get("entity_version") != entity_version:
        raise ValueError("entities_entity_version_mismatch")

    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("frozen_manifest_split_missing")
    assignments = split.get("pair_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("frozen_manifest_pair_assignments_missing")
    if split.get("entity_bundle_hash") != content_sha256:
        raise ValueError("frozen_split_entity_bundle_hash_mismatch")
    if split.get("entity_version") != entity_version:
        raise ValueError("frozen_split_entity_version_mismatch")
    for label in ("molecules", "proteins"):
        if not isinstance(entities.get(label), list) or not entities[label]:
            raise ValueError("frozen_entities_" + label + "_missing")

    # Selected feature snapshot: path + SHA are pinned, bundle/version bound.
    feature_sha256 = sha256_file(feature_file)
    if feature_sha256 != expected_feature_sha:
        raise ValueError("feature_manifest_sha256_mismatch")
    if feature_file.resolve(strict=False) != expected_feature_path.resolve(strict=False):
        raise ValueError("feature_manifest_path_mismatch")
    feature = _load_json_object(feature_file, "feature_manifest")
    if type(feature.get("schema_version")) is not int:
        raise ValueError("feature_manifest_schema_version_invalid")
    if feature.get("bundle_file_sha256") != entities_sha256:
        raise ValueError("feature_bundle_file_sha256_mismatch")
    if feature.get("entity_bundle_hash") != content_sha256:
        raise ValueError("feature_entity_bundle_hash_mismatch")
    if feature.get("entity_version") != entity_version:
        raise ValueError("feature_entity_version_mismatch")

    model = feature.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("feature_model_metadata_missing")
    if model.get("model_id") != ESM_MODEL_ID:
        raise ValueError("feature_model_id_mismatch:" + repr(model.get("model_id")))
    revision = model.get("revision")
    if not _is_text(revision):
        raise ValueError("feature_model_revision_missing")
    if model.get("external_revision_verified") is not False:
        raise ValueError("feature_external_revision_claim")
    embedding_dim = model.get("embedding_dim")
    if type(embedding_dim) is not int or embedding_dim != ESM_EMBEDDING_DIM:
        raise ValueError("feature_manifest_embedding_dim_mismatch:" + repr(embedding_dim))

    feature_rows = feature.get("entities")
    if not isinstance(feature_rows, list) or not feature_rows:
        raise ValueError("feature_entities_missing")
    feature_index: dict[tuple[str, int], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    for position, row in enumerate(feature_rows):
        if not isinstance(row, dict):
            raise ValueError("feature_row_not_mapping:" + str(position))
        accession = row.get("accession")
        taxonomy_id = row.get("taxonomy_id")
        if not _is_text(accession):
            raise ValueError("feature_row_accession_invalid:" + str(position))
        if type(taxonomy_id) is not int:
            raise ValueError("feature_row_taxonomy_invalid:" + str(position))
        key = (accession, taxonomy_id)
        if key in feature_index:
            raise ValueError("feature_entity_duplicate:" + accession)
        for field, expected in (
            ("entity_bundle_hash", content_sha256),
            ("bundle_file_sha256", entities_sha256),
            ("entity_version", entity_version),
        ):
            if row.get(field) != expected:
                raise ValueError("feature_row_" + field + "_mismatch:" + accession)
        if row.get("embedding_dim") != embedding_dim:
            raise ValueError("feature_row_embedding_dim_mismatch:" + accession)
        if row.get("model_id") != ESM_MODEL_ID:
            raise ValueError("feature_row_model_id_mismatch:" + accession)
        _require_sha256(row.get("sequence_sha256"), "feature_row_sequence_sha256:" + accession)
        status = row.get("status")
        if not _is_text(status):
            raise ValueError("feature_row_status_invalid:" + accession)
        status_counts[status] = status_counts.get(status, 0) + 1
        feature_index[key] = row

    return {
        "manifest": manifest,
        "manifest_path": frozen_file,
        "manifest_sha256": manifest_sha256,
        "entities": entities,
        "entities_path": entities_file,
        "entities_sha256": entities_sha256,
        "content_sha256": content_sha256,
        "entity_version": entity_version,
        "split": split,
        "assignments": assignments,
        "molecules": entities["molecules"],
        "proteins": entities["proteins"],
        "feature_manifest": {
            "manifest": feature,
            "path": feature_file,
            "sha256": feature_sha256,
            "model": dict(model),
            "rows": feature_rows,
            "index": feature_index,
            "status_counts": status_counts,
            "embedding_dim": embedding_dim,
            "revision": revision,
        },
    }


# ---------------------------------------------------------------------------
# Per-row adjudication
# ---------------------------------------------------------------------------

def _adjudicate_pair(
    assignment: Mapping[str, Any],
    *,
    position: int,
    sources: Mapping[str, Any],
    molecule_index: Mapping[str, dict[str, Any]],
    protein_index: Mapping[str, dict[str, Any]],
    feature_index: Mapping[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    pair_key = assignment.get("pair_key")
    molecule_id, taxonomy_id, accession = parse_pair_key(pair_key)
    evidence_ids = _evidence_ids(assignment.get("evidence_ids"), pair_key)

    labels = assignment.get("labels")
    label: int | None = None
    label_name: str | None = None
    reason: str | None = None
    reason_detail: str | None = None
    if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in LABEL_MAPPING:
        reason = "label_missing_or_not_unique_binary"
        reason_detail = repr(labels)
    else:
        label_name = labels[0]
        label = LABEL_MAPPING[label_name]

    molecule = molecule_index.get(molecule_id)
    protein = protein_index.get(accession)
    feature_row = feature_index.get((accession, taxonomy_id))

    molecule_structure_status = molecule.get("structure_status") if molecule else None
    molecule_structure_sha256 = molecule.get("structure_sha256") if molecule else None
    protein_sequence_status = protein.get("sequence_status") if protein else None
    protein_identity_status = protein.get("identity_status") if protein else None
    protein_sequence_sha256 = protein.get("sequence_sha256") if protein else None
    feature_status = feature_row.get("status") if feature_row else None
    feature_cache_key = None
    feature_cache_file_sha256 = None

    if reason is None:
        if molecule is None or protein is None:
            reason = "entity_missing"
            reason_detail = (
                "molecule_missing" if molecule is None else "protein_missing"
            )
        elif protein.get("taxonomy_id") != taxonomy_id:
            raise ValueError("pair_taxonomy_mismatch:" + pair_key)
        elif molecule_structure_status != "verified":
            reason = "molecule_structure_not_verified"
            reason_detail = str(molecule_structure_status)
        elif protein_sequence_status != "verified":
            reason = "protein_sequence_not_verified"
            reason_detail = str(protein_sequence_status)
        elif protein_identity_status != "verified":
            reason = "protein_identity_not_verified"
            reason_detail = str(protein_identity_status)
        else:
            if not _is_sha256(protein_sequence_sha256):
                raise ValueError("protein_sequence_sha256_invalid:" + pair_key)
            if not _is_sha256(molecule_structure_sha256):
                raise ValueError("molecule_structure_sha256_invalid:" + pair_key)
            if not _is_text(protein.get("full_sequence")):
                reason = "protein_sequence_missing"
            elif feature_row is None:
                reason = "feature_row_missing"
            elif feature_status != "ready":
                reason = "feature_status_" + str(feature_status)
                reason_detail = str(feature_row.get("reason"))
            elif feature_row.get("sequence_sha256") != protein_sequence_sha256:
                raise ValueError("feature_sequence_sha256_mismatch:" + pair_key)
            elif feature_row.get("finite") is not True:
                reason = "feature_not_finite"
            elif feature_row.get("test_provenance") is not None:
                reason = "feature_test_only_row"
            else:
                features = feature_row.get("features")
                embedding = (
                    features.get("esm_embedding")
                    if isinstance(features, Mapping) else None
                )
                if not isinstance(embedding, list) or len(embedding) != ESM_EMBEDDING_DIM:
                    reason = "feature_embedding_dim_mismatch"
                else:
                    for value in embedding:
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                        ):
                            reason = "feature_embedding_non_finite"
                            break
                if reason is None and (
                    not _is_sha256(feature_row.get("cache_key"))
                    or not _is_sha256(feature_row.get("cache_file_sha256"))
                ):
                    reason = "feature_cache_binding_missing"
                if reason is None:
                    feature_cache_key = feature_row["cache_key"]
                    feature_cache_file_sha256 = feature_row["cache_file_sha256"]

    included = reason is None
    return {
        "row_index": None,
        "pair_key": pair_key,
        "split": "train",
        "disposition": "included" if included else "excluded",
        "reason_code": "included_eligible" if included else reason,
        "reason_detail": reason_detail,
        "evidence_ids": evidence_ids,
        "molecule_chembl_id": molecule_id,
        "taxonomy_id": taxonomy_id,
        "accession": accession,
        "label": label,
        "label_name": label_name,
        "group_sha256": assignment.get("group_sha256"),
        "provenance_partition": assignment.get("provenance_partition"),
        "molecule_cluster_id": assignment.get("molecule_cluster_id"),
        "protein_cluster_id": assignment.get("protein_cluster_id"),
        "molecule_structure_status": molecule_structure_status,
        "molecule_structure_sha256": molecule_structure_sha256,
        "protein_sequence_status": protein_sequence_status,
        "protein_identity_status": protein_identity_status,
        "protein_sequence_sha256": protein_sequence_sha256,
        "feature_status": feature_status,
        "feature_cache_key": feature_cache_key,
        "feature_cache_file_sha256": feature_cache_file_sha256,
        "feature_model_id": feature_row.get("model_id") if feature_row else None,
        "feature_model_revision": feature_row.get("model_revision") if feature_row else None,
        "feature_sequence_sha256": feature_row.get("sequence_sha256") if feature_row else None,
        "entity_bundle_hash": sources["content_sha256"],
        "entity_version": sources["entity_version"],
        "bundle_file_sha256": sources["entities_sha256"],
        "purpose": PURPOSE,
    }


def _adjudicate_candidates(sources: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    molecule_index = _unique_index(sources["molecules"], "molecule_chembl_id", "molecule")
    protein_index = _unique_index(sources["proteins"], "accession", "protein")
    feature_index = sources["feature_manifest"]["index"]

    ledger: list[dict[str, Any]] = []
    seen_pair_keys: set[str] = set()
    skipped_non_train = 0
    for position, assignment in enumerate(sources["assignments"]):
        if not isinstance(assignment, Mapping):
            raise ValueError("pair_assignment_not_mapping:" + str(position))
        pair_key = assignment.get("pair_key")
        parse_pair_key(pair_key)
        if pair_key in seen_pair_keys:
            raise ValueError("duplicate_pair_key:" + pair_key)
        seen_pair_keys.add(pair_key)
        if assignment.get("split") != "train":
            skipped_non_train += 1
            continue
        ledger.append(_adjudicate_pair(
            assignment,
            position=position,
            sources=sources,
            molecule_index=molecule_index,
            protein_index=protein_index,
            feature_index=feature_index,
        ))

    ledger.sort(key=lambda row: row["pair_key"])
    included = [row for row in ledger if row["disposition"] == "included"]
    for row_index, row in enumerate(included):
        row["row_index"] = row_index
    return ledger, skipped_non_train


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------

def _assemble_manifest(
    sources: Mapping[str, Any], ledger: list[dict[str, Any]], skipped_non_train: int,
) -> dict[str, Any]:
    included = [row for row in ledger if row["disposition"] == "included"]
    excluded = [row for row in ledger if row["disposition"] == "excluded"]
    class_counts = {
        "inactive": sum(1 for row in included if row["label"] == 0),
        "active": sum(1 for row in included if row["label"] == 1),
    }
    exclusion_reason_counts: dict[str, int] = {}
    for row in excluded:
        reason = row["reason_code"]
        exclusion_reason_counts[reason] = exclusion_reason_counts.get(reason, 0) + 1

    blocking_reasons: list[str] = []
    if not included:
        blocking_reasons.append("no_eligible_pairs")
    else:
        if class_counts["inactive"] == 0:
            blocking_reasons.append("missing_class:inactive")
        if class_counts["active"] == 0:
            blocking_reasons.append("missing_class:active")

    split = sources["split"]
    feature_manifest = sources["feature_manifest"]
    model = feature_manifest["model"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blocking_reasons else "ready",
        "blocking_reasons": blocking_reasons,
        "purpose": PURPOSE,
        "purpose_statement": PURPOSE_STATEMENT,
        "data_use": {
            "train_only": True,
            "evaluation": "none (no independent holdout)",
            "generalization_claims": "none",
            "fit_scope": "engineering development in-sample fit diagnostics only",
        },
        "created_at": _utc_now(),
        "created_by": {
            "module": "training.development_manifest",
            "python": platform.python_version(),
            "eligibility_rules_version": ELIGIBILITY_RULES_VERSION,
        },
        "label_mapping": dict(LABEL_MAPPING),
        "label_domain": [0, 1],
        "eligibility_rules_version": ELIGIBILITY_RULES_VERSION,
        "candidate_rule": CANDIDATE_RULE,
        "source": {
            "frozen_manifest_path": str(sources["manifest_path"]),
            "frozen_manifest_sha256": sources["manifest_sha256"],
            "entities_path": str(sources["entities_path"]),
            "entities_sha256": sources["entities_sha256"],
            "feature_manifest_path": str(feature_manifest["path"]),
            "feature_manifest_sha256": feature_manifest["sha256"],
            "entity_bundle_hash": sources["content_sha256"],
            "entity_version": sources["entity_version"],
            "bundle_file_sha256": sources["entities_sha256"],
            "esm_model_id": model.get("model_id"),
            "esm_model_revision": feature_manifest["revision"],
            "external_revision_verified": model.get("external_revision_verified"),
            "embedding_dim": feature_manifest["embedding_dim"],
            "label_policy_sha256": split.get("label_policy_sha256"),
            "split_policy_sha256": split.get("split_policy_sha256"),
            "split_sha256": split.get("split_sha256"),
            "frozen_manifest_training_ready": split.get("training_ready"),
            "frozen_pair_assignments": len(sources["assignments"]),
            "skipped_non_train_assignments": skipped_non_train,
            "feature_status_counts": dict(feature_manifest["status_counts"]),
        },
        "counts": {
            "candidates": len(ledger),
            "included": len(included),
            "excluded": len(excluded),
            "class_counts": class_counts,
            "molecule_count": len({row["molecule_chembl_id"] for row in included}),
            "protein_count": len({row["accession"] for row in included}),
        },
        "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "row_order": [row["pair_key"] for row in included],
        "pairs": included,
        "ledger": ledger,
        "excluded": excluded,
    }


def build_development_manifest(
    frozen_manifest: os.PathLike[str] | str,
    entities_path: os.PathLike[str] | str,
    feature_manifest: os.PathLike[str] | str,
    *,
    expected_feature_manifest_sha256: Any = None,
    expected_feature_manifest_path: Any = None,
) -> dict[str, Any]:
    """Build the complete train-only development manifest from frozen sources."""
    sources = load_verified_sources(
        frozen_manifest,
        entities_path,
        feature_manifest,
        expected_feature_manifest_sha256=expected_feature_manifest_sha256,
        expected_feature_manifest_path=expected_feature_manifest_path,
    )
    ledger, skipped_non_train = _adjudicate_candidates(sources)
    return _assemble_manifest(sources, ledger, skipped_non_train)


# ---------------------------------------------------------------------------
# New-file-only output and run-root guard
# ---------------------------------------------------------------------------

def resolve_manifest_run_root(
    out_path: os.PathLike[str] | str,
) -> Path:
    """Require a brand-new P1_03_Artifacts_<32 hex>/development_manifest.json."""
    unresolved = Path(os.path.expanduser(os.fspath(out_path)))
    if not unresolved.is_absolute():
        raise ValueError("out_path_must_be_absolute")
    requested = Path(os.path.abspath(unresolved))
    if requested.name != RUN_MANIFEST_FILENAME:
        raise ValueError("manifest_out_filename_must_be_development_manifest.json")
    run_root = requested.parent
    if RUN_ROOT_PATTERN.fullmatch(run_root.name) is None:
        raise ValueError("run_root_name_must_be_P1_03_Artifacts_<32 hex>")
    if run_root.resolve(strict=False).parent != PROJECT_ROOT.resolve():
        raise ValueError("run_root_must_be_direct_child_of_project_root")
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError("run_root_already_exists:" + str(run_root))
    run_root.mkdir()
    return requested


def write_new_manifest(
    path: os.PathLike[str] | str, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the manifest file; never overwrite an existing byte stream."""
    destination = Path(os.path.expanduser(os.fspath(path)))
    if not destination.is_absolute():
        raise ValueError("manifest_path_must_be_absolute")
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest_not_mapping")
    if not destination.parent.is_dir():
        raise ValueError("manifest_parent_missing:" + str(destination.parent))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("manifest_already_exists:" + str(destination))
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(manifest), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "status": manifest.get("status"),
    }


# ---------------------------------------------------------------------------
# Consumer-side loader (used by training.pair_pipeline development mode)
# ---------------------------------------------------------------------------

def load_development_manifest(
    path: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Load and structurally validate an existing development manifest."""
    source = Path(os.path.expanduser(os.fspath(path)))
    if not source.is_absolute():
        raise ValueError("development_manifest_path_must_be_absolute")
    if not source.is_file():
        raise ValueError("development_manifest_missing:" + str(source))
    manifest = _load_json_object(source, "development_manifest")
    manifest_sha256 = sha256_file(source)
    if type(manifest.get("schema_version")) is not int:
        raise ValueError("development_manifest_schema_version_invalid")
    if manifest.get("purpose") != PURPOSE:
        raise ValueError("development_manifest_purpose_mismatch")
    status = manifest.get("status")
    if status not in ("ready", "blocked"):
        raise ValueError("development_manifest_status_invalid:" + repr(status))
    source_record = manifest.get("source")
    if not isinstance(source_record, Mapping):
        raise ValueError("development_manifest_source_missing")
    for field in (
        "frozen_manifest_path",
        "frozen_manifest_sha256",
        "entities_path",
        "entities_sha256",
        "feature_manifest_path",
        "feature_manifest_sha256",
        "entity_bundle_hash",
        "entity_version",
    ):
        if field.endswith("_path"):
            if not _is_text(source_record.get(field)):
                raise ValueError("development_manifest_source_invalid:" + field)
        else:
            _require_sha256(source_record.get(field), "development_manifest_source:" + field)

    row_order = manifest.get("row_order")
    pairs = manifest.get("pairs")
    ledger = manifest.get("ledger")
    counts = manifest.get("counts")
    if not isinstance(row_order, list) or not isinstance(pairs, list):
        raise ValueError("development_manifest_pairs_missing")
    if not isinstance(ledger, list) or not isinstance(counts, Mapping):
        raise ValueError("development_manifest_ledger_missing")
    if len(pairs) != len(row_order) or len(set(row_order)) != len(row_order):
        raise ValueError("development_manifest_row_order_mismatch")
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise ValueError("development_manifest_pair_not_mapping:" + str(index))
        if pair.get("row_index") != index or pair.get("pair_key") != row_order[index]:
            raise ValueError("development_manifest_row_order_mismatch:" + str(index))
        if pair.get("label") not in (0, 1):
            raise ValueError("development_manifest_pair_label_invalid:" + str(index))
        if pair.get("disposition") != "included":
            raise ValueError("development_manifest_pair_disposition_invalid:" + str(index))
        for field in ("molecule_chembl_id", "taxonomy_id", "accession", "label_name"):
            if pair.get(field) is None:
                raise ValueError("development_manifest_pair_field_missing:" + field)
    included_keys = {pair["pair_key"] for pair in pairs}
    ledger_keys = [row.get("pair_key") for row in ledger]
    if len(ledger_keys) != len(set(ledger_keys)):
        raise ValueError("development_manifest_ledger_duplicate_pair_key")
    if not included_keys <= set(ledger_keys):
        raise ValueError("development_manifest_ledger_incomplete")
    included_from_ledger = {
        row["pair_key"] for row in ledger
        if isinstance(row, Mapping) and row.get("disposition") == "included"
    }
    if included_from_ledger != included_keys:
        raise ValueError("development_manifest_ledger_inconsistent")
    recomputed = {
        "inactive": sum(1 for pair in pairs if pair["label"] == 0),
        "active": sum(1 for pair in pairs if pair["label"] == 1),
    }
    if counts.get("class_counts") != recomputed:
        raise ValueError("development_manifest_class_counts_mismatch")
    if counts.get("included") != len(pairs):
        raise ValueError("development_manifest_included_count_mismatch")
    return {
        "manifest": manifest,
        "path": source,
        "sha256": manifest_sha256,
        "status": status,
        "blocking_reasons": list(manifest.get("blocking_reasons") or []),
        "source": dict(source_record),
        "counts": dict(counts),
        "row_order": list(row_order),
        "pairs": pairs,
        "ledger": ledger,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="training.development_manifest",
        description=(
            "Create a train-only, source-bound Phase 03 development manifest "
            "for engineering fit diagnostics (CPU only, no training)."
        ),
    )
    parser.add_argument("--manifest", required=True, help="frozen corpus manifest.json (read-only)")
    parser.add_argument("--entities", required=True, help="frozen entities.json (read-only)")
    parser.add_argument("--features", required=True, help="selected feature_manifest.json (read-only)")
    parser.add_argument(
        "--out",
        required=True,
        help="new P1_03_Artifacts_<32 hex>/development_manifest.json",
    )
    args = parser.parse_args(argv)
    try:
        manifest = build_development_manifest(
            args.manifest, args.entities, args.features,
        )
        out_path = resolve_manifest_run_root(args.out)
        record = write_new_manifest(out_path, manifest)
    except Exception as exc:  # noqa: BLE001 - structured blocked report
        print(json.dumps({
            "status": "blocked",
            "reason": type(exc).__name__ + ":" + str(exc),
        }, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": manifest["status"],
        "out": record["path"],
        "sha256": record["sha256"],
        "counts": manifest["counts"],
        "blocking_reasons": manifest["blocking_reasons"],
        "row_order": manifest["row_order"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

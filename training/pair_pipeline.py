"""CPU preprocessing and the real two-process CUDA training tracer CLI.

Responsibilities of this CPU-side module (D-01/D-03/D-04):

* enumerate only the frozen manifest's train pair assignments, join each
  pair_key to the verified molecule/protein entity records and to one ready,
  source-bound ESM snapshot row;
* reject duplicate keys, identity/label conflicts and tampered bindings, and
  record eligibility exclusions with stable reasons;
* export numeric-only compressed NPZ arrays plus a source-bound JSON sidecar;
* drive the registered CUDA interpreter as an isolated subprocess (no RDKit or
  pytest into the GPU environment), then reload the checkpoint on CPU and check
  fixed-input logits against the GPU output within atol/rtol 1e-6.

The GPU-side module training.gpu_model is imported only for checkpoint and
prediction helpers; its import closure is torch/NumPy/stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from training import development_manifest, gpu_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APL_ROOT = PROJECT_ROOT / "APL"

ESM_MODEL_ID = "facebook/esm2_t6_8M_UR50D"
ESM_EMBEDDING_DIM = 320
MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048
INPUT_DIM = MORGAN_NUM_BITS + ESM_EMBEDDING_DIM

PURPOSE = "engineering_development_tracer"
PURPOSE_STATEMENT = (
    "engineering-development tracer fit diagnostics only; no independent "
    "holdout, no AUC/probability calibration/ranking or generalisation claim"
)
PURPOSE_DEVELOPMENT = "engineering_development_fit_diagnostic"
PURPOSE_DEVELOPMENT_STATEMENT = (
    "manifest-bound engineering-development in-sample fit diagnostics only; no "
    "independent holdout, no AUC/probability calibration/ranking or "
    "generalisation claim"
)
LABEL_MAPPING = {"inactive": 0, "active": 1}
EXCLUSION_RULES_VERSION = "phase03-tracer-eligibility-v1"
FIT_DIAGNOSTIC_PER_CLASS = 8
FIT_DIAGNOSTIC_SAMPLE_SEED = 42

EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "46d13b05db30c15e39cc4b455a5ab3ec1a0faf6265017b32301b08212da0c1c5"
)
GPU_PYTHON_DEFAULT = "E:/paper/DDA-Reproduction/.conda/python.exe"
RUN_ROOT_PATTERN = re.compile(r"P1_03_Artifacts_[0-9a-f]{32}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ALLCLOSE_ATOL = 1e-6
ALLCLOSE_RTOL = 1e-6

assert INPUT_DIM == gpu_model.INPUT_DIM


# ---------------------------------------------------------------------------
# File/hash helpers
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


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(name + "_missing:" + str(path))
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(name + "_not_mapping:" + str(path))
    return value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("json_already_exists:" + str(path))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return sha256_file(path)


def _write_jsonl_new(path: Path, rows: list[Mapping[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("jsonl_already_exists:" + str(path))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False))
            stream.write("\n")
    return sha256_file(path)


def _write_text_new(path: Path, text: str) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("text_already_exists:" + str(path))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return sha256_file(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Frozen input validation
# ---------------------------------------------------------------------------

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
        if not isinstance(value, str) or not value:
            raise ValueError(label + "_record_missing_" + key + ":" + str(position))
        if value in index:
            raise ValueError(label + "_duplicate_" + key + ":" + value)
        index[value] = record
    return index


def load_frozen_inputs(manifest_path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load and cross-check the frozen corpus manifest plus its entities bundle."""
    path = Path(os.path.expanduser(os.fspath(manifest_path)))
    if not path.is_absolute():
        raise ValueError("manifest_path_must_be_absolute")
    if not path.is_file():
        raise ValueError("frozen_manifest_missing:" + str(path))
    manifest = _load_json_object(path, "frozen_manifest")
    manifest_sha256 = sha256_file(path)
    if type(manifest.get("schema_version")) is not int:
        raise ValueError("frozen_manifest_schema_version_invalid")
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("frozen_manifest_split_missing")
    assignments = split.get("pair_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("frozen_manifest_pair_assignments_missing")
    entity_meta = manifest.get("entities")
    if not isinstance(entity_meta, Mapping):
        raise ValueError("frozen_manifest_entities_missing")
    entities_relative = entity_meta.get("path")
    if not isinstance(entities_relative, str) or not entities_relative:
        raise ValueError("frozen_manifest_entities_path_invalid")
    relative_path = Path(entities_relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("frozen_manifest_entities_path_not_relative")
    entities_path = (path.parent / relative_path).resolve(strict=False)
    if not entities_path.is_file():
        raise ValueError("frozen_entities_missing:" + str(entities_path))
    entities = _load_json_object(entities_path, "frozen_entities")
    entities_sha256 = sha256_file(entities_path)
    expected_entities_sha256 = _require_sha256(
        entity_meta.get("file_sha256"), "entities_file_sha256",
    )
    if expected_entities_sha256 != entities_sha256:
        raise ValueError("entities_file_sha256_mismatch")
    content_sha256 = _require_sha256(entity_meta.get("content_sha256"), "entity_content_sha256")
    entity_version = entity_meta.get("entity_version")
    if not isinstance(entity_version, str) or not entity_version:
        raise ValueError("frozen_manifest_entity_version_missing")
    if entities.get("content_sha256") != content_sha256:
        raise ValueError("entities_content_sha256_mismatch")
    if entities.get("entity_version") != entity_version:
        raise ValueError("entities_entity_version_mismatch")
    if split.get("entity_bundle_hash") != content_sha256:
        raise ValueError("frozen_split_entity_bundle_hash_mismatch")
    if split.get("entity_version") != entity_version:
        raise ValueError("frozen_split_entity_version_mismatch")
    for key in ("label_policy_sha256", "split_policy_sha256", "split_sha256"):
        value = split.get(key)
        if value is not None and not _is_sha256(value):
            raise ValueError("frozen_split_" + key + "_invalid")
    for label in ("molecules", "proteins"):
        if not isinstance(entities.get(label), list):
            raise ValueError("frozen_entities_" + label + "_missing")
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_sha256": manifest_sha256,
        "entities": entities,
        "entities_path": entities_path,
        "entities_sha256": entities_sha256,
        "content_sha256": content_sha256,
        "entity_version": entity_version,
        "assignments": assignments,
        "molecules": entities["molecules"],
        "proteins": entities["proteins"],
        "split": split,
    }


def load_feature_manifest(
    feature_manifest_path: os.PathLike[str] | str,
    frozen: Mapping[str, Any],
    *,
    expected_sha256: Any = None,
) -> dict[str, Any]:
    """Load and cross-check the selected ready-ESM snapshot against the frozen bundle."""
    path = Path(os.path.expanduser(os.fspath(feature_manifest_path)))
    if not path.is_absolute():
        raise ValueError("feature_manifest_path_must_be_absolute")
    if not path.is_file():
        raise ValueError("feature_manifest_missing:" + str(path))
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("feature_manifest_sha256_mismatch")
    manifest = _load_json_object(path, "feature_manifest")
    if manifest.get("bundle_file_sha256") != frozen["entities_sha256"]:
        raise ValueError("feature_bundle_file_sha256_mismatch")
    if manifest.get("entity_bundle_hash") != frozen["content_sha256"]:
        raise ValueError("feature_entity_bundle_hash_mismatch")
    if manifest.get("entity_version") != frozen["entity_version"]:
        raise ValueError("feature_entity_version_mismatch")
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("feature_model_metadata_missing")
    if model.get("model_id") != ESM_MODEL_ID:
        raise ValueError("feature_model_id_mismatch:" + repr(model.get("model_id")))
    revision = model.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("feature_model_revision_missing")
    if model.get("external_revision_verified") is not False:
        raise ValueError("feature_external_revision_claim")
    embedding_dim = model.get("embedding_dim")
    if type(embedding_dim) is not int or embedding_dim != ESM_EMBEDDING_DIM:
        raise ValueError("feature_manifest_embedding_dim_mismatch:" + repr(embedding_dim))
    rows = manifest.get("entities")
    if not isinstance(rows, list) or not rows:
        raise ValueError("feature_entities_missing")
    index: dict[tuple[str, int], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("feature_row_not_mapping:" + str(position))
        accession = row.get("accession")
        taxonomy_id = row.get("taxonomy_id")
        if not isinstance(accession, str) or not accession:
            raise ValueError("feature_row_accession_invalid:" + str(position))
        if type(taxonomy_id) is not int:
            raise ValueError("feature_row_taxonomy_invalid:" + str(position))
        key = (accession, taxonomy_id)
        if key in index:
            raise ValueError("feature_entity_duplicate:" + accession)
        for field, expected in (
            ("entity_bundle_hash", frozen["content_sha256"]),
            ("bundle_file_sha256", frozen["entities_sha256"]),
            ("entity_version", frozen["entity_version"]),
        ):
            if row.get(field) != expected:
                raise ValueError("feature_row_" + field + "_mismatch:" + accession)
        if row.get("embedding_dim") != embedding_dim:
            raise ValueError("feature_row_embedding_dim_mismatch:" + accession)
        status = row.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError("feature_row_status_invalid:" + accession)
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "ready":
            if row.get("finite") is not True:
                raise ValueError("feature_ready_row_not_finite:" + accession)
            if row.get("test_provenance") is not None:
                raise ValueError("feature_test_only_row_rejected:" + accession)
            features = row.get("features")
            if not isinstance(features, Mapping):
                raise ValueError("feature_ready_row_features_missing:" + accession)
            embedding = features.get("esm_embedding")
            if not isinstance(embedding, list) or len(embedding) != embedding_dim:
                raise ValueError("feature_embedding_dim_mismatch:" + accession)
            for value in embedding:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("feature_embedding_not_numeric:" + accession)
                if not math.isfinite(float(value)):
                    raise ValueError("feature_embedding_non_finite:" + accession)
            _require_sha256(row.get("cache_key"), "feature_cache_key:" + accession)
            _require_sha256(row.get("cache_file_sha256"), "feature_cache_file_sha256:" + accession)
            _require_sha256(row.get("sequence_sha256"), "feature_sequence_sha256:" + accession)
        index[key] = row
    return {
        "manifest": manifest,
        "path": path,
        "sha256": actual_sha256,
        "model": dict(model),
        "rows": rows,
        "index": index,
        "status_counts": status_counts,
        "embedding_dim": embedding_dim,
        "revision": revision,
    }


# ---------------------------------------------------------------------------
# Pair rows and arrays
# ---------------------------------------------------------------------------

def _fingerprint_vector(
    extractor: Any, canonical_smiles: Any, pair_key: str,
) -> tuple[np.ndarray | None, str | None]:
    if not isinstance(canonical_smiles, str) or not canonical_smiles.strip():
        return None, "missing_canonical_smiles"
    if not extractor.load_from_smiles(canonical_smiles, name=pair_key):
        return None, "smiles_parse_failed"
    fingerprints = extractor.compute_fingerprints()
    morgan = fingerprints.get("morgan") if isinstance(fingerprints, Mapping) else None
    if not isinstance(morgan, np.ndarray):
        return None, "morgan_fingerprint_failed"
    if morgan.shape != (MORGAN_NUM_BITS,):
        return None, "morgan_fingerprint_dim_mismatch"
    if not np.issubdtype(morgan.dtype, np.integer):
        return None, "morgan_fingerprint_not_integer"
    bits = morgan.astype(np.int64)
    if not bool(np.isin(bits, np.array([0, 1], dtype=np.int64)).all()):
        return None, "morgan_fingerprint_not_binary"
    return bits.astype(np.float32), None


def build_pair_rows(
    frozen: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    extractor: Any,
) -> dict[str, Any]:
    """Join train assignments to verified entities and ready frozen ESM features."""
    molecule_index = _unique_index(frozen["molecules"], "molecule_chembl_id", "molecule")
    protein_index = _unique_index(frozen["proteins"], "accession", "protein")
    feature_index = feature_manifest["index"]

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    skipped_non_train = 0
    seen_pair_keys: set[str] = set()

    for position, assignment in enumerate(frozen["assignments"]):
        if not isinstance(assignment, Mapping):
            raise ValueError("pair_assignment_not_mapping:" + str(position))
        pair_key = assignment.get("pair_key")
        molecule_id, taxonomy_id, accession = parse_pair_key(pair_key)
        if pair_key in seen_pair_keys:
            raise ValueError("duplicate_pair_key:" + pair_key)
        seen_pair_keys.add(pair_key)
        split = assignment.get("split")
        if split != "train":
            skipped_non_train += 1
            continue
        labels = assignment.get("labels")
        if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in LABEL_MAPPING:
            raise ValueError("pair_label_not_unique_binary:" + pair_key)
        label_name = labels[0]
        label = LABEL_MAPPING[label_name]
        evidence_ids = assignment.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            evidence_ids = []

        reason: str | None = None
        molecule = molecule_index.get(molecule_id)
        protein = protein_index.get(accession)
        feature_row = feature_index.get((accession, taxonomy_id))
        sequence_sha256: str | None = None
        if molecule is None or protein is None:
            reason = "entity_missing"
        elif protein.get("taxonomy_id") != taxonomy_id:
            raise ValueError("pair_taxonomy_mismatch:" + pair_key)
        elif molecule.get("structure_status") != "verified":
            reason = "molecule_structure_not_verified"
        elif protein.get("sequence_status") != "verified":
            reason = "protein_sequence_not_verified"
        elif protein.get("identity_status") != "verified":
            reason = "protein_identity_not_verified"
        else:
            sequence_sha256 = _require_sha256(
                protein.get("sequence_sha256"), "protein_sequence_sha256:" + pair_key,
            )
            if not isinstance(protein.get("full_sequence"), str) or not protein["full_sequence"]:
                reason = "protein_sequence_missing"
            elif feature_row is None:
                reason = "feature_row_missing"
            elif feature_row.get("status") != "ready":
                reason = "feature_status_" + str(feature_row.get("status"))
            elif feature_row.get("sequence_sha256") != sequence_sha256:
                raise ValueError("feature_sequence_sha256_mismatch:" + pair_key)

        morgan: np.ndarray | None = None
        embedding: np.ndarray | None = None
        if reason is None:
            morgan, fingerprint_reason = _fingerprint_vector(
                extractor, molecule.get("canonical_smiles"), pair_key,
            )
            if fingerprint_reason is not None:
                reason = fingerprint_reason
            else:
                embedding = np.asarray(
                    feature_row["features"]["esm_embedding"], dtype=np.float32,
                )
                if embedding.shape != (ESM_EMBEDDING_DIM,) or not bool(np.isfinite(embedding).all()):
                    raise ValueError("feature_embedding_invalid:" + pair_key)

        if reason is not None:
            exclusions.append({
                "pair_key": pair_key,
                "molecule_chembl_id": molecule_id,
                "accession": accession,
                "taxonomy_id": taxonomy_id,
                "label": label,
                "label_name": label_name,
                "reason": reason,
            })
            continue

        rows.append({
            "pair_key": pair_key,
            "molecule_chembl_id": molecule_id,
            "taxonomy_id": taxonomy_id,
            "accession": accession,
            "label": label,
            "label_name": label_name,
            "evidence_ids": [str(value) for value in evidence_ids],
            "sequence_sha256": sequence_sha256,
            "cache_key": feature_row["cache_key"],
            "cache_file_sha256": feature_row["cache_file_sha256"],
            "morgan": morgan,
            "esm_embedding": embedding,
        })

    rows.sort(key=lambda row: row["pair_key"])
    exclusions.sort(key=lambda item: item["pair_key"])
    if not rows:
        raise ValueError("no_eligible_pairs")
    class_counts = {
        "inactive": sum(1 for row in rows if row["label"] == 0),
        "active": sum(1 for row in rows if row["label"] == 1),
    }
    return {
        "rows": rows,
        "exclusions": exclusions,
        "skipped_non_train": skipped_non_train,
        "class_counts": class_counts,
        "pair_count": len(rows),
    }


def load_training_arrays(path: os.PathLike[str] | str) -> tuple[np.ndarray, np.ndarray]:
    """Read the numeric NPZ with allow_pickle=False and validate it."""
    source = Path(os.path.expanduser(os.fspath(path)))
    if not source.is_file():
        raise ValueError("training_arrays_missing:" + str(source))
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays.files) != {"features", "labels"}:
            raise ValueError("arrays_keys_invalid:" + ",".join(sorted(arrays.files)))
        features = arrays["features"]
        labels = arrays["labels"]
    return gpu_model.validate_arrays(features, labels, expected_dim=INPUT_DIM)


def prepare_pair_arrays(
    *,
    manifest_path: os.PathLike[str] | str,
    feature_manifest_path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    expected_feature_manifest_sha256: Any = None,
) -> dict[str, Any]:
    """Create a new run directory with source-bound arrays and JSON metadata."""
    out_root = Path(os.path.expanduser(os.fspath(out_dir)))
    if not out_root.is_absolute():
        raise ValueError("out_dir_must_be_absolute")
    if out_root.exists() or out_root.is_symlink():
        raise FileExistsError("output_directory_already_exists:" + str(out_root))
    if not out_root.parent.is_dir():
        raise ValueError("out_parent_missing:" + str(out_root.parent))
    out_root.mkdir()
    preprocess_dir = out_root / "preprocess"
    preprocess_dir.mkdir()

    frozen = load_frozen_inputs(manifest_path)
    feature_manifest = load_feature_manifest(
        feature_manifest_path,
        frozen,
        expected_sha256=expected_feature_manifest_sha256,
    )

    from ligand_features import LigandFeatureExtractor  # noqa: PLC0415 - CPU-only dependency

    extractor = LigandFeatureExtractor(output_dir=str(preprocess_dir))
    result = build_pair_rows(frozen, feature_manifest, extractor)
    rows = result["rows"]

    matrix = np.stack([
        np.concatenate([row["morgan"], row["esm_embedding"]])
        for row in rows
    ]).astype(np.float32)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    if matrix.shape != (len(rows), INPUT_DIM):
        raise RuntimeError("combined_feature_shape_mismatch:" + str(matrix.shape))
    if not bool(np.isfinite(matrix).all()):
        raise RuntimeError("combined_feature_non_finite")

    arrays_path = out_root / "training_arrays.npz"
    np.savez_compressed(arrays_path, features=matrix, labels=labels)
    reloaded_features, reloaded_labels = load_training_arrays(arrays_path)
    if not np.array_equal(reloaded_features, matrix) or not np.array_equal(reloaded_labels, labels):
        raise RuntimeError("training_arrays_roundtrip_mismatch")
    arrays_sha256 = sha256_file(arrays_path)

    pair_records = []
    for row_index, row in enumerate(rows):
        pair_records.append({
            "row_index": row_index,
            "pair_key": row["pair_key"],
            "molecule_chembl_id": row["molecule_chembl_id"],
            "taxonomy_id": row["taxonomy_id"],
            "accession": row["accession"],
            "label": row["label"],
            "label_name": row["label_name"],
            "evidence_ids": row["evidence_ids"],
            "sequence_sha256": row["sequence_sha256"],
            "cache_key": row["cache_key"],
            "cache_file_sha256": row["cache_file_sha256"],
            "morgan_bit_count": int(row["morgan"].sum()),
        })

    split = frozen["split"]
    input_manifest = {
        "schema_version": 1,
        "mode": "frozen-v1-train-tracer",
        "purpose": PURPOSE,
        "purpose_statement": PURPOSE_STATEMENT,
        "created_at": _utc_now(),
        "source": {
            "frozen_manifest_path": str(frozen["manifest_path"]),
            "frozen_manifest_sha256": frozen["manifest_sha256"],
            "entities_path": str(frozen["entities_path"]),
            "entities_sha256": frozen["entities_sha256"],
            "entity_bundle_hash": frozen["content_sha256"],
            "entity_version": frozen["entity_version"],
            "feature_manifest_path": str(feature_manifest["path"]),
            "feature_manifest_sha256": feature_manifest["sha256"],
            "feature_status_counts": dict(feature_manifest["status_counts"]),
            "esm_model_id": feature_manifest["model"].get("model_id"),
            "esm_model_revision": feature_manifest["revision"],
            "external_revision_verified": feature_manifest["model"].get(
                "external_revision_verified",
            ),
            "embedding_dim": feature_manifest["embedding_dim"],
            "label_policy_sha256": split.get("label_policy_sha256"),
            "split_policy_sha256": split.get("split_policy_sha256"),
            "split_sha256": split.get("split_sha256"),
            "frozen_split_training_ready": split.get("training_ready"),
            "train_pair_assignments": len(frozen["assignments"]),
            "morgan": {
                "radius": MORGAN_RADIUS,
                "num_bits": MORGAN_NUM_BITS,
                "dtype": "float32",
                "source": "LigandFeatureExtractor.compute_fingerprints()['morgan']",
            },
            "input_dim": INPUT_DIM,
            "fingerprint_dim": MORGAN_NUM_BITS,
            "protein_embedding_dim": ESM_EMBEDDING_DIM,
        },
        "arrays": {
            "file": "training_arrays.npz",
            "sha256": arrays_sha256,
            "bytes": arrays_path.stat().st_size,
            "dtype": "float32",
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "allow_pickle": False,
        },
        "label_mapping": dict(LABEL_MAPPING),
        "class_counts": dict(result["class_counts"]),
        "pair_count": result["pair_count"],
        "pairs": pair_records,
        "exclusions": result["exclusions"],
        "exclusion_count": len(result["exclusions"]),
        "exclusion_rules_version": EXCLUSION_RULES_VERSION,
        "skipped_non_train_assignments": result["skipped_non_train"],
    }
    input_manifest_path = out_root / "input_manifest.json"
    input_manifest_sha256 = _write_json_new(input_manifest_path, input_manifest)
    return {
        "out_dir": out_root,
        "arrays_path": arrays_path,
        "input_manifest_path": input_manifest_path,
        "arrays_sha256": arrays_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "pair_count": result["pair_count"],
        "class_counts": dict(result["class_counts"]),
        "pairs": pair_records,
        "exclusions": result["exclusions"],
        "exclusion_count": len(result["exclusions"]),
        "source": input_manifest["source"],
        "feature_status_counts": dict(feature_manifest["status_counts"]),
    }


# ---------------------------------------------------------------------------
# Real two-process tracer
# ---------------------------------------------------------------------------

def resolve_run_root(out_dir: os.PathLike[str] | str) -> Path:
    """Require a brand-new P1_03_Artifacts_<32 hex> directory under the project root."""
    unresolved = Path(os.path.expanduser(os.fspath(out_dir)))
    if not unresolved.is_absolute():
        raise ValueError("run_root_must_be_absolute")
    requested = Path(os.path.abspath(unresolved))
    if RUN_ROOT_PATTERN.fullmatch(requested.name) is None:
        raise ValueError("run_root_name_must_be_P1_03_Artifacts_<32 hex>")
    resolved = requested.resolve(strict=False)
    if resolved.parent != PROJECT_ROOT:
        raise ValueError("run_root_must_be_direct_child_of_project_root")
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("run_root_already_exists:" + str(requested))
    return requested


def validate_gpu_python(gpu_python: os.PathLike[str] | str) -> Path:
    """Pin the GPU interpreter to the registered DDA Conda interpreter."""
    candidate = Path(os.path.expanduser(os.fspath(gpu_python)))
    if not candidate.is_absolute() or not candidate.is_file():
        raise ValueError("gpu_python_missing:" + str(candidate))
    resolved = candidate.resolve()
    registered = Path(GPU_PYTHON_DEFAULT).resolve()
    if resolved != registered:
        raise ValueError("gpu_python_must_be_registered_cuda_interpreter:" + str(resolved))
    if resolved == Path(sys.executable).resolve():
        raise ValueError("gpu_python_must_not_be_the_current_test_interpreter")
    return resolved


def _git_code_version() -> dict[str, Any]:
    try:
        commit_run = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(APL_ROOT), capture_output=True, text=True, timeout=30,
        )
        if commit_run.returncode != 0:
            return {"git_commit": None, "reason": "git_rev_parse_exit_" + str(commit_run.returncode)}
        commit = commit_run.stdout.strip()
        if SHA256_PATTERN.fullmatch(commit) is None or len(commit) != 40:
            return {"git_commit": None, "reason": "git_commit_format_invalid"}
        status_run = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(APL_ROOT), capture_output=True, text=True, timeout=30,
        )
        dirty = None
        if status_run.returncode == 0:
            dirty = [
                line.strip() for line in status_run.stdout.splitlines() if line.strip()
            ]
        return {"git_commit": commit, "dirty_paths": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"git_commit": None, "reason": type(exc).__name__}


# ---------------------------------------------------------------------------
# Shared GPU subprocess / artefact validation helpers
# ---------------------------------------------------------------------------

def _gpu_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(APL_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _gpu_command(
    gpu_interpreter: Path,
    prepared: Mapping[str, Any],
    params: Mapping[str, Any],
) -> list[str]:
    return [
        str(gpu_interpreter), "-B", "-X", "utf8", "-m", "training.gpu_model",
        "--arrays", str(prepared["arrays_path"]),
        "--out", str(prepared["out_dir"]),
        "--seed", str(params["seed"]),
        "--steps", str(params["steps"]),
        "--batch-size", str(params["batch_size"]),
        "--learning-rate", repr(params["learning_rate"]),
        "--hidden-dim", str(params["hidden_dim"]),
    ]


def _run_gpu_process(
    gpu_command: list[str],
    run_root: Path,
    *,
    failure_code: str = "gpu_training_failed",
    log_name: str = "gpu_process.log",
) -> dict[str, Any]:
    completed = subprocess.run(
        gpu_command,
        cwd=str(APL_ROOT),
        env=_gpu_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    process_log = (
        "command: " + json.dumps(gpu_command, ensure_ascii=False) + "\n"
        + "exit_code: " + str(completed.returncode) + "\n"
        + "--- stdout ---\n" + completed.stdout + "\n"
        + "--- stderr ---\n" + completed.stderr + "\n"
    )
    process_log_path = run_root / log_name
    process_log_sha256 = _write_text_new(process_log_path, process_log)
    if completed.returncode != 0:
        raise RuntimeError(
            failure_code + ":exit=" + str(completed.returncode)
            + ":" + completed.stderr.strip()[-2000:]
        )
    return {
        "command": gpu_command,
        "returncode": completed.returncode,
        "process_log_path": process_log_path,
        "process_log_sha256": process_log_sha256,
    }


def _validate_gpu_result(
    *,
    gpu_result: Mapping[str, Any],
    prepared: Mapping[str, Any],
    params: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    if gpu_result.get("status") != "trained" or gpu_result.get("device") != "cuda":
        raise RuntimeError("gpu_result_not_cuda_trained")
    runtime = gpu_result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("gpu_result_runtime_missing")
    if runtime.get("cuda_available") is not True:
        raise RuntimeError("gpu_result_cuda_not_available")
    tf32 = runtime.get("tf32")
    if (
        not isinstance(tf32, Mapping)
        or tf32.get("cudnn_allow_tf32") is not False
        or tf32.get("cuda_matmul_allow_tf32") is not False
    ):
        raise RuntimeError("gpu_tf32_not_disabled")
    if tf32.get("float32_dtype") != "torch.float32":
        raise RuntimeError("gpu_float32_not_recorded")
    if gpu_result.get("arrays_sha256") != prepared["arrays_sha256"]:
        raise RuntimeError("gpu_arrays_binding_mismatch")
    if gpu_result.get("input_manifest_sha256") != prepared["input_manifest_sha256"]:
        raise RuntimeError("gpu_input_manifest_binding_mismatch")
    training = gpu_result.get("training")
    if not isinstance(training, Mapping):
        raise RuntimeError("gpu_training_record_missing")
    trainable_parameter_names = training.get("trainable_parameter_names")
    if not isinstance(trainable_parameter_names, list) or not trainable_parameter_names:
        raise RuntimeError("gpu_trainable_parameter_names_missing")
    for name in trainable_parameter_names:
        if not isinstance(name, str) or not name:
            raise RuntimeError("gpu_trainable_parameter_names_invalid")
        lowered = name.lower()
        if "esm" in lowered or "backbone" in lowered:
            raise RuntimeError("gpu_frozen_esm_has_trainable_parameters:" + name)
    if training.get("frozen_feature_input") is not True:
        raise RuntimeError("gpu_frozen_feature_input_not_recorded")
    checkpoint_record = gpu_result.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise RuntimeError("gpu_checkpoint_record_missing")
    checkpoint_path = run_root / "checkpoint.pt"
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_record.get("sha256") != actual_checkpoint_sha256:
        raise RuntimeError("checkpoint_sha256_mismatch")
    loss_records = gpu_result.get("loss_records")
    if not isinstance(loss_records, list) or len(loss_records) != params["steps"]:
        raise RuntimeError("gpu_loss_record_count_mismatch")
    for record in loss_records:
        if not isinstance(record, Mapping):
            raise RuntimeError("gpu_loss_record_invalid")
        for field in ("loss", "gradient_norm", "parameter_delta"):
            value = record.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError("gpu_loss_record_non_finite:" + field)
    total_delta = gpu_result.get("total_parameter_delta")
    if (
        isinstance(total_delta, bool)
        or not isinstance(total_delta, (int, float))
        or not math.isfinite(float(total_delta))
        or float(total_delta) <= 0.0
    ):
        raise RuntimeError("gpu_trainable_head_did_not_update")
    if gpu_result.get("gpu_reload", {}).get("passed") is not True:
        raise RuntimeError("gpu_checkpoint_reload_not_confirmed")
    return {
        "runtime": dict(runtime),
        "tf32": dict(tf32),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": actual_checkpoint_sha256,
        "loss_records": loss_records,
        "total_delta": float(total_delta),
        "training": dict(training),
    }


def _cpu_portable_reload(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    prepared: Mapping[str, Any],
    gpu_result: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    cpu_tf32 = gpu_model.configure_runtime()
    cpu_runtime = gpu_model.cuda_runtime_report()
    payload = gpu_model.load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        expected_sha256=checkpoint_sha256,
        expect_input_sha256=prepared["arrays_sha256"],
    )
    if payload["step"] != params["steps"] or payload["seed"] != params["seed"]:
        raise RuntimeError("cpu_checkpoint_step_or_seed_mismatch")
    if payload["config"] != gpu_result.get("config"):
        raise RuntimeError("cpu_checkpoint_config_mismatch")
    if payload["input_metadata"]["pair_keys"] != [
        pair["pair_key"] for pair in prepared["pairs"]
    ]:
        raise RuntimeError("cpu_checkpoint_pair_key_binding_mismatch")
    features, labels = load_training_arrays(prepared["arrays_path"])
    cpu_model = gpu_model.build_model(payload["config"])
    cpu_model.load_state_dict(payload["model_state_dict"])
    cpu_logits = gpu_model.predict_arrays(cpu_model, features, device="cpu")
    gpu_logits = np.asarray(gpu_result["logits"], dtype=np.float32)
    if gpu_logits.shape != cpu_logits.shape:
        raise RuntimeError("cpu_gpu_logit_shape_mismatch")
    max_abs_diff = float(np.max(np.abs(cpu_logits - gpu_logits)))
    agreement = bool(np.allclose(
        cpu_logits, gpu_logits, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL,
    ))
    if not agreement:
        raise RuntimeError("cpu_portability_logit_mismatch:" + repr(max_abs_diff))
    return {
        "payload": payload,
        "cpu_tf32": cpu_tf32,
        "cpu_runtime": cpu_runtime,
        "cpu_logits": cpu_logits,
        "max_abs_diff": max_abs_diff,
        "agreement": agreement,
    }


def train_real_pairs(
    *,
    manifest_path: os.PathLike[str] | str,
    feature_manifest_path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    gpu_python: os.PathLike[str] | str = GPU_PYTHON_DEFAULT,
    seed: Any = 42,
    steps: Any = 1,
    batch_size: Any = gpu_model.DEFAULT_BATCH_SIZE,
    learning_rate: Any = gpu_model.DEFAULT_LR,
    hidden_dim: Any = gpu_model.DEFAULT_HIDDEN_DIM,
) -> dict[str, Any]:
    """CPU prepare -> CUDA subprocess train -> CPU portable reload tracer."""
    params = gpu_model.validate_training_params(
        seed=seed, steps=steps, batch_size=batch_size,
        learning_rate=learning_rate, hidden_dim=hidden_dim,
    )
    run_root = resolve_run_root(out_dir)
    gpu_interpreter = validate_gpu_python(gpu_python)

    prepared = prepare_pair_arrays(
        manifest_path=manifest_path,
        feature_manifest_path=feature_manifest_path,
        out_dir=run_root,
        expected_feature_manifest_sha256=EXPECTED_FEATURE_MANIFEST_SHA256,
    )

    gpu_command = _gpu_command(gpu_interpreter, prepared, params)
    process_log = _run_gpu_process(gpu_command, run_root)
    process_log_sha256 = process_log["process_log_sha256"]

    gpu_result_path = run_root / "gpu_result.json"
    gpu_result = _load_json_object(gpu_result_path, "gpu_result")
    if gpu_result.get("status") != "trained" or gpu_result.get("device") != "cuda":
        raise RuntimeError("gpu_result_not_cuda_trained")
    runtime = gpu_result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("gpu_result_runtime_missing")
    if runtime.get("cuda_available") is not True:
        raise RuntimeError("gpu_result_cuda_not_available")
    tf32 = runtime.get("tf32")
    if not isinstance(tf32, Mapping) or tf32.get("cudnn_allow_tf32") is not False             or tf32.get("cuda_matmul_allow_tf32") is not False:
        raise RuntimeError("gpu_tf32_not_disabled")
    if tf32.get("float32_dtype") != "torch.float32":
        raise RuntimeError("gpu_float32_not_recorded")
    if gpu_result.get("arrays_sha256") != prepared["arrays_sha256"]:
        raise RuntimeError("gpu_arrays_binding_mismatch")
    if gpu_result.get("input_manifest_sha256") != prepared["input_manifest_sha256"]:
        raise RuntimeError("gpu_input_manifest_binding_mismatch")
    checkpoint_record = gpu_result.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise RuntimeError("gpu_checkpoint_record_missing")
    checkpoint_path = run_root / "checkpoint.pt"
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_record.get("sha256") != actual_checkpoint_sha256:
        raise RuntimeError("checkpoint_sha256_mismatch")
    loss_records = gpu_result.get("loss_records")
    if not isinstance(loss_records, list) or len(loss_records) != params["steps"]:
        raise RuntimeError("gpu_loss_record_count_mismatch")
    for record in loss_records:
        if not isinstance(record, Mapping):
            raise RuntimeError("gpu_loss_record_invalid")
        for field in ("loss", "gradient_norm", "parameter_delta"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float))                     or not math.isfinite(float(value)):
                raise RuntimeError("gpu_loss_record_non_finite:" + field)
    total_delta = gpu_result.get("total_parameter_delta")
    if isinstance(total_delta, bool) or not isinstance(total_delta, (int, float))             or not math.isfinite(float(total_delta)) or float(total_delta) <= 0.0:
        raise RuntimeError("gpu_trainable_head_did_not_update")
    if gpu_result.get("gpu_reload", {}).get("passed") is not True:
        raise RuntimeError("gpu_checkpoint_reload_not_confirmed")

    # CPU portable reload of the exact same checkpoint bytes.
    cpu_tf32 = gpu_model.configure_runtime()
    cpu_runtime = gpu_model.cuda_runtime_report()
    payload = gpu_model.load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        expected_sha256=actual_checkpoint_sha256,
        expect_input_sha256=prepared["arrays_sha256"],
    )
    if payload["step"] != params["steps"] or payload["seed"] != params["seed"]:
        raise RuntimeError("cpu_checkpoint_step_or_seed_mismatch")
    if payload["config"] != gpu_result.get("config"):
        raise RuntimeError("cpu_checkpoint_config_mismatch")
    if payload["input_metadata"]["pair_keys"] != [pair["pair_key"] for pair in prepared["pairs"]]:
        raise RuntimeError("cpu_checkpoint_pair_key_binding_mismatch")
    features, labels = load_training_arrays(prepared["arrays_path"])
    cpu_model = gpu_model.build_model(payload["config"])
    cpu_model.load_state_dict(payload["model_state_dict"])
    cpu_logits = gpu_model.predict_arrays(cpu_model, features, device="cpu")
    gpu_logits = np.asarray(gpu_result["logits"], dtype=np.float32)
    if gpu_logits.shape != cpu_logits.shape:
        raise RuntimeError("cpu_gpu_logit_shape_mismatch")
    max_abs_diff = float(np.max(np.abs(cpu_logits - gpu_logits)))
    agreement = bool(np.allclose(
        cpu_logits, gpu_logits, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL,
    ))
    if not agreement:
        raise RuntimeError("cpu_portability_logit_mismatch:" + repr(max_abs_diff))

    prediction_rows = []
    for row_index, pair in enumerate(prepared["pairs"]):
        prediction_rows.append({
            "pair_key": pair["pair_key"],
            "row_index": row_index,
            "label": int(pair["label"]),
            "label_name": pair["label_name"],
            "logit": float(cpu_logits[row_index]),
            "device": "cpu",
            "checkpoint_sha256": actual_checkpoint_sha256,
            "atol": ALLCLOSE_ATOL,
            "rtol": ALLCLOSE_RTOL,
            "purpose": PURPOSE,
        })
    predictions_path = run_root / "predictions.jsonl"
    predictions_sha256 = _write_jsonl_new(predictions_path, prediction_rows)
    loss_path = run_root / "loss.jsonl"
    loss_sha256 = sha256_file(loss_path)

    run_record = {
        "schema_version": 1,
        "run_id": run_root.name,
        "created_at": _utc_now(),
        "purpose": PURPOSE,
        "purpose_statement": PURPOSE_STATEMENT,
        "inputs": {
            **prepared["source"],
            "arrays_path": str(prepared["arrays_path"]),
            "arrays_sha256": prepared["arrays_sha256"],
            "input_manifest_path": str(prepared["input_manifest_path"]),
            "input_manifest_sha256": prepared["input_manifest_sha256"],
            "pair_count": prepared["pair_count"],
            "class_counts": dict(prepared["class_counts"]),
            "exclusion_count": prepared["exclusion_count"],
            "exclusions": prepared["exclusions"],
        },
        "command": {
            "train_cli": [sys.executable, "-m", "training.pair_pipeline", "train"]
            + sys.argv[1:],
            "gpu_command": gpu_command,
            "gpu_python": str(gpu_interpreter),
        },
        "environment": {
            "preprocess_python": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "torch": cpu_runtime.get("torch"),
                "cuda_available": cpu_runtime.get("cuda_available"),
            },
            "gpu_runtime": dict(runtime),
        },
        "diagnostics": {
            "gpu_tf32": dict(tf32),
            "cpu_tf32": dict(cpu_tf32),
            "float32_dtype": "torch.float32",
            "cpu_cuda_available": cpu_runtime.get("cuda_available"),
        },
        "training": {
            **dict(gpu_result.get("training") or {}),
            "loss_records": loss_records,
            "initial_loss": gpu_result.get("initial_loss"),
            "final_loss": gpu_result.get("final_loss"),
            "total_parameter_delta": float(total_delta),
            "predicted_class_counts": gpu_result.get("training", {}).get(
                "predicted_class_counts",
            ),
        },
        "portability": {
            "gpu_load_map_location": "cuda",
            "cpu_load_map_location": "cpu",
            "gpu_reload_max_abs_diff": gpu_result.get("gpu_reload", {}).get("max_abs_diff"),
            "cpu_reload_max_abs_diff": max_abs_diff,
            "atol": ALLCLOSE_ATOL,
            "rtol": ALLCLOSE_RTOL,
            "passed": agreement,
        },
        "outputs": {
            "training_arrays.npz": prepared["arrays_sha256"],
            "input_manifest.json": prepared["input_manifest_sha256"],
            "checkpoint.pt": actual_checkpoint_sha256,
            "loss.jsonl": loss_sha256,
            "gpu_result.json": sha256_file(gpu_result_path),
            "predictions.jsonl": predictions_sha256,
            "gpu_process.log": process_log_sha256,
        },
        "data_use": {
            "evaluation": "none (no independent holdout)",
            "generalization_claims": "none",
            "fit_scope": "engineering tracer fit diagnostics only",
        },
        "code_version": _git_code_version(),
    }
    run_record_path = run_root / "run.json"
    _write_json_new(run_record_path, run_record)
    return {
        "status": "trained",
        "run_id": run_root.name,
        "out": str(run_root),
        "device": "cuda",
        "device_name": runtime.get("device_name"),
        "pair_count": prepared["pair_count"],
        "class_counts": dict(prepared["class_counts"]),
        "steps": params["steps"],
        "final_loss": gpu_result.get("final_loss"),
        "total_parameter_delta": float(total_delta),
        "cpu_reload_max_abs_diff": max_abs_diff,
        "checkpoint_sha256": actual_checkpoint_sha256,
        "predictions_path": str(predictions_path),
        "run_record_path": str(run_record_path),
    }


# ---------------------------------------------------------------------------
# Development-manifest (train-only) fit pipeline
# ---------------------------------------------------------------------------

def resolve_new_output_dir(out_dir: os.PathLike[str] | str) -> Path:
    """Require a brand-new absolute output directory outside the APL source tree."""
    unresolved = Path(os.path.expanduser(os.fspath(out_dir)))
    if not unresolved.is_absolute():
        raise ValueError("out_dir_must_be_absolute")
    requested = Path(os.path.abspath(unresolved))
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("output_directory_already_exists:" + str(requested))
    if not requested.parent.is_dir():
        raise ValueError("out_parent_missing:" + str(requested.parent))
    resolved = requested.resolve(strict=False)
    apl_root = APL_ROOT.resolve()
    if resolved == apl_root or str(resolved).startswith(str(apl_root) + os.sep):
        raise ValueError("output_dir_must_not_be_inside_APL_source")
    requested.mkdir()
    return requested


def select_fit_diagnostic_sample(
    pairs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    seed: Any = FIT_DIAGNOSTIC_SAMPLE_SEED,
    per_class: Any = FIT_DIAGNOSTIC_PER_CLASS,
) -> dict[str, Any]:
    """Deterministically select a class-balanced training-only diagnostic sample."""
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("fit_diagnostic_requires_pairs")
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid_fit_diagnostic_seed:" + repr(seed))
    if type(per_class) is not int or per_class < 1:
        raise ValueError("invalid_fit_diagnostic_per_class:" + repr(per_class))
    inactive = [pair for pair in pairs if pair.get("label") == 0]
    active = [pair for pair in pairs if pair.get("label") == 1]
    if not inactive or not active:
        raise ValueError("fit_diagnostic_requires_both_classes")
    generator = np.random.default_rng(seed)

    def _pick(records: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        order = generator.permutation(len(records)).tolist()
        return [records[index] for index in order[: min(per_class, len(records))]]

    selected = _pick(inactive) + _pick(active)
    selected.sort(key=lambda pair: pair["row_index"])
    return {
        "plan_version": 1,
        "seed": seed,
        "per_class": per_class,
        "sample_rows": [int(pair["row_index"]) for pair in selected],
        "pair_keys": [pair["pair_key"] for pair in selected],
        "labels": [int(pair["label"]) for pair in selected],
        "class_counts": {
            "inactive": sum(1 for pair in selected if pair["label"] == 0),
            "active": sum(1 for pair in selected if pair["label"] == 1),
        },
        "criterion": {
            "version": gpu_model.FIT_DIAGNOSTIC_CRITERION_VERSION,
            "max_steps": gpu_model.MAX_STEPS,
            "min_relative_bce_drop": gpu_model.FIT_DIAGNOSTIC_MIN_RELATIVE_LOSS_DROP,
            "min_correct_per_class": gpu_model.FIT_DIAGNOSTIC_MIN_CORRECT_PER_CLASS,
        },
    }


def _load_development_sources(dev: Mapping[str, Any]) -> dict[str, Any]:
    """Re-verify every source binding declared by a development manifest."""
    declared = dev["source"]
    sources = development_manifest.load_verified_sources(
        declared["frozen_manifest_path"],
        declared["entities_path"],
        declared["feature_manifest_path"],
        expected_feature_manifest_sha256=declared["feature_manifest_sha256"],
        expected_feature_manifest_path=declared["feature_manifest_path"],
    )
    for field, actual in (
        ("frozen_manifest_sha256", sources["manifest_sha256"]),
        ("entities_sha256", sources["entities_sha256"]),
        ("entity_bundle_hash", sources["content_sha256"]),
        ("entity_version", sources["entity_version"]),
        ("feature_manifest_sha256", sources["feature_manifest"]["sha256"]),
    ):
        if declared.get(field) != actual:
            raise ValueError("development_manifest_source_binding_mismatch:" + field)
    return sources


def prepare_development_pair_arrays(
    *,
    development_manifest_path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Materialise only the manifest's declared included rows with re-verified sources."""
    dev = development_manifest.load_development_manifest(development_manifest_path)
    if dev["status"] != "ready":
        raise ValueError(
            "development_manifest_training_blocked:"
            + ",".join(dev["blocking_reasons"])
        )
    sources = _load_development_sources(dev)

    declared_pairs = dev["pairs"]
    row_order = list(dev["row_order"])
    assignments_by_key: dict[str, Mapping[str, Any]] = {}
    for assignment in sources["assignments"]:
        if not isinstance(assignment, Mapping):
            continue
        pair_key = assignment.get("pair_key")
        if isinstance(pair_key, str) and pair_key not in assignments_by_key:
            assignments_by_key[pair_key] = assignment
    selected_assignments: list[Mapping[str, Any]] = []
    for pair_key in row_order:
        assignment = assignments_by_key.get(pair_key)
        if assignment is None:
            raise ValueError("development_manifest_pair_missing_from_source:" + pair_key)
        selected_assignments.append(assignment)

    out_root = resolve_new_output_dir(out_dir)
    preprocess_dir = out_root / "preprocess"
    preprocess_dir.mkdir()

    from ligand_features import LigandFeatureExtractor  # noqa: PLC0415 - CPU-only dependency

    extractor = LigandFeatureExtractor(output_dir=str(preprocess_dir))
    filtered_sources = dict(sources)
    filtered_sources["assignments"] = selected_assignments
    result = build_pair_rows(filtered_sources, sources["feature_manifest"], extractor)
    if result["exclusions"]:
        raise ValueError(
            "development_manifest_included_pair_became_ineligible:"
            + result["exclusions"][0]["pair_key"]
        )
    rows = result["rows"]
    if len(rows) != len(declared_pairs):
        raise ValueError("development_manifest_row_count_mismatch")
    if [row["pair_key"] for row in rows] != row_order:
        raise ValueError("development_manifest_row_order_mismatch")
    for row, declared_pair in zip(rows, declared_pairs):
        checks = (
            ("label", row["label"], declared_pair["label"]),
            ("label_name", row["label_name"], declared_pair["label_name"]),
            ("molecule_chembl_id", row["molecule_chembl_id"], declared_pair["molecule_chembl_id"]),
            ("taxonomy_id", row["taxonomy_id"], declared_pair["taxonomy_id"]),
            ("accession", row["accession"], declared_pair["accession"]),
            ("sequence_sha256", row["sequence_sha256"], declared_pair["feature_sequence_sha256"]),
            ("cache_key", row["cache_key"], declared_pair["feature_cache_key"]),
            ("cache_file_sha256", row["cache_file_sha256"], declared_pair["feature_cache_file_sha256"]),
        )
        for field, actual, expected in checks:
            if actual != expected:
                raise ValueError(
                    "development_manifest_pair_binding_mismatch:"
                    + row["pair_key"] + ":" + field
                )

    matrix = np.stack([
        np.concatenate([row["morgan"], row["esm_embedding"]])
        for row in rows
    ]).astype(np.float32)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    if matrix.shape != (len(rows), INPUT_DIM):
        raise RuntimeError("combined_feature_shape_mismatch:" + str(matrix.shape))
    if not bool(np.isfinite(matrix).all()):
        raise RuntimeError("combined_feature_non_finite")

    arrays_path = out_root / "training_arrays.npz"
    np.savez_compressed(arrays_path, features=matrix, labels=labels)
    reloaded_features, reloaded_labels = load_training_arrays(arrays_path)
    if not np.array_equal(reloaded_features, matrix) or not np.array_equal(reloaded_labels, labels):
        raise RuntimeError("training_arrays_roundtrip_mismatch")
    arrays_sha256 = sha256_file(arrays_path)

    pair_records = []
    for row_index, row in enumerate(rows):
        pair_records.append({
            "row_index": row_index,
            "pair_key": row["pair_key"],
            "molecule_chembl_id": row["molecule_chembl_id"],
            "taxonomy_id": row["taxonomy_id"],
            "accession": row["accession"],
            "label": row["label"],
            "label_name": row["label_name"],
            "evidence_ids": row["evidence_ids"],
            "sequence_sha256": row["sequence_sha256"],
            "cache_key": row["cache_key"],
            "cache_file_sha256": row["cache_file_sha256"],
        })
    fit_diagnostic = select_fit_diagnostic_sample(pair_records)

    source = {
        "frozen_manifest_path": str(sources["manifest_path"]),
        "frozen_manifest_sha256": sources["manifest_sha256"],
        "entities_path": str(sources["entities_path"]),
        "entities_sha256": sources["entities_sha256"],
        "entity_bundle_hash": sources["content_sha256"],
        "entity_version": sources["entity_version"],
        "feature_manifest_path": str(sources["feature_manifest"]["path"]),
        "feature_manifest_sha256": sources["feature_manifest"]["sha256"],
        "feature_status_counts": dict(sources["feature_manifest"]["status_counts"]),
        "esm_model_id": sources["feature_manifest"]["model"].get("model_id"),
        "esm_model_revision": sources["feature_manifest"]["revision"],
        "external_revision_verified": sources["feature_manifest"]["model"].get(
            "external_revision_verified",
        ),
        "embedding_dim": sources["feature_manifest"]["embedding_dim"],
        "label_policy_sha256": sources["split"].get("label_policy_sha256"),
        "split_policy_sha256": sources["split"].get("split_policy_sha256"),
        "split_sha256": sources["split"].get("split_sha256"),
        "frozen_split_training_ready": sources["split"].get("training_ready"),
        "frozen_pair_assignments": len(sources["assignments"]),
        "candidate_rule": dev["manifest"].get("candidate_rule"),
        "eligibility_rules_version": dev["manifest"].get(
            "eligibility_rules_version",
        ),
        "row_order": list(row_order),
        "morgan": {
            "radius": MORGAN_RADIUS,
            "num_bits": MORGAN_NUM_BITS,
            "dtype": "float32",
            "source": "LigandFeatureExtractor.compute_fingerprints()['morgan']",
        },
        "input_dim": INPUT_DIM,
        "fingerprint_dim": MORGAN_NUM_BITS,
        "protein_embedding_dim": ESM_EMBEDDING_DIM,
    }
    input_manifest = {
        "schema_version": 1,
        "mode": "development-manifest-fit-diagnostic",
        "purpose": PURPOSE_DEVELOPMENT,
        "purpose_statement": PURPOSE_DEVELOPMENT_STATEMENT,
        "created_at": _utc_now(),
        "development_manifest": {
            "path": str(dev["path"]),
            "sha256": dev["sha256"],
            "status": dev["status"],
            "counts": dict(dev["counts"]),
            "purpose": dev["manifest"].get("purpose"),
        },
        "source": source,
        "row_order": list(row_order),
        "arrays": {
            "file": "training_arrays.npz",
            "sha256": arrays_sha256,
            "bytes": arrays_path.stat().st_size,
            "dtype": "float32",
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "allow_pickle": False,
        },
        "label_mapping": dict(LABEL_MAPPING),
        "class_counts": dict(result["class_counts"]),
        "pair_count": result["pair_count"],
        "pairs": pair_records,
        "exclusions": [],
        "exclusion_count": 0,
        "exclusion_rules_version": EXCLUSION_RULES_VERSION,
        "skipped_non_train_assignments": 0,
        "sampling": gpu_model.SAMPLING_CLASS_BALANCED,
        "fit_diagnostic": fit_diagnostic,
    }
    input_manifest_path = out_root / "input_manifest.json"
    input_manifest_sha256 = _write_json_new(input_manifest_path, input_manifest)
    return {
        "out_dir": out_root,
        "arrays_path": arrays_path,
        "input_manifest_path": input_manifest_path,
        "arrays_sha256": arrays_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "pair_count": result["pair_count"],
        "class_counts": dict(result["class_counts"]),
        "pairs": pair_records,
        "exclusions": [],
        "exclusion_count": 0,
        "source": source,
        "feature_status_counts": dict(sources["feature_manifest"]["status_counts"]),
        "row_order": list(row_order),
        "fit_diagnostic": fit_diagnostic,
        "development_manifest_path": dev["path"],
        "development_manifest_sha256": dev["sha256"],
        "development_manifest_status": dev["status"],
        "development_manifest_counts": dict(dev["counts"]),
    }


def _validate_gpu_fit_diagnostic(
    fit_record: Any, plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(fit_record, Mapping):
        raise RuntimeError("gpu_fit_diagnostic_missing")
    status = fit_record.get("status")
    if status not in ("passed", "failed"):
        raise RuntimeError("gpu_fit_diagnostic_status_invalid:" + repr(status))
    reasons = fit_record.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise RuntimeError("gpu_fit_diagnostic_reasons_invalid")
    if status == "failed" and not reasons:
        raise RuntimeError("gpu_fit_diagnostic_failure_without_reason")
    if fit_record.get("criterion_version") != gpu_model.FIT_DIAGNOSTIC_CRITERION_VERSION:
        raise RuntimeError("gpu_fit_diagnostic_criterion_mismatch")
    small = fit_record.get("small_sample")
    if not isinstance(small, Mapping):
        raise RuntimeError("gpu_fit_diagnostic_small_sample_missing")
    if (
        small.get("sample_rows") != list(plan["sample_rows"])
        or small.get("sample_pair_keys") != list(plan["pair_keys"])
        or small.get("sample_labels") != list(plan["labels"])
    ):
        raise RuntimeError("gpu_fit_diagnostic_sample_binding_mismatch")
    if small.get("device") != "cuda":
        raise RuntimeError("gpu_fit_diagnostic_device_not_cuda")
    if small.get("sampling") != gpu_model.SAMPLING_CLASS_BALANCED:
        raise RuntimeError("gpu_fit_diagnostic_sampling_mismatch")
    criterion = small.get("criterion")
    if not isinstance(criterion, Mapping):
        raise RuntimeError("gpu_fit_diagnostic_criterion_missing")
    if criterion.get("criterion_version") != gpu_model.FIT_DIAGNOSTIC_CRITERION_VERSION:
        raise RuntimeError("gpu_fit_diagnostic_criterion_mismatch")
    if criterion.get("status") != small.get("status"):
        raise RuntimeError("gpu_fit_diagnostic_criterion_status_mismatch")
    if criterion.get("min_relative_bce_drop") != (
        gpu_model.FIT_DIAGNOSTIC_MIN_RELATIVE_LOSS_DROP
    ):
        raise RuntimeError("gpu_fit_diagnostic_criterion_mismatch")
    if criterion.get("min_correct_per_class") != (
        gpu_model.FIT_DIAGNOSTIC_MIN_CORRECT_PER_CLASS
    ):
        raise RuntimeError("gpu_fit_diagnostic_criterion_mismatch")
    full = fit_record.get("full_manifest")
    if not isinstance(full, Mapping):
        raise RuntimeError("gpu_fit_diagnostic_full_manifest_missing")
    for field in ("majority_collapse", "class_counts", "prediction_metrics"):
        if field not in full:
            raise RuntimeError("gpu_fit_diagnostic_full_manifest_missing:" + field)
    if not isinstance(full.get("majority_collapse"), bool):
        raise RuntimeError("gpu_fit_diagnostic_full_manifest_invalid")
    return dict(fit_record)


def train_development_pairs(
    *,
    development_manifest_path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    gpu_python: os.PathLike[str] | str = GPU_PYTHON_DEFAULT,
    seed: Any = 42,
    steps: Any = 1000,
    batch_size: Any = gpu_model.DEFAULT_BATCH_SIZE,
    learning_rate: Any = gpu_model.DEFAULT_LR,
    hidden_dim: Any = gpu_model.DEFAULT_HIDDEN_DIM,
) -> dict[str, Any]:
    """CPU manifest prep -> bounded CUDA diagnostic -> run record (no CPU fallback)."""
    params = gpu_model.validate_training_params(
        seed=seed, steps=steps, batch_size=batch_size,
        learning_rate=learning_rate, hidden_dim=hidden_dim,
    )
    gpu_interpreter = validate_gpu_python(gpu_python)
    prepared = prepare_development_pair_arrays(
        development_manifest_path=development_manifest_path,
        out_dir=out_dir,
    )
    run_root = prepared["out_dir"]

    gpu_command = _gpu_command(gpu_interpreter, prepared, params)
    process_log = _run_gpu_process(gpu_command, run_root)
    process_log_sha256 = process_log["process_log_sha256"]
    gpu_result_path = run_root / "gpu_result.json"
    gpu_result = _load_json_object(gpu_result_path, "gpu_result")
    validated = _validate_gpu_result(
        gpu_result=gpu_result, prepared=prepared, params=params, run_root=run_root,
    )
    runtime = validated["runtime"]
    tf32 = validated["tf32"]
    checkpoint_path = validated["checkpoint_path"]
    actual_checkpoint_sha256 = validated["checkpoint_sha256"]
    loss_records = validated["loss_records"]
    total_delta = validated["total_delta"]
    fit_record = _validate_gpu_fit_diagnostic(
        gpu_result.get("fit_diagnostic"), prepared["fit_diagnostic"],
    )
    portability = _cpu_portable_reload(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=actual_checkpoint_sha256,
        prepared=prepared,
        gpu_result=gpu_result,
        params=params,
    )
    payload = portability["payload"]
    cpu_tf32 = portability["cpu_tf32"]
    cpu_runtime = portability["cpu_runtime"]
    cpu_logits = portability["cpu_logits"]
    max_abs_diff = portability["max_abs_diff"]
    agreement = portability["agreement"]

    prediction_rows = []
    for row_index, pair in enumerate(prepared["pairs"]):
        prediction_rows.append({
            "pair_key": pair["pair_key"],
            "row_index": row_index,
            "label": int(pair["label"]),
            "label_name": pair["label_name"],
            "logit": float(cpu_logits[row_index]),
            "device": "cpu",
            "checkpoint_sha256": actual_checkpoint_sha256,
            "atol": ALLCLOSE_ATOL,
            "rtol": ALLCLOSE_RTOL,
            "purpose": PURPOSE_DEVELOPMENT,
            "development_manifest_sha256": prepared["development_manifest_sha256"],
        })
    predictions_path = run_root / "predictions.jsonl"
    predictions_sha256 = _write_jsonl_new(predictions_path, prediction_rows)
    loss_sha256 = sha256_file(run_root / "loss.jsonl")
    fit_diagnostic_sha256 = sha256_file(run_root / "fit_diagnostic.json")

    run_record = {
        "schema_version": 1,
        "status": "trained",
        "device": "cuda",
        "device_name": runtime.get("device_name"),
        "run_id": run_root.name,
        "created_at": _utc_now(),
        "purpose": PURPOSE_DEVELOPMENT,
        "purpose_statement": PURPOSE_DEVELOPMENT_STATEMENT,
        "inputs": {
            "mode": "development-manifest-fit-diagnostic",
            "development_manifest_path": str(prepared["development_manifest_path"]),
            "development_manifest_sha256": prepared["development_manifest_sha256"],
            "development_manifest_status": prepared["development_manifest_status"],
            "development_manifest_counts": dict(prepared["development_manifest_counts"]),
            "row_order": list(prepared["row_order"]),
            "source": dict(prepared["source"]),
            "arrays_path": str(prepared["arrays_path"]),
            "arrays_sha256": prepared["arrays_sha256"],
            "input_manifest_path": str(prepared["input_manifest_path"]),
            "input_manifest_sha256": prepared["input_manifest_sha256"],
            "pair_count": prepared["pair_count"],
            "class_counts": dict(prepared["class_counts"]),
            "exclusion_count": prepared["exclusion_count"],
            "exclusions": [],
        },
        "command": {
            "train_cli": [sys.executable, "-m", "training.pair_pipeline", "train"]
            + sys.argv[1:],
            "gpu_command": gpu_command,
            "gpu_python": str(gpu_interpreter),
        },
        "environment": {
            "preprocess_python": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "torch": cpu_runtime.get("torch"),
                "cuda_available": cpu_runtime.get("cuda_available"),
            },
            "gpu_runtime": dict(runtime),
        },
        "diagnostics": {
            "gpu_tf32": dict(tf32),
            "cpu_tf32": dict(cpu_tf32),
            "float32_dtype": "torch.float32",
            "cpu_cuda_available": cpu_runtime.get("cuda_available"),
        },
        "training": {
            **dict(gpu_result.get("training") or {}),
            "loss_records": loss_records,
            "initial_loss": gpu_result.get("initial_loss"),
            "final_loss": gpu_result.get("final_loss"),
            "total_parameter_delta": total_delta,
            "evaluation": gpu_result.get("evaluation"),
        },
        "fit_diagnostic": fit_record,
        "fit_diagnostic_sha256": fit_diagnostic_sha256,
        "fit_diagnostic_plan": dict(prepared["fit_diagnostic"]),
        "portability": {
            "gpu_load_map_location": "cuda",
            "cpu_load_map_location": "cpu",
            "gpu_reload_max_abs_diff": gpu_result.get("gpu_reload", {}).get("max_abs_diff"),
            "cpu_reload_max_abs_diff": max_abs_diff,
            "atol": ALLCLOSE_ATOL,
            "rtol": ALLCLOSE_RTOL,
            "passed": agreement,
        },
        "outputs": {
            "training_arrays.npz": prepared["arrays_sha256"],
            "input_manifest.json": prepared["input_manifest_sha256"],
            "checkpoint.pt": actual_checkpoint_sha256,
            "loss.jsonl": loss_sha256,
            "gpu_result.json": sha256_file(gpu_result_path),
            "predictions.jsonl": predictions_sha256,
            "gpu_process.log": process_log_sha256,
            "fit_diagnostic.json": fit_diagnostic_sha256,
        },
        "data_use": {
            "evaluation": "none (no independent holdout)",
            "generalization_claims": "none",
            "fit_scope": "engineering development in-sample fit diagnostics only",
            "checkpoint_resume": "not part of this plan",
        },
        "code_version": _git_code_version(),
    }
    run_record_path = run_root / "run.json"
    _write_json_new(run_record_path, run_record)
    return {
        "status": "trained",
        "run_id": run_root.name,
        "out": str(run_root),
        "device": "cuda",
        "device_name": runtime.get("device_name"),
        "pair_count": prepared["pair_count"],
        "class_counts": dict(prepared["class_counts"]),
        "steps": params["steps"],
        "fit_diagnostic_status": fit_record["status"],
        "final_loss": gpu_result.get("final_loss"),
        "total_parameter_delta": total_delta,
        "cpu_reload_max_abs_diff": max_abs_diff,
        "checkpoint_sha256": actual_checkpoint_sha256,
        "predictions_path": str(predictions_path),
        "run_record_path": str(run_record_path),
    }


# ---------------------------------------------------------------------------
# Standalone checkpoint prediction
# ---------------------------------------------------------------------------

PREDICTION_SCHEMA_VERSION = 1
PREDICTION_MODE = "standalone-checkpoint-prediction"


def _validate_prediction_pair_keys(pair_keys: Any) -> list[str]:
    if pair_keys is None:
        raise ValueError("predict_pair_keys_empty")
    if not isinstance(pair_keys, (list, tuple)):
        raise ValueError("predict_pair_keys_invalid")
    keys = list(pair_keys)
    if not keys:
        raise ValueError("predict_pair_keys_empty")
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("predict_pair_key_invalid")
    if len(set(keys)) != len(keys):
        raise ValueError("predict_pair_key_duplicate")
    return keys


def _model_id(config: Mapping[str, Any]) -> str:
    return (
        "pair-head-" + str(config["architecture"])
        + "-h" + str(config["hidden_dim"])
    )


def _data_id(development_manifest_sha256: str) -> str:
    return "development-manifest-" + development_manifest_sha256[:16]


def _validate_prediction_checkpoint_binding(
    payload: Mapping[str, Any],
    *,
    dev: Mapping[str, Any],
    feature_manifest_sha256: str,
    pair_keys: list[str],
) -> list[int]:
    metadata = payload["input_metadata"]
    if metadata["purpose"] != PURPOSE_DEVELOPMENT:
        raise ValueError(
            "predict_checkpoint_purpose_mismatch:" + str(metadata["purpose"])
        )
    if metadata.get("development_manifest_sha256") != dev["sha256"]:
        raise ValueError("predict_development_manifest_binding_mismatch")
    if metadata.get("feature_manifest_sha256") != feature_manifest_sha256:
        raise ValueError("predict_feature_manifest_binding_mismatch")
    training_pair_keys = metadata["pair_keys"]
    row_by_key: dict[str, int] = {}
    for row_index, key in enumerate(training_pair_keys):
        row_by_key.setdefault(key, row_index)
    declared_keys = set(dev["row_order"])
    training_rows: list[int] = []
    for key in pair_keys:
        if key not in declared_keys:
            raise ValueError("predict_pair_key_not_in_manifest:" + key)
        if key not in row_by_key:
            raise ValueError("predict_pair_key_not_in_checkpoint:" + key)
        training_rows.append(row_by_key[key])
    return training_rows


def _build_prediction_rows(
    *,
    sources: Mapping[str, Any],
    dev: Mapping[str, Any],
    pair_keys: list[str],
    extractor: Any,
) -> list[dict[str, Any]]:
    """Materialise only the requested rows, preserving the request order."""
    assignments_by_key: dict[str, Mapping[str, Any]] = {}
    for assignment in sources["assignments"]:
        if not isinstance(assignment, Mapping):
            continue
        pair_key = assignment.get("pair_key")
        if isinstance(pair_key, str) and pair_key not in assignments_by_key:
            assignments_by_key[pair_key] = assignment
    selected_assignments: list[Mapping[str, Any]] = []
    for pair_key in pair_keys:
        assignment = assignments_by_key.get(pair_key)
        if assignment is None:
            raise ValueError("predict_pair_missing_from_source:" + pair_key)
        selected_assignments.append(assignment)
    filtered_sources = dict(sources)
    filtered_sources["assignments"] = selected_assignments
    result = build_pair_rows(filtered_sources, sources["feature_manifest"], extractor)
    if result["exclusions"]:
        exclusion = result["exclusions"][0]
        raise ValueError(
            "predict_pair_ineligible:" + exclusion["pair_key"]
            + ":" + str(exclusion.get("reason")),
        )
    rows_by_key = {row["pair_key"]: row for row in result["rows"]}
    declared_by_key = {pair["pair_key"]: pair for pair in dev["pairs"]}
    ordered_rows: list[dict[str, Any]] = []
    for pair_key in pair_keys:
        row = rows_by_key.get(pair_key)
        if row is None:
            raise ValueError("predict_pair_row_missing:" + pair_key)
        declared_pair = declared_by_key.get(pair_key)
        if declared_pair is None:
            raise ValueError("predict_pair_missing_from_manifest:" + pair_key)
        checks = (
            ("label", row["label"], declared_pair["label"]),
            ("label_name", row["label_name"], declared_pair["label_name"]),
            ("molecule_chembl_id", row["molecule_chembl_id"], declared_pair["molecule_chembl_id"]),
            ("taxonomy_id", row["taxonomy_id"], declared_pair["taxonomy_id"]),
            ("accession", row["accession"], declared_pair["accession"]),
            ("sequence_sha256", row["sequence_sha256"], declared_pair["feature_sequence_sha256"]),
            ("cache_key", row["cache_key"], declared_pair["feature_cache_key"]),
            ("cache_file_sha256", row["cache_file_sha256"], declared_pair["feature_cache_file_sha256"]),
        )
        for field, actual, expected in checks:
            if actual != expected:
                raise ValueError(
                    "predict_pair_manifest_binding_mismatch:" + pair_key + ":" + field
                )
        ordered_rows.append(row)
    return ordered_rows


def _prediction_feature_matrix(rows: list[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.stack([
        np.concatenate([row["morgan"], row["esm_embedding"]])
        for row in rows
    ]).astype(np.float32)
    if matrix.shape != (len(rows), INPUT_DIM):
        raise RuntimeError("prediction_feature_shape_mismatch:" + str(matrix.shape))
    if not bool(np.isfinite(matrix).all()):
        raise RuntimeError("prediction_feature_non_finite")
    return matrix


def _gpu_predict_command(
    gpu_interpreter: Path,
    *,
    checkpoint_path: Path,
    arrays_path: Path,
    request_path: Path,
    out_root: Path,
) -> list[str]:
    return [
        str(gpu_interpreter), "-B", "-X", "utf8", "-m", "training.gpu_model",
        "--predict-arrays", str(arrays_path),
        "--predict-request", str(request_path),
        "--checkpoint", str(checkpoint_path),
        "--out", str(out_root),
    ]


def _validate_gpu_prediction(
    gpu_prediction: Any,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    prediction_arrays_path: Path,
    prediction_arrays_sha256: str,
    pair_keys: list[str],
    training_row_indices: list[int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(gpu_prediction, Mapping):
        raise RuntimeError("gpu_prediction_missing")
    if gpu_prediction.get("status") != "predicted":
        raise RuntimeError("gpu_prediction_status_invalid")
    if gpu_prediction.get("device") != "cuda":
        raise RuntimeError("gpu_prediction_not_cuda")
    runtime = gpu_prediction.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("gpu_prediction_runtime_missing")
    if runtime.get("cuda_available") is not True:
        raise RuntimeError("gpu_prediction_cuda_not_available")
    if runtime.get("prediction_device") != "cuda":
        raise RuntimeError("gpu_prediction_device_not_cuda")
    tf32 = runtime.get("tf32")
    if (
        not isinstance(tf32, Mapping)
        or tf32.get("cudnn_allow_tf32") is not False
        or tf32.get("cuda_matmul_allow_tf32") is not False
    ):
        raise RuntimeError("gpu_prediction_tf32_not_disabled")
    if tf32.get("float32_dtype") != "torch.float32":
        raise RuntimeError("gpu_prediction_float32_not_recorded")
    if gpu_prediction.get("config") != dict(config):
        raise RuntimeError("gpu_prediction_model_config_mismatch")
    checkpoint_record = gpu_prediction.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise RuntimeError("gpu_prediction_checkpoint_record_missing")
    if checkpoint_record.get("sha256") != checkpoint_sha256:
        raise RuntimeError("gpu_prediction_checkpoint_sha256_mismatch")
    if Path(str(checkpoint_record.get("path"))).resolve(strict=False) != (
        checkpoint_path.resolve(strict=False)
    ):
        raise RuntimeError("gpu_prediction_checkpoint_path_mismatch")
    arrays_record = gpu_prediction.get("prediction_arrays")
    if not isinstance(arrays_record, Mapping):
        raise RuntimeError("gpu_prediction_arrays_record_missing")
    if arrays_record.get("sha256") != prediction_arrays_sha256:
        raise RuntimeError("gpu_prediction_arrays_sha256_mismatch")
    if Path(str(arrays_record.get("path"))).resolve(strict=False) != (
        prediction_arrays_path.resolve(strict=False)
    ):
        raise RuntimeError("gpu_prediction_arrays_path_mismatch")
    if gpu_prediction.get("pair_keys") != list(pair_keys):
        raise RuntimeError("gpu_prediction_pair_keys_mismatch")
    if gpu_prediction.get("training_row_indices") != list(training_row_indices):
        raise RuntimeError("gpu_prediction_training_row_indices_mismatch")
    logits = gpu_prediction.get("logits")
    if not isinstance(logits, list) or len(logits) != len(pair_keys):
        raise RuntimeError("gpu_prediction_logits_invalid")
    for value in logits:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError("gpu_prediction_logits_non_finite")
    return dict(gpu_prediction)


def _cpu_prediction_reload(
    payload: Mapping[str, Any],
    features: np.ndarray,
    gpu_logits: np.ndarray,
) -> dict[str, Any]:
    cpu_tf32 = gpu_model.configure_runtime()
    cpu_runtime = gpu_model.cuda_runtime_report()
    cpu_model = gpu_model.build_model(payload["config"])
    cpu_model.load_state_dict(payload["model_state_dict"])
    cpu_logits = gpu_model.predict_arrays(cpu_model, features, device="cpu")
    gpu_array = np.asarray(gpu_logits, dtype=np.float32)
    if gpu_array.shape != cpu_logits.shape:
        raise RuntimeError("cpu_gpu_prediction_shape_mismatch")
    max_abs_diff = float(np.max(np.abs(
        np.asarray(cpu_logits, dtype=np.float64)
        - np.asarray(gpu_array, dtype=np.float64)
    )))
    agreement = bool(np.allclose(
        cpu_logits, gpu_array, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL,
    ))
    if not agreement:
        raise RuntimeError("cpu_portability_prediction_logit_mismatch:" + repr(max_abs_diff))
    return {
        "cpu_tf32": cpu_tf32,
        "cpu_runtime": cpu_runtime,
        "cpu_logits": cpu_logits,
        "max_abs_diff": max_abs_diff,
        "agreement": agreement,
    }


def _import_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for module_name in ("torch", "numpy", "rdkit"):
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            versions[module_name] = {
                "version": str(version) if version is not None else None,
                "module_path": getattr(module, "__file__", None),
            }
        except Exception as exc:  # noqa: BLE001 - missing version is recorded, not guessed
            versions[module_name] = {
                "version": None,
                "module_path": None,
                "reason": type(exc).__name__ + ":" + str(exc),
            }
    return versions


def predict_pairs(
    *,
    development_manifest_path: os.PathLike[str] | str,
    checkpoint_path: os.PathLike[str] | str,
    pair_keys: Any,
    out_dir: os.PathLike[str] | str,
    gpu_python: os.PathLike[str] | str = GPU_PYTHON_DEFAULT,
    expected_checkpoint_sha256: Any = None,
) -> dict[str, Any]:
    """Predict requested development pairs from a trained checkpoint on CUDA.

    The CPU side only re-verifies the declared development sources, materialises
    the requested numeric rows and performs the portable reload check.  The new
    isolated CUDA process consumes the checkpoint plus that numeric batch and
    records every observed environment/hash.
    """
    requested = _validate_prediction_pair_keys(pair_keys)
    gpu_interpreter = validate_gpu_python(gpu_python)
    dev = development_manifest.load_development_manifest(development_manifest_path)
    if dev["status"] != "ready":
        raise ValueError(
            "development_manifest_prediction_blocked:"
            + ",".join(dev["blocking_reasons"])
        )
    sources = _load_development_sources(dev)
    feature_manifest_sha256 = sources["feature_manifest"]["sha256"]
    checkpoint_file = Path(os.path.expanduser(os.fspath(checkpoint_path)))
    if not checkpoint_file.is_absolute():
        raise ValueError("checkpoint_path_must_be_absolute")
    if not checkpoint_file.is_file():
        raise ValueError("checkpoint_missing:" + str(checkpoint_file))
    if expected_checkpoint_sha256 is not None:
        _require_sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    checkpoint_sha256 = sha256_file(checkpoint_file)
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint_sha256_mismatch")
    payload = gpu_model.load_checkpoint(
        checkpoint_file,
        map_location="cpu",
        expected_sha256=checkpoint_sha256,
    )
    training_row_indices = _validate_prediction_checkpoint_binding(
        payload,
        dev=dev,
        feature_manifest_sha256=feature_manifest_sha256,
        pair_keys=requested,
    )

    out_root = resolve_new_output_dir(out_dir)
    preprocess_dir = out_root / "preprocess"
    preprocess_dir.mkdir()
    from ligand_features import LigandFeatureExtractor  # noqa: PLC0415 - CPU-only dependency

    extractor = LigandFeatureExtractor(output_dir=str(preprocess_dir))
    rows = _build_prediction_rows(
        sources=sources,
        dev=dev,
        pair_keys=requested,
        extractor=extractor,
    )
    features = _prediction_feature_matrix(rows)
    arrays_path = out_root / "prediction_arrays.npz"
    np.savez_compressed(arrays_path, features=features)
    arrays_sha256 = sha256_file(arrays_path)
    request = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "purpose": PURPOSE_DEVELOPMENT,
        "pair_keys": list(requested),
        "training_row_indices": list(training_row_indices),
        "checkpoint_sha256": checkpoint_sha256,
        "training_arrays_sha256": payload["input_metadata"]["arrays_sha256"],
        "prediction_arrays_sha256": arrays_sha256,
        "development_manifest_sha256": dev["sha256"],
        "feature_manifest_sha256": feature_manifest_sha256,
    }
    request_path = out_root / "prediction_request.json"
    request_sha256 = _write_json_new(request_path, request)
    gpu_command = _gpu_predict_command(
        gpu_interpreter,
        checkpoint_path=checkpoint_file,
        arrays_path=arrays_path,
        request_path=request_path,
        out_root=out_root,
    )
    process_log = _run_gpu_process(
        gpu_command, out_root,
        failure_code="gpu_prediction_failed", log_name="gpu_process.log",
    )
    process_log_sha256 = process_log["process_log_sha256"]
    gpu_prediction_path = out_root / "gpu_prediction.json"
    gpu_prediction = _load_json_object(gpu_prediction_path, "gpu_prediction")
    validated = _validate_gpu_prediction(
        gpu_prediction,
        checkpoint_path=checkpoint_file,
        checkpoint_sha256=checkpoint_sha256,
        prediction_arrays_path=arrays_path,
        prediction_arrays_sha256=arrays_sha256,
        pair_keys=requested,
        training_row_indices=training_row_indices,
        config=payload["config"],
    )
    gpu_runtime = dict(validated["runtime"])
    gpu_logits = np.asarray(validated["logits"], dtype=np.float32)
    portability = _cpu_prediction_reload(payload, features, gpu_logits)
    cpu_logits = portability["cpu_logits"]
    cpu_tf32 = portability["cpu_tf32"]
    cpu_runtime = portability["cpu_runtime"]

    model_id = _model_id(payload["config"])
    data_id = _data_id(dev["sha256"])
    prediction_rows = []
    for index, pair_key in enumerate(requested):
        prediction_rows.append({
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "run_id": out_root.name,
            "pair_key": pair_key,
            "training_row_index": int(training_row_indices[index]),
            "model_id": model_id,
            "data_id": data_id,
            "purpose": PURPOSE_DEVELOPMENT,
            "purpose_class": "engineering-development",
            "purpose_statement": PURPOSE_DEVELOPMENT_STATEMENT,
            "engineering_development": True,
            "label": int(rows[index]["label"]),
            "label_name": rows[index]["label_name"],
            "logit": float(gpu_logits[index]),
            "cpu_logit": float(cpu_logits[index]),
            "device": "cuda",
            "checkpoint_sha256": checkpoint_sha256,
            "development_manifest_sha256": dev["sha256"],
            "feature_manifest_sha256": feature_manifest_sha256,
            "atol": ALLCLOSE_ATOL,
            "rtol": ALLCLOSE_RTOL,
        })
    predictions_path = out_root / "predictions.jsonl"
    predictions_sha256 = _write_jsonl_new(predictions_path, prediction_rows)
    gpu_prediction_sha256 = sha256_file(gpu_prediction_path)
    run_record = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "status": "predicted",
        "mode": PREDICTION_MODE,
        "run_id": out_root.name,
        "created_at": _utc_now(),
        "purpose": PURPOSE_DEVELOPMENT,
        "purpose_class": "engineering-development",
        "purpose_statement": PURPOSE_DEVELOPMENT_STATEMENT,
        "device": "cuda",
        "device_name": gpu_runtime.get("device_name"),
        "model": {
            "model_id": model_id,
            "config": dict(payload["config"]),
            "checkpoint_path": str(checkpoint_file),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": payload["step"],
            "checkpoint_global_step": payload["global_step"],
            "checkpoint_seed": payload["seed"],
            "checkpoint_purpose": payload["input_metadata"]["purpose"],
            "checkpoint_training_config": dict(payload["training_config"]),
            "checkpoint_optimizer_config": dict(payload["optimizer_config"]),
        },
        "data": {
            "data_id": data_id,
            "development_manifest_path": str(dev["path"]),
            "development_manifest_sha256": dev["sha256"],
            "development_manifest_status": dev["status"],
            "development_manifest_counts": dict(dev["counts"]),
            "frozen_manifest_path": str(sources["manifest_path"]),
            "frozen_manifest_sha256": sources["manifest_sha256"],
            "entities_path": str(sources["entities_path"]),
            "entities_sha256": sources["entities_sha256"],
            "feature_manifest_path": str(sources["feature_manifest"]["path"]),
            "feature_manifest_sha256": feature_manifest_sha256,
            "source": dict(dev["source"]),
        },
        "inputs": {
            "pair_keys": list(requested),
            "pair_count": len(requested),
            "training_row_indices": list(training_row_indices),
            "prediction_arrays_path": str(arrays_path),
            "prediction_arrays_sha256": arrays_sha256,
            "prediction_request_path": str(request_path),
            "prediction_request_sha256": request_sha256,
        },
        "command": {
            "predict_cli": [
                sys.executable, "-m", "training.pair_pipeline", "predict",
            ] + list(sys.argv[1:]),
            "gpu_command": gpu_command,
            "gpu_python": str(gpu_interpreter),
        },
        "environment": {
            "preprocess_python": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "imports": _import_versions(),
                "cuda_available": cpu_runtime.get("cuda_available"),
            },
            "gpu_runtime": gpu_runtime,
            "checkpoint_training_runtime": dict(payload["runtime"]),
        },
        "diagnostics": {
            "gpu_tf32": dict(gpu_runtime.get("tf32") or {}),
            "cpu_tf32": dict(cpu_tf32),
            "float32_dtype": "torch.float32",
            "cpu_cuda_available": cpu_runtime.get("cuda_available"),
        },
        "portability": {
            "gpu_load_map_location": "cpu",
            "cpu_load_map_location": "cpu",
            "max_abs_diff": portability["max_abs_diff"],
            "atol": ALLCLOSE_ATOL,
            "rtol": ALLCLOSE_RTOL,
            "passed": portability["agreement"],
        },
        "outputs": {
            "prediction_arrays.npz": arrays_sha256,
            "prediction_request.json": request_sha256,
            "gpu_prediction.json": gpu_prediction_sha256,
            "gpu_process.log": process_log_sha256,
            "predictions.jsonl": predictions_sha256,
        },
        "predictions": prediction_rows,
        "data_use": {
            "evaluation": "none (no independent holdout)",
            "generalization_claims": "none",
            "formal_metrics": "none",
            "scope": "engineering-development prediction only",
        },
        "code_version": _git_code_version(),
    }
    run_record_path = out_root / "run.json"
    _write_json_new(run_record_path, run_record)
    return {
        "status": "predicted",
        "run_id": out_root.name,
        "out": str(out_root),
        "device": "cuda",
        "device_name": gpu_runtime.get("device_name"),
        "pair_keys": list(requested),
        "logits": [float(value) for value in gpu_logits],
        "cpu_reload_max_abs_diff": portability["max_abs_diff"],
        "checkpoint_sha256": checkpoint_sha256,
        "predictions_path": str(predictions_path),
        "run_record_path": str(run_record_path),
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="training.pair_pipeline",
        description="Phase 03 pair tracer: CPU frozen-input prep and CUDA training.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    train = subcommands.add_parser("train")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="frozen corpus manifest.json (read-only)")
    source.add_argument(
        "--development-manifest",
        help="train-only development_manifest.json from training.development_manifest",
    )
    train.add_argument("--features", help="selected feature_manifest.json (read-only)")
    train.add_argument("--out", required=True, help="new P1_03_Artifacts_<32 hex> directory")
    train.add_argument("--gpu-python", default=GPU_PYTHON_DEFAULT)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=gpu_model.DEFAULT_BATCH_SIZE)
    train.add_argument("--learning-rate", type=float, default=gpu_model.DEFAULT_LR)
    train.add_argument("--hidden-dim", type=int, default=gpu_model.DEFAULT_HIDDEN_DIM)
    predict = subcommands.add_parser("predict")
    predict.add_argument(
        "--development-manifest",
        required=True,
        help="train-only development_manifest.json the checkpoint was bound to",
    )
    predict.add_argument("--checkpoint", required=True, help="checkpoint.pt to consume")
    predict.add_argument(
        "--pair-key",
        action="append",
        dest="pair_keys",
        default=None,
        help="development pair_key to predict (repeatable, order preserved)",
    )
    predict.add_argument("--out", required=True, help="new prediction output directory")
    predict.add_argument("--gpu-python", default=GPU_PYTHON_DEFAULT)
    predict.add_argument("--expected-checkpoint-sha256", default=None)
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            if args.development_manifest is not None:
                summary = train_development_pairs(
                    development_manifest_path=args.development_manifest,
                    out_dir=args.out,
                    gpu_python=args.gpu_python,
                    seed=args.seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    hidden_dim=args.hidden_dim,
                )
            else:
                if not args.features:
                    raise ValueError("features_argument_required_with_manifest")
                summary = train_real_pairs(
                    manifest_path=args.manifest,
                    feature_manifest_path=args.features,
                    out_dir=args.out,
                    gpu_python=args.gpu_python,
                    seed=args.seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    hidden_dim=args.hidden_dim,
                )
        elif args.command == "predict":
            summary = predict_pairs(
                development_manifest_path=args.development_manifest,
                checkpoint_path=args.checkpoint,
                pair_keys=args.pair_keys,
                out_dir=args.out,
                gpu_python=args.gpu_python,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            )
        else:  # pragma: no cover - argparse restricts the command set
            raise ValueError("unsupported_command:" + str(args.command))
    except Exception as exc:  # noqa: BLE001 - structured blocked report, no CPU fallback
        print(json.dumps({
            "status": "blocked",
            "reason": type(exc).__name__ + ":" + str(exc),
        }, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

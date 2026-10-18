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

from training import gpu_model


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
LABEL_MAPPING = {"inactive": 0, "active": 1}
EXCLUSION_RULES_VERSION = "phase03-tracer-eligibility-v1"

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

    gpu_command = [
        str(gpu_interpreter), "-B", "-X", "utf8", "-m", "training.gpu_model",
        "--arrays", str(prepared["arrays_path"]),
        "--out", str(run_root),
        "--seed", str(params["seed"]),
        "--steps", str(params["steps"]),
        "--batch-size", str(params["batch_size"]),
        "--learning-rate", repr(params["learning_rate"]),
        "--hidden-dim", str(params["hidden_dim"]),
    ]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(APL_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        gpu_command,
        cwd=str(APL_ROOT),
        env=environment,
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
    process_log_path = run_root / "gpu_process.log"
    process_log_sha256 = _write_text_new(process_log_path, process_log)
    if completed.returncode != 0:
        raise RuntimeError(
            "gpu_training_failed:exit=" + str(completed.returncode)
            + ":" + completed.stderr.strip()[-2000:]
        )

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
        help="engineering-development manifest (plan 03-02; not implemented here)",
    )
    train.add_argument("--features", help="selected feature_manifest.json (read-only)")
    train.add_argument("--out", required=True, help="new P1_03_Artifacts_<32 hex> directory")
    train.add_argument("--gpu-python", default=GPU_PYTHON_DEFAULT)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=gpu_model.DEFAULT_BATCH_SIZE)
    train.add_argument("--learning-rate", type=float, default=gpu_model.DEFAULT_LR)
    train.add_argument("--hidden-dim", type=int, default=gpu_model.DEFAULT_HIDDEN_DIM)
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            if args.development_manifest is not None:
                raise ValueError(
                    "development_manifest_mode_deferred_to_plan_03_02"
                )
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

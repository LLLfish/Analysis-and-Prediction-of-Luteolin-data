"""Fail-closed input contract tests for the Phase 03 development-fit pipeline.

Covers:

* gpu_model feature-block / label / pair-key validation before any tensor or
  PairHead call (stable error codes);
* NPZ + input_manifest rejection for object dtype, missing/extra keys,
  NaN/Inf, row-count/identity mismatches and illegal budget values;
* development-manifest consumption on CPU: only declared included rows are
  materialised, source hashes are re-verified, tampered/blocked manifests fail
  closed before the GPU subprocess;
* one synthetic end-to-end development fit run through the real CUDA
  interpreter (tiny fixture, engineering probe only - not real-data training).

The real-frozen-data 1000-step CUDA fit command is intentionally NOT executed
here; it stays deferred while the user's GPU gate is closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from training import development_manifest, gpu_model, pair_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPU_PYTHON = Path("E:/paper/DDA-Reproduction/.conda/python.exe")

ESM_DIM = gpu_model.INPUT_ESM_DIM
FP_DIM = gpu_model.INPUT_FP_DIM
INPUT_DIM = gpu_model.INPUT_DIM
CONTENT_SHA = "2" * 64
ENTITY_VERSION = "entities-v2-" + "2" * 16
MODEL_REVISION = "d" * 40


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _embedding(signal: float, seed: int) -> list:
    values = [signal]
    for offset in range(1, ESM_DIM):
        values.append(round(math.sin(seed + offset * 0.01) * 0.02, 8))
    return values


def _synthetic_features(labels, *, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    morgan = rng.integers(0, 2, size=(len(labels), FP_DIM)).astype(np.float32)
    esm = np.empty((len(labels), ESM_DIM), dtype=np.float32)
    for index, label in enumerate(labels):
        signal = 1.0 if int(label) == 1 else -1.0
        esm[index] = np.asarray(_embedding(signal, index), dtype=np.float32)
    return np.concatenate([morgan, esm], axis=1).astype(np.float32)


def _build_synthetic_dev_sources(root: Path) -> dict:
    """Eight source-bound pairs: four active, four inactive, both classes ready."""
    active_flags = [True, True, True, True, False, False, False, False]
    smiles = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCl", "CCBr", "CCF", "CC(=O)O"]
    molecules = []
    proteins = []
    feature_rows = []
    assignments = []
    for index, (is_active, smiles_value) in enumerate(zip(active_flags, smiles)):
        molecule_id = "CHEMBL92%04d" % (index + 1)
        accession = "P92%03d" % (index + 1)
        sequence = "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLL" + accession
        sequence_sha = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        molecules.append({
            "molecule_chembl_id": molecule_id,
            "canonical_smiles": smiles_value,
            "structure_status": "verified",
            "structure_sha256": hashlib.sha256(molecule_id.encode("utf-8")).hexdigest(),
        })
        proteins.append({
            "accession": accession,
            "taxonomy_id": 9606,
            "sequence_status": "verified",
            "identity_status": "verified",
            "full_sequence": sequence,
            "sequence_sha256": sequence_sha,
        })
        feature_rows.append({
            "accession": accession,
            "taxonomy_id": 9606,
            "sequence_length": len(sequence),
            "sequence_sha256": sequence_sha,
            "entity_version": ENTITY_VERSION,
            "entity_bundle_hash": CONTENT_SHA,
            "bundle_file_sha256": None,
            "model_id": development_manifest.ESM_MODEL_ID,
            "model_revision": MODEL_REVISION,
            "embedding_dim": ESM_DIM,
            "status": "ready",
            "reason": None,
            "finite": True,
            "test_provenance": None,
            "features": {"esm_embedding": _embedding(1.0 if is_active else -1.0, index)},
            "cache_key": hashlib.sha256(("cache:" + accession).encode("utf-8")).hexdigest(),
            "cache_file_sha256": hashlib.sha256(
                ("cache-file:" + accession).encode("utf-8")
            ).hexdigest(),
        })
        assignments.append({
            "pair_key": molecule_id + "|9606|" + accession,
            "labels": ["active" if is_active else "inactive"],
            "split": "train",
            "evidence_ids": ["ev-" + str(index + 1)],
        })
    entities = {
        "schema_version": 2,
        "content_sha256": CONTENT_SHA,
        "entity_version": ENTITY_VERSION,
        "molecules": molecules,
        "proteins": proteins,
    }
    entities_path = root / "frozen" / "entities.json"
    _dump_json(entities_path, entities)
    entities_sha = _sha256(entities_path)
    for row in feature_rows:
        row["bundle_file_sha256"] = entities_sha
    feature_manifest = {
        "schema_version": 1,
        "mode": "local_only_feature_snapshot",
        "status": "blocked",
        "bundle_file_sha256": entities_sha,
        "entity_bundle_hash": CONTENT_SHA,
        "entity_version": ENTITY_VERSION,
        "model": {
            "model_id": development_manifest.ESM_MODEL_ID,
            "revision": MODEL_REVISION,
            "external_revision_verified": False,
            "embedding_dim": ESM_DIM,
        },
        "entities": feature_rows,
        "status_counts": {"ready": len(feature_rows)},
    }
    feature_manifest_path = root / "features" / "feature_manifest.json"
    _dump_json(feature_manifest_path, feature_manifest)
    manifest = {
        "schema_version": 2,
        "entities": {
            "path": "entities.json",
            "file_sha256": entities_sha,
            "content_sha256": CONTENT_SHA,
            "entity_version": ENTITY_VERSION,
            "schema_version": 2,
        },
        "split": {
            "pair_assignments": list(reversed(assignments)),
            "entity_bundle_hash": CONTENT_SHA,
            "entity_version": ENTITY_VERSION,
            "label_policy_sha256": "b" * 64,
            "split_policy_sha256": "d" * 64,
            "split_sha256": "e" * 64,
            "training_ready": False,
        },
    }
    manifest_path = root / "frozen" / "manifest.json"
    _dump_json(manifest_path, manifest)
    return {
        "root": root,
        "manifest_path": manifest_path,
        "entities_path": entities_path,
        "feature_manifest_path": feature_manifest_path,
        "assignments": assignments,
    }


def _build_dev_manifest(synthetic: dict, target: Path) -> Path:
    manifest = development_manifest.build_development_manifest(
        synthetic["manifest_path"],
        synthetic["entities_path"],
        synthetic["feature_manifest_path"],
        expected_feature_manifest_path=synthetic["feature_manifest_path"],
        expected_feature_manifest_sha256=_sha256(synthetic["feature_manifest_path"]),
    )
    return Path(development_manifest.write_new_manifest(target, manifest)["path"])


@pytest.fixture()
def synthetic_dev(tmp_path):
    return _build_synthetic_dev_sources(tmp_path / "synthetic_dev")


# ---------------------------------------------------------------------------
# Manual NPZ/JSON run-directory helper for gpu_model.run_training rejections
# ---------------------------------------------------------------------------

def _default_pairs(pair_keys, labels) -> list:
    records = []
    for index, (pair_key, label) in enumerate(zip(pair_keys, labels)):
        molecule_id, taxonomy_text, accession = pair_key.split("|")
        records.append({
            "row_index": index,
            "pair_key": pair_key,
            "molecule_chembl_id": molecule_id,
            "taxonomy_id": int(taxonomy_text),
            "accession": accession,
            "label": int(label),
            "label_name": "active" if int(label) == 1 else "inactive",
            "evidence_ids": ["ev-synthetic-" + str(index)],
        })
    return records


def _write_run_dir(
    tmp_path: Path,
    *,
    features=None,
    labels=None,
    pair_keys=None,
    save_arrays=None,
    manifest_mutator=None,
    include_purpose: bool = True,
) -> dict:
    if pair_keys is None:
        pair_keys = ["CHEMBL900001|9606|P90001", "CHEMBL900002|9606|P90002"]
    if labels is None:
        labels = np.asarray([1, 0], dtype=np.int64)
    if features is None:
        features = _synthetic_features(labels)
    out = tmp_path / "run"
    out.mkdir(parents=True)
    arrays_path = out / "training_arrays.npz"
    if save_arrays is None:
        np.savez_compressed(arrays_path, features=features, labels=labels)
    else:
        save_arrays(arrays_path)
    arrays_sha = _sha256(arrays_path)
    pairs = _default_pairs(pair_keys, labels)
    manifest = {
        "schema_version": 1,
        "mode": "synthetic-gpu-model-test",
        "purpose": pair_pipeline.PURPOSE,
        "arrays": {
            "file": "training_arrays.npz",
            "sha256": arrays_sha,
            "bytes": arrays_path.stat().st_size,
            "dtype": "float32",
            "shape": [int(np.asarray(features).shape[0]), INPUT_DIM],
            "allow_pickle": False,
        },
        "label_mapping": {"inactive": 0, "active": 1},
        "class_counts": {
            "inactive": int((labels == 0).sum()),
            "active": int((labels == 1).sum()),
        },
        "pair_count": len(pairs),
        "pairs": pairs,
        "exclusions": [],
        "exclusion_count": 0,
    }
    if not include_purpose:
        manifest.pop("purpose")
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    input_manifest_path = out / "input_manifest.json"
    _dump_json(input_manifest_path, manifest)
    return {
        "out": out,
        "arrays_path": arrays_path,
        "input_manifest_path": input_manifest_path,
        "features": features,
        "labels": labels,
        "pair_keys": pair_keys,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Pure validation functions
# ---------------------------------------------------------------------------

def test_validate_training_input_accepts_bound_numeric_arrays():
    labels = np.asarray([0, 1], dtype=np.int64)
    features = _synthetic_features(labels)
    features_array, labels_array = gpu_model.validate_training_input(
        features, labels, pair_keys=[
            "CHEMBL900001|9606|P90001",
            "CHEMBL900002|9606|P90002",
        ],
    )
    assert features_array.shape == (2, INPUT_DIM)
    assert features_array.dtype == np.float32
    assert bool(np.isfinite(features_array).all())
    assert labels_array.tolist() == [0, 1]


def test_feature_block_validation_rejects_boundaries():
    good = _synthetic_features([1, 0])
    with pytest.raises(ValueError, match="empty_features"):
        gpu_model.validate_feature_blocks(good[:0])
    with pytest.raises(ValueError, match="feature_dim_mismatch"):
        gpu_model.validate_feature_blocks(good[:, :-1])
    with pytest.raises(ValueError, match="features_not_2d"):
        gpu_model.validate_feature_blocks(good[0])
    with pytest.raises(ValueError, match="features_dtype_not_floating"):
        gpu_model.validate_feature_blocks(good.astype(np.int64))
    non_binary = good.copy()
    non_binary[0, 0] = 2.0
    with pytest.raises(ValueError, match="morgan_block_not_binary"):
        gpu_model.validate_feature_blocks(non_binary)
    morgan_nan = good.copy()
    morgan_nan[0, 5] = np.nan
    with pytest.raises(ValueError, match="morgan_block_non_finite"):
        gpu_model.validate_feature_blocks(morgan_nan)
    esm_nan = good.copy()
    esm_nan[0, FP_DIM + 3] = np.nan
    with pytest.raises(ValueError, match="esm_block_non_finite"):
        gpu_model.validate_feature_blocks(esm_nan)
    esm_inf = good.copy()
    esm_inf[1, FP_DIM + 4] = np.inf
    with pytest.raises(ValueError, match="esm_block_non_finite"):
        gpu_model.validate_feature_blocks(esm_inf)


def test_label_validation_rejects_domain_count_and_dtype():
    features = _synthetic_features([1, 0])
    with pytest.raises(ValueError, match="label_out_of_domain"):
        gpu_model.validate_training_input(features, np.asarray([0, 2]))
    with pytest.raises(ValueError, match="label_dtype_not_integer"):
        gpu_model.validate_training_input(features, np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="label_count_mismatch"):
        gpu_model.validate_training_input(features, np.asarray([0, 1, 0]))
    with pytest.raises(ValueError, match="labels_not_1d"):
        gpu_model.validate_training_input(features, np.asarray([[0], [1]]))


def test_pair_key_validation_rejects_duplicates_and_missing_keys():
    labels = np.asarray([1, 0])
    features = _synthetic_features(labels)
    with pytest.raises(ValueError, match="pair_key_duplicate"):
        gpu_model.validate_training_input(
            features, labels, pair_keys=["a|9606|P1", "a|9606|P1"],
        )
    with pytest.raises(ValueError, match="pair_key_count_mismatch"):
        gpu_model.validate_training_input(features, labels, pair_keys=["a|9606|P1"])
    with pytest.raises(ValueError, match="pair_key_invalid"):
        gpu_model.validate_training_input(features, labels, pair_keys=["", "b|9606|P2"])


# ---------------------------------------------------------------------------
# run_training rejections (fail before CUDA/model use)
# ---------------------------------------------------------------------------

def test_run_training_rejects_object_dtype_npz_before_cuda(tmp_path):
    class _Payload:
        pass

    run = _write_run_dir(
        tmp_path,
        save_arrays=lambda path: np.savez(
            path,
            features=np.array([_Payload(), _Payload()], dtype=object),
            labels=np.array([0, 1], dtype=np.int64),
        ),
    )
    with pytest.raises(ValueError, match="arrays_require_pickle_rejected"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"],
            out_dir=run["out"],
            seed=42,
            steps=1,
            batch_size=2,
        )


def test_run_training_rejects_missing_and_extra_npz_keys(tmp_path):
    run = _write_run_dir(
        tmp_path,
        save_arrays=lambda path: np.savez(
            path, features=_synthetic_features([1, 0]),
        ),
    )
    with pytest.raises(ValueError, match="arrays_keys_invalid"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"], out_dir=run["out"], steps=1, batch_size=2,
        )
    run2 = _write_run_dir(
        tmp_path / "second",
        save_arrays=lambda path: np.savez(
            path,
            features=_synthetic_features([1, 0]),
            labels=np.asarray([1, 0], dtype=np.int64),
            extra=np.asarray([1, 2], dtype=np.int64),
        ),
    )
    with pytest.raises(ValueError, match="arrays_keys_invalid"):
        gpu_model.run_training(
            arrays_path=run2["arrays_path"], out_dir=run2["out"], steps=1, batch_size=2,
        )


def test_run_training_rejects_nonfinite_arrays(tmp_path):
    features = _synthetic_features([1, 0])
    features[0, 7] = np.nan
    run = _write_run_dir(tmp_path, features=features, labels=np.asarray([1, 0]))
    with pytest.raises(ValueError, match="morgan_block_non_finite"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"], out_dir=run["out"], steps=1, batch_size=2,
        )


def test_run_training_rejects_rows_mismatch(tmp_path):
    run = _write_run_dir(tmp_path)
    manifest = _load_json(run["input_manifest_path"])
    manifest["pairs"] = manifest["pairs"][:1]
    _dump_json(run["input_manifest_path"], manifest)
    with pytest.raises(ValueError, match="input_manifest_rows_mismatch"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"], out_dir=run["out"], steps=1, batch_size=2,
        )


def test_run_training_rejects_pair_metadata_mismatches(tmp_path):
    labels = np.asarray([1, 0])

    def duplicate(manifest):
        manifest["pairs"][1]["pair_key"] = manifest["pairs"][0]["pair_key"]

    run = _write_run_dir(tmp_path / "dup", labels=labels, manifest_mutator=duplicate)
    with pytest.raises(ValueError, match="input_manifest_pair_keys_not_unique"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"], out_dir=run["out"], steps=1, batch_size=2,
        )

    def bad_label(manifest):
        manifest["pairs"][0]["label"] = 2

    run = _write_run_dir(tmp_path / "label", labels=labels, manifest_mutator=bad_label)
    with pytest.raises(ValueError, match="input_manifest_pair_label_invalid"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"], out_dir=run["out"], steps=1, batch_size=2,
        )

    def bad_index(manifest):
        manifest["pairs"][1]["row_index"] = 0

    run = _write_run_dir(tmp_path / "index", labels=labels, manifest_mutator=bad_index)
    with pytest.raises(ValueError, match="input_manifest_pair_row_index_mismatch"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"], out_dir=run["out"], steps=1, batch_size=2,
        )

    def missing_field(manifest):
        manifest["pairs"][0].pop("accession")

    run = _write_run_dir(tmp_path / "field", labels=labels, manifest_mutator=missing_field)
    with pytest.raises(ValueError, match="input_manifest_pair_metadata_invalid"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"], out_dir=run["out"], steps=1, batch_size=2,
        )


def test_run_training_rejects_budget_bounds_before_model(tmp_path):
    run = _write_run_dir(tmp_path)
    base = {
        "arrays_path": run["arrays_path"],
        "out_dir": run["out"],
        "seed": 42,
        "steps": 1,
        "batch_size": 2,
    }
    for override, code in (
        ({"steps": 0}, "invalid_steps"),
        ({"steps": 1001}, "invalid_steps"),
        ({"steps": True}, "invalid_steps"),
        ({"steps": 1.5}, "invalid_steps"),
        ({"batch_size": 0}, "invalid_batch_size"),
        ({"batch_size": 17}, "invalid_batch_size"),
        ({"batch_size": True}, "invalid_batch_size"),
        ({"seed": True}, "invalid_seed"),
        ({"learning_rate": True}, "invalid_learning_rate"),
    ):
        with pytest.raises(ValueError, match=code):
            gpu_model.run_training(**{**base, **override})


def test_run_training_rejects_missing_purpose_and_arrays_hash(tmp_path):
    run = _write_run_dir(tmp_path / "nopurpose", include_purpose=False)
    with pytest.raises(ValueError, match="input_manifest_purpose_missing"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"], out_dir=run["out"], steps=1, batch_size=2,
        )
    run2 = _write_run_dir(tmp_path / "badhash")
    manifest = _load_json(run2["input_manifest_path"])
    manifest["arrays"]["sha256"] = "0" * 64
    _dump_json(run2["input_manifest_path"], manifest)
    with pytest.raises(ValueError, match="input_arrays_hash_mismatch"):
        gpu_model.run_training(
            arrays_path=run2["arrays_path"], out_dir=run2["out"], steps=1, batch_size=2,
        )


# ---------------------------------------------------------------------------
# Development-manifest consumption (CPU preprocessing before GPU)
# ---------------------------------------------------------------------------

def test_prepare_development_arrays_consumes_declared_rows(synthetic_dev, tmp_path):
    target = _build_dev_manifest(synthetic_dev, tmp_path / "dev_root" / "development_manifest.json")
    prepared = pair_pipeline.prepare_development_pair_arrays(
        development_manifest_path=str(target),
        out_dir=str(tmp_path / "fit_run"),
    )
    assert prepared["pair_count"] == 8
    assert prepared["class_counts"] == {"inactive": 4, "active": 4}
    dev = development_manifest.load_development_manifest(target)
    assert prepared["development_manifest_sha256"] == dev["sha256"]
    features, labels = pair_pipeline.load_training_arrays(prepared["arrays_path"])
    assert features.shape == (8, INPUT_DIM)
    assert labels.tolist() == [pair["label"] for pair in dev["pairs"]]
    input_manifest = _load_json(prepared["input_manifest_path"])
    assert input_manifest["development_manifest"]["sha256"] == dev["sha256"]
    assert input_manifest["development_manifest"]["status"] == "ready"
    assert input_manifest["purpose"] == pair_pipeline.PURPOSE_DEVELOPMENT
    assert input_manifest["pair_count"] == 8
    assert [pair["pair_key"] for pair in input_manifest["pairs"]] == dev["row_order"]
    fit = input_manifest["fit_diagnostic"]
    assert fit["seed"] == 42
    assert fit["per_class"] == pair_pipeline.FIT_DIAGNOSTIC_PER_CLASS
    assert len(fit["rows"]) == len(fit["pair_keys"]) == len(fit["labels"])
    assert set(fit["labels"]) == {0, 1}
    for row, key, label in zip(fit["rows"], fit["pair_keys"], fit["labels"]):
        assert dev["row_order"][row] == key
        assert int(labels[row]) == label
    assert input_manifest["sampling"] == gpu_model.SAMPLING_CLASS_BALANCED
    assert "NaN" not in Path(prepared["input_manifest_path"]).read_text(encoding="utf-8")


def test_prepare_development_arrays_rejects_tampered_binding(synthetic_dev, tmp_path):
    target = _build_dev_manifest(synthetic_dev, tmp_path / "dev_root" / "development_manifest.json")
    dev = _load_json(target)
    dev["pairs"][0]["label"] = 1 - dev["pairs"][0]["label"]
    _dump_json(target, dev)
    with pytest.raises(ValueError, match="development_manifest_class_counts_mismatch"):
        pair_pipeline.prepare_development_pair_arrays(
            development_manifest_path=str(target),
            out_dir=str(tmp_path / "fit_run"),
        )

    target2 = _build_dev_manifest(
        synthetic_dev, tmp_path / "dev_root2" / "development_manifest.json",
    )
    dev2 = _load_json(target2)
    dev2["source"]["feature_manifest_sha256"] = "0" * 64
    _dump_json(target2, dev2)
    with pytest.raises(ValueError, match="feature_manifest_sha256_mismatch"):
        pair_pipeline.prepare_development_pair_arrays(
            development_manifest_path=str(target2),
            out_dir=str(tmp_path / "fit_run2"),
        )


def test_blocked_development_manifest_prevents_training(tmp_path):
    synthetic = _build_synthetic_dev_sources(tmp_path / "synthetic_blocked")
    manifest_json = _load_json(synthetic["manifest_path"])
    manifest_json["split"]["pair_assignments"] = [
        assignment for assignment in manifest_json["split"]["pair_assignments"]
        if assignment["labels"] == ["active"]
    ]
    _dump_json(synthetic["manifest_path"], manifest_json)
    dev_manifest = development_manifest.build_development_manifest(
        synthetic["manifest_path"],
        synthetic["entities_path"],
        synthetic["feature_manifest_path"],
        expected_feature_manifest_path=synthetic["feature_manifest_path"],
        expected_feature_manifest_sha256=_sha256(synthetic["feature_manifest_path"]),
    )
    assert dev_manifest["status"] == "blocked"
    target = Path(development_manifest.write_new_manifest(
        tmp_path / "dev_root" / "development_manifest.json", dev_manifest,
    )["path"])
    with pytest.raises(ValueError, match="development_manifest_training_blocked"):
        pair_pipeline.prepare_development_pair_arrays(
            development_manifest_path=str(target),
            out_dir=str(tmp_path / "fit_run"),
        )


# ---------------------------------------------------------------------------
# Synthetic end-to-end development fit through the real CUDA interpreter
# ---------------------------------------------------------------------------

def test_synthetic_development_fit_end_to_end(synthetic_dev, tmp_path, capsys):
    if not GPU_PYTHON.is_file():
        pytest.fail("registered CUDA interpreter is missing: " + str(GPU_PYTHON))
    target = _build_dev_manifest(
        synthetic_dev, tmp_path / "dev_root" / "development_manifest.json",
    )
    out_dir = tmp_path / "fit"
    exit_code = pair_pipeline.main([
        "train",
        "--development-manifest", str(target),
        "--out", str(out_dir),
        "--gpu-python", str(GPU_PYTHON),
        "--seed", "42",
        "--steps", "1000",
        "--batch-size", "16",
        "--learning-rate", "0.01",
        "--hidden-dim", "16",
    ])
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err + captured.out

    run = _load_json(out_dir / "run.json")
    assert run["status"] == "trained"
    assert run["device"] == "cuda"
    assert run["device_name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert run["inputs"]["development_manifest_sha256"] == _sha256(target)
    assert run["inputs"]["pair_count"] == 8
    assert run["inputs"]["class_counts"] == {"inactive": 4, "active": 4}
    assert run["training"]["steps"] == 1000
    assert run["training"]["sampling"] == gpu_model.SAMPLING_CLASS_BALANCED
    assert len(run["training"]["loss_records"]) == 1000
    for record in run["training"]["loss_records"]:
        assert math.isfinite(record["loss"])
        assert math.isfinite(record["gradient_norm"])
        assert record["gradient_norm"] > 0.0
        assert math.isfinite(record["parameter_delta"])
    assert run["training"]["total_parameter_delta"] > 0.0
    assert run["training"]["frozen_feature_input"] is True
    assert set(run["training"]["trainable_parameter_names"]) == {
        "input_layer.weight", "input_layer.bias",
        "output_layer.weight", "output_layer.bias",
    }
    assert run["portability"]["passed"] is True
    assert run["portability"]["cpu_reload_max_abs_diff"] <= 1e-6

    fit = _load_json(out_dir / "fit_diagnostic.json")
    assert fit["status"] == "passed", json.dumps(fit["reasons"])
    small = fit["small_sample"]
    assert small["criterion"]["relative_drop"] >= 0.10
    assert small["criterion"]["per_class_correct"]["inactive"] >= 1
    assert small["criterion"]["per_class_correct"]["active"] >= 1
    assert fit["full_manifest"]["majority_collapse"] is False
    assert len(fit["small_sample"]["sample_pair_keys"]) >= 2

    checkpoint = out_dir / "checkpoint.pt"
    assert checkpoint.is_file()
    assert run["outputs"]["checkpoint.pt"] == _sha256(checkpoint)
    predictions = (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(predictions) == 8
    prediction_rows = [json.loads(line) for line in predictions]
    assert [row["pair_key"] for row in prediction_rows] == run["inputs"]["row_order"]
    assert all(row["device"] == "cpu" for row in prediction_rows)
    assert all(math.isfinite(row["logit"]) for row in prediction_rows)
    assert (out_dir / "gpu_process.log").is_file()

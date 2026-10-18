"""Standalone checkpoint-consumer prediction tests (T3-05 / T3-06).

Covers the independent predict CLI: request ordering, empty/duplicate/unknown
pair rejection, development-manifest and feature/checkpoint hash binding
failures, and one synthetic end-to-end development train -> isolated CUDA
prediction run.  The synthetic fixture is an engineering probe only; the real
frozen-data train/predict acceptance commands stay deferred while the user's
GPU gate is closed, and no formal effect metric is emitted anywhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from training import development_manifest, gpu_model, pair_pipeline


APL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPU_PYTHON = Path("E:/paper/DDA-Reproduction/.conda/python.exe")
ESM_DIM = gpu_model.INPUT_ESM_DIM
FP_DIM = gpu_model.INPUT_FP_DIM
INPUT_DIM = gpu_model.INPUT_DIM
CONTENT_SHA = "2" * 64
ENTITY_VERSION = "entities-v2-" + "2" * 16
MODEL_REVISION = "d" * 40
FORMAL_METRIC_KEYS = {
    "auc",
    "auc_score",
    "roc_auc",
    "accuracy",
    "calibration",
    "ranking",
    "precision",
    "recall",
    "f1",
}


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_json(path, value) -> None:
    path = Path(path)
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


def _synthetic_dev_sources(root: Path) -> dict:
    """Eight source-bound development pairs: four active, four inactive."""
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
            "features": {
                "esm_embedding": _embedding(
                    1.0 if is_active else -1.0, index,
                ),
            },
            "cache_key": hashlib.sha256(
                ("cache:" + accession).encode("utf-8"),
            ).hexdigest(),
            "cache_file_sha256": hashlib.sha256(
                ("cache-file:" + accession).encode("utf-8"),
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


def _build_dev_manifest(sources: dict, target: Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = development_manifest.build_development_manifest(
        sources["manifest_path"],
        sources["entities_path"],
        sources["feature_manifest_path"],
        expected_feature_manifest_path=sources["feature_manifest_path"],
        expected_feature_manifest_sha256=_sha256(sources["feature_manifest_path"]),
    )
    return Path(development_manifest.write_new_manifest(target, manifest)["path"])


@pytest.fixture()
def synthetic_dev(tmp_path):
    return _synthetic_dev_sources(tmp_path / "synthetic_dev")


@pytest.fixture()
def canonical_feature_manifest_pin(synthetic_dev, monkeypatch):
    """Pin pair_pipeline's canonical feature snapshot to the synthetic source (F2)."""
    monkeypatch.setattr(
        pair_pipeline,
        "EXPECTED_FEATURE_MANIFEST_SHA256",
        _sha256(synthetic_dev["feature_manifest_path"]),
    )
    monkeypatch.setattr(
        pair_pipeline,
        "EXPECTED_FEATURE_MANIFEST_PATH",
        Path(synthetic_dev["feature_manifest_path"]),
    )
    return synthetic_dev["feature_manifest_path"]


def _craft_checkpoint(
    tmp_path: Path,
    name: str,
    *,
    metadata_mutator=None,
    purpose: str = pair_pipeline.PURPOSE_DEVELOPMENT,
) -> dict:
    torch.manual_seed(13)
    model = gpu_model.PairHead(hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.randn(4, INPUT_DIM) * 0.05
    loss = model(features).sum()
    loss.backward()
    optimizer.step()
    metadata = {
        "arrays_sha256": "a" * 64,
        "input_manifest_sha256": "b" * 64,
        "pair_keys": ["CRAFTED1|9606|P1", "CRAFTED2|9606|P2"],
        "purpose": purpose,
        "rows": 2,
    }
    if metadata_mutator is not None:
        metadata_mutator(metadata)
    path = tmp_path / (name + ".pt")
    gpu_model.save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        seed=7,
        config=model.config(),
        input_metadata=metadata,
        runtime={"training_device": "cpu"},
    )
    return {"path": path, "metadata": metadata}


def _gpu_environment() -> dict:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(APL_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _collect_keys(value, found: set) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key).lower())
            _collect_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_keys(item, found)


def _assert_no_formal_metric_fields(record: dict) -> None:
    found: set = set()
    _collect_keys(record, found)
    assert not (found & FORMAL_METRIC_KEYS), sorted(found & FORMAL_METRIC_KEYS)


# ---------------------------------------------------------------------------
# CPU contract: request validation before any output or GPU work
# ---------------------------------------------------------------------------

def test_predict_cli_rejects_empty_and_duplicate_pair_keys(
    synthetic_dev, tmp_path, capsys,
):
    target = _build_dev_manifest(
        synthetic_dev, tmp_path / "dev_root" / "development_manifest.json",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-used")
    empty_code = pair_pipeline.main([
        "predict",
        "--development-manifest", str(target),
        "--checkpoint", str(checkpoint),
        "--out", str(tmp_path / "prediction_empty"),
    ])
    empty_capture = capsys.readouterr()
    assert empty_code == 2
    assert "predict_pair_keys_empty" in empty_capture.err
    assert not (tmp_path / "prediction_empty").exists()

    dev = development_manifest.load_development_manifest(target)
    duplicate_key = dev["row_order"][0]
    duplicate_code = pair_pipeline.main([
        "predict",
        "--development-manifest", str(target),
        "--checkpoint", str(checkpoint),
        "--pair-key", duplicate_key,
        "--pair-key", duplicate_key,
        "--out", str(tmp_path / "prediction_duplicate"),
    ])
    duplicate_capture = capsys.readouterr()
    assert duplicate_code == 2
    assert "predict_pair_key_duplicate" in duplicate_capture.err
    assert not (tmp_path / "prediction_duplicate").exists()


def test_predict_rejects_unknown_pair_key_before_output(
    synthetic_dev, canonical_feature_manifest_pin, tmp_path,
):
    target = _build_dev_manifest(
        synthetic_dev, tmp_path / "dev_root" / "development_manifest.json",
    )
    dev = development_manifest.load_development_manifest(target)
    feature_sha = _sha256(synthetic_dev["feature_manifest_path"])
    crafted = _craft_checkpoint(
        tmp_path,
        "unknown_key",
        metadata_mutator=lambda metadata: metadata.update({
            "pair_keys": list(dev["row_order"]),
            "rows": len(dev["row_order"]),
            "development_manifest_sha256": dev["sha256"],
            "feature_manifest_sha256": feature_sha,
        }),
    )
    with pytest.raises(ValueError, match="predict_pair_key_not_in_manifest"):
        pair_pipeline.predict_pairs(
            development_manifest_path=str(target),
            checkpoint_path=str(crafted["path"]),
            pair_keys=["CHEMBL999999|9606|P999999"],
            out_dir=str(tmp_path / "prediction_unknown"),
            gpu_python=str(GPU_PYTHON),
        )
    assert not (tmp_path / "prediction_unknown").exists()


def test_predict_rejects_stale_manifest_and_checkpoint_bindings(
    synthetic_dev, canonical_feature_manifest_pin, tmp_path,
):
    target = _build_dev_manifest(
        synthetic_dev, tmp_path / "dev_root" / "development_manifest.json",
    )
    dev = development_manifest.load_development_manifest(target)
    feature_sha = _sha256(synthetic_dev["feature_manifest_path"])
    pair_key = dev["row_order"][0]

    def metadata_for(dev_sha256: str = None, feature_sha256: str = None):
        def mutate(metadata):
            metadata["pair_keys"] = list(dev["row_order"])
            metadata["rows"] = len(dev["row_order"])
            metadata["development_manifest_sha256"] = dev_sha256 or dev["sha256"]
            metadata["feature_manifest_sha256"] = feature_sha256 or feature_sha
        return mutate

    valid = _craft_checkpoint(tmp_path, "valid", metadata_mutator=metadata_for())
    # A byte-level rewrite keeps the manifest structurally valid but changes its
    # SHA-256, exactly like consuming a stale development manifest.
    stale_target = tmp_path / "stale_root" / "development_manifest.json"
    stale_target.parent.mkdir(parents=True)
    stale_target.write_text(
        json.dumps(_load_json(target), ensure_ascii=False, indent=1) + "\n\n",
        encoding="utf-8",
    )
    assert _sha256(stale_target) != dev["sha256"]
    with pytest.raises(ValueError, match="predict_development_manifest_binding_mismatch"):
        pair_pipeline.predict_pairs(
            development_manifest_path=str(stale_target),
            checkpoint_path=str(valid["path"]),
            pair_keys=[pair_key],
            out_dir=str(tmp_path / "prediction_stale"),
            gpu_python=str(GPU_PYTHON),
        )
    with pytest.raises(ValueError, match="predict_feature_manifest_binding_mismatch"):
        bad_feature = _craft_checkpoint(
            tmp_path,
            "bad_feature",
            metadata_mutator=metadata_for(feature_sha256="0" * 64),
        )
        pair_pipeline.predict_pairs(
            development_manifest_path=str(target),
            checkpoint_path=str(bad_feature["path"]),
            pair_keys=[pair_key],
            out_dir=str(tmp_path / "prediction_bad_feature"),
            gpu_python=str(GPU_PYTHON),
        )
    tracer_checkpoint = _craft_checkpoint(
        tmp_path,
        "tracer_purpose",
        purpose=pair_pipeline.PURPOSE,
        metadata_mutator=metadata_for(),
    )
    with pytest.raises(ValueError, match="predict_checkpoint_purpose_mismatch"):
        pair_pipeline.predict_pairs(
            development_manifest_path=str(target),
            checkpoint_path=str(tracer_checkpoint["path"]),
            pair_keys=[pair_key],
            out_dir=str(tmp_path / "prediction_bad_purpose"),
            gpu_python=str(GPU_PYTHON),
        )
    for name in ("prediction_stale", "prediction_bad_feature", "prediction_bad_purpose"):
        assert not (tmp_path / name).exists()


def test_predict_rejects_checkpoint_sha_pin_and_unregistered_gpu(
    synthetic_dev, canonical_feature_manifest_pin, tmp_path, capsys,
):
    target = _build_dev_manifest(
        synthetic_dev, tmp_path / "dev_root" / "development_manifest.json",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    exit_code = pair_pipeline.main([
        "predict",
        "--development-manifest", str(target),
        "--checkpoint", str(checkpoint),
        "--pair-key", "CHEMBL920001|9606|P92001",
        "--out", str(tmp_path / "prediction_pin"),
        "--expected-checkpoint-sha256", "0" * 64,
    ])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "checkpoint_sha256_mismatch" in captured.err
    with pytest.raises(ValueError, match="gpu_python_must_be_registered_cuda_interpreter"):
        pair_pipeline.predict_pairs(
            development_manifest_path=str(target),
            checkpoint_path=str(checkpoint),
            pair_keys=["CHEMBL920001|9606|P92001"],
            out_dir=str(tmp_path / "prediction_gpu"),
            gpu_python=sys.executable,
        )


def test_gpu_prediction_request_rejects_duplicates_and_bad_hashes(tmp_path):
    base = {
        "schema_version": 1,
        "purpose": pair_pipeline.PURPOSE_DEVELOPMENT,
        "pair_keys": ["CHEMBL920001|9606|P92001"],
        "training_row_indices": [0],
        "checkpoint_sha256": "a" * 64,
        "training_arrays_sha256": "b" * 64,
        "prediction_arrays_sha256": "c" * 64,
        "development_manifest_sha256": "d" * 64,
        "feature_manifest_sha256": "e" * 64,
    }
    validated = gpu_model._validate_prediction_request(dict(base))
    assert validated["pair_keys"] == base["pair_keys"]
    duplicate = dict(base)
    duplicate["pair_keys"] = ["CHEMBL920001|9606|P92001", "CHEMBL920001|9606|P92001"]
    duplicate["training_row_indices"] = [0, 1]
    with pytest.raises(ValueError, match="prediction_pair_key_duplicate"):
        gpu_model._validate_prediction_request(duplicate)
    bad_hash = dict(base)
    bad_hash["checkpoint_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="invalid_sha256:checkpoint_sha256"):
        gpu_model._validate_prediction_request(bad_hash)
    empty = dict(base)
    empty["pair_keys"] = []
    empty["training_row_indices"] = []
    with pytest.raises(ValueError, match="prediction_pair_keys_empty"):
        gpu_model._validate_prediction_request(empty)


# ---------------------------------------------------------------------------
# Synthetic end-to-end: development train then isolated CUDA prediction
# ---------------------------------------------------------------------------

def test_synthetic_development_train_then_standalone_prediction(
    synthetic_dev, canonical_feature_manifest_pin, tmp_path, capsys,
):
    if not GPU_PYTHON.is_file():
        pytest.fail("registered CUDA interpreter is missing: " + str(GPU_PYTHON))
    target = _build_dev_manifest(
        synthetic_dev, tmp_path / "dev_root" / "development_manifest.json",
    )
    dev = development_manifest.load_development_manifest(target)
    train_out = tmp_path / "train"
    exit_code = pair_pipeline.main([
        "train",
        "--development-manifest", str(target),
        "--out", str(train_out),
        "--gpu-python", str(GPU_PYTHON),
        "--seed", "42",
        "--steps", "2",
        "--batch-size", "4",
        "--learning-rate", "0.01",
        "--hidden-dim", "8",
    ])
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err + captured.stdout
    checkpoint_path = train_out / "checkpoint.pt"
    assert checkpoint_path.is_file()
    checkpoint_sha256 = _sha256(checkpoint_path)
    training_sha256 = _sha256(train_out / "training_arrays.npz")

    # F6/F8: the training run record of the dev path is auditable.
    train_run = _load_json(train_out / "run.json")
    train_preprocess = train_run["environment"]["preprocess_python"]
    assert train_preprocess["platform"] == platform.platform()
    assert train_preprocess["system"] == platform.system()
    assert train_preprocess["imports"]["torch"]["version"] == torch.__version__
    assert "exclusions" not in train_run["inputs"]
    assert train_run["inputs"]["exclusion_count"] == dev["counts"]["excluded"]
    assert train_run["inputs"]["exclusion_source"]["source"] == "development_manifest"
    assert train_run["inputs"]["exclusion_source"]["sha256"] == dev["sha256"]

    requested = [dev["row_order"][0], dev["row_order"][2]]
    prediction_out = tmp_path / "prediction"
    arguments = [
        "predict",
        "--development-manifest", str(target),
        "--checkpoint", str(checkpoint_path),
        "--out", str(prediction_out),
        "--gpu-python", str(GPU_PYTHON),
    ]
    for pair_key in requested:
        arguments.extend(["--pair-key", pair_key])
    prediction_code = pair_pipeline.main(arguments)
    prediction_capture = capsys.readouterr()
    assert prediction_code == 0, prediction_capture.err + prediction_capture.stdout

    run = _load_json(prediction_out / "run.json")
    assert run["status"] == "predicted"
    assert run["mode"] == pair_pipeline.PREDICTION_MODE
    assert run["device"] == "cuda"
    assert run["device_name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert run["run_id"] == prediction_out.name
    assert run["purpose"] == pair_pipeline.PURPOSE_DEVELOPMENT
    assert run["purpose_class"] == "engineering-development"
    assert run["model"]["model_id"] == "pair-head-mlp_relu_logit_v1-h8"
    assert run["model"]["checkpoint_sha256"] == checkpoint_sha256
    assert run["model"]["checkpoint_step"] > 0
    assert run["data"]["data_id"].startswith("development-manifest-")
    assert run["data"]["development_manifest_sha256"] == _sha256(target)
    assert run["data"]["development_manifest_sha256"] == dev["sha256"]
    assert run["data"]["feature_manifest_sha256"] == _sha256(
        synthetic_dev["feature_manifest_path"],
    )
    assert run["inputs"]["pair_keys"] == requested
    assert run["inputs"]["prediction_arrays_sha256"] == _sha256(
        prediction_out / "prediction_arrays.npz",
    )
    assert run["inputs"]["prediction_request_sha256"] == _sha256(
        prediction_out / "prediction_request.json",
    )
    assert run["environment"]["gpu_runtime"]["cuda_available"] is True
    assert run["environment"]["gpu_runtime"]["torch"] == "2.9.1+cu128"
    assert run["environment"]["checkpoint_training_runtime"]["training_device"] == "cuda"
    imports = run["environment"]["preprocess_python"]["imports"]
    assert imports["torch"]["version"] == torch.__version__
    assert imports["numpy"]["version"] == np.__version__
    preprocess = run["environment"]["preprocess_python"]
    assert preprocess["platform"] == platform.platform()
    assert preprocess["system"] == platform.system()
    assert preprocess["machine"] == platform.machine()
    assert preprocess["python_full_version"] == sys.version
    assert run["portability"]["passed"] is True
    assert run["portability"]["atol"] == 1e-6
    assert run["portability"]["rtol"] == 1e-6
    assert run["portability"]["max_abs_diff"] <= 1e-6
    assert run["outputs"]["prediction_arrays.npz"] == _sha256(
        prediction_out / "prediction_arrays.npz",
    )
    assert run["outputs"]["prediction_request.json"] == _sha256(
        prediction_out / "prediction_request.json",
    )
    assert run["outputs"]["gpu_prediction.json"] == _sha256(
        prediction_out / "gpu_prediction.json",
    )
    assert run["outputs"]["gpu_process.log"] == _sha256(
        prediction_out / "gpu_process.log",
    )
    assert run["outputs"]["predictions.jsonl"] == _sha256(
        prediction_out / "predictions.jsonl",
    )
    _assert_no_formal_metric_fields(run)

    predictions = [
        json.loads(line)
        for line in (prediction_out / "predictions.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    assert [row["pair_key"] for row in predictions] == requested
    assert len(predictions) == len(requested)
    for row in predictions:
        assert math.isfinite(row["logit"])
        assert math.isfinite(row["cpu_logit"])
        assert row["device"] == "cuda"
        assert row["run_id"] == prediction_out.name
        assert row["model_id"] == run["model"]["model_id"]
        assert row["data_id"] == run["data"]["data_id"]
        assert row["purpose"] == pair_pipeline.PURPOSE_DEVELOPMENT
        assert row["purpose_class"] == "engineering-development"
        assert row["engineering_development"] is True
        assert row["checkpoint_sha256"] == checkpoint_sha256
        assert abs(row["logit"] - row["cpu_logit"]) <= 1e-6
    dev_by_key = {pair["pair_key"]: pair for pair in dev["pairs"]}
    for row in predictions:
        assert row["label"] == dev_by_key[row["pair_key"]]["label"]
        assert row["label_name"] == dev_by_key[row["pair_key"]]["label_name"]

    gpu_prediction = _load_json(prediction_out / "gpu_prediction.json")
    assert gpu_prediction["status"] == "predicted"
    assert gpu_prediction["device"] == "cuda"
    assert gpu_prediction["checkpoint"]["sha256"] == checkpoint_sha256
    assert gpu_prediction["training_row_indices"] == [
        dev["row_order"].index(key) for key in requested
    ]
    process_log = (prediction_out / "gpu_process.log").read_text(encoding="utf-8")
    assert "training.gpu_model" in process_log
    assert "--checkpoint" in process_log
    assert "exit_code: 0" in process_log

    # The prediction request binds the actual training arrays hash of the
    # checkpoint, not the freshly materialised prediction batch.
    request = _load_json(prediction_out / "prediction_request.json")
    assert request["training_arrays_sha256"] == training_sha256
    assert request["checkpoint_sha256"] == checkpoint_sha256
    assert request["pair_keys"] == requested
    assert request["development_manifest_sha256"] == _sha256(target)
    assert request["feature_manifest_sha256"] == _sha256(
        synthetic_dev["feature_manifest_path"],
    )

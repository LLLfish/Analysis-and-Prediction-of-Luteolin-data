"""Contract and CUDA behavior tests for the Phase 03 pair-training tracer.

Three groups:

* CPU contract tests against a synthetic two-pair fixture (no real data);
* a synthetic-fixture CUDA roundtrip that drives the real GPU entry point,
  exercises checkpoint reload on GPU and CPU portability (allowed tiny
  environment/behavior probe, not phase model training);
* a read-only CPU preprocessing check against the frozen v1 corpus and the
  registered ready-ESM snapshot.

The real-frozen-data GPU CLI acceptance run (training.pair_pipeline train) is
deferred while the user's GPU gate is closed and is NOT executed by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from training import gpu_model, pair_pipeline


APL_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = Path("E:/paper/DDA-Reproduction/.conda/python.exe")
REAL_FROZEN_MANIFEST = Path(
    "E:/木犀草素数据分析与预测/P1_02_03_Artifacts_6142fd3706ff480a8aa4f013cfe1c381/"
    "a4a30ac7dcee4478a4c3edee39e3b984/frozen/corpus-formal-v1-45fdfee33b6d/manifest.json"
)
REAL_FEATURE_MANIFEST = Path(
    "E:/木犀草素数据分析与预测/P1_02_04_Artifacts_847e7b27ed5942e19af64e2bd65b8d2f/"
    "01d118a1293346d38a58f116f7c86e2a/feature_manifest.json"
)


class _UnsafePayload:
    """Deliberately pickled object used to prove the restricted loader rejects it."""


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_json(path: Path, value, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, allow_nan=allow_nan, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def _load_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _embedding(seed: int, dim: int = pair_pipeline.ESM_EMBEDDING_DIM) -> list:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(dim) * 0.05).astype(np.float32).tolist()


def _build_synthetic_workspace(root: Path) -> dict:
    """Two synthetic pairs (one active, one inactive) with full source bindings."""
    content_sha256 = "a" * 64
    entity_version = "entities-v2-" + "a" * 16
    proteins = [
        {
            "accession": "P90001",
            "taxonomy_id": 9606,
            "sequence_status": "verified",
            "identity_status": "verified",
            "full_sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP",
        },
        {
            "accession": "P90002",
            "taxonomy_id": 9606,
            "sequence_status": "verified",
            "identity_status": "verified",
            "full_sequence": "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLL",
        },
    ]
    for protein in proteins:
        protein["sequence_sha256"] = hashlib.sha256(
            protein["accession"].encode("utf-8")
        ).hexdigest()
    entities = {
        "schema_version": 2,
        "content_sha256": content_sha256,
        "entity_version": entity_version,
        "molecules": [
            {
                "molecule_chembl_id": "CHEMBL900001",
                "canonical_smiles": "CCO",
                "structure_status": "verified",
            },
            {
                "molecule_chembl_id": "CHEMBL900002",
                "canonical_smiles": "c1ccccc1",
                "structure_status": "verified",
            },
        ],
        "proteins": proteins,
    }
    frozen_dir = root / "frozen"
    entities_path = frozen_dir / "entities.json"
    _dump_json(entities_path, entities)
    entities_sha256 = _sha256(entities_path)

    embeddings = {}
    feature_rows = []
    for index, protein in enumerate(proteins, start=1):
        embedding = _embedding(index)
        embeddings[protein["accession"]] = embedding
        feature_rows.append({
            "accession": protein["accession"],
            "taxonomy_id": protein["taxonomy_id"],
            "sequence_sha256": protein["sequence_sha256"],
            "entity_version": entity_version,
            "entity_bundle_hash": content_sha256,
            "bundle_file_sha256": entities_sha256,
            "embedding_dim": pair_pipeline.ESM_EMBEDDING_DIM,
            "status": "ready",
            "finite": True,
            "test_provenance": None,
            "features": {"esm_embedding": embedding},
            "cache_key": hashlib.sha256(
                ("cache:" + protein["accession"]).encode("utf-8")
            ).hexdigest(),
            "cache_file_sha256": hashlib.sha256(
                ("cache-file:" + protein["accession"]).encode("utf-8")
            ).hexdigest(),
            "model_id": pair_pipeline.ESM_MODEL_ID,
            "model_revision": "c" * 40,
            "pooling_policy": "residue-mask-mean-v1:max_tokens=1024",
            "max_tokens": 1024,
        })
    feature_manifest = {
        "schema_version": 1,
        "status": "ready",
        "bundle_file_sha256": entities_sha256,
        "entity_bundle_hash": content_sha256,
        "entity_version": entity_version,
        "model": {
            "model_id": pair_pipeline.ESM_MODEL_ID,
            "revision": "c" * 40,
            "external_revision_verified": False,
            "embedding_dim": pair_pipeline.ESM_EMBEDDING_DIM,
        },
        "entities": feature_rows,
        "status_counts": {"ready": len(feature_rows)},
    }
    features_dir = root / "features"
    feature_manifest_path = features_dir / "feature_manifest.json"
    _dump_json(feature_manifest_path, feature_manifest)

    # Assignment order is deliberately reversed w.r.t. pair_key order so a test
    # can detect joins performed by array position.
    manifest = {
        "schema_version": 2,
        "entities": {
            "path": "entities.json",
            "file_sha256": entities_sha256,
            "content_sha256": content_sha256,
            "entity_version": entity_version,
        },
        "split": {
            "pair_assignments": [
                {
                    "pair_key": "CHEMBL900002|9606|P90002",
                    "labels": ["inactive"],
                    "split": "train",
                    "evidence_ids": ["evidence-v1-synthetic-b"],
                },
                {
                    "pair_key": "CHEMBL900001|9606|P90001",
                    "labels": ["active"],
                    "split": "train",
                    "evidence_ids": ["evidence-v1-synthetic-a"],
                },
            ],
            "entity_bundle_hash": content_sha256,
            "entity_version": entity_version,
            "label_policy_sha256": "b" * 64,
            "split_policy_sha256": "d" * 64,
            "split_sha256": "e" * 64,
            "training_ready": False,
        },
    }
    manifest_path = frozen_dir / "manifest.json"
    _dump_json(manifest_path, manifest)
    return {
        "root": root,
        "manifest_path": manifest_path,
        "entities_path": entities_path,
        "feature_manifest_path": feature_manifest_path,
        "embeddings": embeddings,
        "manifest": manifest,
        "feature_manifest": feature_manifest,
    }


@pytest.fixture()
def synthetic(tmp_path):
    return _build_synthetic_workspace(tmp_path / "synthetic")


def _prepare(synthetic, out_dir):
    return pair_pipeline.prepare_pair_arrays(
        manifest_path=str(synthetic["manifest_path"]),
        feature_manifest_path=str(synthetic["feature_manifest_path"]),
        out_dir=str(out_dir),
    )


# ---------------------------------------------------------------------------
# CPU contract tests: synthetic source binding and pair/label alignment
# ---------------------------------------------------------------------------

def test_prepare_synthetic_contract_and_pair_order(synthetic, tmp_path):
    prepared = _prepare(synthetic, tmp_path / "run")
    assert prepared["pair_count"] == 2
    assert prepared["class_counts"] == {"inactive": 1, "active": 1}
    assert prepared["exclusion_count"] == 0

    pair_keys = [pair["pair_key"] for pair in prepared["pairs"]]
    assert pair_keys == [
        "CHEMBL900001|9606|P90001",
        "CHEMBL900002|9606|P90002",
    ]
    assert pair_keys == sorted(pair_keys)
    assert [pair["label"] for pair in prepared["pairs"]] == [1, 0]

    features, labels = pair_pipeline.load_training_arrays(prepared["arrays_path"])
    assert features.shape == (2, pair_pipeline.INPUT_DIM)
    assert features.dtype == np.float32
    assert labels.dtype.kind in ("i", "u")
    assert labels.tolist() == [1, 0]
    assert bool(np.isfinite(features).all())

    morgan = features[:, : pair_pipeline.MORGAN_NUM_BITS]
    esm = features[:, pair_pipeline.MORGAN_NUM_BITS:]
    assert set(np.unique(morgan).tolist()) <= {0.0, 1.0}
    # The active pair (row 0) must carry the P90001 embedding even though the
    # assignment array listed it second.
    assert np.allclose(esm[0], np.asarray(synthetic["embeddings"]["P90001"], dtype=np.float32))

    manifest = _load_json(prepared["input_manifest_path"])
    assert manifest["purpose"] == pair_pipeline.PURPOSE
    assert manifest["arrays"]["sha256"] == _sha256(prepared["arrays_path"])
    assert manifest["source"]["feature_manifest_sha256"] == _sha256(
        synthetic["feature_manifest_path"]
    )
    assert manifest["source"]["entities_sha256"] == _sha256(synthetic["entities_path"])
    assert manifest["label_mapping"] == {"inactive": 0, "active": 1}
    raw = Path(prepared["input_manifest_path"]).read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw


def test_prepare_rejects_duplicate_pair_key(synthetic, tmp_path):
    manifest = _load_json(synthetic["manifest_path"])
    duplicate = dict(manifest["split"]["pair_assignments"][0])
    duplicate["labels"] = ["active"]
    manifest["split"]["pair_assignments"].append(duplicate)
    _dump_json(synthetic["manifest_path"], manifest)
    with pytest.raises(ValueError, match="duplicate_pair_key"):
        _prepare(synthetic, tmp_path / "run")


def test_prepare_rejects_label_conflict(synthetic, tmp_path):
    manifest = _load_json(synthetic["manifest_path"])
    manifest["split"]["pair_assignments"][0]["labels"] = ["active", "inactive"]
    _dump_json(synthetic["manifest_path"], manifest)
    with pytest.raises(ValueError, match="pair_label_not_unique_binary"):
        _prepare(synthetic, tmp_path / "run")


def test_prepare_rejects_pair_taxonomy_mismatch(synthetic, tmp_path):
    manifest = _load_json(synthetic["manifest_path"])
    manifest["split"]["pair_assignments"][0]["pair_key"] = "CHEMBL900002|10090|P90002"
    _dump_json(synthetic["manifest_path"], manifest)
    with pytest.raises(ValueError, match="pair_taxonomy_mismatch"):
        _prepare(synthetic, tmp_path / "run")


def test_prepare_rejects_feature_bundle_hash_mismatch(synthetic, tmp_path):
    feature_manifest = _load_json(synthetic["feature_manifest_path"])
    feature_manifest["entity_bundle_hash"] = "f" * 64
    _dump_json(synthetic["feature_manifest_path"], feature_manifest)
    with pytest.raises(ValueError, match="feature_entity_bundle_hash_mismatch"):
        _prepare(synthetic, tmp_path / "run")


def test_prepare_rejects_feature_manifest_sha_mismatch(synthetic, tmp_path):
    with pytest.raises(ValueError, match="feature_manifest_sha256_mismatch"):
        pair_pipeline.prepare_pair_arrays(
            manifest_path=str(synthetic["manifest_path"]),
            feature_manifest_path=str(synthetic["feature_manifest_path"]),
            out_dir=str(tmp_path / "run"),
            expected_feature_manifest_sha256="0" * 64,
        )


def test_prepare_rejects_nonfinite_embedding(synthetic, tmp_path):
    feature_manifest = _load_json(synthetic["feature_manifest_path"])
    embedding = feature_manifest["entities"][0]["features"]["esm_embedding"]
    embedding[3] = float("nan")
    _dump_json(synthetic["feature_manifest_path"], feature_manifest, allow_nan=True)
    with pytest.raises(ValueError, match="feature_embedding_non_finite"):
        _prepare(synthetic, tmp_path / "run")


def test_prepare_rejects_wrong_embedding_dim(synthetic, tmp_path):
    feature_manifest = _load_json(synthetic["feature_manifest_path"])
    feature_manifest["entities"][0]["features"]["esm_embedding"] = (
        feature_manifest["entities"][0]["features"]["esm_embedding"][:-1]
    )
    _dump_json(synthetic["feature_manifest_path"], feature_manifest)
    with pytest.raises(ValueError, match="feature_embedding_dim_mismatch"):
        _prepare(synthetic, tmp_path / "run")


def test_prepare_rejects_test_only_feature_row(synthetic, tmp_path):
    feature_manifest = _load_json(synthetic["feature_manifest_path"])
    feature_manifest["entities"][0]["test_provenance"] = {"source": "test"}
    _dump_json(synthetic["feature_manifest_path"], feature_manifest)
    with pytest.raises(ValueError, match="feature_test_only_row_rejected"):
        _prepare(synthetic, tmp_path / "run")


def test_prepare_excludes_missing_feature_row(synthetic, tmp_path):
    feature_manifest = _load_json(synthetic["feature_manifest_path"])
    feature_manifest["entities"] = feature_manifest["entities"][:1]
    _dump_json(synthetic["feature_manifest_path"], feature_manifest)
    prepared = _prepare(synthetic, tmp_path / "run")
    assert prepared["pair_count"] == 1
    assert prepared["exclusion_count"] == 1
    exclusion = prepared["exclusions"][0]
    assert exclusion["pair_key"] == "CHEMBL900002|9606|P90002"
    assert exclusion["reason"] == "feature_row_missing"


def test_prepare_refuses_existing_output_directory(synthetic, tmp_path):
    existing = tmp_path / "run"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="output_directory_already_exists"):
        _prepare(synthetic, existing)


def test_npz_reader_rejects_object_arrays(tmp_path):
    bad = tmp_path / "bad_arrays.npz"
    np.savez(
        bad,
        features=np.array([_UnsafePayload(), _UnsafePayload()], dtype=object),
        labels=np.array([0, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError):
        pair_pipeline.load_training_arrays(bad)


# ---------------------------------------------------------------------------
# CPU contract tests: bounded budget, device selection, checkpoint, prediction
# ---------------------------------------------------------------------------

def test_train_arrays_requires_cuda_without_silent_fallback():
    if torch.cuda.is_available():
        pytest.fail(
            "the CPU preprocessing interpreter unexpectedly exposes CUDA; "
            "run this suite with E:/Anaconda/python.exe"
        )
    features = np.zeros((2, pair_pipeline.INPUT_DIM), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="training_device_must_be_cuda"):
        gpu_model.train_arrays(features, labels, seed=1, steps=1, device="cpu")
    with pytest.raises(RuntimeError, match="cuda_unavailable_for_training"):
        gpu_model.train_arrays(features, labels, seed=1, steps=1, device="cuda")


def test_validate_training_params_rejects_invalid_budget():
    base = dict(seed=1, steps=1, batch_size=2, learning_rate=1e-3, hidden_dim=8)
    for override in (
        {"steps": 0},
        {"steps": 1001},
        {"steps": True},
        {"steps": 1.5},
        {"batch_size": 0},
        {"batch_size": 17},
        {"batch_size": True},
        {"learning_rate": 0.0},
        {"learning_rate": -1.0},
        {"learning_rate": float("nan")},
        {"seed": -1},
        {"seed": True},
        {"hidden_dim": 0},
        {"hidden_dim": True},
    ):
        with pytest.raises(ValueError):
            gpu_model.validate_training_params(**{**base, **override})


def test_predict_preserves_row_order_and_single_logit():
    rng = np.random.default_rng(11)
    features = (rng.standard_normal((3, pair_pipeline.INPUT_DIM)) * 0.05).astype(np.float32)
    model = gpu_model.PairHead(hidden_dim=8)
    logits = gpu_model.predict_arrays(model, features, device="cpu")
    assert logits.shape == (3,)
    assert bool(np.isfinite(logits).all())
    reversed_logits = gpu_model.predict_arrays(model, features[::-1].copy(), device="cpu")
    assert np.allclose(logits[::-1], reversed_logits, atol=1e-7, rtol=1e-7)
    single = np.array([
        gpu_model.predict_arrays(model, features[index:index + 1], device="cpu")[0]
        for index in range(3)
    ])
    assert np.allclose(logits, single, atol=1e-7, rtol=1e-7)


def test_checkpoint_roundtrip_and_restricted_loading(tmp_path):
    rng = np.random.default_rng(5)
    features = (rng.standard_normal((2, pair_pipeline.INPUT_DIM)) * 0.05).astype(np.float32)
    torch.manual_seed(7)
    model = gpu_model.PairHead(hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.from_numpy(features)).sum()
    loss.backward()
    optimizer.step()
    baseline = gpu_model.predict_arrays(model, features, device="cpu")

    metadata = {
        "arrays_sha256": "a" * 64,
        "input_manifest_sha256": "b" * 64,
        "pair_keys": ["CHEMBL900001|9606|P90001", "CHEMBL900002|9606|P90002"],
        "purpose": pair_pipeline.PURPOSE,
        "rows": 2,
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    record = gpu_model.save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=1,
        seed=7,
        config=model.config(),
        input_metadata=metadata,
        runtime={"device": "cpu"},
    )
    assert checkpoint_path.is_file()
    assert record["sha256"] == _sha256(checkpoint_path)

    payload = gpu_model.load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        expected_sha256=record["sha256"],
        expect_input_sha256="a" * 64,
    )
    assert payload["step"] == 1
    assert payload["seed"] == 7
    assert payload["config"] == model.config()
    assert payload["input_metadata"]["pair_keys"] == metadata["pair_keys"]

    restored = gpu_model.build_model(payload["config"])
    restored.load_state_dict(payload["model_state_dict"])
    restored_logits = gpu_model.predict_arrays(restored, features, device="cpu")
    assert np.array_equal(restored_logits, baseline)

    with pytest.raises(ValueError, match="checkpoint_sha256_mismatch"):
        gpu_model.load_checkpoint(checkpoint_path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="checkpoint_input_binding_mismatch"):
        gpu_model.load_checkpoint(checkpoint_path, expect_input_sha256="f" * 64)
    with pytest.raises(FileExistsError, match="checkpoint_already_exists"):
        gpu_model.save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=1,
            seed=7,
            config=model.config(),
            input_metadata=metadata,
        )

    unsafe_path = tmp_path / "unsafe.pt"
    torch.save({"format_version": 1, "evil": _UnsafePayload()}, unsafe_path)
    with pytest.raises(ValueError, match="checkpoint_load_rejected"):
        gpu_model.load_checkpoint(unsafe_path)

    garbage_path = tmp_path / "garbage.pt"
    garbage_path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="checkpoint_load_rejected"):
        gpu_model.load_checkpoint(garbage_path)

    missing_keys = tmp_path / "missing_keys.pt"
    torch.save({"format_version": 1}, missing_keys)
    with pytest.raises(ValueError, match="checkpoint_missing_keys"):
        gpu_model.load_checkpoint(missing_keys)


def test_run_root_and_gpu_interpreter_guards(tmp_path):
    run_name = "P1_03_Artifacts_" + "0123456789abcdef0123456789abcdef"
    resolved = pair_pipeline.resolve_run_root(pair_pipeline.PROJECT_ROOT / run_name)
    assert resolved.name == run_name
    with pytest.raises(ValueError, match="run_root_name_must_be"):
        pair_pipeline.resolve_run_root(tmp_path / "not-a-run-root")
    with pytest.raises(ValueError, match="gpu_python_must_be_registered_cuda_interpreter"):
        pair_pipeline.validate_gpu_python(sys.executable)
    if GPU_PYTHON.is_file():
        assert pair_pipeline.validate_gpu_python(GPU_PYTHON) == GPU_PYTHON.resolve()


def test_cli_rejects_noncanonical_run_root(synthetic, tmp_path):
    exit_code = pair_pipeline.main([
        "train",
        "--manifest", str(synthetic["manifest_path"]),
        "--features", str(synthetic["feature_manifest_path"]),
        "--out", str(tmp_path / "not_a_run_root"),
    ])
    assert exit_code == 2


def test_git_code_version_records_real_apl_commit():
    """F1/CR-01: a 40-hex git commit must be recorded, never null."""
    record = pair_pipeline._git_code_version()
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(APL_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    expected_commit = completed.stdout.strip()
    commit = record["git_commit"]
    assert isinstance(commit, str)
    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)
    assert commit == expected_commit
    assert isinstance(record["dirty_paths"], list)
    assert all(isinstance(path, str) for path in record["dirty_paths"])


def test_environment_record_contains_platform_and_import_versions():
    """F6/WR-06: OS/platform/Python and package versions are recordable on CPU."""
    record = pair_pipeline._environment_record()
    assert record["python"] == platform.python_version()
    assert record["python_full_version"] == sys.version
    assert record["executable"] == sys.executable
    assert record["platform"] == platform.platform()
    assert record["system"] == platform.system()
    assert record["machine"] == platform.machine()
    assert isinstance(record["cuda_available"], bool)
    imports = record["imports"]
    for module_name in ("torch", "numpy", "rdkit"):
        assert module_name in imports
        assert "version" in imports[module_name]
        assert "module_path" in imports[module_name]


def test_output_location_guards_reject_frozen_and_apl_paths(tmp_path, monkeypatch):
    """F3/WR-01: new output roots are confined to fresh P1_03 GUID roots."""
    run_name = "P1_03_Artifacts_" + "0123456789abcdef0123456789abcdef"
    accepted_tmp = pair_pipeline.resolve_new_output_dir(tmp_path / "accepted_run")
    assert accepted_tmp.is_dir()
    with pytest.raises(ValueError, match="output_dir_must_not_be_inside_APL_source"):
        pair_pipeline.resolve_new_output_dir(pair_pipeline.APL_ROOT / run_name)
    with pytest.raises(ValueError, match="output_dir_under_project_root_must_be"):
        pair_pipeline.resolve_new_output_dir(
            pair_pipeline.PROJECT_ROOT / "not_a_run_root",
        )

    # Real GPU CLI entry points must reject the APL source tree before CUDA.
    with pytest.raises(ValueError, match="out_dir_must_not_be_inside_APL_source"):
        gpu_model.run_training(
            arrays_path=tmp_path / "missing.npz",
            out_dir=pair_pipeline.APL_ROOT,
            steps=1,
            batch_size=2,
        )
    with pytest.raises(ValueError, match="out_dir_must_not_be_inside_APL_source"):
        gpu_model.run_prediction(
            checkpoint_path=tmp_path / "missing.pt",
            arrays_path=tmp_path / "missing.npz",
            request_path=tmp_path / "missing.json",
            out_dir=pair_pipeline.APL_ROOT,
        )

    fake_project = tmp_path / "fake_project"
    fake_apl = fake_project / "APL"
    fake_apl.mkdir(parents=True)
    monkeypatch.setattr(gpu_model, "PROJECT_ROOT", fake_project)
    monkeypatch.setattr(gpu_model, "APL_ROOT", fake_apl)
    p1_03_root = fake_project / run_name
    p1_03_child = p1_03_root / "fit"
    p1_03_child.mkdir(parents=True)
    frozen_child = fake_project / "P1_02_03_Artifacts_frozen" / "child"
    frozen_child.mkdir(parents=True)
    assert gpu_model._validate_output_location(p1_03_root) == p1_03_root
    assert gpu_model._validate_output_location(p1_03_child) == p1_03_child
    with pytest.raises(ValueError, match="out_dir_must_not_be_inside_APL_source"):
        gpu_model._validate_output_location(fake_apl)
    with pytest.raises(ValueError, match="out_dir_under_project_root_must_be"):
        gpu_model._validate_output_location(fake_project)
    with pytest.raises(ValueError, match="out_dir_under_project_root_must_be"):
        gpu_model._validate_output_location(frozen_child)
    with pytest.raises(ValueError, match="out_dir_must_be_absolute"):
        gpu_model._validate_output_location("relative/out")
    with pytest.raises(ValueError, match="out_dir_missing"):
        gpu_model._validate_output_location(fake_project / "missing_dir")


def test_shared_training_evidence_validation_rejects_tampered_records():
    """F4/WR-02: the shared evidence gate proves head-only training and frozen ESM."""
    good = {
        "training": {
            "trainable_parameter_names": [
                "input_layer.weight", "input_layer.bias",
                "output_layer.weight", "output_layer.bias",
            ],
            "frozen_feature_input": True,
        },
    }
    validated = pair_pipeline._validate_training_evidence(good)
    assert validated["frozen_feature_input"] is True
    with pytest.raises(RuntimeError, match="gpu_training_record_missing"):
        pair_pipeline._validate_training_evidence({"training": None})
    with pytest.raises(RuntimeError, match="gpu_trainable_parameter_names_missing"):
        pair_pipeline._validate_training_evidence({
            "training": {
                "trainable_parameter_names": [],
                "frozen_feature_input": True,
            },
        })
    for forbidden_name in ("esm.encoder.weight", "backbone.head.weight"):
        with pytest.raises(
            RuntimeError, match="gpu_frozen_esm_has_trainable_parameters",
        ):
            pair_pipeline._validate_training_evidence({
                "training": {
                    "trainable_parameter_names": [forbidden_name],
                    "frozen_feature_input": True,
                },
            })
    with pytest.raises(RuntimeError, match="gpu_frozen_feature_input_not_recorded"):
        pair_pipeline._validate_training_evidence({
            "training": {
                "trainable_parameter_names": ["output_layer.weight"],
                "frozen_feature_input": False,
            },
        })


# ---------------------------------------------------------------------------
# CUDA behavior: synthetic fixture through the real GPU entry point
# ---------------------------------------------------------------------------

def test_synthetic_cuda_train_checkpoint_cpu_reload(synthetic, tmp_path):
    if not GPU_PYTHON.is_file():
        pytest.fail("registered CUDA interpreter is missing: " + str(GPU_PYTHON))
    run_dir = tmp_path / "gpu_run"
    prepared = _prepare(synthetic, run_dir)

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(APL_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    environment["PYTHONHASHSEED"] = "0"
    command = [
        str(GPU_PYTHON), "-B", "-X", "utf8", "-m", "training.gpu_model",
        "--arrays", str(prepared["arrays_path"]),
        "--out", str(run_dir),
        "--seed", "42",
        "--steps", "2",
        "--batch-size", "2",
        "--learning-rate", "0.01",
        "--hidden-dim", "16",
    ]
    completed = subprocess.run(
        command,
        cwd=str(APL_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    assert completed.returncode == 0, (
        "GPU tracer subprocess failed (exit {})\nSTDOUT:\n{}\nSTDERR:\n{}".format(
            completed.returncode, completed.stdout, completed.stderr,
        )
    )
    gpu_result = _load_json(run_dir / "gpu_result.json")
    assert gpu_result["status"] == "trained"
    assert gpu_result["device"] == "cuda"
    runtime = gpu_result["runtime"]
    assert runtime["cuda_available"] is True
    assert runtime["torch"] == "2.9.1+cu128"
    assert runtime["torch_cuda"] == "12.8"
    assert runtime["device_name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert runtime["capability"] == [12, 0]
    assert "sm_120" in runtime["arch_list"]
    assert runtime["tf32"]["cudnn_allow_tf32"] is False
    assert runtime["tf32"]["cuda_matmul_allow_tf32"] is False
    assert runtime["tf32"]["float32_dtype"] == "torch.float32"

    assert gpu_result["arrays_sha256"] == prepared["arrays_sha256"]
    assert gpu_result["input_manifest_sha256"] == prepared["input_manifest_sha256"]
    assert len(gpu_result["loss_records"]) == 2
    for record in gpu_result["loss_records"]:
        assert math.isfinite(record["loss"])
        assert math.isfinite(record["gradient_norm"])
        assert record["gradient_norm"] > 0.0
        assert math.isfinite(record["parameter_delta"])
    assert gpu_result["total_parameter_delta"] > 0.0
    assert len(gpu_result["logits"]) == 2
    assert all(math.isfinite(value) for value in gpu_result["logits"])
    assert gpu_result["gpu_reload"]["passed"] is True
    assert gpu_result["gpu_reload"]["max_abs_diff"] <= 1e-6
    # F4/WR-02: the legacy tracer path records the frozen-ESM/head-only evidence.
    assert gpu_result["training"]["frozen_feature_input"] is True
    assert gpu_result["training"]["trainable_parameter_names"]
    assert all(
        "esm" not in name.lower() and "backbone" not in name.lower()
        for name in gpu_result["training"]["trainable_parameter_names"]
    )

    checkpoint_path = run_dir / "checkpoint.pt"
    assert checkpoint_path.is_file()
    assert gpu_result["checkpoint"]["sha256"] == _sha256(checkpoint_path)
    assert (run_dir / "loss.jsonl").is_file()
    assert len((run_dir / "loss.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    # CPU portable reload of the same bytes; fixed input must agree within 1e-6.
    payload = gpu_model.load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        expected_sha256=gpu_result["checkpoint"]["sha256"],
        expect_input_sha256=prepared["arrays_sha256"],
    )
    assert payload["step"] == 2
    assert payload["seed"] == 42
    features, labels = pair_pipeline.load_training_arrays(prepared["arrays_path"])
    assert labels.tolist() == [1, 0]
    cpu_model = gpu_model.build_model(payload["config"])
    cpu_model.load_state_dict(payload["model_state_dict"])
    cpu_logits = gpu_model.predict_arrays(cpu_model, features, device="cpu")
    gpu_logits = np.asarray(gpu_result["logits"], dtype=np.float32)
    assert cpu_logits.shape == gpu_logits.shape
    max_abs_diff = float(np.max(np.abs(cpu_logits - gpu_logits)))
    assert max_abs_diff <= 1e-6
    assert np.allclose(cpu_logits, gpu_logits, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Read-only real frozen inputs: CPU preprocessing contract (no GPU training)
# ---------------------------------------------------------------------------

def test_real_frozen_inputs_prepare_60_bound_pairs(tmp_path):
    if not REAL_FROZEN_MANIFEST.is_file() or not REAL_FEATURE_MANIFEST.is_file():
        pytest.fail(
            "frozen Phase 02 manifest/feature snapshot is required for this "
            "read-only preprocessing contract test"
        )
    prepared = pair_pipeline.prepare_pair_arrays(
        manifest_path=str(REAL_FROZEN_MANIFEST),
        feature_manifest_path=str(REAL_FEATURE_MANIFEST),
        out_dir=str(tmp_path / "real_run"),
        expected_feature_manifest_sha256=pair_pipeline.EXPECTED_FEATURE_MANIFEST_SHA256,
    )
    assert prepared["pair_count"] == 60
    assert prepared["class_counts"] == {"active": 55, "inactive": 5}
    assert prepared["exclusion_count"] == 0
    pairs = prepared["pairs"]
    pair_keys = [pair["pair_key"] for pair in pairs]
    assert pair_keys == sorted(pair_keys)
    assert len(set(pair_keys)) == 60
    assert len({pair["molecule_chembl_id"] for pair in pairs}) == 41
    assert len({pair["accession"] for pair in pairs}) == 26
    features, labels = pair_pipeline.load_training_arrays(prepared["arrays_path"])
    assert features.shape == (60, pair_pipeline.INPUT_DIM)
    assert int(labels.sum()) == 55
    assert bool(np.isfinite(features).all())
    manifest = _load_json(prepared["input_manifest_path"])
    assert manifest["source"]["feature_manifest_sha256"] == (
        pair_pipeline.EXPECTED_FEATURE_MANIFEST_SHA256
    )
    assert manifest["source"]["external_revision_verified"] is False
    assert manifest["purpose"] == pair_pipeline.PURPOSE
    text = Path(prepared["input_manifest_path"]).read_text(encoding="utf-8").lower()
    for forbidden_key in ('"auc"', '"auc_score"', '"roc_auc"'):
        assert forbidden_key not in text

"""Training-only fit-diagnostic tests (T3-03 / T3-06, in-sample only).

The deterministic class-balanced sampler, the falsifiable 10%-BCE-drop /
one-correct-per-class criterion and the full-manifest majority baselines are
tested as pure functions on the CPU.  One tiny synthetic two-class fixture is
then trained through the registered CUDA interpreter to prove that the GPU
diagnostic path records finite per-step loss/gradient/parameter deltas, keeps
the ESM input frozen, and writes the diagnostic JSON.  This is an engineering
probe on synthetic data, not real-data training, and it makes no generalisation
claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from training import gpu_model, pair_pipeline


APL_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = Path("E:/paper/DDA-Reproduction/.conda/python.exe")
INPUT_DIM = gpu_model.INPUT_DIM
FP_DIM = gpu_model.INPUT_FP_DIM
ESM_DIM = gpu_model.INPUT_ESM_DIM


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


def _synthetic_features(labels, *, seed: int = 5) -> np.ndarray:
    """Linearly separable synthetic rows: ESM dim 0 carries the class signal."""
    rng = np.random.default_rng(seed)
    rows = len(labels)
    morgan = rng.integers(0, 2, size=(rows, FP_DIM)).astype(np.float32)
    esm = (rng.standard_normal((rows, ESM_DIM)) * 0.02).astype(np.float32)
    for index, label in enumerate(labels):
        esm[index, 0] = 1.0 if int(label) == 1 else -1.0
    return np.concatenate([morgan, esm], axis=1).astype(np.float32)


def _pairs(pair_keys, labels) -> list:
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


def _pair_keys(count: int) -> list:
    return [
        "CHEMBL93%04d|9606|P93%03d" % (index + 1, index + 1)
        for index in range(count)
    ]


def _write_gpu_run(
    root: Path,
    *,
    labels,
    features=None,
    fit_diagnostic=None,
    steps: int = 1000,
    batch_size: int = 16,
) -> dict:
    labels_array = np.asarray(labels, dtype=np.int64)
    keys = _pair_keys(len(labels_array))
    out = root / "gpu_run"
    out.mkdir(parents=True)
    if features is None:
        features = _synthetic_features(labels_array)
    arrays_path = out / "training_arrays.npz"
    np.savez_compressed(arrays_path, features=features, labels=labels_array)
    arrays_sha = _sha256(arrays_path)
    pairs = _pairs(keys, labels_array)
    manifest = {
        "schema_version": 1,
        "mode": "synthetic-fit-diagnostic-test",
        "purpose": pair_pipeline.PURPOSE,
        "purpose_statement": "synthetic engineering probe",
        "arrays": {
            "file": "training_arrays.npz",
            "sha256": arrays_sha,
            "bytes": arrays_path.stat().st_size,
            "dtype": "float32",
            "shape": [len(labels_array), INPUT_DIM],
            "allow_pickle": False,
        },
        "label_mapping": {"inactive": 0, "active": 1},
        "class_counts": {
            "inactive": int((labels_array == 0).sum()),
            "active": int((labels_array == 1).sum()),
        },
        "pair_count": len(pairs),
        "pairs": pairs,
        "exclusions": [],
        "exclusion_count": 0,
    }
    if fit_diagnostic is not None:
        manifest["fit_diagnostic"] = fit_diagnostic
    input_manifest_path = out / "input_manifest.json"
    _dump_json(input_manifest_path, manifest)
    return {
        "out": out,
        "arrays_path": arrays_path,
        "input_manifest_path": input_manifest_path,
        "labels": labels_array,
        "features": features,
        "steps": steps,
        "batch_size": batch_size,
    }


def _fit_plan(labels, *, sample_rows=None, seed: int = 42, per_class: int = 8) -> dict:
    labels_array = np.asarray(labels, dtype=np.int64)
    keys = _pair_keys(len(labels_array))
    if sample_rows is None:
        sample_rows = list(range(len(labels_array)))
    in_range = [row for row in sample_rows if 0 <= row < len(labels_array)]
    return {
        "plan_version": 1,
        "seed": seed,
        "per_class": per_class,
        "sample_rows": [int(row) for row in sample_rows],
        "pair_keys": [
            keys[row] if 0 <= row < len(keys) else "CHEMBL999999|9606|P999999"
            for row in sample_rows
        ],
        "labels": [
            int(labels_array[row]) if 0 <= row < len(labels_array) else 0
            for row in sample_rows
        ],
        "class_counts": {
            "inactive": sum(1 for row in in_range if int(labels_array[row]) == 0),
            "active": sum(1 for row in in_range if int(labels_array[row]) == 1),
        },
        "criterion": {
            "version": gpu_model.FIT_DIAGNOSTIC_CRITERION_VERSION,
            "max_steps": gpu_model.MAX_STEPS,
            "min_relative_bce_drop": gpu_model.FIT_DIAGNOSTIC_MIN_RELATIVE_LOSS_DROP,
            "min_correct_per_class": gpu_model.FIT_DIAGNOSTIC_MIN_CORRECT_PER_CLASS,
        },
    }


# ---------------------------------------------------------------------------
# Deterministic class-balanced sampler
# ---------------------------------------------------------------------------

def test_class_balanced_sampler_is_deterministic_and_covers_both_classes():
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    first = gpu_model.class_balanced_batch_indices(
        labels, seed=42, steps=12, batch_size=8,
    )
    second = gpu_model.class_balanced_batch_indices(
        labels, seed=42, steps=12, batch_size=8,
    )
    assert first == second
    assert len(first) == 12
    assert all(len(batch) == 8 for batch in first)
    assert all(len(set(batch)) == 8 for batch in first)
    for batch in first:
        batch_labels = labels[np.asarray(batch)]
        assert set(batch_labels.tolist()) == {0, 1}
        assert int((batch_labels == 0).sum()) == 4
    small = gpu_model.class_balanced_batch_indices(
        labels, seed=7, steps=3, batch_size=3,
    )
    assert [len(batch) for batch in small] == [3, 3, 3]
    for batch in small:
        batch_labels = labels[np.asarray(batch)]
        assert 1 <= int((batch_labels == 0).sum()) <= 2
        assert 1 <= int((batch_labels == 1).sum()) <= 2
    with pytest.raises(ValueError, match="class_balanced_sampling_requires_both_classes"):
        gpu_model.class_balanced_batch_indices(
            np.asarray([0, 0, 1], dtype=np.int64)[:2],
            seed=1, steps=1, batch_size=2,
        )


# ---------------------------------------------------------------------------
# Falsifiable fit criterion and majority baselines (pure functions)
# ---------------------------------------------------------------------------

def test_fit_criterion_passes_on_fixed_two_class_sample():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    final_logits = np.asarray([-2.0, 2.0, -1.5, 1.5], dtype=np.float32)
    criterion = gpu_model.evaluate_fit_criterion(
        initial_loss=0.7, final_loss=0.5,
        initial_logits=np.asarray([0.1, -0.1, 0.2, -0.2], dtype=np.float32),
        final_logits=final_logits, labels=labels,
    )
    assert criterion["status"] == "passed"
    assert criterion["reasons"] == []
    assert criterion["relative_drop"] >= 0.10
    assert criterion["per_class_correct"] == {"inactive": 2, "active": 2}
    assert criterion["per_class_total"] == {"inactive": 2, "active": 2}
    assert criterion["criterion_version"] == gpu_model.FIT_DIAGNOSTIC_CRITERION_VERSION


def test_fit_criterion_fails_without_ten_percent_bce_drop():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    criterion = gpu_model.evaluate_fit_criterion(
        initial_loss=0.70, final_loss=0.695,
        initial_logits=np.zeros(4, dtype=np.float32),
        final_logits=np.asarray([-2.0, 2.0, -2.0, 2.0], dtype=np.float32),
        labels=labels,
    )
    assert criterion["status"] == "failed"
    assert "relative_bce_drop_below_threshold" in criterion["reasons"]
    assert criterion["relative_drop"] < 0.10


def test_fit_criterion_fails_when_a_class_is_never_correct():
    labels = np.asarray([0, 1], dtype=np.int64)
    criterion = gpu_model.evaluate_fit_criterion(
        initial_loss=0.7, final_loss=0.1,
        initial_logits=np.zeros(2, dtype=np.float32),
        final_logits=np.asarray([-0.5, -0.5], dtype=np.float32),
        labels=labels,
    )
    assert criterion["status"] == "failed"
    assert "per_class_correct_below_one" in criterion["reasons"]
    assert criterion["per_class_correct"] == {"inactive": 1, "active": 0}


def test_majority_baseline_and_majority_collapse_detection():
    labels = np.asarray([0] * 5 + [1] * 55, dtype=np.int64)
    baseline = gpu_model.majority_baseline(labels)
    assert baseline["majority_class"] == "active"
    assert baseline["majority_class_count"] == 55
    assert baseline["majority_class_accuracy"] == pytest.approx(55 / 60)
    collapsed = gpu_model.summarize_full_manifest_fit(
        initial_bce=0.7, final_bce=0.31,
        final_logits=np.full(60, 1.0, dtype=np.float32),
        labels=labels,
    )
    assert collapsed["majority_collapse"] is True
    assert collapsed["predicted_class_counts"] == {"negative": 0, "positive": 60}
    assert collapsed["prediction_metrics"]["per_class_correct"]["inactive"] == 0
    both_classes = np.full(60, 1.0, dtype=np.float32)
    both_classes[:5] = -1.0
    recovered = gpu_model.summarize_full_manifest_fit(
        initial_bce=0.7, final_bce=0.4, final_logits=both_classes, labels=labels,
    )
    assert recovered["majority_collapse"] is False
    assert recovered["relative_bce_drop"] == pytest.approx((0.7 - 0.4) / 0.7)


# ---------------------------------------------------------------------------
# gpu_model rejects malformed fit plans before CUDA/model use
# ---------------------------------------------------------------------------

def test_run_training_rejects_invalid_fit_diagnostic_plans(tmp_path):
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    keys = _pair_keys(4)

    def expect_error(plan, code):
        run = _write_gpu_run(tmp_path / code, labels=labels, fit_diagnostic=plan)
        with pytest.raises(ValueError, match=code):
            gpu_model.run_training(
                arrays_path=run["arrays_path"], out_dir=run["out"],
                steps=1, batch_size=2,
            )

    bad_rows = _fit_plan(labels, sample_rows=[0, 1, 2, 4])
    expect_error(bad_rows, "input_manifest_fit_diagnostic_invalid")

    bad_keys = _fit_plan(labels)
    bad_keys["pair_keys"][0] = "CHEMBL999999|9606|P999999"
    expect_error(bad_keys, "fit_diagnostic_sample_binding_mismatch")

    bad_criterion = _fit_plan(labels)
    bad_criterion["criterion"]["min_relative_bce_drop"] = 0.5
    expect_error(bad_criterion, "fit_diagnostic_criterion_mismatch")

    single_class = _fit_plan(np.asarray([1, 1, 1, 1], dtype=np.int64))
    run = _write_gpu_run(
        tmp_path / "single_class",
        labels=np.asarray([1, 1, 1, 1]),
        fit_diagnostic=single_class,
    )
    with pytest.raises(ValueError, match="fit_diagnostic_sample_requires_both_classes"):
        gpu_model.run_training(
            arrays_path=run["arrays_path"], out_dir=run["out"],
            steps=1, batch_size=2,
        )
    assert len(keys) == 4  # fixture keys stay aligned with labels


# ---------------------------------------------------------------------------
# CUDA diagnostic behavior on the synthetic two-class fixture
# ---------------------------------------------------------------------------

def test_synthetic_cuda_fit_diagnostic_records_real_evidence(tmp_path):
    if not GPU_PYTHON.is_file():
        pytest.fail("registered CUDA interpreter is missing: " + str(GPU_PYTHON))
    labels = np.asarray([0] * 8 + [1] * 8, dtype=np.int64)
    run = _write_gpu_run(tmp_path, labels=labels, fit_diagnostic=_fit_plan(labels))

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(APL_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    environment["PYTHONHASHSEED"] = "0"
    command = [
        str(GPU_PYTHON), "-B", "-X", "utf8", "-m", "training.gpu_model",
        "--arrays", str(run["arrays_path"]),
        "--out", str(run["out"]),
        "--seed", "42",
        "--steps", "1000",
        "--batch-size", "16",
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
        "GPU fit-diagnostic subprocess failed (exit {})\nSTDOUT:\n{}\nSTDERR:\n{}".format(
            completed.returncode, completed.stdout, completed.stderr,
        )
    )
    result = _load_json(run["out"] / "gpu_result.json")
    assert result["status"] == "trained"
    assert result["device"] == "cuda"
    assert result["runtime"]["cuda_available"] is True
    assert result["runtime"]["device_name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert "sm_120" in result["runtime"]["arch_list"]
    assert result["runtime"]["tf32"]["cudnn_allow_tf32"] is False
    assert result["runtime"]["tf32"]["cuda_matmul_allow_tf32"] is False
    assert result["training"]["sampling"] == gpu_model.SAMPLING_CLASS_BALANCED
    assert result["training"]["frozen_feature_input"] is True
    assert set(result["training"]["trainable_parameter_names"]) == {
        "input_layer.weight", "input_layer.bias",
        "output_layer.weight", "output_layer.bias",
    }
    assert all("esm" not in name.lower() for name in result["training"]["trainable_parameter_names"])
    assert len(result["loss_records"]) == 1000
    for record in result["loss_records"]:
        assert math.isfinite(record["loss"])
        assert math.isfinite(record["gradient_norm"])
        assert record["gradient_norm"] > 0.0
        assert math.isfinite(record["parameter_delta"])
    assert result["total_parameter_delta"] > 0.0
    assert result["gpu_reload"]["passed"] is True

    fit = result["fit_diagnostic"]
    assert fit is not None
    assert fit["status"] == "passed", json.dumps(fit["reasons"])
    small = fit["small_sample"]
    assert small["device"] == "cuda"
    assert small["sampling"] == gpu_model.SAMPLING_CLASS_BALANCED
    assert small["steps_run"] <= gpu_model.MAX_STEPS
    assert small["criterion"]["relative_drop"] >= 0.10
    assert small["criterion"]["per_class_correct"]["inactive"] >= 1
    assert small["criterion"]["per_class_correct"]["active"] >= 1
    assert set(small["sample_pair_keys"]) == set(fit["small_sample"]["sample_pair_keys"])
    assert fit["full_manifest"]["majority_collapse"] is False
    assert fit["full_manifest"]["class_counts"] == {"inactive": 8, "active": 8}
    assert fit["full_manifest"]["prediction_metrics"]["per_class_total"] == {
        "inactive": 8, "active": 8,
    }
    text = json.dumps(fit, ensure_ascii=False)
    assert "NaN" not in text and "Infinity" not in text
    diagnostic_path = run["out"] / "fit_diagnostic.json"
    assert diagnostic_path.is_file()
    assert _load_json(diagnostic_path)["status"] == "passed"
    assert result["arrays_sha256"] == _sha256(run["arrays_path"])

"""Weights-only checkpoint recovery and CUDA resume-equivalence tests (T3-04).

Groups:

* CPU contract tests for the v2 checkpoint schema: weights-only loading,
  Python/NumPy/CPU-Torch recovery state, optimizer-config binding and
  fail-closed rejection of tampered/corrupt state;
* a real-CUDA equivalence test on a tiny synthetic fixture: two uninterrupted
  steps versus one step + save + resume + one step must agree on model,
  optimizer, step and RNG state and on fixed-input logits within 1e-6;
* a CPU portable reload of that same checkpoint against the GPU logits.

The synthetic fixture is an environment/behavior probe, not real-data
training; the real-frozen-data acceptance run stays deferred while the user's
GPU gate is closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from training import gpu_model, pair_pipeline


APL_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = Path("E:/paper/DDA-Reproduction/.conda/python.exe")
FP_DIM = gpu_model.INPUT_FP_DIM
ESM_DIM = gpu_model.INPUT_ESM_DIM
INPUT_DIM = gpu_model.INPUT_DIM
PAIR_KEYS = ["CHEMBL94%04d|9606|P94%03d" % (index + 1, index + 1) for index in range(8)]


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


def _synthetic_features(labels) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(11)
    morgan = rng.integers(0, 2, size=(len(labels_array), FP_DIM)).astype(np.float32)
    esm = (rng.standard_normal((len(labels_array), ESM_DIM)) * 0.02).astype(np.float32)
    for index, label in enumerate(labels_array):
        esm[index, 0] = 1.0 if int(label) == 1 else -1.0
    return np.concatenate([morgan, esm], axis=1).astype(np.float32)


def _pair_records(labels) -> list:
    records = []
    for index, (pair_key, label) in enumerate(zip(PAIR_KEYS, labels)):
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


def _build_run_template(root: Path, labels) -> dict:
    """One run-directory template: numeric arrays plus a bound input manifest."""
    labels_array = np.asarray(labels, dtype=np.int64)
    template = root / "template"
    template.mkdir(parents=True)
    features = _synthetic_features(labels_array)
    np.savez_compressed(template / "training_arrays.npz", features=features, labels=labels_array)
    arrays_sha256 = _sha256(template / "training_arrays.npz")
    pairs = _pair_records(labels_array)
    manifest = {
        "schema_version": 1,
        "mode": "synthetic-resume-recovery-test",
        "purpose": pair_pipeline.PURPOSE,
        "purpose_statement": "synthetic engineering probe",
        "arrays": {
            "file": "training_arrays.npz",
            "sha256": arrays_sha256,
            "bytes": (template / "training_arrays.npz").stat().st_size,
            "dtype": "float32",
            "shape": [int(features.shape[0]), INPUT_DIM],
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
    _dump_json(template / "input_manifest.json", manifest)
    return {
        "template": template,
        "features": features,
        "labels": labels_array,
        "arrays_sha256": arrays_sha256,
    }


def _make_run(template: dict, target: Path) -> Path:
    target.mkdir(parents=True)
    shutil.copy2(template["template"] / "training_arrays.npz", target / "training_arrays.npz")
    shutil.copy2(template["template"] / "input_manifest.json", target / "input_manifest.json")
    return target


def _gpu_environment() -> dict:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(APL_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _gpu_call(arguments: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    command = [str(GPU_PYTHON), "-B", "-X", "utf8", "-m", "training.gpu_model"] + arguments
    return subprocess.run(
        command,
        cwd=str(APL_ROOT),
        env=_gpu_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _assert_process_ok(completed: subprocess.CompletedProcess, label: str) -> None:
    assert completed.returncode == 0, (
        "{} failed (exit {})\nSTDOUT:\n{}\nSTDERR:\n{}".format(
            label, completed.returncode, completed.stdout, completed.stderr,
        )
    )


def _tensor_max_abs_diff(left, right) -> float:
    left_array = np.asarray(left.detach().to("cpu").to(torch.float64), dtype=np.float64)
    right_array = np.asarray(right.detach().to("cpu").to(torch.float64), dtype=np.float64)
    return float(np.max(np.abs(left_array - right_array)))


# ---------------------------------------------------------------------------
# CPU contract: v2 weights-only checkpoint and recovery state
# ---------------------------------------------------------------------------

def _cpu_checkpoint(tmp_path: Path) -> dict:
    torch.manual_seed(21)
    features = torch.randn(4, INPUT_DIM) * 0.05
    model = gpu_model.PairHead(hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(features).sum()
    loss.backward()
    optimizer.step()
    metadata = {
        "arrays_sha256": "a" * 64,
        "input_manifest_sha256": "b" * 64,
        "pair_keys": ["KEYSYNTH1|9606|P0001", "KEYSYNTH2|9606|P0002"],
        "purpose": pair_pipeline.PURPOSE,
        "rows": 2,
    }
    path = tmp_path / "checkpoint.pt"
    record = gpu_model.save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        seed=7,
        config=model.config(),
        input_metadata=metadata,
        runtime={"training_device": "cpu", "device_name": None},
        training_config={"batch_size": 2, "sampling": gpu_model.SAMPLING_LEGACY},
    )
    return {"path": path, "record": record, "metadata": metadata}


def test_checkpoint_v2_is_weights_only_with_recovery_state(tmp_path):
    saved = _cpu_checkpoint(tmp_path)
    path = saved["path"]
    payload = gpu_model.load_checkpoint(
        path,
        expected_sha256=saved["record"]["sha256"],
        expect_input_sha256="a" * 64,
    )
    assert payload["format_version"] == gpu_model.CHECKPOINT_FORMAT_VERSION
    assert payload["step"] == payload["global_step"] == 1
    assert payload["seed"] == 7
    assert payload["optimizer_config"]["name"] == "adam"
    assert payload["optimizer_config"]["learning_rate"] == 1e-3
    assert payload["training_config"] == {
        "batch_size": 2, "sampling": gpu_model.SAMPLING_LEGACY,
    }
    for field in (
        "torch_rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_generator_state",
    ):
        assert field in payload
    # The restricted unpickler must read the payload without any allowlist work.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) >= set(gpu_model._CHECKPOINT_KEYS)
    assert isinstance(raw["python_rng_state"], tuple)
    assert isinstance(raw["numpy_rng_state"], dict)
    assert isinstance(raw["torch_cpu_generator_state"], torch.Tensor)


def test_apply_recovery_rng_states_restores_exact_state():
    states = gpu_model._capture_recovery_rng_states(gpu_model._make_cpu_generator(42))
    random_state = states["python_rng_state"]
    numpy_state = states["numpy_rng_state"]
    torch_state = states["torch_rng_state"]
    generator_state = states["torch_cpu_generator_state"]
    # Move every RNG and confirm the checkpoint copy wins.
    import random
    random.random()
    np.random.rand(5)
    torch.rand(5)
    generator = gpu_model._make_cpu_generator(99)
    torch.randperm(8, generator=generator)
    restored = gpu_model.apply_recovery_rng_states(states, generator=generator)
    assert random.getstate() == random_state
    assert gpu_model._serialize_numpy_rng_state() == numpy_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.equal(generator.get_state(), generator_state)
    assert restored["torch_rng_state"].shape == torch_state.shape


def test_load_checkpoint_rejects_tampered_state(tmp_path):
    saved = _cpu_checkpoint(tmp_path)
    base = torch.load(saved["path"], map_location="cpu", weights_only=True)

    def tampered(name: str, mutator) -> Path:
        payload = torch.load(saved["path"], map_location="cpu", weights_only=True)
        mutator(payload)
        target = tmp_path / (name + ".pt")
        torch.save(payload, target)
        return target

    unknown = tampered("unknown", lambda payload: payload.update({"rogue": 1}))
    with pytest.raises(ValueError, match="checkpoint_unknown_keys"):
        gpu_model.load_checkpoint(unknown)

    missing_optimizer = tampered(
        "missing_optimizer",
        lambda payload: payload.pop("optimizer_state_dict"),
    )
    with pytest.raises(ValueError, match="checkpoint_missing_keys"):
        gpu_model.load_checkpoint(missing_optimizer)

    def _bad_version(payload):
        payload["format_version"] = 99
    with pytest.raises(ValueError, match="checkpoint_format_unsupported"):
        gpu_model.load_checkpoint(tampered("version", _bad_version))

    def _global_step(payload):
        payload["global_step"] = 5
    with pytest.raises(ValueError, match="checkpoint_global_step_mismatch"):
        gpu_model.load_checkpoint(tampered("global_step", _global_step))

    def _nan_weight(payload):
        weight = payload["model_state_dict"]["input_layer.weight"].clone()
        weight[0, 0] = float("nan")
        payload["model_state_dict"]["input_layer.weight"] = weight
    with pytest.raises(ValueError, match="checkpoint_model_state_non_finite"):
        gpu_model.load_checkpoint(tampered("nan_weight", _nan_weight))

    def _bad_shape(payload):
        payload["model_state_dict"]["input_layer.weight"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="checkpoint_model_state_shape_mismatch"):
        gpu_model.load_checkpoint(tampered("shape", _bad_shape))

    def _empty_optimizer(payload):
        payload["optimizer_state_dict"] = {}
    with pytest.raises(ValueError, match="checkpoint_optimizer_state_invalid"):
        gpu_model.load_checkpoint(tampered("empty_optimizer", _empty_optimizer))

    def _optimizer_lr(payload):
        payload["optimizer_config"]["learning_rate"] = 0.5
    with pytest.raises(ValueError, match="checkpoint_optimizer_config_mismatch"):
        gpu_model.load_checkpoint(tampered("optimizer_lr", _optimizer_lr))

    def _bad_rng(payload):
        payload["numpy_rng_state"] = {"algorithm": "MT19937"}
    with pytest.raises(ValueError, match="checkpoint_numpy_rng_state_invalid"):
        gpu_model.load_checkpoint(tampered("rng", _bad_rng))

    with pytest.raises(ValueError, match="checkpoint_sha256_mismatch"):
        gpu_model.load_checkpoint(saved["path"], expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="checkpoint_input_binding_mismatch"):
        gpu_model.load_checkpoint(saved["path"], expect_input_sha256="f" * 64)
    assert base["step"] == 1


# ---------------------------------------------------------------------------
# CPU contract: resume has no silent CPU branch
# ---------------------------------------------------------------------------

def test_resume_training_has_no_cpu_fallback(tmp_path):
    if torch.cuda.is_available():
        pytest.fail(
            "the CPU preprocessing interpreter unexpectedly exposes CUDA; "
            "run this suite with E:/Anaconda/python.exe",
        )
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(RuntimeError, match="cuda_unavailable_for_training"):
        gpu_model.resume_training(
            resume_run=source,
            out_dir=out,
            additional_steps=1,
        )


def test_resume_cli_reports_blocked_without_cuda(tmp_path, capsys):
    if torch.cuda.is_available():
        pytest.fail("run this suite with E:/Anaconda/python.exe")
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    exit_code = gpu_model.main([
        "--resume-run", str(source),
        "--out", str(out),
        "--steps", "1",
    ])
    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.err.strip())
    assert payload["status"] == "blocked"
    assert "cuda_unavailable_for_training" in payload["reason"]


# ---------------------------------------------------------------------------
# CUDA behavior: uninterrupted two steps versus one step plus resume
# ---------------------------------------------------------------------------

def test_cuda_resume_matches_uninterrupted_trajectory(tmp_path):
    if not GPU_PYTHON.is_file():
        pytest.fail("registered CUDA interpreter is missing: " + str(GPU_PYTHON))
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    template = _build_run_template(tmp_path, labels)
    continuous = _make_run(template, tmp_path / "continuous")
    interrupted = _make_run(template, tmp_path / "interrupted")
    resumed = tmp_path / "resumed"
    resumed.mkdir(parents=True)
    common = [
        "--batch-size", "4",
        "--learning-rate", "0.01",
        "--hidden-dim", "8",
        "--seed", "42",
    ]
    _assert_process_ok(_gpu_call([
        "--arrays", str(continuous / "training_arrays.npz"),
        "--out", str(continuous),
        "--steps", "2",
        *common,
    ]), "uninterrupted two-step run")
    _assert_process_ok(_gpu_call([
        "--arrays", str(interrupted / "training_arrays.npz"),
        "--out", str(interrupted),
        "--steps", "1",
        *common,
    ]), "first interrupted step")
    _assert_process_ok(_gpu_call([
        "--resume-run", str(interrupted),
        "--out", str(resumed),
        "--steps", "1",
    ]), "resume step")

    continuous_result = _load_json(continuous / "gpu_result.json")
    resume_result = _load_json(resumed / "resume_result.json")
    first_result = _load_json(interrupted / "gpu_result.json")
    assert first_result["device"] == "cuda"
    assert resume_result["status"] == "resumed"
    assert resume_result["device"] == "cuda"
    assert resume_result["start_step"] == 1
    assert resume_result["additional_steps"] == 1
    assert resume_result["final_step"] == 2
    assert resume_result["runtime"]["training_device"] == "cuda"
    assert resume_result["runtime"]["resumed_from"]["start_step"] == 1
    assert resume_result["runtime"]["resumed_from"]["checkpoint_sha256"] == (
        first_result["checkpoint"]["sha256"]
    )

    uninterrupted = gpu_model.load_checkpoint(
        continuous / "checkpoint.pt",
        expected_sha256=continuous_result["checkpoint"]["sha256"],
        expect_input_sha256=template["arrays_sha256"],
    )
    continued = gpu_model.load_checkpoint(
        resumed / "checkpoint.pt",
        expected_sha256=resume_result["checkpoint"]["sha256"],
    )
    assert uninterrupted["step"] == continued["step"] == 2
    assert uninterrupted["global_step"] == continued["global_step"] == 2
    assert uninterrupted["seed"] == continued["seed"] == 42
    assert uninterrupted["config"] == continued["config"]
    assert uninterrupted["optimizer_config"] == continued["optimizer_config"]
    assert uninterrupted["training_config"] == continued["training_config"]
    for name, tensor in uninterrupted["model_state_dict"].items():
        other = continued["model_state_dict"][name]
        assert _tensor_max_abs_diff(tensor, other) <= 1e-6
        assert torch.allclose(tensor, other, atol=1e-6, rtol=1e-6)
    for param_id, entry in uninterrupted["optimizer_state_dict"]["state"].items():
        other_entry = continued["optimizer_state_dict"]["state"][param_id]
        assert set(entry) == set(other_entry)
        assert _tensor_max_abs_diff(entry["step"], other_entry["step"]) <= 1e-6
        for field in ("exp_avg", "exp_avg_sq"):
            assert _tensor_max_abs_diff(entry[field], other_entry[field]) <= 1e-6
            assert torch.allclose(entry[field], other_entry[field], atol=1e-6, rtol=1e-6)
    assert torch.equal(uninterrupted["torch_rng_state"], continued["torch_rng_state"])
    assert uninterrupted["python_rng_state"] == continued["python_rng_state"]
    assert uninterrupted["numpy_rng_state"] == continued["numpy_rng_state"]
    assert torch.equal(
        uninterrupted["torch_cpu_generator_state"],
        continued["torch_cpu_generator_state"],
    )

    continuous_logits = np.asarray(continuous_result["logits"], dtype=np.float64)
    resumed_logits = np.asarray(resume_result["logits"], dtype=np.float64)
    assert continuous_logits.shape == resumed_logits.shape == (len(labels),)
    assert float(np.max(np.abs(continuous_logits - resumed_logits))) <= 1e-6
    assert np.allclose(continuous_logits, resumed_logits, atol=1e-6, rtol=1e-6)
    assert math.isfinite(resume_result["final_loss"])
    assert math.isfinite(resume_result["gpu_reload"]["max_abs_diff"])
    assert resume_result["gpu_reload"]["passed"] is True

    runtime = uninterrupted["runtime"]
    assert runtime["training_device"] == "cuda"
    assert runtime["device_name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert runtime["capability"] == [12, 0]
    assert runtime["torch"] == "2.9.1+cu128"
    assert runtime["torch_cuda"] == "12.8"

    # CPU portable reload of the exact uninterrupted checkpoint bytes.
    features, stored_labels = pair_pipeline.load_training_arrays(
        continuous / "training_arrays.npz",
    )
    assert np.array_equal(features, template["features"])
    assert stored_labels.tolist() == labels.tolist()
    cpu_model = gpu_model.build_model(uninterrupted["config"])
    cpu_model.load_state_dict(uninterrupted["model_state_dict"])
    cpu_logits = gpu_model.predict_arrays(cpu_model, features, device="cpu")
    assert cpu_logits.shape == continuous_logits.shape
    assert float(np.max(np.abs(cpu_logits.astype(np.float64) - continuous_logits))) <= 1e-6
    assert np.allclose(cpu_logits, continuous_logits, atol=1e-6, rtol=1e-6)

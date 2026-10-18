"""CUDA training head, safe checkpoint and portable prediction for pair logits.

Import closure: PyTorch, NumPy and the standard library only.  The registered
CUDA interpreter (E:/paper/DDA-Reproduction/.conda/python.exe) can therefore
import this module without RDKit or pytest.  The CPU preprocessing half lives in
training.pair_pipeline.

Contract highlights:
- real training accepts only device "cuda"; there is no silent CPU fallback;
- TF32 is explicitly disabled for both cuDNN and CUDA matmul, and the resulting
  float32 diagnostics are returned to the caller for the run record;
- checkpoints contain state dicts plus primitive/tensor metadata only, are
  written atomically inside a new run directory, refuse to overwrite, and are
  always read back with torch.load(..., weights_only=True, map_location=...);
- predict_arrays emits one finite float32 logit per input row in input order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


INPUT_FP_DIM = 2048
INPUT_ESM_DIM = 320
INPUT_DIM = INPUT_FP_DIM + INPUT_ESM_DIM

DEFAULT_HIDDEN_DIM = 64
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-3
DEFAULT_STEPS = 1
MAX_STEPS = 1000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 16

SAMPLING_LEGACY = "legacy_permutation"
SAMPLING_CLASS_BALANCED = "class_balanced"
VALID_SAMPLING_MODES = (SAMPLING_LEGACY, SAMPLING_CLASS_BALANCED)

FIT_DIAGNOSTIC_CRITERION_VERSION = "phase03-fit-diagnostic-v1"
FIT_DIAGNOSTIC_MIN_RELATIVE_LOSS_DROP = 0.10
FIT_DIAGNOSTIC_MIN_CORRECT_PER_CLASS = 1
FIT_DIAGNOSTIC_RELATIVE_DROP_EPSILON = 1e-12
FIT_DIAGNOSTIC_SCOPE = (
    "training-only in-sample engineering fit diagnostic; no independent "
    "holdout, no generalisation/AUC/calibration/ranking claim"
)

CHECKPOINT_FORMAT_VERSION = 1
MODEL_ARCHITECTURE = "mlp_relu_logit_v1"
ALLCLOSE_ATOL = 1e-6
ALLCLOSE_RTOL = 1e-6

_CHECKPOINT_KEYS = (
    "format_version",
    "model_state_dict",
    "optimizer_state_dict",
    "step",
    "seed",
    "config",
    "input_metadata",
    "torch_rng_state",
    "runtime",
    "created_at",
)
_SHA256_PATTERN_LENGTH = 64


# ---------------------------------------------------------------------------
# Small validation helpers (stable error codes are part of the contract)
# ---------------------------------------------------------------------------

def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_PATTERN_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError("invalid_sha256:" + name)
    return value


def _require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError("invalid_" + name + ":" + repr(value))
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("invalid_" + name + ":" + repr(value))
    return value


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_params(
    *,
    seed: Any,
    steps: Any,
    batch_size: Any,
    learning_rate: Any,
    hidden_dim: Any,
) -> dict[str, Any]:
    """Validate the bounded training budget (T3-03 / edge coverage)."""
    checked_seed = _require_int(seed, "seed", 0, 2 ** 63 - 1)
    checked_steps = _require_int(steps, "steps", 1, MAX_STEPS)
    checked_batch = _require_int(batch_size, "batch_size", MIN_BATCH_SIZE, MAX_BATCH_SIZE)
    checked_hidden = _require_positive_int(hidden_dim, "hidden_dim")
    if checked_hidden > 1024:
        raise ValueError("invalid_hidden_dim:" + repr(hidden_dim))
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError("invalid_learning_rate:" + repr(learning_rate))
    checked_lr = float(learning_rate)
    if not math.isfinite(checked_lr) or checked_lr <= 0.0:
        raise ValueError("invalid_learning_rate:" + repr(learning_rate))
    return {
        "seed": checked_seed,
        "steps": checked_steps,
        "batch_size": checked_batch,
        "learning_rate": checked_lr,
        "hidden_dim": checked_hidden,
    }


def validate_features(features: Any, *, expected_dim: int = INPUT_DIM) -> np.ndarray:
    """Validate a numeric finite float feature matrix before tensor creation."""
    if not isinstance(features, np.ndarray):
        raise ValueError("features_not_numpy_array")
    if features.ndim != 2:
        raise ValueError("features_not_2d")
    if features.shape[0] < 1:
        raise ValueError("empty_features")
    if features.shape[1] != expected_dim:
        raise ValueError("feature_dim_mismatch:" + str(features.shape[1]))
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError("features_dtype_not_floating:" + str(features.dtype))
    if not bool(np.isfinite(features).all()):
        raise ValueError("non_finite_feature")
    return np.ascontiguousarray(features, dtype=np.float32)


def validate_labels(labels: Any, row_count: int) -> np.ndarray:
    """Validate integer binary labels aligned to the feature rows."""
    if not isinstance(labels, np.ndarray):
        raise ValueError("labels_not_numpy_array")
    if labels.ndim != 1:
        raise ValueError("labels_not_1d")
    if labels.shape[0] != row_count:
        raise ValueError("label_count_mismatch")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("label_dtype_not_integer:" + str(labels.dtype))
    values = np.ascontiguousarray(labels.astype(np.int64))
    if not bool(np.isin(values, np.array([0, 1], dtype=np.int64)).all()):
        raise ValueError("label_out_of_domain")
    return values


def validate_arrays(features: Any, labels: Any, *, expected_dim: int = INPUT_DIM):
    features_array = validate_features(features, expected_dim=expected_dim)
    labels_array = validate_labels(labels, int(features_array.shape[0]))
    return features_array, labels_array


def validate_feature_blocks(features: Any, *, expected_dim: int = INPUT_DIM) -> np.ndarray:
    """Validate the Morgan(2048)/ESM(320) block contract before tensor creation."""
    if not isinstance(features, np.ndarray):
        raise ValueError("features_not_numpy_array")
    if features.ndim != 2:
        raise ValueError("features_not_2d")
    if features.shape[0] < 1:
        raise ValueError("empty_features")
    if features.shape[1] != expected_dim:
        raise ValueError("feature_dim_mismatch:" + str(features.shape[1]))
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError("features_dtype_not_floating:" + str(features.dtype))
    morgan = features[:, :INPUT_FP_DIM]
    esm = features[:, INPUT_FP_DIM:]
    if morgan.shape[1] != INPUT_FP_DIM:
        raise ValueError("morgan_block_dim_mismatch:" + str(morgan.shape[1]))
    if not bool(np.isfinite(morgan).all()):
        raise ValueError("morgan_block_non_finite")
    if not bool(np.isin(morgan, np.array([0.0, 1.0], dtype=morgan.dtype)).all()):
        raise ValueError("morgan_block_not_binary")
    if esm.shape[1] != INPUT_ESM_DIM:
        raise ValueError("esm_block_dim_mismatch:" + str(esm.shape[1]))
    if not bool(np.isfinite(esm).all()):
        raise ValueError("esm_block_non_finite")
    return np.ascontiguousarray(features, dtype=np.float32)


def validate_training_input(
    features: Any, labels: Any, *, pair_keys: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fail-closed validation shared by legacy and development training runs."""
    features_array = validate_feature_blocks(features)
    labels_array = validate_labels(labels, int(features_array.shape[0]))
    if pair_keys is not None:
        if not isinstance(pair_keys, (list, tuple)):
            raise ValueError("pair_keys_not_sequence")
        keys = list(pair_keys)
        if len(keys) != int(features_array.shape[0]):
            raise ValueError("pair_key_count_mismatch")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("pair_key_invalid")
        if len(set(keys)) != len(keys):
            raise ValueError("pair_key_duplicate")
    return features_array, labels_array


# ---------------------------------------------------------------------------
# Runtime configuration (TF32 off, float32 on) and device report
# ---------------------------------------------------------------------------

def configure_runtime() -> dict[str, Any]:
    """Disable TF32 for both cuDNN and CUDA matmul and record float32 settings."""
    cudnn = torch.backends.cudnn
    cuda_matmul = getattr(torch.backends.cuda, "matmul", None)
    if cuda_matmul is None:
        raise RuntimeError("torch_backends_cuda_matmul_unavailable")
    cudnn.allow_tf32 = False
    cuda_matmul.allow_tf32 = False
    cudnn.deterministic = True
    cudnn.benchmark = False
    diagnostics = {
        "float32_dtype": "torch.float32",
        "cudnn_allow_tf32": bool(cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(cuda_matmul.allow_tf32),
        "cudnn_deterministic": bool(cudnn.deterministic),
        "cudnn_benchmark": bool(cudnn.benchmark),
    }
    if diagnostics["cudnn_allow_tf32"] or diagnostics["cuda_matmul_allow_tf32"]:
        raise RuntimeError("tf32_disable_failed")
    return diagnostics


def cuda_runtime_report() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        # str() keeps torch.TorchVersion (a str subclass) out of the checkpoint
        # payload, which must remain loadable with weights_only=True.
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": available,
        "device_count": int(torch.cuda.device_count()),
        "device_name": None,
        "capability": None,
        "arch_list": [],
    }
    if available:
        report["device_name"] = torch.cuda.get_device_name(0)
        report["capability"] = list(torch.cuda.get_device_capability(0))
        report["arch_list"] = list(torch.cuda.get_arch_list())
    return report


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _validate_model_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("model_config_not_mapping")
    architecture = config.get("architecture")
    if architecture != MODEL_ARCHITECTURE:
        raise ValueError("unsupported_model_architecture:" + repr(architecture))
    input_dim = config.get("input_dim")
    if type(input_dim) is not int or input_dim != INPUT_DIM:
        raise ValueError("model_input_dim_mismatch")
    hidden_dim = config.get("hidden_dim")
    if type(hidden_dim) is not int or not 1 <= hidden_dim <= 1024:
        raise ValueError("invalid_hidden_dim:" + repr(hidden_dim))
    return {
        "architecture": MODEL_ARCHITECTURE,
        "input_dim": INPUT_DIM,
        "hidden_dim": hidden_dim,
        "activation": "relu",
        "output_dim": 1,
    }


def build_model(config: Any) -> "PairHead":
    checked = _validate_model_config(config)
    return PairHead(input_dim=checked["input_dim"], hidden_dim=checked["hidden_dim"])


class PairHead(nn.Module):
    """A small deterministic MLP head producing one raw logit per pair row."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        checked_input = _require_positive_int(input_dim, "input_dim")
        checked_hidden = _require_positive_int(hidden_dim, "hidden_dim")
        if checked_input != INPUT_DIM:
            raise ValueError("model_input_dim_mismatch:" + repr(input_dim))
        self.input_dim = checked_input
        self.hidden_dim = checked_hidden
        self.input_layer = nn.Linear(checked_input, checked_hidden)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(checked_hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise ValueError("model_input_not_tensor")
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("model_input_shape_mismatch")
        return self.output_layer(self.activation(self.input_layer(features))).squeeze(-1)

    def config(self) -> dict[str, Any]:
        return {
            "architecture": MODEL_ARCHITECTURE,
            "input_dim": int(self.input_dim),
            "hidden_dim": int(self.hidden_dim),
            "activation": "relu",
            "output_dim": 1,
        }


# ---------------------------------------------------------------------------
# Training on CUDA (ESM/fingerprint input stays outside autograd)
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))


def _snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to("cpu").clone()
        for name, parameter in model.named_parameters()
    }


def _parameter_delta(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        before = snapshot[name]
        total += float((parameter.detach().to("cpu") - before).pow(2).sum().item())
    return math.sqrt(total)


def _gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu().item())
    return math.sqrt(total)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    inactive = int((labels == 0).sum())
    active = int((labels == 1).sum())
    return {"inactive": inactive, "active": active}


def _predicted_class_counts(logits: np.ndarray) -> dict[str, int]:
    positive = int((logits > 0.0).sum())
    return {"negative": int(logits.shape[0]) - positive, "positive": positive}


def class_balanced_batch_indices(
    labels: Any, *, seed: Any, steps: Any, batch_size: Any,
) -> list[list[int]]:
    """Deterministic class-balanced sampling for engineering fit diagnostics.

    Every batch contains both classes whenever both are present in the labels;
    the leading class alternates each step.  Row labels/order are never changed.
    """
    labels_array = validate_labels(labels, int(np.asarray(labels).shape[0]))
    checked_seed = _require_int(seed, "seed", 0, 2 ** 63 - 1)
    checked_steps = _require_int(steps, "steps", 1, MAX_STEPS)
    checked_batch = _require_int(
        batch_size, "batch_size", MIN_BATCH_SIZE, MAX_BATCH_SIZE,
    )
    inactive = np.flatnonzero(labels_array == 0)
    active = np.flatnonzero(labels_array == 1)
    if inactive.size == 0 or active.size == 0:
        raise ValueError("class_balanced_sampling_requires_both_classes")
    generator = np.random.default_rng(checked_seed)
    generator.shuffle(inactive)
    generator.shuffle(active)
    pools = {
        0: [int(index) for index in inactive],
        1: [int(index) for index in active],
    }
    positions = {0: 0, 1: 0}
    batches: list[list[int]] = []
    for step in range(checked_steps):
        second_class_count = checked_batch // 2
        counts = {0: second_class_count, 1: checked_batch - second_class_count}
        if step % 2 == 1:
            counts = {0: counts[1], 1: counts[0]}
        batch: list[int] = []
        for class_label in (0, 1):
            pool = pools[class_label]
            position = positions[class_label]
            for _ in range(counts[class_label]):
                batch.append(pool[position % len(pool)])
                position += 1
            positions[class_label] = position
        batches.append(batch)
    return batches


def prediction_metrics(logits: Any, labels: Any) -> dict[str, Any]:
    """Confusion / per-class correctness for raw logits at the zero threshold."""
    logits_array = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits_array.shape[0] != labels_array.shape[0] or logits_array.size == 0:
        raise ValueError("prediction_metric_shape_mismatch")
    if not bool(np.isin(labels_array, np.array([0, 1], dtype=np.int64)).all()):
        raise ValueError("label_out_of_domain")
    if not bool(np.isfinite(logits_array).all()):
        raise ValueError("prediction_metric_non_finite_logits")
    predicted_positive = logits_array > 0.0
    inactive = labels_array == 0
    active = labels_array == 1
    return {
        "predicted_class_counts": {
            "negative": int((~predicted_positive).sum()),
            "positive": int(predicted_positive.sum()),
        },
        "per_class_total": {
            "inactive": int(inactive.sum()),
            "active": int(active.sum()),
        },
        "per_class_correct": {
            "inactive": int((inactive & ~predicted_positive).sum()),
            "active": int((active & predicted_positive).sum()),
        },
        "confusion": {
            "inactive_predicted_inactive": int((inactive & ~predicted_positive).sum()),
            "inactive_predicted_active": int((inactive & predicted_positive).sum()),
            "active_predicted_inactive": int((active & ~predicted_positive).sum()),
            "active_predicted_active": int((active & predicted_positive).sum()),
        },
    }


def majority_baseline(labels: Any) -> dict[str, Any]:
    """Majority-class baseline used to expose constant-majority predictions."""
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_array.size == 0:
        raise ValueError("empty_labels")
    if not bool(np.isin(labels_array, np.array([0, 1], dtype=np.int64)).all()):
        raise ValueError("label_out_of_domain")
    inactive = int((labels_array == 0).sum())
    active = int((labels_array == 1).sum())
    majority_class = "active" if active >= inactive else "inactive"
    majority_count = max(active, inactive)
    return {
        "majority_class": majority_class,
        "majority_class_count": majority_count,
        "class_counts": {"inactive": inactive, "active": active},
        "majority_class_accuracy": majority_count / int(labels_array.size),
    }


def _mean_bce_from_logits(logits: Any, labels: Any) -> float:
    logits_array = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels_array = np.asarray(labels, dtype=np.float64).reshape(-1)
    if logits_array.shape != labels_array.shape or logits_array.size == 0:
        raise ValueError("prediction_metric_shape_mismatch")
    if not bool(np.isfinite(logits_array).all()):
        raise ValueError("prediction_metric_non_finite_logits")
    values = np.logaddexp(0.0, logits_array) - labels_array * logits_array
    return float(values.mean())


def evaluate_fit_criterion(
    *,
    initial_loss: Any,
    final_loss: Any,
    initial_logits: Any,
    final_logits: Any,
    labels: Any,
) -> dict[str, Any]:
    """Falsifiable fixed-sample fit criterion: >=10% BCE drop and each class hit."""
    if (
        isinstance(initial_loss, bool) or not isinstance(initial_loss, (int, float))
        or isinstance(final_loss, bool) or not isinstance(final_loss, (int, float))
        or not math.isfinite(float(initial_loss))
        or not math.isfinite(float(final_loss))
        or float(initial_loss) <= 0.0
    ):
        raise ValueError("fit_diagnostic_loss_invalid")
    metrics = prediction_metrics(final_logits, labels)
    initial_value = float(initial_loss)
    final_value = float(final_loss)
    relative_drop = (initial_value - final_value) / initial_value
    reasons: list[str] = []
    if not (relative_drop + FIT_DIAGNOSTIC_RELATIVE_DROP_EPSILON
            >= FIT_DIAGNOSTIC_MIN_RELATIVE_LOSS_DROP):
        reasons.append("relative_bce_drop_below_threshold")
    for label_name in ("inactive", "active"):
        if metrics["per_class_correct"][label_name] < FIT_DIAGNOSTIC_MIN_CORRECT_PER_CLASS:
            reasons.append("per_class_correct_below_one")
            break
    return {
        "criterion_version": FIT_DIAGNOSTIC_CRITERION_VERSION,
        "scope": FIT_DIAGNOSTIC_SCOPE,
        "status": "failed" if reasons else "passed",
        "reasons": reasons,
        "initial_bce": initial_value,
        "final_bce": final_value,
        "relative_drop": relative_drop,
        "min_relative_bce_drop": FIT_DIAGNOSTIC_MIN_RELATIVE_LOSS_DROP,
        "min_correct_per_class": FIT_DIAGNOSTIC_MIN_CORRECT_PER_CLASS,
        "per_class_correct": metrics["per_class_correct"],
        "per_class_total": metrics["per_class_total"],
        "confusion": metrics["confusion"],
        "predicted_class_counts": metrics["predicted_class_counts"],
    }


def summarize_full_manifest_fit(
    *, initial_bce: Any, final_bce: Any, final_logits: Any, labels: Any,
) -> dict[str, Any]:
    """In-sample loss/prediction summary used to detect majority-class collapse."""
    initial_value = float(initial_bce)
    final_value = float(final_bce)
    if not (math.isfinite(initial_value) and math.isfinite(final_value)) or initial_value <= 0.0:
        raise ValueError("fit_diagnostic_loss_invalid")
    metrics = prediction_metrics(final_logits, labels)
    baseline = majority_baseline(labels)
    total = baseline["class_counts"]["inactive"] + baseline["class_counts"]["active"]
    if baseline["majority_class"] == "active":
        majority_predictions = metrics["predicted_class_counts"]["positive"]
    else:
        majority_predictions = metrics["predicted_class_counts"]["negative"]
    return {
        "scope": FIT_DIAGNOSTIC_SCOPE,
        "initial_bce": initial_value,
        "final_bce": final_value,
        "relative_bce_drop": (initial_value - final_value) / initial_value,
        "class_counts": baseline["class_counts"],
        "predicted_class_counts": metrics["predicted_class_counts"],
        "prediction_metrics": metrics,
        "majority_baseline": baseline,
        "majority_collapse": bool(majority_predictions == total),
    }


def train_arrays(
    features: Any,
    labels: Any,
    *,
    seed: Any,
    steps: Any,
    batch_size: Any = DEFAULT_BATCH_SIZE,
    learning_rate: Any = DEFAULT_LR,
    device: Any = "cuda",
    hidden_dim: Any = DEFAULT_HIDDEN_DIM,
    sampling: Any = SAMPLING_LEGACY,
) -> dict[str, Any]:
    """Run the bounded CUDA optimizer update used by the Phase 03 tracer.

    The Morgan/ESM feature matrix is a constant tensor: only the small head
    parameters produce gradients.  Returns the model, optimizer and a JSON-ready
    record bundle; raises instead of falling back to CPU on any CUDA problem.
    """
    params = validate_training_params(
        seed=seed, steps=steps, batch_size=batch_size,
        learning_rate=learning_rate, hidden_dim=hidden_dim,
    )
    if not isinstance(device, str) or device != "cuda":
        raise ValueError("training_device_must_be_cuda:" + repr(device))
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("cuda_unavailable_for_training")
    if sampling not in VALID_SAMPLING_MODES:
        raise ValueError("invalid_sampling_mode:" + repr(sampling))

    diagnostics = configure_runtime()
    runtime = cuda_runtime_report()
    runtime["tf32"] = diagnostics

    features_array, labels_array = validate_arrays(features, labels)
    _seed_everything(params["seed"])

    device_object = torch.device("cuda")
    inputs = torch.from_numpy(features_array).to(device=device_object)
    inputs.requires_grad_(False)
    targets = torch.from_numpy(labels_array.astype(np.float32)).to(device_object)

    model = PairHead(hidden_dim=params["hidden_dim"]).to(device_object)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_function = nn.BCEWithLogitsLoss()

    row_count = int(features_array.shape[0])
    if sampling == SAMPLING_CLASS_BALANCED:
        balanced_batches = class_balanced_batch_indices(
            labels_array,
            seed=params["seed"],
            steps=params["steps"],
            batch_size=params["batch_size"],
        )
        order: list[int] = []
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(params["seed"])
        order = torch.randperm(row_count, generator=generator).tolist()

    initial_logits = predict_arrays(model, features_array, device="cuda")
    initial_evaluation_bce = _mean_bce_from_logits(initial_logits, labels_array)
    model.train()

    initial_snapshot = _snapshot_parameters(model)
    previous_snapshot = initial_snapshot
    loss_records: list[dict[str, Any]] = []

    for step_index in range(1, params["steps"] + 1):
        if sampling == SAMPLING_CLASS_BALANCED:
            indices = balanced_batches[step_index - 1]
            batch_rows = len(indices)
        else:
            batch_rows = min(params["batch_size"], row_count)
            indices = [
                order[((step_index - 1) * params["batch_size"] + offset) % row_count]
                for offset in range(batch_rows)
            ]
        batch_index = torch.tensor(indices, dtype=torch.long, device=device_object)
        batch_inputs = inputs.index_select(0, batch_index)
        batch_targets = targets.index_select(0, batch_index)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        loss = loss_function(logits, batch_targets)
        loss_value = float(loss.detach().cpu().item())
        if not math.isfinite(loss_value):
            raise RuntimeError("non_finite_loss_at_step:" + str(step_index))
        loss.backward()
        gradient_norm = _gradient_norm(model)
        if not math.isfinite(gradient_norm):
            raise RuntimeError("non_finite_gradient_norm_at_step:" + str(step_index))
        if gradient_norm <= 0.0:
            raise RuntimeError("zero_gradient_at_step:" + str(step_index))
        optimizer.step()
        parameter_delta = _parameter_delta(model, previous_snapshot)
        previous_snapshot = _snapshot_parameters(model)
        loss_records.append({
            "step": step_index,
            "batch_rows": batch_rows,
            "loss": loss_value,
            "gradient_norm": gradient_norm,
            "parameter_delta": parameter_delta,
        })

    torch.cuda.synchronize()
    total_parameter_delta = _parameter_delta(model, initial_snapshot)
    if not (math.isfinite(total_parameter_delta) and total_parameter_delta > 0.0):
        raise RuntimeError("trainable_head_parameters_did_not_change")

    logits = predict_arrays(model, features_array, device="cuda")
    final_evaluation_bce = _mean_bce_from_logits(logits, labels_array)
    evaluation = summarize_full_manifest_fit(
        initial_bce=initial_evaluation_bce,
        final_bce=final_evaluation_bce,
        final_logits=logits,
        labels=labels_array,
    )
    runtime["synchronize_ok"] = True

    return {
        "model": model,
        "optimizer": optimizer,
        "config": model.config(),
        "runtime": runtime,
        "logits": [float(value) for value in logits],
        "records": {
            "loss_records": loss_records,
            "initial_loss": float(loss_records[0]["loss"]),
            "final_loss": float(loss_records[-1]["loss"]),
            "total_parameter_delta": total_parameter_delta,
            "class_counts": _class_counts(labels_array),
            "predicted_class_counts": _predicted_class_counts(logits),
            "row_count": row_count,
            "optimizer": "adam",
            "loss_function": "bce_with_logits",
            "device": "cuda",
            "frozen_feature_input": True,
            "sampling": sampling,
            "trainable_parameter_names": [
                name for name, _ in model.named_parameters()
            ],
            "evaluation": evaluation,
        },
    }


# ---------------------------------------------------------------------------
# Training-only fixed two-class fit diagnostic
# ---------------------------------------------------------------------------

def _validate_sample_rows(sample_rows: Any, row_count: int) -> list[int]:
    if not isinstance(sample_rows, (list, tuple)) or not sample_rows:
        raise ValueError("fit_diagnostic_sample_rows_invalid")
    rows: list[int] = []
    for value in sample_rows:
        if type(value) is not int or value < 0 or value >= row_count:
            raise ValueError("fit_diagnostic_sample_rows_invalid:" + repr(value))
        rows.append(value)
    if len(set(rows)) != len(rows):
        raise ValueError("fit_diagnostic_sample_rows_duplicate")
    return rows


def run_fit_diagnostic(
    features: Any,
    labels: Any,
    *,
    sample_rows: Any,
    pair_keys: Any,
    seed: Any,
    steps: Any,
    batch_size: Any,
    learning_rate: Any = DEFAULT_LR,
    hidden_dim: Any = DEFAULT_HIDDEN_DIM,
    device: Any = "cuda",
) -> dict[str, Any]:
    """Train a separate small two-class head on a fixed sample (in-sample probe).

    The same fixed sample is evaluated before and after training; the diagnostic
    passes only when the final BCE drops by at least 10% and every class has at
    least one correct prediction.  This is a training-engineering diagnostic, not
    generalisation evidence, and it never mutates source labels or manifests.
    """
    params = validate_training_params(
        seed=seed, steps=steps, batch_size=batch_size,
        learning_rate=learning_rate, hidden_dim=hidden_dim,
    )
    if not isinstance(device, str) or device != "cuda":
        raise ValueError("fit_diagnostic_device_must_be_cuda:" + repr(device))
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("cuda_unavailable_for_fit_diagnostic")

    features_array, labels_array = validate_training_input(features, labels)
    row_count = int(features_array.shape[0])
    rows = _validate_sample_rows(sample_rows, row_count)
    if not isinstance(pair_keys, (list, tuple)) or len(pair_keys) != len(rows):
        raise ValueError("fit_diagnostic_sample_pair_keys_invalid")
    sample_pair_keys: list[str] = []
    for key in pair_keys:
        if not isinstance(key, str) or not key:
            raise ValueError("fit_diagnostic_sample_pair_keys_invalid")
        sample_pair_keys.append(key)
    sample_labels = labels_array[np.asarray(rows, dtype=np.int64)]
    sample_class_counts = _class_counts(sample_labels)
    if sample_class_counts["inactive"] < 1 or sample_class_counts["active"] < 1:
        raise ValueError("fit_diagnostic_sample_requires_both_classes")

    configure_runtime()
    _seed_everything(params["seed"])
    device_object = torch.device("cuda")
    sample_features = np.ascontiguousarray(features_array[rows], dtype=np.float32)
    inputs = torch.from_numpy(sample_features).to(device=device_object)
    inputs.requires_grad_(False)
    targets = torch.from_numpy(sample_labels.astype(np.float32)).to(device_object)

    model = PairHead(hidden_dim=params["hidden_dim"]).to(device_object)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_function = nn.BCEWithLogitsLoss()

    model.eval()
    with torch.no_grad():
        initial_logits_tensor = model(inputs)
        initial_loss = float(loss_function(initial_logits_tensor, targets).item())
    initial_logits = initial_logits_tensor.detach().to("cpu").to(torch.float32).numpy()

    initial_snapshot = _snapshot_parameters(model)
    previous_snapshot = initial_snapshot
    loss_records: list[dict[str, Any]] = []
    batches = class_balanced_batch_indices(
        sample_labels,
        seed=params["seed"],
        steps=params["steps"],
        batch_size=params["batch_size"],
    )
    model.train()
    for step_index, indices in enumerate(batches, start=1):
        batch_index = torch.tensor(indices, dtype=torch.long, device=device_object)
        batch_inputs = inputs.index_select(0, batch_index)
        batch_targets = targets.index_select(0, batch_index)
        optimizer.zero_grad(set_to_none=True)
        logits_tensor = model(batch_inputs)
        loss = loss_function(logits_tensor, batch_targets)
        loss_value = float(loss.detach().cpu().item())
        if not math.isfinite(loss_value):
            raise RuntimeError("fit_diagnostic_non_finite_loss_at_step:" + str(step_index))
        loss.backward()
        gradient_norm = _gradient_norm(model)
        if not math.isfinite(gradient_norm):
            raise RuntimeError(
                "fit_diagnostic_non_finite_gradient_norm_at_step:" + str(step_index)
            )
        if gradient_norm <= 0.0:
            raise RuntimeError("fit_diagnostic_zero_gradient_at_step:" + str(step_index))
        optimizer.step()
        parameter_delta = _parameter_delta(model, previous_snapshot)
        previous_snapshot = _snapshot_parameters(model)
        loss_records.append({
            "step": step_index,
            "batch_rows": len(indices),
            "loss": loss_value,
            "gradient_norm": gradient_norm,
            "parameter_delta": parameter_delta,
        })

    torch.cuda.synchronize()
    total_parameter_delta = _parameter_delta(model, initial_snapshot)
    if not (math.isfinite(total_parameter_delta) and total_parameter_delta > 0.0):
        raise RuntimeError("fit_diagnostic_head_parameters_did_not_change")

    model.eval()
    with torch.no_grad():
        final_logits_tensor = model(inputs)
        final_loss = float(loss_function(final_logits_tensor, targets).item())
    final_logits = final_logits_tensor.detach().to("cpu").to(torch.float32).numpy()
    if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
        raise RuntimeError("fit_diagnostic_non_finite_evaluation_loss")

    criterion = evaluate_fit_criterion(
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_logits=initial_logits,
        final_logits=final_logits,
        labels=sample_labels,
    )
    return {
        "status": criterion["status"],
        "reasons": criterion["reasons"],
        "scope": FIT_DIAGNOSTIC_SCOPE,
        "criterion_version": FIT_DIAGNOSTIC_CRITERION_VERSION,
        "device": "cuda",
        "sampling": SAMPLING_CLASS_BALANCED,
        "seed": params["seed"],
        "steps": params["steps"],
        "steps_run": params["steps"],
        "batch_size": params["batch_size"],
        "learning_rate": params["learning_rate"],
        "hidden_dim": params["hidden_dim"],
        "sample_rows": rows,
        "sample_pair_keys": sample_pair_keys,
        "sample_labels": [int(value) for value in sample_labels.tolist()],
        "sample_class_counts": sample_class_counts,
        "criterion": criterion,
        "initial_bce": criterion["initial_bce"],
        "final_bce": criterion["final_bce"],
        "relative_drop": criterion["relative_drop"],
        "prediction_metrics": prediction_metrics(final_logits, sample_labels),
        "total_parameter_delta": total_parameter_delta,
        "trainable_parameter_names": [
            name for name, _ in model.named_parameters()
        ],
        "frozen_feature_input": True,
        "loss_records": loss_records,
    }


# ---------------------------------------------------------------------------
# Portable inference
# ---------------------------------------------------------------------------

def predict_arrays(
    model: "PairHead",
    features: Any,
    *,
    device: Any = "cpu",
    batch_size: Any = 16,
) -> np.ndarray:
    """Return one finite float32 logit per row, preserving input row order."""
    if not isinstance(model, PairHead):
        raise TypeError("model_must_be_pair_head")
    if not isinstance(device, str) or device not in ("cpu", "cuda"):
        raise ValueError("predict_device_not_supported:" + repr(device))
    if device == "cuda" and not bool(torch.cuda.is_available()):
        raise RuntimeError("cuda_unavailable_for_prediction")
    checked_batch = _require_positive_int(batch_size, "batch_size")
    features_array = validate_features(features, expected_dim=model.input_dim)

    target = torch.device(device)
    model = model.to(target)
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(features_array.shape[0]), checked_batch):
            batch = torch.from_numpy(features_array[start:start + checked_batch]).to(target)
            outputs.append(model(batch).detach().to("cpu"))
    logits = torch.cat(outputs, dim=0).to(torch.float32).numpy()
    if logits.shape != (features_array.shape[0],):
        raise RuntimeError("logit_shape_mismatch")
    if not bool(np.isfinite(logits).all()):
        raise RuntimeError("non_finite_logits")
    if device == "cuda":
        torch.cuda.synchronize()
    return logits


# ---------------------------------------------------------------------------
# Safe, atomic checkpoint
# ---------------------------------------------------------------------------

def _validate_input_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("input_metadata_not_mapping")
    arrays_sha256 = _require_sha256(metadata.get("arrays_sha256"), "arrays_sha256")
    input_manifest_sha256 = _require_sha256(
        metadata.get("input_manifest_sha256"), "input_manifest_sha256",
    )
    pair_keys = metadata.get("pair_keys")
    if not isinstance(pair_keys, list) or not pair_keys:
        raise ValueError("input_metadata_pair_keys_invalid")
    if any(not isinstance(key, str) or not key for key in pair_keys):
        raise ValueError("input_metadata_pair_keys_invalid")
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("input_metadata_pair_keys_not_unique")
    purpose = metadata.get("purpose")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("input_metadata_purpose_invalid")
    rows = metadata.get("rows")
    if type(rows) is not int or rows != len(pair_keys):
        raise ValueError("input_metadata_rows_invalid")
    return {
        "arrays_sha256": arrays_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "pair_keys": list(pair_keys),
        "purpose": purpose,
        "rows": rows,
    }


def save_checkpoint(
    path: os.PathLike[str] | str,
    *,
    model: "PairHead",
    optimizer: torch.optim.Optimizer,
    step: Any,
    seed: Any,
    config: Any,
    input_metadata: Any,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save model/optimizer state plus primitive and tensor metadata."""
    destination = Path(os.path.expanduser(os.fspath(path)))
    if not destination.is_absolute():
        raise ValueError("checkpoint_path_must_be_absolute")
    if not isinstance(model, PairHead):
        raise TypeError("model_must_be_pair_head")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer_required")
    checked_step = _require_int(step, "step", 1, MAX_STEPS)
    checked_seed = _require_int(seed, "seed", 0, 2 ** 63 - 1)
    checked_config = _validate_model_config(config)
    checked_metadata = _validate_input_metadata(input_metadata)
    if not destination.parent.is_dir():
        raise ValueError("checkpoint_parent_missing:" + str(destination.parent))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("checkpoint_already_exists:" + str(destination))

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": checked_step,
        "seed": checked_seed,
        "config": checked_config,
        "input_metadata": checked_metadata,
        "torch_rng_state": torch.get_rng_state(),
        "runtime": dict(runtime) if runtime is not None else {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    temporary = destination.parent / (
        "." + destination.name + ".tmp-" + uuid.uuid4().hex
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "format_version": CHECKPOINT_FORMAT_VERSION,
    }


def load_checkpoint(
    path: os.PathLike[str] | str,
    *,
    map_location: Any = "cpu",
    expected_sha256: Any = None,
    expect_input_sha256: Any = None,
) -> dict[str, Any]:
    """Load a checkpoint with the restricted weights-only unpickler."""
    if not isinstance(map_location, str) or map_location not in ("cpu", "cuda"):
        raise ValueError("unsupported_map_location:" + repr(map_location))
    if map_location == "cuda" and not bool(torch.cuda.is_available()):
        raise RuntimeError("cuda_unavailable_for_checkpoint_load")
    source = Path(os.path.expanduser(os.fspath(path)))
    if not source.is_file():
        raise ValueError("checkpoint_missing:" + str(source))
    actual_sha256 = sha256_file(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("checkpoint_sha256_mismatch")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any unsafe/corrupt payload is rejected
        raise ValueError("checkpoint_load_rejected:" + type(exc).__name__) from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint_not_mapping")
    missing = [key for key in _CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise ValueError("checkpoint_missing_keys:" + ",".join(missing))
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint_format_unsupported")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise ValueError("checkpoint_model_state_invalid")
    if not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in payload["model_state_dict"].items()
    ):
        raise ValueError("checkpoint_model_state_invalid")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise ValueError("checkpoint_optimizer_state_invalid")
    checked_config = _validate_model_config(payload["config"])
    checked_metadata = _validate_input_metadata(payload["input_metadata"])
    _require_int(payload["step"], "step", 1, MAX_STEPS)
    _require_int(payload["seed"], "seed", 0, 2 ** 63 - 1)
    if expect_input_sha256 is not None and checked_metadata["arrays_sha256"] != expect_input_sha256:
        raise ValueError("checkpoint_input_binding_mismatch")
    payload["config"] = checked_config
    payload["input_metadata"] = checked_metadata
    payload["sha256"] = actual_sha256
    return payload


# ---------------------------------------------------------------------------
# GPU-side CLI used by training.pair_pipeline
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("input_manifest_missing:" + str(path))
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("input_manifest_not_mapping")
    return value


def _write_jsonl_new(path: Path, rows: list[Mapping[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("jsonl_already_exists:" + str(path))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False))
            stream.write("\n")
    return sha256_file(path)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("json_already_exists:" + str(path))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return sha256_file(path)


def _validate_pair_metadata(pairs: list[Any]) -> None:
    """Reject missing identity fields, bad labels and row-order mismatches."""
    for position, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise ValueError("input_manifest_pair_metadata_invalid:" + str(position))
        for field in ("pair_key", "molecule_chembl_id", "accession", "label", "label_name"):
            if field not in pair or pair.get(field) in (None, ""):
                raise ValueError("input_manifest_pair_metadata_invalid:" + field)
        row_index = pair.get("row_index")
        if row_index is not None:
            if type(row_index) is not int or row_index != position:
                raise ValueError(
                    "input_manifest_pair_row_index_mismatch:" + str(position)
                )
        label = pair.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("input_manifest_pair_label_invalid:" + str(position))
        taxonomy_id = pair.get("taxonomy_id")
        if taxonomy_id is not None and type(taxonomy_id) is not int:
            raise ValueError("input_manifest_pair_metadata_invalid:taxonomy_id")


def _validate_fit_diagnostic_plan(
    plan: Any, pair_keys: list[str], labels_array: np.ndarray,
) -> dict[str, Any]:
    """Validate the declared fixed two-class diagnostic sample before training."""
    if not isinstance(plan, Mapping):
        raise ValueError("input_manifest_fit_diagnostic_invalid")
    if type(plan.get("plan_version")) is not int or plan.get("plan_version") != 1:
        raise ValueError("input_manifest_fit_diagnostic_invalid:plan_version")
    checked_seed = _require_int(plan.get("seed"), "seed", 0, 2 ** 63 - 1)
    checked_per_class = _require_int(
        plan.get("per_class"), "per_class", 1, MAX_STEPS,
    )
    try:
        rows = _validate_sample_rows(plan.get("sample_rows"), len(pair_keys))
    except ValueError as exc:
        raise ValueError(
            "input_manifest_fit_diagnostic_invalid:" + str(exc),
        ) from exc
    declared_keys = plan.get("pair_keys")
    declared_labels = plan.get("labels")
    if (
        not isinstance(declared_keys, list)
        or not isinstance(declared_labels, list)
        or len(declared_keys) != len(rows)
        or len(declared_labels) != len(rows)
    ):
        raise ValueError("input_manifest_fit_diagnostic_invalid:sample_metadata")
    expected_keys = [pair_keys[row] for row in rows]
    if [value if isinstance(value, str) else None for value in declared_keys] != expected_keys:
        raise ValueError("fit_diagnostic_sample_binding_mismatch")
    expected_labels = [int(labels_array[row]) for row in rows]
    parsed_labels: list[int] = []
    for value in declared_labels:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("input_manifest_fit_diagnostic_invalid:labels")
        parsed_labels.append(value)
    if parsed_labels != expected_labels:
        raise ValueError("fit_diagnostic_sample_binding_mismatch")
    class_counts = {
        "inactive": sum(1 for value in expected_labels if value == 0),
        "active": sum(1 for value in expected_labels if value == 1),
    }
    if class_counts["inactive"] < 1 or class_counts["active"] < 1:
        raise ValueError("fit_diagnostic_sample_requires_both_classes")
    declared_counts = plan.get("class_counts")
    if declared_counts is not None and dict(declared_counts) != class_counts:
        raise ValueError("fit_diagnostic_sample_binding_mismatch:class_counts")
    criterion = plan.get("criterion")
    if not isinstance(criterion, Mapping):
        raise ValueError("fit_diagnostic_criterion_mismatch")
    if criterion.get("version") != FIT_DIAGNOSTIC_CRITERION_VERSION:
        raise ValueError("fit_diagnostic_criterion_mismatch:version")
    if criterion.get("max_steps") != MAX_STEPS:
        raise ValueError("fit_diagnostic_criterion_mismatch:max_steps")
    if criterion.get("min_relative_bce_drop") != FIT_DIAGNOSTIC_MIN_RELATIVE_LOSS_DROP:
        raise ValueError("fit_diagnostic_criterion_mismatch:min_relative_bce_drop")
    if criterion.get("min_correct_per_class") != FIT_DIAGNOSTIC_MIN_CORRECT_PER_CLASS:
        raise ValueError("fit_diagnostic_criterion_mismatch:min_correct_per_class")
    return {
        "seed": checked_seed,
        "per_class": checked_per_class,
        "sample_rows": rows,
        "pair_keys": expected_keys,
        "labels": expected_labels,
        "class_counts": class_counts,
    }


def run_training(
    *,
    arrays_path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    seed: Any = 42,
    steps: Any = DEFAULT_STEPS,
    batch_size: Any = DEFAULT_BATCH_SIZE,
    learning_rate: Any = DEFAULT_LR,
    hidden_dim: Any = DEFAULT_HIDDEN_DIM,
) -> dict[str, Any]:
    """Full GPU-side run: validate inputs, train, checkpoint, reload, record."""
    out_root = Path(os.path.expanduser(os.fspath(out_dir)))
    if not out_root.is_absolute():
        raise ValueError("out_dir_must_be_absolute")
    if not out_root.is_dir():
        raise ValueError("out_dir_missing:" + str(out_root))
    arrays_file = Path(os.path.expanduser(os.fspath(arrays_path)))
    if not arrays_file.is_file():
        raise ValueError("arrays_missing:" + str(arrays_file))
    input_manifest_path = out_root / "input_manifest.json"
    input_manifest = _read_json(input_manifest_path)
    arrays_sha256 = sha256_file(arrays_file)
    arrays_record = input_manifest.get("arrays")
    if not isinstance(arrays_record, Mapping) or arrays_record.get("sha256") != arrays_sha256:
        raise ValueError("input_arrays_hash_mismatch")
    purpose = input_manifest.get("purpose")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("input_manifest_purpose_missing")
    pairs = input_manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("input_manifest_pairs_missing")
    pair_keys = [pair.get("pair_key") for pair in pairs if isinstance(pair, Mapping)]
    if len(pair_keys) != len(pairs) or any(not isinstance(key, str) or not key for key in pair_keys):
        raise ValueError("input_manifest_pair_keys_invalid")
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("input_manifest_pair_keys_not_unique")
    _validate_pair_metadata(pairs)
    input_manifest_sha256 = sha256_file(input_manifest_path)

    array_names: tuple[str, ...] | None = None
    features: Any = None
    labels: Any = None
    load_error: ValueError | None = None
    try:
        with np.load(arrays_file, allow_pickle=False) as arrays:
            array_names = tuple(sorted(arrays.files))
            if set(array_names) == {"features", "labels"}:
                features = arrays["features"]
                labels = arrays["labels"]
    except ValueError as exc:  # noqa: BLE001 - object pickles are rejected here
        load_error = exc
    except Exception as exc:  # noqa: BLE001 - any unreadable payload is rejected
        raise ValueError("arrays_require_pickle_rejected:" + type(exc).__name__) from exc
    if load_error is not None:
        message = str(load_error)
        if "allow_pickle" in message or "Object arrays" in message or "object" in message:
            raise ValueError(
                "arrays_require_pickle_rejected:" + type(load_error).__name__,
            ) from load_error
        raise ValueError("arrays_load_rejected:" + type(load_error).__name__) from load_error
    if array_names is None:
        raise ValueError("arrays_load_rejected:no_arrays")
    if set(array_names) != {"features", "labels"}:
        raise ValueError("arrays_keys_invalid:" + ",".join(array_names))

    params = validate_training_params(
        seed=seed, steps=steps, batch_size=batch_size,
        learning_rate=learning_rate, hidden_dim=hidden_dim,
    )
    features_array, labels_array = validate_training_input(
        features, labels, pair_keys=pair_keys,
    )
    if int(features_array.shape[0]) != len(pair_keys):
        raise ValueError("input_manifest_rows_mismatch")
    fit_request = None
    if input_manifest.get("fit_diagnostic") is not None:
        fit_request = _validate_fit_diagnostic_plan(
            input_manifest.get("fit_diagnostic"), pair_keys, labels_array,
        )
    sampling = (
        SAMPLING_CLASS_BALANCED if fit_request is not None else SAMPLING_LEGACY
    )

    result = train_arrays(
        features_array,
        labels_array,
        seed=params["seed"],
        steps=params["steps"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        device="cuda",
        hidden_dim=params["hidden_dim"],
        sampling=sampling,
    )
    model = result["model"]
    optimizer = result["optimizer"]
    config = result["config"]
    runtime = result["runtime"]
    input_metadata = {
        "arrays_sha256": arrays_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "pair_keys": pair_keys,
        "purpose": purpose,
        "rows": len(pair_keys),
    }
    checkpoint = save_checkpoint(
        out_root / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=params["steps"],
        seed=params["seed"],
        config=config,
        input_metadata=input_metadata,
        runtime=runtime,
    )

    # GPU-side portable reload: same weights, same device, same logits.
    reloaded = load_checkpoint(
        out_root / "checkpoint.pt",
        map_location="cuda",
        expected_sha256=checkpoint["sha256"],
        expect_input_sha256=arrays_sha256,
    )
    reloaded_model = build_model(reloaded["config"]).to("cuda")
    reloaded_model.load_state_dict(reloaded["model_state_dict"])
    reloaded_logits = predict_arrays(reloaded_model, features_array, device="cuda")
    trained_logits = np.asarray(result["logits"], dtype=np.float32)
    reload_max_abs_diff = float(np.max(np.abs(reloaded_logits - trained_logits)))
    reload_passed = bool(np.allclose(
        reloaded_logits, trained_logits, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL,
    ))
    if not reload_passed:
        raise RuntimeError("gpu_checkpoint_reload_logit_mismatch:" + repr(reload_max_abs_diff))

    loss_sha256 = _write_jsonl_new(out_root / "loss.jsonl", result["records"]["loss_records"])
    training_evaluation = result["records"]["evaluation"]
    fit_record: dict[str, Any] | None = None
    fit_diagnostic_sha256: str | None = None
    if fit_request is not None:
        diagnostic = run_fit_diagnostic(
            features_array,
            labels_array,
            sample_rows=fit_request["sample_rows"],
            pair_keys=fit_request["pair_keys"],
            seed=fit_request["seed"],
            steps=params["steps"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            hidden_dim=params["hidden_dim"],
            device="cuda",
        )
        reasons = list(diagnostic["criterion"]["reasons"])
        if training_evaluation["majority_collapse"]:
            reasons.append("full_manifest_predictions_only_majority_class")
        fit_record = {
            "schema_version": 1,
            "status": "passed" if not reasons else "failed",
            "reasons": reasons,
            "criterion_version": FIT_DIAGNOSTIC_CRITERION_VERSION,
            "scope": FIT_DIAGNOSTIC_SCOPE,
            "plan": fit_request,
            "small_sample": diagnostic,
            "full_manifest": training_evaluation,
            "data_use": {
                "evaluation": "none (no independent holdout)",
                "generalization_claims": "none",
                "fit_scope": (
                    "engineering development in-sample fit diagnostic only"
                ),
            },
        }
        fit_diagnostic_sha256 = _write_json_new(
            out_root / "fit_diagnostic.json", fit_record,
        )
    gpu_result = {
        "schema_version": 1,
        "status": "trained",
        "device": "cuda",
        "runtime": runtime,
        "config": config,
        "training": {
            "seed": params["seed"],
            "steps": params["steps"],
            "batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
            "hidden_dim": params["hidden_dim"],
            "optimizer": "adam",
            "loss_function": "bce_with_logits",
            "row_count": result["records"]["row_count"],
            "frozen_feature_input": True,
            "class_counts": result["records"]["class_counts"],
            "predicted_class_counts": result["records"]["predicted_class_counts"],
            "sampling": sampling,
            "trainable_parameter_names": result["records"][
                "trainable_parameter_names"
            ],
        },
        "loss_records": result["records"]["loss_records"],
        "initial_loss": result["records"]["initial_loss"],
        "final_loss": result["records"]["final_loss"],
        "total_parameter_delta": result["records"]["total_parameter_delta"],
        "logits": result["logits"],
        "evaluation": training_evaluation,
        "fit_diagnostic": fit_record,
        "fit_diagnostic_sha256": fit_diagnostic_sha256,
        "gpu_reload": {
            "max_abs_diff": reload_max_abs_diff,
            "atol": ALLCLOSE_ATOL,
            "rtol": ALLCLOSE_RTOL,
            "passed": reload_passed,
        },
        "arrays_sha256": arrays_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "checkpoint": checkpoint,
        "loss_jsonl_sha256": loss_sha256,
    }
    _write_json_new(out_root / "gpu_result.json", gpu_result)
    return {
        "status": "trained",
        "out": str(out_root),
        "device": "cuda",
        "device_name": runtime.get("device_name"),
        "steps": params["steps"],
        "final_loss": gpu_result["final_loss"],
        "total_parameter_delta": gpu_result["total_parameter_delta"],
        "gpu_reload_max_abs_diff": reload_max_abs_diff,
        "checkpoint_sha256": checkpoint["sha256"],
        "sampling": sampling,
        "fit_diagnostic_status": fit_record["status"] if fit_record else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="training.gpu_model",
        description="CUDA-only pair-head training/checkpoint runner (Phase 03 tracer).",
    )
    parser.add_argument("--arrays", required=True, help="training_arrays.npz from pair_pipeline")
    parser.add_argument("--out", required=True, help="existing new run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args(argv)
    try:
        summary = run_training(
            arrays_path=args.arrays,
            out_dir=args.out,
            seed=args.seed,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
        )
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

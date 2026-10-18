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
import random
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

CHECKPOINT_FORMAT_VERSION = 2
MODEL_ARCHITECTURE = "mlp_relu_logit_v1"
OPTIMIZER_NAME = "adam"
ALLCLOSE_ATOL = 1e-6
ALLCLOSE_RTOL = 1e-6

# The v2 checkpoint schema is the reproducible-resume contract: weights-only
# state dicts plus primitive/tensor recovery fields.  Nothing else is accepted.
_CHECKPOINT_KEYS = (
    "format_version",
    "model_state_dict",
    "optimizer_state_dict",
    "step",
    "global_step",
    "seed",
    "config",
    "optimizer_config",
    "training_config",
    "input_metadata",
    "torch_rng_state",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_generator_state",
    "runtime",
    "created_at",
)
_OPTIONAL_INPUT_METADATA_KEYS = (
    "development_manifest_sha256",
    "feature_manifest_sha256",
    "frozen_manifest_sha256",
    "entities_sha256",
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
    # Python, NumPy and CPU-Torch global RNGs are all seeded so the recovery
    # state saved in a checkpoint is reproducible across processes.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _execute_optimizer_steps(
    model: "PairHead",
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    row_count: int,
    labels_array: np.ndarray,
    params: Mapping[str, Any],
    sampling: str,
    start_step: int,
    total_steps: int,
    generator: torch.Generator | None = None,
) -> tuple[list[dict[str, Any]], torch.Generator]:
    """Run optimizer steps (start_step, total_steps] with reproducible CPU sampling.

    Legacy sampling draws one CPU-generator permutation per step; the generator
    state is part of the checkpoint so a resumed run draws exactly the batches
    the uninterrupted run would have drawn.  Class-balanced diagnostic sampling
    is a pure deterministic function of (seed, labels, steps, batch) and is
    therefore regenerated identically on resume.
    """
    if type(start_step) is not int or type(total_steps) is not int:
        raise ValueError("training_step_range_invalid")
    if start_step < 0 or total_steps <= start_step or total_steps > MAX_STEPS:
        raise ValueError("training_step_range_invalid")
    if sampling not in VALID_SAMPLING_MODES:
        raise ValueError("invalid_sampling_mode:" + repr(sampling))
    device_object = torch.device("cuda")
    loss_function = nn.BCEWithLogitsLoss()
    if sampling == SAMPLING_CLASS_BALANCED:
        balanced_batches = class_balanced_batch_indices(
            labels_array,
            seed=params["seed"],
            steps=total_steps,
            batch_size=params["batch_size"],
        )
    else:
        balanced_batches = []
        if generator is None:
            generator = _make_cpu_generator(params["seed"])
    previous_snapshot = _snapshot_parameters(model)
    loss_records: list[dict[str, Any]] = []
    for step_index in range(start_step + 1, total_steps + 1):
        if sampling == SAMPLING_CLASS_BALANCED:
            indices = balanced_batches[step_index - 1]
            batch_rows = len(indices)
        else:
            order = torch.randperm(row_count, generator=generator)
            batch_rows = min(int(params["batch_size"]), row_count)
            indices = order[:batch_rows].tolist()
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
    return loss_records, generator


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
    runtime["training_device"] = "cuda"

    features_array, labels_array = validate_arrays(features, labels)
    _seed_everything(params["seed"])

    device_object = torch.device("cuda")
    inputs = torch.from_numpy(features_array).to(device=device_object)
    inputs.requires_grad_(False)
    targets = torch.from_numpy(labels_array.astype(np.float32)).to(device_object)

    model = PairHead(hidden_dim=params["hidden_dim"]).to(device_object)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    row_count = int(features_array.shape[0])
    initial_logits = predict_arrays(model, features_array, device="cuda")
    initial_evaluation_bce = _mean_bce_from_logits(initial_logits, labels_array)
    model.train()

    initial_snapshot = _snapshot_parameters(model)
    loss_records, sampling_generator = _execute_optimizer_steps(
        model,
        optimizer,
        inputs,
        targets,
        row_count=row_count,
        labels_array=labels_array,
        params=params,
        sampling=sampling,
        start_step=0,
        total_steps=params["steps"],
    )

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
        "sampling_generator": sampling_generator,
        "training_params": dict(params),
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
# Reproducible recovery state (weights-only compatible primitives/tensors)
# ---------------------------------------------------------------------------

def _make_cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _serialize_numpy_rng_state() -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    if algorithm != "MT19937":
        raise RuntimeError("numpy_rng_algorithm_unexpected:" + repr(algorithm))
    return {
        "algorithm": "MT19937",
        "keys": [int(value) for value in keys],
        "pos": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _validate_numpy_rng_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint_numpy_rng_state_invalid")
    if value.get("algorithm") != "MT19937":
        raise ValueError("checkpoint_numpy_rng_state_invalid:algorithm")
    keys = value.get("keys")
    if (
        not isinstance(keys, list)
        or len(keys) != 624
        or any(type(item) is not int or item < 0 or item > 2 ** 32 - 1 for item in keys)
    ):
        raise ValueError("checkpoint_numpy_rng_state_invalid:keys")
    position = value.get("pos")
    if type(position) is not int or not 0 <= position <= 624:
        raise ValueError("checkpoint_numpy_rng_state_invalid:pos")
    has_gauss = value.get("has_gauss")
    if type(has_gauss) is not int or has_gauss not in (0, 1):
        raise ValueError("checkpoint_numpy_rng_state_invalid:has_gauss")
    cached = value.get("cached_gaussian")
    if (
        isinstance(cached, bool)
        or not isinstance(cached, (int, float))
        or not math.isfinite(float(cached))
    ):
        raise ValueError("checkpoint_numpy_rng_state_invalid:cached_gaussian")
    return {
        "algorithm": "MT19937",
        "keys": [int(item) for item in keys],
        "pos": position,
        "has_gauss": has_gauss,
        "cached_gaussian": float(cached),
    }


def _validate_python_rng_state(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("checkpoint_python_rng_state_invalid")
    version, internal, gauss = value
    if type(version) is not int or version != 3:
        raise ValueError("checkpoint_python_rng_state_invalid:version")
    if (
        not isinstance(internal, tuple)
        or not internal
        or any(type(item) is not int or item < 0 for item in internal)
    ):
        raise ValueError("checkpoint_python_rng_state_invalid:internal")
    if gauss is not None and (
        isinstance(gauss, bool)
        or not isinstance(gauss, (int, float))
        or not math.isfinite(float(gauss))
    ):
        raise ValueError("checkpoint_python_rng_state_invalid:gauss")
    return value


def _validate_rng_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise ValueError("checkpoint_" + name + "_invalid")
    tensor = value.detach().to("cpu")
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError("checkpoint_" + name + "_invalid:shape")
    return tensor


def _capture_recovery_rng_states(generator: "torch.Generator | None" = None) -> dict[str, Any]:
    """Capture Python/NumPy/CPU-Torch RNG state as weights-only-safe values."""
    if generator is None:
        generator = torch.Generator(device="cpu")
    return {
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": _serialize_numpy_rng_state(),
        "torch_cpu_generator_state": generator.get_state(),
    }


def _validate_recovery_rng_states(states: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(states, Mapping):
        raise ValueError("checkpoint_rng_states_invalid")
    if set(states) != {
        "torch_rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_generator_state",
    }:
        raise ValueError("checkpoint_rng_states_invalid:keys")
    return {
        "torch_rng_state": _validate_rng_tensor(
            states["torch_rng_state"], "torch_rng_state",
        ),
        "python_rng_state": _validate_python_rng_state(states["python_rng_state"]),
        "numpy_rng_state": _validate_numpy_rng_state(states["numpy_rng_state"]),
        "torch_cpu_generator_state": _validate_rng_tensor(
            states["torch_cpu_generator_state"], "torch_cpu_generator_state",
        ),
    }


def apply_recovery_rng_states(
    payload: Mapping[str, Any], *, generator: "torch.Generator | None" = None,
) -> dict[str, Any]:
    """Restore Python/NumPy/CPU-Torch RNG state from a loaded checkpoint."""
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint_rng_payload_not_mapping")
    states = _validate_recovery_rng_states({
        "torch_rng_state": payload.get("torch_rng_state"),
        "python_rng_state": payload.get("python_rng_state"),
        "numpy_rng_state": payload.get("numpy_rng_state"),
        "torch_cpu_generator_state": payload.get("torch_cpu_generator_state"),
    })
    torch.set_rng_state(states["torch_rng_state"])
    random.setstate(states["python_rng_state"])
    numpy_state = states["numpy_rng_state"]
    np.random.set_state((
        numpy_state["algorithm"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["pos"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    if generator is not None:
        generator.set_state(states["torch_cpu_generator_state"])
    return states


# ---------------------------------------------------------------------------
# Optimizer / training configuration contracts
# ---------------------------------------------------------------------------

def _build_optimizer_config(optimizer: Any) -> dict[str, Any]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer_required")
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError("optimizer_must_be_adam")
    groups = optimizer.param_groups
    if len(groups) != 1:
        raise ValueError("optimizer_param_group_count_unsupported")
    group = groups[0]
    return {
        "name": OPTIMIZER_NAME,
        "learning_rate": float(group["lr"]),
        "betas": [float(value) for value in group["betas"]],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
    }


def _validate_optimizer_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint_optimizer_config_invalid")
    if config.get("name") != OPTIMIZER_NAME:
        raise ValueError("checkpoint_optimizer_config_invalid:name")
    learning_rate = config.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise ValueError("checkpoint_optimizer_config_invalid:learning_rate")
    betas = config.get("betas")
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) < 1.0
            for value in betas
        )
    ):
        raise ValueError("checkpoint_optimizer_config_invalid:betas")
    eps = config.get("eps")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        raise ValueError("checkpoint_optimizer_config_invalid:eps")
    weight_decay = config.get("weight_decay")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) < 0.0
    ):
        raise ValueError("checkpoint_optimizer_config_invalid:weight_decay")
    return {
        "name": OPTIMIZER_NAME,
        "learning_rate": float(learning_rate),
        "betas": [float(value) for value in betas],
        "eps": float(eps),
        "weight_decay": float(weight_decay),
    }


def _validate_training_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint_training_config_invalid")
    batch_size = _require_int(
        config.get("batch_size"), "batch_size", MIN_BATCH_SIZE, MAX_BATCH_SIZE,
    )
    sampling = config.get("sampling")
    if sampling not in VALID_SAMPLING_MODES:
        raise ValueError("checkpoint_training_config_invalid:sampling")
    return {"batch_size": batch_size, "sampling": sampling}


def _expected_model_state_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    model = build_model(config)
    return {
        name: tuple(tensor.shape) for name, tensor in model.state_dict().items()
    }


def _validate_model_state_dict(
    model_state: Any, config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("checkpoint_model_state_invalid")
    if not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in model_state.items()
    ):
        raise ValueError("checkpoint_model_state_invalid")
    expected = _expected_model_state_shapes(config)
    if set(model_state) != set(expected):
        raise ValueError("checkpoint_model_state_keys_mismatch")
    normalized: dict[str, torch.Tensor] = {}
    for name, shape in expected.items():
        tensor = model_state[name]
        if tuple(tensor.shape) != shape:
            raise ValueError("checkpoint_model_state_shape_mismatch:" + name)
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise ValueError("checkpoint_model_state_non_finite:" + name)
        normalized[name] = tensor
    return normalized


def _validate_optimizer_state_dict(
    optimizer_state: Any, config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint_optimizer_state_invalid")
    if set(optimizer_state) != {"state", "param_groups"}:
        raise ValueError("checkpoint_optimizer_state_invalid:keys")
    param_groups = optimizer_state["param_groups"]
    if not isinstance(param_groups, list) or len(param_groups) != 1:
        raise ValueError("checkpoint_optimizer_state_invalid:param_groups")
    group = param_groups[0]
    if not isinstance(group, Mapping):
        raise ValueError("checkpoint_optimizer_state_invalid:param_group")
    model = build_model(config)
    expected_shapes = [tuple(parameter.shape) for parameter in model.parameters()]
    params = group.get("params")
    if not isinstance(params, list) or params != list(range(len(expected_shapes))):
        raise ValueError("checkpoint_optimizer_state_invalid:params")
    learning_rate = group.get("lr")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise ValueError("checkpoint_optimizer_state_invalid:lr")
    entries = optimizer_state["state"]
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("checkpoint_optimizer_state_invalid:state")
    required_entry_keys = {"step", "exp_avg", "exp_avg_sq"}
    normalized_entries: dict[int, dict[str, torch.Tensor]] = {}
    for param_id, entry in entries.items():
        if type(param_id) is not int or not 0 <= param_id < len(expected_shapes):
            raise ValueError("checkpoint_optimizer_state_invalid:param_id")
        if not isinstance(entry, Mapping) or set(entry) != required_entry_keys:
            raise ValueError("checkpoint_optimizer_state_invalid:entry")
        step_tensor = entry["step"]
        if not isinstance(step_tensor, torch.Tensor) or step_tensor.numel() != 1:
            raise ValueError("checkpoint_optimizer_state_invalid:step")
        if not bool(torch.isfinite(step_tensor.detach()).all().item()):
            raise ValueError("checkpoint_optimizer_state_invalid:step:non_finite")
        normalized_entry: dict[str, torch.Tensor] = {"step": step_tensor}
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = entry[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("checkpoint_optimizer_state_invalid:" + name)
            if tuple(tensor.shape) != expected_shapes[param_id]:
                raise ValueError("checkpoint_optimizer_state_invalid:" + name + ":shape")
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                raise ValueError("checkpoint_optimizer_state_invalid:" + name + ":non_finite")
            normalized_entry[name] = tensor
        normalized_entries[param_id] = normalized_entry
    return {
        "state": normalized_entries,
        "param_groups": [dict(group)],
    }


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
    validated = {
        "arrays_sha256": arrays_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "pair_keys": list(pair_keys),
        "purpose": purpose,
        "rows": rows,
    }
    for name in _OPTIONAL_INPUT_METADATA_KEYS:
        value = metadata.get(name)
        if value is None:
            continue
        validated[name] = _require_sha256(value, name)
    return validated


def _validate_cuda_training_runtime(runtime: Any) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise ValueError("checkpoint_runtime_not_mapping")
    if runtime.get("training_device") != "cuda":
        raise ValueError("checkpoint_runtime_training_device_must_be_cuda")
    if runtime.get("cuda_available") is not True:
        raise ValueError("checkpoint_runtime_cuda_not_available")
    device_name = runtime.get("device_name")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("checkpoint_runtime_device_name_missing")
    capability = runtime.get("capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int for value in capability)
    ):
        raise ValueError("checkpoint_runtime_capability_invalid")
    torch_version = runtime.get("torch")
    if not isinstance(torch_version, str) or not torch_version:
        raise ValueError("checkpoint_runtime_torch_version_missing")
    cuda_version = runtime.get("torch_cuda")
    if cuda_version is not None and not isinstance(cuda_version, str):
        raise ValueError("checkpoint_runtime_cuda_version_invalid")
    return dict(runtime)


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
    training_config: Mapping[str, Any] | None = None,
    optimizer_config: Mapping[str, Any] | None = None,
    recovery_rng_states: Mapping[str, Any] | None = None,
    require_cuda_runtime: bool = False,
) -> dict[str, Any]:
    """Atomically save the reproducible-resume checkpoint contract.

    The payload is intentionally restricted to state dicts plus primitive and
    tensor values so the restricted weights-only unpickler can always read it.
    """
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
    checked_optimizer_config = (
        _build_optimizer_config(optimizer)
        if optimizer_config is None
        else _validate_optimizer_config(optimizer_config)
    )
    optimizer_learning_rate = float(optimizer.param_groups[0]["lr"])
    if checked_optimizer_config["learning_rate"] != optimizer_learning_rate:
        raise ValueError("optimizer_config_learning_rate_mismatch")
    checked_training_config = (
        {"batch_size": DEFAULT_BATCH_SIZE, "sampling": SAMPLING_LEGACY}
        if training_config is None
        else _validate_training_config(training_config)
    )
    if recovery_rng_states is None:
        recovery_rng_states = _capture_recovery_rng_states()
    checked_rng_states = _validate_recovery_rng_states(recovery_rng_states)
    checked_runtime = dict(runtime) if runtime is not None else {}
    if require_cuda_runtime:
        _validate_cuda_training_runtime(checked_runtime)
    if not destination.parent.is_dir():
        raise ValueError("checkpoint_parent_missing:" + str(destination.parent))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("checkpoint_already_exists:" + str(destination))

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": checked_step,
        "global_step": checked_step,
        "seed": checked_seed,
        "config": checked_config,
        "optimizer_config": checked_optimizer_config,
        "training_config": checked_training_config,
        "input_metadata": checked_metadata,
        "torch_rng_state": checked_rng_states["torch_rng_state"],
        "python_rng_state": checked_rng_states["python_rng_state"],
        "numpy_rng_state": checked_rng_states["numpy_rng_state"],
        "torch_cpu_generator_state": checked_rng_states["torch_cpu_generator_state"],
        "runtime": checked_runtime,
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
        "step": checked_step,
        "global_step": checked_step,
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
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected_checkpoint_sha256")
    if expect_input_sha256 is not None:
        _require_sha256(expect_input_sha256, "expect_input_sha256")
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
    unknown = sorted(set(payload) - set(_CHECKPOINT_KEYS))
    if unknown:
        raise ValueError("checkpoint_unknown_keys:" + ",".join(unknown))
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint_format_unsupported")
    checked_config = _validate_model_config(payload["config"])
    checked_metadata = _validate_input_metadata(payload["input_metadata"])
    checked_step = _require_int(payload["step"], "step", 1, MAX_STEPS)
    checked_global_step = _require_int(payload["global_step"], "global_step", 1, MAX_STEPS)
    if checked_global_step != checked_step:
        raise ValueError("checkpoint_global_step_mismatch")
    _require_int(payload["seed"], "seed", 0, 2 ** 63 - 1)
    checked_model_state = _validate_model_state_dict(
        payload["model_state_dict"], checked_config,
    )
    checked_optimizer_state = _validate_optimizer_state_dict(
        payload["optimizer_state_dict"], checked_config,
    )
    checked_optimizer_config = _validate_optimizer_config(payload["optimizer_config"])
    group_learning_rate = checked_optimizer_state["param_groups"][0].get("lr")
    if (
        isinstance(group_learning_rate, bool)
        or not isinstance(group_learning_rate, (int, float))
        or float(group_learning_rate) != checked_optimizer_config["learning_rate"]
    ):
        raise ValueError("checkpoint_optimizer_config_mismatch")
    checked_training_config = _validate_training_config(payload["training_config"])
    checked_rng_states = _validate_recovery_rng_states({
        "torch_rng_state": payload["torch_rng_state"],
        "python_rng_state": payload["python_rng_state"],
        "numpy_rng_state": payload["numpy_rng_state"],
        "torch_cpu_generator_state": payload["torch_cpu_generator_state"],
    })
    if not isinstance(payload["runtime"], Mapping):
        raise ValueError("checkpoint_runtime_not_mapping")
    if not isinstance(payload["created_at"], str) or not payload["created_at"]:
        raise ValueError("checkpoint_created_at_invalid")
    if expect_input_sha256 is not None and (
        checked_metadata["arrays_sha256"] != expect_input_sha256
    ):
        raise ValueError("checkpoint_input_binding_mismatch")
    payload["model_state_dict"] = checked_model_state
    payload["optimizer_state_dict"] = checked_optimizer_state
    payload["optimizer_config"] = checked_optimizer_config
    payload["training_config"] = checked_training_config
    payload["config"] = checked_config
    payload["input_metadata"] = checked_metadata
    payload["step"] = checked_step
    payload["global_step"] = checked_global_step
    payload["torch_rng_state"] = checked_rng_states["torch_rng_state"]
    payload["python_rng_state"] = checked_rng_states["python_rng_state"]
    payload["numpy_rng_state"] = checked_rng_states["numpy_rng_state"]
    payload["torch_cpu_generator_state"] = checked_rng_states["torch_cpu_generator_state"]
    payload["sha256"] = actual_sha256
    return payload


# ---------------------------------------------------------------------------
# Resume and standalone checkpoint prediction
# ---------------------------------------------------------------------------

def _load_run_arrays(
    arrays_file: Path, *, expected_sha256: str, expected_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    actual_sha256 = sha256_file(arrays_file)
    if actual_sha256 != expected_sha256:
        raise ValueError("resume_arrays_sha256_mismatch")
    try:
        with np.load(arrays_file, allow_pickle=False) as arrays:
            if set(arrays.files) != {"features", "labels"}:
                raise ValueError("arrays_keys_invalid:" + ",".join(sorted(arrays.files)))
            features = arrays["features"]
            labels = arrays["labels"]
    except ValueError as exc:
        raise ValueError("arrays_require_pickle_rejected:" + type(exc).__name__) from exc
    features_array, labels_array = validate_arrays(features, labels)
    if int(features_array.shape[0]) != expected_rows:
        raise ValueError("resume_arrays_rows_mismatch")
    return features_array, labels_array


def _validate_resume_source_bindings(
    payload: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    input_manifest_sha256: str,
) -> None:
    metadata = payload["input_metadata"]
    if input_manifest_sha256 != metadata["input_manifest_sha256"]:
        raise ValueError("resume_input_manifest_sha256_mismatch")
    if input_manifest.get("purpose") != metadata["purpose"]:
        raise ValueError("resume_input_manifest_purpose_mismatch")
    pairs = input_manifest.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("resume_input_manifest_pairs_missing")
    _validate_pair_metadata(pairs)
    pair_keys = [pair["pair_key"] for pair in pairs]
    if pair_keys != metadata["pair_keys"]:
        raise ValueError("resume_input_manifest_pair_keys_mismatch")
    development_record = input_manifest.get("development_manifest")
    if "development_manifest_sha256" in metadata:
        if not isinstance(development_record, Mapping) or (
            development_record.get("sha256") != metadata["development_manifest_sha256"]
        ):
            raise ValueError("resume_development_manifest_binding_mismatch")
    source_record = input_manifest.get("source")
    for binding in (
        "frozen_manifest_sha256",
        "entities_sha256",
        "feature_manifest_sha256",
    ):
        if binding not in metadata:
            continue
        if not isinstance(source_record, Mapping):
            raise ValueError("resume_source_binding_missing:" + binding)
        if source_record.get(binding) != metadata[binding]:
            raise ValueError("resume_source_binding_mismatch:" + binding)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: Any,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _assert_optimizer_matches_config(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any],
) -> None:
    checked = _validate_optimizer_config(config)
    groups = optimizer.param_groups
    if len(groups) != 1:
        raise RuntimeError("resume_optimizer_param_group_count_mismatch")
    group = groups[0]
    if float(group["lr"]) != checked["learning_rate"]:
        raise RuntimeError("checkpoint_optimizer_config_mismatch:learning_rate")
    if [float(value) for value in group["betas"]] != checked["betas"]:
        raise RuntimeError("checkpoint_optimizer_config_mismatch:betas")
    if float(group["eps"]) != checked["eps"]:
        raise RuntimeError("checkpoint_optimizer_config_mismatch:eps")
    if float(group["weight_decay"]) != checked["weight_decay"]:
        raise RuntimeError("checkpoint_optimizer_config_mismatch:weight_decay")


def resume_training(
    *,
    resume_run: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    additional_steps: Any,
    expected_checkpoint_sha256: Any = None,
) -> dict[str, Any]:
    """Continue a CUDA optimizer trajectory from a weights-only checkpoint.

    The prior run directory supplies its checkpoint, numeric arrays and input
    manifest.  The new directory receives a v2 checkpoint at the final step plus
    resume_result.json proving the continuation.  There is deliberately no
    CPU fallback: any missing CUDA capability raises.
    """
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("cuda_unavailable_for_training")
    source_root = Path(os.path.expanduser(os.fspath(resume_run)))
    if not source_root.is_absolute():
        raise ValueError("resume_run_must_be_absolute")
    if not source_root.is_dir():
        raise ValueError("resume_run_missing:" + str(source_root))
    out_root = Path(os.path.expanduser(os.fspath(out_dir)))
    if not out_root.is_absolute():
        raise ValueError("out_dir_must_be_absolute")
    if not out_root.is_dir():
        raise ValueError("out_dir_missing:" + str(out_root))
    destination_checkpoint = out_root / "checkpoint.pt"
    if destination_checkpoint.exists() or destination_checkpoint.is_symlink():
        raise FileExistsError("checkpoint_already_exists:" + str(destination_checkpoint))
    source_checkpoint = source_root / "checkpoint.pt"
    arrays_file = source_root / "training_arrays.npz"
    input_manifest_path = source_root / "input_manifest.json"
    for path, code in (
        (source_checkpoint, "resume_checkpoint_missing"),
        (arrays_file, "resume_arrays_missing"),
        (input_manifest_path, "resume_input_manifest_missing"),
    ):
        if not path.is_file():
            raise ValueError(code + ":" + str(path))
    additional = _require_int(additional_steps, "additional_steps", 1, MAX_STEPS)
    arrays_sha256 = sha256_file(arrays_file)
    payload = load_checkpoint(
        source_checkpoint,
        map_location="cpu",
        expected_sha256=expected_checkpoint_sha256,
        expect_input_sha256=arrays_sha256,
    )
    start_step = payload["global_step"]
    total_steps = start_step + additional
    if total_steps > MAX_STEPS:
        raise ValueError("resume_total_steps_exceed_max")
    runtime = payload["runtime"]
    _validate_cuda_training_runtime(runtime)
    diagnostics = configure_runtime()
    current_runtime = cuda_runtime_report()
    current_runtime["tf32"] = diagnostics
    current_runtime["training_device"] = "cuda"
    for field in ("device_name", "capability", "torch", "torch_cuda"):
        if runtime.get(field) != current_runtime.get(field):
            raise RuntimeError("checkpoint_runtime_identity_mismatch:" + field)

    input_manifest = _read_json(input_manifest_path)
    _validate_resume_source_bindings(
        payload,
        input_manifest,
        sha256_file(input_manifest_path),
    )
    features_array, labels_array = _load_run_arrays(
        arrays_file,
        expected_sha256=arrays_sha256,
        expected_rows=payload["input_metadata"]["rows"],
    )
    if int(features_array.shape[0]) != len(payload["input_metadata"]["pair_keys"]):
        raise ValueError("resume_pair_key_count_mismatch")

    device_object = torch.device("cuda")
    model = build_model(payload["config"]).to(device_object)
    model.load_state_dict(payload["model_state_dict"])
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=payload["optimizer_config"]["learning_rate"],
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device_object)
    _assert_optimizer_matches_config(optimizer, payload["optimizer_config"])

    inputs = torch.from_numpy(features_array).to(device=device_object)
    inputs.requires_grad_(False)
    targets = torch.from_numpy(labels_array.astype(np.float32)).to(device_object)
    sampling_generator = _make_cpu_generator(payload["seed"])
    apply_recovery_rng_states(payload, generator=sampling_generator)

    loss_records, sampling_generator = _execute_optimizer_steps(
        model,
        optimizer,
        inputs,
        targets,
        row_count=int(features_array.shape[0]),
        labels_array=labels_array,
        params={
            "seed": payload["seed"],
            "batch_size": payload["training_config"]["batch_size"],
        },
        sampling=payload["training_config"]["sampling"],
        start_step=start_step,
        total_steps=total_steps,
        generator=sampling_generator,
    )
    torch.cuda.synchronize()
    logits = predict_arrays(model, features_array, device="cuda")
    if not bool(np.isfinite(logits).all()):
        raise RuntimeError("resume_non_finite_logits")

    resumed_runtime = dict(current_runtime)
    resumed_runtime["resumed_from"] = {
        "run": str(source_root),
        "checkpoint_path": str(source_checkpoint),
        "checkpoint_sha256": payload["sha256"],
        "start_step": start_step,
        "additional_steps": additional,
    }
    checkpoint = save_checkpoint(
        destination_checkpoint,
        model=model,
        optimizer=optimizer,
        step=total_steps,
        seed=payload["seed"],
        config=payload["config"],
        input_metadata=payload["input_metadata"],
        runtime=resumed_runtime,
        training_config=payload["training_config"],
        optimizer_config=payload["optimizer_config"],
        recovery_rng_states=_capture_recovery_rng_states(sampling_generator),
        require_cuda_runtime=True,
    )

    reloaded = load_checkpoint(
        destination_checkpoint,
        map_location="cuda",
        expected_sha256=checkpoint["sha256"],
        expect_input_sha256=arrays_sha256,
    )
    reloaded_model = build_model(reloaded["config"]).to("cuda")
    reloaded_model.load_state_dict(reloaded["model_state_dict"])
    reloaded_logits = predict_arrays(reloaded_model, features_array, device="cuda")
    reload_max_abs_diff = float(np.max(np.abs(
        np.asarray(reloaded_logits, dtype=np.float64)
        - np.asarray(logits, dtype=np.float64)
    )))
    reload_passed = bool(np.allclose(
        reloaded_logits, logits, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL,
    ))
    if not reload_passed:
        raise RuntimeError("resume_checkpoint_reload_logit_mismatch:" + repr(reload_max_abs_diff))

    result = {
        "schema_version": 1,
        "status": "resumed",
        "device": "cuda",
        "runtime": resumed_runtime,
        "config": payload["config"],
        "optimizer_config": payload["optimizer_config"],
        "training_config": payload["training_config"],
        "seed": payload["seed"],
        "start_step": start_step,
        "additional_steps": additional,
        "final_step": total_steps,
        "loss_records": loss_records,
        "final_loss": loss_records[-1]["loss"],
        "logits": [float(value) for value in logits],
        "checkpoint": checkpoint,
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "sha256": payload["sha256"],
        },
        "input_metadata": payload["input_metadata"],
        "gpu_reload": {
            "max_abs_diff": reload_max_abs_diff,
            "atol": ALLCLOSE_ATOL,
            "rtol": ALLCLOSE_RTOL,
            "passed": reload_passed,
        },
    }
    _write_json_new(out_root / "resume_result.json", result)
    return {
        "status": "resumed",
        "out": str(out_root),
        "device": "cuda",
        "start_step": start_step,
        "additional_steps": additional,
        "final_step": total_steps,
        "final_loss": result["final_loss"],
        "checkpoint_sha256": checkpoint["sha256"],
        "gpu_reload_max_abs_diff": reload_max_abs_diff,
    }


def _validate_prediction_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValueError("prediction_request_not_mapping")
    if request.get("schema_version") != 1:
        raise ValueError("prediction_request_schema_unsupported")
    purpose = request.get("purpose")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("prediction_request_purpose_invalid")
    pair_keys = request.get("pair_keys")
    if not isinstance(pair_keys, list) or not pair_keys:
        raise ValueError("prediction_pair_keys_empty")
    if any(not isinstance(key, str) or not key for key in pair_keys):
        raise ValueError("prediction_pair_key_invalid")
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("prediction_pair_key_duplicate")
    row_indices = request.get("training_row_indices")
    if (
        not isinstance(row_indices, list)
        or len(row_indices) != len(pair_keys)
        or any(type(value) is not int or value < 0 for value in row_indices)
    ):
        raise ValueError("prediction_training_row_indices_invalid")
    if len(set(row_indices)) != len(row_indices):
        raise ValueError("prediction_training_row_indices_duplicate")
    validated = {
        "schema_version": 1,
        "purpose": purpose,
        "pair_keys": list(pair_keys),
        "training_row_indices": list(row_indices),
    }
    for name in (
        "checkpoint_sha256",
        "training_arrays_sha256",
        "prediction_arrays_sha256",
        "development_manifest_sha256",
        "feature_manifest_sha256",
    ):
        validated[name] = _require_sha256(request.get(name), name)
    return validated


def run_prediction(
    *,
    checkpoint_path: os.PathLike[str] | str,
    arrays_path: os.PathLike[str] | str,
    request_path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Consume only a checkpoint plus a bound numeric prediction batch on CUDA."""
    destination = Path(os.path.expanduser(os.fspath(out_dir)))
    if not destination.is_absolute():
        raise ValueError("out_dir_must_be_absolute")
    if not destination.is_dir():
        raise ValueError("out_dir_missing:" + str(destination))
    result_path = destination / "gpu_prediction.json"
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("gpu_prediction_already_exists:" + str(result_path))
    checkpoint_file = Path(os.path.expanduser(os.fspath(checkpoint_path)))
    arrays_file = Path(os.path.expanduser(os.fspath(arrays_path)))
    request_file = Path(os.path.expanduser(os.fspath(request_path)))
    for path, code in (
        (checkpoint_file, "prediction_checkpoint_missing"),
        (arrays_file, "prediction_arrays_missing"),
        (request_file, "prediction_request_missing"),
    ):
        if not path.is_file():
            raise ValueError(code + ":" + str(path))
    request = _validate_prediction_request(_read_json(request_file))
    actual_checkpoint_sha256 = sha256_file(checkpoint_file)
    if actual_checkpoint_sha256 != request["checkpoint_sha256"]:
        raise ValueError("prediction_checkpoint_sha256_mismatch")
    actual_arrays_sha256 = sha256_file(arrays_file)
    if actual_arrays_sha256 != request["prediction_arrays_sha256"]:
        raise ValueError("prediction_arrays_sha256_mismatch")
    try:
        with np.load(arrays_file, allow_pickle=False) as arrays:
            if set(arrays.files) != {"features"}:
                raise ValueError("prediction_arrays_keys_invalid")
            features = arrays["features"]
    except ValueError as exc:
        raise ValueError("prediction_arrays_load_rejected:" + type(exc).__name__) from exc
    features_array = validate_feature_blocks(features)
    if int(features_array.shape[0]) != len(request["pair_keys"]):
        raise ValueError("prediction_arrays_rows_mismatch")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("cuda_unavailable_for_prediction")
    payload = load_checkpoint(
        checkpoint_file,
        map_location="cpu",
        expected_sha256=actual_checkpoint_sha256,
    )
    metadata = payload["input_metadata"]
    if metadata["purpose"] != request["purpose"]:
        raise ValueError("prediction_purpose_mismatch")
    if metadata["arrays_sha256"] != request["training_arrays_sha256"]:
        raise ValueError("prediction_training_arrays_binding_mismatch")
    if metadata.get("development_manifest_sha256") != (
        request["development_manifest_sha256"]
    ):
        raise ValueError("prediction_development_manifest_binding_mismatch")
    if metadata.get("feature_manifest_sha256") != request["feature_manifest_sha256"]:
        raise ValueError("prediction_feature_manifest_binding_mismatch")
    training_pair_keys = metadata["pair_keys"]
    for pair_key, row_index in zip(
        request["pair_keys"], request["training_row_indices"],
    ):
        if row_index >= len(training_pair_keys):
            raise ValueError("prediction_training_row_index_out_of_range")
        if training_pair_keys[row_index] != pair_key:
            raise ValueError("prediction_pair_key_binding_mismatch:" + pair_key)
    diagnostics = configure_runtime()
    runtime = cuda_runtime_report()
    runtime["tf32"] = diagnostics
    runtime["prediction_device"] = "cuda"
    model = build_model(payload["config"]).to("cuda")
    model.load_state_dict(payload["model_state_dict"])
    logits = predict_arrays(model, features_array, device="cuda")
    if not bool(np.isfinite(logits).all()):
        raise RuntimeError("prediction_non_finite_logits")
    result = {
        "schema_version": 1,
        "status": "predicted",
        "device": "cuda",
        "runtime": runtime,
        "config": payload["config"],
        "checkpoint": {
            "path": str(checkpoint_file),
            "sha256": actual_checkpoint_sha256,
            "step": payload["step"],
            "global_step": payload["global_step"],
            "seed": payload["seed"],
            "purpose": payload["input_metadata"]["purpose"],
        },
        "prediction_arrays": {
            "path": str(arrays_file),
            "sha256": actual_arrays_sha256,
            "rows": int(features_array.shape[0]),
        },
        "request": {
            "path": str(request_file),
            "sha256": sha256_file(request_file),
        },
        "pair_keys": list(request["pair_keys"]),
        "training_row_indices": list(request["training_row_indices"]),
        "logits": [float(value) for value in logits],
        "training_runtime": dict(payload["runtime"]),
    }
    _write_json_new(result_path, result)
    return {
        "status": "predicted",
        "out": str(destination),
        "device": "cuda",
        "device_name": runtime.get("device_name"),
        "pair_keys": list(request["pair_keys"]),
        "logits": result["logits"],
        "checkpoint_sha256": actual_checkpoint_sha256,
        "prediction_arrays_sha256": actual_arrays_sha256,
        "gpu_prediction_path": str(result_path),
    }

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
    development_record = input_manifest.get("development_manifest")
    if isinstance(development_record, Mapping):
        input_metadata["development_manifest_sha256"] = _require_sha256(
            development_record.get("sha256"), "development_manifest_sha256",
        )
    source_record = input_manifest.get("source")
    if isinstance(source_record, Mapping):
        for binding in (
            "frozen_manifest_sha256",
            "entities_sha256",
            "feature_manifest_sha256",
        ):
            value = source_record.get(binding)
            if value is not None:
                input_metadata[binding] = _require_sha256(value, binding)
    checkpoint = save_checkpoint(
        out_root / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=params["steps"],
        seed=params["seed"],
        config=config,
        input_metadata=input_metadata,
        runtime=runtime,
        training_config={
            "batch_size": params["batch_size"],
            "sampling": sampling,
        },
        recovery_rng_states=_capture_recovery_rng_states(
            result["sampling_generator"],
        ),
        require_cuda_runtime=True,
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
        description="CUDA-only pair-head training/checkpoint/prediction runner.",
    )
    parser.add_argument("--arrays", help="training_arrays.npz from pair_pipeline")
    parser.add_argument("--out", required=True, help="existing new run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument(
        "--resume-run",
        help="existing run directory whose checkpoint.pt is resumed",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help="optional sha256 pin for a resumed or predicted checkpoint",
    )
    parser.add_argument("--predict-arrays", help="numeric prediction batch NPZ")
    parser.add_argument("--predict-request", help="prediction request JSON")
    parser.add_argument("--checkpoint", help="checkpoint.pt for standalone prediction")
    args = parser.parse_args(argv)
    try:
        if args.resume_run is not None:
            summary = resume_training(
                resume_run=args.resume_run,
                out_dir=args.out,
                additional_steps=args.steps,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            )
        elif (
            args.predict_arrays is not None
            or args.predict_request is not None
            or args.checkpoint is not None
        ):
            if not (args.predict_arrays and args.predict_request and args.checkpoint):
                raise ValueError("prediction_arguments_incomplete")
            summary = run_prediction(
                checkpoint_path=args.checkpoint,
                arrays_path=args.predict_arrays,
                request_path=args.predict_request,
                out_dir=args.out,
            )
        else:
            if not args.arrays:
                raise ValueError("arrays_argument_required")
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

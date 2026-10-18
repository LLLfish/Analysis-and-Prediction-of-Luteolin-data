"""Behavior test for the registered DDA CUDA interpreter (Phase 03 D-05/D-06).

The test runner itself is the CPU preprocessing interpreter
(`E:/Anaconda/python.exe`).  The only interpreter allowed to answer CUDA
questions is the pinned, pre-existing DDA Conda interpreter, and it is invoked
as an isolated subprocess; its CPU build is never imported into the runner.

The subprocess must do more than enumerate devices: it executes a real 2x2 CUDA
matmul, calls `sum().backward()` and `torch.cuda.synchronize()`, and reports
the exact forward/backward values plus the sm_120 architecture facts.  No
environment is created, installed, upgraded or modified by this test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


GPU_PYTHON = Path("E:/paper/DDA-Reproduction/.conda/python.exe")

EXPECTED_TORCH_BUILD = "2.9.1+cu128"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5060 Ti"
EXPECTED_CAPABILITY = [12, 0]
EXPECTED_ARCH = "sm_120"
EXPECTED_MATMUL = [[5.0, 11.0], [11.0, 25.0]]
EXPECTED_GRAD = [[8.0, 12.0], [8.0, 12.0]]
EXPECTED_LOSS = 52.0
EXPECTED_ELEMENTWISE_PROBE = 20.0

_CHILD_SCRIPT = r"""
import json
import sys

import torch


def main():
    report = {
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if not report["cuda_available"]:
        print(json.dumps(report))
        return 1
    report["device_name"] = torch.cuda.get_device_name(0)
    report["capability"] = list(torch.cuda.get_device_capability(0))
    report["arch_list"] = list(torch.cuda.get_arch_list())

    a = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float32, requires_grad=True,
    )
    product = a @ a.t()
    loss = product.sum()
    loss.backward()
    torch.cuda.synchronize()

    report["matmul"] = product.detach().cpu().tolist()
    report["grad"] = a.grad.detach().cpu().tolist()
    report["loss"] = float(loss.detach().cpu())
    report["loss_finite"] = bool(torch.isfinite(loss).item())
    report["grad_finite"] = bool(torch.isfinite(a.grad).all().item())
    report["used_cuda_tensor"] = bool(product.is_cuda and a.grad.is_cuda)
    report["dtype"] = str(product.dtype)
    report["synchronize_ok"] = True

    # Independent elementwise kernel so the probe cannot pass on metadata alone.
    probe = (a * 2.0).sum().item()
    torch.cuda.synchronize()
    report["elementwise_probe"] = float(probe)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


@pytest.fixture(scope="module")
def cuda_report():
    """Run the pinned CUDA interpreter once and return its JSON behavior report."""
    if not GPU_PYTHON.is_file():
        pytest.fail(f"registered CUDA interpreter is missing: {GPU_PYTHON}")
    if Path(sys.executable).resolve() == GPU_PYTHON.resolve():
        pytest.fail(
            "this test must run under the CPU preprocessing interpreter, "
            "not under the GPU interpreter under test"
        )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [str(GPU_PYTHON), "-B", "-X", "utf8", "-c", _CHILD_SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        env=env,
    )
    if completed.returncode != 0:
        pytest.fail(
            "CUDA subprocess failed (exit {}).\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                completed.returncode, completed.stdout, completed.stderr,
            )
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        pytest.fail("CUDA subprocess produced no JSON report")
    return json.loads(lines[-1])


def test_registered_gpu_interpreter_reports_expected_runtime(cuda_report):
    assert Path(cuda_report["executable"]).resolve() == GPU_PYTHON.resolve()
    assert cuda_report["python"].startswith("3.11."), cuda_report["python"]
    assert cuda_report["torch"] == EXPECTED_TORCH_BUILD, cuda_report["torch"]
    assert cuda_report["torch_cuda"] == EXPECTED_TORCH_CUDA, cuda_report["torch_cuda"]
    assert cuda_report["cuda_available"] is True
    assert cuda_report["device_count"] >= 1
    assert cuda_report["device_name"] == EXPECTED_DEVICE_NAME, cuda_report["device_name"]
    assert cuda_report["capability"] == EXPECTED_CAPABILITY, cuda_report["capability"]
    assert EXPECTED_ARCH in cuda_report["arch_list"], cuda_report["arch_list"]


def test_registered_gpu_interpreter_executes_sm120_forward_backward(cuda_report):
    assert cuda_report["dtype"] == "torch.float32", cuda_report["dtype"]
    assert cuda_report["used_cuda_tensor"] is True
    assert cuda_report["loss_finite"] is True
    assert cuda_report["grad_finite"] is True
    assert cuda_report["matmul"] == EXPECTED_MATMUL, cuda_report["matmul"]
    assert cuda_report["grad"] == EXPECTED_GRAD, cuda_report["grad"]
    assert cuda_report["loss"] == pytest.approx(EXPECTED_LOSS, abs=0.0)
    assert cuda_report["elementwise_probe"] == pytest.approx(
        EXPECTED_ELEMENTWISE_PROBE, abs=0.0,
    )
    assert cuda_report["synchronize_ok"] is True

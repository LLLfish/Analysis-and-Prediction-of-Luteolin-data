"""Phase 03 minimal pair-training tracer package.

Two modules only:

* training.pair_pipeline -- CPU preprocessing, frozen-input binding, CLI.
* training.gpu_model     -- CUDA-only head training, safe checkpoint, portable
                            prediction; import closure torch/NumPy/stdlib.

The package initialiser deliberately imports nothing so the registered CUDA
interpreter can import training.gpu_model without RDKit or pytest installed.
"""

from __future__ import annotations

__all__ = ["gpu_model", "pair_pipeline"]

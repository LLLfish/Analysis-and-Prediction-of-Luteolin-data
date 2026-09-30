import os
import json
import logging
import ntpath
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from logging_config import setup_logger
logger = setup_logger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None

    class _TorchUnavailableModule:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to construct CNN3D")

    class _UnavailableNN:
        Module = _TorchUnavailableModule

    nn = _UnavailableNN()
    logger.warning("PyTorch not installed. CNN features will be limited.")

RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not installed. 3D molecule processing will be limited.")




class CNN3D(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: List[int] = None, out_dim: int = 64):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        layers = []
        in_c = in_channels
        for h in hidden_channels:
            layers.append(nn.Conv3d(in_c, h, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(h))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool3d(2))
            in_c = h
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(hidden_channels[-1], out_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


class MolecularCNN:
    def __init__(
        self,
        output_dir: str = "",
        grid_size: int = 32,
        voxel_size: float = 1.0,
        *,
        weights_status: str = "unverified",
        model=None,
    ):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "features", "cnn")
        self.output_dir = os.path.abspath(output_dir)
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.model = model
        self.weights_status = weights_status
        self.status = (
            __import__("data_contracts").StageStatus("skipped_untrained", "verified_trained_weights_not_loaded")
            if TORCH_AVAILABLE
            else __import__("data_contracts").StageStatus("unavailable", "pytorch_unavailable")
        )

    def _has_verified_weights(self) -> bool:
        # P0 has no checkpoint loader, so caller-controlled flags are never evidence.
        return False

    def molecule_to_grid(self, mol) -> Optional[np.ndarray]:
        if not RDKIT_AVAILABLE or mol is None:
            return None
        try:
            conf = mol.GetConformer()
            grid = np.zeros((1, self.grid_size, self.grid_size, self.grid_size))
            center = np.array([self.grid_size / 2] * 3)
            for atom_idx in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(atom_idx)
                atomic_num = mol.GetAtomWithIdx(atom_idx).GetAtomicNum()
                x = int(pos.x / self.voxel_size + center[0])
                y = int(pos.y / self.voxel_size + center[1])
                z = int(pos.z / self.voxel_size + center[2])
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size and 0 <= z < self.grid_size:
                    grid[0, x, y, z] = min(atomic_num / 50.0, 1.0)
            logger.info("Generated 3D grid: %s", str(grid.shape))
            return grid
        except Exception as e:
            logger.error("Molecule to grid conversion failed: %s", e)
            return None

    def extract_features(self, mol_grid: np.ndarray) -> Optional[np.ndarray]:
        if not TORCH_AVAILABLE:
            self.status = __import__("data_contracts").StageStatus("unavailable", "pytorch_unavailable")
            return None
        if not self._has_verified_weights():
            self.status = __import__("data_contracts").StageStatus(
                "skipped_untrained", "verified_trained_weights_not_loaded",
            )
            return None
        if not isinstance(mol_grid, np.ndarray) or mol_grid.ndim != 4 or not np.isfinite(mol_grid).all():
            self.status = __import__("data_contracts").StageStatus("error", "invalid_molecular_grid")
            return None
        try:
            with torch.no_grad():
                tensor = torch.from_numpy(mol_grid).float().unsqueeze(0).to(
                    next(self.model.parameters()).device
                )
                features = self.model(tensor)
            if not torch.is_tensor(features) or features.ndim != 2 or not torch.isfinite(features).all():
                self.status = __import__("data_contracts").StageStatus("error", "cnn_output_shape_invalid")
                return None
            self.status = __import__("data_contracts").StageStatus("ready")
            logger.info("CNN features extracted: %s", str(tuple(features.shape)))
            return features.detach().cpu().numpy()
        except Exception as exc:
            self.status = __import__("data_contracts").StageStatus(
                "error", f"cnn_inference_failed:{type(exc).__name__}",
            )
            logger.error("CNN feature extraction failed: %s", exc)
            return None

    def save_features(self, features: np.ndarray, filename: str = "cnn_features.npy"):
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or os.path.basename(filename) != filename
            or ntpath.basename(filename) != filename
            or os.path.isabs(filename)
            or ntpath.isabs(filename)
            or "/" in filename
            or "\\" in filename
            or any(character in filename for character in '<>:"|?*')
            or filename.endswith((" ", "."))
        ):
            raise ValueError("filename must be a basename without path separators")
        output_root = os.path.realpath(self.output_dir)
        os.makedirs(output_root, exist_ok=True)
        filepath = os.path.realpath(os.path.join(output_root, filename))
        if os.path.normcase(os.path.commonpath((output_root, filepath))) != os.path.normcase(output_root):
            raise ValueError("feature path resolves outside the configured output root")
        np.save(filepath, features)
        logger.info("Saved CNN features: %s", filepath)

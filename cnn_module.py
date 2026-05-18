import os
import json
import logging
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
    logger.warning("PyTorch not installed. CNN features will be limited.")

RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not installed. 3D molecule processing will be limited.")

DEEPCHEM_AVAILABLE = False
try:
    import deepchem as dc
    DEEPCHEM_AVAILABLE = True
except ImportError:
    logger.warning("DeepChem not installed. Advanced CNN features will be unavailable.")


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
    def __init__(self, output_dir: str = "", grid_size: int = 32, voxel_size: float = 1.0):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "features", "cnn")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.model = None
        if TORCH_AVAILABLE:
            self.model = CNN3D()

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
        if self.model is None or not TORCH_AVAILABLE:
            return None
        try:
            with torch.no_grad():
                tensor = torch.from_numpy(mol_grid).float().unsqueeze(0)
                features = self.model(tensor).numpy()
                logger.info("CNN features extracted: %s", str(features.shape))
                return features
        except Exception as e:
            logger.error("CNN feature extraction failed: %s", e)
            return None

    def save_features(self, features: np.ndarray, filename: str = "cnn_features.npy"):
        filepath = os.path.join(self.output_dir, filename)
        np.save(filepath, features)
        logger.info("Saved CNN features: %s", filepath)
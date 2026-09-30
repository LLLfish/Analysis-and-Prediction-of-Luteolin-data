import os
import json
import logging
import ntpath
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
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
            raise RuntimeError("PyTorch is required to construct Transformer models")

    class _UnavailableNN:
        Module = _TorchUnavailableModule

    nn = _UnavailableNN()
    logger.warning("PyTorch not installed. Transformer features will be limited.")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerFusion(nn.Module):
    def __init__(self, gnn_dim: int = 64, cnn_dim: int = 64, seq_dim: int = 64,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.gnn_proj = nn.Linear(gnn_dim, d_model) if gnn_dim != d_model else nn.Identity()
        self.cnn_proj = nn.Linear(cnn_dim, d_model) if cnn_dim != d_model else nn.Identity()
        self.seq_proj = nn.Linear(seq_dim, d_model) if seq_dim != d_model else nn.Identity()
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.fusion_layer = nn.Linear(d_model * 3, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, gnn_feat, cnn_feat, seq_feat):
        gnn = self.gnn_proj(gnn_feat).unsqueeze(1)
        cnn = self.cnn_proj(cnn_feat).unsqueeze(1)
        seq = self.seq_proj(seq_feat).unsqueeze(1)
        x = torch.cat([gnn, cnn, seq], dim=1)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        fused = x.view(x.size(0), -1)
        fused = self.fusion_layer(fused)
        output = self.output_proj(fused)
        return output


class FeatureFusion:
    def __init__(
        self,
        output_dir: str = "",
        *,
        weights_status: str = "unverified",
        model=None,
    ):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "features", "fusion")
        self.output_dir = os.path.abspath(output_dir)
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

    def fuse_features(self, gnn_features: np.ndarray, cnn_features: np.ndarray,
                      seq_features: np.ndarray) -> Optional[np.ndarray]:
        if not TORCH_AVAILABLE:
            self.status = __import__("data_contracts").StageStatus("unavailable", "pytorch_unavailable")
            return None
        if not self._has_verified_weights():
            self.status = __import__("data_contracts").StageStatus(
                "skipped_untrained", "verified_trained_weights_not_loaded",
            )
            return None
        arrays = (gnn_features, cnn_features, seq_features)
        if any(
            not isinstance(value, np.ndarray)
            or value.size != 64
            or not np.isfinite(value).all()
            for value in arrays
        ):
            self.status = __import__("data_contracts").StageStatus("error", "fusion_input_shape_or_value_invalid")
            return None
        try:
            tensors = [torch.from_numpy(value.reshape(1, 64)).float() for value in arrays]
            with torch.no_grad():
                output = self.model(*tensors)
            if not torch.is_tensor(output) or output.ndim != 2 or not torch.isfinite(output).all():
                self.status = __import__("data_contracts").StageStatus("error", "fusion_output_shape_invalid")
                return None
            self.status = __import__("data_contracts").StageStatus("ready")
            logger.info("Features fused via Transformer: %s", str(tuple(output.shape)))
            return output.detach().cpu().numpy()
        except Exception as exc:
            self.status = __import__("data_contracts").StageStatus(
                "error", f"transformer_inference_failed:{type(exc).__name__}",
            )
            logger.error("Transformer fusion failed: %s", exc)
            return None

    def save_fused_features(self, features: np.ndarray, filename: str = "fused_features.npy"):
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
        logger.info("Saved fused features: %s", filepath)

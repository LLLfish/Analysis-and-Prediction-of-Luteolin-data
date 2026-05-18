import os
import json
import logging
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
    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(_BASE_DIR, "features", "fusion")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None
        if TORCH_AVAILABLE:
            self.model = TransformerFusion()

    def fuse_features(self, gnn_features: np.ndarray, cnn_features: np.ndarray,
                      seq_features: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None or not TORCH_AVAILABLE:
            logger.warning("Transformer not available, using concatenation")
            try:
                fused = np.concatenate([
                    gnn_features.flatten(),
                    cnn_features.flatten(),
                    seq_features.flatten()
                ])
                return fused.reshape(1, -1)
            except Exception as e:
                logger.error("Feature concatenation failed: %s", e)
                return None
        try:
            with torch.no_grad():
                gnn_t = torch.from_numpy(gnn_features).float()
                cnn_t = torch.from_numpy(cnn_features).float()
                seq_t = torch.from_numpy(seq_features).float()
                if gnn_t.dim() == 1:
                    gnn_t = gnn_t.unsqueeze(0)
                if cnn_t.dim() == 1:
                    cnn_t = cnn_t.unsqueeze(0)
                if seq_t.dim() == 1:
                    seq_t = seq_t.unsqueeze(0)
                output = self.model(gnn_t, cnn_t, seq_t)
                logger.info("Features fused via Transformer: %s", str(output.shape))
                return output.numpy()
        except Exception as e:
            logger.error("Transformer fusion failed: %s", e)
            return None

    def save_fused_features(self, features: np.ndarray, filename: str = "fused_features.npy"):
        filepath = os.path.join(self.output_dir, filename)
        np.save(filepath, features)
        logger.info("Saved fused features: %s", filepath)
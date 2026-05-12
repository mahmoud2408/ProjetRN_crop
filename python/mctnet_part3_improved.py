from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from mctnet import MCTNet, build_mctnet


def compute_vi_timeseries_improved(X: np.ndarray) -> np.ndarray:
    """
    Compute VI-enriched temporal features.

    Args:
        X: Sentinel-2 tensor of shape (N, 36, 10).

    Returns:
        Tensor of shape (N, 36, 14) with 10 spectral bands followed by
        NDVI, IRECI, MTCI and S2REP.
    """
    eps = 1e-8
    X = np.asarray(X, dtype=np.float32)
    valid = np.any(np.abs(X) > 0.0, axis=-1, keepdims=True)

    b4 = X[:, :, 2]
    b5 = X[:, :, 3]
    b6 = X[:, :, 4]
    b7 = X[:, :, 5]
    b8 = X[:, :, 6]

    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ireci = (b7 - b4) / (b5 + eps)
    mtci = (b6 - b5) / (b5 - b4 + eps)
    s2rep = 705.0 + 35.0 * ((((b4 + b7) * 0.5) - b5) / (b6 - b5 + eps))

    ndvi = np.clip(ndvi, -1.5, 1.5)
    ireci = np.clip(ireci, -5.0, 5.0)
    mtci = np.clip(mtci, -5.0, 5.0)
    s2rep = np.clip(s2rep, 650.0, 850.0)

    vi = np.stack([ndvi, ireci, mtci, s2rep], axis=-1).astype(np.float32)
    vi = np.where(valid, vi, 0.0).astype(np.float32)
    return np.concatenate([X, vi], axis=-1).astype(np.float32)


def normalize_vi_columns_improved(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize only the 4 VI channels from train statistics.

    Args:
        X_tr: Train tensor of shape (N_train, 36, 14).
        X_va: Validation tensor of shape (N_val, 36, 14).
        X_te: Test tensor of shape (N_test, 36, 14).

    Returns:
        Tuple containing normalized train/val/test tensors and VI mean/std,
        where each mean/std array has shape (4,).
    """
    train = np.array(X_tr, dtype=np.float32, copy=True)
    val = np.array(X_va, dtype=np.float32, copy=True)
    test = np.array(X_te, dtype=np.float32, copy=True)

    train_valid = np.any(np.abs(train[:, :, :10]) > 0.0, axis=-1)
    vi_train = train[:, :, 10:14]

    if np.any(train_valid):
        vi_values = vi_train[train_valid]
        vi_mean = vi_values.mean(axis=0).astype(np.float32)
        vi_std = vi_values.std(axis=0).astype(np.float32)
    else:
        vi_mean = np.zeros(4, dtype=np.float32)
        vi_std = np.ones(4, dtype=np.float32)

    vi_std = np.where(vi_std < 1e-8, 1.0, vi_std).astype(np.float32)

    def _apply_norm(X: np.ndarray) -> np.ndarray:
        out = np.array(X, dtype=np.float32, copy=True)
        valid = np.any(np.abs(out[:, :, :10]) > 0.0, axis=-1)
        normalized = (out[:, :, 10:14] - vi_mean.reshape(1, 1, 4)) / vi_std.reshape(1, 1, 4)
        normalized = np.where(valid[:, :, None], normalized, 0.0).astype(np.float32)
        out[:, :, 10:14] = normalized
        return out

    return _apply_norm(train), _apply_norm(val), _apply_norm(test), vi_mean, vi_std


def build_class_weights(
    y_train: np.ndarray,
    num_classes: int,
    device: torch.device,
    power: float = 0.5,
) -> torch.Tensor:
    """
    Build normalized class weights from class frequencies.

    Args:
        y_train: Training labels of shape (N,).
        num_classes: Number of classes.
        device: Target device.
        power: Frequency inverse exponent. `1.0` matches inverse-frequency
            weighting, while `0.5` is a softer weighting.

    Returns:
        Tensor of shape (num_classes,).
    """
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    frequencies = counts / counts.sum()
    if power <= 0.0:
        weights = np.ones(num_classes, dtype=np.float64)
    else:
        weights = np.power(np.clip(frequencies, 1e-12, None), -power)
    weights = weights * (float(num_classes) / weights.sum())
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


class StaticLateFusionHead(nn.Module):
    """
    Late-fusion encoder for static environmental covariates.

    Input shape:
        x: (N, E_static)

    Output shape:
        (N, hidden_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode static covariates.

        Args:
            x: Tensor of shape (N, E_static).

        Returns:
            Tensor of shape (N, hidden_dim).
        """
        return self.net(x)


class MCTNetPart3Improved(nn.Module):
    """
    Improved Part 3 model built on the original MCTNet backbone.

    Input shapes:
        x: (N, 36, C_temporal)
        valid_mask: (N, 36)
        dynamic_env: optional (N, 36, E_dynamic)
        static_env: optional (N, E_static)

    Output shape:
        logits: (N, num_classes)
    """

    def __init__(
        self,
        temporal_input_dim: int = 14,
        static_input_dim: int = 0,
        num_classes: int = 5,
        seq_len: int = 36,
        n_stages: int = 3,
        n_heads: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
        static_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.temporal_input_dim = temporal_input_dim
        self.static_input_dim = static_input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.temporal_backbone: MCTNet = build_mctnet(
            num_classes=num_classes,
            input_dim=temporal_input_dim,
            seq_len=seq_len,
            n_stages=n_stages,
            n_heads=n_heads,
            kernel_size=kernel_size,
            dropout=dropout,
            use_alpe=True,
            use_missing_mask=True,
            use_cnn_branch=True,
            use_transformer_branch=True,
        )
        self.temporal_feature_dim = int(self.temporal_backbone.final_seq_len)
        self.static_feature_dim = int(static_hidden_dim if static_input_dim > 0 else 0)
        self.static_head = (
            StaticLateFusionHead(input_dim=static_input_dim, hidden_dim=static_hidden_dim, dropout=dropout)
            if static_input_dim > 0 else None
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.temporal_feature_dim + self.static_feature_dim, num_classes)

    def forward(
        self,
        x: Tensor,
        valid_mask: Optional[Tensor] = None,
        dynamic_env: Optional[Tensor] = None,
        static_env: Optional[Tensor] = None,
        return_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """
        Run a forward pass.

        Args:
            x: Tensor of shape (N, 36, C_temporal_base).
            valid_mask: Optional tensor of shape (N, 36).
            dynamic_env: Optional tensor of shape (N, 36, E_dynamic).
            static_env: Optional tensor of shape (N, E_static).
            return_features: Whether to also return intermediate features.

        Returns:
            Logits of shape (N, num_classes), optionally with a feature dict.
        """
        temporal_features = self.temporal_backbone.extract_pooled_features(
            x=x,
            valid_mask=valid_mask,
            dynamic_env=dynamic_env,
            static_env=None,
            return_stage_features=False,
        )

        feature_parts = [temporal_features]
        static_features: Optional[Tensor] = None
        if self.static_head is not None:
            if static_env is None:
                static_features = temporal_features.new_zeros((temporal_features.shape[0], self.static_feature_dim))
            else:
                static_features = self.static_head(static_env)
            feature_parts.append(static_features)

        fused_features = torch.cat(feature_parts, dim=-1)
        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)

        if return_features:
            return logits, {
                'temporal_features': temporal_features,
                'static_features': static_features if static_features is not None else temporal_features.new_zeros((temporal_features.shape[0], 0)),
                'fused_features': fused_features,
            }
        return logits

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_sinusoidal_positional_encoding(
    seq_len: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Build a sinusoidal positional encoding.

    Args:
        seq_len: Temporal length.
        dim: Embedding dimension.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape (seq_len, dim).
    """
    encoding = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    if dim == 0:
        return encoding
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    even_indices = torch.arange(0, dim, 2, device=device, dtype=dtype)
    div_term = torch.exp(even_indices * (-math.log(10000.0) / max(dim, 1)))
    phase = position * div_term
    encoding[:, 0::2] = torch.sin(phase)
    odd_width = encoding[:, 1::2].shape[1]
    if odd_width > 0:
        encoding[:, 1::2] = torch.cos(phase[:, :odd_width])
    return encoding


class ECA(nn.Module):
    """
    Efficient Channel Attention for 1D temporal features.

    Input shape:
        x: (N, C, T)

    Output shape:
        (N, C, T)
    """

    def __init__(self, channels: int, k: int = 3) -> None:
        super().__init__()
        padding = (k - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply channel attention.

        Args:
            x: Tensor of shape (N, C, T).

        Returns:
            Tensor of shape (N, C, T).
        """
        weights = x.mean(dim=-1, keepdim=True).transpose(1, 2)
        weights = self.conv(weights)
        weights = torch.sigmoid(weights).transpose(1, 2)
        return x * weights


class ALPEPlus(nn.Module):
    """
    Adaptive Learnable Positional Encoding with masked temporal support.

    Input shape:
        mask: (N, T)

    Output shape:
        (N, T, input_dim)
    """

    def __init__(self, seq_len: int = 36, input_dim: int = 14, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size, padding=padding, bias=True)
        self.eca = ECA(channels=input_dim, k=3)

    def forward(self, mask: Tensor) -> Tensor:
        """
        Build masked learnable positional encodings.

        Args:
            mask: Tensor of shape (N, T).

        Returns:
            Tensor of shape (N, T, input_dim).
        """
        encoding = _build_sinusoidal_positional_encoding(
            seq_len=self.seq_len,
            dim=self.input_dim,
            device=mask.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(mask.shape[0], -1, -1)
        masked_encoding = encoding * mask.unsqueeze(-1).to(encoding.dtype)
        masked_encoding = masked_encoding.transpose(1, 2)
        masked_encoding = self.conv(masked_encoding)
        masked_encoding = self.eca(masked_encoding)
        return masked_encoding.transpose(1, 2)


class CNNSubModulePlus(nn.Module):
    """
    Residual temporal CNN sub-module.

    Input shape:
        x: (N, C, T)

    Output shape:
        (N, out_dim, T)
    """

    def __init__(self, in_dim: int, out_dim: int, kernel: int = 3) -> None:
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=kernel, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=kernel, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the residual CNN branch.

        Args:
            x: Tensor of shape (N, C, T).

        Returns:
            Tensor of shape (N, out_dim, T).
        """
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        return out


class GRUSubModule(nn.Module):
    """
    Temporal GRU sub-module.

    Input shape:
        x: (N, T, in_dim)

    Output shape:
        (N, T, out_dim)
    """

    def __init__(self, in_dim: int, hidden: int, bidirectional: bool = False) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the GRU branch.

        Args:
            x: Tensor of shape (N, T, in_dim).

        Returns:
            Tensor of shape (N, T, out_dim).
        """
        out, _ = self.gru(x)
        return out


class CTFusionGRU(nn.Module):
    """
    CNN-GRU fusion stage with optional temporal pooling.

    Input shape:
        x: (N, T, in_dim)

    Output shape:
        (N, T_out, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        cnn_dim: int,
        gru_hidden: int,
        kernel: int = 3,
        bidirectional: bool = False,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = CNNSubModulePlus(in_dim=in_dim, out_dim=cnn_dim, kernel=kernel)
        self.gru = GRUSubModule(in_dim=in_dim, hidden=gru_hidden, bidirectional=bidirectional)
        self.pool = pool
        self.out_dim = cnn_dim + self.gru.out_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply one fusion stage.

        Args:
            x: Tensor of shape (N, T, in_dim).

        Returns:
            Tensor of shape (N, T_out, out_dim).
        """
        cnn_out = self.cnn(x.transpose(1, 2))
        gru_out = self.gru(x)
        if self.pool:
            cnn_out = F.max_pool1d(cnn_out, kernel_size=2, stride=2)
            gru_out = F.max_pool1d(gru_out.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        cnn_out = cnn_out.transpose(1, 2)
        return torch.cat([cnn_out, gru_out], dim=-1)


class MCTNetGRUPlus(nn.Module):
    """
    MCTNet-GRU+ architecture.

    Input shape:
        x: (N, 36, input_dim)
        mask: (N, 36)

    Output shape:
        logits: (N, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 14,
        num_classes: int = 5,
        seq_len: int = 36,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.alpe = ALPEPlus(seq_len=seq_len, input_dim=input_dim)
        self.stage1 = CTFusionGRU(in_dim=input_dim, cnn_dim=32, gru_hidden=32, kernel=3, bidirectional=True, pool=True)
        self.stage2 = CTFusionGRU(in_dim=96, cnn_dim=64, gru_hidden=40, kernel=3, bidirectional=False, pool=True)
        self.stage3 = CTFusionGRU(in_dim=104, cnn_dim=80, gru_hidden=40, kernel=3, bidirectional=False, pool=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(120, num_classes)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Run MCTNet-GRU+ forward.

        Args:
            x: Tensor of shape (N, 36, input_dim).
            mask: Optional tensor of shape (N, 36).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        alpe_enc = self.alpe(mask.to(x.device))
        x = x + alpe_enc.to(x.dtype)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.max(dim=1).values
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


def compute_vi_timeseries(X: np.ndarray) -> np.ndarray:
    """
    Compute vegetation indices and append them to Sentinel-2 features.

    Args:
        X: Sentinel-2 tensor of shape (N, 36, 10).

    Returns:
        Tensor of shape (N, 36, 14).
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
    s2rep = 705.0 + 35.0 * (((b4 + b7) * 0.5) - b5) / (b6 - b5 + eps)

    vi = np.stack([ndvi, ireci, mtci, s2rep], axis=-1)
    vi = np.clip(vi, -5.0, 5.0).astype(np.float32)
    vi = np.where(valid, vi, 0.0).astype(np.float32)
    return np.concatenate([X, vi], axis=-1).astype(np.float32)


def build_class_weights(y_train: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Build normalized inverse-frequency class weights.

    Args:
        y_train: Training labels of shape (N,).
        num_classes: Number of classes.
        device: Target device.

    Returns:
        Tensor of shape (num_classes,).
    """
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    frequencies = counts / counts.sum()
    inverse = 1.0 / np.clip(frequencies, 1e-12, None)
    weights = inverse * (float(num_classes) / inverse.sum())
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def normalize_vi_columns(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize only the 4 vegetation-index channels.

    Args:
        X_tr: Train tensor of shape (N_train, 36, 14).
        X_va: Validation tensor of shape (N_val, 36, 14).
        X_te: Test tensor of shape (N_test, 36, 14).

    Returns:
        Normalized train, val, test tensors and VI mean/std of shape (4,).
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

"""
Technical summary
-----------------
This model combines the strongest compatible ideas from the supplied papers.

1. From the MCTNet paper, it keeps masked Sentinel-2 temporal modeling and an
   ALPE-style learnable positional encoding so missing observations remain part
   of the design instead of being ignored.
2. From the Geo-CBAM-CNN paper, it adds explicit channel and temporal attention
   so the network can suppress uninformative dates and emphasize red-edge-rich
   periods that matter for crop discrimination.
3. From the red-edge CNN-RNN paper, it assumes the 14-channel per-date
   representation and adds a 2D time-feature branch plus a bidirectional GRU to
   capture phenology after local spectral-temporal pattern extraction.
4. From the GEDI/Sentinel fusion paper and the BKA-CNN multi-source idea, it
   keeps dynamic climate covariates in the temporal stream while encoding soil
   and topography separately through a late-fusion static branch.
5. From the supervised-to-unsupervised paper, the architecture exposes a
   reconstruction head so the training script can add masked self-supervised
   regularization on top of supervised classification.

The design is meant to be a stronger replacement for plain MCTNet on the
current project data, with a better chance of exceeding 0.90 OA/F1 by combining
red-edge enrichment, heterogeneous fusion, local-global attention, and
phenology-aware recurrent modeling.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def build_sinusoidal_positional_encoding(
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


def masked_mean(x: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
    """
    Compute a masked mean across the temporal axis.

    Args:
        x: Tensor of shape (B, T, C).
        valid_mask: Optional mask of shape (B, T).

    Returns:
        Tensor of shape (B, C).
    """
    if valid_mask is None:
        return x.mean(dim=1)
    weights = valid_mask.to(x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


class EfficientChannelAttention1D(nn.Module):
    """
    Efficient channel attention over 1D temporal features.

    Input shape:
        x: (B, C, T)

    Output shape:
        (B, C, T)
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply efficient channel attention.

        Args:
            x: Tensor of shape (B, C, T).

        Returns:
            Tensor of shape (B, C, T).
        """
        weights = x.mean(dim=-1, keepdim=True).transpose(1, 2)
        weights = self.conv(weights)
        weights = torch.sigmoid(weights).transpose(1, 2)
        return x * weights


class ChannelTemporalCBAM1D(nn.Module):
    """
    CBAM-style attention adapted to temporal feature maps.

    Input shape:
        x: (B, C, T)

    Output shape:
        (B, C, T)
    """

    def __init__(self, channels: int, reduction: int = 8, temporal_kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        padding = (temporal_kernel_size - 1) // 2
        self.temporal_conv = nn.Conv1d(2, 1, kernel_size=temporal_kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply channel attention followed by temporal attention.

        Args:
            x: Tensor of shape (B, C, T).

        Returns:
            Tensor of shape (B, C, T).
        """
        avg_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)
        channel_weights = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)).unsqueeze(-1)
        x_channel = x * channel_weights

        avg_map = x_channel.mean(dim=1, keepdim=True)
        max_map = x_channel.amax(dim=1, keepdim=True)
        temporal_weights = torch.sigmoid(self.temporal_conv(torch.cat([avg_map, max_map], dim=1)))
        return x_channel * temporal_weights


class ALPEEnhanced(nn.Module):
    """
    ALPE-style masked positional encoding.

    Input shape:
        valid_mask: (B, T)

    Output shape:
        (B, T, D)
    """

    def __init__(self, seq_len: int, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.seq_len = seq_len
        self.dim = dim
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, bias=True)
        self.eca = EfficientChannelAttention1D(channels=dim, kernel_size=3)

    def forward(self, valid_mask: Tensor) -> Tensor:
        """
        Build masked learnable positional encodings.

        Args:
            valid_mask: Tensor of shape (B, T).

        Returns:
            Tensor of shape (B, T, D).
        """
        batch_size = int(valid_mask.shape[0])
        encoding = build_sinusoidal_positional_encoding(
            seq_len=self.seq_len,
            dim=self.dim,
            device=valid_mask.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        masked = encoding * valid_mask.unsqueeze(-1).to(encoding.dtype)
        masked = masked.transpose(1, 2)
        masked = self.conv(masked)
        masked = self.eca(masked)
        return masked.transpose(1, 2)


class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise separable temporal convolution.

    Input shape:
        x: (B, C_in, T)

    Output shape:
        (B, C_out, T)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply a depthwise separable temporal convolution.

        Args:
            x: Tensor of shape (B, C_in, T).

        Returns:
            Tensor of shape (B, C_out, T).
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.act(out)


class MultiScaleTemporalBlock(nn.Module):
    """
    Multi-scale temporal feature extractor with CBAM-style refinement.

    Input shape:
        x: (B, T, D)

    Output shape:
        (B, T, D)
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.branch3 = DepthwiseSeparableConv1D(dim, dim, kernel_size=3)
        self.branch5 = DepthwiseSeparableConv1D(dim, dim, kernel_size=5)
        self.branch7 = DepthwiseSeparableConv1D(dim, dim, kernel_size=7)
        self.fuse = nn.Sequential(
            nn.Conv1d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )
        self.attn = ChannelTemporalCBAM1D(channels=dim, reduction=8, temporal_kernel_size=7)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply multi-scale temporal convolutions and attention refinement.

        Args:
            x: Tensor of shape (B, T, D).
            valid_mask: Optional tensor of shape (B, T).

        Returns:
            Tensor of shape (B, T, D).
        """
        residual = x
        x_1d = x.transpose(1, 2)
        out = torch.cat([self.branch3(x_1d), self.branch5(x_1d), self.branch7(x_1d)], dim=1)
        out = self.fuse(out)
        out = self.attn(out)
        out = out.transpose(1, 2)
        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).to(out.dtype)
        out = self.norm(residual + self.dropout(out))
        return out


class SpectroTemporal2DBlock(nn.Module):
    """
    2D encoder over the time-feature map inspired by 2D CNN-GRU crop models.

    Input shape:
        x: (B, T, C_in)

    Output shape:
        (B, T, D)
    """

    def __init__(self, input_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.project = nn.Linear(64, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        """
        Encode the time-feature matrix with 2D convolutions.

        Args:
            x: Tensor of shape (B, T, C_in).
            valid_mask: Optional tensor of shape (B, T).

        Returns:
            Tensor of shape (B, T, D).
        """
        feature_map = x.unsqueeze(1)
        out = self.encoder(feature_map)
        out = out.mean(dim=-1).transpose(1, 2)
        out = self.project(out)
        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).to(out.dtype)
        return self.dropout(out)


class TransformerEncoderBlock(nn.Module):
    """
    Mask-aware transformer encoder block.

    Input shape:
        x: (B, T, D)

    Output shape:
        (B, T, D)
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, ffn_expansion: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply one transformer encoder block.

        Args:
            x: Tensor of shape (B, T, D).
            valid_mask: Optional tensor of shape (B, T).

        Returns:
            Tensor of shape (B, T, D).
        """
        key_padding_mask = None if valid_mask is None else valid_mask.eq(0)
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout1(attn_out)
        x = x + self.ffn(self.norm2(x))
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        return x


class BranchFusionGate(nn.Module):
    """
    Learnable gating across local, 2D, and transformer branches.

    Input shapes:
        local_branch: (B, T, D)
        cnn2d_branch: (B, T, D)
        transformer_branch: (B, T, D)
        valid_mask: optional (B, T)

    Output shape:
        fused: (B, T, D)
        gate_weights: (B, 3)
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
        )
        self.project = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        local_branch: Tensor,
        cnn2d_branch: Tensor,
        transformer_branch: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Fuse the three temporal branches.

        Args:
            local_branch: Tensor of shape (B, T, D).
            cnn2d_branch: Tensor of shape (B, T, D).
            transformer_branch: Tensor of shape (B, T, D).
            valid_mask: Optional tensor of shape (B, T).

        Returns:
            Tuple (fused, gate_weights).
        """
        summary = torch.cat(
            [
                masked_mean(local_branch, valid_mask),
                masked_mean(cnn2d_branch, valid_mask),
                masked_mean(transformer_branch, valid_mask),
            ],
            dim=-1,
        )
        gate_weights = torch.softmax(self.gate(summary), dim=-1)
        stacked = torch.stack([local_branch, cnn2d_branch, transformer_branch], dim=2)
        fused_weighted = (stacked * gate_weights[:, None, :, None]).sum(dim=2)
        fused_concat = torch.cat([fused_weighted, local_branch, transformer_branch], dim=-1)
        return self.project(fused_concat), gate_weights


class TemporalAttentionPooling(nn.Module):
    """
    Mask-aware attention pooling over time.

    Input shape:
        x: (B, T, D)

    Output shape:
        pooled: (B, D)
        weights: (B, T)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Pool a temporal sequence into one feature vector.

        Args:
            x: Tensor of shape (B, T, D).
            valid_mask: Optional tensor of shape (B, T).

        Returns:
            Tuple (pooled, weights).
        """
        logits = self.score(x).squeeze(-1)
        if valid_mask is not None:
            logits = logits.masked_fill(valid_mask.eq(0), -1e9)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class StaticContextEncoder(nn.Module):
    """
    Encoder for static environmental covariates.

    Input shape:
        x: (B, E_static)

    Output shape:
        (B, H_static)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
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
            x: Tensor of shape (B, E_static).

        Returns:
            Tensor of shape (B, H_static).
        """
        return self.net(x)


class EnhancedMCTNet(nn.Module):
    """
    Enhanced hybrid architecture for crop classification.

    Input shapes:
        x: (B, T, C_temporal)
        valid_mask: optional (B, T)
        static_env: optional (B, E_static)

    Output shapes:
        logits: (B, num_classes)
        reconstruction: optional (B, T, C_temporal)
    """

    def __init__(
        self,
        temporal_input_dim: int,
        num_classes: int,
        seq_len: int = 36,
        static_input_dim: int = 0,
        model_dim: int = 96,
        num_heads: int = 6,
        num_transformer_layers: int = 2,
        gru_hidden_dim: int = 64,
        static_hidden_dim: int = 32,
        dropout: float = 0.1,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.temporal_input_dim = temporal_input_dim
        self.static_input_dim = static_input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers

        self.input_projection = nn.Sequential(
            nn.Linear(temporal_input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.alpe = ALPEEnhanced(seq_len=seq_len, dim=model_dim, kernel_size=3)
        self.local_branch = MultiScaleTemporalBlock(dim=model_dim, dropout=dropout)
        self.spectrotemporal_branch = SpectroTemporal2DBlock(input_dim=temporal_input_dim, out_dim=model_dim, dropout=dropout)
        self.transformer_branch = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.branch_fusion = BranchFusionGate(dim=model_dim, dropout=dropout)
        self.post_fusion_norm = nn.LayerNorm(model_dim)
        self.bigru = nn.GRU(
            input_size=model_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_projection = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )
        self.temporal_pool = TemporalAttentionPooling(dim=model_dim)
        self.reconstruction_head = nn.Linear(model_dim, temporal_input_dim)

        if static_input_dim > 0:
            self.static_encoder = StaticContextEncoder(
                input_dim=static_input_dim,
                hidden_dim=static_hidden_dim,
                dropout=dropout,
            )
            self.static_gate = nn.Sequential(
                nn.Linear(static_hidden_dim, model_dim),
                nn.Sigmoid(),
            )
            classifier_input_dim = model_dim + static_hidden_dim
        else:
            self.static_encoder = None
            self.static_gate = None
            classifier_input_dim = model_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, max(model_dim, 64)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(model_dim, 64), num_classes),
        )

    def encode_temporal(
        self,
        x: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Encode temporal inputs before late fusion.

        Args:
            x: Tensor of shape (B, T, C_temporal).
            valid_mask: Optional tensor of shape (B, T).

        Returns:
            Tuple (sequence_features, aux) where sequence_features has shape
            (B, T, D) and aux stores gate/pooling information.
        """
        projected = self.input_projection(x)
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=projected.dtype)
        projected = projected + self.alpe(valid_mask=valid_mask).to(projected.dtype)
        projected = projected * valid_mask.unsqueeze(-1).to(projected.dtype)

        local_features = self.local_branch(projected, valid_mask=valid_mask)
        spectrotemporal_features = self.spectrotemporal_branch(x, valid_mask=valid_mask)

        transformer_features = projected
        for block in self.transformer_branch:
            transformer_features = block(transformer_features, valid_mask=valid_mask)

        fused, gate_weights = self.branch_fusion(
            local_branch=local_features,
            cnn2d_branch=spectrotemporal_features,
            transformer_branch=transformer_features,
            valid_mask=valid_mask,
        )
        fused = self.post_fusion_norm(fused + projected)
        gru_features, _ = self.bigru(fused)
        gru_features = self.gru_projection(gru_features)
        gru_features = gru_features * valid_mask.unsqueeze(-1).to(gru_features.dtype)

        return gru_features, {
            'gate_weights': gate_weights,
        }

    def forward(
        self,
        x: Tensor,
        valid_mask: Optional[Tensor] = None,
        static_env: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Run the enhanced model.

        Args:
            x: Tensor of shape (B, T, C_temporal).
            valid_mask: Optional tensor of shape (B, T).
            static_env: Optional tensor of shape (B, E_static).
            return_aux: Whether to return auxiliary tensors.

        Returns:
            Dict containing at least `logits` of shape (B, num_classes) and
            `reconstruction` of shape (B, T, C_temporal).
        """
        sequence_features, aux = self.encode_temporal(x=x, valid_mask=valid_mask)
        reconstruction = self.reconstruction_head(sequence_features)
        pooled, pool_weights = self.temporal_pool(sequence_features, valid_mask=valid_mask)

        static_features: Optional[Tensor] = None
        if self.static_encoder is not None:
            if static_env is None:
                static_features = pooled.new_zeros((pooled.shape[0], self.classifier[0].in_features - pooled.shape[1]))
            else:
                static_features = self.static_encoder(static_env)
                pooled = pooled * self.static_gate(static_features)
            classifier_input = torch.cat([pooled, static_features], dim=-1)
        else:
            classifier_input = pooled

        logits = self.classifier(classifier_input)
        outputs: Dict[str, Tensor] = {
            'logits': logits,
            'reconstruction': reconstruction,
        }
        if return_aux:
            outputs['sequence_features'] = sequence_features
            outputs['gate_weights'] = aux['gate_weights']
            outputs['pool_weights'] = pool_weights
        return outputs

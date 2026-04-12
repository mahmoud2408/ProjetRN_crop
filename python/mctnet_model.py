

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MCTNetConfig:
    input_dim: int = 10
    num_classes: int = 4
    seq_len: int = 36
    n_stages: int = 3
    stage_dims: Tuple[int, ...] = (10, 20, 40)
    n_heads: int = 5
    kernel_size: int = 3
    mlp_ratio: float = 6.0
    dropout: float = 0.1
    classifier_hidden_dim: int = 64
    pool_type: str = 'max'
    use_alpe: bool = True
    use_missing_mask: bool = True
    use_cnn_branch: bool = True
    use_transformer_branch: bool = True

    def __post_init__(self) -> None:
        if self.n_stages != len(self.stage_dims):
            raise ValueError('n_stages must match len(stage_dims).')
        for stage_dim in self.stage_dims:
            if stage_dim % self.n_heads != 0:
                raise ValueError(
                    f'stage_dim={stage_dim} must be divisible by n_heads={self.n_heads}.'
                )
        if self.pool_type not in {'max', 'avg'}:
            raise ValueError("pool_type must be 'max' or 'avg'.")


def build_sinusoidal_positional_encoding(
    seq_len: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) *
        (-math.log(10000.0) / dim)
    )

    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class EfficientChannelAttention1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        weights = x.mean(dim=-1, keepdim=True)        # [B, C, 1]
        weights = weights.transpose(1, 2)             # [B, 1, C]
        weights = self.conv(weights)
        weights = torch.sigmoid(weights).transpose(1, 2)  # [B, C, 1]
        return x * weights


class ALPE(nn.Module):
    def __init__(self, dim: int, kernel_size: int, use_missing_mask: bool = True) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.use_missing_mask = use_missing_mask
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.eca = EfficientChannelAttention1D(channels=dim, kernel_size=3)

    def forward(self, valid_mask: Tensor, seq_len: int, dim: int) -> Tensor:
        # valid_mask: [B, T] with 1 for valid observations, 0 for missing ones.
        batch_size = valid_mask.shape[0]
        pe = build_sinusoidal_positional_encoding(
            seq_len=seq_len,
            dim=dim,
            device=valid_mask.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(batch_size, -1, -1)     # [B, T, D]

        if self.use_missing_mask:
            pe = pe * valid_mask.unsqueeze(-1).to(pe.dtype)

        pe = pe.transpose(1, 2)                        # [B, D, T]
        pe = self.conv(pe)
        pe = self.eca(pe)
        return pe.transpose(1, 2)                      # [B, T, D]


class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, C]
        residual = x.transpose(1, 2)                   # [B, C, T]
        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(residual)
        out = self.relu(out)
        return out.transpose(1, 2)                     # [B, T, C_out]


class TransformerEncoderSubmodule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        use_alpe: bool,
        kernel_size: int,
        use_missing_mask: bool,
        use_absolute_pe: bool,
    ) -> None:
        super().__init__()
        self.input_proj = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        )
        self.use_alpe = use_alpe
        self.use_absolute_pe = use_absolute_pe
        self.alpe = ALPE(
            dim=out_dim,
            kernel_size=kernel_size,
            use_missing_mask=use_missing_mask,
        ) if use_alpe else None
        self.self_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(out_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        x = self.input_proj(x)

        if self.alpe is not None and valid_mask is not None:
            x = x + self.alpe(valid_mask=valid_mask, seq_len=x.size(1), dim=x.size(2))
        elif self.use_absolute_pe:
            pe = build_sinusoidal_positional_encoding(
                seq_len=x.size(1),
                dim=x.size(2),
                device=x.device,
                dtype=x.dtype,
            ).unsqueeze(0)
            x = x + pe

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = valid_mask.eq(0)

        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class CTFusionBlock(nn.Module):
    def __init__(self, config: MCTNetConfig, stage_index: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        use_alpe_in_stage = config.use_alpe and stage_index == 0
        use_absolute_pe_in_stage = (not config.use_alpe) and stage_index == 0
        self.use_cnn_branch = config.use_cnn_branch
        self.use_transformer_branch = config.use_transformer_branch

        self.cnn_branch = (
            ResidualCNNBlock(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=config.kernel_size,
            )
            if self.use_cnn_branch else None
        )
        self.transformer_branch = (
            TransformerEncoderSubmodule(
                in_dim=in_dim,
                out_dim=out_dim,
                n_heads=config.n_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                use_alpe=use_alpe_in_stage,
                kernel_size=config.kernel_size,
                use_missing_mask=config.use_missing_mask,
                use_absolute_pe=use_absolute_pe_in_stage,
            )
            if self.use_transformer_branch else None
        )

        branch_multiplier = int(self.use_cnn_branch) + int(self.use_transformer_branch)
        if branch_multiplier == 0:
            raise ValueError('At least one of CNN or Transformer branches must be enabled.')

        fusion_in_dim = out_dim * branch_multiplier
        self.fusion_proj = (
            nn.Identity() if fusion_in_dim == out_dim else nn.Linear(fusion_in_dim, out_dim)
        )
        self.fusion_norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        outputs: List[Tensor] = []
        if self.cnn_branch is not None:
            outputs.append(self.cnn_branch(x))
        if self.transformer_branch is not None:
            outputs.append(self.transformer_branch(x, valid_mask=valid_mask))

        fused = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        fused = self.fusion_proj(fused)
        return self.fusion_norm(fused)


class SequenceDownsampler(nn.Module):
    def __init__(self, pool_type: str = 'max') -> None:
        super().__init__()
        if pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        else:
            raise ValueError("pool_type must be 'max' or 'avg'.")

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)                          # [B, C, T]
        x = self.pool(x)
        return x.transpose(1, 2)                       # [B, T/2, C]


class ChannelWiseGlobalMaxPool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # Paper wording suggests global max pooling along the channel dimension.
        # x: [B, T, C] -> [B, T]
        return x.max(dim=-1).values


class MCTNet(nn.Module):
    def __init__(self, config: MCTNetConfig) -> None:
        super().__init__()
        self.config = config
        blocks: List[nn.Module] = []
        pools: List[nn.Module] = []

        in_dim = config.input_dim
        for stage_index, out_dim in enumerate(config.stage_dims):
            blocks.append(CTFusionBlock(config, stage_index, in_dim=in_dim, out_dim=out_dim))
            if stage_index < config.n_stages - 1:
                pools.append(SequenceDownsampler(pool_type=config.pool_type))
            in_dim = out_dim

        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList(pools)
        self.final_pool = ChannelWiseGlobalMaxPool()

        final_seq_len = config.seq_len
        for _ in range(config.n_stages - 1):
            final_seq_len //= 2
        self.final_seq_len = final_seq_len

        self.classifier = nn.Sequential(
            nn.Linear(final_seq_len, config.classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

    def extract_pooled_features(
        self,
        x: Tensor,
        valid_mask: Tensor,
        return_stage_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        stage_features: Dict[str, Tensor] = {}
        current_x = x
        current_mask = valid_mask

        for stage_index, block in enumerate(self.blocks):
            current_x = block(current_x, valid_mask=current_mask)
            stage_features[f'stage_{stage_index + 1}'] = current_x

            if stage_index < len(self.pools):
                current_x = self.pools[stage_index](current_x)
                # Implementation choice:
                # use max pooling on the validity mask so a pooled step is considered
                # valid if at least one original observation was valid.
                pooled_mask = F.max_pool1d(
                    current_mask.unsqueeze(1).float(),
                    kernel_size=2,
                    stride=2,
                )
                current_mask = pooled_mask.squeeze(1).to(valid_mask.dtype)

        pooled = self.final_pool(current_x)

        if return_stage_features:
            stage_features['pooled'] = pooled
            return pooled, stage_features
        return pooled

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor,
        return_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        if return_features:
            pooled, stage_features = self.extract_pooled_features(
                x=x,
                valid_mask=valid_mask,
                return_stage_features=True,
            )
            logits = self.classifier(pooled)
            return logits, stage_features

        pooled = self.extract_pooled_features(
            x=x,
            valid_mask=valid_mask,
            return_stage_features=False,
        )
        logits = self.classifier(pooled)
        return logits

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


def build_mctnet(
    num_classes: int,
    input_dim: int = 10,
    seq_len: int = 36,
    n_stages: int = 3,
    n_heads: int = 5,
    kernel_size: int = 3,
    dropout: float = 0.1,
    use_alpe: bool = True,
    use_missing_mask: bool = True,
    use_cnn_branch: bool = True,
    use_transformer_branch: bool = True,
) -> MCTNet:
    config = MCTNetConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        seq_len=seq_len,
        n_stages=n_stages,
        stage_dims=(10, 20, 40)[:n_stages],
        n_heads=n_heads,
        kernel_size=kernel_size,
        dropout=dropout,
        use_alpe=use_alpe,
        use_missing_mask=use_missing_mask,
        use_cnn_branch=use_cnn_branch,
        use_transformer_branch=use_transformer_branch,
    )
    return MCTNet(config)

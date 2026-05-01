from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def next_multiple(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor) * divisor)


def build_sinusoidal_positional_encoding(
    seq_len: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    encoding = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    if dim == 0:
        return encoding

    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    even_indices = torch.arange(0, dim, 2, device=device, dtype=dtype)
    div_term = torch.exp(even_indices * (-math.log(10000.0) / dim))
    phase = position * div_term

    encoding[:, 0::2] = torch.sin(phase)

    odd_width = encoding[:, 1::2].shape[1]
    if odd_width > 0:
        encoding[:, 1::2] = torch.cos(phase[:, :odd_width])

    return encoding


@dataclass
class MCTNetConfig:
    """
    Paper-aligned defaults:
      - n_stages = 3
      - n_heads = 5
      - kernel_size = 3

    Implementation choice:
      stage_dims are auto-expanded from input_dim because the paper only fixes
      the tuned hyperparameters above for the 10-band Sentinel-2 setting.
    """

    input_dim: int = 10
    num_classes: int = 5
    seq_len: int = 36
    n_stages: int = 3
    n_heads: int = 5
    kernel_size: int = 3
    dropout: float = 0.1
    ffn_expansion: int = 4
    pool_type: str = 'max'
    use_alpe: bool = True
    use_missing_mask: bool = True
    use_cnn_branch: bool = True
    use_transformer_branch: bool = True
    stage_dims: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.stage_dims is None:
            stage1_dim = max(10, next_multiple(self.input_dim, self.n_heads))
            dims = [stage1_dim]
            for _ in range(1, self.n_stages):
                dims.append(dims[-1] * 2)
            self.stage_dims = tuple(dims)

        if len(self.stage_dims) != self.n_stages:
            raise ValueError('len(stage_dims) must match n_stages')

        for stage_dim in self.stage_dims:
            if stage_dim % self.n_heads != 0:
                raise ValueError(f'stage_dim={stage_dim} must be divisible by n_heads={self.n_heads}')

        if self.pool_type not in {'max', 'avg'}:
            raise ValueError("pool_type must be 'max' or 'avg'")


class EfficientChannelAttention1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        weights = x.mean(dim=-1, keepdim=True)          # [B, C, 1]
        weights = weights.transpose(1, 2)               # [B, 1, C]
        weights = self.conv(weights)
        weights = torch.sigmoid(weights).transpose(1, 2)
        return x * weights


class ALPE(nn.Module):
    """
    Paper:
      ALPE(t) = ECA(Conv1D(PE(t) * mask))
    """

    def __init__(self, dim: int, kernel_size: int, use_missing_mask: bool = True) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.use_missing_mask = use_missing_mask
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, bias=True)
        self.eca = EfficientChannelAttention1D(channels=dim, kernel_size=3)

    def forward(self, valid_mask: Tensor, seq_len: int, dim: int) -> Tensor:
        batch_size = valid_mask.shape[0]
        positional_encoding = build_sinusoidal_positional_encoding(
            seq_len=seq_len,
            dim=dim,
            device=valid_mask.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(batch_size, -1, -1)      # [B, T, D]

        if self.use_missing_mask:
            positional_encoding = positional_encoding * valid_mask.unsqueeze(-1).to(positional_encoding.dtype)

        positional_encoding = positional_encoding.transpose(1, 2)  # [B, D, T]
        positional_encoding = self.conv(positional_encoding)
        positional_encoding = self.eca(positional_encoding)
        return positional_encoding.transpose(1, 2)                  # [B, T, D]


class ResidualCNNSubmodule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x.transpose(1, 2)
        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(residual)
        out = self.relu(out)
        return out.transpose(1, 2)


class TransformerSubmodule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int,
        dropout: float,
        kernel_size: int,
        use_alpe: bool,
        use_missing_mask: bool,
        use_absolute_pe: bool,
        ffn_expansion: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.alpe = ALPE(out_dim, kernel_size=kernel_size, use_missing_mask=use_missing_mask) if use_alpe else None
        self.use_absolute_pe = use_absolute_pe
        self.self_attention = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        ffn_hidden_dim = out_dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, out_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        x = self.input_projection(x)

        if self.alpe is not None and valid_mask is not None:
            x = x + self.alpe(valid_mask=valid_mask, seq_len=x.size(1), dim=x.size(2))
        elif self.use_absolute_pe:
            x = x + build_sinusoidal_positional_encoding(
                seq_len=x.size(1),
                dim=x.size(2),
                device=x.device,
                dtype=x.dtype,
            ).unsqueeze(0)

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = valid_mask.eq(0)

        attention_out, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attention_out))
        x = self.norm2(x + self.ffn(x))
        return x


class CTFusionBlock(nn.Module):
    """
    Implementation choice:
      the paper text does not specify the exact fusion operator between the CNN
      and Transformer branches, so this code concatenates both outputs and
      projects them back to the stage width.
    """

    def __init__(self, config: MCTNetConfig, stage_index: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.use_cnn_branch = config.use_cnn_branch
        self.use_transformer_branch = config.use_transformer_branch

        use_alpe_in_stage = config.use_alpe and stage_index == 0
        use_absolute_pe_in_stage = (not config.use_alpe) and stage_index == 0

        self.cnn_branch = (
            ResidualCNNSubmodule(in_channels=in_dim, out_channels=out_dim, kernel_size=config.kernel_size)
            if self.use_cnn_branch else None
        )
        self.transformer_branch = (
            TransformerSubmodule(
                in_dim=in_dim,
                out_dim=out_dim,
                n_heads=config.n_heads,
                dropout=config.dropout,
                kernel_size=config.kernel_size,
                use_alpe=use_alpe_in_stage,
                use_missing_mask=config.use_missing_mask,
                use_absolute_pe=use_absolute_pe_in_stage,
                ffn_expansion=config.ffn_expansion,
            )
            if self.use_transformer_branch else None
        )

        branch_count = int(self.use_cnn_branch) + int(self.use_transformer_branch)
        if branch_count == 0:
            raise ValueError('At least one branch must be enabled')

        fusion_input_dim = out_dim * branch_count
        self.fusion_projection = nn.Identity() if fusion_input_dim == out_dim else nn.Linear(fusion_input_dim, out_dim)
        self.fusion_norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        outputs: List[Tensor] = []
        if self.cnn_branch is not None:
            outputs.append(self.cnn_branch(x))
        if self.transformer_branch is not None:
            outputs.append(self.transformer_branch(x, valid_mask=valid_mask))

        fused = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        fused = self.fusion_projection(fused)
        return self.fusion_norm(fused)


class SequencePooling(nn.Module):
    def __init__(self, pool_type: str = 'max') -> None:
        super().__init__()
        if pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        else:
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.pool(x)
        return x.transpose(1, 2)


class ChannelWiseGlobalMaxPool(nn.Module):
    """
    Paper wording:
      the last CTFusion output is globally max pooled along the channel
      dimension to obtain a one-dimensional vector.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.max(dim=-1).values


class MCTNet(nn.Module):
    def __init__(self, config: MCTNetConfig) -> None:
        super().__init__()
        self.config = config

        blocks: List[nn.Module] = []
        pools: List[nn.Module] = []

        in_dim = config.input_dim
        for stage_index, out_dim in enumerate(config.stage_dims):
            blocks.append(CTFusionBlock(config=config, stage_index=stage_index, in_dim=in_dim, out_dim=out_dim))
            if stage_index < config.n_stages - 1:
                pools.append(SequencePooling(pool_type=config.pool_type))
            in_dim = out_dim

        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList(pools)
        self.final_pool = ChannelWiseGlobalMaxPool()

        final_seq_len = config.seq_len
        for _ in range(config.n_stages - 1):
            final_seq_len = final_seq_len // 2
        self.final_seq_len = final_seq_len

        # Paper: a linear layer with Softmax activation.
        # In PyTorch training we return logits and keep Softmax outside.
        self.classifier = nn.Linear(self.final_seq_len, config.num_classes)

    def fuse_modalities(
        self,
        x: Tensor,
        dynamic_env: Optional[Tensor] = None,
        static_env: Optional[Tensor] = None,
    ) -> Tensor:
        parts: List[Tensor] = [x]

        if dynamic_env is not None and dynamic_env.numel() > 0:
            if dynamic_env.dim() != 3:
                raise ValueError('dynamic_env must have shape [B, T, E_dynamic].')
            if dynamic_env.size(1) != x.size(1):
                raise ValueError('dynamic_env sequence length must match Sentinel-2 sequence length.')
            parts.append(dynamic_env)

        if static_env is not None and static_env.numel() > 0:
            if static_env.dim() != 2:
                raise ValueError('static_env must have shape [B, E_static].')
            static_broadcast = static_env.unsqueeze(1).expand(-1, x.size(1), -1)
            parts.append(static_broadcast)

        fused = torch.cat(parts, dim=-1)
        if fused.size(-1) != self.config.input_dim:
            raise ValueError(
                f'Fused input feature dimension mismatch: expected {self.config.input_dim}, got {fused.size(-1)}.'
            )
        return fused

    def extract_pooled_features(
        self,
        x: Tensor,
        valid_mask: Optional[Tensor],
        dynamic_env: Optional[Tensor] = None,
        static_env: Optional[Tensor] = None,
        return_stage_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        stage_features: Dict[str, Tensor] = {}
        current_x = self.fuse_modalities(x=x, dynamic_env=dynamic_env, static_env=static_env)
        current_valid_mask = valid_mask

        for stage_index, block in enumerate(self.blocks):
            current_x = block(current_x, valid_mask=current_valid_mask)
            stage_features[f'stage_{stage_index + 1}'] = current_x

            if stage_index < len(self.pools):
                current_x = self.pools[stage_index](current_x)
                if current_valid_mask is not None:
                    pooled_mask = F.max_pool1d(
                        current_valid_mask.unsqueeze(1).float(),
                        kernel_size=2,
                        stride=2,
                    )
                    current_valid_mask = pooled_mask.squeeze(1).to(current_valid_mask.dtype)

        pooled = self.final_pool(current_x)
        if return_stage_features:
            stage_features['pooled'] = pooled
            return pooled, stage_features
        return pooled

    def forward(
        self,
        x: Tensor,
        valid_mask: Optional[Tensor] = None,
        dynamic_env: Optional[Tensor] = None,
        static_env: Optional[Tensor] = None,
        return_features: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        if return_features:
            pooled, stage_features = self.extract_pooled_features(
                x=x,
                valid_mask=valid_mask,
                dynamic_env=dynamic_env,
                static_env=static_env,
                return_stage_features=True,
            )
            logits = self.classifier(pooled)
            return logits, stage_features

        pooled = self.extract_pooled_features(
            x=x,
            valid_mask=valid_mask,
            dynamic_env=dynamic_env,
            static_env=static_env,
            return_stage_features=False,
        )
        return self.classifier(pooled)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


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
    stage_dims: Optional[Tuple[int, ...]] = None,
) -> MCTNet:
    config = MCTNetConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        seq_len=seq_len,
        n_stages=n_stages,
        n_heads=n_heads,
        kernel_size=kernel_size,
        dropout=dropout,
        use_alpe=use_alpe,
        use_missing_mask=use_missing_mask,
        use_cnn_branch=use_cnn_branch,
        use_transformer_branch=use_transformer_branch,
        stage_dims=stage_dims,
    )
    return MCTNet(config)

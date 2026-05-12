"""
Technical summary
-----------------
This training script operationalizes the paper-derived upgrade path behind the
new enhanced crop classifier.

1. From MCTNet, it preserves masked temporal training with missing-observation
   support and a CNN-Transformer hybrid backbone.
2. From the Geo-CBAM-CNN paper, it trains a stronger attention-enhanced local
   branch rather than relying on plain convolutions.
3. From the red-edge CNN-RNN paper, it uses 14 per-date Sentinel-2 features and
   explicitly trains a hybrid local-global-phenology architecture.
4. From the tree-crop and BKA-CNN fusion papers, it avoids naive fusion by
   keeping dynamic climate covariates in the temporal stream and soil/topography
   in a late static branch.
5. From the supervised-to-unsupervised paper, it adds a masked temporal
   reconstruction objective so the encoder learns from both labels and the raw
   time-series structure.

The intent is to improve beyond the current baseline by combining stronger
feature engineering, heterogeneous fusion, branch attention, and auxiliary
self-supervised regularization in a production-ready PyTorch pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_enhanced import (
    DEFAULT_CONFIGS,
    EnhancedCropDataset,
    EnhancedPreparedSplits,
    prepare_enhanced_splits,
    resolve_dataset_paths,
)
from enhanced_mctnet import EnhancedMCTNet


PART1_BASELINE: Dict[str, Dict[str, float]] = {
    'arkansas': {'OA': 0.968, 'Kappa': 0.951, 'F1': 0.933},
    'california': {'OA': 0.852, 'Kappa': 0.806, 'F1': 0.829},
}


def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds.

    Args:
        seed: Integer seed.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_inputs(x: Tensor, dynamic_env: Tensor) -> Tensor:
    """
    Concatenate spectral-VI features and dynamic covariates.

    Args:
        x: Tensor of shape (B, 36, 14).
        dynamic_env: Tensor of shape (B, 36, E_dyn).

    Returns:
        Tensor of shape (B, 36, 14 + E_dyn).
    """
    if dynamic_env.numel() == 0:
        return x
    return torch.cat([x, dynamic_env], dim=-1)


def apply_temporal_masking(
    x: Tensor,
    valid_mask: Tensor,
    mask_rate: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Randomly mask valid time steps for self-supervised reconstruction.

    Args:
        x: Temporal tensor of shape (B, T, C).
        valid_mask: Validity mask of shape (B, T).
        mask_rate: Fraction of valid time steps to mask.

    Returns:
        Tuple (x_masked, valid_mask_masked, masked_positions) where
        masked_positions has shape (B, T) and marks the dropped valid steps.
    """
    if mask_rate <= 0.0:
        masked_positions = torch.zeros_like(valid_mask, dtype=torch.bool)
        return x, valid_mask, masked_positions

    random_mask = torch.rand_like(valid_mask)
    masked_positions = (random_mask < mask_rate) & valid_mask.gt(0)

    x_masked = x.clone()
    x_masked[masked_positions] = 0.0
    valid_mask_masked = valid_mask.clone()
    valid_mask_masked[masked_positions] = 0.0
    return x_masked, valid_mask_masked, masked_positions


def build_class_weights(
    y_train: np.ndarray,
    num_classes: int,
    device: torch.device,
    power: float = 0.5,
) -> Tensor:
    """
    Build softened inverse-frequency class weights.

    Args:
        y_train: Training labels of shape (N,).
        num_classes: Number of classes.
        device: Target device.
        power: Inverse-frequency exponent.

    Returns:
        Tensor of shape (num_classes,).
    """
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    frequencies = counts / counts.sum()
    weights = np.power(np.clip(frequencies, 1e-12, None), -power)
    weights = weights * (float(num_classes) / weights.sum())
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def build_sample_weights(y_train: np.ndarray, power: float = 0.5) -> np.ndarray:
    """
    Build per-sample weights for a weighted sampler.

    Args:
        y_train: Training labels of shape (N,).
        power: Inverse-frequency exponent.

    Returns:
        Array of shape (N,).
    """
    num_classes = int(np.max(y_train)) + 1 if y_train.size > 0 else 0
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    class_weights = np.power(counts / counts.sum(), -power)
    return class_weights[y_train.astype(np.int64)].astype(np.float64)


def build_dataloaders(
    prepared: EnhancedPreparedSplits,
    batch_size: int,
    use_weighted_sampler: bool,
    sampler_power: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test dataloaders.

    Args:
        prepared: Prepared splits object.
        batch_size: Batch size.
        use_weighted_sampler: Whether to use balanced sampling.
        sampler_power: Inverse-frequency exponent for the sampler.

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    train_dataset = EnhancedCropDataset(
        x=prepared.x_train,
        valid_mask=prepared.valid_mask_train,
        y=prepared.y_train,
        dynamic_env=prepared.dynamic_env_train,
        static_env=prepared.static_env_train,
    )
    val_dataset = EnhancedCropDataset(
        x=prepared.x_val,
        valid_mask=prepared.valid_mask_val,
        y=prepared.y_val,
        dynamic_env=prepared.dynamic_env_val,
        static_env=prepared.static_env_val,
    )
    test_dataset = EnhancedCropDataset(
        x=prepared.x_test,
        valid_mask=prepared.valid_mask_test,
        y=prepared.y_test,
        dynamic_env=prepared.dynamic_env_test,
        static_env=prepared.static_env_test,
    )

    train_loader_kwargs: Dict[str, Any] = {
        'batch_size': batch_size,
        'num_workers': 0,
        'pin_memory': torch.cuda.is_available(),
    }
    if use_weighted_sampler:
        sample_weights = build_sample_weights(prepared.y_train, power=sampler_power)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader_kwargs['sampler'] = sampler
    else:
        train_loader_kwargs['shuffle'] = True

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def evaluate(
    model: EnhancedMCTNet,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Evaluate the model on one split.

    Args:
        model: Enhanced model.
        loader: Validation or test DataLoader.
        device: Target device.
        class_names: Ordered class names.

    Returns:
        Dict with OA, Kappa, F1_macro, report, and confusion matrix.
    """
    model.eval()
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device, non_blocking=True)
            valid_mask = batch['valid_mask'].to(device, non_blocking=True)
            dynamic_env = batch['dynamic_env'].to(device, non_blocking=True)
            static_env = batch['static_env'].to(device, non_blocking=True)

            model_input = build_model_inputs(x=x, dynamic_env=dynamic_env)
            outputs = model(
                x=model_input,
                valid_mask=valid_mask,
                static_env=static_env if static_env.shape[-1] > 0 else None,
                return_aux=False,
            )
            preds = outputs['logits'].argmax(dim=1).cpu().numpy()
            y_true_parts.append(batch['y'].numpy())
            y_pred_parts.append(preds)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    oa = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {'OA': oa, 'Kappa': kappa, 'F1_macro': f1_macro, 'report': report, 'cm': cm}


def train_one_epoch(
    model: EnhancedMCTNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    classification_criterion: nn.Module,
    reconstruction_criterion: nn.Module,
    device: torch.device,
    reconstruction_weight: float,
    temporal_mask_rate: float,
    gradient_clip_norm: float,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: Enhanced model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        classification_criterion: Classification loss.
        reconstruction_criterion: Reconstruction loss.
        device: Target device.
        reconstruction_weight: Auxiliary-loss weight.
        temporal_mask_rate: Fraction of valid time steps to mask.
        gradient_clip_norm: Maximum gradient norm.

    Returns:
        Dict with averaged train_loss, cls_loss, and recon_loss.
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recon_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        valid_mask = batch['valid_mask'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        dynamic_env = batch['dynamic_env'].to(device, non_blocking=True)
        static_env = batch['static_env'].to(device, non_blocking=True)

        model_input = build_model_inputs(x=x, dynamic_env=dynamic_env)
        original_target = model_input.detach()
        masked_input, masked_valid_mask, masked_positions = apply_temporal_masking(
            x=model_input,
            valid_mask=valid_mask,
            mask_rate=temporal_mask_rate,
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            x=masked_input,
            valid_mask=masked_valid_mask,
            static_env=static_env if static_env.shape[-1] > 0 else None,
            return_aux=False,
        )
        logits = outputs['logits']
        reconstruction = outputs['reconstruction']

        cls_loss = classification_criterion(logits, y)

        if reconstruction_weight > 0.0 and masked_positions.any():
            recon_mask = masked_positions.unsqueeze(-1).expand_as(reconstruction)
            recon_values = reconstruction_criterion(reconstruction, original_target)
            recon_loss = recon_values[recon_mask].mean()
        else:
            recon_loss = logits.new_zeros(())

        loss = cls_loss + reconstruction_weight * recon_loss
        loss.backward()
        if gradient_clip_norm > 0.0:
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_cls_loss += float(cls_loss.item()) * batch_size
        total_recon_loss += float(recon_loss.item()) * batch_size
        total_count += batch_size

    return {
        'loss': total_loss / max(total_count, 1),
        'cls_loss': total_cls_loss / max(total_count, 1),
        'recon_loss': total_recon_loss / max(total_count, 1),
    }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build a warmup + cosine scheduler.

    Args:
        optimizer: Optimizer.
        epochs: Number of epochs.

    Returns:
        Scheduler instance stepped once per epoch.
    """
    warmup_epochs = min(10, max(2, epochs // 10))
    if warmup_epochs >= epochs:
        warmup_epochs = max(1, epochs - 1)
    warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-5)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], title: str, output_path: Path) -> None:
    """
    Plot and save a normalized confusion matrix.

    Args:
        cm: Confusion matrix of shape (C, C).
        class_names: Ordered class names.
        title: Figure title.
        output_path: PNG destination.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = cm.astype(np.float64)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(normalized, row_sums, out=np.zeros_like(normalized), where=row_sums != 0)

    fig, axis = plt.subplots(figsize=(max(6, len(class_names) * 1.25), max(5, len(class_names) * 1.0)))
    image = axis.imshow(normalized, cmap='Blues', vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha='right')
    axis.set_yticklabels(class_names)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('True')
    axis.set_title(title)

    for row_idx in range(normalized.shape[0]):
        for col_idx in range(normalized.shape[1]):
            axis.text(
                col_idx,
                row_idx,
                f'{normalized[row_idx, col_idx]:.2f}',
                ha='center',
                va='center',
                color='white' if normalized[row_idx, col_idx] > 0.5 else 'black',
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_learning_curves(
    train_losses: Sequence[float],
    val_f1s: Sequence[float],
    config_name: str,
    output_path: Path,
) -> None:
    """
    Plot and save learning curves.

    Args:
        train_losses: Per-epoch train losses.
        val_f1s: Per-epoch validation macro-F1 scores.
        config_name: Experiment name.
        output_path: PNG destination.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, color='#2563eb')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train loss')
    axes[0].set_title(f'{config_name} - Train loss')
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_f1s, color='#dc2626')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation F1 macro')
    axes[1].set_title(f'{config_name} - Validation F1')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_ablation_comparison(summary_csv: Path, output_dir: Path) -> None:
    """
    Plot grouped OA/Kappa/F1 comparison against the original MCTNet baseline.

    Args:
        summary_csv: Summary CSV path.
        output_dir: Output directory.

    Returns:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_csv)

    comparison_rows: List[Dict[str, Any]] = []
    for state, metrics in PART1_BASELINE.items():
        comparison_rows.append(
            {
                'state': state,
                'config': 'mctnet_original',
                'OA': float(metrics['OA']),
                'Kappa': float(metrics['Kappa']),
                'F1_macro': float(metrics['F1']),
            }
        )

    for _, row in summary.iterrows():
        comparison_rows.append(
            {
                'state': str(row['state']),
                'config': str(row['config']),
                'OA': float(row['OA']),
                'Kappa': float(row['Kappa']),
                'F1_macro': float(row['F1_macro']),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    config_order = [config for config in ['mctnet_original'] + DEFAULT_CONFIGS if config in comparison_df['config'].unique()]
    state_order = [state for state in ['arkansas', 'california'] if state in comparison_df['state'].unique()]
    metrics_order = ['OA', 'Kappa', 'F1_macro']
    colors = ['#2563eb', '#dc2626', '#16a34a', '#d97706']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(config_order))
    width = 0.8 / max(len(state_order), 1)

    for axis, metric_name in zip(axes, metrics_order):
        for state_idx, state in enumerate(state_order):
            state_values: List[float] = []
            for config in config_order:
                subset = comparison_df[(comparison_df['state'] == state) & (comparison_df['config'] == config)]
                state_values.append(float(subset.iloc[0][metric_name]) if not subset.empty else np.nan)
            offset = (state_idx - (len(state_order) - 1) / 2) * width
            axis.bar(
                x + offset,
                state_values,
                width=width,
                color=colors[state_idx % len(colors)],
                label=state.title(),
            )
        axis.set_xticks(x)
        axis.set_xticklabels(config_order, rotation=20, ha='right')
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel(metric_name)
        axis.set_title(metric_name)
        axis.grid(axis='y', alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=max(len(state_order), 1))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / 'enhanced_comparison_barplot.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def run_single_experiment(
    npz_path: Path,
    json_path: Path,
    output_dir: Path,
    config_name: str,
    args: Any,
) -> Dict[str, Any]:
    """
    Run one enhanced experiment.

    Args:
        npz_path: Bundle path.
        json_path: Metadata path.
        output_dir: Output directory.
        config_name: Ablation config name.
        args: Namespace-like configuration.

    Returns:
        Result dict with test metrics and artifact paths.
    """
    prepared = prepare_enhanced_splits(
        npz_path=npz_path,
        json_path=json_path,
        config_name=config_name,
        use_env_covariates=bool(getattr(args, 'use_env_covariates', True)),
    )

    state_dir = output_dir / prepared.state
    state_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        prepared=prepared,
        batch_size=int(args.batch_size),
        use_weighted_sampler=bool(getattr(args, 'use_weighted_sampler', True)),
        sampler_power=float(getattr(args, 'sampler_power', getattr(args, 'class_weight_power', 0.5))),
    )

    device = torch.device('cpu' if bool(getattr(args, 'cpu', False)) or not torch.cuda.is_available() else 'cuda')
    model = EnhancedMCTNet(
        temporal_input_dim=prepared.temporal_input_dim,
        static_input_dim=prepared.static_input_dim,
        num_classes=len(prepared.class_names),
        seq_len=int(prepared.x_train.shape[1]),
        model_dim=int(getattr(args, 'model_dim', 96)),
        num_heads=int(getattr(args, 'num_heads', 6)),
        num_transformer_layers=int(getattr(args, 'num_transformer_layers', 2)),
        gru_hidden_dim=int(getattr(args, 'gru_hidden_dim', 64)),
        static_hidden_dim=int(getattr(args, 'static_hidden_dim', 32)),
        dropout=float(args.dropout),
        ffn_expansion=int(getattr(args, 'ffn_expansion', 4)),
    ).to(device)

    class_weights = build_class_weights(
        y_train=prepared.y_train,
        num_classes=len(prepared.class_names),
        device=device,
        power=float(getattr(args, 'class_weight_power', 0.5)),
    )
    classification_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(getattr(args, 'label_smoothing', 0.05)),
    )
    reconstruction_criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = AdamW(
        model.parameters(),
        lr=float(getattr(args, 'learning_rate', getattr(args, 'lr', 1e-3))),
        weight_decay=float(args.weight_decay),
    )
    scheduler = build_scheduler(optimizer=optimizer, epochs=int(args.epochs))

    checkpoint_path = state_dir / f'best_enhanced_mctnet_{config_name}.pt'
    history_path = state_dir / f'history_enhanced_{config_name}.json'
    confusion_png = state_dir / f'confusion_matrix_enhanced_{config_name}.png'
    curves_png = state_dir / f'learning_curves_enhanced_{config_name}.png'

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    train_losses: List[float] = []
    val_f1s: List[float] = []
    history_rows: List[Dict[str, float]] = []

    print(
        f'[{prepared.state}][{config_name}] '
        f'temporal_input_dim={prepared.temporal_input_dim} '
        f'static_input_dim={prepared.static_input_dim} '
        f'num_classes={len(prepared.class_names)}'
    )

    for epoch_idx in range(1, int(args.epochs) + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            classification_criterion=classification_criterion,
            reconstruction_criterion=reconstruction_criterion,
            device=device,
            reconstruction_weight=float(getattr(args, 'reconstruction_weight', 0.2)),
            temporal_mask_rate=float(getattr(args, 'temporal_mask_rate', 0.15)),
            gradient_clip_norm=float(getattr(args, 'gradient_clip_norm', 1.0)),
        )
        scheduler.step()
        val_metrics = evaluate(model=model, loader=val_loader, device=device, class_names=prepared.class_names)

        train_losses.append(float(train_stats['loss']))
        val_f1s.append(float(val_metrics['F1_macro']))
        history_rows.append(
            {
                'epoch': epoch_idx,
                'train_loss': float(train_stats['loss']),
                'train_cls_loss': float(train_stats['cls_loss']),
                'train_recon_loss': float(train_stats['recon_loss']),
                'val_OA': float(val_metrics['OA']),
                'val_Kappa': float(val_metrics['Kappa']),
                'val_F1_macro': float(val_metrics['F1_macro']),
                'lr': float(optimizer.param_groups[0]['lr']),
            }
        )

        if float(val_metrics['F1_macro']) > best_val_f1:
            best_val_f1 = float(val_metrics['F1_macro'])
            best_epoch = epoch_idx
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if epoch_idx % 10 == 0 or epoch_idx == int(args.epochs):
            print(
                f'  epoch {epoch_idx:03d}/{int(args.epochs)} | '
                f'loss={float(train_stats["loss"]):.4f} | '
                f'cls={float(train_stats["cls_loss"]):.4f} | '
                f'recon={float(train_stats["recon_loss"]):.4f} | '
                f'val_OA={float(val_metrics["OA"]):.4f} | '
                f'val_Kappa={float(val_metrics["Kappa"]):.4f} | '
                f'val_F1={float(val_metrics["F1_macro"]):.4f}'
            )

        if patience_counter >= int(args.early_stopping_patience):
            print(f'  early stopping at epoch {epoch_idx}')
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    val_metrics_best = evaluate(model=model, loader=val_loader, device=device, class_names=prepared.class_names)
    test_metrics = evaluate(model=model, loader=test_loader, device=device, class_names=prepared.class_names)

    plot_confusion_matrix(
        cm=np.asarray(test_metrics['cm'], dtype=np.int64),
        class_names=prepared.class_names,
        title=f'{prepared.state} - enhanced - {config_name}',
        output_path=confusion_png,
    )
    plot_learning_curves(
        train_losses=train_losses,
        val_f1s=val_f1s,
        config_name=f'{prepared.state}-enhanced-{config_name}',
        output_path=curves_png,
    )

    history_payload = {
        'state': prepared.state,
        'config': config_name,
        'temporal_input_dim': prepared.temporal_input_dim,
        'static_input_dim': prepared.static_input_dim,
        'dynamic_feature_names': prepared.dynamic_feature_names,
        'static_feature_names': prepared.static_feature_names,
        'best_epoch': best_epoch,
        'best_val_F1_macro': float(best_val_f1),
        'vi_mean': prepared.vi_mean.astype(float).tolist(),
        'vi_std': prepared.vi_std.astype(float).tolist(),
        'epochs': history_rows,
        'final_validation_report': str(val_metrics_best['report']),
        'final_test_report': str(test_metrics['report']),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding='utf-8')

    result = {
        'state': prepared.state,
        'config': config_name,
        'OA': float(test_metrics['OA']),
        'Kappa': float(test_metrics['Kappa']),
        'F1_macro': float(test_metrics['F1_macro']),
        'best_epoch': int(best_epoch),
        'temporal_input_dim': int(prepared.temporal_input_dim),
        'static_input_dim': int(prepared.static_input_dim),
        'checkpoint_path': str(checkpoint_path),
        'history_path': str(history_path),
        'confusion_matrix_png': str(confusion_png),
        'learning_curves_png': str(curves_png),
    }
    print(
        f'[{prepared.state}][{config_name}] '
        f'OA={result["OA"]:.4f} | '
        f'Kappa={result["Kappa"]:.4f} | '
        f'F1={result["F1_macro"]:.4f} | '
        f'best_epoch={best_epoch}'
    )
    return result


def run_full_ablation(
    processed_env_dir: Path,
    output_dir: Path,
    states: List[str],
    args: Any,
) -> List[Dict[str, Any]]:
    """
    Run the full enhanced ablation study.

    Args:
        processed_env_dir: Directory containing .npz/.json bundles.
        output_dir: Output directory.
        states: State slugs.
        args: Namespace-like configuration object.

    Returns:
        List of per-run result dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    configs = list(getattr(args, 'configs', DEFAULT_CONFIGS))

    for state in states:
        npz_path, json_path = resolve_dataset_paths(processed_env_dir=processed_env_dir, state=state)
        for config_name in configs:
            result = run_single_experiment(
                npz_path=npz_path,
                json_path=json_path,
                output_dir=output_dir,
                config_name=config_name,
                args=args,
            )
            results.append(result)

    summary_csv = output_dir / 'enhanced_ablation_summary.csv'
    fieldnames = [
        'state',
        'config',
        'OA',
        'Kappa',
        'F1_macro',
        'best_epoch',
        'temporal_input_dim',
        'static_input_dim',
        'checkpoint_path',
        'history_path',
        'confusion_matrix_png',
        'learning_curves_png',
    ]
    with summary_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return results


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description='Train the enhanced crop-mapping model.')
    parser.add_argument('--processed-env-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--states', nargs='+', default=['arkansas', 'california'])
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model-dim', type=int, default=96)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--num-transformer-layers', type=int, default=2)
    parser.add_argument('--gru-hidden-dim', type=int, default=64)
    parser.add_argument('--static-hidden-dim', type=int, default=32)
    parser.add_argument('--ffn-expansion', type=int, default=4)
    parser.add_argument('--temporal-mask-rate', type=float, default=0.15)
    parser.add_argument('--reconstruction-weight', type=float, default=0.2)
    parser.add_argument('--class-weight-power', type=float, default=0.5)
    parser.add_argument('--sampler-power', type=float, default=0.5)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0)
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--use-env-covariates', dest='use_env_covariates', action='store_true')
    parser.add_argument('--disable-env-covariates', dest='use_env_covariates', action='store_false')
    parser.add_argument('--use-weighted-sampler', dest='use_weighted_sampler', action='store_true')
    parser.add_argument('--disable-weighted-sampler', dest='use_weighted_sampler', action='store_false')
    parser.set_defaults(use_env_covariates=True, use_weighted_sampler=True)
    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    results = run_full_ablation(
        processed_env_dir=Path(args.processed_env_dir),
        output_dir=output_dir,
        states=list(args.states),
        args=args,
    )
    plot_ablation_comparison(
        summary_csv=output_dir / 'enhanced_ablation_summary.csv',
        output_dir=output_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

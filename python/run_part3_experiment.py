from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from mctnet_gru_plus import MCTNetGRUPlus, build_class_weights, compute_vi_timeseries, normalize_vi_columns


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


class CropDatasetVI(Dataset):
    """
    Dataset for Part 3 with VI-enriched S2 inputs and optional environmental early fusion.

    Args:
        x: Tensor of shape (N, 36, 14).
        mask: Tensor of shape (N, 36).
        y: Tensor of shape (N,).
        env: Optional tensor of shape (N, 36, 8).
        config_indices: Optional list of selected environmental indices.

    Returns:
        __getitem__ returns (x_fused, mask, y) with x_fused of shape (36, 14+E).
    """

    def __init__(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        env: Optional[np.ndarray] = None,
        config_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.x = torch.from_numpy(x).float()
        self.mask = torch.from_numpy(mask).float()
        self.y = torch.from_numpy(y).long()
        self.env = None if env is None else torch.from_numpy(env).float()
        self.config_indices = list(config_indices or [])

    def __len__(self) -> int:
        """
        Get dataset length.

        Args:
            None.

        Returns:
            Number of samples.
        """
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get one fused sample.

        Args:
            index: Sample index.

        Returns:
            Tuple (x_fused, mask, y) with shapes (36, 14+E), (36,), ().
        """
        x_item = self.x[index]
        if self.env is not None and self.config_indices:
            env_item = self.env[index][:, self.config_indices]
            x_item = torch.cat([x_item, env_item], dim=-1)
        return x_item, self.mask[index], self.y[index]


def train_one_epoch_part3(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one training epoch.

    Args:
        model: Part 3 model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Target device.

    Returns:
        Mean epoch loss as float.
    """
    model.train()
    total_loss = 0.0
    total_count = 0

    for x_batch, mask_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        mask_batch = mask_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch, mask_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = int(x_batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate_part3(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Evaluate a model.

    Args:
        model: Part 3 model.
        loader: Validation or test DataLoader.
        device: Target device.
        class_names: Ordered class names.

    Returns:
        Dict with OA, Kappa, F1_macro, report, cm.
    """
    model.eval()
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    with torch.no_grad():
        for x_batch, mask_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            logits = model(x_batch, mask_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true_parts.append(y_batch.numpy())
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


def _resolve_class_names(metadata: Dict[str, Any]) -> List[str]:
    """
    Resolve ordered class names from metadata.

    Args:
        metadata: Metadata dict containing class_name_to_index.

    Returns:
        List of class names of length num_classes.
    """
    mapping = metadata['class_name_to_index']
    class_names: List[Optional[str]] = [None] * int(metadata['num_classes'])
    for class_name, class_idx in mapping.items():
        class_names[int(class_idx)] = class_name
    return [class_name if class_name is not None else f'class_{idx}' for idx, class_name in enumerate(class_names)]


def _extract_config_indices(metadata: Dict[str, Any], config_name: str, env_dim: int) -> List[int]:
    """
    Resolve environmental indices for one ablation config.

    Args:
        metadata: Metadata dict containing ablation_configs.
        config_name: Ablation config name.
        env_dim: Environmental feature dimension.

    Returns:
        List of selected indices inside the temporal environmental tensor.
    """
    specs = metadata.get('ablation_configs', {})
    if config_name not in specs:
        return []
    spec = specs[config_name]
    if isinstance(spec, list):
        return [int(index) for index in spec if 0 <= int(index) < env_dim]
    if isinstance(spec, dict):
        dynamic_indices = [int(index) for index in spec.get('dynamic_indices', [])]
        static_indices = [int(index) for index in spec.get('static_indices', [])]
        dynamic_count = len(metadata.get('environmental_covariates', {}).get('dynamic_columns', []))
        static_offset = dynamic_count if dynamic_count > 0 else max(env_dim - len(static_indices), 0)
        combined = dynamic_indices + [static_offset + index for index in static_indices]
        return [index for index in combined if 0 <= index < env_dim]
    return []


def _broadcast_static_env(static_env: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Broadcast static covariates over time.

    Args:
        static_env: Static tensor of shape (N, E_static).
        seq_len: Temporal length.

    Returns:
        Temporal tensor of shape (N, seq_len, E_static).
    """
    return np.repeat(static_env[:, None, :], seq_len, axis=1).astype(np.float32)


def _load_env_splits(
    bundle: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    seq_len: int,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, List[int]]]:
    """
    Load temporal environmental tensors from a bundle.

    Args:
        bundle: NPZ content loaded into memory.
        metadata: Metadata dict.
        seq_len: Temporal length.

    Returns:
        Tuple (env_splits, config_map) where env_splits maps split names to
        tensors of shape (N, 36, E) or None if unavailable.
    """
    if all(f'env_{split_name}' in bundle for split_name in ['train', 'val', 'test']):
        env_splits = {
            split_name: np.asarray(bundle[f'env_{split_name}'], dtype=np.float32)
            for split_name in ['train', 'val', 'test']
        }
        env_dim = int(env_splits['train'].shape[-1])
        config_map = {
            config_name: _extract_config_indices(metadata, config_name, env_dim)
            for config_name in metadata.get('ablation_configs', {})
        }
        return env_splits, config_map

    if all(f'dynamic_env_{split_name}' in bundle for split_name in ['train', 'val', 'test']) and all(
        f'static_env_{split_name}' in bundle for split_name in ['train', 'val', 'test']
    ):
        env_splits = {}
        for split_name in ['train', 'val', 'test']:
            dynamic_env = np.asarray(bundle[f'dynamic_env_{split_name}'], dtype=np.float32)
            static_env = np.asarray(bundle[f'static_env_{split_name}'], dtype=np.float32)
            static_broadcast = _broadcast_static_env(static_env, seq_len)
            env_splits[split_name] = np.concatenate([dynamic_env, static_broadcast], axis=-1).astype(np.float32)
        env_dim = int(env_splits['train'].shape[-1])
        config_map = {
            config_name: _extract_config_indices(metadata, config_name, env_dim)
            for config_name in metadata.get('ablation_configs', {})
        }
        return env_splits, config_map

    return None, {}


def _resolve_dataset_paths(processed_env_dir: Path, state: str) -> Tuple[Path, Path]:
    """
    Resolve dataset paths for one state.

    Args:
        processed_env_dir: Directory containing processed bundles.
        state: State slug.

    Returns:
        Tuple (npz_path, json_path).
    """
    env_npz = processed_env_dir / f'{state}_mctnet_env_dataset.npz'
    env_json = processed_env_dir / f'{state}_mctnet_env_dataset.json'
    base_npz = processed_env_dir / f'{state}_mctnet_dataset.npz'
    base_json = processed_env_dir / f'{state}_mctnet_dataset.json'
    if env_npz.exists() and env_json.exists():
        return env_npz, env_json
    if base_npz.exists() and base_json.exists():
        return base_npz, base_json
    raise FileNotFoundError(f'Dataset files not found for state={state} in {processed_env_dir}')


def _build_loaders(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    mask_train: np.ndarray,
    mask_val: np.ndarray,
    mask_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    env_splits: Optional[Dict[str, np.ndarray]],
    config_indices: Sequence[int],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation and test dataloaders.

    Args:
        x_train: Train tensor of shape (N_train, 36, 14).
        x_val: Validation tensor of shape (N_val, 36, 14).
        x_test: Test tensor of shape (N_test, 36, 14).
        mask_train: Train mask tensor of shape (N_train, 36).
        mask_val: Validation mask tensor of shape (N_val, 36).
        mask_test: Test mask tensor of shape (N_test, 36).
        y_train: Train labels of shape (N_train,).
        y_val: Validation labels of shape (N_val,).
        y_test: Test labels of shape (N_test,).
        env_splits: Optional environmental tensors by split.
        config_indices: Selected environmental indices.
        batch_size: Batch size.

    Returns:
        Tuple of DataLoaders (train_loader, val_loader, test_loader).
    """
    train_dataset = CropDatasetVI(
        x=x_train,
        mask=mask_train,
        y=y_train,
        env=None if env_splits is None else env_splits['train'],
        config_indices=config_indices,
    )
    val_dataset = CropDatasetVI(
        x=x_val,
        mask=mask_val,
        y=y_val,
        env=None if env_splits is None else env_splits['val'],
        config_indices=config_indices,
    )
    test_dataset = CropDatasetVI(
        x=x_test,
        mask=mask_test,
        y=y_test,
        env=None if env_splits is None else env_splits['test'],
        config_indices=config_indices,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
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


def run_part3_single(
    npz_path: Path,
    json_path: Path,
    output_dir: Path,
    config_name: str = 'baseline',
    use_env_covariates: bool = False,
    args: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run one Part 3 experiment.

    Args:
        npz_path: Input bundle path.
        json_path: Metadata path.
        output_dir: Root output directory.
        config_name: Ablation config name.
        use_env_covariates: Whether to fuse environmental covariates.
        args: Namespace-like object with training hyperparameters.

    Returns:
        Dict with state, config, OA, Kappa, F1_macro and experiment metadata.
    """
    if args is None:
        args = SimpleNamespace(
            epochs=200,
            batch_size=32,
            learning_rate=1e-3,
            weight_decay=1e-4,
            dropout=0.1,
            early_stopping_patience=20,
            seed=2021,
            cpu=False,
        )

    with np.load(npz_path, allow_pickle=True) as data:
        bundle = {key: data[key] for key in data.files}
    metadata = json.loads(json_path.read_text(encoding='utf-8'))

    state = str(metadata.get('state_slug', metadata.get('state_name', npz_path.stem))).lower().replace(' ', '_')
    state_dir = output_dir / state
    state_dir.mkdir(parents=True, exist_ok=True)

    x_train_raw = np.asarray(bundle['x_train'], dtype=np.float32)
    x_val_raw = np.asarray(bundle['x_val'], dtype=np.float32)
    x_test_raw = np.asarray(bundle['x_test'], dtype=np.float32)
    mask_train = np.asarray(bundle['valid_mask_train'], dtype=np.float32)
    mask_val = np.asarray(bundle['valid_mask_val'], dtype=np.float32)
    mask_test = np.asarray(bundle['valid_mask_test'], dtype=np.float32)
    y_train = np.asarray(bundle['y_train'], dtype=np.int64)
    y_val = np.asarray(bundle['y_val'], dtype=np.int64)
    y_test = np.asarray(bundle['y_test'], dtype=np.int64)

    x_train_vi = compute_vi_timeseries(x_train_raw)
    x_val_vi = compute_vi_timeseries(x_val_raw)
    x_test_vi = compute_vi_timeseries(x_test_raw)
    x_train_vi, x_val_vi, x_test_vi, vi_mean, vi_std = normalize_vi_columns(x_train_vi, x_val_vi, x_test_vi)

    env_splits, config_map = _load_env_splits(bundle, metadata, seq_len=int(x_train_vi.shape[1]))
    if use_env_covariates and config_name != 'baseline':
        config_indices = config_map.get(config_name, [])
        env_active = env_splits
    else:
        config_indices = []
        env_active = None

    train_loader, val_loader, test_loader = _build_loaders(
        x_train=x_train_vi,
        x_val=x_val_vi,
        x_test=x_test_vi,
        mask_train=mask_train,
        mask_val=mask_val,
        mask_test=mask_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        env_splits=env_active,
        config_indices=config_indices,
        batch_size=int(args.batch_size),
    )

    input_dim = int(x_train_vi.shape[-1] + len(config_indices))
    num_classes = int(metadata['num_classes'])
    device = torch.device('cpu' if bool(getattr(args, 'cpu', False)) or not torch.cuda.is_available() else 'cuda')
    class_names = _resolve_class_names(metadata)

    model = MCTNetGRUPlus(
        input_dim=input_dim,
        num_classes=num_classes,
        seq_len=int(x_train_vi.shape[1]),
        dropout=float(args.dropout),
    ).to(device)
    class_weights = build_class_weights(y_train=y_train, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=float(getattr(args, 'learning_rate', getattr(args, 'lr', 1e-3))), weight_decay=float(args.weight_decay))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(args.epochs))

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    train_losses: List[float] = []
    val_f1s: List[float] = []
    history_rows: List[Dict[str, float]] = []

    checkpoint_path = state_dir / f'best_mctnet_gruplus_{config_name}.pt'
    history_path = state_dir / f'history_{config_name}.json'
    confusion_png = state_dir / f'confusion_matrix_{config_name}.png'
    curves_png = state_dir / f'learning_curves_{config_name}.png'

    print(f'[{state}][{config_name}] input_dim={input_dim} num_classes={num_classes}')

    for epoch_idx in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch_part3(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_metrics = evaluate_part3(model, val_loader, device, class_names)

        train_losses.append(train_loss)
        val_f1s.append(float(val_metrics['F1_macro']))
        history_rows.append(
            {
                'epoch': epoch_idx,
                'train_loss': float(train_loss),
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
                f'loss={train_loss:.4f} | '
                f'val_OA={float(val_metrics["OA"]):.4f} | '
                f'val_Kappa={float(val_metrics["Kappa"]):.4f} | '
                f'val_F1={float(val_metrics["F1_macro"]):.4f}'
            )

        if patience_counter >= int(args.early_stopping_patience):
            print(f'  early stopping at epoch {epoch_idx}')
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate_part3(model, test_loader, device, class_names)
    val_metrics_best = evaluate_part3(model, val_loader, device, class_names)

    plot_confusion_matrix(
        cm=np.asarray(test_metrics['cm'], dtype=np.int64),
        class_names=class_names,
        title=f'{state} - {config_name}',
        output_path=confusion_png,
    )
    plot_learning_curves(
        train_losses=train_losses,
        val_f1s=val_f1s,
        config_name=f'{state}-{config_name}',
        output_path=curves_png,
    )

    history_payload = {
        'state': state,
        'config': config_name,
        'input_dim': input_dim,
        'env_indices': list(config_indices),
        'best_epoch': best_epoch,
        'best_val_F1_macro': float(best_val_f1),
        'vi_mean': vi_mean.astype(float).tolist(),
        'vi_std': vi_std.astype(float).tolist(),
        'epochs': history_rows,
        'final_validation_report': str(val_metrics_best['report']),
        'final_test_report': str(test_metrics['report']),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding='utf-8')

    result = {
        'state': state,
        'config': config_name,
        'OA': float(test_metrics['OA']),
        'Kappa': float(test_metrics['Kappa']),
        'F1_macro': float(test_metrics['F1_macro']),
        'best_epoch': int(best_epoch),
        'input_dim': int(input_dim),
        'checkpoint_path': str(checkpoint_path),
        'history_path': str(history_path),
        'confusion_matrix_png': str(confusion_png),
        'learning_curves_png': str(curves_png),
    }
    print(
        f'[{state}][{config_name}] '
        f'OA={result["OA"]:.4f} | '
        f'Kappa={result["Kappa"]:.4f} | '
        f'F1={result["F1_macro"]:.4f} | '
        f'best_epoch={best_epoch}'
    )
    return result


def run_part3_full_ablation(
    processed_env_dir: Path,
    output_dir: Path,
    states: List[str],
    args: Any,
) -> List[Dict[str, Any]]:
    """
    Run the full Part 3 ablation.

    Args:
        processed_env_dir: Directory containing .npz/.json bundles.
        output_dir: Output directory.
        states: State list.
        args: Namespace-like object with configs and hyperparameters.

    Returns:
        List of result dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    configs = list(getattr(args, 'configs', ['baseline', 'climate', 'soil', 'topography', 'all']))

    for state in states:
        npz_path, json_path = _resolve_dataset_paths(processed_env_dir, state)
        for config_name in configs:
            result = run_part3_single(
                npz_path=npz_path,
                json_path=json_path,
                output_dir=output_dir,
                config_name=config_name,
                use_env_covariates=bool(getattr(args, 'use_env_covariates', False)),
                args=args,
            )
            results.append(result)

    summary_csv = output_dir / 'part3_ablation_summary.csv'
    fieldnames = [
        'state',
        'config',
        'OA',
        'Kappa',
        'F1_macro',
        'best_epoch',
        'input_dim',
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
    config_order = [config for config in ['mctnet_original', 'baseline', 'climate', 'soil', 'topography', 'all'] if config in comparison_df['config'].unique()]
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
    fig.savefig(output_dir / 'part3_comparison_barplot.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """
    Command-line entry point.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description='Run Part 3 MCTNet-GRU+ experiments.')
    parser.add_argument('--processed-env-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--states', nargs='+', default=['arkansas', 'california'])
    parser.add_argument('--configs', nargs='+', default=['baseline', 'climate', 'soil', 'topography', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--early-stopping-patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--use-env-covariates', action='store_true')
    args = parser.parse_args()

    set_seed(int(args.seed))
    results = run_part3_full_ablation(
        processed_env_dir=Path(args.processed_env_dir),
        output_dir=Path(args.output_dir),
        states=list(args.states),
        args=args,
    )
    plot_ablation_comparison(
        summary_csv=Path(args.output_dir) / 'part3_ablation_summary.csv',
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

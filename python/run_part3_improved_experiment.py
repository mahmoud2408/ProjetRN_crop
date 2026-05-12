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
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from mctnet_part3_improved import (
    MCTNetPart3Improved,
    build_class_weights,
    compute_vi_timeseries_improved,
    normalize_vi_columns_improved,
)


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


class CropDatasetImproved(Dataset):
    """
    Dataset for the improved Part 3 pipeline.

    Args:
        x: Tensor of shape (N, 36, 14).
        mask: Tensor of shape (N, 36).
        y: Tensor of shape (N,).
        dynamic_env: Optional tensor of shape (N, 36, E_dynamic).
        static_env: Optional tensor of shape (N, E_static).

    Returns:
        __getitem__ returns (x, mask, y, dynamic_env, static_env).
    """

    def __init__(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        dynamic_env: Optional[np.ndarray] = None,
        static_env: Optional[np.ndarray] = None,
    ) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32)).float()
        self.mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).float()
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()
        if dynamic_env is None:
            dynamic_env = np.zeros((len(x), x.shape[1], 0), dtype=np.float32)
        if static_env is None:
            static_env = np.zeros((len(x), 0), dtype=np.float32)
        self.dynamic_env = torch.from_numpy(np.asarray(dynamic_env, dtype=np.float32)).float()
        self.static_env = torch.from_numpy(np.asarray(static_env, dtype=np.float32)).float()

    def __len__(self) -> int:
        """
        Get dataset length.

        Args:
            None.

        Returns:
            Number of samples.
        """
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Get one sample.

        Args:
            index: Sample index.

        Returns:
            Tuple (x, mask, y, dynamic_env, static_env).
        """
        return (
            self.x[index],
            self.mask[index],
            self.y[index],
            self.dynamic_env[index],
            self.static_env[index],
        )


def train_one_epoch_improved(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
) -> float:
    """
    Run one training epoch.

    Args:
        model: Improved model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Target device.
        gradient_clip_norm: Maximum gradient norm.

    Returns:
        Mean epoch loss as float.
    """
    model.train()
    total_loss = 0.0
    total_count = 0

    for x_batch, mask_batch, y_batch, dynamic_env_batch, static_env_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        mask_batch = mask_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        dynamic_env_batch = dynamic_env_batch.to(device, non_blocking=True)
        static_env_batch = static_env_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            x=x_batch,
            valid_mask=mask_batch,
            dynamic_env=dynamic_env_batch if dynamic_env_batch.shape[-1] > 0 else None,
            static_env=static_env_batch if static_env_batch.shape[-1] > 0 else None,
        )
        loss = criterion(logits, y_batch)
        loss.backward()
        if gradient_clip_norm > 0.0:
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        batch_size = int(x_batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate_improved(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Evaluate a model.

    Args:
        model: Improved model.
        loader: Validation or test DataLoader.
        device: Target device.
        class_names: Ordered class names.

    Returns:
        Dict with OA, Kappa, F1_macro, report and confusion matrix.
    """
    model.eval()
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    with torch.no_grad():
        for x_batch, mask_batch, y_batch, dynamic_env_batch, static_env_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            dynamic_env_batch = dynamic_env_batch.to(device, non_blocking=True)
            static_env_batch = static_env_batch.to(device, non_blocking=True)
            logits = model(
                x=x_batch,
                valid_mask=mask_batch,
                dynamic_env=dynamic_env_batch if dynamic_env_batch.shape[-1] > 0 else None,
                static_env=static_env_batch if static_env_batch.shape[-1] > 0 else None,
            )
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


def _extract_config_indices(
    metadata: Dict[str, Any],
    config_name: str,
    dynamic_dim: int,
    static_dim: int,
) -> Tuple[List[int], List[int]]:
    """
    Resolve dynamic and static environmental indices for one config.

    Args:
        metadata: Metadata dict containing ablation_configs.
        config_name: Ablation config name.
        dynamic_dim: Number of dynamic environmental channels.
        static_dim: Number of static environmental channels.

    Returns:
        Tuple (dynamic_indices, static_indices).
    """
    specs = metadata.get('ablation_configs', {})
    if config_name not in specs:
        return [], []

    spec = specs[config_name]
    if isinstance(spec, dict):
        dynamic_indices = [int(index) for index in spec.get('dynamic_indices', []) if 0 <= int(index) < dynamic_dim]
        static_indices = [int(index) for index in spec.get('static_indices', []) if 0 <= int(index) < static_dim]
        return dynamic_indices, static_indices

    if isinstance(spec, list):
        dynamic_count = len(metadata.get('environmental_covariates', {}).get('dynamic_columns', []))
        dynamic_indices = [int(index) for index in spec if 0 <= int(index) < min(dynamic_dim, dynamic_count)]
        static_indices = [int(index) - dynamic_count for index in spec if dynamic_count <= int(index) < dynamic_count + static_dim]
        static_indices = [index for index in static_indices if 0 <= index < static_dim]
        return dynamic_indices, static_indices

    return [], []


def _load_environment_splits(
    bundle: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    """
    Load dynamic and static environmental splits from a bundle.

    Args:
        bundle: NPZ content loaded into memory.
        metadata: Metadata dict.

    Returns:
        Tuple (dynamic_splits, static_splits), each mapping split names to
        arrays, or (None, None) if unavailable.
    """
    split_names = ['train', 'val', 'test']
    if all(f'dynamic_env_{split_name}' in bundle for split_name in split_names) and all(
        f'static_env_{split_name}' in bundle for split_name in split_names
    ):
        dynamic_splits = {
            split_name: np.asarray(bundle[f'dynamic_env_{split_name}'], dtype=np.float32)
            for split_name in split_names
        }
        static_splits = {
            split_name: np.asarray(bundle[f'static_env_{split_name}'], dtype=np.float32)
            for split_name in split_names
        }
        return dynamic_splits, static_splits

    if all(f'env_{split_name}' in bundle for split_name in split_names):
        dynamic_count = len(metadata.get('environmental_covariates', {}).get('dynamic_columns', []))
        static_count = len(metadata.get('environmental_covariates', {}).get('static_columns', []))
        dynamic_splits: Dict[str, np.ndarray] = {}
        static_splits: Dict[str, np.ndarray] = {}
        for split_name in split_names:
            env_tensor = np.asarray(bundle[f'env_{split_name}'], dtype=np.float32)
            dynamic_splits[split_name] = env_tensor[:, :, :dynamic_count].astype(np.float32)
            if static_count > 0:
                static_splits[split_name] = env_tensor[:, 0, dynamic_count:dynamic_count + static_count].astype(np.float32)
            else:
                static_splits[split_name] = np.zeros((env_tensor.shape[0], 0), dtype=np.float32)
        return dynamic_splits, static_splits

    return None, None


def _resolve_dataset_paths(processed_env_dir: Path, state: str) -> Tuple[Path, Path]:
    """
    Resolve dataset paths for one state.

    Args:
        processed_env_dir: Directory containing processed bundles.
        state: State slug.

    Returns:
        Tuple (npz_path, json_path).
    """
    candidates = [
        (
            processed_env_dir / f'{state}_mctnet_env_dataset.npz',
            processed_env_dir / f'{state}_mctnet_env_dataset.json',
        ),
        (
            processed_env_dir / f'{state}_mctnet_env_dataset_FULL_TEMPORAL.npz',
            processed_env_dir / f'{state}_mctnet_env_dataset_FULL_TEMPORAL.json',
        ),
        (
            processed_env_dir / f'{state}_mctnet_dataset.npz',
            processed_env_dir / f'{state}_mctnet_dataset.json',
        ),
    ]
    for npz_path, json_path in candidates:
        if npz_path.exists() and json_path.exists():
            return npz_path, json_path
    raise FileNotFoundError(f'Dataset files not found for state={state} in {processed_env_dir}')


def _select_split_features(
    split_tensor: np.ndarray,
    indices: Sequence[int],
    axis: int = -1,
) -> np.ndarray:
    """
    Select a feature subset from one split tensor.

    Args:
        split_tensor: Input array with feature dimension on `axis`.
        indices: Selected feature indices.
        axis: Feature axis.

    Returns:
        Selected array with the same leading dimensions.
    """
    if len(indices) == 0:
        shape = list(split_tensor.shape)
        shape[axis] = 0
        return np.zeros(shape, dtype=np.float32)
    return np.take(np.asarray(split_tensor, dtype=np.float32), indices=list(indices), axis=axis).astype(np.float32)


def _build_sample_weights(y_train: np.ndarray, power: float = 0.5) -> np.ndarray:
    """
    Build per-sample weights for balanced minibatch sampling.

    Args:
        y_train: Training labels of shape (N,).
        power: Frequency inverse exponent.

    Returns:
        Array of shape (N,).
    """
    num_classes = int(np.max(y_train)) + 1 if y_train.size > 0 else 0
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    class_weights = np.power(counts / counts.sum(), -power)
    return class_weights[y_train.astype(np.int64)].astype(np.float64)


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
    dynamic_env_train: Optional[np.ndarray],
    dynamic_env_val: Optional[np.ndarray],
    dynamic_env_test: Optional[np.ndarray],
    static_env_train: Optional[np.ndarray],
    static_env_val: Optional[np.ndarray],
    static_env_test: Optional[np.ndarray],
    batch_size: int,
    use_weighted_sampler: bool,
    sampler_power: float,
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
        dynamic_env_train: Optional train tensor of shape (N_train, 36, E_dynamic).
        dynamic_env_val: Optional validation tensor of shape (N_val, 36, E_dynamic).
        dynamic_env_test: Optional test tensor of shape (N_test, 36, E_dynamic).
        static_env_train: Optional train tensor of shape (N_train, E_static).
        static_env_val: Optional validation tensor of shape (N_val, E_static).
        static_env_test: Optional test tensor of shape (N_test, E_static).
        batch_size: Batch size.
        use_weighted_sampler: Whether to use a weighted sampler.
        sampler_power: Frequency inverse exponent for the sampler.

        Returns:
            Tuple of DataLoaders (train_loader, val_loader, test_loader).
    """
    train_dataset = CropDatasetImproved(
        x=x_train,
        mask=mask_train,
        y=y_train,
        dynamic_env=dynamic_env_train,
        static_env=static_env_train,
    )
    val_dataset = CropDatasetImproved(
        x=x_val,
        mask=mask_val,
        y=y_val,
        dynamic_env=dynamic_env_val,
        static_env=static_env_val,
    )
    test_dataset = CropDatasetImproved(
        x=x_test,
        mask=mask_test,
        y=y_test,
        dynamic_env=dynamic_env_test,
        static_env=static_env_test,
    )

    train_loader_kwargs: Dict[str, Any] = {
        'batch_size': batch_size,
        'num_workers': 0,
        'pin_memory': torch.cuda.is_available(),
    }
    if use_weighted_sampler:
        sample_weights = _build_sample_weights(y_train, power=sampler_power)
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


def run_part3_single_improved(
    npz_path: Path,
    json_path: Path,
    output_dir: Path,
    config_name: str = 'baseline',
    use_env_covariates: bool = True,
    args: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run one improved Part 3 experiment.

    Args:
        npz_path: Input bundle path.
        json_path: Metadata path.
        output_dir: Root output directory.
        config_name: Ablation config name.
        use_env_covariates: Whether to use environmental covariates.
        args: Namespace-like object with training hyperparameters.

    Returns:
        Dict with state, config, OA, Kappa, F1_macro and experiment metadata.
    """
    if args is None:
        args = SimpleNamespace(
            epochs=250,
            batch_size=32,
            learning_rate=1e-3,
            weight_decay=1e-4,
            dropout=0.1,
            early_stopping_patience=30,
            seed=2021,
            cpu=False,
            class_weight_power=0.5,
            label_smoothing=0.05,
            use_weighted_sampler=True,
            gradient_clip_norm=1.0,
            static_hidden_dim=32,
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

    x_train_vi = compute_vi_timeseries_improved(x_train_raw)
    x_val_vi = compute_vi_timeseries_improved(x_val_raw)
    x_test_vi = compute_vi_timeseries_improved(x_test_raw)
    x_train_vi, x_val_vi, x_test_vi, vi_mean, vi_std = normalize_vi_columns_improved(
        x_train_vi,
        x_val_vi,
        x_test_vi,
    )

    dynamic_splits, static_splits = _load_environment_splits(bundle, metadata)
    dynamic_dim = 0 if dynamic_splits is None else int(dynamic_splits['train'].shape[-1])
    static_dim = 0 if static_splits is None else int(static_splits['train'].shape[-1])

    if use_env_covariates and config_name != 'baseline':
        dynamic_indices, static_indices = _extract_config_indices(
            metadata=metadata,
            config_name=config_name,
            dynamic_dim=dynamic_dim,
            static_dim=static_dim,
        )
    else:
        dynamic_indices, static_indices = [], []

    dynamic_env_train = None
    dynamic_env_val = None
    dynamic_env_test = None
    if dynamic_splits is not None:
        dynamic_env_train = _select_split_features(dynamic_splits['train'], dynamic_indices, axis=-1)
        dynamic_env_val = _select_split_features(dynamic_splits['val'], dynamic_indices, axis=-1)
        dynamic_env_test = _select_split_features(dynamic_splits['test'], dynamic_indices, axis=-1)

    static_env_train = None
    static_env_val = None
    static_env_test = None
    if static_splits is not None:
        static_env_train = _select_split_features(static_splits['train'], static_indices, axis=-1)
        static_env_val = _select_split_features(static_splits['val'], static_indices, axis=-1)
        static_env_test = _select_split_features(static_splits['test'], static_indices, axis=-1)

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
        dynamic_env_train=dynamic_env_train,
        dynamic_env_val=dynamic_env_val,
        dynamic_env_test=dynamic_env_test,
        static_env_train=static_env_train,
        static_env_val=static_env_val,
        static_env_test=static_env_test,
        batch_size=int(args.batch_size),
        use_weighted_sampler=bool(getattr(args, 'use_weighted_sampler', True)),
        sampler_power=float(getattr(args, 'class_weight_power', 0.5)),
    )

    temporal_input_dim = int(x_train_vi.shape[-1] + len(dynamic_indices))
    static_input_dim = int(len(static_indices))
    num_classes = int(metadata['num_classes'])
    device = torch.device('cpu' if bool(getattr(args, 'cpu', False)) or not torch.cuda.is_available() else 'cuda')
    class_names = _resolve_class_names(metadata)

    model = MCTNetPart3Improved(
        temporal_input_dim=temporal_input_dim,
        static_input_dim=static_input_dim,
        num_classes=num_classes,
        seq_len=int(x_train_vi.shape[1]),
        dropout=float(args.dropout),
        static_hidden_dim=int(getattr(args, 'static_hidden_dim', 32)),
    ).to(device)
    class_weights = build_class_weights(
        y_train=y_train,
        num_classes=num_classes,
        device=device,
        power=float(getattr(args, 'class_weight_power', 0.5)),
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(getattr(args, 'label_smoothing', 0.05)),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(getattr(args, 'learning_rate', getattr(args, 'lr', 1e-3))),
        weight_decay=float(args.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(args.epochs))

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    train_losses: List[float] = []
    val_f1s: List[float] = []
    history_rows: List[Dict[str, float]] = []

    checkpoint_path = state_dir / f'best_mctnet_part3_improved_{config_name}.pt'
    history_path = state_dir / f'history_improved_{config_name}.json'
    confusion_png = state_dir / f'confusion_matrix_improved_{config_name}.png'
    curves_png = state_dir / f'learning_curves_improved_{config_name}.png'

    print(
        f'[{state}][{config_name}] '
        f'temporal_input_dim={temporal_input_dim} '
        f'static_input_dim={static_input_dim} '
        f'num_classes={num_classes}'
    )

    for epoch_idx in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch_improved(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip_norm=float(getattr(args, 'gradient_clip_norm', 1.0)),
        )
        scheduler.step()
        val_metrics = evaluate_improved(model, val_loader, device, class_names)

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
    test_metrics = evaluate_improved(model, test_loader, device, class_names)
    val_metrics_best = evaluate_improved(model, val_loader, device, class_names)

    plot_confusion_matrix(
        cm=np.asarray(test_metrics['cm'], dtype=np.int64),
        class_names=class_names,
        title=f'{state} - improved - {config_name}',
        output_path=confusion_png,
    )
    plot_learning_curves(
        train_losses=train_losses,
        val_f1s=val_f1s,
        config_name=f'{state}-improved-{config_name}',
        output_path=curves_png,
    )

    history_payload = {
        'state': state,
        'config': config_name,
        'temporal_input_dim': temporal_input_dim,
        'static_input_dim': static_input_dim,
        'dynamic_indices': list(dynamic_indices),
        'static_indices': list(static_indices),
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
        'temporal_input_dim': int(temporal_input_dim),
        'static_input_dim': int(static_input_dim),
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


def run_part3_full_ablation_improved(
    processed_env_dir: Path,
    output_dir: Path,
    states: List[str],
    args: Any,
) -> List[Dict[str, Any]]:
    """
    Run the full improved Part 3 ablation.

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
            result = run_part3_single_improved(
                npz_path=npz_path,
                json_path=json_path,
                output_dir=output_dir,
                config_name=config_name,
                use_env_covariates=bool(getattr(args, 'use_env_covariates', True)),
                args=args,
            )
            results.append(result)

    summary_csv = output_dir / 'part3_improved_ablation_summary.csv'
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


def plot_ablation_comparison_improved(summary_csv: Path, output_dir: Path) -> None:
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
    config_order = [
        config for config in ['mctnet_original', 'baseline', 'climate', 'soil', 'topography', 'all']
        if config in comparison_df['config'].unique()
    ]
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
    fig.savefig(output_dir / 'part3_improved_comparison_barplot.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """
    Command-line entry point.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description='Run improved Part 3 MCTNet experiments.')
    parser.add_argument('--processed-env-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--states', nargs='+', default=['arkansas', 'california'])
    parser.add_argument('--configs', nargs='+', default=['baseline', 'climate', 'soil', 'topography', 'all'])
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--class-weight-power', type=float, default=0.5)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0)
    parser.add_argument('--static-hidden-dim', type=int, default=32)
    parser.add_argument('--use-env-covariates', dest='use_env_covariates', action='store_true')
    parser.add_argument('--disable-env-covariates', dest='use_env_covariates', action='store_false')
    parser.add_argument('--use-weighted-sampler', dest='use_weighted_sampler', action='store_true')
    parser.add_argument('--disable-weighted-sampler', dest='use_weighted_sampler', action='store_false')
    parser.set_defaults(use_env_covariates=True, use_weighted_sampler=True)
    args = parser.parse_args()

    set_seed(int(args.seed))
    results = run_part3_full_ablation_improved(
        processed_env_dir=Path(args.processed_env_dir),
        output_dir=Path(args.output_dir),
        states=list(args.states),
        args=args,
    )
    plot_ablation_comparison_improved(
        summary_csv=Path(args.output_dir) / 'part3_improved_ablation_summary.csv',
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

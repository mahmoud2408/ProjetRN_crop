from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from mctnet import build_mctnet


DEFAULT_ABLATION_CONFIGS: List[str] = ['climate', 'soil', 'topography', 'all']


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MCTNetEnvDataset(Dataset):
    """
    Real-data dataset.

    - x_s2: [36, 10]
    - dynamic_env: [36, E_dynamic]
    - static_env: [E_static]

    The static features remain compact on disk and are broadcast inside the
    model forward pass, which keeps the CSV faithful to the source format.
    """

    def __init__(
        self,
        bundle: Dict[str, np.ndarray],
        split: str,
        dynamic_indices: List[int],
        static_indices: List[int],
    ) -> None:
        self.x = torch.from_numpy(bundle[f'x_{split}']).float()
        self.valid_mask = torch.from_numpy(bundle[f'valid_mask_{split}']).float()
        self.y = torch.from_numpy(bundle[f'y_{split}']).long()
        self.dynamic_env = torch.from_numpy(bundle[f'dynamic_env_{split}']).float()
        self.static_env = torch.from_numpy(bundle[f'static_env_{split}']).float()
        self.dynamic_indices = dynamic_indices
        self.static_indices = static_indices

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        dynamic_env = self.dynamic_env[index][:, self.dynamic_indices] if self.dynamic_indices else self.dynamic_env[index][:, :0]
        static_env = self.static_env[index][self.static_indices] if self.static_indices else self.static_env[index][:0]

        return {
            'x': self.x[index],
            'dynamic_env': dynamic_env,
            'static_env': static_env,
            'valid_mask': self.valid_mask[index],
            'y': self.y[index],
        }


def build_class_weights(
    y_train: np.ndarray,
    metadata: Dict,
    boost_classes: Optional[List[str]] = None,
    boost_factor: float = 1.0,
) -> Tuple[Tensor, Optional[WeightedRandomSampler]]:
    num_classes = len(metadata['class_name_to_index'])
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)
    class_weights = counts.sum() / (num_classes * counts.astype(np.float64))

    if boost_classes:
        for class_name in boost_classes:
            class_idx = metadata['class_name_to_index'].get(class_name)
            if class_idx is not None:
                class_weights[int(class_idx)] *= float(boost_factor)

    sample_weights = class_weights[y_train.astype(np.int64)]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return torch.as_tensor(class_weights, dtype=torch.float32), sampler


def build_dataloaders(
    bundle: Dict[str, np.ndarray],
    metadata: Dict,
    dynamic_indices: List[int],
    static_indices: List[int],
    batch_size: int,
    num_workers: int = 0,
    boost_classes: Optional[List[str]] = None,
    boost_factor: float = 1.0,
) -> Tuple[Dict[str, DataLoader], Tensor]:
    datasets: Dict[str, MCTNetEnvDataset] = {
        split_name: MCTNetEnvDataset(
            bundle=bundle,
            split=split_name,
            dynamic_indices=dynamic_indices,
            static_indices=static_indices,
        )
        for split_name in ['train', 'val', 'test']
    }

    class_weights, sampler = build_class_weights(
        y_train=bundle['y_train'],
        metadata=metadata,
        boost_classes=boost_classes,
        boost_factor=boost_factor,
    )

    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return loaders, class_weights


def get_ablation_config(metadata: Dict, config_name: str) -> Dict[str, object]:
    configs = metadata.get('ablation_configs', {})
    if config_name not in configs:
        raise KeyError(f'Unknown ablation config: {config_name}')
    return dict(configs[config_name])


def confusion_matrix_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        confusion[int(truth), int(pred)] += 1
    return confusion


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    confusion = confusion_matrix_from_predictions(y_true, y_pred, num_classes)
    total = confusion.sum()
    oa = float(np.trace(confusion) / total) if total > 0 else 0.0

    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    per_class_f1: List[float] = []

    for class_idx in range(num_classes):
        tp = confusion[class_idx, class_idx]
        fp = col_sums[class_idx] - tp
        fn = row_sums[class_idx] - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append(2.0 * precision * recall / (precision + recall))

    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    expected_accuracy = float((row_sums * col_sums).sum() / (total ** 2)) if total > 0 else 0.0
    if expected_accuracy == 1.0:
        kappa = 1.0
    else:
        kappa = float((oa - expected_accuracy) / (1.0 - expected_accuracy))

    return {'oa': oa, 'macro_f1': macro_f1, 'kappa': kappa}


def format_metric_triplet(metrics: Dict[str, float]) -> str:
    return (
        f'OA={metrics["oa"]:.4f}, '
        f'Kappa={metrics["kappa"]:.4f}, '
        f'F1={metrics["macro_f1"]:.4f}'
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    context_manager = torch.enable_grad() if is_training else torch.no_grad()
    with context_manager:
        for batch in loader:
            x = batch['x'].to(device, non_blocking=True)
            dynamic_env = batch['dynamic_env'].to(device, non_blocking=True)
            static_env = batch['static_env'].to(device, non_blocking=True)
            valid_mask = batch['valid_mask'].to(device, non_blocking=True)
            targets = batch['y'].to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(
                x=x,
                dynamic_env=dynamic_env if dynamic_env.shape[-1] > 0 else None,
                static_env=static_env if static_env.shape[-1] > 0 else None,
                valid_mask=valid_mask,
            )
            loss = criterion(logits, targets)

            if is_training:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            all_targets.append(targets.detach().cpu().numpy())
            all_predictions.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    metrics = classification_metrics(y_true, y_pred, num_classes=model.config.num_classes)
    metrics['loss'] = running_loss / max(len(loader.dataset), 1)
    return metrics


def plot_training_curves(history: Dict[str, Dict[str, float]], output_path: Path, title_prefix: str) -> None:
    epochs = sorted(history.keys(), key=lambda key: int(key.split('_')[1]))
    epoch_numbers = [int(epoch_key.split('_')[1]) for epoch_key in epochs]

    train_loss = [history[epoch_key]['train_loss'] for epoch_key in epochs]
    val_loss = [history[epoch_key]['val_loss'] for epoch_key in epochs]
    train_oa = [history[epoch_key]['train_oa'] for epoch_key in epochs]
    val_oa = [history[epoch_key]['val_oa'] for epoch_key in epochs]
    train_kappa = [history[epoch_key]['train_kappa'] for epoch_key in epochs]
    val_kappa = [history[epoch_key]['val_kappa'] for epoch_key in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    series = [
        (train_loss, val_loss, 'Loss', f'{title_prefix} - Loss'),
        (train_oa, val_oa, 'OA', f'{title_prefix} - OA'),
        (train_kappa, val_kappa, 'Kappa', f'{title_prefix} - Kappa'),
    ]

    for axis, (train_values, val_values, ylabel, title) in zip(axes, series):
        axis.plot(epoch_numbers, train_values, label='Train', color='#2563eb')
        axis.plot(epoch_numbers, val_values, label='Val', color='#dc2626')
        axis.set_xlabel('Epoch')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix_normalized(
    confusion: np.ndarray,
    class_names: List[str],
    output_png: Path,
    title: str,
) -> None:
    normalized = confusion.astype(np.float64)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(normalized, row_sums, out=np.zeros_like(normalized), where=row_sums != 0)

    fig, axis = plt.subplots(figsize=(max(6, len(class_names) * 1.4), max(5, len(class_names) * 1.1)))
    image = axis.imshow(normalized, cmap='Blues', vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha='right')
    axis.set_yticklabels(class_names)
    axis.set_xlabel('Predicted class')
    axis.set_ylabel('True class')
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
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                x=batch['x'].to(device, non_blocking=True),
                dynamic_env=batch['dynamic_env'].to(device, non_blocking=True) if batch['dynamic_env'].shape[-1] > 0 else None,
                static_env=batch['static_env'].to(device, non_blocking=True) if batch['static_env'].shape[-1] > 0 else None,
                valid_mask=batch['valid_mask'].to(device, non_blocking=True),
            )
            y_true_parts.append(batch['y'].cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)


def train_one_configuration(
    config_name: str,
    config_spec: Dict[str, object],
    bundle: Dict[str, np.ndarray],
    metadata: Dict,
    args: argparse.Namespace,
    state_output_dir: Path,
) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    num_classes = len(metadata['class_name_to_index'])
    dynamic_indices = list(config_spec['dynamic_indices'])
    static_indices = list(config_spec['static_indices'])
    input_dim = int(config_spec['input_dim'])

    model = build_mctnet(
        num_classes=num_classes,
        input_dim=input_dim,
        seq_len=metadata['sequence_length'],
        n_stages=args.n_stages,
        n_heads=args.n_heads,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_alpe=not args.no_alpe,
        use_missing_mask=not args.no_mask,
        use_cnn_branch=not args.no_cnn,
        use_transformer_branch=not args.no_trans,
    ).to(device)

    loaders, class_weights = build_dataloaders(
        bundle=bundle,
        metadata=metadata,
        dynamic_indices=dynamic_indices,
        static_indices=static_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        boost_classes=getattr(args, 'boost_classes', None),
        boost_factor=float(getattr(args, 'boost_factor', 1.0)),
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_kappa = -math.inf
    best_test_metrics: Dict[str, float] = {}
    best_val_metrics: Dict[str, float] = {}
    best_epoch_idx = 0
    history: Dict[str, Dict[str, float]] = {}
    patience_counter = 0
    last_train_metrics: Dict[str, float] = {}
    last_val_metrics: Dict[str, float] = {}
    last_test_metrics: Dict[str, float] = {}
    last_epoch_idx = 0

    checkpoint_path = state_output_dir / f'best_{config_name}.pt'
    curves_path = state_output_dir / f'loss_curves_{config_name}.png'
    history_path = state_output_dir / f'history_{config_name}.json'

    print(
        f'  Config={config_name:<11s} '
        f'input_dim={input_dim:<3d} '
        f'dynamic={len(dynamic_indices):<2d} '
        f'static={len(static_indices):<2d} '
        f'params={model.count_parameters():,}'
    )

    for epoch_idx in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders['val'], criterion, device)
        test_metrics = run_epoch(model, loaders['test'], criterion, device)
        last_train_metrics = dict(train_metrics)
        last_val_metrics = dict(val_metrics)
        last_test_metrics = dict(test_metrics)
        last_epoch_idx = epoch_idx

        history_key = f'epoch_{epoch_idx:03d}'
        history[history_key] = {
            'train_loss': train_metrics['loss'],
            'train_oa': train_metrics['oa'],
            'train_macro_f1': train_metrics['macro_f1'],
            'train_kappa': train_metrics['kappa'],
            'val_loss': val_metrics['loss'],
            'val_oa': val_metrics['oa'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_kappa': val_metrics['kappa'],
            'test_loss': test_metrics['loss'],
            'test_oa': test_metrics['oa'],
            'test_macro_f1': test_metrics['macro_f1'],
            'test_kappa': test_metrics['kappa'],
        }

        if val_metrics['kappa'] > best_val_kappa:
            best_val_kappa = val_metrics['kappa']
            best_test_metrics = dict(test_metrics)
            best_val_metrics = dict(val_metrics)
            best_epoch_idx = epoch_idx
            patience_counter = 0
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'num_classes': num_classes,
                        'input_dim': input_dim,
                        'seq_len': metadata['sequence_length'],
                        'n_stages': args.n_stages,
                        'n_heads': args.n_heads,
                        'kernel_size': args.kernel_size,
                        'dropout': args.dropout,
                        'use_alpe': not args.no_alpe,
                        'use_missing_mask': not args.no_mask,
                        'use_cnn_branch': not args.no_cnn,
                        'use_transformer_branch': not args.no_trans,
                    },
                    'ablation_config': config_name,
                    'config_spec': config_spec,
                    'best_val_kappa': best_val_kappa,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if epoch_idx % 10 == 0 or epoch_idx == args.epochs:
            print(
                f'    epoch {epoch_idx:03d}/{args.epochs} | '
                f'val({format_metric_triplet(val_metrics)}) | '
                f'test({format_metric_triplet(test_metrics)})'
            )

        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            print(f'    early stopping at epoch {epoch_idx}')
            break

    history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    plot_training_curves(history, curves_path, title_prefix=f'{metadata["state_name"]} - {config_name}')
    best_test_metrics['loss_curves_png'] = str(curves_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_model = build_mctnet(**checkpoint['model_config']).to(device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_val_eval_metrics = run_epoch(best_model, loaders['val'], criterion, device)
    best_test_eval_metrics = run_epoch(best_model, loaders['test'], criterion, device)

    y_true, y_pred = evaluate_predictions(best_model, loaders['test'], device)
    confusion = confusion_matrix_from_predictions(y_true, y_pred, num_classes=num_classes)

    class_names: List[Optional[str]] = [None] * num_classes
    for class_name, class_idx in metadata['class_name_to_index'].items():
        class_names[int(class_idx)] = class_name
    resolved_class_names = [class_name if class_name is not None else f'class_{idx}' for idx, class_name in enumerate(class_names)]

    confusion_png = state_output_dir / f'confusion_matrix_{config_name}.png'
    confusion_csv = state_output_dir / f'confusion_matrix_{config_name}.csv'
    confusion_npy = state_output_dir / f'confusion_matrix_{config_name}.npy'

    np.savetxt(confusion_csv, confusion, delimiter=',', fmt='%d')
    np.save(confusion_npy, confusion)
    plot_confusion_matrix_normalized(
        confusion=confusion,
        class_names=resolved_class_names,
        output_png=confusion_png,
        title=f'{metadata["state_name"]} - {config_name}',
    )

    print(
        f'    end training {config_name} | '
        f'last_epoch={last_epoch_idx:03d} | '
        f'val({format_metric_triplet(last_val_metrics)}) | '
        f'test({format_metric_triplet(last_test_metrics)})'
    )
    print(
        f'    best checkpoint {config_name} | '
        f'epoch={best_epoch_idx:03d} | '
        f'val({format_metric_triplet(best_val_eval_metrics)}) | '
        f'test({format_metric_triplet(best_test_eval_metrics)})'
    )

    best_test_metrics = dict(best_test_eval_metrics)
    best_test_metrics['best_epoch'] = best_epoch_idx
    best_test_metrics['best_val_oa'] = best_val_eval_metrics['oa']
    best_test_metrics['best_val_macro_f1'] = best_val_eval_metrics['macro_f1']
    best_test_metrics['best_val_kappa'] = best_val_eval_metrics['kappa']
    best_test_metrics['last_epoch'] = last_epoch_idx
    best_test_metrics['last_val_oa'] = last_val_metrics['oa']
    best_test_metrics['last_val_macro_f1'] = last_val_metrics['macro_f1']
    best_test_metrics['last_val_kappa'] = last_val_metrics['kappa']
    best_test_metrics['last_test_oa'] = last_test_metrics['oa']
    best_test_metrics['last_test_macro_f1'] = last_test_metrics['macro_f1']
    best_test_metrics['last_test_kappa'] = last_test_metrics['kappa']
    best_test_metrics['confusion_matrix_png'] = str(confusion_png)
    best_test_metrics['confusion_matrix_csv'] = str(confusion_csv)
    best_test_metrics['confusion_matrix_npy'] = str(confusion_npy)
    return best_test_metrics


def plot_ablation_barplot(results: Dict[str, Dict[str, Dict[str, float]]], metric: str, output_png: Path) -> None:
    if not results:
        return

    state_names = list(results.keys())
    config_names = list(next(iter(results.values())).keys())
    x = np.arange(len(config_names))
    width = 0.8 / max(len(state_names), 1)
    colors = ['#2563eb', '#16a34a', '#dc2626', '#d97706']

    fig, axis = plt.subplots(figsize=(max(10, len(config_names) * 1.8), 6))
    for state_idx, state_name in enumerate(state_names):
        values = [results[state_name].get(config_name, {}).get(metric, 0.0) for config_name in config_names]
        offset = (state_idx - (len(state_names) - 1) / 2) * width
        bars = axis.bar(
            x + offset,
            values,
            width=width,
            label=state_name.title(),
            color=colors[state_idx % len(colors)],
        )
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    axis.set_ylabel(metric.upper())
    axis.set_title(f'Ablation Study - {metric.upper()}')
    axis.set_xticks(x)
    axis.set_xticklabels([config_name.capitalize() for config_name in config_names], rotation=15)
    axis.set_ylim(0, 1.05)
    axis.grid(axis='y', linestyle='--', alpha=0.4)
    axis.legend()
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_summary_table(results: Dict[str, Dict[str, Dict[str, float]]], output_csv: Path, output_json: Path) -> None:
    rows: List[Dict[str, object]] = []

    for state_name, config_metrics in results.items():
        for config_name, metrics in config_metrics.items():
            rows.append(
                {
                    'state': state_name,
                    'config': config_name,
                    'best_epoch': int(metrics.get('best_epoch', 0)),
                    'oa': round(metrics.get('oa', 0.0), 4),
                    'macro_f1': round(metrics.get('macro_f1', 0.0), 4),
                    'kappa': round(metrics.get('kappa', 0.0), 4),
                    'loss': round(metrics.get('loss', 0.0), 4),
                    'best_val_oa': round(metrics.get('best_val_oa', 0.0), 4),
                    'best_val_macro_f1': round(metrics.get('best_val_macro_f1', 0.0), 4),
                    'best_val_kappa': round(metrics.get('best_val_kappa', 0.0), 4),
                    'last_epoch': int(metrics.get('last_epoch', 0)),
                    'last_val_oa': round(metrics.get('last_val_oa', 0.0), 4),
                    'last_val_macro_f1': round(metrics.get('last_val_macro_f1', 0.0), 4),
                    'last_val_kappa': round(metrics.get('last_val_kappa', 0.0), 4),
                    'last_test_oa': round(metrics.get('last_test_oa', 0.0), 4),
                    'last_test_macro_f1': round(metrics.get('last_test_macro_f1', 0.0), 4),
                    'last_test_kappa': round(metrics.get('last_test_kappa', 0.0), 4),
                    'confusion_matrix_png': metrics.get('confusion_matrix_png', ''),
                    'loss_curves_png': metrics.get('loss_curves_png', ''),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'state',
                'config',
                'best_epoch',
                'oa',
                'macro_f1',
                'kappa',
                'loss',
                'best_val_oa',
                'best_val_macro_f1',
                'best_val_kappa',
                'last_epoch',
                'last_val_oa',
                'last_val_macro_f1',
                'last_val_kappa',
                'last_test_oa',
                'last_test_macro_f1',
                'last_test_kappa',
                'confusion_matrix_png',
                'loss_curves_png',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_json.write_text(json.dumps(results, indent=2), encoding='utf-8')


def run_ablation_experiment(
    processed_env_dir: Path,
    output_dir: Path,
    states: List[str],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for state_slug in states:
        npz_path = processed_env_dir / f'{state_slug}_mctnet_env_dataset.npz'
        json_path = processed_env_dir / f'{state_slug}_mctnet_env_dataset.json'

        if not npz_path.exists() or not json_path.exists():
            print(f'[WARN] missing dataset for {state_slug}, skipping')
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            bundle = {key: data[key] for key in data.files}
        metadata = json.loads(json_path.read_text(encoding='utf-8'))

        state_output_dir = output_dir / state_slug
        state_output_dir.mkdir(parents=True, exist_ok=True)
        results[state_slug] = {}

        print(f'\n{"=" * 72}')
        print(f'State: {metadata["state_name"]}')
        print(f'Spectral shape: {metadata["feature_shape_per_sample"]}')
        print(f'Dynamic env shape: {metadata["dynamic_env_shape_per_sample"]}')
        print(f'Static env shape: {metadata["static_env_shape_per_sample"]}')

        for config_name in args.configs:
            config_spec = get_ablation_config(metadata, config_name)
            set_seed(args.seed)
            test_metrics = train_one_configuration(
                config_name=config_name,
                config_spec=config_spec,
                bundle=bundle,
                metadata=metadata,
                args=args,
                state_output_dir=state_output_dir,
            )
            results[state_slug][config_name] = test_metrics

    if not results:
        print('[WARN] no ablation results were produced')
        return results

    for metric in ['oa', 'macro_f1', 'kappa']:
        plot_ablation_barplot(results, metric=metric, output_png=output_dir / f'ablation_{metric}_barplot.png')

    save_summary_table(
        results=results,
        output_csv=output_dir / 'ablation_summary.csv',
        output_json=output_dir / 'ablation_results.json',
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the 5-way MCTNet ablation study with temporal climate and static covariates.')
    parser.add_argument('--processed-env-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--states', nargs='+', default=['arkansas'])
    parser.add_argument('--configs', nargs='+', default=DEFAULT_ABLATION_CONFIGS)

    # Paper-aligned defaults.
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--n-stages', type=int, default=3)
    parser.add_argument('--n-heads', type=int, default=5)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--optimizer', default='adam', choices=['adam'])

    # Stable training defaults.
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--early-stopping-patience', type=int, default=25)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--boost-classes', nargs='*', default=None)
    parser.add_argument('--boost-factor', type=float, default=1.0)

    # Paper ablations.
    parser.add_argument('--no-alpe', action='store_true')
    parser.add_argument('--no-mask', action='store_true')
    parser.add_argument('--no-cnn', action='store_true')
    parser.add_argument('--no-trans', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_ablation_experiment(
        processed_env_dir=Path(args.processed_env_dir),
        output_dir=Path(args.output_dir),
        states=args.states,
        args=args,
    )


if __name__ == '__main__':
    main()

"""
Training script for MCTNet on the processed crop datasets (.npz + .json).

Expected inputs:
  - dataset .npz produced by build_mctnet_dataset.py
  - metadata .json produced by build_mctnet_dataset.py

Paper-driven defaults:
  - n_stage = 3
  - n_head = 5
  - kernel_size = 3
  - learning_rate = 0.001
  - optimizer = Adam

Implementation choices:
  - batch_size defaults to 32
  - early stopping on validation kappa
  - checkpointing of the best validation model
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from mctnet_model import MCTNet, build_mctnet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CropNPZDataset(Dataset):
    def __init__(self, bundle: Dict[str, np.ndarray], split: str) -> None:
        self.x = torch.from_numpy(bundle[f'x_{split}']).float()
        self.valid_mask = torch.from_numpy(bundle[f'valid_mask_{split}']).float()
        self.y = torch.from_numpy(bundle[f'y_{split}']).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            'x': self.x[index],
            'valid_mask': self.valid_mask[index],
            'y': self.y[index],
        }


def load_bundle(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_dataloaders(
    bundle: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split in ['train', 'val', 'test']:
        dataset = CropNPZDataset(bundle=bundle, split=split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def confusion_matrix_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    cm = confusion_matrix_from_predictions(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class_f1: List[float] = []
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)

    for class_idx in range(num_classes):
        tp = cm[class_idx, class_idx]
        fp = col_sums[class_idx] - tp
        fn = row_sums[class_idx] - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append(2.0 * precision * recall / (precision + recall))

    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    expected_accuracy = float((row_sums * col_sums).sum() / (total ** 2)) if total > 0 else 0.0
    if expected_accuracy == 1.0:
        kappa = 1.0
    else:
        kappa = float((accuracy - expected_accuracy) / (1.0 - expected_accuracy))

    return {
        'oa': accuracy,
        'macro_f1': macro_f1,
        'kappa': kappa,
    }


def run_epoch(
    model: MCTNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        valid_mask = batch['valid_mask'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x=x, valid_mask=valid_mask)
        loss = criterion(logits, y)

        if is_training:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * x.size(0)
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(predictions, axis=0)
    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=model.config.num_classes,
    )
    metrics['loss'] = running_loss / len(loader.dataset)
    return metrics


def save_checkpoint(
    checkpoint_path: Path,
    model: MCTNet,
    metadata: Dict[str, object],
    args: argparse.Namespace,
    best_val_metrics: Dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_state_dict': model.state_dict(),
        'model_config': asdict(model.config),
        'metadata': metadata,
        'training_args': vars(args),
        'best_val_metrics': best_val_metrics,
    }
    torch.save(payload, checkpoint_path)


def train_model(
    args: argparse.Namespace,
    bundle: Dict[str, np.ndarray],
    metadata: Dict[str, object],
) -> Tuple[MCTNet, Dict[str, Dict[str, float]]]:
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    loaders = build_dataloaders(
        bundle=bundle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(metadata['class_name_to_index'])
    model = build_mctnet(
        num_classes=num_classes,
        input_dim=int(bundle['x_train'].shape[-1]),
        seq_len=int(bundle['x_train'].shape[1]),
        n_stages=args.n_stages,
        n_heads=args.n_heads,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_alpe=not args.no_alpe,
        use_missing_mask=not args.no_mask,
        use_cnn_branch=not args.no_cnn,
        use_transformer_branch=not args.no_trans,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_kappa = float('-inf')
    best_val_metrics: Dict[str, float] = {}
    best_test_metrics: Dict[str, float] = {}
    patience_counter = 0
    history: Dict[str, Dict[str, float]] = {}

    print(f'Device: {device}')
    print(f'Parameter count: {model.count_parameters()}')
    print(f'Classes: {metadata["class_name_to_index"]}')

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=loaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = run_epoch(
            model=model,
            loader=loaders['val'],
            criterion=criterion,
            optimizer=None,
            device=device,
        )
        test_metrics = run_epoch(
            model=model,
            loader=loaders['test'],
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history[f'epoch_{epoch:03d}'] = {
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

        print(
            f'Epoch {epoch:03d} | '
            f'train loss={train_metrics["loss"]:.4f} oa={train_metrics["oa"]:.4f} '
            f'f1={train_metrics["macro_f1"]:.4f} kappa={train_metrics["kappa"]:.4f} | '
            f'val loss={val_metrics["loss"]:.4f} oa={val_metrics["oa"]:.4f} '
            f'f1={val_metrics["macro_f1"]:.4f} kappa={val_metrics["kappa"]:.4f} | '
            f'test oa={test_metrics["oa"]:.4f} f1={test_metrics["macro_f1"]:.4f} '
            f'kappa={test_metrics["kappa"]:.4f}'
        )

        if val_metrics['kappa'] > best_val_kappa:
            best_val_kappa = val_metrics['kappa']
            best_val_metrics = dict(val_metrics)
            best_test_metrics = dict(test_metrics)
            patience_counter = 0
            save_checkpoint(
                checkpoint_path=Path(args.checkpoint_path),
                model=model,
                metadata=metadata,
                args=args,
                best_val_metrics=best_val_metrics,
            )
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print(f'Early stopping at epoch {epoch}.')
            break

    metrics_payload = {
        'best_val': best_val_metrics,
        'test_at_best_val': best_test_metrics,
        'history': history,
    }
    return model, metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train MCTNet on processed crop datasets.')
    parser.add_argument('--dataset-npz', required=True, help='Path to the processed .npz dataset.')
    parser.add_argument('--metadata-json', required=True, help='Path to the metadata .json file.')
    parser.add_argument('--output-dir', required=True, help='Directory for checkpoints and metrics.')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Adam learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Adam weight decay.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Model dropout.')
    parser.add_argument('--n-stages', type=int, default=3, choices=[1, 2, 3], help='Number of MCTNet stages.')
    parser.add_argument('--n-heads', type=int, default=5, help='Number of attention heads.')
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel size for CNN and ALPE Conv1D.')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed.')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers.')
    parser.add_argument('--early-stopping-patience', type=int, default=15, help='Validation patience.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training.')
    parser.add_argument('--no-alpe', action='store_true', help='Disable ALPE and use no learnable PE.')
    parser.add_argument('--no-mask', action='store_true', help='Disable the missing-data mask inside ALPE.')
    parser.add_argument('--no-cnn', action='store_true', help='Disable the CNN branch.')
    parser.add_argument('--no-trans', action='store_true', help='Disable the Transformer branch.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path = str(output_dir / 'best_mctnet.pt')
    metrics_path = output_dir / 'metrics.json'

    bundle = load_bundle(Path(args.dataset_npz))
    metadata = json.loads(Path(args.metadata_json).read_text(encoding='utf-8'))

    _, metrics_payload = train_model(args=args, bundle=bundle, metadata=metadata)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding='utf-8')

    print('Best validation metrics:', metrics_payload['best_val'])
    print('Test metrics at best validation epoch:', metrics_payload['test_at_best_val'])
    print(f'Checkpoint written to: {args.checkpoint_path}')
    print(f'Metrics written to: {metrics_path}')


if __name__ == '__main__':
    main()

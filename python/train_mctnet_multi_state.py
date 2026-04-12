

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from mctnet_model import MCTNet, MCTNetConfig
from train_mctnet import (
    build_dataloaders,
    classification_metrics,
    confusion_matrix_from_predictions,
    load_bundle,
    set_seed,
    train_model,
)


def load_best_model(checkpoint_path: Path, device: torch.device) -> MCTNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = MCTNetConfig(**checkpoint['model_config'])
    model = MCTNet(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_best_checkpoint_on_test(
    bundle: Dict[str, np.ndarray],
    metadata: Dict[str, object],
    checkpoint_path: Path,
    batch_size: int,
    num_workers: int,
    force_cpu: bool,
) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    model = load_best_model(checkpoint_path=checkpoint_path, device=device)
    loaders = build_dataloaders(bundle=bundle, batch_size=batch_size, num_workers=num_workers)

    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loaders['test']:
            x = batch['x'].to(device, non_blocking=True)
            valid_mask = batch['valid_mask'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)

            logits = model(x=x, valid_mask=valid_mask)
            preds = logits.argmax(dim=1)

            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_parts, axis=0)
    y_pred = np.concatenate(y_pred_parts, axis=0)

    num_classes = len(metadata['class_name_to_index'])
    cm = confusion_matrix_from_predictions(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    class_names = [None] * num_classes
    for class_name, class_index in metadata['class_name_to_index'].items():
        class_names[int(class_index)] = class_name

    return cm, metrics, class_names


def save_confusion_matrix_csv(cm: np.ndarray, class_names: List[str], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['true/pred'] + class_names)
        for class_name, row in zip(class_names, cm.tolist()):
            writer.writerow([class_name] + row)


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    output_png: Path,
    normalize: bool = True,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    matrix = cm.astype(np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Normalized confusion matrix' if normalize else 'Confusion matrix')

    threshold = matrix.max() / 2.0 if matrix.size > 0 else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = f'{value:.2f}' if normalize else str(int(value))
            ax.text(
                col_idx,
                row_idx,
                text,
                ha='center',
                va='center',
                color='white' if value > threshold else 'black',
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_train_args(base_args: argparse.Namespace, state_output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_npz='',
        metadata_json='',
        output_dir=str(state_output_dir),
        checkpoint_path=str(state_output_dir / 'best_mctnet.pt'),
        epochs=base_args.epochs,
        batch_size=base_args.batch_size,
        learning_rate=base_args.learning_rate,
        weight_decay=base_args.weight_decay,
        dropout=base_args.dropout,
        n_stages=base_args.n_stages,
        n_heads=base_args.n_heads,
        kernel_size=base_args.kernel_size,
        seed=base_args.seed,
        num_workers=base_args.num_workers,
        early_stopping_patience=base_args.early_stopping_patience,
        cpu=base_args.cpu,
        no_alpe=base_args.no_alpe,
        no_mask=base_args.no_mask,
        no_cnn=base_args.no_cnn,
        no_trans=base_args.no_trans,
    )


def run_state_training(
    state_slug: str,
    processed_dir: Path,
    output_dir: Path,
    base_args: argparse.Namespace,
) -> Dict[str, object]:
    dataset_npz = processed_dir / f'{state_slug}_mctnet_dataset.npz'
    metadata_json = processed_dir / f'{state_slug}_mctnet_dataset.json'

    if not dataset_npz.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_npz}')
    if not metadata_json.exists():
        raise FileNotFoundError(f'Metadata not found: {metadata_json}')

    state_output_dir = output_dir / state_slug
    state_output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(dataset_npz)
    metadata = json.loads(metadata_json.read_text(encoding='utf-8'))

    train_args = make_train_args(base_args=base_args, state_output_dir=state_output_dir)
    train_args.dataset_npz = str(dataset_npz)
    train_args.metadata_json = str(metadata_json)

    print('=' * 100)
    print(f'Training state: {state_slug}')
    print(f'Dataset: {dataset_npz}')
    print(f'Output dir: {state_output_dir}')

    _, metrics_payload = train_model(args=train_args, bundle=bundle, metadata=metadata)

    metrics_path = state_output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding='utf-8')

    checkpoint_path = state_output_dir / 'best_mctnet.pt'
    cm, test_metrics, class_names = evaluate_best_checkpoint_on_test(
        bundle=bundle,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        batch_size=base_args.batch_size,
        num_workers=base_args.num_workers,
        force_cpu=base_args.cpu,
    )

    np.save(state_output_dir / 'test_confusion_matrix.npy', cm)
    save_confusion_matrix_csv(
        cm=cm,
        class_names=class_names,
        output_csv=state_output_dir / 'test_confusion_matrix.csv',
    )
    save_confusion_matrix_plot(
        cm=cm,
        class_names=class_names,
        output_png=state_output_dir / 'test_confusion_matrix.png',
        normalize=True,
    )

    evaluation_payload = {
        'test_metrics_from_best_checkpoint': test_metrics,
        'class_names': class_names,
    }
    (state_output_dir / 'test_evaluation.json').write_text(
        json.dumps(evaluation_payload, indent=2),
        encoding='utf-8',
    )

    summary_row = {
        'state_slug': state_slug,
        'state_name': metadata['state_name'],
        'num_classes': len(metadata['class_name_to_index']),
        'best_val_oa': metrics_payload['best_val'].get('oa', 0.0),
        'best_val_macro_f1': metrics_payload['best_val'].get('macro_f1', 0.0),
        'best_val_kappa': metrics_payload['best_val'].get('kappa', 0.0),
        'test_oa': test_metrics.get('oa', 0.0),
        'test_macro_f1': test_metrics.get('macro_f1', 0.0),
        'test_kappa': test_metrics.get('kappa', 0.0),
        'checkpoint_path': str(checkpoint_path),
        'metrics_path': str(metrics_path),
        'confusion_matrix_png': str(state_output_dir / 'test_confusion_matrix.png'),
    }

    print(f'Finished state: {state_slug}')
    print(json.dumps(summary_row, indent=2))
    return summary_row


def write_summary_csv(summary_rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return

    fieldnames = list(summary_rows[0].keys())
    with output_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def run_multi_state_experiment(
    processed_dir: Path,
    output_dir: Path,
    states: List[str],
    base_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []

    for state_slug in states:
        summary_rows.append(
            run_state_training(
                state_slug=state_slug,
                processed_dir=processed_dir,
                output_dir=output_dir,
                base_args=base_args,
            )
        )

    summary_json = output_dir / 'summary.json'
    summary_csv = output_dir / 'summary.csv'
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding='utf-8')
    write_summary_csv(summary_rows=summary_rows, output_csv=summary_csv)

    print('=' * 100)
    print(f'Global summary written to: {summary_json}')
    print(f'Global summary CSV written to: {summary_csv}')
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train MCTNet on multiple processed states.')
    parser.add_argument('--processed-dir', required=True, help='Directory containing processed .npz/.json datasets.')
    parser.add_argument('--output-dir', required=True, help='Output directory for training runs.')
    parser.add_argument(
        '--states',
        nargs='+',
        default=['arkansas', 'california'],
        help='State slugs to train, e.g. arkansas california.',
    )
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
    parser.add_argument('--no-alpe', action='store_true', help='Disable ALPE.')
    parser.add_argument('--no-mask', action='store_true', help='Disable missing-data mask in ALPE.')
    parser.add_argument('--no-cnn', action='store_true', help='Disable CNN branch.')
    parser.add_argument('--no-trans', action='store_true', help='Disable Transformer branch.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = run_multi_state_experiment(
        processed_dir=processed_dir,
        output_dir=output_dir,
        states=args.states,
        base_args=args,
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == '__main__':
    main()

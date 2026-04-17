"""
run_mctnet_training.py
======================
Script d'entraînement autonome pour MCTNet.
Peut être utilisé directement en ligne de commande OU importé par
train_mctnet_multi_state.py (il exporte le même contrat d'API que train_mctnet.py).

Usage CLI :
    python run_mctnet_training.py \
        --dataset-npz /path/arkansas_mctnet_dataset.npz \
        --metadata-json /path/arkansas_mctnet_dataset.json \
        --output-dir /path/runs/arkansas \
        --epochs 100
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from mctnet_model import MCTNet, MCTNetConfig, build_mctnet


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class CropNPZDataset(Dataset):
    """Charge les tenseurs (x, valid_mask, y) depuis un bundle NPZ."""

    def __init__(self, bundle: Dict[str, np.ndarray], split: str) -> None:
        self.x          = torch.from_numpy(bundle[f'x_{split}']).float()
        self.valid_mask = torch.from_numpy(bundle[f'valid_mask_{split}']).float()
        self.y          = torch.from_numpy(bundle[f'y_{split}']).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            'x':          self.x[index],
            'valid_mask': self.valid_mask[index],
            'y':          self.y[index],
        }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_bundle(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_dataloaders(
    bundle:      Dict[str, np.ndarray],
    batch_size:  int,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split in ['train', 'val', 'test']:
        dataset = CropNPZDataset(bundle, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def confusion_matrix_from_predictions(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    cm        = confusion_matrix_from_predictions(y_true, y_pred, num_classes)
    total     = cm.sum()
    accuracy  = float(np.trace(cm) / total) if total > 0 else 0.0
    row_sums  = cm.sum(axis=1)
    col_sums  = cm.sum(axis=0)

    per_class_f1: List[float] = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = col_sums[i] - tp
        fn = row_sums[i] - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1.append(
            2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        )

    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    exp_acc  = float((row_sums * col_sums).sum() / total ** 2) if total > 0 else 0.0
    kappa    = (
        1.0 if exp_acc == 1.0
        else float((accuracy - exp_acc) / (1.0 - exp_acc))
    )
    return {'oa': accuracy, 'macro_f1': macro_f1, 'kappa': kappa}


# ---------------------------------------------------------------------------
# Boucle d'une époque
# ---------------------------------------------------------------------------

def run_epoch(
    model:     MCTNet,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_preds:   List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            x          = batch['x'].to(device, non_blocking=True)
            valid_mask = batch['valid_mask'].to(device, non_blocking=True)
            y          = batch['y'].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x=x, valid_mask=valid_mask)
            loss   = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss  += float(loss.item()) * x.size(0)
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    y_true  = np.concatenate(all_targets)
    y_pred  = np.concatenate(all_preds)
    metrics = classification_metrics(y_true, y_pred, model.config.num_classes)
    metrics['loss'] = running_loss / max(len(loader.dataset), 1)
    return metrics


# ---------------------------------------------------------------------------
# Sauvegarde checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_path: Path,
    model:           MCTNet,
    metadata:        Dict,
    args:            argparse.Namespace,
    best_val_metrics: Dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config':     asdict(model.config),
        'metadata':         metadata,
        'training_args':    vars(args),
        'best_val_metrics': best_val_metrics,
    }, checkpoint_path)


# ---------------------------------------------------------------------------
# Visualisation des courbes d'apprentissage
# ---------------------------------------------------------------------------

def plot_training_curves(
    history:    Dict[str, Dict[str, float]],
    output_png: Path,
    state_name: str = '',
) -> None:
    epochs = sorted(history.keys(), key=lambda k: int(k.split('_')[1]))
    ep_nums      = [int(k.split('_')[1]) for k in epochs]
    train_loss   = [history[k]['train_loss']   for k in epochs]
    val_loss     = [history[k]['val_loss']     for k in epochs]
    train_oa     = [history[k]['train_oa']     for k in epochs]
    val_oa       = [history[k]['val_oa']       for k in epochs]
    train_kappa  = [history[k]['train_kappa']  for k in epochs]
    val_kappa    = [history[k]['val_kappa']    for k in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    title_prefix = f'{state_name} – ' if state_name else ''

    for ax, (train_vals, val_vals, ylabel, title) in zip(axes, [
        (train_loss,  val_loss,  'Loss',    f'{title_prefix}Loss'),
        (train_oa,    val_oa,    'OA',      f'{title_prefix}Overall Accuracy'),
        (train_kappa, val_kappa, 'Kappa',   f'{title_prefix}Kappa'),
    ]):
        ax.plot(ep_nums, train_vals, label='Train', color='#2563eb')
        ax.plot(ep_nums, val_vals,   label='Val',   color='#dc2626')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: List[str],
    output_png:  Path,
    state_name:  str = '',
) -> None:
    matrix = cm.astype(np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix   = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.4),
                                    max(5, len(class_names) * 1.2)))
    img = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    title = 'Normalized Confusion Matrix'
    if state_name:
        title = f'{state_name} – {title}'
    ax.set_title(title)

    thresh = 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i,j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, j] > thresh else 'black')

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Boucle d'entraînement principale
# ---------------------------------------------------------------------------

def train_model(
    args:     argparse.Namespace,
    bundle:   Dict[str, np.ndarray],
    metadata: Dict,
) -> Tuple[MCTNet, Dict]:
    device  = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    loaders = build_dataloaders(bundle, args.batch_size, args.num_workers)

    num_classes = len(metadata['class_name_to_index'])
    model = build_mctnet(
        num_classes           = num_classes,
        input_dim             = int(bundle['x_train'].shape[-1]),
        seq_len               = int(bundle['x_train'].shape[1]),
        n_stages              = args.n_stages,
        n_heads               = args.n_heads,
        kernel_size           = args.kernel_size,
        dropout               = args.dropout,
        use_alpe              = not args.no_alpe,
        use_missing_mask      = not args.no_mask,
        use_cnn_branch        = not args.no_cnn,
        use_transformer_branch= not args.no_trans,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_val_kappa    = float('-inf')
    best_val_metrics: Dict[str, float] = {}
    best_test_metrics: Dict[str, float] = {}
    patience_counter  = 0
    history: Dict[str, Dict[str, float]] = {}

    checkpoint_path = Path(args.checkpoint_path)

    print(f'\n{"="*70}')
    print(f'État          : {metadata["state_name"]}')
    print(f'Device        : {device}')
    print(f'Paramètres    : {model.count_parameters():,}')
    print(f'Classes ({num_classes}) : {list(metadata["class_name_to_index"].keys())}')
    print(f'Splits        : train={len(loaders["train"].dataset)} '
          f'val={len(loaders["val"].dataset)} '
          f'test={len(loaders["test"].dataset)}')
    print(f'{"="*70}')

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val_m   = run_epoch(model, loaders['val'],   criterion, device)
        test_m  = run_epoch(model, loaders['test'],  criterion, device)
        scheduler.step()

        history[f'epoch_{epoch:03d}'] = {
            'train_loss':     train_m['loss'],  'train_oa':    train_m['oa'],
            'train_macro_f1': train_m['macro_f1'], 'train_kappa': train_m['kappa'],
            'val_loss':       val_m['loss'],    'val_oa':      val_m['oa'],
            'val_macro_f1':   val_m['macro_f1'],  'val_kappa':   val_m['kappa'],
            'test_loss':      test_m['loss'],   'test_oa':     test_m['oa'],
            'test_macro_f1':  test_m['macro_f1'], 'test_kappa':  test_m['kappa'],
        }

        print(
            f'Epoch {epoch:03d}/{args.epochs} | '
            f'Loss {train_m["loss"]:.4f}/{val_m["loss"]:.4f} | '
            f'OA {train_m["oa"]:.4f}/{val_m["oa"]:.4f} | '
            f'F1 {train_m["macro_f1"]:.4f}/{val_m["macro_f1"]:.4f} | '
            f'κ {train_m["kappa"]:.4f}/{val_m["kappa"]:.4f} | '
            f'Test OA {test_m["oa"]:.4f}'
        )

        if val_m['kappa'] > best_val_kappa:
            best_val_kappa    = val_m['kappa']
            best_val_metrics  = dict(val_m)
            best_test_metrics = dict(test_m)
            patience_counter  = 0
            save_checkpoint(checkpoint_path, model, metadata, args, best_val_metrics)
            print(f'  ✓ Nouveau meilleur checkpoint (val κ={best_val_kappa:.4f})')
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print(f'  Early stopping déclenché à l\'époque {epoch}.')
            break

    # Sauvegarde des courbes d'apprentissage
    output_dir = Path(args.output_dir)
    plot_training_curves(
        history,
        output_dir / 'training_curves.png',
        state_name=metadata['state_name'],
    )

    metrics_payload = {
        'best_val':          best_val_metrics,
        'test_at_best_val':  best_test_metrics,
        'history':           history,
    }
    return model, metrics_payload


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Entraîne MCTNet sur un dataset .npz.')
    p.add_argument('--dataset-npz',   required=True,  help='Chemin vers le fichier .npz.')
    p.add_argument('--metadata-json', required=True,  help='Chemin vers le fichier .json.')
    p.add_argument('--output-dir',    required=True,  help='Dossier de sortie.')
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch-size',    type=int,   default=32)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay',  type=float, default=1e-4)
    p.add_argument('--dropout',       type=float, default=0.1)
    p.add_argument('--n-stages',      type=int,   default=3, choices=[1, 2, 3])
    p.add_argument('--n-heads',       type=int,   default=5)
    p.add_argument('--kernel-size',   type=int,   default=3)
    p.add_argument('--seed',          type=int,   default=2021)
    p.add_argument('--num-workers',   type=int,   default=0)
    p.add_argument('--early-stopping-patience', type=int, default=15)
    p.add_argument('--cpu',      action='store_true')
    p.add_argument('--no-alpe',  action='store_true')
    p.add_argument('--no-mask',  action='store_true')
    p.add_argument('--no-cnn',   action='store_true')
    p.add_argument('--no-trans', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path = str(output_dir / 'best_mctnet.pt')

    bundle   = load_bundle(Path(args.dataset_npz))
    metadata = json.loads(Path(args.metadata_json).read_text(encoding='utf-8'))

    model, metrics_payload = train_model(args=args, bundle=bundle, metadata=metadata)

    # Sauvegarde métriques
    metrics_path = output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding='utf-8')

    # Matrice de confusion sur le test set depuis le meilleur checkpoint
    checkpoint = torch.load(args.checkpoint_path,
                            map_location='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    best_model = MCTNet(MCTNetConfig(**checkpoint['model_config']))
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    best_model.to(device)

    loaders   = build_dataloaders(bundle, args.batch_size, args.num_workers)
    y_true_l, y_pred_l = [], []
    with torch.no_grad():
        for batch in loaders['test']:
            logits = best_model(batch['x'].to(device), batch['valid_mask'].to(device))
            y_true_l.append(batch['y'].numpy())
            y_pred_l.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true_l)
    y_pred = np.concatenate(y_pred_l)
    num_classes  = len(metadata['class_name_to_index'])
    cm           = confusion_matrix_from_predictions(y_true, y_pred, num_classes)
    class_names  = [None] * num_classes
    for name, idx in metadata['class_name_to_index'].items():
        class_names[int(idx)] = name

    plot_confusion_matrix(cm, class_names, output_dir / 'test_confusion_matrix.png',
                          state_name=metadata['state_name'])

    print(f'\n{"="*70}')
    print(f'Meilleure val  : {metrics_payload["best_val"]}')
    print(f'Test au meilleur checkpoint : {metrics_payload["test_at_best_val"]}')
    print(f'Checkpoint : {args.checkpoint_path}')
    print(f'Métriques  : {metrics_path}')
    print(f'Courbes    : {output_dir / "training_curves.png"}')
    print(f'Conf. mat. : {output_dir / "test_confusion_matrix.png"}')


if __name__ == '__main__':
    main()

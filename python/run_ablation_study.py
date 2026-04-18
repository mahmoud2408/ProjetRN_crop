"""
run_ablation_study.py
=====================
Étude d'ablation Partie 2 — Impact des covariables environnementales sur MCTNet.

5 configurations — sélection de 8 covariables (3 clim + 3 sol + 2 topo) :
  baseline   | Sentinel-2 uniquement                 (input_dim = 10)
  climate    | S2 + Climat    (3 var.)               (input_dim = 13)
  soil       | S2 + Sol       (3 var.)               (input_dim = 13)
  topography | S2 + Topographie (2 var.)             (input_dim = 12)
  all        | S2 + tout      (8 var.)               (input_dim = 18)

Stratégie de fusion : Early Fusion — les covariables statiques sont répétées
sur les 36 pas de temps et concaténées aux 10 bandes Sentinel-2 avant d'entrer
dans MCTNet. L'architecture interne du modèle n'est pas modifiée.

Usage CLI :
    python run_ablation_study.py \
        --processed-env-dir /path/processed_env \
        --output-dir        /path/ablation_runs \
        --states arkansas california \
        --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from mctnet_model import build_mctnet
from run_mctnet_training import (
    classification_metrics,
    confusion_matrix_from_predictions,
    plot_confusion_matrix,
    run_epoch,
    save_checkpoint,
    set_seed,
)

# ---------------------------------------------------------------------------
# Noms des configurations (doit correspondre aux clés du JSON metadata)
# ---------------------------------------------------------------------------
ABLATION_CONFIG_NAMES: List[str] = [
    'baseline',
    'climate',
    'soil',
    'topography',
    'all',
]


# ---------------------------------------------------------------------------
# Dataset avec early fusion des covariables environnementales
# ---------------------------------------------------------------------------

class CropAblationDataset(Dataset):
    """
    Dataset PyTorch pour l'ablation study.

    Si env_indices est vide  → baseline : x garde sa forme [36, 10].
    Sinon                    → x devient [36, 10 + len(env_indices)]
    via répétition des covariables statiques sur chaque pas de temps.
    """

    def __init__(
        self,
        bundle:      Dict[str, np.ndarray],
        split:       str,
        env_indices: List[int],
    ) -> None:
        self.x          = torch.from_numpy(bundle[f'x_{split}']).float()
        self.valid_mask = torch.from_numpy(bundle[f'valid_mask_{split}']).float()
        self.y          = torch.from_numpy(bundle[f'y_{split}']).long()
        self.env        = torch.from_numpy(bundle[f'env_{split}']).float()
        self.env_indices = env_indices

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        x_seq = self.x[idx]                          # [36, 10]

        if self.env_indices:
            env_vals = self.env[idx][self.env_indices]               # [E]
            env_seq  = env_vals.unsqueeze(0).expand(x_seq.shape[0], -1)  # [36, E]
            x_seq    = torch.cat([x_seq, env_seq], dim=1)           # [36, 10+E]

        return {
            'x':          x_seq,
            'valid_mask': self.valid_mask[idx],
            'y':          self.y[idx],
        }


def build_ablation_dataloaders(
    bundle:      Dict[str, np.ndarray],
    env_indices: List[int],
    batch_size:  int,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split in ['train', 'val', 'test']:
        ds = CropAblationDataset(bundle, split, env_indices)
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == 'train'),
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )
    return loaders


def get_env_indices(metadata: Dict, config_name: str) -> List[int]:
    """Renvoie les indices env à partir du fichier JSON de métadonnées."""
    ablation_configs = metadata.get('ablation_configs', {})
    if config_name in ablation_configs:
        return list(ablation_configs[config_name])
    env_meta = metadata.get('environmental_covariates', {})
    if config_name == 'baseline':
        return []
    if config_name == 'all':
        return list(range(len(env_meta.get('all_columns', []))))
    group_indices = env_meta.get('group_to_indices', {})
    return list(group_indices.get(config_name, []))


# ---------------------------------------------------------------------------
# Entraînement d'une configuration
# ---------------------------------------------------------------------------

def train_one_config(
    config_name:  str,
    env_indices:  List[int],
    bundle:       Dict[str, np.ndarray],
    metadata:     Dict,
    args:         SimpleNamespace,
    output_dir:   Path,
) -> Dict[str, float]:
    """
    Entraîne MCTNet pour une configuration d'ablation et retourne les
    meilleures métriques sur le test set.
    """
    input_dim   = 10 + len(env_indices)
    num_classes = len(metadata['class_name_to_index'])
    device      = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    loaders = build_ablation_dataloaders(bundle, env_indices, args.batch_size, args.num_workers)
    model   = build_mctnet(
        num_classes            = num_classes,
        input_dim              = input_dim,
        seq_len                = 36,
        n_stages               = args.n_stages,
        n_heads                = args.n_heads,
        kernel_size            = args.kernel_size,
        dropout                = args.dropout,
        use_alpe               = not getattr(args, 'no_alpe',  False),
        use_missing_mask       = not getattr(args, 'no_mask',  False),
        use_cnn_branch         = not getattr(args, 'no_cnn',   False),
        use_transformer_branch = not getattr(args, 'no_trans', False),
    ).to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_val_kappa     = float('-inf')
    best_test_metrics: Dict[str, float] = {}
    patience_counter   = 0
    history: Dict[str, Dict[str, float]] = {}
    ckpt_path          = output_dir / f'best_{config_name}.pt'

    print(f'\n  Config: {config_name.upper():12s} | '
          f'input_dim={input_dim} | '
          f'#params={model.count_parameters():,} | '
          f'device={device}')

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val_m   = run_epoch(model, loaders['val'],   criterion, device)
        test_m  = run_epoch(model, loaders['test'],  criterion, device)
        scheduler.step()

        history[f'epoch_{epoch:03d}'] = {
            'train_loss': train_m['loss'], 'train_oa': train_m['oa'],
            'train_kappa': train_m['kappa'],
            'val_loss':   val_m['loss'],   'val_oa':   val_m['oa'],
            'val_kappa':  val_m['kappa'],
            'test_oa':    test_m['oa'],    'test_kappa': test_m['kappa'],
            'test_macro_f1': test_m['macro_f1'],
        }

        if val_m['kappa'] > best_val_kappa:
            best_val_kappa    = val_m['kappa']
            best_test_metrics = dict(test_m)
            patience_counter  = 0
            # Checkpoint léger (config_name stocké pour recharge facile)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': input_dim, 'num_classes': num_classes,
                    'seq_len': 36, 'n_stages': args.n_stages,
                    'n_heads': args.n_heads, 'kernel_size': args.kernel_size,
                    'dropout': args.dropout,
                },
                'ablation_config': config_name,
                'env_indices': env_indices,
                'best_val_kappa': best_val_kappa,
            }, ckpt_path)
        else:
            patience_counter += 1

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(f'    Epoch {epoch:03d}/{args.epochs} | '
                  f'Val OA={val_m["oa"]:.4f} κ={val_m["kappa"]:.4f} | '
                  f'Test OA={test_m["oa"]:.4f} F1={test_m["macro_f1"]:.4f}')

        if patience_counter >= args.early_stopping_patience:
            print(f'    Early stopping à l\'époque {epoch}.')
            break

    # Sauvegarde de l'historique
    (output_dir / f'history_{config_name}.json').write_text(
        json.dumps(history, indent=2), encoding='utf-8'
    )
    print(f'  => Test : OA={best_test_metrics.get("oa",0):.4f} '
          f'F1={best_test_metrics.get("macro_f1",0):.4f} '
          f'κ={best_test_metrics.get("kappa",0):.4f}')
    return best_test_metrics


# ---------------------------------------------------------------------------
# Matrice de confusion pour la meilleure config
# ---------------------------------------------------------------------------

def compute_and_save_confusion_matrix(
    bundle:       Dict[str, np.ndarray],
    metadata:     Dict,
    ckpt_path:    Path,
    output_png:   Path,
    batch_size:   int = 64,
    force_cpu:    bool = False,
) -> np.ndarray:
    from mctnet_model import MCTNet, MCTNetConfig
    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    ckpt   = torch.load(ckpt_path, map_location=device)
    cfg    = ckpt['model_config']

    model = build_mctnet(
        num_classes = cfg['num_classes'],
        input_dim   = cfg['input_dim'],
        seq_len     = cfg.get('seq_len', 36),
        n_stages    = cfg.get('n_stages', 3),
        n_heads     = cfg.get('n_heads', 5),
        kernel_size = cfg.get('kernel_size', 3),
        dropout     = cfg.get('dropout', 0.1),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    env_indices = ckpt.get('env_indices', [])
    loader = build_ablation_dataloaders(bundle, env_indices, batch_size)['test']

    yt_l, yp_l = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['x'].to(device), batch['valid_mask'].to(device))
            yt_l.append(batch['y'].numpy())
            yp_l.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(yt_l)
    y_pred = np.concatenate(yp_l)
    n_cls  = len(metadata['class_name_to_index'])
    cm     = confusion_matrix_from_predictions(y_true, y_pred, n_cls)

    class_names = [None] * n_cls
    for name, idx in metadata['class_name_to_index'].items():
        class_names[int(idx)] = name

    plot_confusion_matrix(cm, class_names, output_png,
                          state_name=f'{metadata["state_name"]} – {ckpt["ablation_config"]}')
    return cm


# ---------------------------------------------------------------------------
# Graphiques comparatifs (Partie D)
# ---------------------------------------------------------------------------

def plot_ablation_barplot(
    results:    Dict[str, Dict[str, Dict[str, float]]],
    metric:     str,
    output_png: Path,
) -> None:
    """
    results[state][config][metric] → barplot côte à côte AR / CA.
    """
    states  = list(results.keys())
    configs = list(next(iter(results.values())).keys())
    x       = np.arange(len(configs))
    width   = 0.8 / max(len(states), 1)
    colors  = ['#2563eb', '#16a34a', '#dc2626', '#d97706']

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.8), 6))
    for i, state in enumerate(states):
        vals   = [results[state].get(cfg, {}).get(metric, 0) for cfg in configs]
        offset = (i - (len(states) - 1) / 2) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=state.title(), color=colors[i % len(colors)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    baseline_vals = [results[s].get('baseline', {}).get(metric, None) for s in states]
    ax.axhline(np.nanmean([v for v in baseline_vals if v]),
               color='grey', linestyle='--', linewidth=1, alpha=0.7,
               label='Baseline (moyenne)')

    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(f'Ablation Study — {metric.upper()} par configuration', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in configs], fontsize=10)
    ymin = max(0, min(v for s in states for v in [results[s].get(c, {}).get(metric, 1) for c in configs]) - 0.05)
    ax.set_ylim(ymin, 1.01)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Barplot sauvegardé → {output_png}')


def plot_all_metrics_grid(
    results:    Dict[str, Dict[str, Dict[str, float]]],
    output_png: Path,
) -> None:
    """Grille 1×3 comparant OA, Macro F1 et Kappa pour les deux états."""
    metrics = [('oa', 'Overall Accuracy (OA)'),
               ('macro_f1', 'Macro F1'),
               ('kappa', 'Kappa')]
    states  = list(results.keys())
    configs = list(next(iter(results.values())).keys())
    x       = np.arange(len(configs))
    colors  = ['#2563eb', '#16a34a']
    width   = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, (metric, title) in zip(axes, metrics):
        for i, state in enumerate(states):
            vals   = [results[state].get(cfg, {}).get(metric, 0) for cfg in configs]
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, label=state.title(), color=colors[i])
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in configs], fontsize=9, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(fontsize=9)

    fig.suptitle('Ablation Study — Résumé des métriques', fontsize=13)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  Grille des métriques → {output_png}')


# ---------------------------------------------------------------------------
# Tableau récapitulatif
# ---------------------------------------------------------------------------

def save_summary_table(
    results:    Dict[str, Dict[str, Dict[str, float]]],
    output_csv: Path,
    output_json: Path,
) -> None:
    rows = []
    for state, configs in results.items():
        for config, metrics in configs.items():
            rows.append({
                'state':    state,
                'config':   config,
                'oa':       round(metrics.get('oa',       0), 4),
                'macro_f1': round(metrics.get('macro_f1', 0), 4),
                'kappa':    round(metrics.get('kappa',    0), 4),
                'loss':     round(metrics.get('loss',     0), 4),
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['state', 'config', 'oa', 'macro_f1', 'kappa', 'loss']
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_json.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nTableau CSV → {output_csv}')
    print(f'JSON complet → {output_json}')

    # Affichage console
    print(f'\n{"État":12s} {"Config":12s} {"OA":>6s} {"F1":>6s} {"Kappa":>6s}')
    print('─' * 48)
    for row in rows:
        print(f'{row["state"]:12s} {row["config"]:12s} '
              f'{row["oa"]:6.4f} {row["macro_f1"]:6.4f} {row["kappa"]:6.4f}')


# ---------------------------------------------------------------------------
# Boucle principale multi-états / multi-configs
# ---------------------------------------------------------------------------

def run_ablation_experiment(
    processed_env_dir: Path,
    output_dir:        Path,
    states:            List[str],
    args:              SimpleNamespace,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Lance l'ablation complète sur tous les états et toutes les configs."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for state_slug in states:
        npz_path  = processed_env_dir / f'{state_slug}_mctnet_env_dataset.npz'
        json_path = processed_env_dir / f'{state_slug}_mctnet_env_dataset.json'

        if not npz_path.exists():
            print(f'[WARN] Dataset manquant : {npz_path} — ignoré.')
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            bundle = {k: data[k] for k in data.files}
        metadata = json.loads(json_path.read_text(encoding='utf-8'))

        state_output = output_dir / state_slug
        state_output.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*70}')
        print(f'État : {metadata["state_name"]}  |  '
              f'{metadata["num_classes"]} classes : '
              f'{list(metadata["class_name_to_index"].keys())}')
        print(f'Env shape : [N, {metadata["environmental_covariates"]["n_env_features"]}]')

        results[state_slug] = {}
        configs_to_run = getattr(args, 'configs', ABLATION_CONFIG_NAMES)

        for config_name in configs_to_run:
            env_indices = get_env_indices(metadata, config_name)
            set_seed(args.seed)

            test_metrics = train_one_config(
                config_name  = config_name,
                env_indices  = env_indices,
                bundle       = bundle,
                metadata     = metadata,
                args         = args,
                output_dir   = state_output,
            )
            results[state_slug][config_name] = test_metrics

        # Matrice de confusion de la meilleure config (celle avec le kappa max)
        best_config = max(
            results[state_slug],
            key=lambda c: results[state_slug][c].get('kappa', 0)
        )
        best_ckpt = state_output / f'best_{best_config}.pt'
        if best_ckpt.exists():
            compute_and_save_confusion_matrix(
                bundle      = bundle,
                metadata    = metadata,
                ckpt_path   = best_ckpt,
                output_png  = state_output / f'confusion_matrix_best_{best_config}.png',
                force_cpu   = getattr(args, 'cpu', False),
            )

    # ── Graphiques et tableaux de synthèse ────────────────────────────────────
    for metric in ['oa', 'macro_f1', 'kappa']:
        plot_ablation_barplot(
            results,
            metric     = metric,
            output_png = output_dir / f'ablation_{metric}_barplot.png',
        )
    plot_all_metrics_grid(results, output_dir / 'ablation_metrics_grid.png')
    save_summary_table(
        results,
        output_csv  = output_dir / 'ablation_summary.csv',
        output_json = output_dir / 'ablation_results.json',
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Ablation study MCTNet — covariables environnementales.')
    p.add_argument('--processed-env-dir', required=True,
                   help='Dossier contenant les *_mctnet_env_dataset.npz/.json')
    p.add_argument('--output-dir', required=True,
                   help='Dossier de sortie des runs d\'ablation.')
    p.add_argument('--states', nargs='+', default=['arkansas', 'california'])
    p.add_argument('--configs', nargs='+', default=ABLATION_CONFIG_NAMES,
                   help='Sous-ensemble de configs à lancer.')
    p.add_argument('--epochs',        type=int,   default=50)
    p.add_argument('--batch-size',    type=int,   default=32)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay',  type=float, default=1e-4)
    p.add_argument('--dropout',       type=float, default=0.1)
    p.add_argument('--n-stages',      type=int,   default=3, choices=[1, 2, 3])
    p.add_argument('--n-heads',       type=int,   default=5)
    p.add_argument('--kernel-size',   type=int,   default=3)
    p.add_argument('--seed',          type=int,   default=2021)
    p.add_argument('--num-workers',   type=int,   default=0)
    p.add_argument('--early-stopping-patience', type=int, default=10)
    p.add_argument('--cpu',      action='store_true')
    p.add_argument('--no-alpe',  action='store_true')
    p.add_argument('--no-mask',  action='store_true')
    p.add_argument('--no-cnn',   action='store_true')
    p.add_argument('--no-trans', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_ablation_experiment(
        processed_env_dir = Path(args.processed_env_dir),
        output_dir        = Path(args.output_dir),
        states            = args.states,
        args              = args,
    )


if __name__ == '__main__':
    main()
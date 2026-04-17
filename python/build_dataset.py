import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Configuration des classes et bandes selon l'étude
# ---------------------------------------------------------
TARGET_CLASSES_PER_STATE: Dict[str, List[str]] = {
    'california': ['pistachio', 'almond', 'alfalfa', 'rice', 'grapes', 'others'],
    'arkansas': ['corn', 'cotton', 'rice', 'soybeans', 'others'],
}

CANONICAL_NAME_MAP: Dict[str, str] = {
    'pistachio': 'pistachio', 'pistachios': 'pistachio',
    'almond': 'almond', 'almonds': 'almond',
    'alfalfa': 'alfalfa',
    'rice': 'rice',
    'grape': 'grapes', 'grapes': 'grapes',
    'corn': 'corn',
    'cotton': 'cotton',
    'soybean': 'soybeans', 'soybeans': 'soybeans',
    'other': 'others', 'others': 'others',
}

S2_BANDS: List[str] = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
N_TIME_STEPS: int = 36
TRAIN_VAL_PER_CLASS: int = 300
TRAIN_FRACTION_WITHIN_TRAIN_VAL: float = 0.8


def ordered_feature_columns() -> List[str]:
    return [f'{band}_t{t:02d}' for t in range(1, N_TIME_STEPS + 1) for band in S2_BANDS]


def ordered_valid_columns() -> List[str]:
    return [f'valid_t{t:02d}' for t in range(1, N_TIME_STEPS + 1)]


def detect_state(dataframe: pd.DataFrame) -> str:
    raw = str(dataframe['state_name'].iloc[0]).lower().strip()
    if 'arkan' in raw or 'aran' in raw:
        return 'arkansas'
    if 'californ' in raw:
        return 'california'
    return raw.replace(' ', '_')


def canonicalize_label(raw_name: str, targets: List[str]) -> str:
    """Standardise le nom de la classe, gère les pluriels et assigne à 'others' si non ciblé."""
    normalized = str(raw_name).lower().strip()
    canonical = CANONICAL_NAME_MAP.get(normalized, 'others')
    if canonical in targets:
        return canonical
    return 'others'


def enforce_target_classes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Applique le filtrage strict des classes selon l'État."""
    state_key = detect_state(dataframe)
    targets = TARGET_CLASSES_PER_STATE.get(state_key)
    if targets is None:
        return dataframe

    df = dataframe.copy()
    df['label_final_name'] = df['label_final_name'].apply(
        lambda n: canonicalize_label(n, targets)
    )
    # Assigne un code temporaire pour 'others'
    df.loc[df['label_final_name'] == 'others', 'label_final_code'] = 999
    return df


def make_class_order(dataframe: pd.DataFrame, state_key: str) -> List[str]:
    """Garantit que 'others' possède toujours l'index le plus élevé."""
    targets = TARGET_CLASSES_PER_STATE.get(state_key, [])
    present = set(dataframe['label_final_name'].unique())
    ordered = [c for c in targets if c != 'others' and c in present]
    if 'others' in present:
        ordered.append('others')
    return ordered


def make_label_mapping(dataframe: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, int]]:
    state_key = detect_state(dataframe)
    class_order = make_class_order(dataframe, state_key)

    # name_to_idx génère les index finaux (0 à N-1)
    name_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(class_order)}

    code_to_idx: Dict[int, int] = {}
    for name, idx in name_to_idx.items():
        codes = dataframe[dataframe['label_final_name'] == name]['label_final_code'].unique()
        for code in codes:
            code_to_idx[int(code)] = idx

    return name_to_idx, code_to_idx


def split_like_paper(dataframe: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    """Sépare les données (Train: 240, Val: 60, Test: Reste) par classe[cite: 384]."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for class_name in dataframe['label_final_name'].unique():
        class_df = dataframe[dataframe['label_final_name'] == class_name]
        indices = class_df.index.to_numpy()
        rng.shuffle(indices)

        n_total = len(indices)
        n_train_val = min(TRAIN_VAL_PER_CLASS, n_total - 1)
        n_train = int(n_train_val * TRAIN_FRACTION_WITHIN_TRAIN_VAL)

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train_val])
        test_idx.extend(indices[n_train_val:])

    return {
        'train': dataframe.loc[train_idx],
        'val': dataframe.loc[val_idx],
        'test': dataframe.loc[test_idx],
    }


def pack_split(
        df_split: pd.DataFrame,
        feature_cols: List[str],
        valid_cols: List[str],
        code_to_idx: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transforme les DataFrames en tenseurs Numpy avec les bonnes dimensions."""
    x = df_split[feature_cols].to_numpy(dtype='float32').reshape(
        -1, N_TIME_STEPS, len(S2_BANDS)
    )
    v = df_split[valid_cols].to_numpy(dtype='float32')
    # Création du masque des valeurs manquantes (1.0 = manquant, 0.0 = valide)
    m = np.repeat((1.0 - v)[:, :, np.newaxis], len(S2_BANDS), axis=2).astype('float32')
    y = np.array(
        [code_to_idx[int(c)] for c in df_split['label_final_code']], dtype='int64'
    )
    return x, v, m, y


def save_bundle(
        bundle: Dict[str, np.ndarray],
        metadata: Dict,
        npz_path: Path,
        json_path: Path,
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **bundle)
    json_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def build_dataset_bundle(
        csv_path: Path,
        normalize_reflectance: bool = True,
        reflectance_scale: float = 10000.0,
        split_seed: int = 2021,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    df = pd.read_csv(csv_path)
    df = df.sort_values('sample_id').reset_index(drop=True)
    df = enforce_target_classes(df)

    name_to_idx, code_to_idx = make_label_mapping(df)
    splits = split_like_paper(df, split_seed)

    f_cols = ordered_feature_columns()
    v_cols = ordered_valid_columns()

    bundle: Dict[str, np.ndarray] = {}
    split_counts: Dict[str, Dict[str, int]] = {}

    for split_name, split_df in splits.items():
        x, v, m, y = pack_split(split_df, f_cols, v_cols, code_to_idx)
        if normalize_reflectance:
            x = x / reflectance_scale
        bundle[f'x_{split_name}'] = x
        bundle[f'valid_mask_{split_name}'] = v
        bundle[f'missing_mask_{split_name}'] = m
        bundle[f'y_{split_name}'] = y
        split_counts[split_name] = (
            split_df['label_final_name'].value_counts().sort_index().to_dict()
        )

    state_name = str(df['state_name'].iloc[0])
    metadata = {
        'state_name': state_name,
        'source_csv': str(csv_path),
        'n_samples_total': int(len(df)),
        'class_name_to_index': name_to_idx,
        'num_classes': len(name_to_idx),
        'split_counts': split_counts,
        'samples_per_split': {k: len(v) for k, v in splits.items()},
        'paper_settings': {
            'train_val_per_class': TRAIN_VAL_PER_CLASS,
            'train_fraction_within_train_val': TRAIN_FRACTION_WITHIN_TRAIN_VAL,
            'normalize_reflectance': normalize_reflectance,
            'reflectance_scale': reflectance_scale if normalize_reflectance else None,
            'split_seed': split_seed,
        },
    }
    return bundle, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare les données GEE pour MCTNet.")
    parser.add_argument('--input-csv', nargs='+', required=True, help="Chemins vers les fichiers CSV.")
    parser.add_argument('--output-dir', required=True, help="Dossier de sortie pour les .npz et .json.")
    parser.add_argument('--split-seed', type=int, default=2021)
    parser.add_argument('--reflectance-scale', type=float, default=10000.0)
    parser.add_argument('--disable-normalize-reflectance', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    normalize = not args.disable_normalize_reflectance

    for csv_file in args.input_csv:
        csv_path = Path(csv_file)
        bundle, metadata = build_dataset_bundle(
            csv_path=csv_path,
            normalize_reflectance=normalize,
            reflectance_scale=args.reflectance_scale,
            split_seed=args.split_seed,
        )
        slug = detect_state(pd.DataFrame({'state_name': [metadata['state_name']]}))
        npz_path = out_dir / f'{slug}_mctnet_dataset.npz'
        json_path = out_dir / f'{slug}_mctnet_dataset.json'

        save_bundle(bundle, metadata, npz_path, json_path)

        print(f"\n[{metadata['state_name']}] Traitement terminé.")
        print(f"Classes ({metadata['num_classes']}) : {list(metadata['class_name_to_index'].keys())}")
        print(f"  -> Sauvegardé dans : {npz_path}")
        for split, counts in metadata['split_counts'].items():
            print(f"  {split.capitalize()}: {counts}")


if __name__ == '__main__':
    main()
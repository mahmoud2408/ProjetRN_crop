"""
Build the MCTNet-ready dataset from Google Earth Engine CSV exports.

Paper-driven settings:
  - 36 temporal observations
  - 10 Sentinel-2 spectral bands
  - Missing values kept as 0
  - Randomly sampled 10,000 points per state
  - Classes under 5% merged to "others"
  - 300 samples per final class for train+val, split 8:2

Implementation choices:
  - Reflectance is optionally scaled from Sentinel-2 SR integers to [0, 1]
    by dividing by 10000.
  - The paper does not publish a random seed; this script uses one so the
    train/validation split is reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

S2_BANDS: List[str] = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
N_TIME_STEPS = 36
TRAIN_VAL_PER_CLASS = 300
TRAIN_FRACTION_WITHIN_TRAIN_VAL = 0.8


def ordered_feature_columns() -> List[str]:
    columns: List[str] = []
    for time_idx in range(1, N_TIME_STEPS + 1):
        suffix = f't{time_idx:02d}'
        for band in S2_BANDS:
            columns.append(f'{band}_{suffix}')
    return columns


def ordered_valid_columns() -> List[str]:
    return [f'valid_t{time_idx:02d}' for time_idx in range(1, N_TIME_STEPS + 1)]


def make_class_order(dataframe: pd.DataFrame) -> List[str]:
    counts = dataframe['label_final_name'].value_counts()
    ordered = [name for name in counts.index.tolist() if name != 'others']
    if 'others' in counts.index:
        ordered.append('others')
    return ordered


def make_label_mapping(dataframe: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    class_order = make_class_order(dataframe)
    name_to_index = {class_name: idx for idx, class_name in enumerate(class_order)}

    code_lookup = (
        dataframe[['label_final_name', 'label_final_code']]
        .drop_duplicates()
        .sort_values(['label_final_name', 'label_final_code'])
    )
    name_to_original_code = {
        str(row.label_final_name): int(row.label_final_code)
        for row in code_lookup.itertuples(index=False)
    }

    return name_to_index, name_to_original_code


def reshape_features(
    dataframe: pd.DataFrame,
    normalize_reflectance: bool,
    reflectance_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_columns = ordered_feature_columns()
    valid_columns = ordered_valid_columns()

    x = dataframe[feature_columns].to_numpy(dtype=np.float32)
    x = x.reshape(len(dataframe), N_TIME_STEPS, len(S2_BANDS))

    if normalize_reflectance:
        x /= reflectance_scale

    valid_mask = dataframe[valid_columns].to_numpy(dtype=np.uint8)
    missing_mask = (1 - valid_mask).astype(np.uint8)
    missing_mask = np.repeat(missing_mask[:, :, None], len(S2_BANDS), axis=2)

    return x, valid_mask, missing_mask


def split_like_paper(
    dataframe: pd.DataFrame,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_frames: List[pd.DataFrame] = []
    val_frames: List[pd.DataFrame] = []
    test_frames: List[pd.DataFrame] = []

    train_count = int(TRAIN_VAL_PER_CLASS * TRAIN_FRACTION_WITHIN_TRAIN_VAL)
    val_count = TRAIN_VAL_PER_CLASS - train_count

    for class_name, class_frame in dataframe.groupby('label_final_name', sort=False):
        class_frame = class_frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        class_frame = class_frame.reset_index(drop=True)

        if len(class_frame) < TRAIN_VAL_PER_CLASS:
            raise ValueError(
                f'Classe "{class_name}" avec {len(class_frame)} echantillons, '
                f'impossible d appliquer la separation du papier ({TRAIN_VAL_PER_CLASS}).'
            )

        train_val_frame = class_frame.iloc[:TRAIN_VAL_PER_CLASS].copy()
        test_frame = class_frame.iloc[TRAIN_VAL_PER_CLASS:].copy()
        train_frame = train_val_frame.iloc[:train_count].copy()
        val_frame = train_val_frame.iloc[train_count:train_count + val_count].copy()

        train_frames.append(train_frame)
        val_frames.append(val_frame)
        test_frames.append(test_frame)

    split_frames = {
        'train': pd.concat(train_frames, axis=0, ignore_index=True),
        'val': pd.concat(val_frames, axis=0, ignore_index=True),
        'test': pd.concat(test_frames, axis=0, ignore_index=True),
    }

    for split_name, split_frame in split_frames.items():
        split_frames[split_name] = split_frame.sort_values('sample_id').reset_index(drop=True)

    return split_frames


def encode_targets(
    dataframe: pd.DataFrame,
    name_to_index: Dict[str, int],
) -> np.ndarray:
    return dataframe['label_final_name'].map(name_to_index).to_numpy(dtype=np.int64)


def pack_split(
    dataframe: pd.DataFrame,
    name_to_index: Dict[str, int],
    normalize_reflectance: bool,
    reflectance_scale: float,
) -> Dict[str, np.ndarray]:
    x, valid_mask, missing_mask = reshape_features(
        dataframe=dataframe,
        normalize_reflectance=normalize_reflectance,
        reflectance_scale=reflectance_scale,
    )

    return {
        'x': x.astype(np.float32),
        'valid_mask': valid_mask.astype(np.uint8),
        'missing_mask': missing_mask.astype(np.uint8),
        'y': encode_targets(dataframe, name_to_index).astype(np.int64),
        'label_final_code': dataframe['label_final_code'].to_numpy(dtype=np.int64),
        'longitude': dataframe['longitude'].to_numpy(dtype=np.float32),
        'latitude': dataframe['latitude'].to_numpy(dtype=np.float32),
        'sample_id': dataframe['sample_id'].astype(str).to_numpy(),
    }


def build_dataset_bundle(
    csv_path: Path,
    normalize_reflectance: bool,
    reflectance_scale: float,
    split_seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.sort_values('sample_id').reset_index(drop=True)

    required_columns = set(
        ['sample_id', 'state_name', 'label_final_name', 'label_final_code'] +
        ordered_feature_columns() +
        ordered_valid_columns()
    )
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f'Colonnes manquantes dans {csv_path.name}: {sorted(missing_columns)}')

    name_to_index, name_to_original_code = make_label_mapping(dataframe)
    split_frames = split_like_paper(dataframe, seed=split_seed)

    bundle: Dict[str, np.ndarray] = {}
    split_counts: Dict[str, Dict[str, int]] = {}

    for split_name, split_frame in split_frames.items():
        packed = pack_split(
            dataframe=split_frame,
            name_to_index=name_to_index,
            normalize_reflectance=normalize_reflectance,
            reflectance_scale=reflectance_scale,
        )
        for key, value in packed.items():
            bundle[f'{key}_{split_name}'] = value
        split_counts[split_name] = split_frame['label_final_name'].value_counts().sort_index().to_dict()

    state_name = str(dataframe['state_name'].iloc[0])
    metadata: Dict[str, object] = {
        'state_name': state_name,
        'source_csv': str(csv_path),
        'n_samples_total': int(len(dataframe)),
        's2_bands': S2_BANDS,
        'n_time_steps': N_TIME_STEPS,
        'feature_shape_per_sample': [N_TIME_STEPS, len(S2_BANDS)],
        'class_name_to_index': name_to_index,
        'class_name_to_original_code': name_to_original_code,
        'split_counts': split_counts,
        'paper_settings': {
            'train_val_per_class': TRAIN_VAL_PER_CLASS,
            'train_fraction_within_train_val': TRAIN_FRACTION_WITHIN_TRAIN_VAL,
            'missing_values_marked_with_zero': True,
            'input_1_shape': [N_TIME_STEPS, len(S2_BANDS)],
            'input_2_shape': [N_TIME_STEPS],
        },
        'implementation_choices': {
            'normalize_reflectance': normalize_reflectance,
            'reflectance_scale': reflectance_scale if normalize_reflectance else None,
            'split_seed': split_seed,
        },
    }

    return bundle, metadata


def save_bundle(
    bundle: Dict[str, np.ndarray],
    metadata: Dict[str, object],
    output_npz: Path,
    output_json: Path,
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **bundle)
    output_json.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert GEE CSV exports into MCTNet-ready NumPy datasets.'
    )
    parser.add_argument(
        '--input-csv',
        nargs='+',
        required=True,
        help='One or more CSV files exported from Google Earth Engine.',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for .npz and .json files.',
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=2021,
        help='Reproducible seed for the train/val split.',
    )
    parser.add_argument(
        '--reflectance-scale',
        type=float,
        default=10000.0,
        help='Sentinel-2 SR scale factor used when normalize-reflectance is enabled.',
    )
    parser.add_argument(
        '--disable-normalize-reflectance',
        action='store_true',
        help='Keep Sentinel-2 values exactly as exported by GEE.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    normalize_reflectance = not args.disable_normalize_reflectance

    for input_csv in args.input_csv:
        csv_path = Path(input_csv)
        bundle, metadata = build_dataset_bundle(
            csv_path=csv_path,
            normalize_reflectance=normalize_reflectance,
            reflectance_scale=args.reflectance_scale,
            split_seed=args.split_seed,
        )

        state_slug = metadata['state_name'].lower().replace(' ', '_')
        output_npz = output_dir / f'{state_slug}_mctnet_dataset.npz'
        output_json = output_dir / f'{state_slug}_mctnet_dataset.json'
        save_bundle(bundle, metadata, output_npz, output_json)

        print(f'[{metadata["state_name"]}] dataset ecrit : {output_npz}')
        print(f'[{metadata["state_name"]}] metadonnees : {output_json}')


if __name__ == '__main__':
    main()

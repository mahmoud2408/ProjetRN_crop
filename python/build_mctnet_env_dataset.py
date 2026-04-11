"""
Build MCTNet-ready datasets extended with environmental covariates.

Expected input:
  - CSV exported from gee/mctnet_env_covariates_prep_2021.js

Outputs:
  - .npz bundle with Sentinel-2 tensors + raw environmental covariates
  - .json metadata describing covariate groups
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from build_mctnet_dataset import (
    make_label_mapping,
    ordered_feature_columns,
    ordered_valid_columns,
    pack_split,
    save_bundle,
    split_like_paper,
)

CLIMATE_COLUMNS = [
    'climate_pr_sum_mm',
    'climate_tmmn_mean_c',
    'climate_tmmx_mean_c',
    'climate_aet_sum_mm',
    'climate_pet_sum_mm',
    'climate_vpd_mean_kpa',
]

SOIL_COLUMNS = [
    'soil_clay_0cm_pct',
    'soil_sand_0cm_pct',
    'soil_soc_0cm_gkg',
    'soil_phh2o_0cm',
]

TOPOGRAPHY_COLUMNS = [
    'topo_elevation_m',
    'topo_slope_deg',
    'topo_aspect_sin',
    'topo_aspect_cos',
]

ENV_GROUPS = {
    'climate': CLIMATE_COLUMNS,
    'soil': SOIL_COLUMNS,
    'topography': TOPOGRAPHY_COLUMNS,
}

ALL_ENV_COLUMNS = CLIMATE_COLUMNS + SOIL_COLUMNS + TOPOGRAPHY_COLUMNS


def extract_env_array(dataframe: pd.DataFrame):
    return dataframe[ALL_ENV_COLUMNS].to_numpy(dtype='float32')


def build_env_dataset_bundle(
    csv_path: Path,
    normalize_reflectance: bool,
    reflectance_scale: float,
    split_seed: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.sort_values('sample_id').reset_index(drop=True)

    required_columns = set(
        ['sample_id', 'state_name', 'label_final_name', 'label_final_code']
        + ordered_feature_columns()
        + ordered_valid_columns()
        + ALL_ENV_COLUMNS
    )
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f'Colonnes manquantes dans {csv_path.name}: {sorted(missing_columns)}')

    name_to_index, name_to_original_code = make_label_mapping(dataframe)
    split_frames = split_like_paper(dataframe, seed=split_seed)

    bundle: Dict[str, object] = {}
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

        bundle[f'env_{split_name}'] = extract_env_array(split_frame)
        split_counts[split_name] = split_frame['label_final_name'].value_counts().sort_index().to_dict()

    env_column_to_index = {column_name: idx for idx, column_name in enumerate(ALL_ENV_COLUMNS)}
    env_group_to_indices = {
        group_name: [env_column_to_index[column_name] for column_name in column_names]
        for group_name, column_names in ENV_GROUPS.items()
    }

    state_name = str(dataframe['state_name'].iloc[0])
    metadata: Dict[str, object] = {
        'state_name': state_name,
        'source_csv': str(csv_path),
        'n_samples_total': int(len(dataframe)),
        's2_bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'n_time_steps': 36,
        'feature_shape_per_sample': [36, 10],
        'class_name_to_index': name_to_index,
        'class_name_to_original_code': name_to_original_code,
        'split_counts': split_counts,
        'environmental_covariates': {
            'all_columns': ALL_ENV_COLUMNS,
            'group_to_columns': ENV_GROUPS,
            'column_to_index': env_column_to_index,
            'group_to_indices': env_group_to_indices,
        },
        'paper_settings': {
            'train_val_per_class': 300,
            'train_fraction_within_train_val': 0.8,
            'missing_values_marked_with_zero': True,
            'input_1_shape': [36, 10],
            'input_2_shape': [36],
        },
        'implementation_choices': {
            'normalize_reflectance': normalize_reflectance,
            'reflectance_scale': reflectance_scale if normalize_reflectance else None,
            'split_seed': split_seed,
            'environmental_covariates_are_static_per_point': True,
        },
    }

    return bundle, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert GEE CSV exports with environmental covariates into MCTNet-ready datasets.'
    )
    parser.add_argument('--input-csv', nargs='+', required=True, help='CSV files exported from GEE.')
    parser.add_argument('--output-dir', required=True, help='Output directory for .npz/.json files.')
    parser.add_argument('--split-seed', type=int, default=2021, help='Reproducible split seed.')
    parser.add_argument('--reflectance-scale', type=float, default=10000.0, help='Sentinel-2 SR scale factor.')
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
        bundle, metadata = build_env_dataset_bundle(
            csv_path=csv_path,
            normalize_reflectance=normalize_reflectance,
            reflectance_scale=args.reflectance_scale,
            split_seed=args.split_seed,
        )

        state_slug = metadata['state_name'].lower().replace(' ', '_')
        output_npz = output_dir / f'{state_slug}_mctnet_env_dataset.npz'
        output_json = output_dir / f'{state_slug}_mctnet_env_dataset.json'
        save_bundle(bundle, metadata, output_npz, output_json)

        print(f'[{metadata["state_name"]}] env dataset ecrit : {output_npz}')
        print(f'[{metadata["state_name"]}] env metadonnees : {output_json}')


if __name__ == '__main__':
    main()

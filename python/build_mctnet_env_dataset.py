"""
build_mctnet_env_dataset.py
===========================
Convertit les CSV GEE (Sentinel-2 + covariables environnementales) en
fichiers .npz/.json compatibles avec MCTNet.

Dépend de build_dataset.py pour les utilitaires partagés.

Usage :
    python build_mctnet_env_dataset.py \
        --input-csv mctnet_env_samples_AR_2021.csv mctnet_env_samples_CA_2021.csv \
        --output-dir /path/to/processed_env

Sélection retenue (8 variables sur 14 disponibles dans GEE) :
  Climat (3) : climate_vpd_mean_kpa, climate_pr_sum_mm, climate_tmmn_mean_c
  Sol    (3) : soil_clay_0cm_pct, soil_sand_0cm_pct, soil_phh2o_0cm
  Topo   (2) : topo_elevation_m, topo_slope_deg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Importation depuis le module corrigé (build_dataset.py)
from build_dataset import (
    detect_state,
    enforce_target_classes,
    make_label_mapping,
    ordered_feature_columns,
    ordered_valid_columns,
    pack_split,
    save_bundle,
    split_like_paper,
)

# ---------------------------------------------------------------------------
# Définition des groupes de covariables environnementales
# Les noms correspondent EXACTEMENT aux CONFIG.climateBands / soilBands /
# topographyBands du script GEE mctnet_env_covariates_prep_2021.js
# ---------------------------------------------------------------------------

CLIMATE_COLUMNS: List[str] = [
    'climate_pr_sum_mm',
    'climate_tmmn_mean_c',
    'climate_vpd_mean_kpa',
]

SOIL_COLUMNS: List[str] = [
    'soil_clay_0cm_pct',
    'soil_sand_0cm_pct',
    'soil_phh2o_0cm',
]

TOPOGRAPHY_COLUMNS: List[str] = [
    'topo_elevation_m',
    'topo_slope_deg',
]

ENV_GROUPS: Dict[str, List[str]] = {
    'climate':    CLIMATE_COLUMNS,
    'soil':       SOIL_COLUMNS,
    'topography': TOPOGRAPHY_COLUMNS,
}

ALL_ENV_COLUMNS: List[str] = CLIMATE_COLUMNS + SOIL_COLUMNS + TOPOGRAPHY_COLUMNS

# Noms des configurations d'ablation (Partie 2 de l'énoncé)
ABLATION_CONFIGS: Dict[str, List[str]] = {
    'baseline':    [],
    'climate':     CLIMATE_COLUMNS,
    'soil':        SOIL_COLUMNS,
    'topography':  TOPOGRAPHY_COLUMNS,
    'all':         ALL_ENV_COLUMNS,
}


def extract_env_array(dataframe: pd.DataFrame) -> np.ndarray:
    """Extrait la matrice des covariables env. → shape [N, 8]."""
    return dataframe[ALL_ENV_COLUMNS].to_numpy(dtype='float32')


def build_env_dataset_bundle(
    csv_path:              Path,
    normalize_reflectance: bool  = True,
    reflectance_scale:     float = 10000.0,
    split_seed:            int   = 2021,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Lit un CSV GEE avec covariables environnementales et produit un bundle
    contenant, pour chaque split (train/val/test) :
      - x_{split}           : [N, 36, 10]  séries temporelles Sentinel-2
      - valid_mask_{split}  : [N, 36]      masque de disponibilité
      - missing_mask_{split}: [N, 36, 10]  masque de valeurs manquantes
      - y_{split}           : [N]          étiquettes entières
      - env_{split}         : [N, 8]       covariables environnementales statiques (3 clim + 3 sol + 2 topo)
    """
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.sort_values('sample_id').reset_index(drop=True)

    # ── Vérification des colonnes ─────────────────────────────────────────────
    required_columns = set(
        ['sample_id', 'state_name', 'label_final_name', 'label_final_code']
        + ordered_feature_columns()
        + ordered_valid_columns()
        + ALL_ENV_COLUMNS
    )
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f'Colonnes manquantes dans {csv_path.name} '
            f'({len(missing_columns)} colonnes) : {sorted(missing_columns)[:5]} ...'
        )

    # ── Filtrage des classes cibles ───────────────────────────────────────────
    dataframe = enforce_target_classes(dataframe)

    # make_label_mapping retourne (name_to_idx: Dict[str,int], code_to_idx: Dict[int,int])
    name_to_idx, code_to_idx = make_label_mapping(dataframe)

    # Reconstruction du dictionnaire classe -> code CDL original (avant groupement)
    original_code_map: Dict[str, int] = {}
    for name, idx in name_to_idx.items():
        rows = dataframe[dataframe['label_final_name'] == name]['label_final_code']
        # Pour 'others' il y a plusieurs codes ; on stocke la liste
        codes = rows.unique().tolist()
        original_code_map[name] = codes[0] if len(codes) == 1 else codes

    # ── Splits reproducibles ─────────────────────────────────────────────────
    splits = split_like_paper(dataframe, split_seed)

    f_cols = ordered_feature_columns()
    v_cols = ordered_valid_columns()

    bundle: Dict[str, np.ndarray] = {}
    split_counts: Dict[str, Dict[str, int]] = {}

    for split_name, split_df in splits.items():
        x, v, m, y = pack_split(split_df, f_cols, v_cols, code_to_idx)
        if normalize_reflectance:
            x = x / reflectance_scale

        bundle[f'x_{split_name}']            = x
        bundle[f'valid_mask_{split_name}']   = v
        bundle[f'missing_mask_{split_name}'] = m
        bundle[f'y_{split_name}']            = y
        bundle[f'env_{split_name}']          = extract_env_array(split_df)

        split_counts[split_name] = (
            split_df['label_final_name'].value_counts().sort_index().to_dict()
        )

    # ── Métadonnées ───────────────────────────────────────────────────────────
    col_to_idx = {col: i for i, col in enumerate(ALL_ENV_COLUMNS)}
    group_to_indices = {
        grp: [col_to_idx[c] for c in cols]
        for grp, cols in ENV_GROUPS.items()
    }

    state_name = str(dataframe['state_name'].iloc[0])
    metadata: Dict = {
        'state_name':              state_name,
        'source_csv':              str(csv_path),
        'n_samples_total':         int(len(dataframe)),
        's2_bands':                ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'],
        'n_time_steps':            36,
        'feature_shape_per_sample': [36, 10],
        'class_name_to_index':     name_to_idx,
        'class_name_to_original_code': original_code_map,
        'num_classes':             len(name_to_idx),
        'split_counts':            split_counts,
        'samples_per_split':       {k: len(v) for k, v in splits.items()},
        'environmental_covariates': {
            'all_columns':       ALL_ENV_COLUMNS,
            'n_env_features':    len(ALL_ENV_COLUMNS),
            'group_to_columns':  ENV_GROUPS,
            'column_to_index':   col_to_idx,
            'group_to_indices':  group_to_indices,
        },
        'ablation_configs': {
            name: [col_to_idx[c] for c in cols]
            for name, cols in ABLATION_CONFIGS.items()
        },
        'paper_settings': {
            'train_val_per_class':             300,
            'train_fraction_within_train_val': 0.8,
            'missing_values_marked_with_zero': True,
        },
        'implementation_choices': {
            'normalize_reflectance': normalize_reflectance,
            'reflectance_scale':     reflectance_scale if normalize_reflectance else None,
            'split_seed':            split_seed,
            'env_covariates_are_static_per_point': True,
            'env_shape_per_sample':  [len(ALL_ENV_COLUMNS)],  # [8]
            'fusion_strategy':       'early_fusion_repeat_across_time',
        },
    }

    return bundle, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convertit les CSV GEE (S2 + covariables) en datasets MCTNet .npz/.json.'
    )
    parser.add_argument('--input-csv',  nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split-seed', type=int,   default=2021)
    parser.add_argument('--reflectance-scale', type=float, default=10000.0)
    parser.add_argument('--disable-normalize-reflectance', action='store_true')
    args = parser.parse_args()

    output_dir          = Path(args.output_dir)
    normalize           = not args.disable_normalize_reflectance

    for csv_file in args.input_csv:
        csv_path = Path(csv_file)
        bundle, metadata = build_env_dataset_bundle(
            csv_path              = csv_path,
            normalize_reflectance = normalize,
            reflectance_scale     = args.reflectance_scale,
            split_seed            = args.split_seed,
        )

        slug      = detect_state(pd.DataFrame({'state_name': [metadata['state_name']]}))
        npz_path  = output_dir / f'{slug}_mctnet_env_dataset.npz'
        json_path = output_dir / f'{slug}_mctnet_env_dataset.json'
        save_bundle(bundle, metadata, npz_path, json_path)

        print(f'\n[{metadata["state_name"]}]')
        print(f'  {metadata["num_classes"]} classes : {list(metadata["class_name_to_index"].keys())}')
        for split, counts in metadata['split_counts'].items():
            print(f'  {split:5s}: {counts}')
        print(f'  env shape : [N, {metadata["environmental_covariates"]["n_env_features"]}]')
        print(f'  -> {npz_path}')


if __name__ == '__main__':
    main()
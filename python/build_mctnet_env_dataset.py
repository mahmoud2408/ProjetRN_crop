from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paper-aligned constants
# ---------------------------------------------------------------------------

S2_BANDS: List[str] = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
DEFAULT_SEQUENCE_LENGTH: int = 36
TRAIN_VAL_PER_CLASS: int = 300
TRAIN_FRACTION_WITHIN_TRAIN_VAL: float = 0.8

STATE_DISPLAY_NAMES: Dict[str, str] = {
    'arkansas': 'Arkansas',
    'california': 'California',
}

PAPER_CLASS_ORDER: Dict[str, List[str]] = {
    'arkansas': ['soybeans', 'rice', 'corn', 'cotton', 'others'],
    'california': ['grapes', 'rice', 'alfalfa', 'almond', 'pistachio', 'others'],
}

# The real Arkansas CSV uses numeric labels directly in the data file.
STATE_LABEL_CODE_TO_NAME: Dict[str, Dict[int, str]] = {
    'arkansas': {
        1: 'corn',
        2: 'cotton',
        3: 'rice',
        5: 'soybeans',
        999: 'others',
    },
    'california': {
        3: 'rice',
        36: 'alfalfa',
        69: 'grapes',
        75: 'almond',
        204: 'pistachio',
        999: 'others',
    },
}

ENV_GROUP_PATTERNS: Dict[str, Tuple[str, ...]] = {
    'climate': ('climate', 'temp', 'dewpoint', 'precip', 'rain', 'vpd', 'tmmn', 'tmmx', 'pr'),
    'soil': ('soil', 'clay', 'sand', 'ph', 'texture', 'oc', 'organic'),
    'topography': ('topo', 'elevation', 'slope', 'aspect', 'landform', 'dem'),
}

LABEL_CODE_CANDIDATES: List[str] = ['label_final_code', 'label', 'class', 'crop_label']
LABEL_NAME_CANDIDATES: List[str] = ['label_final_name', 'label_name', 'class_name', 'crop_name']
ID_COLUMN_CANDIDATES: List[str] = ['sample_id', 'system:index', 'point_id', 'id']
LOCAL_CSV_CACHE_DIR = Path('/tmp/mctnet_csv_cache')


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TemporalGroup:
    base_name: str
    indices: List[int]
    columns: List[str]


@dataclass
class DatasetSchema:
    state_slug: str
    state_name: str
    sample_id_column: str
    label_code_column: str
    label_name_column: Optional[str]
    spectral_time_indices: List[int]
    spectral_groups: Dict[str, TemporalGroup]
    valid_mask_group: Optional[TemporalGroup]
    dynamic_env_groups: Dict[str, TemporalGroup]
    dynamic_env_base_order: List[str]
    static_env_columns: List[str]
    group_to_dynamic_columns: Dict[str, List[str]]
    group_to_static_columns: Dict[str, List[str]]

    @property
    def sequence_length(self) -> int:
        return len(self.spectral_time_indices)


# ---------------------------------------------------------------------------
# Resilient CSV reading
# ---------------------------------------------------------------------------

def is_colab_drive_path(path: Path) -> bool:
    return str(path).startswith('/content/drive/')


def stage_csv_locally(csv_path: Path, force_refresh: bool = False) -> Path:
    if not is_colab_drive_path(csv_path):
        return csv_path

    file_stat = csv_path.stat()
    signature = f'{csv_path.resolve()}::{file_stat.st_size}::{int(file_stat.st_mtime)}'
    cache_name = f'{csv_path.stem}_{md5(signature.encode("utf-8")).hexdigest()[:12]}{csv_path.suffix}'
    LOCAL_CSV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_csv_path = LOCAL_CSV_CACHE_DIR / cache_name

    if force_refresh and local_csv_path.exists():
        local_csv_path.unlink()

    if not local_csv_path.exists():
        shutil.copy2(csv_path, local_csv_path)

    return local_csv_path


def read_csv_resilient(csv_path: Path, **kwargs) -> pd.DataFrame:
    errors: List[str] = []
    for attempt_idx in range(3):
        try:
            local_path = stage_csv_locally(csv_path, force_refresh=(attempt_idx > 0))
            return pd.read_csv(local_path, **kwargs)
        except OSError as exc:
            errors.append(f'tentative {attempt_idx + 1}: {exc}')
            time.sleep(2 * (attempt_idx + 1))

    raise OSError(
        f"Impossible de lire le CSV {csv_path} apres 3 tentatives. "
        f"Erreurs: {' | '.join(errors)}"
    )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def detect_state_from_path(csv_path: Path) -> str:
    lowered = csv_path.stem.lower()
    if 'ark' in lowered:
        return 'arkansas'
    if 'calif' in lowered or '_ca' in lowered:
        return 'california'
    return lowered.replace(' ', '_')


def detect_state(dataframe: pd.DataFrame, csv_path: Optional[Path] = None) -> str:
    if 'state_name' in dataframe.columns and dataframe['state_name'].notna().any():
        state_name = str(dataframe['state_name'].dropna().iloc[0]).strip().lower()
        if 'ark' in state_name:
            return 'arkansas'
        if 'calif' in state_name:
            return 'california'
        return state_name.replace(' ', '_')

    if csv_path is not None:
        return detect_state_from_path(csv_path)

    return 'unknown'


def state_display_name(state_slug: str) -> str:
    return STATE_DISPLAY_NAMES.get(state_slug, state_slug.replace('_', ' ').title())


def parse_temporal_column(column_name: str) -> Optional[Tuple[str, int]]:
    match = re.match(r'^(?P<base>.+)_t(?P<index>\d+)$', column_name)
    if not match:
        return None
    return match.group('base'), int(match.group('index'))


def collect_temporal_groups(columns: Iterable[str]) -> Dict[str, Dict[int, str]]:
    groups: Dict[str, Dict[int, str]] = {}
    for column_name in columns:
        parsed = parse_temporal_column(column_name)
        if parsed is None:
            continue
        base_name, time_index = parsed
        groups.setdefault(base_name, {})[time_index] = column_name
    return groups


def temporal_group_from_mapping(base_name: str, mapping: Dict[int, str]) -> TemporalGroup:
    sorted_indices = sorted(mapping)
    return TemporalGroup(
        base_name=base_name,
        indices=sorted_indices,
        columns=[mapping[index] for index in sorted_indices],
    )


def classify_environmental_group(feature_name: str) -> str:
    lowered = feature_name.lower()
    for group_name, tokens in ENV_GROUP_PATTERNS.items():
        if any(token in lowered for token in tokens):
            return group_name
    return 'other'


def select_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    existing = set(columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def ensure_numeric_frame(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    numeric = dataframe.loc[:, columns].apply(pd.to_numeric, errors='coerce')
    return numeric


def ensure_sample_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    sample_id_column = select_first_existing(df.columns, ID_COLUMN_CANDIDATES)
    if sample_id_column is None:
        df['sample_id'] = np.arange(len(df)).astype(str)
    else:
        df['sample_id'] = df[sample_id_column].astype(str)
    return df


# ---------------------------------------------------------------------------
# Label handling
# ---------------------------------------------------------------------------

def guess_labels_csv_path(data_csv_path: Path) -> Optional[Path]:
    candidate_names = [
        data_csv_path.name.replace('mctnet_', 'mctnet_samples_'),
        'mctnet_samples_AR_2021.csv',
        'mctnet_samples_CA_2021.csv',
    ]

    search_roots = [data_csv_path.parent]
    if data_csv_path.parent.parent.exists():
        search_roots.append(data_csv_path.parent.parent)

    for search_root in search_roots:
        for candidate_name in candidate_names:
            candidate = search_root / candidate_name
            if candidate.exists():
                return candidate

    return None


def resolve_labels_csv_path(data_csv_path: Path, labels_csv_path: Optional[Path]) -> Optional[Path]:
    if labels_csv_path is not None:
        return labels_csv_path
    return guess_labels_csv_path(data_csv_path)


def resolve_join_keys(dataframe: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[str, str]:
    best_pair: Optional[Tuple[str, str]] = None
    best_overlap = -1

    for left_key in ID_COLUMN_CANDIDATES:
        if left_key not in dataframe.columns:
            continue
        left_values = dataframe[left_key].astype(str)

        for right_key in ID_COLUMN_CANDIDATES:
            if right_key not in labels_df.columns:
                continue
            right_values = labels_df[right_key].astype(str)
            overlap = int(left_values.isin(set(right_values)).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = (left_key, right_key)

    if best_pair is None or best_overlap <= 0:
        raise ValueError(
            'Impossible de trouver une cle de jointure exploitable entre le CSV de donnees '
            f'et le CSV de labels. Colonnes candidates: {ID_COLUMN_CANDIDATES}'
        )

    return best_pair


def merge_with_labels(
    dataframe: pd.DataFrame,
    data_csv_path: Path,
    labels_csv_path: Optional[Path],
) -> pd.DataFrame:
    if any(column in dataframe.columns for column in LABEL_CODE_CANDIDATES):
        return dataframe.copy()

    resolved_labels_path = resolve_labels_csv_path(data_csv_path, labels_csv_path)
    if resolved_labels_path is None or not resolved_labels_path.exists():
        raise FileNotFoundError(
            'Les labels ne sont ni integres au CSV ni disponibles dans un CSV externe detecte automatiquement.'
        )

    labels_df = read_csv_resilient(resolved_labels_path)
    left_key, right_key = resolve_join_keys(dataframe, labels_df)

    label_code_column = select_first_existing(labels_df.columns, LABEL_CODE_CANDIDATES)
    label_name_column = select_first_existing(labels_df.columns, LABEL_NAME_CANDIDATES)
    if label_code_column is None and label_name_column is None:
        raise ValueError(f'Le CSV de labels {resolved_labels_path} ne contient aucune colonne de label exploitable.')

    merge_columns = [right_key]
    if label_code_column is not None:
        merge_columns.append(label_code_column)
    if label_name_column is not None:
        merge_columns.append(label_name_column)
    if 'state_name' in labels_df.columns and 'state_name' not in dataframe.columns:
        merge_columns.append('state_name')

    labels_subset = labels_df[merge_columns].copy().drop_duplicates(subset=[right_key])
    merged = dataframe.merge(labels_subset, left_on=left_key, right_on=right_key, how='left', suffixes=('', '_label'))

    if right_key != left_key and right_key in merged.columns:
        merged = merged.drop(columns=[right_key])

    missing_all = 0
    if label_code_column is not None:
        missing_all = int(merged[label_code_column].isna().sum())
    elif label_name_column is not None:
        missing_all = int(merged[label_name_column].isna().sum())
    if missing_all > 0:
        raise ValueError(f'{missing_all} lignes n ont pas recu de label apres fusion.')

    return merged


def canonicalize_integrated_labels(
    dataframe: pd.DataFrame,
    state_slug: str,
) -> pd.DataFrame:
    df = dataframe.copy()

    label_code_column = select_first_existing(df.columns, LABEL_CODE_CANDIDATES)
    label_name_column = select_first_existing(df.columns, LABEL_NAME_CANDIDATES)

    if label_code_column is None and label_name_column is None:
        raise ValueError('Aucune colonne de label exploitable n a ete trouvee dans le CSV fusionne.')

    if label_code_column is not None:
        df['label_final_code'] = pd.to_numeric(df[label_code_column], errors='coerce').astype('Int64')
    else:
        df['label_final_code'] = pd.Series([pd.NA] * len(df), dtype='Int64')

    if label_name_column is not None:
        df['label_final_name'] = df[label_name_column].astype(str).str.strip().str.lower()
    else:
        mapping = STATE_LABEL_CODE_TO_NAME.get(state_slug, {})
        df['label_final_name'] = df['label_final_code'].map(mapping)

    unresolved_mask = df['label_final_name'].isna()
    if unresolved_mask.any():
        unresolved_codes = df.loc[unresolved_mask, 'label_final_code'].astype('Int64')
        df.loc[unresolved_mask, 'label_final_name'] = unresolved_codes.map(
            lambda value: f'class_{int(value)}' if pd.notna(value) else 'unknown'
        )

    df['label_final_name'] = df['label_final_name'].astype(str).str.strip().str.lower()
    if state_slug in PAPER_CLASS_ORDER:
        valid_names = set(PAPER_CLASS_ORDER[state_slug])
        df.loc[~df['label_final_name'].isin(valid_names), 'label_final_name'] = 'others'

    if df['label_final_code'].isna().any():
        name_to_code = {
            value: key
            for key, value in STATE_LABEL_CODE_TO_NAME.get(state_slug, {}).items()
        }
        filled_codes = df['label_final_name'].map(name_to_code)
        df.loc[df['label_final_code'].isna(), 'label_final_code'] = filled_codes[df['label_final_code'].isna()]

    df['label_final_code'] = df['label_final_code'].fillna(-1).astype(int)
    return df


def make_label_mapping(dataframe: pd.DataFrame, state_slug: str) -> Tuple[Dict[str, int], Dict[int, int]]:
    class_names = dataframe['label_final_name'].astype(str).tolist()
    present_classes = set(class_names)

    if state_slug in PAPER_CLASS_ORDER:
        class_order = [class_name for class_name in PAPER_CLASS_ORDER[state_slug] if class_name in present_classes]
        leftovers = sorted(present_classes.difference(class_order))
        class_order.extend(leftovers)
    else:
        class_order = sorted(present_classes)

    name_to_idx = {class_name: class_idx for class_idx, class_name in enumerate(class_order)}
    code_to_idx: Dict[int, int] = {}

    for class_name, class_idx in name_to_idx.items():
        codes = dataframe.loc[dataframe['label_final_name'] == class_name, 'label_final_code'].unique()
        for code in codes:
            code_to_idx[int(code)] = class_idx

    return name_to_idx, code_to_idx


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

def infer_dataset_schema(dataframe: pd.DataFrame, csv_path: Path) -> DatasetSchema:
    df = ensure_sample_id(dataframe)
    temporal_groups = collect_temporal_groups(df.columns)

    spectral_groups: Dict[str, TemporalGroup] = {}
    missing_bands = [band for band in S2_BANDS if band not in temporal_groups]
    if missing_bands:
        raise ValueError(f'Colonnes Sentinel-2 manquantes: {missing_bands}')

    for band in S2_BANDS:
        spectral_groups[band] = temporal_group_from_mapping(band, temporal_groups[band])

    reference_indices = spectral_groups[S2_BANDS[0]].indices
    if len(reference_indices) != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError(
            f'Sequence Sentinel-2 inattendue: {len(reference_indices)} pas trouves, {DEFAULT_SEQUENCE_LENGTH} attendus.'
        )

    for band in S2_BANDS[1:]:
        if spectral_groups[band].indices != reference_indices:
            raise ValueError(f'Les indices temporels de {band} ne correspondent pas au reste des bandes S2.')

    valid_mask_group = None
    if 'valid' in temporal_groups:
        candidate_group = temporal_group_from_mapping('valid', temporal_groups['valid'])
        if len(candidate_group.indices) == len(reference_indices):
            valid_mask_group = candidate_group

    sample_id_column = 'sample_id'
    label_code_column = select_first_existing(df.columns, LABEL_CODE_CANDIDATES) or 'label_final_code'
    label_name_column = select_first_existing(df.columns, LABEL_NAME_CANDIDATES)

    dynamic_env_groups: Dict[str, TemporalGroup] = {}
    dynamic_env_base_order: List[str] = []
    group_to_dynamic_columns: Dict[str, List[str]] = {'climate': [], 'soil': [], 'topography': [], 'other': []}

    reserved_temporal_names = set(S2_BANDS + ['valid'])
    for base_name, mapping in temporal_groups.items():
        if base_name in reserved_temporal_names:
            continue
        group = temporal_group_from_mapping(base_name, mapping)
        if len(group.indices) != len(reference_indices):
            raise ValueError(
                f'La variable temporelle {base_name} a {len(group.indices)} pas, attendu {len(reference_indices)}.'
            )
        dynamic_env_groups[base_name] = group
        dynamic_env_base_order.append(base_name)
        env_group = classify_environmental_group(base_name)
        group_to_dynamic_columns.setdefault(env_group, []).append(base_name)

    used_columns = {column_name for group in spectral_groups.values() for column_name in group.columns}
    if valid_mask_group is not None:
        used_columns.update(valid_mask_group.columns)
    for group in dynamic_env_groups.values():
        used_columns.update(group.columns)

    excluded_static = set(used_columns)
    excluded_static.update(['label_final_name', 'label_final_code', 'label_name', 'label', 'sample_id', 'state_name'])
    excluded_static.update(ID_COLUMN_CANDIDATES)

    static_env_columns: List[str] = []
    group_to_static_columns: Dict[str, List[str]] = {'climate': [], 'soil': [], 'topography': [], 'other': []}

    for column_name in df.columns:
        if column_name in excluded_static:
            continue
        if parse_temporal_column(column_name) is not None:
            continue
        if column_name == 'geometry':
            continue
        numeric_series = pd.to_numeric(df[column_name], errors='coerce')
        if numeric_series.notna().sum() == 0:
            continue
        static_env_columns.append(column_name)
        env_group = classify_environmental_group(column_name)
        group_to_static_columns.setdefault(env_group, []).append(column_name)

    state_slug = detect_state(df, csv_path=csv_path)
    state_name = state_display_name(state_slug)

    return DatasetSchema(
        state_slug=state_slug,
        state_name=state_name,
        sample_id_column=sample_id_column,
        label_code_column=label_code_column,
        label_name_column=label_name_column,
        spectral_time_indices=reference_indices,
        spectral_groups=spectral_groups,
        valid_mask_group=valid_mask_group,
        dynamic_env_groups=dynamic_env_groups,
        dynamic_env_base_order=dynamic_env_base_order,
        static_env_columns=static_env_columns,
        group_to_dynamic_columns=group_to_dynamic_columns,
        group_to_static_columns=group_to_static_columns,
    )


# ---------------------------------------------------------------------------
# Split / normalization
# ---------------------------------------------------------------------------

def split_like_paper(dataframe: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for class_name in sorted(dataframe['label_final_name'].unique()):
        class_df = dataframe[dataframe['label_final_name'] == class_name]
        class_indices = class_df.index.to_numpy()
        rng.shuffle(class_indices)

        if len(class_indices) <= 1:
            test_indices.extend(class_indices.tolist())
            continue

        n_train_val = min(TRAIN_VAL_PER_CLASS, len(class_indices))
        n_train = int(n_train_val * TRAIN_FRACTION_WITHIN_TRAIN_VAL)
        n_val = n_train_val - n_train

        train_indices.extend(class_indices[:n_train].tolist())
        val_indices.extend(class_indices[n_train:n_train + n_val].tolist())
        test_indices.extend(class_indices[n_train_val:].tolist())

    return {
        'train': dataframe.loc[train_indices].reset_index(drop=True),
        'val': dataframe.loc[val_indices].reset_index(drop=True),
        'test': dataframe.loc[test_indices].reset_index(drop=True),
    }


def normalize_reflectance_tensor(x: np.ndarray, reflectance_scale: float) -> np.ndarray:
    return (x / reflectance_scale).astype(np.float32)


def compute_dynamic_normalization(dynamic_env_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if dynamic_env_train.shape[-1] == 0:
        empty = np.zeros((1, 1, 0), dtype=np.float32)
        return empty, empty
    env_mean = np.nanmean(dynamic_env_train, axis=(0, 1), keepdims=True).astype(np.float32)
    env_std = np.nanstd(dynamic_env_train, axis=(0, 1), keepdims=True).astype(np.float32)
    env_std = np.where(env_std < 1e-6, 1.0, env_std).astype(np.float32)
    return env_mean, env_std


def apply_dynamic_normalization(dynamic_env: np.ndarray, env_mean: np.ndarray, env_std: np.ndarray) -> np.ndarray:
    if dynamic_env.shape[-1] == 0:
        return dynamic_env.astype(np.float32)
    normalized = (dynamic_env - env_mean) / env_std
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    return normalized.astype(np.float32)


def compute_static_normalization(static_env_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if static_env_train.shape[-1] == 0:
        empty = np.zeros((1, 0), dtype=np.float32)
        return empty, empty
    env_mean = np.nanmean(static_env_train, axis=0, keepdims=True).astype(np.float32)
    env_std = np.nanstd(static_env_train, axis=0, keepdims=True).astype(np.float32)
    env_std = np.where(env_std < 1e-6, 1.0, env_std).astype(np.float32)
    return env_mean, env_std


def apply_static_normalization(static_env: np.ndarray, env_mean: np.ndarray, env_std: np.ndarray) -> np.ndarray:
    if static_env.shape[-1] == 0:
        return static_env.astype(np.float32)
    normalized = (static_env - env_mean) / env_std
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def build_s2_column_order(schema: DatasetSchema) -> List[str]:
    ordered: List[str] = []
    for time_index in schema.spectral_time_indices:
        for band in S2_BANDS:
            column_name = schema.spectral_groups[band].columns[schema.spectral_groups[band].indices.index(time_index)]
            ordered.append(column_name)
    return ordered


def pack_s2_features(dataframe: pd.DataFrame, schema: DatasetSchema) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered_columns = build_s2_column_order(schema)
    feature_values = ensure_numeric_frame(dataframe, ordered_columns).to_numpy(dtype=np.float32)
    x = feature_values.reshape(len(dataframe), schema.sequence_length, len(S2_BANDS))
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)

    if schema.valid_mask_group is not None:
        valid_values = ensure_numeric_frame(dataframe, schema.valid_mask_group.columns).to_numpy(dtype=np.float32)
        valid_mask = (valid_values > 0).astype(np.float32)
    else:
        # Real Arkansas CSV: missing Sentinel-2 composites are already zero-filled.
        # We rebuild a temporal mask from the spectral tensor.
        valid_mask = np.any(np.abs(x) > 0.0, axis=2).astype(np.float32)

    missing_mask = np.repeat((1.0 - valid_mask)[:, :, np.newaxis], len(S2_BANDS), axis=2).astype(np.float32)
    return x, valid_mask, missing_mask


def pack_dynamic_environmental_features(dataframe: pd.DataFrame, schema: DatasetSchema) -> np.ndarray:
    if not schema.dynamic_env_base_order:
        return np.zeros((len(dataframe), schema.sequence_length, 0), dtype=np.float32)

    dynamic_tensor = np.zeros((len(dataframe), schema.sequence_length, len(schema.dynamic_env_base_order)), dtype=np.float32)
    for env_index, base_name in enumerate(schema.dynamic_env_base_order):
        group = schema.dynamic_env_groups[base_name]
        env_values = ensure_numeric_frame(dataframe, group.columns).to_numpy(dtype=np.float32)
        env_values = np.where(np.isfinite(env_values), env_values, np.nan)
        dynamic_tensor[:, :, env_index] = env_values

    return dynamic_tensor


def pack_static_environmental_features(dataframe: pd.DataFrame, schema: DatasetSchema) -> np.ndarray:
    if not schema.static_env_columns:
        return np.zeros((len(dataframe), 0), dtype=np.float32)

    static_values = ensure_numeric_frame(dataframe, schema.static_env_columns).to_numpy(dtype=np.float32)
    static_values = np.where(np.isfinite(static_values), static_values, np.nan)
    return static_values


def build_ablation_metadata(schema: DatasetSchema) -> Dict[str, Dict[str, object]]:
    dynamic_column_to_index = {
        column_name: column_index for column_index, column_name in enumerate(schema.dynamic_env_base_order)
    }
    static_column_to_index = {
        column_name: column_index for column_index, column_name in enumerate(schema.static_env_columns)
    }

    def group_indices(group_name: str) -> Tuple[List[int], List[int]]:
        dynamic_columns = schema.group_to_dynamic_columns.get(group_name, [])
        static_columns = schema.group_to_static_columns.get(group_name, [])
        return (
            [dynamic_column_to_index[column] for column in dynamic_columns if column in dynamic_column_to_index],
            [static_column_to_index[column] for column in static_columns if column in static_column_to_index],
        )

    configs: Dict[str, Dict[str, object]] = {}
    for config_name, group_names in {
        'baseline': [],
        'climate': ['climate'],
        'soil': ['soil'],
        'topography': ['topography'],
        'all': ['climate', 'soil', 'topography', 'other'],
    }.items():
        dynamic_indices: List[int] = []
        static_indices: List[int] = []
        for group_name in group_names:
            dyn_idx, sta_idx = group_indices(group_name)
            dynamic_indices.extend(dyn_idx)
            static_indices.extend(sta_idx)

        # Preserve column order and remove duplicates.
        dynamic_indices = sorted(set(dynamic_indices))
        static_indices = sorted(set(static_indices))

        configs[config_name] = {
            'dynamic_indices': dynamic_indices,
            'static_indices': static_indices,
            'dynamic_columns': [schema.dynamic_env_base_order[idx] for idx in dynamic_indices],
            'static_columns': [schema.static_env_columns[idx] for idx in static_indices],
            'input_dim': len(S2_BANDS) + len(dynamic_indices) + len(static_indices),
        }

    return configs


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_env_dataset_bundle(
    csv_path: Path,
    labels_csv_path: Optional[Path] = None,
    normalize_reflectance: bool = True,
    reflectance_scale: float = 10000.0,
    normalize_environment: bool = True,
    split_seed: int = 2021,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    dataframe = read_csv_resilient(csv_path)
    resolved_labels_path = resolve_labels_csv_path(csv_path, labels_csv_path)
    dataframe = merge_with_labels(dataframe, data_csv_path=csv_path, labels_csv_path=resolved_labels_path)
    dataframe = ensure_sample_id(dataframe)

    state_slug = detect_state(dataframe, csv_path=csv_path)
    dataframe = canonicalize_integrated_labels(dataframe, state_slug=state_slug)
    schema = infer_dataset_schema(dataframe, csv_path=csv_path)

    dataframe = dataframe.sort_values('sample_id').reset_index(drop=True)
    name_to_idx, code_to_idx = make_label_mapping(dataframe, state_slug=schema.state_slug)
    splits = split_like_paper(dataframe, split_seed)

    bundle: Dict[str, np.ndarray] = {}
    split_counts: Dict[str, Dict[str, int]] = {}
    raw_dynamic_env_splits: Dict[str, np.ndarray] = {}
    raw_static_env_splits: Dict[str, np.ndarray] = {}

    for split_name, split_df in splits.items():
        x, valid_mask, missing_mask = pack_s2_features(split_df, schema)
        dynamic_env = pack_dynamic_environmental_features(split_df, schema)
        static_env = pack_static_environmental_features(split_df, schema)
        y = np.array([code_to_idx[int(label_code)] for label_code in split_df['label_final_code']], dtype=np.int64)

        if normalize_reflectance:
            x = normalize_reflectance_tensor(x, reflectance_scale)

        bundle[f'x_{split_name}'] = x.astype(np.float32)
        bundle[f'valid_mask_{split_name}'] = valid_mask.astype(np.float32)
        bundle[f'missing_mask_{split_name}'] = missing_mask.astype(np.float32)
        bundle[f'y_{split_name}'] = y
        raw_dynamic_env_splits[split_name] = dynamic_env
        raw_static_env_splits[split_name] = static_env

        split_counts[split_name] = split_df['label_final_name'].value_counts().to_dict()

    dynamic_mean = None
    dynamic_std = None
    static_mean = None
    static_std = None

    if normalize_environment:
        dynamic_mean, dynamic_std = compute_dynamic_normalization(raw_dynamic_env_splits['train'])
        static_mean, static_std = compute_static_normalization(raw_static_env_splits['train'])

        for split_name in splits:
            bundle[f'dynamic_env_{split_name}'] = apply_dynamic_normalization(
                raw_dynamic_env_splits[split_name],
                dynamic_mean,
                dynamic_std,
            )
            bundle[f'static_env_{split_name}'] = apply_static_normalization(
                raw_static_env_splits[split_name],
                static_mean,
                static_std,
            )
    else:
        for split_name in splits:
            bundle[f'dynamic_env_{split_name}'] = np.where(
                np.isfinite(raw_dynamic_env_splits[split_name]),
                raw_dynamic_env_splits[split_name],
                0.0,
            ).astype(np.float32)
            bundle[f'static_env_{split_name}'] = np.where(
                np.isfinite(raw_static_env_splits[split_name]),
                raw_static_env_splits[split_name],
                0.0,
            ).astype(np.float32)

    dynamic_column_to_index = {
        column_name: column_index for column_index, column_name in enumerate(schema.dynamic_env_base_order)
    }
    static_column_to_index = {
        column_name: column_index for column_index, column_name in enumerate(schema.static_env_columns)
    }

    metadata = {
        'state_slug': schema.state_slug,
        'state_name': schema.state_name,
        'source_csv': str(csv_path),
        'labels_csv': str(resolved_labels_path) if resolved_labels_path else None,
        'n_samples_total': int(len(dataframe)),
        'samples_per_split': {split_name: int(len(split_df)) for split_name, split_df in splits.items()},
        'split_counts': split_counts,
        'class_name_to_index': name_to_idx,
        'num_classes': len(name_to_idx),
        'sequence_length': schema.sequence_length,
        'feature_shape_per_sample': [schema.sequence_length, len(S2_BANDS)],
        'dynamic_env_shape_per_sample': [schema.sequence_length, len(schema.dynamic_env_base_order)],
        'static_env_shape_per_sample': [len(schema.static_env_columns)],
        'spectral_bands': S2_BANDS,
        'spectral_time_indices': schema.spectral_time_indices,
        'environmental_covariates': {
            'dynamic_columns': schema.dynamic_env_base_order,
            'static_columns': schema.static_env_columns,
            'dynamic_column_to_index': dynamic_column_to_index,
            'static_column_to_index': static_column_to_index,
            'group_to_dynamic_columns': schema.group_to_dynamic_columns,
            'group_to_static_columns': schema.group_to_static_columns,
            'broadcast_static_in_model': True,
        },
        'ablation_configs': build_ablation_metadata(schema),
        'paper_settings': {
            'n_time_steps': schema.sequence_length,
            'n_s2_bands': len(S2_BANDS),
            'train_val_per_class': TRAIN_VAL_PER_CLASS,
            'train_fraction_within_train_val': TRAIN_FRACTION_WITHIN_TRAIN_VAL,
            'missing_s2_values_marked_with_zero': True,
        },
        'implementation_choices': {
            'spectral_parser_accepts_t0_to_t35_and_t01_to_t36': True,
            'temporal_alignment_uses_sorted_suffix_order': True,
            'integrated_labels_supported': True,
            'static_variables_kept_separate_from_temporal_csv_tensor': True,
            'static_variables_broadcast_inside_python_model_pipeline': True,
            'valid_mask_reconstructed_from_zero_filled_s2_when_missing': schema.valid_mask_group is None,
            'reflectance_normalization': normalize_reflectance,
            'reflectance_scale': reflectance_scale if normalize_reflectance else None,
            'environmental_normalization': normalize_environment,
            'split_seed': split_seed,
        },
    }

    if dynamic_mean is not None and dynamic_std is not None:
        metadata['dynamic_environment_normalization'] = {
            'mean': dynamic_mean.reshape(-1).astype(float).tolist(),
            'std': dynamic_std.reshape(-1).astype(float).tolist(),
            'feature_order': schema.dynamic_env_base_order,
        }
    if static_mean is not None and static_std is not None:
        metadata['static_environment_normalization'] = {
            'mean': static_mean.reshape(-1).astype(float).tolist(),
            'std': static_std.reshape(-1).astype(float).tolist(),
            'feature_order': schema.static_env_columns,
        }

    return bundle, metadata


def save_bundle(bundle: Dict[str, np.ndarray], metadata: Dict, npz_path: Path, json_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **bundle)
    json_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build MCTNet datasets from real temporal CSV files with dynamic and static covariates.'
    )
    parser.add_argument('--input-csv', nargs='+', required=True, help='Path(s) to temporal CSV file(s).')
    parser.add_argument(
        '--labels-csv',
        nargs='*',
        default=None,
        help='Optional label CSV path(s). Omit when labels are already inside the source CSV.',
    )
    parser.add_argument('--output-dir', required=True, help='Output directory for NPZ/JSON files.')
    parser.add_argument('--split-seed', type=int, default=2021)
    parser.add_argument('--reflectance-scale', type=float, default=10000.0)
    parser.add_argument('--disable-normalize-reflectance', action='store_true')
    parser.add_argument('--disable-normalize-environment', action='store_true')
    return parser.parse_args()


def expand_labels_paths(input_csv: List[str], labels_csv: Optional[List[str]]) -> List[Optional[Path]]:
    if not labels_csv:
        return [None] * len(input_csv)

    resolved = [Path(path_str) for path_str in labels_csv]
    if len(resolved) == 1 and len(input_csv) > 1:
        return resolved * len(input_csv)
    if len(resolved) != len(input_csv):
        raise ValueError('The number of --labels-csv paths must be 1 or equal to the number of --input-csv paths.')
    return resolved


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    labels_paths = expand_labels_paths(args.input_csv, args.labels_csv)

    for input_csv_str, labels_path in zip(args.input_csv, labels_paths):
        input_csv_path = Path(input_csv_str)
        bundle, metadata = build_env_dataset_bundle(
            csv_path=input_csv_path,
            labels_csv_path=labels_path,
            normalize_reflectance=not args.disable_normalize_reflectance,
            reflectance_scale=args.reflectance_scale,
            normalize_environment=not args.disable_normalize_environment,
            split_seed=args.split_seed,
        )

        state_slug = metadata['state_slug']
        output_npz = output_dir / f'{state_slug}_mctnet_env_dataset.npz'
        output_json = output_dir / f'{state_slug}_mctnet_env_dataset.json'
        save_bundle(bundle, metadata, output_npz, output_json)

        print(f'[{state_slug}] saved')
        print(f'  x_train shape: {bundle["x_train"].shape}')
        print(f'  dynamic_env_train shape: {bundle["dynamic_env_train"].shape}')
        print(f'  static_env_train shape: {bundle["static_env_train"].shape}')
        print(f'  valid_mask_train shape: {bundle["valid_mask_train"].shape}')
        print(f'  classes: {metadata["class_name_to_index"]}')
        print(f'  -> {output_npz}')
        print(f'  -> {output_json}')


if __name__ == '__main__':
    main()

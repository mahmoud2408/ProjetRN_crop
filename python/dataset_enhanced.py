"""
Technical summary
-----------------
This dataset pipeline combines five ideas from the provided papers in a way that
fits the current project data without changing existing files.

1. The MCTNet paper contributes the masked Sentinel-2 time-series setting and
   the importance of preserving missing-observation masks.
2. The red-edge crop-mapping paper contributes the 14-channel per-date
   representation built from 10 Sentinel-2 bands plus NDVI, IRECI, MTCI, and
   S2REP.
3. The Geo-CBAM-CNN paper motivates attention over temporal-spectral features,
   especially around informative red-edge periods.
4. The tree-crop multi-sensor paper motivates heterogeneous fusion instead of
   collapsing every modality into one undifferentiated tensor. Here, dynamic
   climate covariates stay temporal, while soil and topography stay static.
5. The supervised-to-unsupervised paper motivates exposing clean reconstruction
   targets so the training loop can add a masked self-supervised auxiliary loss.

The goal is to give the enhanced model a richer temporal input, cleaner
modality separation, and stable train-time targets for auxiliary learning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


S2_BANDS: List[str] = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
DEFAULT_CONFIGS: List[str] = ['baseline', 'climate', 'soil', 'topography', 'all']


@dataclass
class EnhancedPreparedSplits:
    """
    Prepared tensors for one state and one ablation configuration.

    Attributes:
        state: State slug.
        class_names: Ordered class-name list of length C.
        x_train: Train tensor of shape (N_train, 36, 14).
        x_val: Validation tensor of shape (N_val, 36, 14).
        x_test: Test tensor of shape (N_test, 36, 14).
        valid_mask_train: Train validity mask of shape (N_train, 36).
        valid_mask_val: Validation validity mask of shape (N_val, 36).
        valid_mask_test: Test validity mask of shape (N_test, 36).
        y_train: Train labels of shape (N_train,).
        y_val: Validation labels of shape (N_val,).
        y_test: Test labels of shape (N_test,).
        dynamic_env_train: Train dynamic covariates of shape (N_train, 36, E_dyn).
        dynamic_env_val: Validation dynamic covariates of shape (N_val, 36, E_dyn).
        dynamic_env_test: Test dynamic covariates of shape (N_test, 36, E_dyn).
        static_env_train: Train static covariates of shape (N_train, E_static).
        static_env_val: Validation static covariates of shape (N_val, E_static).
        static_env_test: Test static covariates of shape (N_test, E_static).
        temporal_input_dim: Number of temporal channels seen by the model.
        static_input_dim: Number of static channels seen by the model.
        dynamic_feature_names: Selected dynamic feature names.
        static_feature_names: Selected static feature names.
        vi_mean: Train VI means of shape (4,).
        vi_std: Train VI stds of shape (4,).
        metadata: Full metadata dict loaded from the JSON file.
    """

    state: str
    class_names: List[str]
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    valid_mask_train: np.ndarray
    valid_mask_val: np.ndarray
    valid_mask_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    dynamic_env_train: np.ndarray
    dynamic_env_val: np.ndarray
    dynamic_env_test: np.ndarray
    static_env_train: np.ndarray
    static_env_val: np.ndarray
    static_env_test: np.ndarray
    temporal_input_dim: int
    static_input_dim: int
    dynamic_feature_names: List[str]
    static_feature_names: List[str]
    vi_mean: np.ndarray
    vi_std: np.ndarray
    metadata: Dict[str, Any]


class EnhancedCropDataset(Dataset):
    """
    PyTorch dataset for the enhanced crop classifier.

    Args:
        x: Temporal spectral tensor of shape (N, 36, 14).
        valid_mask: Validity mask of shape (N, 36).
        y: Labels of shape (N,).
        dynamic_env: Dynamic temporal covariates of shape (N, 36, E_dyn).
        static_env: Static covariates of shape (N, E_static).

    Returns:
        Each item is a dict with tensors:
            x: (36, 14)
            valid_mask: (36,)
            y: ()
            dynamic_env: (36, E_dyn)
            static_env: (E_static,)
    """

    def __init__(
        self,
        x: np.ndarray,
        valid_mask: np.ndarray,
        y: np.ndarray,
        dynamic_env: Optional[np.ndarray] = None,
        static_env: Optional[np.ndarray] = None,
    ) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32)).float()
        self.valid_mask = torch.from_numpy(np.asarray(valid_mask, dtype=np.float32)).float()
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()

        if dynamic_env is None:
            dynamic_env = np.zeros((self.x.shape[0], self.x.shape[1], 0), dtype=np.float32)
        if static_env is None:
            static_env = np.zeros((self.x.shape[0], 0), dtype=np.float32)

        self.dynamic_env = torch.from_numpy(np.asarray(dynamic_env, dtype=np.float32)).float()
        self.static_env = torch.from_numpy(np.asarray(static_env, dtype=np.float32)).float()

    def __len__(self) -> int:
        """
        Return dataset size.

        Args:
            None.

        Returns:
            Integer number of samples.
        """
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """
        Return one sample.

        Args:
            index: Sample index.

        Returns:
            Dict containing tensors for one sample.
        """
        return {
            'x': self.x[index],
            'valid_mask': self.valid_mask[index],
            'y': self.y[index],
            'dynamic_env': self.dynamic_env[index],
            'static_env': self.static_env[index],
        }


def resolve_dataset_paths(processed_env_dir: Path, state: str) -> Tuple[Path, Path]:
    """
    Resolve bundle paths for one state.

    Args:
        processed_env_dir: Directory containing processed bundles.
        state: State slug.

    Returns:
        Tuple (npz_path, json_path).
    """
    candidates = [
        (
            processed_env_dir / f'{state}_mctnet_env_dataset.npz',
            processed_env_dir / f'{state}_mctnet_env_dataset.json',
        ),
        (
            processed_env_dir / f'{state}_mctnet_env_dataset_FULL_TEMPORAL.npz',
            processed_env_dir / f'{state}_mctnet_env_dataset_FULL_TEMPORAL.json',
        ),
        (
            processed_env_dir / f'{state}_mctnet_dataset.npz',
            processed_env_dir / f'{state}_mctnet_dataset.json',
        ),
    ]
    for npz_path, json_path in candidates:
        if npz_path.exists() and json_path.exists():
            return npz_path, json_path
    raise FileNotFoundError(f'Dataset files not found for state={state} in {processed_env_dir}')


def load_bundle_and_metadata(npz_path: Path, json_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load a processed bundle and its metadata.

    Args:
        npz_path: Bundle path.
        json_path: Metadata path.

    Returns:
        Tuple (bundle, metadata).
    """
    with np.load(npz_path, allow_pickle=True) as data:
        bundle = {key: data[key] for key in data.files}
    metadata = json.loads(json_path.read_text(encoding='utf-8'))
    return bundle, metadata


def compute_red_edge_features(x: np.ndarray) -> np.ndarray:
    """
    Compute per-date red-edge vegetation indices.

    Args:
        x: Sentinel-2 tensor of shape (N, 36, 10).

    Returns:
        Tensor of shape (N, 36, 14).
    """
    eps = 1e-8
    x = np.asarray(x, dtype=np.float32)
    valid = np.any(np.abs(x) > 0.0, axis=-1, keepdims=True)

    b4 = x[:, :, 2]
    b5 = x[:, :, 3]
    b6 = x[:, :, 4]
    b7 = x[:, :, 5]
    b8 = x[:, :, 6]

    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ireci = (b7 - b4) / (b5 + eps)
    mtci = (b6 - b5) / (b5 - b4 + eps)
    s2rep = 705.0 + 35.0 * ((((b4 + b7) * 0.5) - b5) / (b6 - b5 + eps))

    ndvi = np.clip(ndvi, -1.5, 1.5)
    ireci = np.clip(ireci, -5.0, 5.0)
    mtci = np.clip(mtci, -5.0, 5.0)
    s2rep = np.clip(s2rep, 650.0, 850.0)

    vi = np.stack([ndvi, ireci, mtci, s2rep], axis=-1).astype(np.float32)
    vi = np.where(valid, vi, 0.0).astype(np.float32)
    return np.concatenate([x, vi], axis=-1).astype(np.float32)


def normalize_vi_channels(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize only the four VI channels from train statistics.

    Args:
        x_train: Train tensor of shape (N_train, 36, 14).
        x_val: Validation tensor of shape (N_val, 36, 14).
        x_test: Test tensor of shape (N_test, 36, 14).

    Returns:
        Tuple (x_train_n, x_val_n, x_test_n, vi_mean, vi_std).
    """
    train = np.array(x_train, dtype=np.float32, copy=True)
    val = np.array(x_val, dtype=np.float32, copy=True)
    test = np.array(x_test, dtype=np.float32, copy=True)

    train_valid = np.any(np.abs(train[:, :, :10]) > 0.0, axis=-1)
    vi_train = train[:, :, 10:14]

    if np.any(train_valid):
        vi_values = vi_train[train_valid]
        vi_mean = vi_values.mean(axis=0).astype(np.float32)
        vi_std = vi_values.std(axis=0).astype(np.float32)
    else:
        vi_mean = np.zeros(4, dtype=np.float32)
        vi_std = np.ones(4, dtype=np.float32)

    vi_std = np.where(vi_std < 1e-8, 1.0, vi_std).astype(np.float32)

    def _apply_norm(x_tensor: np.ndarray) -> np.ndarray:
        out = np.array(x_tensor, dtype=np.float32, copy=True)
        valid = np.any(np.abs(out[:, :, :10]) > 0.0, axis=-1)
        vi_norm = (out[:, :, 10:14] - vi_mean.reshape(1, 1, 4)) / vi_std.reshape(1, 1, 4)
        vi_norm = np.where(valid[:, :, None], vi_norm, 0.0).astype(np.float32)
        out[:, :, 10:14] = vi_norm
        return out

    return _apply_norm(train), _apply_norm(val), _apply_norm(test), vi_mean, vi_std


def _resolve_class_names(metadata: Dict[str, Any]) -> List[str]:
    """
    Resolve ordered class names from metadata.

    Args:
        metadata: Metadata dict containing class_name_to_index.

    Returns:
        Ordered class-name list.
    """
    mapping = metadata['class_name_to_index']
    class_names: List[Optional[str]] = [None] * int(metadata['num_classes'])
    for class_name, class_idx in mapping.items():
        class_names[int(class_idx)] = class_name
    return [class_name if class_name is not None else f'class_{idx}' for idx, class_name in enumerate(class_names)]


def _extract_config_indices(
    metadata: Dict[str, Any],
    config_name: str,
    dynamic_dim: int,
    static_dim: int,
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """
    Resolve selected dynamic and static feature indices for one config.

    Args:
        metadata: Metadata dict containing ablation configuration specs.
        config_name: Ablation config name.
        dynamic_dim: Number of available dynamic channels.
        static_dim: Number of available static channels.

    Returns:
        Tuple (dynamic_indices, static_indices, dynamic_names, static_names).
    """
    specs = metadata.get('ablation_configs', {})
    env_info = metadata.get('environmental_covariates', {})
    dynamic_names_all = list(env_info.get('dynamic_columns', []))
    static_names_all = list(env_info.get('static_columns', []))

    if config_name not in specs:
        return [], [], [], []

    spec = specs[config_name]
    if isinstance(spec, dict):
        dynamic_indices = [int(index) for index in spec.get('dynamic_indices', []) if 0 <= int(index) < dynamic_dim]
        static_indices = [int(index) for index in spec.get('static_indices', []) if 0 <= int(index) < static_dim]
        dynamic_names = [dynamic_names_all[index] for index in dynamic_indices if index < len(dynamic_names_all)]
        static_names = [static_names_all[index] for index in static_indices if index < len(static_names_all)]
        return dynamic_indices, static_indices, dynamic_names, static_names

    if isinstance(spec, list):
        dynamic_count = len(dynamic_names_all)
        dynamic_indices = [int(index) for index in spec if 0 <= int(index) < min(dynamic_dim, dynamic_count)]
        static_indices = [int(index) - dynamic_count for index in spec if dynamic_count <= int(index) < dynamic_count + static_dim]
        static_indices = [index for index in static_indices if 0 <= index < static_dim]
        dynamic_names = [dynamic_names_all[index] for index in dynamic_indices if index < len(dynamic_names_all)]
        static_names = [static_names_all[index] for index in static_indices if index < len(static_names_all)]
        return dynamic_indices, static_indices, dynamic_names, static_names

    return [], [], [], []


def _load_environment_splits(
    bundle: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    """
    Load dynamic and static environmental covariates from a bundle.

    Args:
        bundle: Loaded NPZ content.
        metadata: Metadata dict.

    Returns:
        Tuple (dynamic_splits, static_splits), each mapping split names to arrays.
    """
    split_names = ['train', 'val', 'test']

    if all(f'dynamic_env_{split_name}' in bundle for split_name in split_names) and all(
        f'static_env_{split_name}' in bundle for split_name in split_names
    ):
        dynamic_splits = {
            split_name: np.asarray(bundle[f'dynamic_env_{split_name}'], dtype=np.float32)
            for split_name in split_names
        }
        static_splits = {
            split_name: np.asarray(bundle[f'static_env_{split_name}'], dtype=np.float32)
            for split_name in split_names
        }
        return dynamic_splits, static_splits

    if all(f'env_{split_name}' in bundle for split_name in split_names):
        env_info = metadata.get('environmental_covariates', {})
        dynamic_count = len(env_info.get('dynamic_columns', []))
        static_count = len(env_info.get('static_columns', []))
        dynamic_splits: Dict[str, np.ndarray] = {}
        static_splits: Dict[str, np.ndarray] = {}
        for split_name in split_names:
            env_tensor = np.asarray(bundle[f'env_{split_name}'], dtype=np.float32)
            dynamic_splits[split_name] = env_tensor[:, :, :dynamic_count].astype(np.float32)
            if static_count > 0:
                static_splits[split_name] = env_tensor[:, 0, dynamic_count:dynamic_count + static_count].astype(np.float32)
            else:
                static_splits[split_name] = np.zeros((env_tensor.shape[0], 0), dtype=np.float32)
        return dynamic_splits, static_splits

    return None, None


def _select_features(arr: np.ndarray, indices: Sequence[int], axis: int) -> np.ndarray:
    """
    Select a subset of features along one axis.

    Args:
        arr: Input array.
        indices: Selected indices.
        axis: Feature axis.

    Returns:
        Selected array with the same leading dimensions.
    """
    if len(indices) == 0:
        shape = list(arr.shape)
        shape[axis] = 0
        return np.zeros(shape, dtype=np.float32)
    return np.take(np.asarray(arr, dtype=np.float32), indices=list(indices), axis=axis).astype(np.float32)


def prepare_enhanced_splits(
    npz_path: Path,
    json_path: Path,
    config_name: str = 'baseline',
    use_env_covariates: bool = True,
) -> EnhancedPreparedSplits:
    """
    Prepare tensors for the enhanced architecture.

    Args:
        npz_path: Bundle path.
        json_path: Metadata path.
        config_name: One of baseline, climate, soil, topography, all.
        use_env_covariates: Whether to activate environmental covariates.

    Returns:
        An EnhancedPreparedSplits object with all tensors ready for training.
    """
    bundle, metadata = load_bundle_and_metadata(npz_path=npz_path, json_path=json_path)
    state = str(metadata.get('state_slug', metadata.get('state_name', npz_path.stem))).lower().replace(' ', '_')

    x_train = compute_red_edge_features(np.asarray(bundle['x_train'], dtype=np.float32))
    x_val = compute_red_edge_features(np.asarray(bundle['x_val'], dtype=np.float32))
    x_test = compute_red_edge_features(np.asarray(bundle['x_test'], dtype=np.float32))
    x_train, x_val, x_test, vi_mean, vi_std = normalize_vi_channels(x_train=x_train, x_val=x_val, x_test=x_test)

    valid_mask_train = np.asarray(bundle['valid_mask_train'], dtype=np.float32)
    valid_mask_val = np.asarray(bundle['valid_mask_val'], dtype=np.float32)
    valid_mask_test = np.asarray(bundle['valid_mask_test'], dtype=np.float32)
    y_train = np.asarray(bundle['y_train'], dtype=np.int64)
    y_val = np.asarray(bundle['y_val'], dtype=np.int64)
    y_test = np.asarray(bundle['y_test'], dtype=np.int64)

    dynamic_splits, static_splits = _load_environment_splits(bundle=bundle, metadata=metadata)
    dynamic_dim = 0 if dynamic_splits is None else int(dynamic_splits['train'].shape[-1])
    static_dim = 0 if static_splits is None else int(static_splits['train'].shape[-1])

    if use_env_covariates and config_name != 'baseline':
        dynamic_indices, static_indices, dynamic_names, static_names = _extract_config_indices(
            metadata=metadata,
            config_name=config_name,
            dynamic_dim=dynamic_dim,
            static_dim=static_dim,
        )
    else:
        dynamic_indices, static_indices, dynamic_names, static_names = [], [], [], []

    if dynamic_splits is None:
        dynamic_env_train = np.zeros((x_train.shape[0], x_train.shape[1], 0), dtype=np.float32)
        dynamic_env_val = np.zeros((x_val.shape[0], x_val.shape[1], 0), dtype=np.float32)
        dynamic_env_test = np.zeros((x_test.shape[0], x_test.shape[1], 0), dtype=np.float32)
    else:
        dynamic_env_train = _select_features(dynamic_splits['train'], dynamic_indices, axis=-1)
        dynamic_env_val = _select_features(dynamic_splits['val'], dynamic_indices, axis=-1)
        dynamic_env_test = _select_features(dynamic_splits['test'], dynamic_indices, axis=-1)

    if static_splits is None:
        static_env_train = np.zeros((x_train.shape[0], 0), dtype=np.float32)
        static_env_val = np.zeros((x_val.shape[0], 0), dtype=np.float32)
        static_env_test = np.zeros((x_test.shape[0], 0), dtype=np.float32)
    else:
        static_env_train = _select_features(static_splits['train'], static_indices, axis=-1)
        static_env_val = _select_features(static_splits['val'], static_indices, axis=-1)
        static_env_test = _select_features(static_splits['test'], static_indices, axis=-1)

    return EnhancedPreparedSplits(
        state=state,
        class_names=_resolve_class_names(metadata),
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        valid_mask_train=valid_mask_train,
        valid_mask_val=valid_mask_val,
        valid_mask_test=valid_mask_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        dynamic_env_train=dynamic_env_train,
        dynamic_env_val=dynamic_env_val,
        dynamic_env_test=dynamic_env_test,
        static_env_train=static_env_train,
        static_env_val=static_env_val,
        static_env_test=static_env_test,
        temporal_input_dim=int(x_train.shape[-1] + dynamic_env_train.shape[-1]),
        static_input_dim=int(static_env_train.shape[-1]),
        dynamic_feature_names=dynamic_names,
        static_feature_names=static_names,
        vi_mean=vi_mean,
        vi_std=vi_std,
        metadata=metadata,
    )

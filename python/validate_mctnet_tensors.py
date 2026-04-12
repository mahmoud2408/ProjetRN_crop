
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CropTimeSeriesDataset(Dataset):
    def __init__(self, bundle: Dict[str, np.ndarray], split: str) -> None:
        self.x = torch.from_numpy(bundle[f'x_{split}']).float()
        self.valid_mask = torch.from_numpy(bundle[f'valid_mask_{split}']).float()
        self.missing_mask = torch.from_numpy(bundle[f'missing_mask_{split}']).float()
        self.y = torch.from_numpy(bundle[f'y_{split}']).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            'x': self.x[index],
            'valid_mask': self.valid_mask[index],
            'missing_mask': self.missing_mask[index],
            'y': self.y[index],
        }


def load_bundle(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def validate_split_shapes(bundle: Dict[str, np.ndarray], split: str) -> None:
    x = bundle[f'x_{split}']
    valid_mask = bundle[f'valid_mask_{split}']
    missing_mask = bundle[f'missing_mask_{split}']
    y = bundle[f'y_{split}']

    assert x.ndim == 3, f'{split}: x doit etre 3D, recu {x.shape}'
    assert x.shape[1:] == (36, 10), f'{split}: x attendu [N, 36, 10], recu {x.shape}'
    assert valid_mask.shape == (x.shape[0], 36), (
        f'{split}: valid_mask attendu [N, 36], recu {valid_mask.shape}'
    )
    assert missing_mask.shape == (x.shape[0], 36, 10), (
        f'{split}: missing_mask attendu [N, 36, 10], recu {missing_mask.shape}'
    )
    assert y.shape == (x.shape[0],), f'{split}: y attendu [N], recu {y.shape}'

    recovered_missing_mask = np.repeat((1 - valid_mask)[:, :, None], 10, axis=2)
    assert np.array_equal(missing_mask, recovered_missing_mask), (
        f'{split}: missing_mask ne correspond pas a 1 - valid_mask'
    )


def validate_dataloader(bundle: Dict[str, np.ndarray], split: str, batch_size: int) -> None:
    dataset = CropTimeSeriesDataset(bundle, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    assert batch['x'].shape[1:] == (36, 10), f'{split}: batch x incorrect {tuple(batch["x"].shape)}'
    assert batch['valid_mask'].shape[1:] == (36,), (
        f'{split}: batch valid_mask incorrect {tuple(batch["valid_mask"].shape)}'
    )
    assert batch['missing_mask'].shape[1:] == (36, 10), (
        f'{split}: batch missing_mask incorrect {tuple(batch["missing_mask"].shape)}'
    )
    assert batch['y'].ndim == 1, f'{split}: batch y incorrect {tuple(batch["y"].shape)}'

    print(
        f'[{split}] batch OK | '
        f'x={tuple(batch["x"].shape)} | '
        f'valid_mask={tuple(batch["valid_mask"].shape)} | '
        f'missing_mask={tuple(batch["missing_mask"].shape)} | '
        f'y={tuple(batch["y"].shape)}'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate MCTNet dataset tensor dimensions.'
    )
    parser.add_argument('--dataset-npz', required=True, help='Path to the .npz dataset bundle.')
    parser.add_argument(
        '--metadata-json',
        required=False,
        help='Optional metadata JSON produced by build_mctnet_dataset.py',
    )
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for DataLoader validation.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.dataset_npz)
    bundle = load_bundle(npz_path)

    if args.metadata_json:
        metadata = json.loads(Path(args.metadata_json).read_text(encoding='utf-8'))
        print(f"State: {metadata['state_name']}")
        print(f"Classes: {metadata['class_name_to_index']}")

    for split in ['train', 'val', 'test']:
        validate_split_shapes(bundle, split)
        validate_dataloader(bundle, split, args.batch_size)

    print('Toutes les dimensions attendues par le pipeline MCTNet sont correctes.')


if __name__ == '__main__':
    main()

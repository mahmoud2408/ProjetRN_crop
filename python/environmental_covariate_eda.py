from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_mctnet_env_dataset import ALL_ENV_COLUMNS, ENV_GROUPS


def correlation_ratio(categories: np.ndarray, values: np.ndarray) -> float:
    categories = np.asarray(categories)
    values = np.asarray(values, dtype=np.float64)

    valid_mask = ~pd.isna(categories) & ~np.isnan(values)
    categories = categories[valid_mask]
    values = values[valid_mask]

    if values.size == 0:
        return 0.0

    grand_mean = values.mean()
    numerator = 0.0
    denominator = np.sum((values - grand_mean) ** 2)
    if denominator == 0:
        return 0.0

    for category in np.unique(categories):
        group_values = values[categories == category]
        if group_values.size == 0:
            continue
        numerator += group_values.size * (group_values.mean() - grand_mean) ** 2

    eta_squared = numerator / denominator
    return float(np.sqrt(max(eta_squared, 0.0)))


def save_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    output_path: Path,
    cmap: str = 'viridis',
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.6), max(6, len(row_labels) * 0.35)))
    image = ax.imshow(matrix, cmap=cmap, aspect='auto')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_barplot(values: pd.Series, title: str, ylabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    values.sort_values(ascending=False).plot(kind='bar', ax=ax, color='#3b82f6')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_eda(csv_path: Path, output_dir: Path) -> Dict[str, str]:
    dataframe = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = dataframe[ALL_ENV_COLUMNS].describe().T
    summary['missing_count'] = dataframe[ALL_ENV_COLUMNS].isna().sum()
    summary_path = output_dir / 'covariate_summary.csv'
    summary.to_csv(summary_path, encoding='utf-8')

    class_means = dataframe.groupby('label_final_name')[ALL_ENV_COLUMNS].mean()
    class_means_path = output_dir / 'covariate_class_means.csv'
    class_means.to_csv(class_means_path, encoding='utf-8')

    pearson_corr = dataframe[ALL_ENV_COLUMNS].corr(method='pearson')
    pearson_path = output_dir / 'covariate_pearson_corr.csv'
    pearson_corr.to_csv(pearson_path, encoding='utf-8')

    eta_values = pd.Series(
        {
            column_name: correlation_ratio(
                categories=dataframe['label_final_name'].to_numpy(),
                values=dataframe[column_name].to_numpy(),
            )
            for column_name in ALL_ENV_COLUMNS
        }
    )
    eta_path = output_dir / 'covariate_class_eta.csv'
    eta_values.rename('eta').to_csv(eta_path, encoding='utf-8')

    save_heatmap(
        matrix=pearson_corr.to_numpy(),
        row_labels=pearson_corr.index.tolist(),
        col_labels=pearson_corr.columns.tolist(),
        title='Pearson correlation between environmental covariates',
        output_path=output_dir / 'covariate_pearson_heatmap.png',
        cmap='coolwarm',
    )
    save_heatmap(
        matrix=class_means.to_numpy(),
        row_labels=class_means.index.tolist(),
        col_labels=class_means.columns.tolist(),
        title='Class-wise mean values of environmental covariates',
        output_path=output_dir / 'covariate_class_means_heatmap.png',
        cmap='viridis',
    )
    save_barplot(
        values=eta_values,
        title='Correlation ratio between crop classes and environmental covariates',
        ylabel='Eta',
        output_path=output_dir / 'covariate_class_eta_barplot.png',
    )

    group_summary = {
        group_name: dataframe[column_names].describe().T.to_dict()
        for group_name, column_names in ENV_GROUPS.items()
    }
    group_summary_path = output_dir / 'covariate_group_summary.json'
    group_summary_path.write_text(json.dumps(group_summary, indent=2), encoding='utf-8')

    return {
        'summary_csv': str(summary_path),
        'class_means_csv': str(class_means_path),
        'pearson_corr_csv': str(pearson_path),
        'eta_csv': str(eta_path),
        'pearson_heatmap_png': str(output_dir / 'covariate_pearson_heatmap.png'),
        'class_means_heatmap_png': str(output_dir / 'covariate_class_means_heatmap.png'),
        'eta_barplot_png': str(output_dir / 'covariate_class_eta_barplot.png'),
        'group_summary_json': str(group_summary_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run exploratory analysis on environmental covariates.')
    parser.add_argument('--input-csv', required=True, help='CSV exported from GEE with environmental covariates.')
    parser.add_argument('--output-dir', required=True, help='Output directory for EDA artifacts.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_eda(csv_path=Path(args.input_csv), output_dir=Path(args.output_dir))
    print(json.dumps(artifacts, indent=2))


if __name__ == '__main__':
    main()
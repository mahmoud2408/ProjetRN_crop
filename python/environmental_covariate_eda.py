from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_mctnet_env_dataset import (
    canonicalize_integrated_labels,
    detect_state,
    infer_dataset_schema,
    merge_with_labels,
    read_csv_resilient,
)


def correlation_ratio(categories: np.ndarray, values: np.ndarray) -> float:
    categories = np.asarray(categories)
    values = np.asarray(values, dtype=np.float64)

    valid_mask = ~pd.isna(categories) & ~np.isnan(values)
    categories = categories[valid_mask]
    values = values[valid_mask]

    if values.size == 0:
        return 0.0

    grand_mean = values.mean()
    denominator = np.sum((values - grand_mean) ** 2)
    if denominator == 0.0:
        return 0.0

    numerator = 0.0
    for category in np.unique(categories):
        category_values = values[categories == category]
        if category_values.size == 0:
            continue
        numerator += category_values.size * (category_values.mean() - grand_mean) ** 2

    eta_squared = numerator / denominator
    return float(np.sqrt(max(eta_squared, 0.0)))


def save_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    column_labels: List[str],
    title: str,
    output_path: Path,
    cmap: str = 'viridis',
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(max(8, len(column_labels) * 0.8), max(6, len(row_labels) * 0.5)))
    image = axis.imshow(matrix, cmap=cmap, aspect='auto')
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(np.arange(len(column_labels)))
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_xticklabels(column_labels, rotation=45, ha='right')
    axis.set_yticklabels(row_labels)
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_barplot(values: pd.Series, title: str, ylabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 6))
    values.sort_values(ascending=False).plot(kind='bar', ax=axis, color='#2563eb')
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def build_temporal_mean_dataframe(dataframe: pd.DataFrame, csv_path: Path) -> tuple[pd.DataFrame, Dict[str, str]]:
    schema = infer_dataset_schema(dataframe, csv_path=csv_path)
    eda_df = pd.DataFrame(index=dataframe.index)

    feature_group: Dict[str, str] = {}

    for base_name in schema.dynamic_env_base_order:
        group = schema.dynamic_env_groups[base_name]
        temporal_values = dataframe[group.columns].apply(pd.to_numeric, errors='coerce').replace(-9999.0, np.nan)
        eda_df[base_name] = temporal_values.mean(axis=1)
        feature_group[base_name] = next(
            (
                group_name
                for group_name, columns in schema.group_to_dynamic_columns.items()
                if base_name in columns
            ),
            'other',
        )

    for column_name in schema.static_env_columns:
        eda_df[column_name] = pd.to_numeric(dataframe[column_name], errors='coerce')
        feature_group[column_name] = next(
            (
                group_name
                for group_name, columns in schema.group_to_static_columns.items()
                if column_name in columns
            ),
            'other',
        )

    eda_df['label_final_name'] = dataframe['label_final_name'].astype(str)
    return eda_df, feature_group


def run_eda(csv_path: Path, output_dir: Path, labels_csv_path: Optional[Path] = None) -> Dict[str, str]:
    raw_df = read_csv_resilient(csv_path)
    merged_df = merge_with_labels(raw_df, data_csv_path=csv_path, labels_csv_path=labels_csv_path)
    state_slug = detect_state(merged_df, csv_path=csv_path)
    merged_df = canonicalize_integrated_labels(merged_df, state_slug=state_slug)

    if 'label_final_name' not in merged_df.columns:
        raise ValueError('Le dataframe fusionne ne contient pas label_final_name.')

    output_dir.mkdir(parents=True, exist_ok=True)
    eda_df, feature_group = build_temporal_mean_dataframe(merged_df, csv_path=csv_path)
    covariate_columns = [column for column in eda_df.columns if column != 'label_final_name']

    summary_df = eda_df[covariate_columns].describe().T
    summary_df['missing_count'] = eda_df[covariate_columns].isna().sum()
    summary_df['group'] = [feature_group[column] for column in summary_df.index]
    summary_csv = output_dir / 'covariate_summary.csv'
    summary_df.to_csv(summary_csv, encoding='utf-8')

    class_means_df = eda_df.groupby('label_final_name')[covariate_columns].mean()
    class_means_csv = output_dir / 'covariate_class_means.csv'
    class_means_df.to_csv(class_means_csv, encoding='utf-8')

    pearson_corr_df = eda_df[covariate_columns].corr(method='pearson')
    pearson_corr_csv = output_dir / 'covariate_pearson_corr.csv'
    pearson_corr_df.to_csv(pearson_corr_csv, encoding='utf-8')

    eta_series = pd.Series(
        {
            column_name: correlation_ratio(
                categories=eda_df['label_final_name'].to_numpy(),
                values=eda_df[column_name].to_numpy(),
            )
            for column_name in covariate_columns
        },
        name='eta',
    )
    eta_csv = output_dir / 'covariate_class_eta.csv'
    eta_series.to_csv(eta_csv, encoding='utf-8')

    save_heatmap(
        matrix=pearson_corr_df.to_numpy(),
        row_labels=pearson_corr_df.index.tolist(),
        column_labels=pearson_corr_df.columns.tolist(),
        title='Pearson correlation of temporal-mean climate and static covariates',
        output_path=output_dir / 'covariate_pearson_heatmap.png',
        cmap='coolwarm',
    )
    save_heatmap(
        matrix=class_means_df.to_numpy(),
        row_labels=class_means_df.index.tolist(),
        column_labels=class_means_df.columns.tolist(),
        title='Class-wise means of environmental covariates',
        output_path=output_dir / 'covariate_class_means_heatmap.png',
        cmap='viridis',
    )
    save_barplot(
        values=eta_series,
        title='Eta correlation ratio with crop classes',
        ylabel='Eta',
        output_path=output_dir / 'covariate_class_eta_barplot.png',
    )

    group_summary = {}
    for group_name in sorted(set(feature_group.values())):
        group_columns = [column for column, column_group in feature_group.items() if column_group == group_name]
        if not group_columns:
            continue
        group_summary[group_name] = eda_df[group_columns].describe().T.to_dict()

    group_summary_json = output_dir / 'covariate_group_summary.json'
    group_summary_json.write_text(json.dumps(group_summary, indent=2), encoding='utf-8')

    temporal_means_csv = output_dir / 'covariate_temporal_means.csv'
    eda_df.to_csv(temporal_means_csv, index=False, encoding='utf-8')

    return {
        'summary_csv': str(summary_csv),
        'class_means_csv': str(class_means_csv),
        'pearson_corr_csv': str(pearson_corr_csv),
        'eta_csv': str(eta_csv),
        'temporal_means_csv': str(temporal_means_csv),
        'pearson_heatmap_png': str(output_dir / 'covariate_pearson_heatmap.png'),
        'class_means_heatmap_png': str(output_dir / 'covariate_class_means_heatmap.png'),
        'eta_barplot_png': str(output_dir / 'covariate_class_eta_barplot.png'),
        'group_summary_json': str(group_summary_json),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EDA for real temporal environmental covariates.')
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--labels-csv', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_eda(
        csv_path=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        labels_csv_path=Path(args.labels_csv) if args.labels_csv else None,
    )
    print(json.dumps(artifacts, indent=2))


if __name__ == '__main__':
    main()

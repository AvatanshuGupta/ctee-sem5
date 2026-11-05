"""Explore the four electrical-load feature datasets and export reports/plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASET_FILES = {
    "train": "train.csv",
    "calibration": "calibration.csv",
    "test_noisy": "test_noisy.csv",
    "test_clean": "test_clean.csv",
}
LABEL_COLUMNS = ("behavior", "composition")
ROOT = Path(__file__).resolve().parent


def _read_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(data_dir / filename)
        for name, filename in DATASET_FILES.items()
    }


def _save_label_distribution_plot(
    datasets: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(LABEL_COLUMNS), figsize=(15, 5))
    for ax, label in zip(axes, LABEL_COLUMNS):
        plot_rows: list[dict[str, Any]] = []
        levels: set[str] = set()
        for dataset_name, frame in datasets.items():
            proportions = frame[label].value_counts(normalize=True, dropna=False)
            for value, proportion in proportions.items():
                category = str(value) if pd.notna(value) else "<missing>"
                levels.add(category)
                plot_rows.append(
                    {
                        "class": category,
                        "dataset": dataset_name,
                        "proportion": proportion,
                    }
                )

        sns.barplot(
            data=pd.DataFrame(plot_rows),
            x="class",
            y="proportion",
            hue="dataset",
            order=sorted(levels),
            hue_order=list(datasets),
            errorbar=None,
            ax=ax,
        )
        ax.set_title(f"{label} distribution")
        ax.set_xlabel(label)
        ax.set_ylabel("Fraction of rows")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Dataset")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_label_cross_tabs_plot(
    datasets: dict[str, pd.DataFrame],
    behavior_levels: list[str],
    composition_levels: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (dataset_name, frame) in zip(axes.flat, datasets.items()):
        table = pd.crosstab(
            frame["behavior"],
            frame["composition"],
            normalize="all",
        ).reindex(
            index=behavior_levels,
            columns=composition_levels,
            fill_value=0,
        )
        sns.heatmap(
            table,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{dataset_name}: behavior × composition")
        ax.set_xlabel("composition")
        ax.set_ylabel("behavior")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_missingness_plot(
    missing_percent: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(22, 5))
    sns.heatmap(
        missing_percent,
        cmap="magma",
        vmin=0,
        vmax=max(1.0, float(missing_percent.to_numpy().max())),
        cbar_kws={"label": "Missing values (%)"},
        ax=ax,
    )
    ax.set_title("Missing numeric values by feature and dataset")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Dataset")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_feature_distributions(
    dataset_name: str,
    frame: pd.DataFrame,
    numeric_columns: list[str],
    output_path: Path,
) -> None:
    columns = 5
    rows = int(np.ceil(len(numeric_columns) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(20, max(4, rows * 2.8)),
        squeeze=False,
    )

    for ax, column in zip(axes.flat, numeric_columns):
        values = (
            frame[column]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        if values.size:
            ax.hist(values, bins=30, color="#2878B5", alpha=0.85)
            ax.set_ylabel("Rows")
        else:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
        ax.set_title(column, fontsize=8)
        ax.tick_params(axis="both", labelsize=6)

    for ax in axes.flat[len(numeric_columns) :]:
        ax.set_visible(False)

    fig.suptitle(f"{dataset_name}: numeric feature distributions", y=1.002)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_correlation_plot(
    correlation: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(19, 17))
    sns.heatmap(
        correlation,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        mask=correlation.isna(),
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title(f"{dataset_name}: numeric feature correlations")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_feature_label_correlation_plot(
    correlation: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 19))
    sns.heatmap(
        correlation,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        mask=correlation.isna(),
        cbar_kws={"label": "Pearson correlation with one-hot label"},
        ax=ax,
    )
    ax.set_title(f"{dataset_name}: feature-to-label correlations")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyze_all_datasets(
    datasets: dict[str, pd.DataFrame] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze the four feature tables, returning reports and saving CSVs/plots."""
    data_dir = ROOT
    if datasets is None:
        datasets = _read_datasets(data_dir)
    else:
        datasets = {name: datasets[name] for name in DATASET_FILES}

    output_root = Path(output_dir) if output_dir is not None else ROOT / "analysis_output"
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    reference_columns = list(datasets["train"].columns)
    missing_labels = set(LABEL_COLUMNS).difference(reference_columns)
    if missing_labels:
        raise ValueError(f"Required label columns are missing: {sorted(missing_labels)}")
    for name, frame in datasets.items():
        if list(frame.columns) != reference_columns:
            raise ValueError(f"{name} does not have the same columns and order as train.")

    numeric_columns = datasets["train"].select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric feature columns were found.")

    overview_rows: list[dict[str, Any]] = []
    schema_by_column: dict[str, dict[str, Any]] = {
        column: {
            "role": "label" if column in LABEL_COLUMNS else "numeric feature"
        }
        for column in reference_columns
    }
    categorical_rows: list[dict[str, Any]] = []
    feature_statistics: dict[str, pd.DataFrame] = {}
    correlations: dict[str, pd.DataFrame] = {}
    feature_label_correlations: dict[str, pd.DataFrame] = {}
    missing_percent_rows: dict[str, pd.Series] = {}

    for name, frame in datasets.items():
        numeric = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
        overview_rows.append(
            {
                "dataset": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "numeric_features": len(numeric_columns),
                "missing_cells": int(frame.isna().sum().sum()),
                "missing_percent": 100 * frame.isna().to_numpy().mean(),
                "duplicate_rows": int(frame.duplicated().sum()),
                "infinite_numeric_values": int(
                    np.isinf(frame[numeric_columns].to_numpy(dtype=float)).sum()
                ),
            }
        )

        for column in reference_columns:
            schema_by_column[column].update(
                {
                    f"{name}_dtype": str(frame[column].dtype),
                    f"{name}_missing_count": int(frame[column].isna().sum()),
                    f"{name}_missing_percent": 100 * frame[column].isna().mean(),
                    f"{name}_unique_values": int(
                        frame[column].nunique(dropna=False)
                    ),
                }
            )

        for label in LABEL_COLUMNS:
            counts = frame[label].value_counts(dropna=False)
            for value, count in counts.items():
                categorical_rows.append(
                    {
                        "dataset": name,
                        "column": label,
                        "value": str(value) if pd.notna(value) else "<missing>",
                        "count": int(count),
                        "percent": 100 * count / len(frame),
                    }
                )

        stats = numeric.describe(
            percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
        ).T
        stats["missing_count"] = frame[numeric_columns].isna().sum()
        stats["missing_percent"] = 100 * frame[numeric_columns].isna().mean()
        stats["zero_count"] = numeric.eq(0).sum()
        stats["unique_count"] = numeric.nunique(dropna=True)
        stats["skew"] = numeric.skew()
        stats["kurtosis"] = numeric.kurtosis()
        feature_statistics[name] = stats
        stats.to_csv(output_root / f"feature_statistics_{name}.csv", index_label="feature")

        correlation = numeric.corr(method="pearson", min_periods=2)
        correlations[name] = correlation
        correlation.to_csv(output_root / f"correlations_{name}.csv", index_label="feature")

        label_indicators = pd.get_dummies(
            frame[list(LABEL_COLUMNS)].astype("category"),
            prefix=list(LABEL_COLUMNS),
            prefix_sep="=",
            dtype=float,
        )
        target_correlation = pd.concat([numeric, label_indicators], axis=1).corr(
            method="pearson",
            min_periods=2,
        ).loc[numeric_columns, label_indicators.columns]
        feature_label_correlations[name] = target_correlation
        target_correlation.to_csv(
            output_root / f"feature_label_correlations_{name}.csv",
            index_label="feature",
        )
        missing_percent_rows[name] = 100 * frame[numeric_columns].isna().mean()

    overview = pd.DataFrame(overview_rows).set_index("dataset")
    schema = pd.DataFrame.from_dict(schema_by_column, orient="index")
    schema.index.name = "column"
    categorical_distributions = pd.DataFrame(categorical_rows)
    missing_percent = pd.DataFrame(missing_percent_rows).T.reindex(
        columns=numeric_columns
    )

    overview.to_csv(output_root / "dataset_overview.csv")
    schema.to_csv(output_root / "columns_and_missingness.csv")
    categorical_distributions.to_csv(
        output_root / "categorical_distributions.csv",
        index=False,
    )

    behavior_levels = sorted(
        {
            str(value)
            for frame in datasets.values()
            for value in frame["behavior"].dropna().unique()
        }
    )
    composition_levels = sorted(
        {
            str(value)
            for frame in datasets.values()
            for value in frame["composition"].dropna().unique()
        }
    )
    joint_distributions: dict[str, pd.DataFrame] = {}
    for name, frame in datasets.items():
        joint = pd.crosstab(
            frame["behavior"],
            frame["composition"],
            normalize="all",
        ).reindex(
            index=behavior_levels,
            columns=composition_levels,
            fill_value=0,
        )
        joint_distributions[name] = joint
        joint.to_csv(output_root / f"label_cross_tab_{name}.csv", index_label="behavior")

    sns.set_theme(style="whitegrid")
    _save_label_distribution_plot(
        datasets,
        plots_dir / "label_distributions.png",
    )
    _save_label_cross_tabs_plot(
        datasets,
        behavior_levels,
        composition_levels,
        plots_dir / "label_cross_tabs.png",
    )
    _save_missingness_plot(
        missing_percent,
        plots_dir / "feature_missingness.png",
    )

    for name, frame in datasets.items():
        _save_feature_distributions(
            name,
            frame,
            numeric_columns,
            plots_dir / f"feature_distributions_{name}.png",
        )
        _save_correlation_plot(
            correlations[name],
            name,
            plots_dir / f"feature_correlations_{name}.png",
        )
        _save_feature_label_correlation_plot(
            feature_label_correlations[name],
            name,
            plots_dir / f"feature_label_correlations_{name}.png",
        )

    print(f"Analysis outputs: {output_root}")
    print(f"Columns ({len(reference_columns)}): {', '.join(reference_columns)}")
    print(f"Numeric features analyzed: {len(numeric_columns)}")
    print(f"Plots saved: {plots_dir}")

    return {
        "overview": overview,
        "columns": schema,
        "categorical_distributions": categorical_distributions,
        "feature_statistics": feature_statistics,
        "correlations": correlations,
        "feature_label_correlations": feature_label_correlations,
        "label_cross_tabs": joint_distributions,
        "missing_percent": missing_percent,
        "output_dir": output_root,
        "plots_dir": plots_dir,
    }


if __name__ == "__main__":
    reports = analyze_all_datasets()
    print("\nDataset overview:")
    print(reports["overview"].to_string())
    print("\nColumn report:")
    print(reports["columns"].to_string())

"""탐색적 데이터 분석 그래프와 다중공선성 진단."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import CASE_FEATURES, FEATURE_UNITS

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_eda(data: pd.DataFrame, figures_dir: Path) -> None:
    """분석에 사용되는 변수의 분포와 강도 관계를 PNG로 저장합니다."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    features = list(dict.fromkeys(f for case in CASE_FEATURES.values() for f in case))
    numeric = data[features + ["Compressive_Strength"]].apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(numeric.corr(method="pearson"), cmap="coolwarm", center=0, annot=True,
                fmt=".2f", square=True, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Pearson Correlation Matrix")
    _save(fig, figures_dir / "pearson_correlation_heatmap.png")

    columns = features + ["Compressive_Strength"]
    ncols = 4
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    for ax, column in zip(axes.flat, columns):
        sns.histplot(numeric[column].dropna(), kde=True, ax=ax)
        ax.set_title(f"{column} ({FEATURE_UNITS.get(column, 'MPa')})")
        ax.set_xlabel(FEATURE_UNITS.get(column, "MPa"))
        ax.set_ylabel("Count")
    for ax in axes.flat[len(columns):]:
        ax.remove()
    _save(fig, figures_dir / "variable_histograms.png")

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows))
    for ax, column in zip(axes.flat, columns):
        sns.boxplot(x=numeric[column], ax=ax, color="#79a9dc")
        ax.set_title(column)
        ax.set_xlabel(FEATURE_UNITS.get(column, "MPa"))
    for ax in axes.flat[len(columns):]:
        ax.remove()
    _save(fig, figures_dir / "variable_boxplots.png")

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    for ax, column in zip(axes.flat, features):
        sns.scatterplot(
            x=numeric[column], y=numeric["Compressive_Strength"],
            s=20, alpha=0.7, ax=ax,
        )
        ax.set_title(f"{column} vs Compressive Strength")
        ax.set_xlabel(f"{column} ({FEATURE_UNITS.get(column, '')})")
        ax.set_ylabel("Compressive Strength (MPa)")
    for ax in axes.flat[len(features):]:
        ax.remove()
    _save(fig, figures_dir / "feature_strength_scatterplots.png")


def calculate_vif(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """각 변수를 나머지 변수로 회귀해 VIF를 계산합니다."""
    x = data[features].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median()).to_numpy(dtype=float)
    rows = []
    for index, feature in enumerate(features):
        y = x[:, index]
        predictors = np.delete(x, index, axis=1)
        predictors = np.column_stack([np.ones(len(predictors)), predictors])
        predicted = predictors @ np.linalg.lstsq(predictors, y, rcond=None)[0]
        residual_ss = float(np.sum((y - predicted) ** 2))
        total_ss = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1 - residual_ss / total_ss if total_ss > 0 else 1.0
        vif = np.inf if r_squared >= 1 - 1e-12 else 1 / (1 - r_squared)
        rows.append({"Variable": feature, "VIF": vif, "Auxiliary_R2": r_squared})
    return pd.DataFrame(rows)

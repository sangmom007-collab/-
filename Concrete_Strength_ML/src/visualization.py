"""예측값, 잔차, 학습곡선, 모델 성능 시각화."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.eda import _save


def create_model_figures(
    predictions: pd.DataFrame, metrics: pd.DataFrame, figures_dir: Path
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    test = predictions[predictions["Split"].str.endswith(" - Test")]
    for (split, case, model), frame in test.groupby(["Split", "Case", "Model"]):
        label = f"{split} - {case} - {model}"
        safe = "_".join(label.lower().replace("-", " ").split())
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(
            data=frame, x="Actual_MPa", y="Predicted_MPa",
            alpha=0.8, s=35, ax=ax,
        )
        low = min(frame["Actual_MPa"].min(), frame["Predicted_MPa"].min())
        high = max(frame["Actual_MPa"].max(), frame["Predicted_MPa"].max())
        ax.plot([low, high], [low, high], "r--", label="Ideal: y = x")
        ax.set(
            title=f"Actual vs Predicted: {label}",
            xlabel="Actual Compressive Strength (MPa)",
            ylabel="Predicted Compressive Strength (MPa)",
        )
        ax.legend()
        _save(fig, figures_dir / f"actual_vs_predicted_{safe}.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            x=frame["Predicted_MPa"], y=frame["Residual_MPa"],
            alpha=0.8, s=35, ax=ax,
        )
        ax.axhline(0, color="red", linestyle="--")
        ax.set(
            title=f"Residual Plot: {label}",
            xlabel="Predicted Compressive Strength (MPa)",
            ylabel="Residual (Actual - Predicted, MPa)",
        )
        _save(fig, figures_dir / f"residual_plot_{safe}.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(frame["Residual_MPa"], kde=True, ax=ax)
        ax.set(
            title=f"Residual Distribution: {label}",
            xlabel="Residual (MPa)",
            ylabel="Count",
        )
        _save(fig, figures_dir / f"residual_distribution_{safe}.png")

    test_metrics = metrics[metrics["Data_Split"].str.startswith("Random Split")]
    for metric, ylabel in [("Test_R2", "Test R²"), ("Test_RMSE", "Test RMSE (MPa)"),
                           ("Test_MAE", "Test MAE (MPa)")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=test_metrics, x="Case", y=metric, hue="Model", ax=ax
        )
        ax.set_title(f"Model Comparison: {ylabel}")
        ax.set_ylabel(ylabel)
        _save(fig, figures_dir / f"{metric.lower()}_comparison.png")


def save_ann_loss_curves(models: dict, figures_dir: Path) -> None:
    for (split, case), estimator in models.items():
        model = estimator.named_steps["model"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, len(model.loss_curve_) + 1), model.loss_curve_)
        if model.validation_scores_ is not None:
            ax2 = ax.twinx()
            ax2.plot(
                range(1, len(model.validation_scores_) + 1),
                model.validation_scores_,
                color="tab:orange",
                alpha=0.7,
            )
            ax2.set_ylabel("Internal validation R²")
        ax.set(
            title=f"ANN Training Loss: {split} - {case}",
            xlabel="Iteration",
            ylabel="Training loss (MPa²-scale objective)",
        )
        safe = "_".join(f"{split}_{case}".lower().replace("-", " ").split())
        _save(fig, figures_dir / f"ann_loss_{safe}.png")

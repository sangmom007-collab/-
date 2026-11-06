"""SHAP 기반 전역 중요도와 주요 변수의 비선형 기여도 시각화."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHAP_FEATURES = {
    "W_C_Ratio": "W/C ratio",
    "UPV": "UPV (m/s)",
    "Rebound_Hardness": "Rebound hardness",
}


def _as_shap_matrix(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[0]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"지원하지 않는 SHAP 값 형태입니다: {array.shape}")
    return array


def run_shap_analysis(
    model_name: str,
    estimator,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    output_dir: Path,
    results_root: Path,
) -> dict:
    """Analyze the selected Case 3 model and save SHAP summary/dependence plots.

    The model is selected using training-only CV; SHAP values are calculated for
    a bounded sample of the held-out test set to keep model-agnostic explainers
    practical on this small-data project.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP가 설치되지 않았습니다. `pip install -r requirements.txt`를 실행하세요."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    features = list(x_train.columns)
    background = x_train.sample(
        n=min(30, len(x_train)), random_state=42
    ).copy()
    explain = x_test.sample(
        n=min(50, len(x_test)), random_state=42
    ).copy()
    underlying_model = estimator.named_steps["model"]

    fallback_reason = None
    if underlying_model.__class__.__name__ == "XGBRegressor":
        transformed_background = background
        transformed_explain = explain
        for _, transformer in list(estimator.steps)[:-1]:
            transformed_background = transformer.transform(transformed_background)
            transformed_explain = transformer.transform(transformed_explain)
        try:
            explainer = shap.TreeExplainer(underlying_model)
            values = explainer.shap_values(transformed_explain)
            plot_data = pd.DataFrame(
                transformed_explain, columns=features, index=explain.index
            )
            explanation_method = "TreeExplainer"
        except ValueError as exc:
            # Some SHAP versions cannot parse XGBoost 3.x JSON model metadata.
            fallback_reason = str(exc)
            explanation_method = "KernelExplainer (TreeExplainer compatibility fallback)"
    else:
        explanation_method = "KernelExplainer"

    if underlying_model.__class__.__name__ != "XGBRegressor" or fallback_reason:
        background_values = background.to_numpy()
        explain_values = explain.to_numpy()

        def predict_array(values):
            return estimator.predict(pd.DataFrame(values, columns=features))

        explainer = shap.KernelExplainer(predict_array, background_values)
        values = explainer.shap_values(
            explain_values, nsamples=100, l1_reg="num_features(10)"
        )
        plot_data = explain

    shap_values = _as_shap_matrix(values)
    if shap_values.shape != plot_data.shape:
        raise ValueError(
            f"SHAP 값과 설명 데이터 크기가 일치하지 않습니다: "
            f"{shap_values.shape} != {plot_data.shape}"
        )

    summary_path = output_dir / "shap_summary_plot.png"
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        plot_data,
        feature_names=features,
        show=False,
        max_display=min(14, len(features)),
    )
    plt.title(f"Case 3 SHAP Summary ({model_name})")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    dependence_path = output_dir / "shap_dependence_plot.png"
    feature_names = [name for name in SHAP_FEATURES if name in features]
    if not feature_names:
        raise ValueError("W_C_Ratio, UPV, Rebound_Hardness 설명 변수를 찾지 못했습니다.")
    fig, axes = plt.subplots(
        1, len(feature_names), figsize=(5.4 * len(feature_names), 4.8)
    )
    axes = np.atleast_1d(axes)
    color_features = {
        "W_C_Ratio": "UPV",
        "UPV": "Rebound_Hardness",
        "Rebound_Hardness": "UPV",
    }
    for ax, feature in zip(axes, feature_names):
        feature_index = features.index(feature)
        color_feature = color_features[feature]
        color_index = features.index(color_feature) if color_feature in features else None
        color_values = (
            plot_data[color_feature].to_numpy()
            if color_index is not None
            else "#2878B5"
        )
        scatter = ax.scatter(
            plot_data[feature],
            shap_values[:, feature_index],
            c=color_values,
            cmap="viridis" if color_index is not None else None,
            alpha=0.8,
            edgecolors="none",
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{SHAP_FEATURES[feature]} contribution")
        ax.set_xlabel(SHAP_FEATURES[feature])
        ax.set_ylabel("SHAP value (MPa contribution)")
        if color_index is not None:
            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label(SHAP_FEATURES.get(color_feature, color_feature))
    fig.suptitle(f"Case 3 SHAP Dependence ({model_name})")
    fig.tight_layout()
    fig.savefig(dependence_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    importance = pd.DataFrame(
        {
            "Feature": features,
            "Mean_Absolute_SHAP": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("Mean_Absolute_SHAP", ascending=False)
    importance.to_csv(
        output_dir / "shap_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "case": "Case 3",
        "model": model_name,
        "explainer": explanation_method,
        "tree_explainer_fallback_reason": fallback_reason,
        "background_rows": len(background),
        "explained_test_rows": len(explain),
        "plots": [str(summary_path), str(dependence_path)],
    }
    (output_dir / "shap_analysis.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for source in (summary_path, dependence_path):
        shutil.copy2(source, results_root / source.name)
    return {"metadata": metadata, "importance": importance}

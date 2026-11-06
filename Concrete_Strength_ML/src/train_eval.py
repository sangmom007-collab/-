"""교차검증, 독립 테스트 평가와 모델별 결과 저장을 담당합니다."""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import CASE_FEATURES, CV_FOLDS, MODEL_NAMES, RANDOM_STATE
from src.evaluation import prediction_frame, regression_metrics
from src.linear_regression import original_scale_coefficients
from src.models import create_model_pipeline


def _cross_validation_splitter(groups: pd.Series | None):
    if groups is None:
        from sklearn.model_selection import KFold

        return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    from sklearn.model_selection import GroupKFold

    fold_count = min(CV_FOLDS, int(groups.nunique()))
    if fold_count < 2:
        raise ValueError("출처 그룹 교차검증에는 학습 데이터에 2개 이상의 출처가 필요합니다.")
    return GroupKFold(n_splits=fold_count)


def _predict_with_optional_std(estimator, x: pd.DataFrame):
    """GPR에서는 예측 평균과 표준편차를 함께, 다른 모델은 평균만 반환합니다."""
    if estimator.named_steps["model"].__class__.__name__ != "GaussianProcessRegressor":
        return estimator.predict(x), None

    transformed = x
    for _, transformer in list(estimator.steps)[:-1]:
        transformed = transformer.transform(transformed)
    prediction, uncertainty = estimator.named_steps["model"].predict(
        transformed, return_std=True
    )
    return prediction, uncertainty


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def evaluate_experiments(
    x: pd.DataFrame,
    y: pd.Series,
    data: pd.DataFrame,
    splits: dict[str, dict],
    models_dir: Path,
) -> dict:
    """모든 모델·Case·분할을 평가하고 fold별/요약 지표를 반환합니다.

    교차검증은 각 최종 학습 분할의 학습 데이터에서만 수행합니다. Pipeline의
    결측 대체와 표준화기는 매 Fold의 학습 부분에서 새로 fit됩니다.
    """
    fold_records = []
    summary_records = []
    prediction_frames = []
    coefficient_frames = []
    fitted_models = {}
    model_config = {}
    shap_candidate = None

    for split_name, split in splits.items():
        train_index = split["train_index"]
        test_index = split["test_index"]
        all_groups = split.get("groups")
        train_groups = all_groups.loc[train_index] if all_groups is not None else None
        splitter = _cross_validation_splitter(train_groups)

        for case_name, features in CASE_FEATURES.items():
            x_train = x.loc[train_index, features]
            y_train = y.loc[train_index]
            x_test = x.loc[test_index, features]
            y_test = y.loc[test_index]

            for model_name in MODEL_NAMES:
                cv_scores = []
                fit_groups = (
                    train_groups.to_numpy()
                    if train_groups is not None
                    else None
                )
                for fold_number, (train_pos, valid_pos) in enumerate(
                    splitter.split(x_train, y_train, groups=fit_groups), start=1
                ):
                    fold_pipeline = create_model_pipeline(
                        model_name, n_features=len(features)
                    )
                    x_fold_train = x_train.iloc[train_pos]
                    y_fold_train = y_train.iloc[train_pos]
                    x_fold_valid = x_train.iloc[valid_pos]
                    y_fold_valid = y_train.iloc[valid_pos]
                    fold_pipeline.fit(x_fold_train, y_fold_train)
                    valid_prediction = fold_pipeline.predict(x_fold_valid)
                    scores = regression_metrics(y_fold_valid, valid_prediction)
                    cv_scores.append(scores)
                    fold_records.append(
                        {
                            "Data_Split": split_name,
                            "Case": case_name,
                            "Model": model_name,
                            "Fold": fold_number,
                            "Train_Rows": len(train_pos),
                            "Validation_Rows": len(valid_pos),
                            **scores,
                        }
                    )

                estimator = create_model_pipeline(model_name, n_features=len(features))
                estimator.fit(x_train, y_train)
                fitted_models[(split_name, case_name, model_name)] = estimator
                joblib.dump(
                    estimator,
                    models_dir
                    / f"{_safe_filename(split_name)}_{_safe_filename(case_name)}_"
                    f"{_safe_filename(model_name)}.joblib",
                )

                train_prediction, train_std = _predict_with_optional_std(
                    estimator, x_train
                )
                test_prediction, test_std = _predict_with_optional_std(
                    estimator, x_test
                )
                train_metrics = regression_metrics(y_train, train_prediction)
                test_metrics = regression_metrics(y_test, test_prediction)
                cv_frame = pd.DataFrame(cv_scores)
                cv_summary = {}
                for metric in ("RMSE", "R2", "MAE"):
                    cv_summary[f"CV_{metric}_Mean"] = float(cv_frame[metric].mean())
                    cv_summary[f"CV_{metric}_Std"] = float(cv_frame[metric].std(ddof=1))

                summary_records.append(
                    {
                        "Data_Split": split_name,
                        "Case": case_name,
                        "Model": model_name,
                        "Train_R2": train_metrics["R2"],
                        "Test_R2": test_metrics["R2"],
                        "Test_RMSE": test_metrics["RMSE"],
                        "Test_MAE": test_metrics["MAE"],
                        "Train_RMSE": train_metrics["RMSE"],
                        "Train_MAE": train_metrics["MAE"],
                        "Train_Rows": len(y_train),
                        "Test_Rows": len(y_test),
                        "CV_Folds": len(cv_frame),
                        **cv_summary,
                    }
                )

                for subset_name, subset_y, subset_prediction, subset_std in [
                    ("Train", y_train, train_prediction, train_std),
                    ("Test", y_test, test_prediction, test_std),
                ]:
                    frame = prediction_frame(
                        data,
                        subset_y,
                        subset_prediction,
                        f"{split_name} - {subset_name}",
                        case_name,
                        model_name,
                    )
                    if subset_std is not None:
                        frame["Prediction_Std_MPa"] = subset_std
                    prediction_frames.append(frame)

                if model_name == "Linear Regression":
                    coefficients = original_scale_coefficients(estimator, features)
                    coefficients.insert(0, "Case", case_name)
                    coefficients.insert(0, "Data_Split", split_name)
                    coefficient_frames.append(coefficients)

                if split_name == "Random Split" and case_name == "Case 3":
                    cv_rmse = cv_summary["CV_RMSE_Mean"]
                    if shap_candidate is None or cv_rmse < shap_candidate["cv_rmse"]:
                        shap_candidate = {
                            "cv_rmse": cv_rmse,
                            "model_name": model_name,
                            "estimator": estimator,
                            "features": features,
                            "x_train": x_train,
                            "x_test": x_test,
                            "y_test": y_test,
                        }

                model_config[f"{split_name} / {case_name} / {model_name}"] = {
                    "features": features,
                    "cv_rmse_mean": cv_summary["CV_RMSE_Mean"],
                    "cv_rmse_std": cv_summary["CV_RMSE_Std"],
                    "cv_folds": len(cv_frame),
                    "test_r2": test_metrics["R2"],
                    "test_rmse_mpa": test_metrics["RMSE"],
                    "test_mae_mpa": test_metrics["MAE"],
                }

    comparison = pd.DataFrame(summary_records)
    folds = pd.DataFrame(fold_records)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame(columns=["Data_Split", "Case", "Variable", "Coefficient"])
    )
    return {
        "comparison": comparison,
        "fold_metrics": folds,
        "predictions": predictions,
        "coefficients": coefficients,
        "fitted_models": fitted_models,
        "model_config": model_config,
        "shap_candidate": shap_candidate,
    }


def save_expanded_comparison(comparison: pd.DataFrame, fold_metrics: pd.DataFrame, results_dir: Path):
    """Save expanded model-by-case summaries at the project results root."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "model_comparison_expanded.csv"
    xlsx_path = results_dir / "model_comparison_expanded.xlsx"
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="Model_Comparison", index=False)
        fold_metrics.to_excel(writer, sheet_name="Fold_Metrics", index=False)
    return csv_path, xlsx_path

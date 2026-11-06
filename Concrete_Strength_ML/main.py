"""압축강도 모델 비교와 SHAP 해석 전체 실행 진입점."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CASE_FEATURES,
    COLUMN_MAP,
    DATA_FILE,
    GPR_ALPHA,
    GPR_OPTIMIZER_RESTARTS,
    MODEL_NAMES,
    RANDOM_STATE,
    RESULTS_ROOT,
    SOURCE_ID_COLUMN,
    SVR_C,
    SVR_EPSILON,
    SVR_GAMMA,
    TEST_SIZE,
    XGBOOST_PARAMETERS,
)
from src.data_loader import load_data, quality_reports
from src.data_split import make_splits
from src.eda import calculate_vif, create_eda
from src.explainability import run_shap_analysis
from src.preprocessing import add_derived_features, model_dataset, outlier_reports
from src.train_eval import evaluate_experiments, save_expanded_comparison
from src.visualization import create_model_figures, save_ann_loss_curves


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)


def run() -> Path:
    run_dir = RESULTS_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    predictions_dir = run_dir / "predictions"
    models_dir = run_dir / "models"
    for directory in (figures_dir, tables_dir, predictions_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=False)

    original = load_data(DATA_FILE)
    descriptive, missing, duplicates = quality_reports(original)
    data, ratio_checks = add_derived_features(original)
    outliers, impossible = outlier_reports(data)
    x_all, y_all = model_dataset(data)
    if len(x_all) < 10:
        raise ValueError(f"유효한 압축강도 데이터가 너무 적습니다: {len(x_all)}행")

    splits = make_splits(x_all, y_all, data.loc[x_all.index])
    split_records = []
    for split_name, split in splits.items():
        groups = split.get("groups")
        train_groups = (
            groups.loc[split["train_index"]] if groups is not None else None
        )
        test_groups = groups.loc[split["test_index"]] if groups is not None else None
        split_records.append(
            {
                "Data_Split": split_name,
                "Train_Rows": len(split["train_index"]),
                "Test_Rows": len(split["test_index"]),
                "Train_Source_Count": (
                    int(train_groups.nunique()) if train_groups is not None else None
                ),
                "Test_Source_Count": (
                    int(test_groups.nunique()) if test_groups is not None else None
                ),
            }
        )
    split_table = pd.DataFrame(split_records)

    evaluation = evaluate_experiments(x_all, y_all, data, splits, models_dir)
    comparison = evaluation["comparison"]
    fold_metrics = evaluation["fold_metrics"]
    predictions = evaluation["predictions"]
    coefficients = evaluation["coefficients"]
    fitted_models = evaluation["fitted_models"]

    data.to_csv(tables_dir / "processed_data.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        predictions_dir / "actual_vs_predicted.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        tables_dir / "model_comparison_expanded.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics.to_csv(
        tables_dir / "model_cv_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    outliers.to_csv(
        tables_dir / "statistical_outlier_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    impossible.to_csv(
        tables_dir / "physical_rule_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_excel(
        tables_dir / "analysis_results.xlsx",
        {
            "Model_Comparison": comparison,
            "Model_CV_Folds": fold_metrics,
            "Descriptive_Stats": descriptive,
            "Missingness": missing,
            "Duplicates": duplicates,
            "Ratio_Checks": ratio_checks,
            "IQR_Zscore_Candidates": outliers,
            "Physical_Candidates": impossible,
            "Linear_Coefficients": coefficients,
            "VIF": calculate_vif(data.loc[y_all.index], CASE_FEATURES["Case 1"]),
            "Data_Splits": split_table,
        },
    )

    # Keep the requested project-level files as the latest-run summary.
    save_expanded_comparison(comparison, fold_metrics, RESULTS_ROOT)

    create_eda(data.loc[y_all.index], figures_dir)
    create_model_figures(predictions, comparison, figures_dir)
    ann_models = {
        (split_name, case_name): estimator
        for (split_name, case_name, model_name), estimator in fitted_models.items()
        if model_name == "ANN"
    }
    save_ann_loss_curves(ann_models, figures_dir)

    shap_candidate = evaluation["shap_candidate"]
    if shap_candidate is None:
        raise RuntimeError("Case 3 학습 결과에서 SHAP 설명 대상 모델을 선택하지 못했습니다.")
    shap_result = run_shap_analysis(
        model_name=shap_candidate["model_name"],
        estimator=shap_candidate["estimator"],
        x_train=shap_candidate["x_train"],
        x_test=shap_candidate["x_test"],
        output_dir=figures_dir,
        results_root=RESULTS_ROOT,
    )
    shap_result["importance"].to_csv(
        tables_dir / "shap_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "input_file": DATA_FILE,
        "input_rows": len(original),
        "rows_with_target": len(y_all),
        "target_unit": "MPa",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "column_mapping": COLUMN_MAP,
        "source_id_available": SOURCE_ID_COLUMN in data.columns,
        "case_features": CASE_FEATURES,
        "models": list(MODEL_NAMES),
        "cross_validation": {
            "folds": 5,
            "random_split": "KFold(shuffle=True, random_state=42)",
            "source_split": "GroupKFold",
            "metrics": ["RMSE", "R2", "MAE"],
            "summary": "fold mean and sample standard deviation",
        },
        "model_parameters": {
            "xgboost": XGBOOST_PARAMETERS,
            "svr": {"kernel": "rbf", "C": SVR_C, "gamma": SVR_GAMMA, "epsilon": SVR_EPSILON},
            "gpr": {"kernel": "ConstantKernel * RBF", "alpha": GPR_ALPHA,
                    "n_restarts_optimizer": GPR_OPTIMIZER_RESTARTS},
        },
        "splits": split_table.to_dict(orient="records"),
        "model_results": evaluation["model_config"],
        "shap": shap_result["metadata"],
        "ratio_note": (
            "W_C_Ratio and Fine_Aggregate_Ratio are calculated as decimal ratios; "
            "original W/B and S/a percentage columns remain separate."
        ),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(_json_safe(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"입력 파일: {DATA_FILE}")
    print(f"분석 데이터: {len(y_all)}행 / 모델: {len(MODEL_NAMES)}종 / Case: 3종")
    print(f"Source_ID: {'있음' if SOURCE_ID_COLUMN in data.columns else '없음'}")
    for split_name in comparison["Data_Split"].drop_duplicates():
        print(f"\n{split_name} 테스트 성능:")
        print(
            comparison[comparison["Data_Split"] == split_name][
                ["Case", "Model", "CV_RMSE_Mean", "CV_RMSE_Std",
                 "Test_R2", "Test_RMSE", "Test_MAE"]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
    print(
        f"\nSHAP 설명 모델: {shap_candidate['model_name']} "
        f"(Case 3 CV RMSE={shap_candidate['cv_rmse']:.4f} MPa)"
    )
    print(f"확장 비교표: {RESULTS_ROOT / 'model_comparison_expanded.xlsx'}")
    print(f"결과 폴더: {run_dir}")
    return run_dir


if __name__ == "__main__":
    run()

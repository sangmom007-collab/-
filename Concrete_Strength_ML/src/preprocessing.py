"""파생변수 계산, 비율 감사, 이상치 후보 탐지."""

import numpy as np
import pandas as pd
from scipy.stats import zscore

from config import CASE_FEATURES, EXISTING_RATIO_COLUMNS, FEATURE_UNITS


def add_derived_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본을 보존하면서 요청된 비율을 소수 형태로 새로 계산합니다."""
    result = data.copy()
    cement = result["Cement"].replace(0, np.nan)
    aggregate_total = (
        result["Fine_Aggregate"] + result["Coarse_Aggregate"]
    ).replace(0, np.nan)
    result["W_C_Ratio"] = result["Water"] / cement
    result["Fine_Aggregate_Ratio"] = result["Fine_Aggregate"] / aggregate_total

    binder_total = (
        result["Cement"] + result["Fly_Ash"] + result["Slag"] + result["Silica_Fume"]
    ).replace(0, np.nan)
    calculated_w_b_percent = 100 * result["Water"] / binder_total
    ratio_checks = []
    comparison_specs = [
        (
            "Existing_W_B_Ratio_Percent",
            "W/B (물/결합재비, 기존 열)",
            calculated_w_b_percent,
        ),
        (
            "Existing_Fine_Aggregate_Ratio_Percent",
            "S/a (잔골재율, 기존 열)",
            100 * result["Fine_Aggregate_Ratio"],
        ),
    ]
    for column, label, calculated in comparison_specs:
        if column not in result:
            continue
        existing = result[column]
        comparable = existing.notna() & calculated.notna()
        difference = existing - calculated
        for index in result.index[comparable]:
            ratio_checks.append(
                {
                    "Excel_Row": result.at[index, "Excel_Row"],
                    "Ratio": label,
                    "Existing_Percent": existing.at[index],
                    "Calculated_Percent": calculated.at[index],
                    "Absolute_Difference_Percentage_Point": abs(difference.at[index]),
                    "Matches_Within_0.02_Point": abs(difference.at[index]) <= 0.02,
                }
            )
    return result, pd.DataFrame(ratio_checks)


def model_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """모든 Case가 동일한 행을 사용하도록 입력 누락을 포함한 한 데이터셋을 만듭니다."""
    features = list(dict.fromkeys(feature for case in CASE_FEATURES.values() for feature in case))
    target = data["Compressive_Strength"]
    valid_target = target.notna()
    x = data.loc[valid_target, features].copy()
    y = target.loc[valid_target].copy()
    # Excel의 "NaN" 문자열 등은 로더에서 결측치로 정규화됩니다.
    for column in x:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x.index.name = "Data_Index"
    y.index.name = "Data_Index"
    return x, y


def outlier_reports(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IQR/Z-score 통계 이상치와 단순 물리 제약 위반을 자동 삭제 없이 기록합니다."""
    numeric_columns = [
        column
        for column in list(dict.fromkeys(f for group in CASE_FEATURES.values() for f in group))
        + ["Compressive_Strength"]
        if column in data
    ]
    records = []
    impossible_rules = {
        column: "값이 0보다 작음"
        for column in [
            "Cement",
            "Water",
            "Coarse_Aggregate",
            "Fine_Aggregate",
            "Fly_Ash",
            "Slag",
            "Silica_Fume",
            "Superplasticizer",
            "AE_Agent",
            "UPV",
            "Rebound_Hardness",
            "Compressive_Strength",
        ]
    }
    impossible_rules["Age"] = "재령이 0 이하"
    impossible = []
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        observed = values.dropna()
        if observed.empty:
            continue
        q1, q3 = observed.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        standardized = pd.Series(np.nan, index=values.index, dtype=float)
        if observed.nunique() > 1:
            standardized.loc[observed.index] = zscore(observed, nan_policy="omit")
        iqr_mask = (values < low) | (values > high)
        z_mask = standardized.abs() > 3
        for index in values.index[iqr_mask.fillna(False) | z_mask.fillna(False)]:
            methods = []
            if bool(iqr_mask.loc[index]):
                methods.append("IQR")
            if bool(z_mask.loc[index]):
                methods.append("Z-score")
            records.append(
                {
                    "Excel_Row": data.at[index, "Excel_Row"],
                    "Variable": column,
                    "Value": values.at[index],
                    "Detection_Method": ", ".join(methods),
                    "IQR_Lower_Bound": low,
                    "IQR_Upper_Bound": high,
                    "Z_Score": standardized.at[index],
                    "Interpretation": "통계적 이상치 후보 (자동 삭제하지 않음)",
                }
            )
        rule = impossible_rules.get(column)
        if rule:
            invalid = values < 0 if column != "Age" else values <= 0
            for index in values.index[invalid.fillna(False)]:
                impossible.append(
                    {
                        "Excel_Row": data.at[index, "Excel_Row"],
                        "Variable": column,
                        "Value": values.at[index],
                        "Rule": rule,
                        "Interpretation": "물리 제약 위반 후보 (검토 필요)",
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(impossible)


def feature_units() -> dict[str, str]:
    return FEATURE_UNITS


def ratio_column_names() -> dict[str, str]:
    return EXISTING_RATIO_COLUMNS

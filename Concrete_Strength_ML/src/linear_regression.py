"""선형회귀 Pipeline 학습과 원단위 회귀계수 추출."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_linear_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def original_scale_coefficients(
    pipeline: Pipeline, features: list[str]
) -> pd.DataFrame:
    """표준화된 회귀계수를 원래 입력 단위 기준으로 변환합니다."""
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    coefficients = model.coef_ / scaler.scale_
    intercept = model.intercept_ - float(np.sum(coefficients * scaler.mean_))
    result = pd.DataFrame(
        {"Variable": features, "Coefficient": coefficients}
    )
    result.loc[len(result)] = ["Intercept", intercept]
    return result

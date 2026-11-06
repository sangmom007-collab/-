"""회귀 모델 평가 지표와 잔차 산출."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def prediction_frame(
    data: pd.DataFrame,
    y: pd.Series,
    prediction: np.ndarray,
    split_name: str,
    case_name: str,
    model_name: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "Data_Index": y.index,
            "Excel_Row": data.loc[y.index, "Excel_Row"].to_numpy(),
            "Split": split_name,
            "Case": case_name,
            "Model": model_name,
            "Actual_MPa": y.to_numpy(),
            "Predicted_MPa": prediction,
        }
    )
    result["Residual_MPa"] = result["Actual_MPa"] - result["Predicted_MPa"]
    return result

"""프로젝트 설정: 실제 Excel 열 이름과 분석 기본값을 관리합니다."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "concrete_data.xlsx"
SHEET_NAME = "DATA"
RESULTS_ROOT = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
ANN_MAX_ITER = 2000
ANN_HIDDEN_LAYERS = [(8,), (16, 8), (32, 16)]
ANN_DEFAULT_HIDDEN_LAYER = (16, 8)
SVR_C = 10.0
SVR_GAMMA = "scale"
SVR_EPSILON = 0.1
XGBOOST_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_jobs": 1,
}
GPR_ALPHA = 0.1
GPR_OPTIMIZER_RESTARTS = 1
MODEL_NAMES = [
    "Linear Regression",
    "ANN",
    "XGBoost",
    "SVR",
    "GPR",
]

# 실제 Excel 열 이름을 표준 변수명으로 연결합니다. 다른 파일을 사용할 때 수정하세요.
COLUMN_MAP = {
    "Cement": "Cement (kg/m^3)",
    "Water": "Water (kg/m^3)",
    "Fine_Aggregate": "Fine_Agg (kg/m^3)",
    "Coarse_Aggregate": "Coarse_Agg (kg/m^3)",
    "Fly_Ash": "Fly_Ash (kg/m^3)",
    "Slag": "Slag (kg/m^3)",
    "Silica_Fume": "Silica_Fume (kg/m^3)",
    "Superplasticizer": "SP (kg/m^3)",
    "AE_Agent": "AE_Agent (kg/m^3)",
    "Age": "Age (day)",
    "UPV": "UPV (m/s)",
    "Rebound_Hardness": "Rebound",
    "Compressive_Strength": "Strength (MPa)",
}

# 기존 파생변수 열은 계산한 변수와 혼동하지 않도록 별도로 감사합니다.
EXISTING_RATIO_COLUMNS = {
    "W_B_Ratio": "W_B_Ratio %",
    "Fine_Aggregate_Ratio": "S_a_Ratio %",
}
SOURCE_ID_COLUMN = "Source_ID"

MIX_FEATURES = [
    "Cement",
    "Water",
    "Coarse_Aggregate",
    "Fine_Aggregate",
    "Fly_Ash",
    "Slag",
    "Silica_Fume",
    "Superplasticizer",
    "AE_Agent",
]
CASE_FEATURES = {
    "Case 1": MIX_FEATURES + ["Age", "W_C_Ratio", "Fine_Aggregate_Ratio"],
    "Case 2": ["UPV", "Rebound_Hardness"],
    "Case 3": MIX_FEATURES
    + ["Age", "UPV", "Rebound_Hardness", "W_C_Ratio", "Fine_Aggregate_Ratio"],
}

FEATURE_UNITS = {
    "Cement": "kg/m³",
    "Water": "kg/m³",
    "Coarse_Aggregate": "kg/m³",
    "Fine_Aggregate": "kg/m³",
    "Fly_Ash": "kg/m³",
    "Slag": "kg/m³",
    "Silica_Fume": "kg/m³",
    "Superplasticizer": "kg/m³",
    "AE_Agent": "kg/m³",
    "Age": "day",
    "UPV": "m/s",
    "Rebound_Hardness": "rebound index",
    "W_C_Ratio": "dimensionless",
    "Fine_Aggregate_Ratio": "dimensionless",
    "Compressive_Strength": "MPa",
}

"""통일된 인터페이스로 회귀 모델 Pipeline을 생성합니다."""

from collections.abc import Mapping

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from config import (
    ANN_DEFAULT_HIDDEN_LAYER,
    ANN_MAX_ITER,
    GPR_ALPHA,
    GPR_OPTIMIZER_RESTARTS,
    RANDOM_STATE,
    SVR_C,
    SVR_EPSILON,
    SVR_GAMMA,
    XGBOOST_PARAMETERS,
)


MODEL_NAMES = ("Linear Regression", "ANN", "XGBoost", "SVR", "GPR")


def create_model_pipeline(
    model_name: str,
    n_features: int,
    settings: Mapping | None = None,
) -> Pipeline:
    """Create a preprocessing/model pipeline with fold-local imputation and scaling."""
    options = dict(settings or {})
    models = {
        "Linear Regression": (
            LinearRegression(),
            True,
        ),
        "ANN": (
            MLPRegressor(
                hidden_layer_sizes=options.get(
                    "hidden_layer_sizes", ANN_DEFAULT_HIDDEN_LAYER
                ),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=ANN_MAX_ITER,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
            ),
            True,
        ),
        "SVR": (
            SVR(
                kernel="rbf",
                C=options.get("C", SVR_C),
                gamma=options.get("gamma", SVR_GAMMA),
                epsilon=options.get("epsilon", SVR_EPSILON),
            ),
            True,
        ),
        "GPR": (
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-2, 1e2))
                * RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e2)),
                alpha=options.get("alpha", GPR_ALPHA),
                normalize_y=True,
                n_restarts_optimizer=options.get(
                    "n_restarts_optimizer", GPR_OPTIMIZER_RESTARTS
                ),
                random_state=RANDOM_STATE,
            ),
            True,
        ),
    }

    if model_name == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost가 필요합니다. 가상환경에서 `pip install -r requirements.txt`를 실행하세요."
            ) from exc
        parameters = {**XGBOOST_PARAMETERS, **options}
        parameters.setdefault("objective", "reg:squarederror")
        parameters.setdefault("random_state", RANDOM_STATE)
        estimator = XGBRegressor(**parameters)
        scale_features = False
    elif model_name in models:
        estimator, scale_features = models[model_name]
    else:
        raise ValueError(
            f"지원하지 않는 모델입니다: {model_name}. 지원 모델: {', '.join(MODEL_NAMES)}"
        )

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def available_models() -> tuple[str, ...]:
    """Return model labels accepted by the factory."""
    return MODEL_NAMES

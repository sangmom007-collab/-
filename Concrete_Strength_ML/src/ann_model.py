"""ANN 구조를 학습 데이터 내부 교차검증으로 선택합니다."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ANN_HIDDEN_LAYERS, ANN_MAX_ITER, RANDOM_STATE


def tune_ann(
    x: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: pd.Series | None = None,
) -> tuple[GridSearchCV, pd.DataFrame]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=ANN_MAX_ITER,
                    random_state=RANDOM_STATE,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={"model__hidden_layer_sizes": ANN_HIDDEN_LAYERS},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )
    fit_args = {"groups": groups} if groups is not None else {}
    search.fit(x, y, **fit_args)
    return search, pd.DataFrame(search.cv_results_)

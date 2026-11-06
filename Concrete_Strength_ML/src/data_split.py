"""모델 공통 데이터 분할과 교차검증 구성."""

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

from config import CV_FOLDS, RANDOM_STATE, TEST_SIZE, SOURCE_ID_COLUMN


def make_splits(
    x: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame
) -> dict[str, dict]:
    """동일한 행 분할을 모든 Case와 모델에 공유합니다."""
    random_train, random_test = train_test_split(
        x.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    splits = {
        "Random Split": {
            "train_index": random_train,
            "test_index": random_test,
            "groups": None,
        }
    }
    if SOURCE_ID_COLUMN in metadata:
        groups = metadata.loc[x.index, SOURCE_ID_COLUMN]
        if groups.isna().any():
            raise ValueError(
                f"{SOURCE_ID_COLUMN} 열에 결측이 있어 출처별 분할을 안전하게 수행할 수 없습니다. "
                "출처를 확인해 값을 채우거나 열을 수정하세요."
            )
        valid = groups.notna()
        if valid.all() and groups.nunique() >= 2:
            group_splitter = GroupShuffleSplit(
                n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            train_pos, test_pos = next(
                group_splitter.split(x, y, groups=groups)
            )
            group_train, group_test = x.index[train_pos], x.index[test_pos]
            splits["Source Group Split"] = {
                "train_index": group_train,
                "test_index": group_test,
                "groups": groups,
            }
    return splits


def make_cv(n_rows: int, groups: pd.Series | None = None):
    """무작위 분할은 KFold, 출처별 분할은 GroupKFold를 반환합니다."""
    folds = min(CV_FOLDS, n_rows)
    if groups is not None:
        folds = min(folds, groups.nunique())
        if folds < 2:
            raise ValueError("그룹 교차검증에는 학습 데이터에 최소 2개 출처가 필요합니다.")
        return GroupKFold(n_splits=folds)
    if folds < 2:
        raise ValueError("교차검증에는 학습 데이터가 최소 2행 필요합니다.")
    return KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

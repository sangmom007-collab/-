"""Excel 파일 로드와 원본 데이터 품질 확인."""

from pathlib import Path

import pandas as pd

from config import COLUMN_MAP, DATA_FILE, SHEET_NAME, SOURCE_ID_COLUMN


def load_data(path: Path = DATA_FILE, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    """Excel을 읽고 표준 변수명과 원본 행 번호를 부여합니다."""
    if not path.is_file():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path}\n"
            "Excel 파일을 프로젝트의 data 폴더에 넣고 config.py의 DATA_FILE을 확인하세요."
        )

    raw = pd.read_excel(path, sheet_name=sheet_name)
    raw.columns = [str(column).strip() for column in raw.columns]
    required = list(COLUMN_MAP.values())
    missing_columns = [column for column in required if column not in raw.columns]
    if missing_columns:
        raise ValueError(
            f"Excel 시트 '{sheet_name}'에 필요한 열이 없습니다: {missing_columns}\n"
            "config.py의 COLUMN_MAP을 실제 Excel 열 이름에 맞게 수정하세요."
        )

    data = raw.rename(columns={value: key for key, value in COLUMN_MAP.items()})
    data.insert(0, "Excel_Row", range(2, len(data) + 2))
    for existing_name, source_name in {
        "Existing_W_B_Ratio_Percent": "W_B_Ratio",
        "Existing_Fine_Aggregate_Ratio_Percent": "Fine_Aggregate_Ratio",
    }.items():
        source_column = next(
            (column for column in raw.columns if column == _existing_column(source_name)),
            None,
        )
        if source_column:
            data[existing_name] = raw[source_column].to_numpy()
    if SOURCE_ID_COLUMN in raw.columns:
        data[SOURCE_ID_COLUMN] = raw[SOURCE_ID_COLUMN].to_numpy()

    numeric_columns = list(COLUMN_MAP) + [
        "Existing_W_B_Ratio_Percent",
        "Existing_Fine_Aggregate_Ratio_Percent",
    ]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def _existing_column(key: str) -> str:
    from config import EXISTING_RATIO_COLUMNS

    return EXISTING_RATIO_COLUMNS[key]


def quality_reports(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """요약 통계, 결측치, 중복 행 정보를 반환합니다."""
    numeric = data.select_dtypes(include="number")
    descriptive = numeric.describe().T.rename_axis("Variable").reset_index()
    missing = pd.DataFrame(
        {
            "Variable": data.columns,
            "Missing_Count": data.isna().sum().to_numpy(),
            "Missing_Percent": (data.isna().mean() * 100).to_numpy(),
            "Data_Type": data.dtypes.astype(str).to_numpy(),
        }
    )
    duplicate_count = int(data.drop(columns=["Excel_Row"]).duplicated().sum())
    duplicates = pd.DataFrame(
        [{"Exact_Duplicate_Rows": duplicate_count, "Rows_Including_Header": len(data) + 1}]
    )
    return descriptive, missing, duplicates

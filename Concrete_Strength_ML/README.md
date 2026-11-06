# 콘크리트 압축강도 예측 모델

Excel 배합정보와 비파괴시험 데이터로 선형회귀, ANN, XGBoost, SVR, GPR을 학습하고
교차검증 및 테스트 성능을 비교합니다. Case 3의 CV 우수 모델에는 SHAP 해석을 수행합니다.
원본 Excel은 수정하지 않습니다.

## 실행 환경

- Python 3.11 이상 권장 (현재 검증은 Python 3.10.8에서도 수행 가능)
- Windows PowerShell, VS Code Python 확장

```powershell
cd "$HOME\Desktop\structure\Concrete_Strength_ML"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
code .
```

VS Code에서 `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv`를 선택합니다.
PowerShell이 가상환경 활성화를 막으면 해당 창에서 아래 명령 후 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 데이터와 실행

기본 파일은 `data\concrete_data.xlsx`, 기본 시트는 `DATA`입니다.
열 이름이 다르면 `config.py`의 `COLUMN_MAP` 및 `SHEET_NAME`을 수정합니다.

```powershell
python main.py
```

누락된 Excel 파일이나 열은 실행 초기에 구체적인 오류와 설정 위치를 안내합니다.

분석 완료 후 발표 자료를 생성하려면 아래 명령을 실행합니다.

```powershell
python make_presentation.py
```

최근 실행 폴더에 실제 분석 그래프와 성능표를 포함한 한국어 PowerPoint 파일을 저장합니다.

## 분석 정책

- 모든 Case에서 동일한 행과 80/20 무작위 분할을 공유하며 난수 시드는 42입니다.
- `Source_ID`가 있을 때만 별도의 출처 그룹 분할과 GroupKFold를 실행합니다.
- Linear Regression, ANN, SVR, GPR은 각 Fold 학습 내부에서 결측치 대체와 표준화를 수행합니다.
  XGBoost는 트리 모델로 표준화를 생략하고 결측치 대체만 Fold 학습 데이터로 fit합니다.
- 무작위 분할은 shuffle이 적용된 5-Fold KFold, 출처별 분할은 GroupKFold를 사용합니다.
  Fold별 RMSE/R²/MAE 및 평균·표준편차를 저장합니다.
- ANN, XGBoost, SVR, GPR 설정은 `config.py`와 `src/models.py`에서 조정할 수 있습니다.
- SHAP 대상 모델은 무작위 분할 Case 3의 학습 데이터 CV RMSE 평균으로 선택합니다.
  테스트 데이터는 선택에 사용하지 않으며, SHAP 시각화는 `results`와 실행별 폴더에 저장합니다.
- `W_C_Ratio`는 `Water/Cement`, `Fine_Aggregate_Ratio`는
  `Fine_Aggregate/(Fine_Aggregate+Coarse_Aggregate)`이며 둘 다 소수 비율입니다.
  원본 W/B 및 S/a 비율 열은 별도 보존하여 계산값과 비교합니다.
- 이상치 후보는 기록만 하고 삭제하지 않습니다.

## 출력

실행마다 `results\run_YYYYMMDD_HHMMSS` 폴더를 만들고 Excel 결과표, Fold/비교 CSV,
300 dpi PNG 그래프, SHAP 설명, 학습된 Joblib Pipeline 및 설정 JSON을 저장합니다.
최신 전체 비교표는 `results\model_comparison_expanded.csv` 및
`results\model_comparison_expanded.xlsx`에도 기록됩니다.
수치 성능은 실행한 모델의 실제 평가 결과만 출력합니다.

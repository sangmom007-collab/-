"""실행 결과를 이용해 한국어 발표용 PowerPoint를 생성합니다."""

import json
from pathlib import Path
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from config import RESULTS_ROOT

NAVY = RGBColor(20, 39, 67)
TEAL = RGBColor(0, 137, 145)
BLUE = RGBColor(52, 104, 168)
PALE = RGBColor(239, 245, 249)
INK = RGBColor(38, 51, 65)
MUTED = RGBColor(98, 112, 126)
WHITE = RGBColor(255, 255, 255)
ORANGE = RGBColor(226, 137, 81)
FONT = "Malgun Gothic"
MODEL_DISPLAY_NAMES = {
    "Linear Regression": "Linear Regression",
    "ANN": "ANN (MLPRegressor)",
    "XGBoost": "XGBoost (XGBRegressor)",
    "SVR": "SVR (RBF kernel)",
    "GPR": "GPR (RBF kernel)",
}


def add_text(slide, text, x, y, w, h, size=18, color=INK, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.03)
    frame.margin_right = Inches(0.03)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.01)
    frame.vertical_anchor = valign
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.space_after = Pt(5)
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color
    return box


def add_bullets(slide, items, x, y, w, h, size=17, color=INK, gap=9):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.06)
    frame.margin_right = Inches(0.04)
    frame.margin_top = Inches(0.04)
    for i, item in enumerate(items):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = f"•  {item}"
        paragraph.space_after = Pt(gap)
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = color
    return box


def add_card(slide, title, body, x, y, w, h, accent=TEAL):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = PALE
    shape.line.fill.background()
    stripe = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.09), Inches(h)
    )
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = accent
    stripe.line.fill.background()
    add_text(slide, title, x + 0.22, y + 0.15, w - 0.38, 0.38, 17, NAVY, True)
    add_text(slide, body, x + 0.22, y + 0.62, w - 0.40, h - 0.72, 13, INK)


def add_header(slide, title, number):
    add_text(slide, title, 0.55, 0.25, 12.1, 0.55, 26, NAVY, True)
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.58), Inches(0.92), Inches(12.15), Inches(0.035)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = TEAL
    rule.line.fill.background()
    add_text(slide, "콘크리트 압축강도 예측 | Capstone Design", 0.58, 7.16, 10.7, 0.2, 9, MUTED)
    add_text(slide, f"{number:02d}", 12.15, 7.13, 0.55, 0.22, 10, MUTED, align=PP_ALIGN.RIGHT)


def add_picture(slide, image_path, x, y, w, h):
    slide.shapes.add_picture(str(image_path), Inches(x), Inches(y), Inches(w), Inches(h))


def make_target_correlation(processed: pd.DataFrame, output: Path) -> None:
    features = [
        "UPV", "Rebound_Hardness", "Superplasticizer", "Cement", "W_C_Ratio",
        "Water", "Coarse_Aggregate", "Fly_Ash", "Fine_Aggregate_Ratio",
        "Fine_Aggregate", "AE_Agent", "Silica_Fume", "Age", "Slag",
    ]
    correlations = (
        processed[features + ["Compressive_Strength"]]
        .corr()["Compressive_Strength"]
        .drop("Compressive_Strength")
        .sort_values()
    )
    labels = {
        "UPV": "초음파속도(UPV)",
        "Rebound_Hardness": "반발경도",
        "Superplasticizer": "고성능감수제",
        "Cement": "시멘트",
        "W_C_Ratio": "물/시멘트비",
        "Water": "물",
        "Coarse_Aggregate": "굵은골재",
        "Fly_Ash": "플라이애쉬",
        "Fine_Aggregate_Ratio": "잔골재율",
        "Fine_Aggregate": "잔골재",
        "AE_Agent": "AE제",
        "Silica_Fume": "실리카퓸",
        "Age": "재령",
        "Slag": "슬래그",
    }
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ["#4c78a8" if value >= 0 else "#e07b63" for value in correlations]
    ax.barh([labels[name] for name in correlations.index], correlations.values, color=colors)
    ax.axvline(0, color="#394957", linewidth=0.8)
    ax.set_xlim(-0.55, 0.85)
    ax.set_xlabel("Pearson 상관계수 (r)")
    ax.set_title("압축강도와 입력변수 간 상관")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_age_distribution(processed: pd.DataFrame, output: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    counts = processed["Age"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(counts.index.astype(str), counts.values, color="#3a9a9f")
    ax.set_title("재령별 데이터 수")
    ax.set_xlabel("재령 (일)")
    ax.set_ylabel("데이터 수")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_results_table(slide, comparison: pd.DataFrame, split_name: str):
    models = [
        model
        for model in ["Linear Regression", "ANN", "XGBoost", "SVR", "GPR"]
        if model in set(comparison.loc[comparison["Data_Split"] == split_name, "Model"])
    ]
    cases = ["Case 1", "Case 2", "Case 3"]
    row_count = 1 + len(cases) * len(models)
    table = slide.shapes.add_table(
        row_count, 6, Inches(0.65), Inches(1.36), Inches(12.05), Inches(4.72)
    ).table
    widths = [1.1, 3.1, 1.5, 1.5, 2.25, 2.6]
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    headers = ["입력 Case", "모델", "Train R²", "Test R²", "Test RMSE (MPa)", "Test MAE (MPa)"]
    for col, value in enumerate(headers):
        cell = table.cell(0, col)
        cell.text = value
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
    for row_index, (case, model) in enumerate(
        [(case, model) for case in cases for model in models], start=1
    ):
        record = comparison.loc[
            (comparison["Data_Split"] == split_name)
            & (comparison["Case"] == case)
            & (comparison["Model"] == model)
        ].iloc[0]
        values = [
            case,
            MODEL_DISPLAY_NAMES.get(model, model),
            f"{record['Train_R2']:.3f}",
            f"{record['Test_R2']:.3f}",
            f"{record['Test_RMSE']:.2f}",
            f"{record['Test_MAE']:.2f}",
        ]
        for col, value in enumerate(values):
            cell = table.cell(row_index, col)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(228, 242, 242) if case == "Case 3" else WHITE
    for row in table.rows:
        row.height = Inches(4.72 / row_count)
        for cell in row.cells:
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
                paragraph.font.name = FONT
                paragraph.font.size = Pt(11 if row_count > 8 else 13)
                paragraph.font.bold = row is table.rows[0]
                paragraph.font.color.rgb = WHITE if row is table.rows[0] else INK


def build_presentation(run_dir: Path) -> Path:
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    processed = pd.read_csv(tables_dir / "processed_data.csv")
    comparison = pd.read_excel(
        tables_dir / "analysis_results.xlsx", sheet_name="Model_Comparison"
    )
    shap_metadata = json.loads(
        (figures_dir / "shap_analysis.json").read_text(encoding="utf-8")
    )
    shap_importance = pd.read_csv(figures_dir / "shap_feature_importance.csv")
    split_summary = pd.read_excel(
        tables_dir / "analysis_results.xlsx", sheet_name="Data_Splits"
    )
    vif = pd.read_excel(tables_dir / "analysis_results.xlsx", sheet_name="VIF")
    source_count = int(processed["Source_ID"].nunique())
    source_sizes = processed["Source_ID"].value_counts()
    group_split = split_summary.loc[
        split_summary["Data_Split"] == "Source Group Split"
    ]
    random_case3 = comparison.loc[
        (comparison["Data_Split"] == "Random Split")
        & (comparison["Case"] == "Case 3")
    ].sort_values("CV_RMSE_Mean")
    random_case2 = comparison.loc[
        (comparison["Data_Split"] == "Random Split")
        & (comparison["Case"] == "Case 2")
    ].sort_values("CV_RMSE_Mean")
    best_random_case3 = random_case3.iloc[0]
    best_random_case2 = random_case2.iloc[0]
    best_model_name = str(best_random_case3["Model"])
    best_model_slug = "_".join(
        f"Random Split - Test - Case 3 - {best_model_name}"
        .lower()
        .replace("-", " ")
        .split()
    )

    correlation_path = figures_dir / "presentation_target_correlation.png"
    age_path = figures_dir / "presentation_age_distribution.png"
    make_target_correlation(processed, correlation_path)
    make_age_distribution(processed, age_path)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. 표지
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.82), Inches(1.12), Inches(0.12), Inches(4.95)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()
    add_text(slide, "CAPSTONE DESIGN  |  토목공학", 1.28, 1.28, 10.3, 0.45, 15, RGBColor(119, 213, 207), True)
    add_text(slide, "배합정보와 비파괴시험 데이터를 활용한\n머신러닝 기반 콘크리트 압축강도 예측", 1.26, 2.0, 10.9, 1.75, 30, WHITE, True)
    add_text(slide, "선형회귀와 인공신경망(ANN)의 성능 비교", 1.3, 4.12, 9.5, 0.6, 20, RGBColor(220, 232, 241))
    add_text(slide, "Excel 실험데이터 356행  |  Train/Test = 80/20  |  Random State = 42", 1.3, 5.45, 10.7, 0.45, 13, RGBColor(193, 207, 220))
    add_text(slide, "발표자: ____________________", 1.3, 6.35, 7.0, 0.35, 13, WHITE)

    # 2. 연구 질문과 전체 흐름
    slide = prs.slides.add_slide(blank)
    add_header(slide, "연구 목적과 분석 흐름", 2)
    add_text(slide, "연구 질문", 0.72, 1.28, 2.0, 0.38, 18, TEAL, True)
    add_text(slide, "배합·재령·비파괴시험 정보를 함께 사용할 때\n압축강도 예측 성능은 얼마나 개선되는가?", 0.75, 1.72, 11.6, 0.9, 25, NAVY, True)
    steps = [
        ("01", "Excel 데이터", "356개 실험 관측치"),
        ("02", "전처리·EDA", "결측·파생비율·상관"),
        ("03", "공통 분할", "학습 284 / 시험 72"),
        ("04", "모델 비교", "Linear · ANN · XGBoost · SVR · GPR"),
    ]
    for i, (num, title, body) in enumerate(steps):
        x = 0.75 + i * 3.12
        add_card(slide, f"{num}  {title}", body, x, 3.35, 2.75, 1.55, TEAL if i < 3 else ORANGE)
        if i < 3:
            add_text(slide, "→", x + 2.78, 3.88, 0.35, 0.45, 24, MUTED, True, PP_ALIGN.CENTER)
    add_text(slide, "평가 지표: R² (설명력) · RMSE (큰 오차에 민감) · MAE (평균 절대 오차)", 0.82, 5.55, 11.8, 0.5, 17, INK, True)

    # 3. 데이터 현황
    slide = prs.slides.add_slide(blank)
    add_header(slide, "데이터 구성과 품질 점검", 3)
    add_card(slide, "분석 규모", "356행 · 입력변수 14개 · 출력변수 1개\n압축강도 평균 35.0 MPa\n범위 9.32–87.28 MPa", 0.72, 1.3, 3.45, 1.55)
    add_card(slide, "결측 및 정제", "UPV 결측 24행 (6.7%)\n문자열 NaN을 결측으로 정규화\n학습 Pipeline의 중앙값 대체", 0.72, 3.05, 3.45, 1.55, ORANGE)
    add_card(slide, "데이터 분할", "Train 284행 (80%)\nTest 72행 (20%)\n모든 Case·모델에 동일 행 사용", 0.72, 4.8, 3.45, 1.55, BLUE)
    add_picture(slide, age_path, 4.48, 1.32, 7.9, 3.5)
    add_text(slide, "재령은 3–2,555일, 14개 값에 분포하며 28일 데이터가 107행으로 가장 많음.", 4.6, 5.02, 7.7, 0.65, 15, INK)
    add_text(
        slide,
        f"Source_ID {source_count}개 출처 확인 (출처별 행 수 {source_sizes.min()}–{source_sizes.max()})",
        4.58, 5.82, 7.9, 0.5, 15, TEAL, True,
    )

    # 4. 입력변수 Case
    slide = prs.slides.add_slide(blank)
    add_header(slide, "입력변수 조합: 세 가지 Case", 4)
    add_card(slide, "Case 1 | 배합·재령", "배합 9개 + 재령 + W/C + S/a\n총 12개 입력변수\n배합설계 기반 기준 모델", 0.75, 1.52, 3.82, 2.0, BLUE)
    add_card(slide, "Case 2 | 비파괴시험", "UPV + 반발경도\n총 2개 입력변수\n현장 비파괴 정보의 예측력 평가", 4.78, 1.52, 3.82, 2.0, ORANGE)
    add_card(slide, "Case 3 | 통합", "배합 9개 + 재령 + UPV·반발경도 + W/C·S/a\n총 14개 입력변수\n정보 통합의 추가 가치 평가", 8.82, 1.52, 3.82, 2.0, TEAL)
    add_text(slide, "파생비율 정의", 0.82, 4.18, 2.2, 0.42, 18, TEAL, True)
    add_text(slide, "W/C = Water ÷ Cement     |     S/a = Fine Aggregate ÷ (Fine + Coarse Aggregate)", 0.85, 4.76, 11.8, 0.5, 20, NAVY, True)
    add_text(slide, "두 비율은 소수로 재계산했습니다. Excel의 기존 W/B(물/결합재비) 및 S/a 열은 별도 보존해 비교했습니다.", 0.85, 5.55, 11.7, 0.75, 15, INK)

    # 5. 모델링·누출 방지
    slide = prs.slides.add_slide(blank)
    add_header(slide, "모델링 설계와 데이터 누출 방지", 5)
    add_card(slide, "전처리 Pipeline", "결측 대체: 학습 데이터 중앙값\n표준화: 학습 데이터로 평균·표준편차 계산\n테스트 데이터에는 학습 변환만 적용", 0.75, 1.45, 5.6, 1.72)
    add_card(slide, "선형회귀", "StandardScaler + LinearRegression\nBaseline 비교 모델\n계수는 원래 단위로 환산해 저장", 6.75, 1.45, 5.6, 1.72, BLUE)
    add_card(slide, "ANN / MLPRegressor", "완전연결 인공신경망\nReLU · Adam · 조기 종료\n학습 데이터 내부 5-Fold CV", 0.75, 3.48, 5.6, 1.72, ORANGE)
    add_card(slide, "비선형·확률 모델", "XGBoost: 트리 앙상블\nSVR: RBF 커널 회귀\nGPR: RBF 커널 + 예측 불확실성", 6.75, 3.48, 5.6, 1.72)
    add_text(slide, "Source_ID 기준 GroupShuffleSplit 및 학습 내부 GroupKFold도 별도로 평가", 0.85, 5.78, 11.7, 0.48, 16, TEAL, True)

    # 6. 상관분석
    slide = prs.slides.add_slide(blank)
    add_header(slide, "탐색적 분석: 상관관계", 6)
    add_picture(slide, correlation_path, 0.55, 1.12, 7.0, 5.55)
    add_bullets(
        slide,
        [
            "UPV와 압축강도: r = +0.73",
            "반발경도와 압축강도: r = +0.72",
            "고성능감수제: r = +0.59, 시멘트: r = +0.55",
            "W/C: r = −0.43, 잔골재율: r = −0.40",
            "재령 단독 상관: r = +0.02",
        ],
        7.78, 1.8, 4.75, 3.55, 16,
    )
    add_text(slide, "주의: Pearson 상관은 선형 연관성 요약이며 인과관계를 의미하지 않습니다. 재령 효과는 배합·출처 등과 함께 해석해야 합니다.", 7.82, 5.45, 4.6, 0.95, 13, MUTED)

    # 7. 전체 변수 상관행렬
    slide = prs.slides.add_slide(blank)
    add_header(slide, "전체 입력변수 상관행렬", 7)
    add_picture(slide, figures_dir / "pearson_correlation_heatmap.png", 0.45, 1.1, 8.35, 5.7)
    add_bullets(
        slide,
        [
            "물–굵은골재 상관: r = +0.93",
            "잔골재–잔골재율: r = +0.68",
            "UPV–반발경도: r = +0.62",
            "배합량 변수는 서로 함께 변하는 경향",
        ],
        9.0, 1.75, 3.65, 3.2, 15,
    )
    add_text(slide, "입력변수 간 상관이 높으면 선형회귀 계수의 불확실성이 커질 수 있습니다.", 9.02, 5.15, 3.55, 1.05, 13, ORANGE, True)

    # 8. 결과표
    slide = prs.slides.add_slide(blank)
    add_header(slide, "테스트 성능 비교: 모델 5종 × Case 3종", 8)
    add_results_table(slide, comparison, "Random Split")
    add_text(
        slide,
        f"Case 3 학습 CV 최우수: {best_model_name} "
        f"(CV RMSE {best_random_case3['CV_RMSE_Mean']:.2f} MPa) · "
        f"Test RMSE {best_random_case3['Test_RMSE']:.2f} MPa",
        0.82, 6.2, 11.8, 0.48, 15, TEAL, True,
    )
    add_text(slide, "모든 성능 수치는 72개 테스트 데이터에서 실제 계산한 값입니다.", 0.83, 6.72, 11.7, 0.3, 11, MUTED)

    # 9. 실험 출처를 분리한 일반화 성능
    if not group_split.empty:
        slide = prs.slides.add_slide(blank)
        add_header(slide, "출처 그룹 분할: 새로운 실험 출처에 대한 평가", 9)
        add_results_table(slide, comparison, "Source Group Split")
        group_record = comparison.loc[
            (comparison["Data_Split"] == "Source Group Split")
            & (comparison["Case"] == "Case 3")
        ].sort_values("CV_RMSE_Mean").iloc[0]
        split_record = group_split.iloc[0]
        add_text(
            slide,
            f"GroupShuffleSplit: 학습 {int(split_record['Train_Rows'])}행 / "
            f"시험 {int(split_record['Test_Rows'])}행 · 출처 "
            f"{int(split_record['Train_Source_Count'])}개 / "
            f"{int(split_record['Test_Source_Count'])}개",
            0.82, 6.13, 11.8, 0.35, 14, INK, True,
        )
        add_text(
            slide,
            f"Case 3 {group_record['Model']}: Test R² {group_record['Test_R2']:.3f}, "
            f"RMSE {group_record['Test_RMSE']:.2f} MPa, "
            f"MAE {group_record['Test_MAE']:.2f} MPa "
            "(모델 선택은 학습 그룹 CV 기준).",
            0.82, 6.52, 11.8, 0.48, 12, TEAL, True,
        )

    # 10. 성능 시각화
    slide = prs.slides.add_slide(blank)
    add_header(slide, "무작위 분할 성능: 통합 Case의 이점", 10)
    add_picture(slide, figures_dir / "test_rmse_comparison.png", 0.52, 1.25, 7.3, 4.65)
    add_card(
        slide,
        f"Case 3 {best_model_name}",
        f"Test R² {best_random_case3['Test_R2']:.3f}\n"
        f"RMSE {best_random_case3['Test_RMSE']:.2f} MPa\n"
        f"MAE {best_random_case3['Test_MAE']:.2f} MPa\n"
        f"CV RMSE {best_random_case3['CV_RMSE_Mean']:.2f} MPa",
        8.15, 1.62, 4.2, 2.0, TEAL,
    )
    add_card(
        slide,
        f"Case 2 {best_random_case2['Model']}",
        f"비파괴 2개 변수\nTest R² {best_random_case2['Test_R2']:.3f}\n"
        f"RMSE {best_random_case2['Test_RMSE']:.2f} MPa · "
        f"MAE {best_random_case2['Test_MAE']:.2f} MPa",
        8.15, 4.03, 4.2, 1.8, ORANGE,
    )
    add_text(slide, "해석: 입력정보의 조합과 데이터 분포에 따라 모델 우열이 달라짐", 0.82, 6.16, 11.4, 0.42, 16, NAVY, True)

    # 11. 최적 모델 예측/잔차 진단
    slide = prs.slides.add_slide(blank)
    add_header(slide, f"최적 조합 진단: Case 3 {best_model_name}", 11)
    add_picture(slide, figures_dir / f"actual_vs_predicted_{best_model_slug}.png", 0.55, 1.25, 6.15, 4.8)
    add_picture(slide, figures_dir / f"residual_plot_{best_model_slug}.png", 6.85, 1.35, 5.85, 4.65)
    add_text(slide, "좌: 예측값이 y=x 기준선 주변에 분포. 우: 잔차는 0 주변에 모이나 일부 큰 오차가 남아 있음.", 0.82, 6.24, 11.7, 0.45, 15, INK)

    # 12. ANN 선택 및 과적합 확인
    slide = prs.slides.add_slide(blank)
    add_header(slide, "ANN (MLPRegressor) 구조와 과적합 점검", 12)
    add_picture(slide, figures_dir / "ann_loss_random_split_case_3.png", 0.55, 1.35, 6.25, 4.75)
    ann_case3 = comparison.loc[
        (comparison["Data_Split"] == "Random Split")
        & (comparison["Case"] == "Case 3")
        & (comparison["Model"] == "ANN")
    ].iloc[0]
    add_card(
        slide,
        "ANN 설정 및 내부 CV",
        "은닉층 구조: (16, 8)\n"
        f"5-Fold CV RMSE: {ann_case3['CV_RMSE_Mean']:.2f} "
        f"± {ann_case3['CV_RMSE_Std']:.2f} MPa",
        7.15, 1.55, 5.15, 1.65,
    )
    add_card(
        slide,
        "Train/Test 비교",
        f"Train R² {ann_case3['Train_R2']:.3f} → "
        f"Test R² {ann_case3['Test_R2']:.3f}\n"
        f"Train RMSE {ann_case3['Train_RMSE']:.2f} → "
        f"Test RMSE {ann_case3['Test_RMSE']:.2f} MPa\n"
        "단일 분할 결과이며 반복 검증이 필요",
        7.15, 3.55, 5.15, 1.85, BLUE,
    )
    add_text(slide, "단, 단일 분할 결과만으로 일반화 성능을 확정할 수는 없습니다.", 7.2, 5.78, 5.0, 0.55, 14, ORANGE, True)

    # 13. 해석 한계
    slide = prs.slides.add_slide(blank)
    add_header(slide, "결과 해석 시 한계와 주의사항", 13)
    top_vif = vif.sort_values("VIF", ascending=False).head(4)
    vif_text = "\n".join(f"{r.Variable}: VIF {r.VIF:.1f}" for r in top_vif.itertuples())
    add_card(slide, "다중공선성", f"VIF가 매우 큰 변수들이 확인됨\n{vif_text}\n선형회귀 계수의 개별 해석에 주의", 0.75, 1.45, 5.6, 2.0, ORANGE)
    add_card(slide, "비율 정의 확인", "기존 W/B 열은 요청한 W/C와 다름\n기존 비율값은 재계산 결과와 불일치\n분석에는 명시된 식으로 만든 W/C·S/a 사용", 6.75, 1.45, 5.6, 2.0)
    add_card(slide, "출처 일반화", "12개 출처 중 3개를 그룹 테스트로 홀드아웃\n출처별 표본 수 6–84행으로 불균형\n한 번의 그룹 분할이므로 결과 변동성 주의", 0.75, 3.82, 5.6, 1.8, BLUE)
    add_card(slide, "데이터 규모와 편중", "총 356행, 재령 분포 불균형\n28일 데이터 107행\n이상치 후보는 삭제하지 않고 별도 기록", 6.75, 3.82, 5.6, 1.8, TEAL)
    add_text(slide, "추가 출처 분할 반복 및 독립 실험 데이터 외부 검증으로 일반화 성능을 더 확인해야 합니다.", 0.85, 6.12, 11.7, 0.48, 15, NAVY, True)

    # 14. SHAP 전역 중요도와 비선형 기여도
    slide = prs.slides.add_slide(blank)
    add_header(slide, f"SHAP 해석: Case 3 {shap_metadata['model']}", 14)
    add_picture(
        slide, figures_dir / "shap_summary_plot.png",
        0.45, 1.28, 6.1, 4.85,
    )
    add_picture(
        slide, figures_dir / "shap_dependence_plot.png",
        6.72, 1.48, 6.05, 2.45,
    )
    top_features = shap_importance.head(4)
    importance_text = "\n".join(
        f"{row.Feature}: 평균 |SHAP| {row.Mean_Absolute_SHAP:.2f} MPa"
        for row in top_features.itertuples()
    )
    add_text(slide, "영향이 큰 변수", 6.88, 4.12, 3.5, 0.35, 17, TEAL, True)
    add_text(slide, importance_text, 6.9, 4.55, 5.6, 1.15, 14, INK)
    add_text(
        slide,
        f"설명기법: {shap_metadata['explainer']}. "
        "SHAP는 선택된 모델의 예측 기여도를 설명하며 인과관계를 뜻하지 않습니다.",
        0.82, 6.28, 11.8, 0.45, 12, MUTED,
    )

    # 15. 결론과 후속 연구
    slide = prs.slides.add_slide(blank)
    add_header(slide, "결론 및 후속 연구", 15)
    group_best = comparison.loc[
        (comparison["Data_Split"] == "Source Group Split")
        & (comparison["Case"] == "Case 3")
    ].sort_values("CV_RMSE_Mean")
    group_text = (
        f"Case 3 {group_best.iloc[0]['Model']} (그룹 CV 선택)\n"
        f"(R² {group_best.iloc[0]['Test_R2']:.3f}, "
        f"RMSE {group_best.iloc[0]['Test_RMSE']:.2f} MPa)"
        if not group_best.empty
        else "출처 그룹 분할 결과를 추가로 평가해야 함"
    )
    add_card(
        slide,
        "결론 1 | 무작위 분할",
        f"Case 3 {best_model_name} (학습 CV 선택)\n"
        f"(R² {best_random_case3['Test_R2']:.3f}, "
        f"RMSE {best_random_case3['Test_RMSE']:.2f} MPa)",
        0.75, 1.45, 5.75, 1.7,
    )
    add_card(slide, "결론 2 | 출처 홀드아웃", group_text, 6.78, 1.45, 5.75, 1.7, ORANGE)
    add_text(slide, "다음 단계", 0.85, 3.72, 2.2, 0.45, 20, TEAL, True)
    add_bullets(
        slide,
        [
            "출처별 반복 그룹 분할로 현재 한 번의 분할 결과 변동성 확인",
            "새로운 독립 연구자료를 외부 테스트셋으로 추가",
            "재령·배합 구간별 오차와 이상치 후보를 공학적으로 검토",
            "다중공선성 완화 후 회귀계수 해석 및 불확실성 평가",
        ],
        0.9, 4.28, 11.7, 1.8, 16,
    )
    add_text(slide, "질문 감사합니다", 0.85, 6.43, 11.6, 0.45, 23, NAVY, True, PP_ALIGN.CENTER)

    output = run_dir / "concrete_strength_ml_presentation.pptx"
    if output.exists():
        output = run_dir / "concrete_strength_ml_presentation_updated.pptx"
    if output.exists():
        output = run_dir / (
            f"concrete_strength_ml_presentation_{datetime.now():%Y%m%d_%H%M%S}.pptx"
        )
    prs.save(output)
    return output


if __name__ == "__main__":
    runs = sorted(RESULTS_ROOT.glob("run_*"))
    if not runs:
        raise FileNotFoundError("results 폴더에 실행 결과가 없습니다. 먼저 python main.py를 실행하세요.")
    presentation = build_presentation(runs[-1])
    print(f"PPTX 생성 완료: {presentation}")

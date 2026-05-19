import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# BÀI 12 — ĐỒ ÁN TÍCH HỢP AIDEOM-VN
# Integrated prototype: Forecast + Readiness + Allocation
# + Labor + Risk + Decision Dashboard
# =========================================================


# ---------------------------------------------------------
# 0. STYLE + COMPONENTS
# ---------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 12px;
            border-radius: 14px;
        }
        .small-note {
            font-size: 14px;
            opacity: 0.84;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def sticker_header(icon, title, subtitle):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, rgba(14,165,233,0.18), rgba(139,92,246,0.14));
            padding: 18px 22px;
            border-radius: 18px;
            margin: 10px 0 18px 0;
            border: 1px solid rgba(255,255,255,0.08);
        ">
            <div style="font-size: 28px; font-weight: 850;">{icon} {title}</div>
            <div style="font-size: 16px; opacity: 0.88; margin-top: 6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def policy_card(icon, title, body, tone="info"):
    colors = {
        "info": "#0ea5e9",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "purple": "#8b5cf6",
        "gray": "#64748b",
    }
    color = colors.get(tone, "#0ea5e9")

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {color};
            background: rgba(255,255,255,0.04);
            padding: 16px 18px;
            border-radius: 14px;
            margin-bottom: 12px;
        ">
            <div style="font-size: 22px; font-weight: 750; margin-bottom: 6px;">
                {icon} {title}
            </div>
            <div style="font-size: 16px; line-height: 1.62;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# 1. BASE DATA
# ---------------------------------------------------------
def get_macro_base():
    """
    Dữ liệu minh họa/calibrated từ các bài 1-11.
    Khi nộp chính thức, sinh viên nên thay bằng dữ liệu chính thức:
    Cục Thống kê, MIC, MOST, MOLISA, World Bank, OECD.
    """
    return pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "GDP": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "K": [16500, 17800, 19600, 21300, 23500, 25900],
        "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    })


def get_scenarios():
    return pd.DataFrame({
        "scenario_id": ["S1", "S2", "S3", "S4", "S5"],
        "scenario_name": [
            "Truyền thống",
            "Số hóa nhanh",
            "AI dẫn dắt",
            "Bao trùm số",
            "Tối ưu cân bằng",
        ],
        "description": [
            "Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu.",
            "Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số.",
            "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu.",
            "Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số.",
            "Kết quả mô phỏng cân bằng giữa tăng trưởng, bao trùm và rủi ro.",
        ],
        "K_share": [0.70, 0.25, 0.20, 0.30, 0.36],
        "D_share": [0.10, 0.45, 0.20, 0.20, 0.26],
        "AI_share": [0.10, 0.15, 0.45, 0.10, 0.18],
        "H_share": [0.10, 0.15, 0.15, 0.40, 0.20],
        "policy_icon": ["🏗️", "💻", "🤖", "🌱", "⚖️"],
    })


def get_region_data():
    return pd.DataFrame({
        "region_code": ["NMM", "RRD", "NCC", "CH", "SE", "MD"],
        "region": [
            "Trung du miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long",
        ],
        "GRDP_pc": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
        "FDI": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
        "DigitalIndex": [38, 78, 55, 32, 82, 48],
        "AIReadiness": [22, 68, 40, 18, 75, 30],
        "TrainedLabor": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
        "RD": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
        "Internet": [72, 92, 84, 68, 94, 78],
        "Gini": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
    })


def get_sector_data():
    return pd.DataFrame({
        "sector_id": list(range(1, 9)),
        "sector": [
            "Nông-Lâm-Thủy sản",
            "CN chế biến chế tạo",
            "Xây dựng",
            "Bán buôn-bán lẻ",
            "Tài chính-Ngân hàng",
            "Logistics-Vận tải",
            "CNTT-Truyền thông",
            "Giáo dục-Đào tạo",
        ],
        "labor_million": [13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15],
        "risk_pct": [18, 42, 25, 38, 52, 35, 28, 22],
        "a1_new_ai_job_per_billion": [8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5],
        "b1_upgrade_job_per_billion": [45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0],
        "c1_displace_job_per_billion": [5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5],
        "d1_retrain_capacity_per_billion": [50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0],
    })


def get_data_sources():
    return pd.DataFrame({
        "Nhóm dữ liệu": [
            "Vĩ mô 2020-2025",
            "Chuyển đổi số và kinh tế số",
            "Khoa học - công nghệ, AI",
            "Lao động - việc làm",
            "Dữ liệu quốc tế đối chiếu",
        ],
        "Nguồn khuyến nghị": [
            "Cục Thống kê Quốc gia",
            "Bộ Thông tin và Truyền thông, QĐ 749/QĐ-TTg",
            "Bộ KH&CN, Nghị quyết 57-NQ/TW",
            "Bộ LĐ-TB&XH, ILO",
            "World Bank, OECD, UNDP",
        ],
        "Biến dùng trong dashboard": [
            "GDP, K, L, năng suất, lao động",
            "Digital Index, tỷ trọng kinh tế số, internet",
            "AI readiness, R&D, doanh nghiệp công nghệ",
            "NetJob, risk tự động hóa, đào tạo lại",
            "FDI, thương mại, so sánh quốc tế",
        ],
        "Ghi chú": [
            "Cần thống nhất năm cơ sở và đơn vị tỷ VND.",
            "Dùng cho M2, M3 và cảnh báo số hóa.",
            "Dùng cho M1, M2, M5 và khuyến nghị AI.",
            "Dùng cho M4 và cảnh báo xã hội.",
            "Dùng để hiệu chỉnh kịch bản khủng hoảng, xanh hóa.",
        ],
    })


# ---------------------------------------------------------
# 2. MATHEMATICAL MODULES
# ---------------------------------------------------------
def cobb_douglas_y(A, K, L, D, AI, H):
    return A * (K ** 0.33) * (L ** 0.42) * (D ** 0.10) * (AI ** 0.08) * (H ** 0.07)


def compute_tfp_base(df):
    out = df.copy()
    out["A_TFP"] = out["GDP"] / (
        out["K"] ** 0.33 *
        out["L"] ** 0.42 *
        out["D"] ** 0.10 *
        out["AI"] ** 0.08 *
        out["H"] ** 0.07
    )
    return out


def simulate_m1_forecast(scenario_id, annual_budget=1000, shock_multiplier=1.0):
    """
    M1: Dự báo kinh tế 2026-2030 bằng Cobb-Douglas mở rộng.
    """
    macro = compute_tfp_base(get_macro_base())
    scenarios = get_scenarios().set_index("scenario_id")
    s = scenarios.loc[scenario_id]

    years = list(range(2026, 2031))

    K = float(macro["K"].iloc[-1])
    L = float(macro["L"].iloc[-1])
    D = float(macro["D"].iloc[-1])
    AI = float(macro["AI"].iloc[-1])
    H = float(macro["H"].iloc[-1])
    A = float(macro["A_TFP"].mean())

    rows = []

    for year in years:
        K = (1 - 0.045) * K + s["K_share"] * annual_budget
        D = (1 - 0.055) * D + s["D_share"] * annual_budget / 65.0
        AI = (1 - 0.070) * AI + s["AI_share"] * annual_budget / 12.0
        H = (1 - 0.020) * H + s["H_share"] * annual_budget / 125.0
        L = L * 1.004

        A = A * (
            1
            + 0.0022 * D / 100
            + 0.0018 * AI / 100
            + 0.0028 * H / 100
        )

        Y = cobb_douglas_y(A, K, L, D, AI, H) * shock_multiplier

        rows.append({
            "scenario_id": scenario_id,
            "scenario_name": s["scenario_name"],
            "year": year,
            "GDP": Y,
            "K": K,
            "L": L,
            "D": D,
            "AI": AI,
            "H": H,
            "A_TFP": A,
        })

    return pd.DataFrame(rows)


def entropy_weights(X):
    X = np.array(X, dtype=float)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-12)
    X_norm = X_norm + 1e-12
    P = X_norm / X_norm.sum(axis=0)
    k = 1 / np.log(X.shape[0])
    E = -k * np.sum(P * np.log(P), axis=0)
    d = 1 - E
    return d / d.sum()


def topsis_score(df, criteria, benefit_flags, weights=None):
    X = df[criteria].values.astype(float)

    if weights is None:
        weights = entropy_weights(X)
    else:
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

    denom = np.sqrt((X ** 2).sum(axis=0))
    R = X / np.where(denom == 0, 1, denom)
    V = R * weights

    benefit_flags = np.array(benefit_flags, dtype=bool)

    A_plus = np.where(benefit_flags, V.max(axis=0), V.min(axis=0))
    A_minus = np.where(benefit_flags, V.min(axis=0), V.max(axis=0))

    S_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    S_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

    C = S_minus / (S_plus + S_minus + 1e-12)

    out = df.copy()
    out["TOPSIS_score"] = C
    out["Rank"] = out["TOPSIS_score"].rank(ascending=False, method="dense").astype(int)

    weight_df = pd.DataFrame({
        "criteria": criteria,
        "weight": weights,
        "type": ["Benefit" if b else "Cost" for b in benefit_flags],
    })

    return out.sort_values("TOPSIS_score", ascending=False), weight_df


def run_m2_readiness():
    region = get_region_data()
    criteria = [
        "GRDP_pc", "FDI", "DigitalIndex", "AIReadiness",
        "TrainedLabor", "RD", "Internet", "Gini"
    ]
    benefit_flags = [True, True, True, True, True, True, True, False]
    result, weights = topsis_score(region, criteria, benefit_flags)
    return result, weights


def run_m3_allocation(scenario_id, total_budget=80000):
    """
    M3: Phân bổ ngân sách ngành-vùng-thời gian dạng prototype.
    S5 dùng readiness + equity need để cân bằng.
    """
    regions, weights = run_m2_readiness()
    scenarios = get_scenarios().set_index("scenario_id")
    s = scenarios.loc[scenario_id]

    reg = regions.copy()

    reg["need_score"] = (
        (100 - reg["DigitalIndex"]) * 0.35
        + (100 - reg["AIReadiness"]) * 0.35
        + (45 - reg["TrainedLabor"]).clip(lower=0) * 0.30
    )

    reg["readiness_score_norm"] = reg["TOPSIS_score"] / reg["TOPSIS_score"].sum()
    reg["need_score_norm"] = reg["need_score"] / reg["need_score"].sum()

    if scenario_id == "S4":
        reg["region_weight"] = 0.25 * reg["readiness_score_norm"] + 0.75 * reg["need_score_norm"]
    elif scenario_id == "S3":
        reg["region_weight"] = 0.75 * reg["readiness_score_norm"] + 0.25 * reg["need_score_norm"]
    elif scenario_id == "S5":
        reg["region_weight"] = 0.50 * reg["readiness_score_norm"] + 0.50 * reg["need_score_norm"]
    else:
        reg["region_weight"] = 0.60 * reg["readiness_score_norm"] + 0.40 * reg["need_score_norm"]

    reg["region_weight"] = reg["region_weight"] / reg["region_weight"].sum()

    item_shares = {
        "K": s["K_share"],
        "D": s["D_share"],
        "AI": s["AI_share"],
        "H": s["H_share"],
    }

    rows = []
    for _, r in reg.iterrows():
        for item, share in item_shares.items():
            rows.append({
                "scenario_id": scenario_id,
                "scenario_name": s["scenario_name"],
                "region": r["region"],
                "region_code": r["region_code"],
                "item": item,
                "item_budget": total_budget * r["region_weight"] * share,
                "region_weight": r["region_weight"],
            })

    alloc = pd.DataFrame(rows)

    region_total = alloc.groupby(["scenario_id", "scenario_name", "region"], as_index=False).agg(
        total_region_budget=("item_budget", "sum")
    )

    item_total = alloc.groupby(["scenario_id", "scenario_name", "item"], as_index=False).agg(
        total_item_budget=("item_budget", "sum")
    )

    return alloc, region_total, item_total


def run_m4_labor(scenario_id, total_budget=80000):
    """
    M4: Mô phỏng lao động dựa trên ngân sách AI và H của từng kịch bản.
    """
    sectors = get_sector_data()
    scenarios = get_scenarios().set_index("scenario_id")
    s = scenarios.loc[scenario_id]

    ai_budget = total_budget * s["AI_share"] * 0.08
    h_budget = total_budget * s["H_share"] * 0.08

    sectors = sectors.copy()

    risk_weight = sectors["risk_pct"] * sectors["labor_million"]
    risk_weight = risk_weight / risk_weight.sum()

    training_weight = sectors["labor_million"] * (1 + sectors["risk_pct"] / 100)
    training_weight = training_weight / training_weight.sum()

    sectors["x_AI"] = ai_budget * risk_weight
    sectors["x_H"] = h_budget * training_weight

    sectors["risk"] = sectors["risk_pct"] / 100
    sectors["NewJob_AI"] = sectors["a1_new_ai_job_per_billion"] * sectors["x_AI"]
    sectors["UpgradeJob"] = sectors["b1_upgrade_job_per_billion"] * sectors["x_H"]
    sectors["DisplacedJob"] = (
        sectors["c1_displace_job_per_billion"] *
        sectors["risk"] *
        sectors["x_AI"]
    )
    sectors["RetrainingCapacity"] = sectors["d1_retrain_capacity_per_billion"] * sectors["x_H"]
    sectors["NetJob"] = sectors["NewJob_AI"] + sectors["UpgradeJob"] - sectors["DisplacedJob"]
    sectors["RetrainingGap"] = (sectors["DisplacedJob"] - sectors["RetrainingCapacity"]).clip(lower=0)

    return sectors


def run_m5_risk(scenario_id, total_budget=80000):
    """
    M5: Cảnh báo rủi ro cyber, môi trường, phụ thuộc công nghệ, xã hội.
    """
    scenarios = get_scenarios().set_index("scenario_id")
    s = scenarios.loc[scenario_id]

    cyber = 100 * (0.15 * s["D_share"] + 0.50 * s["AI_share"] - 0.20 * s["H_share"])
    emission = 100 * (0.45 * s["K_share"] + 0.30 * s["AI_share"] - 0.15 * s["D_share"])
    dependency = 100 * (0.55 * s["AI_share"] + 0.20 * s["D_share"] - 0.25 * s["H_share"])
    social = 100 * (0.45 * s["AI_share"] - 0.50 * s["H_share"] + 0.10 * s["K_share"])

    rows = [
        {"risk_type": "Cyber risk", "score": max(0, cyber), "threshold": 18},
        {"risk_type": "Environmental pressure", "score": max(0, emission), "threshold": 28},
        {"risk_type": "Technology dependency", "score": max(0, dependency), "threshold": 22},
        {"risk_type": "Labor-social risk", "score": max(0, social), "threshold": 12},
    ]

    risk = pd.DataFrame(rows)
    risk["status"] = np.where(risk["score"] > risk["threshold"], "Cảnh báo", "Kiểm soát")
    risk["scenario_id"] = scenario_id
    risk["scenario_name"] = s["scenario_name"]

    return risk


def run_integrated_pipeline(total_budget=80000, annual_budget=1000):
    scenarios = get_scenarios()
    all_forecasts = []
    all_alloc = []
    all_region = []
    all_item = []
    all_labor = []
    all_risk = []
    kpi_rows = []

    shock_map = {
        "S1": 1.00,
        "S2": 1.01,
        "S3": 1.015,
        "S4": 0.995,
        "S5": 1.012,
    }

    for sid in scenarios["scenario_id"]:
        f = simulate_m1_forecast(sid, annual_budget=annual_budget, shock_multiplier=shock_map[sid])
        all_forecasts.append(f)

        alloc, region_total, item_total = run_m3_allocation(sid, total_budget=total_budget)
        all_alloc.append(alloc)
        all_region.append(region_total)
        all_item.append(item_total)

        labor = run_m4_labor(sid, total_budget=total_budget)
        labor["scenario_id"] = sid
        labor["scenario_name"] = scenarios.set_index("scenario_id").loc[sid, "scenario_name"]
        all_labor.append(labor)

        risk = run_m5_risk(sid, total_budget=total_budget)
        all_risk.append(risk)

        last = f[f["year"] == 2030].iloc[0]
        labor_total = labor["NetJob"].sum()
        displaced_total = labor["DisplacedJob"].sum()
        risk_avg = risk["score"].mean()
        risk_count = (risk["status"] == "Cảnh báo").sum()

        kpi_rows.append({
            "scenario_id": sid,
            "scenario_name": scenarios.set_index("scenario_id").loc[sid, "scenario_name"],
            "GDP_2030": last["GDP"],
            "D_2030": last["D"],
            "AI_2030": last["AI"],
            "H_2030": last["H"],
            "TFP_2030": last["A_TFP"],
            "NetJob": labor_total,
            "DisplacedJob": displaced_total,
            "AvgRiskScore": risk_avg,
            "RiskWarnings": risk_count,
        })

    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    allocation_df = pd.concat(all_alloc, ignore_index=True)
    region_df = pd.concat(all_region, ignore_index=True)
    item_df = pd.concat(all_item, ignore_index=True)
    labor_df = pd.concat(all_labor, ignore_index=True)
    risk_df = pd.concat(all_risk, ignore_index=True)
    kpi_df = pd.DataFrame(kpi_rows)

    kpi_df["GDP_rank"] = kpi_df["GDP_2030"].rank(ascending=False, method="dense").astype(int)
    kpi_df["NetJob_rank"] = kpi_df["NetJob"].rank(ascending=False, method="dense").astype(int)
    kpi_df["Risk_rank"] = kpi_df["AvgRiskScore"].rank(ascending=True, method="dense").astype(int)

    kpi_df["Composite_score"] = (
        0.40 * (kpi_df["GDP_2030"] / kpi_df["GDP_2030"].max())
        + 0.30 * (kpi_df["NetJob"] / kpi_df["NetJob"].max())
        + 0.20 * (1 - kpi_df["AvgRiskScore"] / kpi_df["AvgRiskScore"].max())
        + 0.10 * (kpi_df["H_2030"] / kpi_df["H_2030"].max())
    )
    kpi_df["Composite_rank"] = kpi_df["Composite_score"].rank(ascending=False, method="dense").astype(int)

    return {
        "forecast": forecast_df,
        "allocation": allocation_df,
        "region_allocation": region_df,
        "item_allocation": item_df,
        "labor": labor_df,
        "risk": risk_df,
        "kpi": kpi_df.sort_values("Composite_rank"),
    }


# ---------------------------------------------------------
# 3. DOWNLOADABLE SUPPORT FILES
# ---------------------------------------------------------
def make_readme_text():
    return """# AIDEOM-VN Prototype

## Structure
- modules/bai12_aideom_vn_integrated.py: Streamlit dashboard module
- data/: official Vietnamese datasets to be added
- tests/: pytest tests for key modules
- reports/: markdown/pdf report
- slides/: 15-slide presentation

## Core modules
M1 Forecast: Cobb-Douglas macro forecast
M2 Readiness: TOPSIS + entropy weight
M3 Allocation: budget allocation by region and investment item
M4 Labor: AI-H-NetJob simulation
M5 Risk: cyber, environment, dependency, labor risk
M6 Dashboard: policy comparison and decision support

## Run
pip install -r requirements.txt
streamlit run app.py

## Policy scenarios
S1 Traditional
S2 Fast digitalization
S3 AI-led
S4 Inclusive digitalization
S5 Balanced optimal
"""


def make_pytest_text():
    return """import pandas as pd
from modules import bai12_aideom_vn_integrated as m

def test_pipeline_runs():
    out = m.run_integrated_pipeline(total_budget=80000, annual_budget=1000)
    assert "kpi" in out
    assert len(out["kpi"]) == 5

def test_scenario_shares_sum_to_one():
    s = m.get_scenarios()
    shares = s[["K_share", "D_share", "AI_share", "H_share"]].sum(axis=1)
    assert all(abs(shares - 1.0) < 1e-9)

def test_topsis_regions():
    result, weights = m.run_m2_readiness()
    assert len(result) == 6
    assert abs(weights["weight"].sum() - 1.0) < 1e-8

def test_labor_netjob_columns():
    labor = m.run_m4_labor("S5")
    assert "NetJob" in labor.columns
    assert "DisplacedJob" in labor.columns

def test_risk_warning_columns():
    risk = m.run_m5_risk("S3")
    assert set(["risk_type", "score", "threshold", "status"]).issubset(risk.columns)
"""


def make_requirements_text():
    return """streamlit
pandas
numpy
plotly
"""


# ---------------------------------------------------------
# 4. DASHBOARD TABS
# ---------------------------------------------------------
def show_overview(total_budget, annual_budget):
    sticker_header(
        "🧠🇻🇳",
        "AIDEOM-VN — Nguyên mẫu mô hình hỗ trợ ra quyết định kinh tế số",
        "Đồ án tích hợp 6 mô-đun: dự báo, sẵn sàng số, phân bổ, lao động, rủi ro và dashboard ra quyết định."
    )

    policy_card(
        "🎯",
        "Mục tiêu đồ án",
        "Xây dựng một pipeline định lượng để so sánh 5 kịch bản chính sách kinh tế số Việt Nam đến 2030. Dashboard không thay thế quyết định chính trị - xã hội, mà cung cấp bằng chứng, cảnh báo và minh họa đánh đổi.",
        "success"
    )

    policy_card(
        "⚖️",
        "Đánh đổi chính sách trung tâm",
        "Kịch bản AI dẫn dắt có thể tạo tăng trưởng cao nhưng làm tăng rủi ro cyber, phụ thuộc công nghệ và dịch chuyển lao động. Kịch bản bao trùm số giúp giảm rủi ro xã hội nhưng có thể tăng trưởng chậm hơn. S5 cố gắng cân bằng các mục tiêu.",
        "warning"
    )

    modules = pd.DataFrame({
        "Module": ["M1", "M2", "M3", "M4", "M5", "M6"],
        "Tên": [
            "Dự báo kinh tế",
            "Đánh giá sẵn sàng số",
            "Tối ưu phân bổ",
            "Mô phỏng lao động",
            "Đánh giá rủi ro",
            "Dashboard ra quyết định",
        ],
        "Đầu vào": [
            "Macro 2020-2025",
            "Regions, sectors",
            "Budget, β-matrix, readiness",
            "AI plans, H plans",
            "Risk parameters",
            "Outputs M1-M5",
        ],
        "Đầu ra": [
            "GDP, TFP, K, D, AI, H 2026-2030",
            "Digital Index + AI Readiness map",
            "Phân bổ ngành-vùng-thời gian",
            "NetJob từng ngành",
            "Cyber, environmental, dependency risk",
            "KPI, cảnh báo, khuyến nghị",
        ],
        "Kỹ thuật chính": [
            "Cobb-Douglas",
            "TOPSIS + entropy",
            "Allocation model",
            "NetJob simulation",
            "Multi-risk scoring",
            "Streamlit + Plotly",
        ],
    })

    st.subheader("Bảng 12.1 — Sáu mô-đun chức năng AIDEOM-VN")
    st.dataframe(modules, use_container_width=True)

    st.subheader("Ảnh 12.1 — Sơ đồ luồng dữ liệu 6 mô-đun")

    labels = [
        "Dữ liệu Việt Nam",
        "M1 Dự báo",
        "M2 Readiness",
        "M3 Phân bổ",
        "M4 Lao động",
        "M5 Rủi ro",
        "M6 Dashboard",
        "KPI 2030",
        "Cảnh báo",
        "Khuyến nghị chính sách",
    ]

    fig_flow = go.Figure(data=[go.Sankey(
        node=dict(
            label=labels,
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.35),
        ),
        link=dict(
            source=[0, 0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 6],
            target=[1, 2, 3, 3, 3, 4, 5, 6, 6, 7, 8, 9],
            value=[25, 20, 25, 20, 20, 18, 18, 16, 16, 20, 15, 15],
        )
    )])
    fig_flow.update_layout(height=560, title="Pipeline AIDEOM-VN: từ dữ liệu đến khuyến nghị")
    st.plotly_chart(fig_flow, use_container_width=True)

    data_sources = get_data_sources()
    st.subheader("Bảng 12.2 — Ghi chú nguồn dữ liệu Việt Nam cần dùng khi hoàn thiện báo cáo")
    st.dataframe(data_sources, use_container_width=True)

    out = run_integrated_pipeline(total_budget=total_budget, annual_budget=annual_budget)
    kpi = out["kpi"]

    best = kpi.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kịch bản tốt nhất", best["scenario_name"], f"Rank {int(best['Composite_rank'])}")
    c2.metric("GDP 2030 cao nhất", f"{kpi['GDP_2030'].max():,.1f}")
    c3.metric("NetJob cao nhất", f"{kpi['NetJob'].max():,.0f}")
    c4.metric("Cảnh báo rủi ro tối đa", f"{int(kpi['RiskWarnings'].max())}")

    st.subheader("Bảng 12.3 — KPI tổng hợp 5 kịch bản đến 2030")
    st.dataframe(kpi.round(3), use_container_width=True)

    fig_kpi = px.bar(
        kpi,
        x="scenario_name",
        y="Composite_score",
        color="scenario_name",
        text="Composite_score",
        title="Ảnh 12.2 — Xếp hạng tổng hợp 5 kịch bản chính sách"
    )
    fig_kpi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_kpi.update_layout(height=480, xaxis_title="Kịch bản", yaxis_title="Composite score", showlegend=False)
    st.plotly_chart(fig_kpi, use_container_width=True)


def show_scenarios(total_budget, annual_budget):
    st.header("12.2. Năm kịch bản chính sách")

    sticker_header(
        "🎛️📌",
        "Năm kịch bản — năm triết lý phát triển khác nhau",
        "Mỗi kịch bản không chỉ là tỷ trọng ngân sách, mà là một quan điểm chính sách về tăng trưởng, công nghệ, bao trùm và rủi ro."
    )

    scenarios = get_scenarios()

    st.subheader("Bảng 12.4 — Cấu trúc phân bổ của 5 kịch bản")
    st.dataframe(scenarios, use_container_width=True)

    fig_stack = px.bar(
        scenarios.melt(
            id_vars=["scenario_id", "scenario_name"],
            value_vars=["K_share", "D_share", "AI_share", "H_share"],
            var_name="Hạng mục",
            value_name="Tỷ trọng"
        ),
        x="scenario_name",
        y="Tỷ trọng",
        color="Hạng mục",
        barmode="stack",
        text="Tỷ trọng",
        title="Ảnh 12.3 — Cơ cấu ngân sách K/D/AI/H theo 5 kịch bản"
    )
    fig_stack.update_layout(height=530, xaxis_title="Kịch bản", yaxis_title="Tỷ trọng ngân sách")
    st.plotly_chart(fig_stack, use_container_width=True)

    for _, s in scenarios.iterrows():
        tone = {
            "S1": "gray",
            "S2": "info",
            "S3": "purple",
            "S4": "success",
            "S5": "warning",
        }[s["scenario_id"]]

        policy_card(
            s["policy_icon"],
            f"{s['scenario_id']} — {s['scenario_name']}",
            f"{s['description']} Cơ cấu: K={s['K_share']:.0%}, D={s['D_share']:.0%}, AI={s['AI_share']:.0%}, H={s['H_share']:.0%}.",
            tone
        )

    out = run_integrated_pipeline(total_budget=total_budget, annual_budget=annual_budget)
    forecast = out["forecast"]
    kpi = out["kpi"]

    st.subheader("Ảnh 12.4 — GDP dự báo 2026-2030 theo kịch bản")
    fig_gdp = px.line(
        forecast,
        x="year",
        y="GDP",
        color="scenario_name",
        markers=True,
        title="Đường dự báo GDP theo M1 Cobb-Douglas mở rộng"
    )
    fig_gdp.update_layout(height=520, xaxis_title="Năm", yaxis_title="GDP mô phỏng, nghìn tỷ VND")
    st.plotly_chart(fig_gdp, use_container_width=True)

    st.subheader("Ảnh 12.5 — So sánh 4 KPI chính năm 2030")
    kpi_long = kpi.melt(
        id_vars=["scenario_id", "scenario_name"],
        value_vars=["GDP_2030", "NetJob", "AvgRiskScore", "H_2030"],
        var_name="KPI",
        value_name="Giá trị"
    )
    fig_kpi = px.bar(
        kpi_long,
        x="scenario_name",
        y="Giá trị",
        color="KPI",
        barmode="group",
        title="Tăng trưởng, việc làm, rủi ro và nhân lực theo kịch bản"
    )
    fig_kpi.update_layout(height=550, xaxis_title="Kịch bản")
    st.plotly_chart(fig_kpi, use_container_width=True)


def show_allocation(total_budget, annual_budget):
    st.header("12.3. Phân bổ ngân sách và bản đồ vùng")

    out = run_integrated_pipeline(total_budget=total_budget, annual_budget=annual_budget)
    allocation = out["allocation"]
    region_alloc = out["region_allocation"]
    item_alloc = out["item_allocation"]

    selected_scenario = st.selectbox(
        "Chọn kịch bản để xem phân bổ chi tiết",
        get_scenarios()["scenario_id"].tolist(),
        format_func=lambda x: f"{x} — {get_scenarios().set_index('scenario_id').loc[x, 'scenario_name']}",
        key="bai12_alloc_scenario"
    )

    alloc_s = allocation[allocation["scenario_id"] == selected_scenario]
    region_s = region_alloc[region_alloc["scenario_id"] == selected_scenario]
    item_s = item_alloc[item_alloc["scenario_id"] == selected_scenario]

    st.subheader("Bảng 12.5 — Phân bổ ngân sách theo vùng và hạng mục")
    pivot = alloc_s.pivot_table(
        index="region",
        columns="item",
        values="item_budget",
        aggfunc="sum"
    )
    st.dataframe(pivot.round(1), use_container_width=True)

    fig_heat = px.imshow(
        pivot,
        text_auto=".0f",
        aspect="auto",
        title=f"Ảnh 12.6 — Heatmap phân bổ ngân sách {selected_scenario}"
    )
    fig_heat.update_layout(height=560)
    st.plotly_chart(fig_heat, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_region = px.bar(
            region_s.sort_values("total_region_budget", ascending=False),
            x="region",
            y="total_region_budget",
            text="total_region_budget",
            title="Tổng ngân sách theo vùng"
        )
        fig_region.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_region.update_layout(height=480, xaxis_tickangle=-25, yaxis_title="Tỷ VND")
        st.plotly_chart(fig_region, use_container_width=True)

    with c2:
        fig_item = px.pie(
            item_s,
            names="item",
            values="total_item_budget",
            hole=0.42,
            title="Cơ cấu ngân sách theo K/D/AI/H"
        )
        fig_item.update_layout(height=480)
        st.plotly_chart(fig_item, use_container_width=True)

    readiness, weights = run_m2_readiness()
    st.subheader("Bảng 12.6 — M2 TOPSIS: bản đồ sẵn sàng số và AI")
    st.dataframe(readiness[["region", "TOPSIS_score", "Rank", "DigitalIndex", "AIReadiness", "TrainedLabor"]].round(4), use_container_width=True)

    fig_ready = px.scatter(
        readiness,
        x="DigitalIndex",
        y="AIReadiness",
        size="GRDP_pc",
        color="TOPSIS_score",
        text="region_code",
        hover_name="region",
        title="Ảnh 12.7 — Bản đồ Digital Index và AI Readiness vùng"
    )
    fig_ready.update_traces(textposition="top center")
    fig_ready.update_layout(height=520)
    st.plotly_chart(fig_ready, use_container_width=True)


def show_labor_risk(total_budget, annual_budget):
    st.header("12.4. Lao động và cảnh báo rủi ro")

    out = run_integrated_pipeline(total_budget=total_budget, annual_budget=annual_budget)
    labor = out["labor"]
    risk = out["risk"]

    selected_scenario = st.selectbox(
        "Chọn kịch bản để xem lao động và rủi ro",
        get_scenarios()["scenario_id"].tolist(),
        format_func=lambda x: f"{x} — {get_scenarios().set_index('scenario_id').loc[x, 'scenario_name']}",
        key="bai12_labor_scenario"
    )

    lab_s = labor[labor["scenario_id"] == selected_scenario].copy()
    risk_s = risk[risk["scenario_id"] == selected_scenario].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NetJob", f"{lab_s['NetJob'].sum():,.0f}")
    c2.metric("DisplacedJob", f"{lab_s['DisplacedJob'].sum():,.0f}")
    c3.metric("Retraining gap", f"{lab_s['RetrainingGap'].sum():,.0f}")
    c4.metric("Risk warnings", f"{int((risk_s['status'] == 'Cảnh báo').sum())}")

    st.subheader("Bảng 12.7 — M4 NetJob theo ngành")
    st.dataframe(
        lab_s[[
            "sector", "labor_million", "risk_pct", "x_AI", "x_H",
            "NewJob_AI", "UpgradeJob", "DisplacedJob", "RetrainingCapacity", "NetJob", "RetrainingGap"
        ]].round(3),
        use_container_width=True
    )

    fig_net = px.bar(
        lab_s.sort_values("NetJob", ascending=False),
        x="sector",
        y="NetJob",
        color="risk_pct",
        text="NetJob",
        title="Ảnh 12.8 — NetJob ròng theo ngành"
    )
    fig_net.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_net.update_layout(height=520, xaxis_tickangle=-25)
    st.plotly_chart(fig_net, use_container_width=True)

    comp = lab_s.melt(
        id_vars=["sector"],
        value_vars=["NewJob_AI", "UpgradeJob", "DisplacedJob"],
        var_name="Thành phần",
        value_name="Việc làm"
    )
    fig_comp = px.bar(
        comp,
        x="sector",
        y="Việc làm",
        color="Thành phần",
        barmode="group",
        title="Ảnh 12.9 — Việc mới, nâng kỹ năng và dịch chuyển"
    )
    fig_comp.update_layout(height=540, xaxis_tickangle=-25)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Bảng 12.8 — M5 Risk dashboard")
    st.dataframe(risk_s.round(3), use_container_width=True)

    fig_risk = px.bar(
        risk_s,
        x="risk_type",
        y="score",
        color="status",
        text="score",
        title="Ảnh 12.10 — Cảnh báo rủi ro theo kịch bản"
    )
    fig_risk.add_scatter(
        x=risk_s["risk_type"],
        y=risk_s["threshold"],
        mode="lines+markers",
        name="Threshold"
    )
    fig_risk.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_risk.update_layout(height=500, yaxis_title="Risk score")
    st.plotly_chart(fig_risk, use_container_width=True)

    warnings = risk_s[risk_s["status"] == "Cảnh báo"]
    if len(warnings) > 0:
        policy_card(
            "🚨",
            "Cảnh báo chính sách",
            "Kịch bản này có rủi ro vượt ngưỡng. Cần bổ sung ràng buộc về an ninh dữ liệu, đào tạo nhân lực, xanh hóa hạ tầng số hoặc giới hạn tốc độ mở rộng AI.",
            "danger"
        )
    else:
        policy_card(
            "✅",
            "Rủi ro đang được kiểm soát",
            "Các chỉ số rủi ro đều dưới ngưỡng. Tuy nhiên, cần tiếp tục theo dõi khi dữ liệu thực tế thay đổi theo quý/năm.",
            "success"
        )


def show_policy_comparison(total_budget, annual_budget):
    st.header("12.5. So sánh kịch bản và khuyến nghị chính sách")

    out = run_integrated_pipeline(total_budget=total_budget, annual_budget=annual_budget)
    kpi = out["kpi"]
    forecast = out["forecast"]
    risk = out["risk"]

    st.subheader("Bảng 12.9 — Tổng hợp KPI 2030 để so sánh chính sách")
    st.dataframe(kpi.round(3), use_container_width=True)

    fig_parallel = px.parallel_coordinates(
        kpi,
        dimensions=["GDP_2030", "NetJob", "AvgRiskScore", "H_2030", "Composite_score"],
        color="Composite_score",
        title="Ảnh 12.11 — Parallel coordinates: đánh đổi giữa tăng trưởng, việc làm và rủi ro"
    )
    fig_parallel.update_layout(height=580)
    st.plotly_chart(fig_parallel, use_container_width=True)

    fig_scatter = px.scatter(
        kpi,
        x="AvgRiskScore",
        y="GDP_2030",
        size="NetJob",
        color="scenario_name",
        text="scenario_id",
        title="Ảnh 12.12 — Bản đồ đánh đổi: GDP cao có đi cùng rủi ro cao không?"
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=520, xaxis_title="Điểm rủi ro trung bình", yaxis_title="GDP 2030")
    st.plotly_chart(fig_scatter, use_container_width=True)

    best = kpi.iloc[0]
    policy_card(
        "🏆",
        f"Khuyến nghị chính: {best['scenario_id']} — {best['scenario_name']}",
        f"Kịch bản này có điểm tổng hợp cao nhất trong mô phỏng. GDP 2030 đạt khoảng {best['GDP_2030']:,.1f}, NetJob khoảng {best['NetJob']:,.0f}, điểm rủi ro trung bình {best['AvgRiskScore']:.2f}.",
        "success"
    )

    policy_card(
        "⚖️",
        "Không có kịch bản tối ưu tuyệt đối",
        "S3 AI dẫn dắt thường có ưu thế tăng trưởng và công nghệ nhưng dễ tăng rủi ro cyber/phụ thuộc. S4 bao trùm số tốt hơn cho nhân lực và xã hội nhưng tăng trưởng có thể thấp hơn. S5 phù hợp khi Chính phủ muốn cân bằng tăng trưởng, bao trùm và an toàn hệ thống.",
        "warning"
    )

    st.subheader("Bảng 12.10 — Ma trận khuyến nghị chính sách")
    reco = pd.DataFrame({
        "Tình huống Việt Nam": [
            "Tăng trưởng yếu, thất nghiệp tăng",
            "Số hóa doanh nghiệp thấp",
            "Cạnh tranh AI/bán dẫn tăng mạnh",
            "Vùng yếu bị bỏ lại phía sau",
            "Rủi ro cyber/phụ thuộc công nghệ tăng",
        ],
        "Kịch bản nên ưu tiên": [
            "S4 hoặc S5",
            "S2 hoặc S5",
            "S3 có kiểm soát, hoặc S5",
            "S4",
            "S5 kèm ràng buộc an ninh và nhân lực",
        ],
        "Lý do": [
            "Cần đào tạo lại, giữ việc làm, tránh sốc xã hội.",
            "D là nền tảng cho SME, dịch vụ công và thanh toán số.",
            "AI tạo bước nhảy năng suất nhưng cần nhân lực và dữ liệu.",
            "Bao trùm vùng giúp giảm khoảng cách số.",
            "Cân bằng AI với H, cyber governance và xanh hóa hạ tầng.",
        ],
        "Liên hệ chính sách": [
            "An sinh, đào tạo lại, thị trường lao động",
            "QĐ 749/QĐ-TTg",
            "Nghị quyết 57-NQ/TW",
            "Phát triển vùng, nông nghiệp số, giáo dục số",
            "Chủ quyền số, an ninh dữ liệu, Net Zero",
        ],
    })
    st.dataframe(reco, use_container_width=True)


def show_technical_handoff(total_budget, annual_budget):
    st.header("12.6. Yêu cầu kỹ thuật, bàn giao và hướng mở rộng")

    sticker_header(
        "🧰📦",
        "Từ dashboard đến sản phẩm nộp cuối kỳ",
        "Phần này giúp nhóm kiểm tra nhanh sản phẩm bàn giao: mã nguồn, dashboard, báo cáo, slide, video demo và hướng nghiên cứu sau đồ án."
    )

    checklist = pd.DataFrame({
        "Hạng mục": [
            "Mã nguồn Python",
            "Dashboard Streamlit",
            "Bộ test pytest",
            "README + requirements.txt",
            "Báo cáo nghiên cứu",
            "Slide thuyết trình",
            "Video demo",
            "GitHub repository",
        ],
        "Yêu cầu đề bài": [
            "≥1.500 dòng, module độc lập",
            "Tối thiểu 4 tab, có tham số tương tác",
            "Chạy ít nhất S1, S3, S5",
            "Rõ cài đặt, dữ liệu, cách chạy",
            "15–25 trang, có 5 hình và 4 bảng",
            "15 slides + thuyết minh 20 phút",
            "3–5 phút demo dashboard",
            "Có version control, commit rõ",
        ],
        "Trạng thái trong module này": [
            "Prototype tích hợp trong một file module",
            "Có 6 tab chức năng",
            "Có đoạn pytest mẫu để tải xuống",
            "Có nội dung mẫu để tải xuống",
            "Có khung IMRaD và KPI để viết",
            "Có checklist nội dung slide",
            "Có checklist demo",
            "Sẵn sàng đẩy lên GitHub",
        ],
        "Gợi ý nâng cấp": [
            "Tách M1-M6 thành 6 file .py riêng.",
            "Thêm auth, cache, upload data.",
            "Tạo thư mục tests/test_aideom.py.",
            "Bổ sung ảnh, link nguồn chính thức.",
            "Xuất PDF bằng Word/LaTeX.",
            "Thiết kế bằng PowerPoint/Canva.",
            "Quay Loom/OBS.",
            "Thêm GitHub Actions test.",
        ],
    })

    st.subheader("Bảng 12.11 — Checklist sản phẩm bàn giao")
    st.dataframe(checklist, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "⬇️ Tải README.md mẫu",
        data=make_readme_text(),
        file_name="README_AIDEOM_VN.md",
        mime="text/markdown"
    )
    c2.download_button(
        "⬇️ Tải pytest mẫu",
        data=make_pytest_text(),
        file_name="test_aideom_vn.py",
        mime="text/plain"
    )
    c3.download_button(
        "⬇️ Tải requirements.txt mẫu",
        data=make_requirements_text(),
        file_name="requirements.txt",
        mime="text/plain"
    )

    st.subheader("Khung báo cáo 15–25 trang")

    report = pd.DataFrame({
        "Phần": [
            "Executive summary",
            "Introduction",
            "Methods",
            "Data",
            "Results",
            "Policy discussion",
            "Limitations",
            "Conclusion",
            "Appendix",
        ],
        "Nội dung cần có": [
            "Tóm tắt mục tiêu, dữ liệu, kết quả chính, khuyến nghị ≤2 trang.",
            "Bối cảnh Việt Nam, NQ 57, QĐ 749, vai trò AIDEOM-VN.",
            "Mô tả M1-M6, công thức Cobb-Douglas, TOPSIS, NetJob, risk score.",
            "Nguồn dữ liệu, đơn vị, năm cơ sở, cách xử lý thiếu dữ liệu.",
            "5 kịch bản, KPI 2030, bảng so sánh, hình minh họa.",
            "Đánh đổi tăng trưởng - bao trùm - rủi ro - môi trường.",
            "Giả định mô hình, dữ liệu minh họa, không thay thế quyết định chính trị.",
            "Khuyến nghị kịch bản, điều kiện triển khai.",
            "Code, test, bảng tham số, mô tả dashboard.",
        ],
        "Số trang gợi ý": ["1–2", "2", "4–5", "2–3", "5–7", "3–4", "1–2", "1", "2–3"],
    })
    st.dataframe(report, use_container_width=True)

    st.subheader("Khung slide 15 trang")

    slides = pd.DataFrame({
        "Slide": list(range(1, 16)),
        "Tiêu đề": [
            "Tên đồ án và nhóm",
            "Vấn đề chính sách",
            "Kiến trúc AIDEOM-VN",
            "Dữ liệu Việt Nam",
            "M1 Forecast",
            "M2 Readiness",
            "M3 Allocation",
            "M4 Labor",
            "M5 Risk",
            "5 kịch bản chính sách",
            "KPI 2030",
            "Đánh đổi chính sách",
            "Khuyến nghị",
            "Giới hạn và hướng mở rộng",
            "Q&A",
        ],
        "Điểm nói chính": [
            "Giới thiệu mục tiêu.",
            "Vì sao cần mô hình tích hợp.",
            "Sơ đồ 6 module.",
            "Nguồn, đơn vị, năm cơ sở.",
            "GDP, TFP, K/D/AI/H.",
            "TOPSIS vùng.",
            "Heatmap phân bổ.",
            "NetJob theo ngành.",
            "Cyber, environment, dependency.",
            "S1-S5.",
            "Bảng tổng hợp KPI.",
            "GDP vs NetJob vs Risk.",
            "Chọn S5 hoặc điều kiện dùng S3/S4.",
            "Dữ liệu, realtime, CGE/DSGE, MARL.",
            "Trả lời câu hỏi.",
        ],
    })
    st.dataframe(slides, use_container_width=True)

    st.subheader("Hướng mở rộng nghiên cứu sau đồ án")

    extensions = pd.DataFrame({
        "Hướng mở rộng": [
            "Bài báo SCIE Q2-Q3 theo use case",
            "Mở rộng sang CGE/DSGE-AI",
            "Tích hợp dữ liệu thời gian thực",
            "Multi-Agent RL",
        ],
        "Ý tưởng triển khai": [
            "Chọn ĐBSCL hoặc chế biến chế tạo, dùng dữ liệu thật và kiểm định độ nhạy.",
            "Liên kết ngành, hộ gia đình, doanh nghiệp, Nhà nước trong cân bằng tổng thể.",
            "Kết nối Open Data, hải quan, Vietstock, dữ liệu quý/tháng.",
            "Mỗi agent đại diện bộ/ngành/vùng với mục tiêu riêng.",
        ],
        "Giá trị khoa học": [
            "Tạo đóng góp học thuật cụ thể, dễ công bố hơn đồ án tổng quát.",
            "Tăng độ đúng kinh tế học vĩ mô, xét phản hồi giá và thị trường.",
            "Biến dashboard thành hệ thống theo dõi cập nhật.",
            "Mô phỏng xung đột và phối hợp chính sách giữa nhiều chủ thể.",
        ],
        "Mức độ khó": ["Trung bình-cao", "Cao", "Trung bình", "Cao"],
    })
    st.dataframe(extensions, use_container_width=True)

    fig_ext = px.bar(
        extensions,
        x="Hướng mở rộng",
        y=[4, 5, 3, 5],
        color="Mức độ khó",
        text=[4, 5, 3, 5],
        title="Ảnh 12.13 — Mức độ khó tương đối của các hướng mở rộng"
    )
    fig_ext.update_layout(height=460, yaxis_title="Điểm khó minh họa 1–5")
    st.plotly_chart(fig_ext, use_container_width=True)

    policy_card(
        "🚀",
        "Đóng góp sáng tạo đề xuất",
        "Ngoài 6 mô-đun bắt buộc, nhóm có thể thêm một mô-đun cảnh báo sớm theo thời gian thực hoặc một use case vùng ĐBSCL. Đây là phần đóng góp vượt khung để đạt mức A+.",
        "purple"
    )


# ---------------------------------------------------------
# 5. RENDER
# ---------------------------------------------------------
def render():
    st.title("🏛️ Bài 12 — AIDEOM-VN: Đồ án tích hợp hỗ trợ ra quyết định")
    inject_css()

    st.markdown("""
    Module này là nguyên mẫu tích hợp cuối kỳ. Dashboard liên kết 6 mô-đun AIDEOM-VN,
    so sánh 5 kịch bản chính sách đến năm 2030 và tạo các bảng/biểu đồ phục vụ báo cáo, slide, demo.
    """)

    with st.sidebar:
        st.header("⚙️ Tham số đồ án")
        total_budget = st.number_input(
            "Tổng ngân sách 2026-2030, tỷ VND",
            min_value=30000,
            max_value=150000,
            value=80000,
            step=5000,
            key="bai12_total_budget"
        )
        annual_budget = st.number_input(
            "Ngân sách mô phỏng hằng năm cho M1, nghìn tỷ VND",
            min_value=500,
            max_value=3000,
            value=1000,
            step=100,
            key="bai12_annual_budget"
        )

        st.markdown("---")
        st.markdown("**Gợi ý nộp bài:**")
        st.markdown("- Chạy dashboard trên Streamlit Cloud")
        st.markdown("- Báo cáo 15–25 trang")
        st.markdown("- Slide 15 trang")
        st.markdown("- Video demo 3–5 phút")

    tabs = st.tabs([
        "12.1 Tổng quan",
        "12.2 Kịch bản",
        "12.3 Phân bổ",
        "12.4 Lao động & rủi ro",
        "12.5 So sánh chính sách",
        "12.6 Bàn giao & mở rộng",
    ])

    with tabs[0]:
        show_overview(total_budget, annual_budget)

    with tabs[1]:
        show_scenarios(total_budget, annual_budget)

    with tabs[2]:
        show_allocation(total_budget, annual_budget)

    with tabs[3]:
        show_labor_risk(total_budget, annual_budget)

    with tabs[4]:
        show_policy_comparison(total_budget, annual_budget)

    with tabs[5]:
        show_technical_handoff(total_budget, annual_budget)

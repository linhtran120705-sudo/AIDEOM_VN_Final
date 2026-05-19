import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# =========================================================
# BÀI 9 — TÁC ĐỘNG AI TỚI THỊ TRƯỜNG LAO ĐỘNG VIỆT NAM
# LP: AI, tự động hóa, đào tạo lại và NetJob theo ngành
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
            opacity: 0.82;
            line-height: 1.55;
        }
        </style>
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


# ---------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------
def get_sector_data():
    df = pd.DataFrame({
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
        "a2_new_digital_job_per_billion": [12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 22.0],
        "b1_upgrade_job_per_billion": [45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0],
        "c1_displace_job_per_billion": [5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5],
        "d1_retrain_capacity_per_billion": [50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0],
    })

    df["risk"] = df["risk_pct"] / 100
    df["labor_jobs"] = df["labor_million"] * 1_000_000
    df["displacement_coef"] = df["c1_displace_job_per_billion"] * df["risk"]
    df["net_ai_coef_without_training"] = (
        df["a1_new_ai_job_per_billion"] - df["displacement_coef"]
    )
    df["retraining_need_ratio"] = (
        df["displacement_coef"] / df["d1_retrain_capacity_per_billion"]
    )
    df["netjob_per_1b_if_1b_AI_1b_H"] = (
        df["a1_new_ai_job_per_billion"] +
        df["b1_upgrade_job_per_billion"] -
        df["displacement_coef"]
    )
    df["vulnerable_group"] = df["sector_id"].isin([1, 3, 4])

    return df


def compute_jobs(df, x_ai, x_h):
    out = df.copy()
    out["x_AI"] = np.array(x_ai, dtype=float)
    out["x_H"] = np.array(x_h, dtype=float)

    out["NewJob_AI"] = out["a1_new_ai_job_per_billion"] * out["x_AI"]
    out["UpgradeJob"] = out["b1_upgrade_job_per_billion"] * out["x_H"]
    out["DisplacedJob"] = out["displacement_coef"] * out["x_AI"]
    out["RetrainingCapacity"] = out["d1_retrain_capacity_per_billion"] * out["x_H"]
    out["NetJob"] = out["NewJob_AI"] + out["UpgradeJob"] - out["DisplacedJob"]

    out["NetJob_per_budget"] = out["NetJob"] / (out["x_AI"] + out["x_H"] + 1e-9)
    out["Displaced_share_labor_pct"] = out["DisplacedJob"] / out["labor_jobs"] * 100
    out["Retraining_gap"] = out["DisplacedJob"] - out["RetrainingCapacity"]
    out["Retraining_gap"] = out["Retraining_gap"].clip(lower=0)
    out["AI_budget_share_pct"] = out["x_AI"] / (out["x_AI"] + out["x_H"] + 1e-9) * 100
    out["H_budget_share_pct"] = out["x_H"] / (out["x_AI"] + out["x_H"] + 1e-9) * 100

    return out


# ---------------------------------------------------------
# 2. OPTIMIZATION MODEL
# ---------------------------------------------------------
def solve_labor_lp(
    total_budget=30000,
    mode="policy_balanced",
    add_5pct_cap=False,
    max_sector_share=0.30,
    min_ai_share=0.18,
    min_h_share=0.25,
    min_vulnerable_h_share=0.18,
    min_manufacturing_h=0.0,
):
    if not PULP_AVAILABLE:
        return None

    df = get_sector_data()
    ids = df["sector_id"].tolist()

    model = pulp.LpProblem("AI_Labor_Market_Vietnam_LP", pulp.LpMaximize)

    x_ai = pulp.LpVariable.dicts("x_AI", ids, lowBound=0, cat="Continuous")
    x_h = pulp.LpVariable.dicts("x_H", ids, lowBound=0, cat="Continuous")

    new_job = {}
    upgrade_job = {}
    displaced = {}
    retrain_cap = {}
    net_job = {}

    row_by_id = {int(row["sector_id"]): row for _, row in df.iterrows()}

    for i in ids:
        row = row_by_id[i]
        new_job[i] = row["a1_new_ai_job_per_billion"] * x_ai[i]
        upgrade_job[i] = row["b1_upgrade_job_per_billion"] * x_h[i]
        displaced[i] = row["displacement_coef"] * x_ai[i]
        retrain_cap[i] = row["d1_retrain_capacity_per_billion"] * x_h[i]
        net_job[i] = new_job[i] + upgrade_job[i] - displaced[i]

    model += pulp.lpSum(net_job[i] for i in ids), "Maximize_total_NetJob"

    model += pulp.lpSum(x_ai[i] + x_h[i] for i in ids) <= total_budget, "C1_Total_budget"

    for i in ids:
        model += net_job[i] >= 0, f"C2_NetJob_nonnegative_sector_{i}"
        model += displaced[i] <= retrain_cap[i], f"C3_Automation_speed_not_exceed_retraining_{i}"

    if add_5pct_cap:
        for i in ids:
            labor_jobs = row_by_id[i]["labor_jobs"]
            model += displaced[i] <= 0.05 * labor_jobs, f"C4_Displaced_not_over_5pct_labor_sector_{i}"

    if mode == "policy_balanced":
        for i in ids:
            model += x_ai[i] + x_h[i] <= max_sector_share * total_budget, f"P1_Sector_cap_{i}"

        model += pulp.lpSum(x_ai[i] for i in ids) >= min_ai_share * total_budget, "P2_Min_AI_share"
        model += pulp.lpSum(x_h[i] for i in ids) >= min_h_share * total_budget, "P3_Min_H_share"

        vulnerable_ids = [1, 3, 4]
        model += (
            pulp.lpSum(x_h[i] for i in vulnerable_ids) >= min_vulnerable_h_share * total_budget
        ), "P4_Min_H_for_vulnerable_sectors"

        if min_manufacturing_h > 0:
            model += x_h[2] >= min_manufacturing_h, "P5_Min_H_manufacturing"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return {
            "status": status,
            "objective": np.nan,
            "result_df": pd.DataFrame(),
            "summary": {},
            "constraints_df": pd.DataFrame(),
        }

    x_ai_val = [pulp.value(x_ai[i]) for i in ids]
    x_h_val = [pulp.value(x_h[i]) for i in ids]

    result_df = compute_jobs(df, x_ai_val, x_h_val)

    constraints_rows = []
    for name, cons in model.constraints.items():
        constraints_rows.append({
            "constraint": name,
            "slack": cons.slack,
            "shadow_price": cons.pi,
            "binding?": abs(cons.slack) <= 1e-5,
        })

    constraints_df = pd.DataFrame(constraints_rows)

    summary = {
        "status": status,
        "objective_total_netjob": float(pulp.value(model.objective)),
        "total_budget_used": float(result_df["x_AI"].sum() + result_df["x_H"].sum()),
        "total_x_AI": float(result_df["x_AI"].sum()),
        "total_x_H": float(result_df["x_H"].sum()),
        "total_new_job": float(result_df["NewJob_AI"].sum()),
        "total_upgrade_job": float(result_df["UpgradeJob"].sum()),
        "total_displaced": float(result_df["DisplacedJob"].sum()),
        "total_retrain_capacity": float(result_df["RetrainingCapacity"].sum()),
        "mode": mode,
        "add_5pct_cap": add_5pct_cap,
    }

    return {
        "status": status,
        "objective": float(pulp.value(model.objective)),
        "result_df": result_df,
        "summary": summary,
        "constraints_df": constraints_df,
    }


def manufacturing_threshold(total_budget=30000, sector_ai_cap=None):
    df = get_sector_data()
    m = df[df["sector_id"] == 2].iloc[0]

    risk = m["risk"]
    a1 = m["a1_new_ai_job_per_billion"]
    b1 = m["b1_upgrade_job_per_billion"]
    c1 = m["c1_displace_job_per_billion"]
    d1 = m["d1_retrain_capacity_per_billion"]

    displace_coef = c1 * risk

    ratio_netjob = max(0, (displace_coef - a1) / b1)
    ratio_retrain = displace_coef / d1
    ratio_required = max(ratio_netjob, ratio_retrain)

    if sector_ai_cap is None:
        x_ai_max_feasible = total_budget / (1 + ratio_required)
    else:
        x_ai_max_feasible = min(sector_ai_cap, total_budget / (1 + ratio_required))

    x_h_min = ratio_required * x_ai_max_feasible

    return {
        "sector": m["sector"],
        "displace_coef": displace_coef,
        "ratio_netjob": ratio_netjob,
        "ratio_retrain": ratio_retrain,
        "ratio_required": ratio_required,
        "x_ai_max_feasible": x_ai_max_feasible,
        "x_h_min": x_h_min,
        "total_budget_needed": x_ai_max_feasible + x_h_min,
    }


def sensitivity_budget_curve(mode="policy_balanced", add_5pct_cap=False):
    budgets = np.arange(15000, 50001, 5000)
    rows = []

    for b in budgets:
        res = solve_labor_lp(total_budget=float(b), mode=mode, add_5pct_cap=add_5pct_cap)

        if res is not None and res["status"] == "Optimal":
            s = res["summary"]
            rows.append({
                "budget": b,
                "total_netjob": s["objective_total_netjob"],
                "total_displaced": s["total_displaced"],
                "total_x_AI": s["total_x_AI"],
                "total_x_H": s["total_x_H"],
                "status": "Optimal",
            })
        else:
            rows.append({
                "budget": b,
                "total_netjob": np.nan,
                "total_displaced": np.nan,
                "total_x_AI": np.nan,
                "total_x_H": np.nan,
                "status": "Infeasible/No solution",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 3. SECTIONS
# ---------------------------------------------------------
def show_context():
    st.header("9.1. Bối cảnh Việt Nam")

    sticker_header(
        "🤖👷‍♀️",
        "AI không chỉ thay thế việc làm — AI tái phân bổ kỹ năng",
        "Bài 9 mô phỏng thị trường lao động Việt Nam khi AI vừa tạo việc làm mới, vừa dịch chuyển việc làm cũ, và Nhà nước phải đầu tư đào tạo lại để NetJob ròng không âm."
    )

    policy_card(
        "🎯",
        "Câu hỏi chính sách trung tâm",
        "Khi tự động hóa tăng tốc trong chế biến chế tạo, bán buôn-bán lẻ, logistics và tài chính, Việt Nam cần phân bổ ngân sách AI và đào tạo lại như thế nào để không ngành nào bị mất việc ròng?",
        "success"
    )

    policy_card(
        "⚖️",
        "Đánh đổi then chốt",
        "Đầu tư AI có thể tạo việc làm mới có năng suất cao hơn, nhưng cũng làm dịch chuyển lao động cũ. Đào tạo lại giúp chuyển lao động bị ảnh hưởng sang vị trí mới, nhưng nếu đào tạo chậm hơn tốc độ tự động hóa thì rủi ro thất nghiệp ròng xuất hiện.",
        "warning"
    )

    policy_card(
        "🇻🇳",
        "Liên hệ Việt Nam",
        "Nghị quyết 57-NQ/TW nhấn mạnh khoa học - công nghệ, đổi mới sáng tạo và chuyển đổi số là động lực phát triển; QĐ 749/QĐ-TTg đặt nền cho chuyển đổi số quốc gia. Vì vậy, chính sách AI cần đi cùng chính sách lao động: đào tạo lại, nâng kỹ năng và bảo vệ nhóm dễ bị tổn thương.",
        "purple"
    )

    df = get_sector_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số ngành phân tích", "8")
    c2.metric("Tổng lao động", f"{df['labor_million'].sum():.2f}", "triệu người")
    c3.metric("Ngân sách mô phỏng", "30.000", "tỷ VND")
    c4.metric("Rủi ro AI cao nhất", f"{df['risk_pct'].max():.0f}%", "Tài chính-NH")

    st.subheader("Ảnh 9.1 — Bản đồ rủi ro tự động hóa và quy mô lao động")

    fig_risk = px.scatter(
        df,
        x="risk_pct",
        y="labor_million",
        size="labor_million",
        color="net_ai_coef_without_training",
        hover_name="sector",
        text="sector_id",
        title="Ngành nào vừa đông lao động vừa có rủi ro tự động hóa cao?",
        labels={
            "risk_pct": "Risk tự động hóa/AI, %",
            "labor_million": "Lao động, triệu người",
            "net_ai_coef_without_training": "Net AI coef trước đào tạo",
        }
    )
    fig_risk.update_traces(textposition="top center")
    fig_risk.update_layout(height=540)
    st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("Ảnh 9.2 — Luồng chính sách: AI → dịch chuyển việc làm → đào tạo lại → NetJob")

    labels = [
        "Ngân sách 30.000 tỷ",
        "Đầu tư AI x_AI",
        "Đào tạo lại x_H",
        "Việc làm AI mới",
        "Việc làm bị dịch chuyển",
        "Nâng cấp kỹ năng",
        "Năng lực đào tạo lại",
        "NetJob ròng",
        "An sinh xã hội",
    ]

    fig_flow = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.3),
            label=labels,
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2, 3, 5, 4, 6, 7],
            target=[1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 8],
            value=[16, 14, 10, 6, 9, 5, 10, 9, 4, 5, 15],
        )
    )])
    fig_flow.update_layout(
        title="Ảnh 9.2 — Logic mô hình lao động dưới tác động AI",
        height=560
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    risk_table = df[[
        "sector_id", "sector", "labor_million", "risk_pct",
        "a1_new_ai_job_per_billion", "b1_upgrade_job_per_billion",
        "displacement_coef", "retraining_need_ratio"
    ]].copy()
    risk_table.columns = [
        "ID", "Ngành", "Lao động, triệu", "Risk, %",
        "a1 việc AI/tỷ", "b1 nâng kỹ năng/tỷ",
        "Hệ số dịch chuyển hiệu dụng", "x_H cần cho 1 tỷ x_AI"
    ]

    st.subheader("Bảng 9.1 — Đọc nhanh rủi ro và nhu cầu đào tạo")
    st.dataframe(risk_table.round(3), use_container_width=True)

    st.info(
        "Cách đọc: hệ số dịch chuyển hiệu dụng = c1 × risk. Nếu hệ số này lớn, mỗi tỷ VND đầu tư AI có thể kéo theo nhiều việc làm cần chuyển đổi. "
        "Tỷ lệ x_H cần cho 1 tỷ x_AI càng cao thì ngành đó càng cần đào tạo lại đi kèm tự động hóa."
    )


def show_math_model():
    st.header("9.2. Mô hình toán học")

    sticker_header(
        "🧮⚙️",
        "Mô hình NetJob: AI tạo việc làm, tự động hóa dịch chuyển việc làm, đào tạo lại hấp thụ cú sốc",
        "Tư duy của bài toán là không cấm AI, mà kiểm soát tốc độ AI sao cho không vượt quá năng lực đào tạo lại lao động."
    )

    st.subheader("1️⃣ Phương trình NetJob ròng")

    st.latex(r"""
    NetJob_{i,t}
    =
    NewJob^{AI}_{i,t}
    +
    UpgradeJob_{i,t}
    -
    DisplacedJob^{Automation}_{i,t}
    """)

    st.markdown("""
    `NetJob` là số việc làm ròng của ngành `i`. Nếu `NetJobᵢ ≥ 0`, ngành đó không bị mất việc ròng.
    Mô hình nhìn thị trường lao động theo ba dòng: **việc mới do AI**, **việc nâng cấp nhờ đào tạo**, và **việc bị dịch chuyển bởi tự động hóa**.
    """)

    st.subheader("2️⃣ Các thành phần của mô hình")

    c1, c2 = st.columns(2)

    with c1:
        st.latex(r"NewJob_i = a_{1i}x^{AI}_i + a_{2i}x^D_i")
        st.latex(r"UpgradeJob_i = b_{1i}x^H_i")
        st.latex(r"DisplacedJob_i = c_{1i}x^{AI}_i risk_i")

    with c2:
        st.latex(r"RetrainingCapacity_i = d_{1i}x^H_i")
        st.latex(r"DisplacedJob_i \leq RetrainingCapacity_i")
        st.latex(r"NetJob_i \geq 0")

    st.warning(
        "Trong bài toán lập trình 9.4, đề bài chỉ tối ưu hai biến x_AI và x_H. Vì vậy, thành phần x_D được trình bày trong mô hình tổng quát nhưng không đưa vào biến quyết định của LP đơn giản."
    )

    variable_table = pd.DataFrame({
        "Ký hiệu": [
            "xᴬᴵᵢ", "xᴴᵢ", "riskᵢ", "a₁ᵢ", "a₂ᵢ", "b₁ᵢ", "c₁ᵢ", "d₁ᵢ",
            "NetJobᵢ", "RetrainingCapacityᵢ"
        ],
        "Ý nghĩa": [
            "Đầu tư AI vào ngành i",
            "Đầu tư đào tạo lại/nhân lực số vào ngành i",
            "Tỷ lệ việc làm có nguy cơ bị tự động hóa một phần",
            "Việc làm AI mới tạo ra trên mỗi tỷ VND đầu tư AI",
            "Việc làm mới từ đầu tư số hóa trên mỗi tỷ VND",
            "Việc làm được nâng cấp trên mỗi tỷ VND đào tạo",
            "Hệ số dịch chuyển việc làm do tự động hóa",
            "Năng lực đào tạo lại trên mỗi tỷ VND",
            "Việc làm ròng sau AI và đào tạo",
            "Số lao động có thể được đào tạo lại/hấp thụ",
        ],
        "Đơn vị": [
            "tỷ VND", "tỷ VND", "%", "việc/tỷ", "việc/tỷ", "việc/tỷ",
            "việc/tỷ", "việc/tỷ", "việc làm", "việc làm"
        ],
        "Ý nghĩa chính sách": [
            "Tốc độ đưa AI vào ngành",
            "Tốc độ nâng kỹ năng để hấp thụ AI",
            "Mức dễ tổn thương trước AI",
            "Lợi ích tạo việc làm mới",
            "Vai trò nền tảng số",
            "Năng lực chuyển đổi kỹ năng",
            "Mặt trái của tự động hóa",
            "Năng lực an sinh chủ động",
            "Mục tiêu cần không âm",
            "Rào chắn an sinh trước tự động hóa",
        ]
    })

    st.subheader("Bảng 9.2 — Chú giải biến và hệ số")
    st.dataframe(variable_table, use_container_width=True)

    st.subheader("3️⃣ Bài toán tối ưu tuyến tính")

    st.latex(r"""
    \max \sum_i NetJob_i
    """)

    st.latex(r"""
    \sum_i(x^{AI}_i+x^H_i)\leq 30000
    """)

    st.latex(r"""
    NetJob_i \geq 0,\quad
    DisplacedJob_i \leq RetrainingCapacity_i,\quad
    x^{AI}_i,x^H_i\geq 0
    """)

    st.markdown("""
    Đây là bài toán tuyến tính vì `NetJob`, `DisplacedJob` và `RetrainingCapacity` đều là hàm tuyến tính của `x_AI` và `x_H`.
    Bài toán có thể giải bằng PuLP, CVXPY hoặc scipy.optimize.linprog. Module này dùng **PuLP/CBC** để ổn định trên Streamlit Cloud.
    """)

    st.subheader("4️⃣ Công thức cải tiến: tốc độ tự động hóa so với năng lực đào tạo")

    st.latex(r"""
    AutomationPressure_i =
    \frac{DisplacedJob_i}{RetrainingCapacity_i+\varepsilon}
    """)

    st.markdown("""
    Nếu `AutomationPressureᵢ ≤ 1`, tốc độ tự động hóa không vượt quá năng lực đào tạo lại.  
    Nếu lớn hơn 1, ngành đó đang có rủi ro thiếu năng lực chuyển đổi lao động.
    """)

    st.subheader("5️⃣ Công thức cải tiến: giới hạn an sinh xã hội")

    st.latex(r"""
    DisplacedJob_i \leq 0.05L_i
    """)

    st.markdown("""
    Đây là ràng buộc mở rộng của câu 9.4.4. Nó nói rằng trong mỗi ngành, số việc làm bị dịch chuyển không được vượt quá 5% tổng lao động ngành.
    Ràng buộc này biến mô hình từ tối đa hóa việc làm thuần túy sang mô hình có **hàng rào an sinh xã hội**.
    """)

    policy_card(
        "🔐",
        "Ý nghĩa sâu của ràng buộc đào tạo lại",
        "Câu “tốc độ tự động hóa không nên vượt quá năng lực đào tạo lại” được biểu diễn chính xác bằng DisplacedJobᵢ ≤ RetrainingCapacityᵢ. Đây là ràng buộc bảo vệ người lao động, không phải ràng buộc kỹ thuật đơn thuần.",
        "success"
    )


def show_sector_data():
    st.header("9.3. Tham số 8 ngành Việt Nam")

    df = get_sector_data()

    display = df[[
        "sector_id", "sector", "labor_million", "risk_pct",
        "a1_new_ai_job_per_billion", "a2_new_digital_job_per_billion",
        "b1_upgrade_job_per_billion", "c1_displace_job_per_billion",
        "d1_retrain_capacity_per_billion"
    ]].copy()

    display.columns = [
        "ID", "Ngành", "Lao động, triệu", "Risk, %",
        "a1 việc/tỷ", "a2 việc/tỷ", "b1 việc/tỷ", "c1 việc/tỷ", "d1 việc/tỷ"
    ]

    st.dataframe(display.round(3), use_container_width=True)

    st.subheader("Ảnh 9.3 — Heatmap tham số việc làm theo ngành")

    heat = display.set_index("Ngành")[[
        "Risk, %", "a1 việc/tỷ", "a2 việc/tỷ", "b1 việc/tỷ", "c1 việc/tỷ", "d1 việc/tỷ"
    ]]

    fig_heat = px.imshow(
        heat,
        text_auto=".1f",
        aspect="auto",
        title="Mỗi ngành mạnh/yếu ở hệ số nào?"
    )
    fig_heat.update_layout(height=560)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Ảnh 9.4 — Hệ số tạo việc làm mới và hệ số dịch chuyển hiệu dụng")

    compare = df[[
        "sector", "a1_new_ai_job_per_billion", "b1_upgrade_job_per_billion",
        "displacement_coef", "risk_pct"
    ]].copy()

    compare_long = compare.melt(
        id_vars=["sector", "risk_pct"],
        value_vars=[
            "a1_new_ai_job_per_billion",
            "b1_upgrade_job_per_billion",
            "displacement_coef"
        ],
        var_name="Chỉ số",
        value_name="Việc làm/tỷ VND"
    )

    compare_long["Chỉ số"] = compare_long["Chỉ số"].replace({
        "a1_new_ai_job_per_billion": "Việc mới do AI a1",
        "b1_upgrade_job_per_billion": "Nâng kỹ năng b1",
        "displacement_coef": "Dịch chuyển hiệu dụng c1×risk",
    })

    fig_compare = px.bar(
        compare_long,
        x="sector",
        y="Việc làm/tỷ VND",
        color="Chỉ số",
        barmode="group",
        title="Tạo việc làm, nâng kỹ năng và rủi ro dịch chuyển theo ngành"
    )
    fig_compare.update_layout(height=570, xaxis_tickangle=-25)
    st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader("Ảnh 9.5 — Tỷ lệ đào tạo cần thiết cho mỗi 1 tỷ VND đầu tư AI")

    ratio_df = df.sort_values("retraining_need_ratio", ascending=False)

    fig_ratio = px.bar(
        ratio_df,
        x="retraining_need_ratio",
        y="sector",
        orientation="h",
        text="retraining_need_ratio",
        color="risk_pct",
        title="x_H tối thiểu cần để hấp thụ dịch chuyển từ 1 tỷ VND x_AI"
    )
    fig_ratio.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_ratio.update_layout(height=520, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_ratio, use_container_width=True)

    high_risk = df.sort_values("risk_pct", ascending=False).iloc[0]
    high_training = df.sort_values("retraining_need_ratio", ascending=False).iloc[0]

    c1, c2 = st.columns(2)
    c1.metric("Risk cao nhất", high_risk["sector"], f"{high_risk['risk_pct']:.0f}%")
    c2.metric("Cần đào tạo kèm AI cao nhất", high_training["sector"], f"{high_training['retraining_need_ratio']:.2f} tỷ H / 1 tỷ AI")

    st.info(
        "Điểm cần chú ý: Tài chính-Ngân hàng có risk cao nhất, nhưng cũng có a1 tạo việc làm AI cao. "
        "Ngành chế biến chế tạo có quy mô lao động lớn và hệ số dịch chuyển hiệu dụng cao, nên là trọng tâm đào tạo lại."
    )


def show_programming_solution():
    st.header("9.4. Giải yêu cầu lập trình")

    if not PULP_AVAILABLE:
        st.error("Chưa cài PuLP. Hãy thêm `pulp` vào requirements.txt.")
        return None

    sticker_header(
        "🧩📈",
        "Từ phương trình NetJob đến phân bổ ngân sách tối ưu",
        "Phần này giải LP, kiểm tra ràng buộc, tính NetJob từng ngành, tìm ngưỡng đào tạo ngành chế biến chế tạo và mô phỏng luồng dịch chuyển lao động."
    )

    st.subheader("Thiết lập mô hình")

    c1, c2, c3 = st.columns(3)

    total_budget = c1.number_input(
        "Ngân sách tổng, tỷ VND",
        min_value=5000,
        max_value=60000,
        value=30000,
        step=1000,
        key="bai9_total_budget"
    )

    mode_label = c2.selectbox(
        "Kiểu mô hình",
        [
            "Mô hình chính sách cân bằng",
            "Mô hình gốc theo đề bài"
        ],
        key="bai9_mode"
    )

    add_5pct_cap = c3.checkbox(
        "Thêm ràng buộc không ngành nào mất quá 5% lao động",
        value=False,
        key="bai9_5pct_cap"
    )

    mode = "policy_balanced" if mode_label == "Mô hình chính sách cân bằng" else "base"

    if mode == "policy_balanced":
        st.markdown("#### Tham số ràng buộc chính sách bổ sung")

        p1, p2, p3, p4 = st.columns(4)

        max_sector_share = p1.slider(
            "Trần ngân sách mỗi ngành",
            min_value=0.15,
            max_value=0.60,
            value=0.30,
            step=0.05,
            key="bai9_sector_cap"
        )

        min_ai_share = p2.slider(
            "Sàn tỷ trọng AI",
            min_value=0.00,
            max_value=0.50,
            value=0.18,
            step=0.02,
            key="bai9_min_ai"
        )

        min_h_share = p3.slider(
            "Sàn tỷ trọng đào tạo H",
            min_value=0.00,
            max_value=0.70,
            value=0.25,
            step=0.05,
            key="bai9_min_h"
        )

        min_vulnerable_h_share = p4.slider(
            "Sàn H cho nhóm dễ tổn thương 1,3,4",
            min_value=0.00,
            max_value=0.50,
            value=0.18,
            step=0.03,
            key="bai9_vulnerable_h"
        )

        min_manufacturing_h = st.number_input(
            "Sàn đào tạo lại cho CN chế biến chế tạo, tỷ VND",
            min_value=0.0,
            max_value=float(total_budget),
            value=0.0,
            step=500.0,
            key="bai9_min_manufacturing_h"
        )
    else:
        max_sector_share = 1.0
        min_ai_share = 0.0
        min_h_share = 0.0
        min_vulnerable_h_share = 0.0
        min_manufacturing_h = 0.0

        st.warning(
            "Mô hình gốc theo đề bài có thể tạo nghiệm góc: dồn ngân sách vào ngành có b1 cao nhất vì không bắt buộc đầu tư AI hoặc đa dạng hóa ngành. "
            "Để thảo luận chính sách hay hơn, nên xem thêm mô hình chính sách cân bằng."
        )

    st.subheader("Câu 9.4.1 — Giải LP bằng PuLP và báo cáo phân bổ tối ưu")

    result = solve_labor_lp(
        total_budget=total_budget,
        mode=mode,
        add_5pct_cap=add_5pct_cap,
        max_sector_share=max_sector_share,
        min_ai_share=min_ai_share,
        min_h_share=min_h_share,
        min_vulnerable_h_share=min_vulnerable_h_share,
        min_manufacturing_h=min_manufacturing_h,
    )

    if result is None or result["status"] != "Optimal":
        st.error(f"Mô hình không tối ưu. Trạng thái: {result['status'] if result else 'Không chạy được'}")
        st.info(
            "Gợi ý: nếu đang bật ràng buộc 5% hoặc sàn AI/H quá cao, hãy giảm sàn AI, tăng ngân sách, hoặc tăng trần ngân sách mỗi ngành."
        )
        return None

    res_df = result["result_df"]
    summary = result["summary"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trạng thái", result["status"])
    k2.metric("Tổng NetJob", f"{summary['objective_total_netjob']:,.0f}", "việc làm")
    k3.metric("Đầu tư AI", f"{summary['total_x_AI']:,.0f}", "tỷ VND")
    k4.metric("Đào tạo H", f"{summary['total_x_H']:,.0f}", "tỷ VND")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Việc AI mới", f"{summary['total_new_job']:,.0f}")
    k6.metric("Việc nâng kỹ năng", f"{summary['total_upgrade_job']:,.0f}")
    k7.metric("Việc bị dịch chuyển", f"{summary['total_displaced']:,.0f}")
    k8.metric("Năng lực đào tạo", f"{summary['total_retrain_capacity']:,.0f}")

    display = res_df[[
        "sector_id", "sector", "x_AI", "x_H", "NewJob_AI", "UpgradeJob",
        "DisplacedJob", "RetrainingCapacity", "NetJob",
        "Displaced_share_labor_pct", "Retraining_gap"
    ]].copy()

    display.columns = [
        "ID", "Ngành", "x_AI, tỷ", "x_H, tỷ", "Việc mới AI",
        "Việc nâng kỹ năng", "Việc dịch chuyển", "Năng lực đào tạo",
        "NetJob", "Dịch chuyển/LĐ, %", "Khoảng thiếu đào tạo"
    ]

    st.dataframe(display.round(3), use_container_width=True)

    st.markdown("#### Kiểm tra ràng buộc và shadow price")
    st.dataframe(result["constraints_df"].round(5), use_container_width=True)

    fig_alloc = px.bar(
        res_df.melt(
            id_vars=["sector"],
            value_vars=["x_AI", "x_H"],
            var_name="Hạng mục",
            value_name="Ngân sách, tỷ VND"
        ),
        x="sector",
        y="Ngân sách, tỷ VND",
        color="Hạng mục",
        barmode="stack",
        title="Ảnh 9.6 — Phân bổ ngân sách tối ưu theo ngành và hạng mục"
    )
    fig_alloc.update_layout(height=560, xaxis_tickangle=-25)
    st.plotly_chart(fig_alloc, use_container_width=True)

    fig_net = px.bar(
        res_df.sort_values("NetJob", ascending=False),
        x="sector",
        y="NetJob",
        color="NetJob",
        text="NetJob",
        title="Ảnh 9.7 — NetJob ròng theo ngành"
    )
    fig_net.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_net.update_layout(height=520, xaxis_tickangle=-25)
    st.plotly_chart(fig_net, use_container_width=True)

    fig_components = px.bar(
        res_df.melt(
            id_vars=["sector"],
            value_vars=["NewJob_AI", "UpgradeJob", "DisplacedJob"],
            var_name="Thành phần",
            value_name="Việc làm"
        ),
        x="sector",
        y="Việc làm",
        color="Thành phần",
        barmode="group",
        title="Ảnh 9.8 — Ba thành phần tạo nên NetJob: việc mới, nâng kỹ năng, dịch chuyển"
    )
    fig_components.update_layout(height=560, xaxis_tickangle=-25)
    st.plotly_chart(fig_components, use_container_width=True)

    st.subheader("Câu 9.4.2 — Ngưỡng đào tạo tối thiểu cho ngành CN chế biến chế tạo")

    threshold = manufacturing_threshold(total_budget=total_budget)

    t1, t2, t3 = st.columns(3)
    t1.metric("Hệ số dịch chuyển c1×risk", f"{threshold['displace_coef']:.3f}", "việc/tỷ")
    t2.metric("x_H cần cho 1 tỷ x_AI", f"{threshold['ratio_required']:.3f}", "tỷ H/tỷ AI")
    t3.metric("x_H tối thiểu khi x_AI tối đa khả thi", f"{threshold['x_h_min']:,.0f}", "tỷ VND")

    xai_grid = np.linspace(0, total_budget, 100)
    xh_netjob = threshold["ratio_netjob"] * xai_grid
    xh_retrain = threshold["ratio_retrain"] * xai_grid
    xh_required = threshold["ratio_required"] * xai_grid
    feasible_total = total_budget - xai_grid

    threshold_df = pd.DataFrame({
        "x_AI ngành 2": xai_grid,
        "x_H cần để NetJob ≥ 0": xh_netjob,
        "x_H cần để Displaced ≤ Retraining": xh_retrain,
        "x_H yêu cầu thực tế": xh_required,
        "Ngân sách còn lại nếu chỉ xét ngành 2": feasible_total,
    })

    fig_threshold = px.line(
        threshold_df.melt(
            id_vars="x_AI ngành 2",
            var_name="Đường ngưỡng",
            value_name="x_H, tỷ VND"
        ),
        x="x_AI ngành 2",
        y="x_H, tỷ VND",
        color="Đường ngưỡng",
        title="Ảnh 9.9 — Ngưỡng đào tạo tối thiểu cho ngành chế biến chế tạo"
    )
    fig_threshold.update_layout(height=540)
    st.plotly_chart(fig_threshold, use_container_width=True)

    policy_card(
        "🏭",
        "Phân tích ngành chế biến chế tạo",
        f"Với ngành chế biến chế tạo, mỗi 1 tỷ VND đầu tư AI cần khoảng {threshold['ratio_required']:.2f} tỷ VND đào tạo lại để tốc độ tự động hóa không vượt quá năng lực đào tạo. Nếu dùng toàn bộ ngân sách trong riêng ngành này theo mức AI tối đa khả thi, cần tối thiểu khoảng {threshold['x_h_min']:,.0f} tỷ VND cho đào tạo.",
        "warning"
    )

    st.subheader("Câu 9.4.3 — Nhóm dễ bị tổn thương và Sankey dịch chuyển lao động")

    vulnerable = res_df[res_df["sector_id"].isin([1, 3, 4])].copy()

    if vulnerable["DisplacedJob"].sum() < 1:
        stress_ai = np.array([1200, 800, 1400], dtype=float)
        stress_h = np.array([600, 500, 800], dtype=float)
        vulnerable = compute_jobs(get_sector_data().query("sector_id in [1,3,4]"), stress_ai, stress_h)
        sankey_note = "Do nghiệm tối ưu hiện tại gần như không tạo luồng dịch chuyển ở nhóm 1,3,4, Sankey dưới đây dùng kịch bản minh họa stress-test để nhìn rõ cơ chế dịch chuyển."
    else:
        sankey_note = "Sankey dưới đây sử dụng trực tiếp nghiệm tối ưu hiện tại."

    st.info(sankey_note)

    labels = [
        "Lao động phổ thông nhóm 1,3,4",
        "Bị dịch chuyển bởi AI/tự động hóa",
        "Được đào tạo lại",
        "Chưa hấp thụ kịp",
        "Việc làm nâng cấp",
        "Việc làm AI mới",
        "NetJob an toàn",
        "Rủi ro an sinh",
    ]

    idx = {label: i for i, label in enumerate(labels)}

    displaced_total = vulnerable["DisplacedJob"].sum()
    retrained_total = np.minimum(vulnerable["DisplacedJob"], vulnerable["RetrainingCapacity"]).sum()
    gap_total = max(0, displaced_total - retrained_total)
    upgrade_total = vulnerable["UpgradeJob"].sum()
    newjob_total = vulnerable["NewJob_AI"].sum()

    links = [
        ("Lao động phổ thông nhóm 1,3,4", "Bị dịch chuyển bởi AI/tự động hóa", displaced_total),
        ("Bị dịch chuyển bởi AI/tự động hóa", "Được đào tạo lại", retrained_total),
        ("Bị dịch chuyển bởi AI/tự động hóa", "Chưa hấp thụ kịp", gap_total),
        ("Được đào tạo lại", "Việc làm nâng cấp", retrained_total),
        ("Việc làm nâng cấp", "NetJob an toàn", upgrade_total),
        ("Việc làm AI mới", "NetJob an toàn", newjob_total),
        ("Chưa hấp thụ kịp", "Rủi ro an sinh", gap_total),
    ]

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            label=labels,
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.4),
        ),
        link=dict(
            source=[idx[s] for s, t, v in links],
            target=[idx[t] for s, t, v in links],
            value=[max(0.001, float(v)) for s, t, v in links],
        )
    )])
    fig_sankey.update_layout(
        title="Ảnh 9.10 — Swimming lane/Sankey: luồng dịch chuyển lao động nhóm dễ bị tổn thương",
        height=600
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.dataframe(vulnerable[[
        "sector", "x_AI", "x_H", "DisplacedJob", "RetrainingCapacity",
        "UpgradeJob", "NewJob_AI", "NetJob"
    ]].round(3), use_container_width=True)

    st.subheader("Câu 9.4.4 — Mở rộng: không ngành nào mất quá 5% lao động")

    base_no_cap = solve_labor_lp(
        total_budget=total_budget,
        mode=mode,
        add_5pct_cap=False,
        max_sector_share=max_sector_share,
        min_ai_share=min_ai_share,
        min_h_share=min_h_share,
        min_vulnerable_h_share=min_vulnerable_h_share,
        min_manufacturing_h=min_manufacturing_h,
    )

    social_cap = solve_labor_lp(
        total_budget=total_budget,
        mode=mode,
        add_5pct_cap=True,
        max_sector_share=max_sector_share,
        min_ai_share=min_ai_share,
        min_h_share=min_h_share,
        min_vulnerable_h_share=min_vulnerable_h_share,
        min_manufacturing_h=min_manufacturing_h,
    )

    compare_rows = []
    for name, obj in [
        ("Không có ràng buộc 5%", base_no_cap),
        ("Có ràng buộc 5%", social_cap),
    ]:
        if obj is not None and obj["status"] == "Optimal":
            compare_rows.append({
                "Kịch bản": name,
                "Trạng thái": obj["status"],
                "Tổng NetJob": obj["summary"]["objective_total_netjob"],
                "Tổng dịch chuyển": obj["summary"]["total_displaced"],
                "x_AI": obj["summary"]["total_x_AI"],
                "x_H": obj["summary"]["total_x_H"],
            })
        else:
            compare_rows.append({
                "Kịch bản": name,
                "Trạng thái": obj["status"] if obj else "Không chạy",
                "Tổng NetJob": np.nan,
                "Tổng dịch chuyển": np.nan,
                "x_AI": np.nan,
                "x_H": np.nan,
            })

    compare_cap = pd.DataFrame(compare_rows)
    st.dataframe(compare_cap.round(3), use_container_width=True)

    fig_cap = px.bar(
        compare_cap,
        x="Kịch bản",
        y="Tổng NetJob",
        color="Trạng thái",
        text="Tổng NetJob",
        title="Ảnh 9.11 — Tác động của ràng buộc an sinh 5% lên tổng NetJob"
    )
    fig_cap.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_cap.update_layout(height=460)
    st.plotly_chart(fig_cap, use_container_width=True)

    if social_cap is not None and social_cap["status"] == "Optimal":
        st.success(
            "Bài toán vẫn khả thi khi thêm ràng buộc 5%. Điều này cho thấy có thể thiết kế chính sách AI theo hướng vừa tối đa hóa NetJob, vừa đặt trần an sinh cho dịch chuyển lao động."
        )
    else:
        st.warning(
            "Bài toán không khả thi với ràng buộc 5% trong cấu hình hiện tại. Cần tăng ngân sách đào tạo, giảm sàn AI, hoặc giảm tốc độ tự động hóa trong ngành rủi ro cao."
        )

    st.subheader("Phân tích nhạy cảm ngân sách")

    curve = sensitivity_budget_curve(mode=mode, add_5pct_cap=add_5pct_cap)

    st.dataframe(curve.round(3), use_container_width=True)

    fig_curve = px.line(
        curve,
        x="budget",
        y="total_netjob",
        markers=True,
        title="Ảnh 9.12 — Đường phản ứng NetJob theo ngân sách"
    )
    fig_curve.update_layout(
        height=480,
        xaxis_title="Ngân sách, tỷ VND",
        yaxis_title="Tổng NetJob"
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    return {
        "result": result,
        "social_cap": social_cap,
        "threshold": threshold,
        "curve": curve,
    }


def show_policy_discussion():
    st.header("9.5. Câu hỏi thảo luận chính sách")

    if not PULP_AVAILABLE:
        st.error("Cần cài PuLP để chạy phần thảo luận chính sách.")
        return

    sticker_header(
        "🧭🇻🇳",
        "Từ nghiệm LP đến chiến lược lao động trong kỷ nguyên AI",
        "Phần này dùng chính kết quả mô hình để trả lời các câu hỏi mở: ngành nào cần đào tạo lại, ngành nào nên tăng AI, và ràng buộc nào bảo vệ an sinh xã hội."
    )

    res = solve_labor_lp(total_budget=30000, mode="policy_balanced", add_5pct_cap=False)
    res_5 = solve_labor_lp(total_budget=30000, mode="policy_balanced", add_5pct_cap=True)

    if res is None or res["status"] != "Optimal":
        st.error("Mô hình chính sách cân bằng không tối ưu nên không thể thảo luận.")
        return

    df = res["result_df"]

    st.subheader("a) Ngành nào cần đầu tư đào tạo lại nhiều nhất? Có khớp với thực tế Việt Nam không?")

    h_rank = df.sort_values("x_H", ascending=False)[[
        "sector", "x_H", "x_AI", "labor_million", "risk_pct", "DisplacedJob", "RetrainingCapacity", "NetJob"
    ]].copy()

    st.dataframe(h_rank.round(3), use_container_width=True)

    fig_a = px.bar(
        h_rank,
        x="sector",
        y="x_H",
        color="risk_pct",
        text="x_H",
        title="Minh chứng câu a — Xếp hạng đầu tư đào tạo lại theo ngành"
    )
    fig_a.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_a.update_layout(height=520, xaxis_tickangle=-25)
    st.plotly_chart(fig_a, use_container_width=True)

    top_h = h_rank.iloc[0]

    policy_card(
        "🎓",
        f"Ngành cần đào tạo lại nhiều nhất: {top_h['sector']}",
        f"Theo nghiệm tối ưu, ngành này nhận khoảng {top_h['x_H']:,.0f} tỷ VND cho đào tạo lại. Điều này phản ánh ba yếu tố: quy mô lao động, risk tự động hóa và hiệu quả đào tạo b1/d1. Với Việt Nam, kết quả này cần được đọc cùng thực tế lao động: ngành đông lao động như nông nghiệp, chế biến chế tạo và bán lẻ thường cần chính sách nâng kỹ năng diện rộng.",
        "success"
    )

    st.markdown("""
    **Diễn giải chính sách:**  
    Nếu mô hình ưu tiên đào tạo cho ngành có quy mô lao động lớn, điều này khớp với thực tế Việt Nam: rủi ro AI không chỉ nằm ở ngành công nghệ cao, mà nằm ở nhóm lao động phổ thông trong các chuỗi sản xuất, bán lẻ, vận tải và nông nghiệp.  
    Chính sách tốt không phải là “đào tạo AI cho tất cả”, mà là **đào tạo đúng tầng kỹ năng**:

    - lao động phổ thông: kỹ năng số cơ bản, vận hành máy móc, an toàn dữ liệu;
    - lao động trung cấp: phân tích dữ liệu, vận hành hệ thống tự động;
    - lao động chuyên môn: AI ứng dụng, robot, quản trị nền tảng số.
    """)

    st.subheader("b) Tài chính-Ngân hàng rủi ro 52% nhưng tạo việc làm AI cao. Chiến lược gì?")

    fin = df[df["sector"] == "Tài chính-Ngân hàng"].iloc[0]

    fin_df = pd.DataFrame({
        "Chỉ báo": [
            "Risk tự động hóa",
            "a1 tạo việc làm AI",
            "Hệ số dịch chuyển hiệu dụng",
            "x_AI tối ưu",
            "x_H tối ưu",
            "NetJob",
            "Dịch chuyển/LĐ",
        ],
        "Giá trị": [
            fin["risk_pct"],
            fin["a1_new_ai_job_per_billion"],
            fin["displacement_coef"],
            fin["x_AI"],
            fin["x_H"],
            fin["NetJob"],
            fin["Displaced_share_labor_pct"],
        ],
        "Diễn giải": [
            "Nguy cơ thay thế một phần cao",
            "Khả năng tạo việc làm AI cũng cao",
            "Áp lực dịch chuyển từ AI",
            "Mức mở rộng AI được mô hình chọn",
            "Mức đào tạo kèm theo",
            "Việc làm ròng sau cân bằng AI-H",
            "Tỷ lệ lao động bị dịch chuyển so với quy mô ngành",
        ]
    })

    st.dataframe(fin_df.round(3), use_container_width=True)

    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(
        x=["NewJob_AI", "UpgradeJob", "DisplacedJob", "NetJob"],
        y=[fin["NewJob_AI"], fin["UpgradeJob"], fin["DisplacedJob"], fin["NetJob"]],
        text=[fin["NewJob_AI"], fin["UpgradeJob"], fin["DisplacedJob"], fin["NetJob"]],
        name="Tài chính-Ngân hàng"
    ))
    fig_b.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_b.update_layout(
        title="Minh chứng câu b — Tài chính-Ngân hàng: AI tạo việc mới nhưng cũng dịch chuyển việc cũ",
        height=470,
        yaxis_title="Việc làm"
    )
    st.plotly_chart(fig_b, use_container_width=True)

    policy_card(
        "🏦🤖",
        "Chiến lược khuyến nghị: AI-augment, không AI-replace",
        "Ngành Tài chính-Ngân hàng nên dùng AI để tăng năng suất nhân viên, tự động hóa tác vụ lặp lại, phát hiện gian lận và cá nhân hóa dịch vụ; đồng thời đào tạo lại nhân viên sang phân tích dữ liệu, quản trị rủi ro mô hình, tư vấn tài chính số và an toàn thông tin.",
        "purple"
    )

    st.markdown("""
    **Lập luận:**  
    Tài chính-Ngân hàng có rủi ro 52% vì nhiều tác vụ văn phòng, nhập liệu, kiểm tra hồ sơ và chăm sóc khách hàng có thể tự động hóa.  
    Nhưng ngành này cũng có hệ số `a1` cao, nghĩa là AI có thể tạo việc làm mới chất lượng cao.  
    Do đó, chiến lược không nên là hạn chế AI, mà là **AI đi kèm tái thiết kế công việc**:

    - giảm việc lặp lại;
    - tăng vai trò phân tích, kiểm soát, tư vấn;
    - đào tạo nhân viên về dữ liệu, an ninh mạng, quản trị mô hình AI;
    - đặt tiêu chuẩn đạo đức và minh bạch thuật toán.
    """)

    st.subheader("c) Có nên đầu tư x_AI vào Nông-Lâm-Thủy sản không?")

    agri = df[df["sector"] == "Nông-Lâm-Thủy sản"].iloc[0]

    agri_df = pd.DataFrame({
        "Chỉ báo": [
            "Lao động",
            "Risk",
            "a1 tạo việc làm AI",
            "b1 nâng kỹ năng",
            "Hệ số dịch chuyển hiệu dụng",
            "x_AI tối ưu",
            "x_H tối ưu",
            "NetJob",
        ],
        "Giá trị": [
            agri["labor_million"],
            agri["risk_pct"],
            agri["a1_new_ai_job_per_billion"],
            agri["b1_upgrade_job_per_billion"],
            agri["displacement_coef"],
            agri["x_AI"],
            agri["x_H"],
            agri["NetJob"],
        ],
        "Ý nghĩa": [
            "Quy mô lao động rất lớn",
            "Rủi ro tự động hóa thấp hơn tài chính/chế biến",
            "Tạo việc làm AI trực tiếp thấp",
            "Đào tạo/nâng kỹ năng có hiệu quả cao",
            "Mức dịch chuyển trên mỗi tỷ AI",
            "Mức AI mô hình khuyến nghị",
            "Mức đào tạo mô hình khuyến nghị",
            "Kết quả việc làm ròng",
        ]
    })

    st.dataframe(agri_df.round(3), use_container_width=True)

    fig_c = px.bar(
        pd.DataFrame({
            "Thành phần": ["NewJob_AI", "UpgradeJob", "DisplacedJob", "NetJob"],
            "Việc làm": [agri["NewJob_AI"], agri["UpgradeJob"], agri["DisplacedJob"], agri["NetJob"]],
        }),
        x="Thành phần",
        y="Việc làm",
        text="Việc làm",
        title="Minh chứng câu c — Nông-Lâm-Thủy sản: AI trực tiếp thấp nhưng đào tạo có vai trò lớn"
    )
    fig_c.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_c.update_layout(height=450)
    st.plotly_chart(fig_c, use_container_width=True)

    policy_card(
        "🌾📱",
        "Không nên hiểu AI nông nghiệp chỉ là robot thay lao động",
        "Vì a1 của Nông-Lâm-Thủy sản thấp, mô hình thường không khuyến nghị dồn x_AI trực tiếp vào ngành này nếu mục tiêu là tối đa hóa NetJob ngắn hạn. Nhưng do quy mô lao động rất lớn, chính sách nên ưu tiên AI hỗ trợ năng suất và đào tạo kỹ năng số cơ bản thay vì tự động hóa thay thế lao động.",
        "warning"
    )

    st.markdown("""
    **Hàm ý thực tiễn:**  
    Với Nông-Lâm-Thủy sản, AI nên đi theo hướng:

    - dự báo thời tiết, sâu bệnh, giá nông sản;
    - truy xuất nguồn gốc và thương mại điện tử nông sản;
    - tưới tiêu thông minh, quản lý vùng trồng;
    - đào tạo kỹ năng số cho hợp tác xã và hộ sản xuất.

    Như vậy, AI không nhất thiết tạo nhiều “việc làm AI” trực tiếp trong nông nghiệp, nhưng có thể tạo **năng suất và thu nhập**, đồng thời giảm rủi ro dịch chuyển lao động.
    """)

    st.subheader("d) Ràng buộc nào biểu diễn “tốc độ tự động hóa không vượt quá năng lực đào tạo lại”?")

    st.latex(r"""
    DisplacedJob_i \leq RetrainingCapacity_i
    """)

    st.latex(r"""
    c_{1i}x^{AI}_i risk_i \leq d_{1i}x^H_i
    """)

    st.markdown("""
    Đây là ràng buộc trung tâm của bài. Vế trái là số việc làm bị dịch chuyển do tự động hóa; vế phải là năng lực đào tạo lại.
    Nếu vế trái lớn hơn vế phải, nghĩa là AI đang đi nhanh hơn khả năng chuyển đổi kỹ năng của người lao động.
    """)

    cap_status = "khả thi" if res_5 is not None and res_5["status"] == "Optimal" else "không khả thi"

    compare = pd.DataFrame([
        {
            "Ràng buộc an sinh": "Không thêm trần 5%",
            "Trạng thái": res["status"],
            "Tổng NetJob": res["summary"]["objective_total_netjob"],
            "Tổng Displaced": res["summary"]["total_displaced"],
        },
        {
            "Ràng buộc an sinh": "Thêm DisplacedJobᵢ ≤ 0,05Lᵢ",
            "Trạng thái": res_5["status"] if res_5 else "Không chạy",
            "Tổng NetJob": res_5["summary"]["objective_total_netjob"] if res_5 and res_5["status"] == "Optimal" else np.nan,
            "Tổng Displaced": res_5["summary"]["total_displaced"] if res_5 and res_5["status"] == "Optimal" else np.nan,
        }
    ])

    st.dataframe(compare.round(3), use_container_width=True)

    fig_d = px.bar(
        compare,
        x="Ràng buộc an sinh",
        y="Tổng NetJob",
        color="Trạng thái",
        text="Tổng NetJob",
        title="Minh chứng câu d — Chi phí của ràng buộc bảo vệ lao động 5%"
    )
    fig_d.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_d.update_layout(height=460)
    st.plotly_chart(fig_d, use_container_width=True)

    policy_card(
        "🛡️",
        "Đề xuất bổ sung ràng buộc an sinh",
        f"Ràng buộc DisplacedJobᵢ ≤ 0,05Lᵢ trong cấu hình hiện tại là {cap_status}. Đây là hàng rào để không ngành nào chịu cú sốc dịch chuyển quá lớn trong một giai đoạn chính sách.",
        "success" if cap_status == "khả thi" else "danger"
    )

    st.markdown("""
    **Có thể bổ sung thêm các ràng buộc chính sách sau:**

    1. **Sàn đào tạo cho ngành đông lao động:** `x_Hᵢ ≥ H_minᵢ` với nông nghiệp, chế biến chế tạo, bán lẻ.
    2. **Trần tốc độ AI trong ngành rủi ro cao:** `x_AIᵢ ≤ AI_capᵢ`.
    3. **Tỷ lệ phối hợp AI-H:** `x_Hᵢ ≥ λᵢx_AIᵢ`.
    4. **Bảo vệ nhóm lao động phổ thông:** `DisplacedLowSkillᵢ ≤ RetrainingLowSkillᵢ`.
    5. **Quỹ chuyển đổi việc làm vùng:** thêm chiều vùng `r` để tránh dịch chuyển lao động tập trung ở một số địa phương.

    **Kết luận chính sách:** Việt Nam không nên làm chậm AI, nhưng phải làm nhanh hơn quá trình đào tạo lại.  
    Chính sách lao động trong kỷ nguyên AI cần đi theo nguyên tắc: **AI tăng tốc đến đâu, kỹ năng và an sinh phải đi kèm đến đó.**
    """)


# ---------------------------------------------------------
# 4. RENDER
# ---------------------------------------------------------
def render():
    st.title("👷‍♀️🤖 Bài 9 — Tác động AI tới thị trường lao động Việt Nam")
    inject_css()

    st.markdown("""
    Bài 9 xây dựng mô hình tuyến tính để mô phỏng tác động của AI và tự động hóa tới việc làm theo 8 ngành Việt Nam.
    Mục tiêu là tìm phân bổ ngân sách AI và đào tạo lại sao cho **NetJob ròng không âm** và tốc độ tự động hóa không vượt quá năng lực đào tạo lại.
    """)

    tabs = st.tabs([
        "9.1 Bối cảnh",
        "9.2 Mô hình toán học",
        "9.3 Tham số ngành",
        "9.4 Giải lập trình",
        "9.5 Chính sách",
    ])

    with tabs[0]:
        show_context()

    with tabs[1]:
        show_math_model()

    with tabs[2]:
        show_sector_data()

    with tabs[3]:
        show_programming_solution()

    with tabs[4]:
        show_policy_discussion()

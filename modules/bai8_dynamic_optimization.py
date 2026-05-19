import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =========================================================
# BÀI 8 — TỐI ƯU ĐỘNG PHÂN BỔ LIÊN THỜI GIAN 2026–2035
# Bản cải tiến: bối cảnh sâu hơn, công thức mở rộng,
# policy cards, growth accounting, policy cockpit.
# =========================================================


# ---------------------------------------------------------
# 1. THAM SỐ MÔ HÌNH
# ---------------------------------------------------------
def get_model_params():
    return {
        "T": 10,
        "years": list(range(2026, 2036)),

        # Cobb-Douglas elasticities
        "alpha_K": 0.33,
        "alpha_L": 0.42,
        "alpha_D": 0.10,
        "alpha_AI": 0.08,
        "alpha_H": 0.07,

        # Depreciation
        "delta_K": 0.05,
        "delta_D": 0.12,
        "delta_AI": 0.15,

        # Human capital
        "theta_H": 0.80,
        "mu_brain_drain": 0.02,

        # TFP spillover
        "phi_D": 0.003,
        "phi_AI": 0.002,
        "phi_H": 0.004,

        # Discount
        "rho": 0.97,

        # Labor path
        "L0": 53.9,
        "labor_growth": 0.004,

        # Initial conditions
        "K0": 27500.0,
        "D0": 20.3,
        "AI0": 86.0,
        "H0": 30.0,
        "A0": 1.0,

        # Investment policy constraints
        "min_consumption_share": 0.55,
        "max_total_investment_share": 0.42,
        "max_single_investment_share": 0.20,
        "min_H_investment_share": 0.03,
        "min_DAI_investment_share": 0.04,

        # Scale for converting investment values into D/AI/H indices
        "scale_K": 1.0,
        "scale_D": 0.004,
        "scale_AI": 0.006,
        "scale_H": 0.003,
    }


# ---------------------------------------------------------
# 2. HÀM TÍNH TOÁN CỐT LÕI
# ---------------------------------------------------------
def utility(C, gamma=1.0):
    C_safe = np.maximum(C, 1e-9)
    if abs(gamma - 1.0) < 1e-9:
        return np.log(C_safe)
    return (C_safe ** (1 - gamma) - 1) / (1 - gamma)


def production(A, K, L, D, AI, H, params):
    return (
        A
        * (K ** params["alpha_K"])
        * (L ** params["alpha_L"])
        * (D ** params["alpha_D"])
        * (AI ** params["alpha_AI"])
        * (H ** params["alpha_H"])
    )


def unpack_decision(z, T):
    z = np.array(z)
    C_share = z[0:T]
    IK_share = z[T:2 * T]
    ID_share = z[2 * T:3 * T]
    IAI_share = z[3 * T:4 * T]
    IH_share = z[4 * T:5 * T]
    return C_share, IK_share, ID_share, IAI_share, IH_share


def simulate_path(z, params=None, rho=None, gamma_utility=1.0, shock_year=None, shock_pct=0.0):
    if params is None:
        params = get_model_params()
    if rho is None:
        rho = params["rho"]

    T = params["T"]
    years = params["years"]
    C_share, IK_share, ID_share, IAI_share, IH_share = unpack_decision(z, T)

    K = np.zeros(T + 1)
    D = np.zeros(T + 1)
    AI = np.zeros(T + 1)
    H = np.zeros(T + 1)
    A = np.zeros(T + 1)
    L = np.zeros(T)

    Y_plan = np.zeros(T)
    Y = np.zeros(T)
    C = np.zeros(T)
    I_K = np.zeros(T)
    I_D = np.zeros(T)
    I_AI = np.zeros(T)
    I_H = np.zeros(T)
    total_I = np.zeros(T)
    welfare_terms = np.zeros(T)

    K[0] = params["K0"]
    D[0] = params["D0"]
    AI[0] = params["AI0"]
    H[0] = params["H0"]
    A[0] = params["A0"]

    penalties = []
    feasible = True

    for t in range(T):
        year = years[t]
        L[t] = params["L0"] * ((1 + params["labor_growth"]) ** t)

        Y_plan[t] = production(A[t], K[t], L[t], D[t], AI[t], H[t], params)
        Y[t] = Y_plan[t]

        if shock_year is not None and year == shock_year:
            Y[t] = Y_plan[t] * (1 - shock_pct)

        C[t] = C_share[t] * Y[t]
        I_K[t] = IK_share[t] * Y[t]
        I_D[t] = ID_share[t] * Y[t]
        I_AI[t] = IAI_share[t] * Y[t]
        I_H[t] = IH_share[t] * Y[t]
        total_I[t] = I_K[t] + I_D[t] + I_AI[t] + I_H[t]

        if C[t] + total_I[t] > Y[t] + 1e-6:
            feasible = False
            penalties.append((C[t] + total_I[t] - Y[t]) ** 2)
        if C[t] <= 0:
            feasible = False
            penalties.append(1e6)

        K[t + 1] = (1 - params["delta_K"]) * K[t] + params["scale_K"] * I_K[t]
        D[t + 1] = (1 - params["delta_D"]) * D[t] + params["scale_D"] * I_D[t]
        AI[t + 1] = (1 - params["delta_AI"]) * AI[t] + params["scale_AI"] * I_AI[t]
        H[t + 1] = H[t] + params["theta_H"] * params["scale_H"] * I_H[t] - params["mu_brain_drain"] * H[t]

        A[t + 1] = A[t] * (
            1
            + params["phi_D"] * D[t] / 100
            + params["phi_AI"] * AI[t] / 100
            + params["phi_H"] * H[t] / 100
        )

        welfare_terms[t] = (rho ** t) * utility(C[t], gamma_utility)

        if min(K[t + 1], D[t + 1], AI[t + 1], H[t + 1], A[t + 1]) <= 0:
            feasible = False
            penalties.append(1e6)

    path = pd.DataFrame({
        "year": years,
        "K": K[:-1],
        "D": D[:-1],
        "AI": AI[:-1],
        "H": H[:-1],
        "A_TFP": A[:-1],
        "L": L,
        "Y_plan": Y_plan,
        "Y": Y,
        "C": C,
        "I_K": I_K,
        "I_D": I_D,
        "I_AI": I_AI,
        "I_H": I_H,
        "Total_investment": total_I,
        "C_share": C_share,
        "IK_share": IK_share,
        "ID_share": ID_share,
        "IAI_share": IAI_share,
        "IH_share": IH_share,
        "Welfare_term": welfare_terms,
    })

    terminal = pd.DataFrame({
        "year": [years[-1] + 1],
        "K": [K[-1]],
        "D": [D[-1]],
        "AI": [AI[-1]],
        "H": [H[-1]],
        "A_TFP": [A[-1]],
    })

    return {
        "path": path,
        "terminal": terminal,
        "welfare": welfare_terms.sum(),
        "feasible": feasible,
        "penalty": float(np.sum(penalties)) if penalties else 0.0,
    }


def objective_slsqp(z, params, rho, gamma_utility, shock_year=None, shock_pct=0.0):
    sim = simulate_path(
        z,
        params=params,
        rho=rho,
        gamma_utility=gamma_utility,
        shock_year=shock_year,
        shock_pct=shock_pct,
    )
    return -sim["welfare"] + 1e6 * sim["penalty"]


def build_constraints(params):
    T = params["T"]

    def budget_share_constraint(z):
        C_share, IK_share, ID_share, IAI_share, IH_share = unpack_decision(z, T)
        return 1.0 - (C_share + IK_share + ID_share + IAI_share + IH_share)

    def max_investment_constraint(z):
        C_share, IK_share, ID_share, IAI_share, IH_share = unpack_decision(z, T)
        return params["max_total_investment_share"] - (IK_share + ID_share + IAI_share + IH_share)

    def min_consumption_constraint(z):
        C_share, IK_share, ID_share, IAI_share, IH_share = unpack_decision(z, T)
        return C_share - params["min_consumption_share"]

    def min_h_constraint(z):
        C_share, IK_share, ID_share, IAI_share, IH_share = unpack_decision(z, T)
        return IH_share - params["min_H_investment_share"]

    def min_dai_constraint(z):
        C_share, IK_share, ID_share, IAI_share, IH_share = unpack_decision(z, T)
        return (ID_share + IAI_share) - params["min_DAI_investment_share"]

    return [
        {"type": "ineq", "fun": budget_share_constraint},
        {"type": "ineq", "fun": max_investment_constraint},
        {"type": "ineq", "fun": min_consumption_constraint},
        {"type": "ineq", "fun": min_h_constraint},
        {"type": "ineq", "fun": min_dai_constraint},
    ]


@st.cache_data(show_spinner=False)
def solve_dynamic_model(rho=0.97, gamma_utility=1.0, shock_year=None, shock_pct=0.0, strategy_hint="balanced"):
    if not SCIPY_AVAILABLE:
        return None

    params = get_model_params()
    T = params["T"]

    if strategy_hint == "front_load":
        t = np.arange(T)
        inv_base = np.clip(0.37 - 0.012 * t, 0.23, 0.40)
    elif strategy_hint == "even":
        inv_base = np.ones(T) * 0.34
    else:
        inv_base = np.linspace(0.36, 0.30, T)

    C0 = 1 - inv_base
    IK0 = inv_base * 0.42
    ID0 = inv_base * 0.18
    IAI0 = inv_base * 0.16
    IH0 = inv_base * 0.24
    z0 = np.concatenate([C0, IK0, ID0, IAI0, IH0])

    lower = np.zeros(5 * T)
    upper = np.ones(5 * T) * params["max_single_investment_share"]
    upper[0:T] = 0.95
    lower[0:T] = params["min_consumption_share"]
    lower[4 * T:5 * T] = params["min_H_investment_share"]
    bounds = list(zip(lower, upper))

    callback_history = []

    def callback(xk):
        sim = simulate_path(xk, params=params, rho=rho, gamma_utility=gamma_utility)
        callback_history.append({
            "iteration": len(callback_history) + 1,
            "welfare": sim["welfare"],
            "penalty": sim["penalty"],
        })

    res = minimize(
        objective_slsqp,
        z0,
        args=(params, rho, gamma_utility, shock_year, shock_pct),
        method="SLSQP",
        bounds=bounds,
        constraints=build_constraints(params),
        callback=callback,
        options={"maxiter": 400, "ftol": 1e-7, "disp": False},
    )

    sim = simulate_path(
        res.x,
        params=params,
        rho=rho,
        gamma_utility=gamma_utility,
        shock_year=shock_year,
        shock_pct=shock_pct,
    )

    return {
        "success": res.success,
        "message": res.message,
        "objective": -res.fun,
        "z": res.x,
        "path": sim["path"],
        "terminal": sim["terminal"],
        "welfare": sim["welfare"],
        "callback": pd.DataFrame(callback_history),
        "params": params,
    }


def simulate_fixed_strategy(strategy="even", rho=0.97, gamma_utility=1.0, shock_year=None, shock_pct=0.0):
    params = get_model_params()
    T = params["T"]

    if strategy == "front_load":
        t = np.arange(T)
        inv = np.clip(0.42 - 0.018 * t, 0.22, 0.42)
    elif strategy == "back_load":
        t = np.arange(T)
        inv = np.clip(0.25 + 0.014 * t, 0.25, 0.40)
    else:
        inv = np.ones(T) * 0.34

    C = 1 - inv
    IK = inv * 0.42
    ID = inv * 0.18
    IAI = inv * 0.16
    IH = inv * 0.24
    z = np.concatenate([C, IK, ID, IAI, IH])

    sim = simulate_path(
        z,
        params=params,
        rho=rho,
        gamma_utility=gamma_utility,
        shock_year=shock_year,
        shock_pct=shock_pct,
    )

    return {
        "strategy": strategy,
        "z": z,
        "path": sim["path"],
        "terminal": sim["terminal"],
        "welfare": sim["welfare"],
        "params": params,
    }


def check_path_constraints(path, params):
    checks = []

    for _, row in path.iterrows():
        year = int(row["year"])
        budget_gap = row["Y"] - row["C"] - row["Total_investment"]
        invest_share = row["Total_investment"] / row["Y"]
        c_share = row["C"] / row["Y"]
        h_share = row["I_H"] / row["Y"]
        dai_share = (row["I_D"] + row["I_AI"]) / row["Y"]

        checks.append({
            "year": year,
            "budget_gap": budget_gap,
            "investment_share": invest_share,
            "consumption_share": c_share,
            "H_share": h_share,
            "D_AI_share": dai_share,
            "budget_ok": budget_gap >= -1e-5,
            "investment_cap_ok": invest_share <= params["max_total_investment_share"] + 1e-5,
            "consumption_floor_ok": c_share >= params["min_consumption_share"] - 1e-5,
            "H_floor_ok": h_share >= params["min_H_investment_share"] - 1e-5,
            "D_AI_floor_ok": dai_share >= params["min_DAI_investment_share"] - 1e-5,
        })

    return pd.DataFrame(checks)


# ---------------------------------------------------------
# 3. HÀM PHỤ GIAO DIỆN, PHÂN TÍCH NÂNG CAO
# ---------------------------------------------------------
def policy_card(icon, title, body, tone="info"):
    colors = {
        "info": "#0ea5e9",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "purple": "#8b5cf6",
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
            <div style="font-size: 22px; font-weight: 700; margin-bottom: 6px;">
                {icon} {title}
            </div>
            <div style="font-size: 16px; line-height: 1.6;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_sticker_header(icon, title, subtitle):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, rgba(14,165,233,0.18), rgba(139,92,246,0.14));
            padding: 18px 22px;
            border-radius: 18px;
            margin: 10px 0 18px 0;
            border: 1px solid rgba(255,255,255,0.08);
        ">
            <div style="font-size: 28px; font-weight: 800;">{icon} {title}</div>
            <div style="font-size: 16px; opacity: 0.88; margin-top: 6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classify_loading_pattern(path):
    early = path[path["year"].between(2026, 2028)]["Total_investment"].sum()
    middle = path[path["year"].between(2029, 2032)]["Total_investment"].sum()
    late = path[path["year"].between(2033, 2035)]["Total_investment"].sum()

    if early > late * 1.08:
        pattern = "Front-loaded"
        meaning = "Đầu tư được đẩy mạnh sớm để tạo hiệu ứng tích lũy vốn và năng suất."
    elif late > early * 1.08:
        pattern = "Back-loaded"
        meaning = "Đầu tư dồn về cuối kỳ, thường phản ánh ưu tiên tiêu dùng hiện tại hoặc ràng buộc ngân sách ngắn hạn."
    else:
        pattern = "Smooth/Balanced"
        meaning = "Đầu tư được làm mượt theo thời gian, cân bằng giữa tiêu dùng hiện tại và tích lũy tương lai."

    return {"early": early, "middle": middle, "late": late, "pattern": pattern, "meaning": meaning}


def calculate_growth_decomposition(path):
    params = get_model_params()
    df = path.copy()

    for col in ["Y", "A_TFP", "K", "L", "D", "AI", "H"]:
        df[f"dln_{col}"] = np.log(df[col]).diff()

    decomp = pd.DataFrame({
        "year": df["year"],
        "TFP": df["dln_A_TFP"],
        "K": params["alpha_K"] * df["dln_K"],
        "L": params["alpha_L"] * df["dln_L"],
        "D": params["alpha_D"] * df["dln_D"],
        "AI": params["alpha_AI"] * df["dln_AI"],
        "H": params["alpha_H"] * df["dln_H"],
        "Actual_dlnY": df["dln_Y"],
    }).dropna().reset_index(drop=True)

    decomp["Model_sum"] = decomp[["TFP", "K", "L", "D", "AI", "H"]].sum(axis=1)

    decomp_long = decomp.melt(
        id_vars=["year"],
        value_vars=["TFP", "K", "L", "D", "AI", "H"],
        var_name="Nguồn đóng góp",
        value_name="Đóng góp log-growth",
    )

    avg = decomp[["TFP", "K", "L", "D", "AI", "H"]].mean().reset_index()
    avg.columns = ["Nguồn đóng góp", "Đóng góp bình quân"]
    denom = avg["Đóng góp bình quân"].sum()
    avg["Tỷ trọng trong tăng trưởng, %"] = np.where(
        abs(denom) < 1e-12,
        0,
        avg["Đóng góp bình quân"] / denom * 100,
    )

    return decomp, decomp_long, avg


def investment_policy_matrix(path):
    df = path.copy()
    return pd.DataFrame({
        "Chỉ báo": [
            "Tổng đầu tư 2026–2028",
            "Tổng đầu tư 2033–2035",
            "AI/H trung bình",
            "Độ lệch chuẩn AI/H",
            "Tăng Y 2026–2035",
            "Tăng TFP 2026–2035",
            "Tăng H 2026–2035",
        ],
        "Giá trị": [
            df[df["year"].between(2026, 2028)]["Total_investment"].sum(),
            df[df["year"].between(2033, 2035)]["Total_investment"].sum(),
            (df["I_AI"] / (df["I_H"] + 1e-9)).mean(),
            (df["I_AI"] / (df["I_H"] + 1e-9)).std(),
            (df["Y"].iloc[-1] / df["Y"].iloc[0] - 1) * 100,
            (df["A_TFP"].iloc[-1] / df["A_TFP"].iloc[0] - 1) * 100,
            (df["H"].iloc[-1] / df["H"].iloc[0] - 1) * 100,
        ],
        "Diễn giải nhanh": [
            "Cường độ đầu tư đầu kỳ.",
            "Cường độ đầu tư cuối kỳ.",
            "Tỷ lệ phối hợp AI với nhân lực số.",
            "Độ ổn định của phối hợp AI-H.",
            "Mức mở rộng sản lượng trong 10 năm.",
            "Tác động năng suất dài hạn.",
            "Mức tích lũy nhân lực số.",
        ],
    })


# ---------------------------------------------------------
# 4. PHẦN 8.1 — BỐI CẢNH
# ---------------------------------------------------------
def show_context():
    st.header("8.1. Bối cảnh Việt Nam")

    add_sticker_header(
        "🇻🇳⏳",
        "Từ tăng trưởng ngắn hạn đến quỹ đạo phát triển dài hạn",
        "Bài 8 không hỏi ‘nên đầu tư bao nhiêu trong một năm’, mà hỏi: Việt Nam nên đi theo quỹ đạo đầu tư nào từ 2026 đến 2035 để tiến gần hơn mục tiêu 2030 và 2045?",
    )

    policy_card(
        "🎯",
        "Vấn đề chính sách trung tâm",
        "Việt Nam muốn đạt mục tiêu thu nhập trung bình cao vào năm 2030 và thu nhập cao vào năm 2045. Muốn vậy, tăng trưởng không thể chỉ dựa vào vốn vật chất truyền thống, mà phải chuyển sang quỹ đạo dựa trên số hóa, AI, TFP và nhân lực chất lượng cao.",
        "success",
    )
    policy_card(
        "⚖️",
        "Đánh đổi liên thời gian",
        "Nếu đầu tư nhiều hôm nay, tiêu dùng hiện tại giảm nhưng năng lực sản xuất tương lai tăng. Nếu ưu tiên tiêu dùng hiện tại, nền kinh tế có thể thiếu nền tảng công nghệ và nhân lực cho giai đoạn sau.",
        "warning",
    )
    policy_card(
        "🧠",
        "Ý nghĩa của mô hình",
        "Mô hình động giúp nhìn thấy độ trễ của đầu tư: K, D, AI và H không chỉ tạo sản lượng tức thời, mà còn tích lũy thành năng lực sản xuất và làm tăng TFP trong nhiều năm.",
        "purple",
    )

    timeline = pd.DataFrame({
        "Mốc chính sách": ["2026", "2030", "2035", "2045"],
        "Câu hỏi phát triển": [
            "Bắt đầu quỹ đạo đầu tư hậu 2020–2025",
            "Có đủ nền tảng để vào nhóm thu nhập trung bình cao không?",
            "Kết quả tích lũy sau 10 năm đầu tư số là gì?",
            "Nền kinh tế có đủ năng lực công nghệ để tiến tới thu nhập cao không?",
        ],
        "Biến mô hình liên quan": [
            "K0, D0, AI0, H0, A0",
            "Y, A_TFP, D, AI, H",
            "K_2035, D_2035, AI_2035, H_2035",
            "TFP dài hạn, nhân lực số, năng lực hấp thụ AI",
        ],
        "Thông điệp": [
            "Điểm xuất phát quyết định tốc độ tích lũy.",
            "Cần đầu tư đủ sớm để có kết quả trước 2030.",
            "Đánh giá chất lượng quỹ đạo, không chỉ mức GDP.",
            "Tầm nhìn dài hạn đòi hỏi tránh dưới đầu tư vào R&D, AI và H.",
        ],
    })

    st.subheader("🗺️ Bảng 8.1 — Khung thời gian chính sách 2026–2045")
    st.dataframe(timeline, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giai đoạn mô phỏng", "2026–2035", "10 năm")
    c2.metric("Biến trạng thái", "5", "K, D, AI, H, A")
    c3.metric("Biến điều khiển", "5", "C, IK, ID, IAI, IH")
    c4.metric("Trọng tâm", "Welfare", "phúc lợi liên thời gian")

    st.subheader("Ảnh 8.1 — Bản đồ tư duy tối ưu động cho Việt Nam")
    labels = [
        "GDP năm t", "Tiêu dùng hiện tại", "Đầu tư vật chất K", "Hạ tầng số D",
        "Năng lực AI", "Nhân lực số H", "Tích lũy vốn", "Lan tỏa TFP",
        "GDP năm t+1", "Phúc lợi xã hội", "Mục tiêu 2030–2045",
    ]
    fig_flow = go.Figure(data=[go.Sankey(
        node=dict(pad=18, thickness=20, line=dict(color="black", width=0.3), label=labels),
        link=dict(
            source=[0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9],
            target=[1, 2, 3, 4, 5, 6, 6, 6, 6, 8, 8, 9, 10],
            value=[45, 14, 8, 7, 10, 14, 8, 7, 10, 25, 20, 45, 45],
        ),
    )])
    fig_flow.update_layout(title="Ảnh 8.1 — Từ GDP hiện tại đến tích lũy vốn, TFP và mục tiêu dài hạn", height=560)
    st.plotly_chart(fig_flow, use_container_width=True)

    strategic_map = pd.DataFrame({
        "Trụ cột": ["K - Vốn vật chất", "D - Hạ tầng số", "AI - Năng lực AI", "H - Nhân lực số", "A - TFP"],
        "Vai trò": [
            "Nền tảng tăng trưởng truyền thống",
            "Hạ tầng cho kinh tế số, dữ liệu và dịch vụ công",
            "Tự động hóa, phân tích dữ liệu, tăng năng suất",
            "Năng lực hấp thụ và vận hành công nghệ",
            "Chất lượng tăng trưởng và năng suất dài hạn",
        ],
        "Rủi ro nếu thiếu": [
            "Thiếu năng lực sản xuất cơ bản",
            "Chuyển đổi số phân mảnh, chi phí giao dịch cao",
            "Không tạo được bước nhảy năng suất",
            "Có công nghệ nhưng thiếu người dùng và quản trị",
            "Tăng trưởng chỉ dựa vào mở rộng đầu vào",
        ],
        "Liên hệ chính sách": [
            "Đầu tư công, hạ tầng kinh tế",
            "QĐ 749/QĐ-TTg về chuyển đổi số quốc gia",
            "Chiến lược AI, Nghị quyết 57-NQ/TW",
            "Đào tạo kỹ năng số, kỹ sư AI, bán dẫn",
            "Mục tiêu tăng trưởng chất lượng cao đến 2030–2045",
        ],
    })
    st.subheader("📌 Bảng 8.2 — Năm trụ cột của quỹ đạo tăng trưởng mới")
    st.dataframe(strategic_map, use_container_width=True)

    st.info("Điểm nhấn của Bài 8: đây là mô hình ‘hy sinh có tính toán’. Một phần tiêu dùng hiện tại được chuyển thành đầu tư để đổi lấy năng lực sản xuất, TFP và phúc lợi tương lai.")


# ---------------------------------------------------------
# 5. PHẦN 8.2 — MÔ HÌNH TOÁN HỌC
# ---------------------------------------------------------
def show_math_model():
    st.header("8.2. Mô hình toán học")

    add_sticker_header(
        "🧮🚀",
        "Mô hình động: tiêu dùng hôm nay hay năng lực sản xuất ngày mai?",
        "Bài toán được xây dựng như một hệ động lực: quyết định đầu tư hôm nay làm thay đổi K, D, AI, H, A trong tương lai.",
    )

    st.subheader("1️⃣ Hàm mục tiêu: tối đa hóa phúc lợi liên thời gian")
    st.latex(r"""
    \max_{\{C_t,I_{K,t},I_{D,t},I_{AI,t},I_{H,t}\}}
    W = \sum_{t=2026}^{2035}\rho^{t-2026}U(C_t)
    """)
    st.markdown("""
    Hàm mục tiêu không tối đa hóa GDP đơn thuần. Nó tối đa hóa **phúc lợi xã hội**, tức là lợi ích từ tiêu dùng qua thời gian.
    Đây là điểm quan trọng: một phương án GDP cao nhưng làm tiêu dùng hiện tại giảm quá mạnh có thể không tối ưu về welfare.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"U(C_t)=\ln(C_t)")
        st.info("Log utility làm trơn tiêu dùng: tăng thêm 1 đơn vị tiêu dùng có giá trị cao hơn khi C còn thấp.")
    with col2:
        st.latex(r"U(C_t)=\frac{C_t^{1-\gamma}-1}{1-\gamma},\quad \gamma=1.5")
        st.info("CRRA phản ánh xã hội ngại biến động tiêu dùng mạnh hơn, phù hợp khi muốn ổn định đời sống.")

    st.subheader("2️⃣ Hàm sản xuất Cobb-Douglas mở rộng")
    st.latex(r"Y_t = A_t K_t^{0.33} L_t^{0.42}D_t^{0.10}AI_t^{0.08}H_t^{0.07}")

    cd_df = pd.DataFrame({
        "Yếu tố": ["K", "L", "D", "AI", "H", "A"],
        "Hệ số / vai trò": ["0.33", "0.42", "0.10", "0.08", "0.07", "TFP"],
        "Diễn giải kinh tế": [
            "Vốn vật chất vẫn là trụ cột tăng trưởng truyền thống.",
            "Lao động có vai trò lớn, phản ánh nền kinh tế còn thâm dụng lao động.",
            "Hạ tầng số tạo nền tảng cho giao dịch, dữ liệu và dịch vụ công.",
            "AI tạo năng suất, tự động hóa và khả năng ra quyết định.",
            "Nhân lực số giúp hấp thụ, vận hành và kiểm soát công nghệ.",
            "Phần tăng trưởng chất lượng, không giải thích trực tiếp bằng đầu vào quan sát.",
        ],
        "Thông điệp chính sách": [
            "Không thể bỏ đầu tư vật chất.",
            "Nâng chất lượng lao động quan trọng hơn chỉ tăng số lượng.",
            "D là điều kiện nền cho AI và dữ liệu.",
            "AI cần đi kèm quản trị và nhân lực.",
            "H là điều kiện để tránh ‘có máy nhưng thiếu người’.",
            "TFP là thước đo chất lượng mô hình tăng trưởng.",
        ],
    })
    st.dataframe(cd_df, use_container_width=True)

    st.subheader("3️⃣ Phương trình chuyển trạng thái")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(r"K_{t+1}=(1-\delta_K)K_t+I_{K,t}")
        st.latex(r"D_{t+1}=(1-\delta_D)D_t+s_D I_{D,t}")
        st.latex(r"AI_{t+1}=(1-\delta_{AI})AI_t+s_{AI} I_{AI,t}")
    with c2:
        st.latex(r"H_{t+1}=H_t+\theta_H s_H I_{H,t}-\mu H_t")
        st.latex(r"A_{t+1}=A_t\left(1+\phi_D\frac{D_t}{100}+\phi_{AI}\frac{AI_t}{100}+\phi_H\frac{H_t}{100}\right)")
        st.latex(r"C_t+I_{K,t}+I_{D,t}+I_{AI,t}+I_{H,t}\leq Y_t")

    st.markdown("""
    Trong code, các hệ số `s_D`, `s_AI`, `s_H` được dùng để chuyển đầu tư tính bằng giá trị sản lượng sang các chỉ số D, AI, H.
    Điều này giúp tránh lỗi quy mô vì K là nghìn tỷ VND, còn D, AI, H là chỉ số/tỷ lệ.
    """)

    params = get_model_params()
    param_df = pd.DataFrame({
        "Nhóm": [
            "Khấu hao", "Khấu hao", "Khấu hao",
            "Chuyển hóa đầu tư", "Chuyển hóa đầu tư", "Chuyển hóa đầu tư",
            "Nhân lực", "Nhân lực",
            "TFP", "TFP", "TFP",
            "Chiết khấu",
        ],
        "Ký hiệu": [
            "δ_K", "δ_D", "δ_AI", "s_D", "s_AI", "s_H",
            "θ_H", "μ", "φ_D", "φ_AI", "φ_H", "ρ",
        ],
        "Giá trị": [
            params["delta_K"], params["delta_D"], params["delta_AI"],
            params["scale_D"], params["scale_AI"], params["scale_H"],
            params["theta_H"], params["mu_brain_drain"],
            params["phi_D"], params["phi_AI"], params["phi_H"], params["rho"],
        ],
        "Ý nghĩa": [
            "Khấu hao vốn vật chất",
            "Khấu hao hạ tầng số",
            "Khấu hao vốn AI",
            "1 đơn vị đầu tư D làm tăng chỉ số D bao nhiêu",
            "1 đơn vị đầu tư AI làm tăng năng lực AI bao nhiêu",
            "1 đơn vị đầu tư H làm tăng vốn nhân lực bao nhiêu",
            "Hiệu quả chuyển đào tạo thành vốn nhân lực",
            "Hao hụt kỹ năng/chảy máu chất xám",
            "D tác động đến TFP",
            "AI tác động đến TFP",
            "H tác động đến TFP",
            "Mức coi trọng tương lai",
        ],
    })
    st.subheader("📋 Bảng 8.3 — Tham số mô hình và ý nghĩa")
    st.dataframe(param_df, use_container_width=True)

    st.subheader("4️⃣ Công thức cải tiến: chỉ số cân bằng AI - Nhân lực")
    st.latex(r"Balance_{AI,H,t}=\frac{I_{AI,t}}{I_{H,t}+\varepsilon}")
    st.markdown("""
    Chỉ số này cho biết AI có đang được đầu tư nhanh hơn nhân lực số hay không.
    Nếu chỉ số quá cao, rủi ro là Việt Nam đầu tư mạnh vào công nghệ nhưng thiếu kỹ sư, chuyên gia dữ liệu và năng lực quản trị AI.
    Nếu chỉ số quá thấp, có thể nhân lực được chuẩn bị tốt nhưng thiếu nền tảng công nghệ để hấp thụ.
    """)

    st.subheader("5️⃣ Công thức cải tiến: áp lực hy sinh tiêu dùng hiện tại")
    st.latex(r"Sacrifice_t = 1-\frac{C_t}{Y_t}=\frac{I_{K,t}+I_{D,t}+I_{AI,t}+I_{H,t}}{Y_t}")
    st.markdown("""
    Đây là tỷ lệ sản lượng được chuyển từ tiêu dùng hiện tại sang đầu tư tương lai.
    Chỉ số này rất quan trọng về chính sách vì đầu tư dài hạn luôn có chi phí xã hội ngắn hạn.
    """)

    st.subheader("6️⃣ Công thức cải tiến: đóng góp tăng trưởng gần đúng")
    st.latex(r"""
    \Delta \ln Y_t
    \approx
    \Delta \ln A_t
    +0.33\Delta \ln K_t
    +0.42\Delta \ln L_t
    +0.10\Delta \ln D_t
    +0.08\Delta \ln AI_t
    +0.07\Delta \ln H_t
    """)
    st.markdown("""
    Phân rã này giúp chuyển mô hình từ “hộp đen tối ưu” thành một bảng giải thích:
    tăng trưởng đến từ vốn, lao động, số hóa, AI, nhân lực hay TFP?
    """)

    initial_df = pd.DataFrame({
        "Biến trạng thái ban đầu 2026": ["K0", "L0", "D0", "AI0", "H0", "A0"],
        "Giá trị": [params["K0"], params["L0"], params["D0"], params["AI0"], params["H0"], params["A0"]],
        "Đơn vị": ["nghìn tỷ VND", "triệu lao động", "% GDP / chỉ số", "nghìn DN số", "% lao động kỹ năng", "chỉ số TFP"],
        "Vai trò trong mô hình": [
            "Nền vốn vật chất ban đầu",
            "Quy mô lao động",
            "Mức số hóa ban đầu",
            "Năng lực AI ban đầu",
            "Nền nhân lực số",
            "Năng suất nhân tố tổng hợp",
        ],
    })
    st.subheader("📍 Bảng 8.4 — Điều kiện ban đầu năm 2026")
    st.dataframe(initial_df, use_container_width=True)

    st.success("Điểm nhấn: mô hình không chỉ dự báo GDP. Nó mô phỏng cách Việt Nam chuyển một phần sản lượng hiện tại thành nền tảng tăng trưởng tương lai thông qua K, D, AI, H và TFP.")


# ---------------------------------------------------------
# 6. PHẦN 8.3 — GIẢI LẬP TRÌNH
# ---------------------------------------------------------
def show_programming_solution():
    st.header("8.3. Giải yêu cầu lập trình")

    st.markdown("""
    Vì hàm Cobb-Douglas và động học TFP tạo bài toán phi tuyến, module này dùng **scipy.optimize.minimize** với thuật toán **SLSQP**.
    Biến quyết định là tỷ trọng tiêu dùng và tỷ trọng đầu tư vào K, D, AI, H trong từng năm.
    """)

    if not SCIPY_AVAILABLE:
        st.error("Chưa cài scipy. Hãy thêm `scipy` vào requirements.txt.")
        return None

    st.subheader("Thiết lập tham số mô phỏng")
    c1, c2, c3 = st.columns(3)
    rho = c1.slider("ρ - hệ số chiết khấu", 0.85, 0.99, 0.97, 0.01, key="bai8_rho")
    utility_type = c2.selectbox("Hàm thỏa dụng", ["Log utility", "CRRA gamma = 1.5"], key="bai8_utility")
    gamma_utility = 1.0 if utility_type == "Log utility" else 1.5
    strategy_hint = c3.selectbox(
        "Gợi ý nghiệm khởi tạo",
        ["balanced", "front_load", "even"],
        format_func=lambda x: {"balanced": "Cân bằng", "front_load": "Front-load", "even": "Trải đều"}[x],
        key="bai8_strategy_hint",
    )

    st.subheader("Câu 8.3.1 — Giải tối ưu động bằng SLSQP")
    with st.spinner("Đang giải tối ưu động bằng SLSQP..."):
        opt = solve_dynamic_model(rho=rho, gamma_utility=gamma_utility, strategy_hint=strategy_hint)

    if opt is None:
        st.error("Không thể chạy tối ưu. Kiểm tra scipy trong requirements.txt.")
        return None

    path = opt["path"]
    params = opt["params"]
    checks = check_path_constraints(path, params)

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Trạng thái", "Tối ưu" if opt["success"] else "Cần kiểm tra")
    c5.metric("Welfare", f"{opt['welfare']:.3f}")
    c6.metric("Y năm 2035", f"{path['Y'].iloc[-1]:,.2f}")
    c7.metric("C năm 2035", f"{path['C'].iloc[-1]:,.2f}")

    if not opt["success"]:
        st.warning(f"SLSQP báo: {opt['message']}")

    st.markdown("#### Bảng quỹ đạo tối ưu")
    st.dataframe(path.round(4), use_container_width=True)

    st.markdown("#### Kiểm tra ràng buộc")
    st.dataframe(checks.round(5), use_container_width=True)

    if checks[["budget_ok", "investment_cap_ok", "consumption_floor_ok", "H_floor_ok", "D_AI_floor_ok"]].all().all():
        st.success("Tất cả ràng buộc chính đều đạt trong nghiệm hiện tại.")
    else:
        st.warning("Một số ràng buộc cần kiểm tra lại. Có thể cần đổi nghiệm khởi tạo hoặc nới tham số.")

    if not opt["callback"].empty:
        fig_cb = px.line(opt["callback"], x="iteration", y="welfare", markers=True, title="Ảnh 8.2 — Quá trình hội tụ của SLSQP")
        fig_cb.update_layout(height=420)
        st.plotly_chart(fig_cb, use_container_width=True)

    st.subheader("Câu 8.3.2 — Quỹ đạo tối ưu của K, D, AI, H, Y, C")

    state_long = path.melt(id_vars="year", value_vars=["K", "D", "AI", "H", "A_TFP"], var_name="Biến trạng thái", value_name="Giá trị")
    fig_state = px.line(state_long, x="year", y="Giá trị", color="Biến trạng thái", markers=True, title="Ảnh 8.3 — Quỹ đạo trạng thái K, D, AI, H và TFP")
    fig_state.update_layout(height=520)
    st.plotly_chart(fig_state, use_container_width=True)

    yc_long = path.melt(id_vars="year", value_vars=["Y", "C", "Total_investment"], var_name="Biến", value_name="Giá trị")
    fig_yc = px.line(yc_long, x="year", y="Giá trị", color="Biến", markers=True, title="Ảnh 8.4 — Sản lượng, tiêu dùng và tổng đầu tư")
    fig_yc.update_layout(height=520)
    st.plotly_chart(fig_yc, use_container_width=True)

    inv_share_long = path.melt(id_vars="year", value_vars=["IK_share", "ID_share", "IAI_share", "IH_share"], var_name="Tỷ trọng đầu tư", value_name="Tỷ trọng trong Y")
    fig_inv = px.area(inv_share_long, x="year", y="Tỷ trọng trong Y", color="Tỷ trọng đầu tư", title="Ảnh 8.5 — Cơ cấu tỷ trọng đầu tư tối ưu theo thời gian")
    fig_inv.update_layout(height=520)
    st.plotly_chart(fig_inv, use_container_width=True)

    # -----------------------------------------------------
    # Phân tích nâng cao: Growth accounting và policy cockpit
    # -----------------------------------------------------
    st.subheader("🔎 Phân tích nâng cao — Tăng trưởng đến từ đâu?")

    decomp, decomp_long, avg_decomp = calculate_growth_decomposition(path)

    st.markdown("""
    Để tránh dashboard chỉ dừng ở việc “vẽ đường”, phần này phân rã tăng trưởng theo logic Cobb-Douglas.
    Mục tiêu là trả lời: **quỹ đạo tối ưu đang dựa vào vốn vật chất, số hóa, AI, nhân lực hay TFP?**
    """)

    st.dataframe(avg_decomp.round(5), use_container_width=True)

    fig_decomp = px.bar(decomp_long, x="year", y="Đóng góp log-growth", color="Nguồn đóng góp", title="Ảnh 8.5b — Phân rã đóng góp tăng trưởng theo Cobb-Douglas")
    fig_decomp.update_layout(height=540)
    st.plotly_chart(fig_decomp, use_container_width=True)

    fig_decomp_avg = px.pie(avg_decomp, names="Nguồn đóng góp", values="Tỷ trọng trong tăng trưởng, %", hole=0.42, title="Ảnh 8.5c — Tỷ trọng đóng góp bình quân vào tăng trưởng")
    fig_decomp_avg.update_layout(height=500)
    st.plotly_chart(fig_decomp_avg, use_container_width=True)

    cockpit = investment_policy_matrix(path)
    st.subheader("🎛️ Bảng 8.5 — Policy cockpit của nghiệm tối ưu")
    st.dataframe(cockpit.round(3), use_container_width=True)

    loading = classify_loading_pattern(path)
    if loading["pattern"] == "Front-loaded":
        policy_card("🚀", "Nghiệm có xu hướng front-loaded", "Mô hình đang đề xuất đầu tư mạnh hơn ở đầu kỳ. Điều này hợp lý khi D, AI và H có hiệu ứng tích lũy, tạo tác động lan tỏa đến TFP và sản lượng các năm sau.", "success")
    elif loading["pattern"] == "Back-loaded":
        policy_card("⏳", "Nghiệm có xu hướng back-loaded", "Mô hình đang giữ tiêu dùng hiện tại cao hơn và dồn đầu tư về cuối kỳ. Đây có thể là dấu hiệu của ưu tiên ngắn hạn hoặc ràng buộc khiến đầu tư sớm chưa đủ hấp dẫn.", "warning")
    else:
        policy_card("⚖️", "Nghiệm có xu hướng cân bằng", "Mô hình làm mượt đầu tư theo thời gian, tránh hy sinh tiêu dùng quá mạnh nhưng vẫn duy trì tích lũy vốn.", "info")

    st.subheader("Câu 8.3.3 — Cú sốc năm 2028: Y giảm 8% so với kế hoạch")
    with st.spinner("Đang giải lại mô hình với cú sốc 2028..."):
        shock = solve_dynamic_model(rho=rho, gamma_utility=gamma_utility, shock_year=2028, shock_pct=0.08, strategy_hint=strategy_hint)

    if shock is not None:
        shock_path = shock["path"]
        shock_compare = path[["year", "Y", "C", "Total_investment", "I_K", "I_D", "I_AI", "I_H"]].rename(columns={
            "Y": "Y_base", "C": "C_base", "Total_investment": "Investment_base",
            "I_K": "IK_base", "I_D": "ID_base", "I_AI": "IAI_base", "I_H": "IH_base",
        })
        shock_tmp = shock_path[["year", "Y", "C", "Total_investment", "I_K", "I_D", "I_AI", "I_H"]].rename(columns={
            "Y": "Y_shock", "C": "C_shock", "Total_investment": "Investment_shock",
            "I_K": "IK_shock", "I_D": "ID_shock", "I_AI": "IAI_shock", "I_H": "IH_shock",
        })
        shock_compare = shock_compare.merge(shock_tmp, on="year")
        st.dataframe(shock_compare.round(3), use_container_width=True)

        shock_long = shock_compare.melt(
            id_vars="year",
            value_vars=["Y_base", "Y_shock", "C_base", "C_shock", "Investment_base", "Investment_shock"],
            var_name="Kịch bản",
            value_name="Giá trị",
        )
        fig_shock = px.line(shock_long, x="year", y="Giá trị", color="Kịch bản", markers=True, title="Ảnh 8.6 — So sánh quỹ đạo trước và sau cú sốc 2028")
        fig_shock.update_layout(height=540)
        st.plotly_chart(fig_shock, use_container_width=True)

        st.info("Diễn giải: khi Y năm 2028 giảm 8%, mô hình phải điều chỉnh đồng thời tiêu dùng và đầu tư. Nếu vẫn muốn duy trì tích lũy dài hạn, tiêu dùng ngắn hạn có thể giảm; nếu ưu tiên ổn định tiêu dùng, đầu tư tương lai sẽ bị chậm lại.")

    st.subheader("Câu 8.3.4 — So sánh chiến lược trải đều và front-load")
    even = simulate_fixed_strategy("even", rho=rho, gamma_utility=gamma_utility)
    front = simulate_fixed_strategy("front_load", rho=rho, gamma_utility=gamma_utility)
    back = simulate_fixed_strategy("back_load", rho=rho, gamma_utility=gamma_utility)

    strategy_df = pd.DataFrame([
        {"Chiến lược": "Tối ưu SLSQP", "Welfare": opt["welfare"], "Y_2035": path["Y"].iloc[-1], "C_2035": path["C"].iloc[-1], "K_2035": path["K"].iloc[-1], "AI_2035": path["AI"].iloc[-1], "H_2035": path["H"].iloc[-1]},
        {"Chiến lược": "Đầu tư trải đều", "Welfare": even["welfare"], "Y_2035": even["path"]["Y"].iloc[-1], "C_2035": even["path"]["C"].iloc[-1], "K_2035": even["path"]["K"].iloc[-1], "AI_2035": even["path"]["AI"].iloc[-1], "H_2035": even["path"]["H"].iloc[-1]},
        {"Chiến lược": "Front-load", "Welfare": front["welfare"], "Y_2035": front["path"]["Y"].iloc[-1], "C_2035": front["path"]["C"].iloc[-1], "K_2035": front["path"]["K"].iloc[-1], "AI_2035": front["path"]["AI"].iloc[-1], "H_2035": front["path"]["H"].iloc[-1]},
        {"Chiến lược": "Back-load", "Welfare": back["welfare"], "Y_2035": back["path"]["Y"].iloc[-1], "C_2035": back["path"]["C"].iloc[-1], "K_2035": back["path"]["K"].iloc[-1], "AI_2035": back["path"]["AI"].iloc[-1], "H_2035": back["path"]["H"].iloc[-1]},
    ])

    st.dataframe(strategy_df.round(3), use_container_width=True)

    fig_strategy = px.bar(strategy_df.sort_values("Welfare", ascending=False), x="Chiến lược", y="Welfare", text="Welfare", title="Ảnh 8.7 — So sánh welfare giữa các chiến lược đầu tư")
    fig_strategy.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_strategy.update_layout(height=460)
    st.plotly_chart(fig_strategy, use_container_width=True)

    strategy_paths = []
    for name, obj in [("Tối ưu SLSQP", {"path": path}), ("Trải đều", even), ("Front-load", front), ("Back-load", back)]:
        tmp = obj["path"][["year", "Y", "C", "Total_investment"]].copy()
        tmp["Chiến lược"] = name
        strategy_paths.append(tmp)
    strategy_paths = pd.concat(strategy_paths, ignore_index=True)

    fig_strategy_line = px.line(strategy_paths, x="year", y="Y", color="Chiến lược", markers=True, title="Ảnh 8.8 — So sánh quỹ đạo sản lượng Y giữa các chiến lược")
    fig_strategy_line.update_layout(height=500)
    st.plotly_chart(fig_strategy_line, use_container_width=True)

    best_strategy = strategy_df.sort_values("Welfare", ascending=False).iloc[0]
    st.success(f"Chiến lược có welfare cao nhất trong mô phỏng là **{best_strategy['Chiến lược']}**. Nếu front-load thắng, mô hình hàm ý đầu tư sớm tạo hiệu ứng tích lũy và TFP sớm hơn. Nếu trải đều hoặc tối ưu SLSQP thắng, mô hình hàm ý cần cân bằng giữa tiêu dùng hiện tại và đầu tư dài hạn.")

    return {"opt": opt, "shock": shock, "even": even, "front": front, "back": back, "strategy_df": strategy_df}


# ---------------------------------------------------------
# 7. PHẦN 8.4 — THẢO LUẬN CHÍNH SÁCH
# ---------------------------------------------------------
def show_policy_discussion():
    st.header("8.4. Câu hỏi thảo luận chính sách")

    if not SCIPY_AVAILABLE:
        st.error("Cần cài scipy để chạy phần thảo luận chính sách.")
        return

    add_sticker_header(
        "🧭📊",
        "Từ nghiệm tối ưu đến quyết định chính sách",
        "Phần này không chỉ trả lời đúng câu hỏi, mà còn biến kết quả mô hình thành lập luận chính sách cho Việt Nam.",
    )

    opt_097 = solve_dynamic_model(rho=0.97, gamma_utility=1.0, strategy_hint="balanced")
    opt_090 = solve_dynamic_model(rho=0.90, gamma_utility=1.0, strategy_hint="balanced")

    if opt_097 is None or opt_090 is None:
        st.error("Không thể giải mô hình để thảo luận.")
        return

    path = opt_097["path"]
    path090 = opt_090["path"]

    st.subheader("a) Quỹ đạo tối ưu có front-loaded hay back-loaded không?")
    loading = classify_loading_pattern(path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Đầu tư 2026–2028", f"{loading['early']:,.2f}")
    c2.metric("Đầu tư 2033–2035", f"{loading['late']:,.2f}")
    c3.metric("Mẫu hình", loading["pattern"])

    invest_phase = pd.DataFrame({"Giai đoạn": ["2026–2028", "2029–2032", "2033–2035"], "Tổng đầu tư": [loading["early"], loading["middle"], loading["late"]]})
    fig_phase = px.bar(invest_phase, x="Giai đoạn", y="Tổng đầu tư", text="Tổng đầu tư", title="Minh chứng câu a — Đầu tư theo ba pha phát triển")
    fig_phase.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
    fig_phase.update_layout(height=460)
    st.plotly_chart(fig_phase, use_container_width=True)

    fig_a = px.area(path.melt(id_vars="year", value_vars=["I_K", "I_D", "I_AI", "I_H"], var_name="Loại đầu tư", value_name="Giá trị"), x="year", y="Giá trị", color="Loại đầu tư", title="Minh chứng câu a — Cấu trúc đầu tư theo thời gian")
    fig_a.update_layout(height=520)
    st.plotly_chart(fig_a, use_container_width=True)

    policy_card("🚦", f"Kết luận mô hình: {loading['pattern']}", loading["meaning"], "success" if loading["pattern"] == "Front-loaded" else "warning")

    st.markdown("""
    **Diễn giải sâu:** Nếu mô hình đề xuất front-load, điều đó không đơn thuần là “chi nhiều ở đầu kỳ”.
    Về bản chất, mô hình đang nói rằng **thời điểm đầu tư quan trọng như quy mô đầu tư**.
    Đầu tư vào D, AI và H càng sớm thì càng có nhiều năm để tích lũy năng lực công nghệ, nâng TFP, tăng năng lực hấp thụ AI và tạo hiệu ứng lan tỏa sang sản lượng tương lai.

    Nhưng front-load cũng có mặt trái: tiêu dùng hiện tại bị nén lại, áp lực ngân sách lớn hơn, và nếu năng lực thực thi yếu thì vốn đầu tư sớm có thể không chuyển hóa thành năng suất.
    """)

    st.info("Liên hệ Việt Nam: để đạt mục tiêu 2030 và tầm nhìn 2045, chính sách cần đầu tư sớm cho hạ tầng số, dữ liệu, AI và nhân lực. Tuy nhiên, đầu tư sớm phải đi kèm năng lực giải ngân, quản trị dự án và đánh giá hiệu quả, nếu không front-load có thể biến thành dàn trải.")

    st.subheader("b) Tỷ lệ đầu tư AI/H có ổn định không? Mô hình ngụ ý gì?")
    ratio_df = path[["year", "I_AI", "I_H", "IAI_share", "IH_share"]].copy()
    ratio_df["AI_H_ratio"] = ratio_df["I_AI"] / (ratio_df["I_H"] + 1e-9)
    ratio_df["AI_minus_H_share"] = ratio_df["IAI_share"] - ratio_df["IH_share"]

    ratio_std = ratio_df["AI_H_ratio"].std()
    ratio_mean = ratio_df["AI_H_ratio"].mean()
    stability = "ổn định" if ratio_std < 0.08 else "biến động"

    c4, c5, c6 = st.columns(3)
    c4.metric("AI/H trung bình", f"{ratio_mean:.3f}")
    c5.metric("Độ lệch chuẩn", f"{ratio_std:.3f}")
    c6.metric("Đánh giá", stability)

    fig_b = px.line(ratio_df, x="year", y="AI_H_ratio", markers=True, title="Minh chứng câu b — Tỷ lệ đầu tư AI/H theo thời gian")
    fig_b.update_layout(height=460)
    st.plotly_chart(fig_b, use_container_width=True)

    inv_ai_h = path.melt(id_vars="year", value_vars=["I_AI", "I_H"], var_name="Loại đầu tư", value_name="Giá trị")
    fig_b2 = px.area(inv_ai_h, x="year", y="Giá trị", color="Loại đầu tư", title="AI không thể đi một mình: so sánh đầu tư AI và nhân lực số")
    fig_b2.update_layout(height=520)
    st.plotly_chart(fig_b2, use_container_width=True)

    policy_card("🤖👩‍💻", "Thông điệp AI - Nhân lực", "Nếu tỷ lệ AI/H ổn định, mô hình hàm ý đầu tư AI và đào tạo nhân lực nên đi đồng thời. Nếu AI tăng nhanh hơn H, rủi ro là công nghệ vượt quá năng lực hấp thụ. Nếu H đi trước AI, đó là chiến lược chuẩn bị nền tảng nhân lực để giảm phụ thuộc công nghệ.", "purple")

    st.markdown("""
    **Diễn giải sâu:** AI là vốn công nghệ, còn H là năng lực hấp thụ. Trong mô hình động, đầu tư AI làm tăng năng lực sản xuất nhưng cũng cần người vận hành, dữ liệu sạch, kỹ năng quản trị và năng lực kiểm soát rủi ro. Vì vậy, **AI và H là hai biến bổ trợ**, không nên tách rời.

    Hàm ý chính sách cho Việt Nam: nếu xây trung tâm AI nhưng thiếu kỹ sư AI, chuyên gia dữ liệu và nhân lực quản trị, hiệu quả thực tế sẽ thấp; nếu chỉ đào tạo nhân lực nhưng thiếu hạ tầng tính toán và môi trường ứng dụng, nhân lực khó phát huy. Chiến lược tốt là “AI + H đồng tiến”.
    """)

    st.subheader("c) Nếu ρ giảm từ 0,97 xuống 0,90 thì kết quả thay đổi thế nào?")
    compare_rho = pd.DataFrame([
        {"Kịch bản": "Dài hạn hơn", "ρ": 0.97, "Welfare": opt_097["welfare"], "Y_2035": path["Y"].iloc[-1], "C_2026": path["C"].iloc[0], "C_2035": path["C"].iloc[-1], "Investment_2026_2028": path[path["year"].between(2026, 2028)]["Total_investment"].sum(), "Investment_2033_2035": path[path["year"].between(2033, 2035)]["Total_investment"].sum(), "AI_2035": path["AI"].iloc[-1], "H_2035": path["H"].iloc[-1], "A_2035": path["A_TFP"].iloc[-1]},
        {"Kịch bản": "Ngắn hạn hơn", "ρ": 0.90, "Welfare": opt_090["welfare"], "Y_2035": path090["Y"].iloc[-1], "C_2026": path090["C"].iloc[0], "C_2035": path090["C"].iloc[-1], "Investment_2026_2028": path090[path090["year"].between(2026, 2028)]["Total_investment"].sum(), "Investment_2033_2035": path090[path090["year"].between(2033, 2035)]["Total_investment"].sum(), "AI_2035": path090["AI"].iloc[-1], "H_2035": path090["H"].iloc[-1], "A_2035": path090["A_TFP"].iloc[-1]},
    ])
    compare_rho["Chênh lệch Y_2035 so với ρ=0.97"] = [0, compare_rho.loc[1, "Y_2035"] - compare_rho.loc[0, "Y_2035"]]
    st.dataframe(compare_rho.round(4), use_container_width=True)

    rho_paths = pd.concat([
        path[["year", "C", "Total_investment", "Y", "AI", "H", "A_TFP"]].assign(rho="ρ = 0.97"),
        path090[["year", "C", "Total_investment", "Y", "AI", "H", "A_TFP"]].assign(rho="ρ = 0.90"),
    ])

    fig_c1 = px.line(rho_paths, x="year", y="Total_investment", color="rho", markers=True, title="Minh chứng câu c — Đầu tư thay đổi khi xã hội ngắn hạn hơn")
    fig_c1.update_layout(height=480)
    st.plotly_chart(fig_c1, use_container_width=True)

    fig_c2 = px.line(rho_paths, x="year", y="C", color="rho", markers=True, title="Minh chứng câu c — Tiêu dùng thay đổi khi ρ giảm")
    fig_c2.update_layout(height=480)
    st.plotly_chart(fig_c2, use_container_width=True)

    fig_c3 = px.line(rho_paths, x="year", y="A_TFP", color="rho", markers=True, title="Minh chứng câu c — TFP dài hạn thay đổi theo tầm nhìn chính sách")
    fig_c3.update_layout(height=480)
    st.plotly_chart(fig_c3, use_container_width=True)

    policy_card("⏰", "ρ là tham số của tầm nhìn chính sách", "Khi ρ = 0.97, tương lai được coi trọng hơn, nên mô hình có xu hướng chấp nhận hy sinh tiêu dùng hiện tại để tích lũy năng lực dài hạn. Khi ρ = 0.90, hiện tại được ưu tiên hơn, làm tăng nguy cơ dưới đầu tư vào R&D, AI và nhân lực.", "danger")

    st.markdown("""
    **Vì sao chính phủ có thể dưới đầu tư vào R&D và AI?**

    R&D, AI và nhân lực số thường có ba đặc điểm: chi phí đến ngay nhưng lợi ích đến muộn; lợi ích lan tỏa nên nhà đầu tư không thu hết toàn bộ lợi ích xã hội; rủi ro thực thi cao, đặc biệt với công nghệ mới, dữ liệu, AI và bán dẫn.

    Khi tầm nhìn chính sách ngắn hạn hơn, mô hình sẽ chiết khấu mạnh lợi ích tương lai. Vì vậy, các khoản đầu tư có độ trễ dài như R&D, AI, nhân lực số dễ bị đánh giá thấp. Đây là lý do kinh tế giải thích hiện tượng “dưới đầu tư” vào đổi mới sáng tạo.
    """)

    st.subheader("📌 Kết luận chiến lược cho Việt Nam")
    conclusion_df = pd.DataFrame({
        "Câu hỏi": ["Đầu tư nên sớm hay muộn?", "AI nên đi trước hay đi cùng nhân lực?", "Tầm nhìn dài hạn quan trọng thế nào?", "Đánh đổi chính sách lớn nhất là gì?"],
        "Kết luận từ mô hình": [f"Quỹ đạo hiện tại: {loading['pattern']}.", f"AI/H trung bình = {ratio_mean:.3f}, mức độ {stability}.", "ρ cao làm tăng trọng lượng của lợi ích tương lai.", "Hy sinh một phần tiêu dùng hiện tại để đổi lấy TFP, AI và H trong tương lai."],
        "Hàm ý Việt Nam": ["Cần đầu tư sớm nhưng có chọn lọc, tránh dàn trải.", "Đào tạo nhân lực phải đi cùng trung tâm dữ liệu, AI và hạ tầng số.", "Chính sách 2030–2045 cần cơ chế vượt chu kỳ ngân sách ngắn hạn.", "Cần cân bằng tăng trưởng, ổn định xã hội, năng lực thực thi và đổi mới sáng tạo."],
    })
    st.dataframe(conclusion_df, use_container_width=True)

    st.success("Thông điệp cuối cùng: Bài 8 cho thấy tăng trưởng dựa trên AI không chỉ là bài toán công nghệ, mà là bài toán thời gian. Ai đầu tư đúng thứ và đúng thời điểm sẽ tích lũy được năng lực sản xuất trước.")


# ---------------------------------------------------------
# 8. HÀM RENDER CHÍNH
# ---------------------------------------------------------
def render():
    st.title("⏳ Bài 8 — Tối ưu động phân bổ liên thời gian 2026–2035")

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 12px;
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    Bài 8 mô phỏng chiến lược phân bổ nguồn lực liên thời gian của Việt Nam bằng mô hình tối ưu động.
    Mục tiêu là tìm quỹ đạo tiêu dùng và đầu tư vào **K, D, AI, H** sao cho tổng phúc lợi xã hội 2026–2035 cao nhất.
    """)

    tabs = st.tabs(["8.1 Bối cảnh", "8.2 Mô hình toán học", "8.3 Giải lập trình", "8.4 Chính sách"])

    with tabs[0]:
        show_context()
    with tabs[1]:
        show_math_model()
    with tabs[2]:
        show_programming_solution()
    with tabs[3]:
        show_policy_discussion()

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
# Dynamic Optimization with Cobb-Douglas, SLSQP
# =========================================================


# ---------------------------------------------------------
# 1. THAM SỐ MÔ HÌNH
# ---------------------------------------------------------
def get_model_params():
    params = {
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

        # Initial conditions from Bài 1 / đề bài
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

        # Scale for investment conversion
        # K is large, D/AI/H are index-like, so investments into D/AI/H are scaled.
        "scale_K": 1.0,
        "scale_D": 0.004,
        "scale_AI": 0.006,
        "scale_H": 0.003,
    }

    return params


def utility(C, gamma=1.0):
    C_safe = np.maximum(C, 1e-9)

    if abs(gamma - 1.0) < 1e-9:
        return np.log(C_safe)

    return (C_safe ** (1 - gamma) - 1) / (1 - gamma)


def production(A, K, L, D, AI, H, params):
    return (
        A *
        (K ** params["alpha_K"]) *
        (L ** params["alpha_L"]) *
        (D ** params["alpha_D"]) *
        (AI ** params["alpha_AI"]) *
        (H ** params["alpha_H"])
    )


def unpack_decision(z, T):
    """
    Decision vector z has 5*T variables:
    C_share, IK_share, ID_share, IAI_share, IH_share
    All are shares of annual output Y_t.
    """

    z = np.array(z)
    C_share = z[0:T]
    IK_share = z[T:2*T]
    ID_share = z[2*T:3*T]
    IAI_share = z[3*T:4*T]
    IH_share = z[4*T:5*T]

    return C_share, IK_share, ID_share, IAI_share, IH_share


def simulate_path(
    z,
    params=None,
    rho=None,
    gamma_utility=1.0,
    shock_year=None,
    shock_pct=0.0,
):
    if params is None:
        params = get_model_params()

    T = params["T"]
    years = params["years"]

    if rho is None:
        rho = params["rho"]

    C_share, IK_share, ID_share, IAI_share, IH_share = unpack_decision(z, T)

    K = np.zeros(T + 1)
    D = np.zeros(T + 1)
    AI = np.zeros(T + 1)
    H = np.zeros(T + 1)
    A = np.zeros(T + 1)
    L = np.zeros(T)

    Y = np.zeros(T)
    Y_plan = np.zeros(T)
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

    feasible = True
    penalties = []

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

        # Budget check
        if C[t] + total_I[t] > Y[t] + 1e-6:
            feasible = False
            penalties.append((C[t] + total_I[t] - Y[t]) ** 2)

        if C[t] <= 0:
            feasible = False
            penalties.append(1e6)

        # Dynamics
        K[t + 1] = (1 - params["delta_K"]) * K[t] + params["scale_K"] * I_K[t]
        D[t + 1] = (1 - params["delta_D"]) * D[t] + params["scale_D"] * I_D[t]
        AI[t + 1] = (1 - params["delta_AI"]) * AI[t] + params["scale_AI"] * I_AI[t]
        H[t + 1] = H[t] + params["theta_H"] * params["scale_H"] * I_H[t] - params["mu_brain_drain"] * H[t]

        A[t + 1] = A[t] * (
            1 +
            params["phi_D"] * D[t] / 100 +
            params["phi_AI"] * AI[t] / 100 +
            params["phi_H"] * H[t] / 100
        )

        welfare_terms[t] = (rho ** t) * utility(C[t], gamma_utility)

        # State positivity
        if min(K[t + 1], D[t + 1], AI[t + 1], H[t + 1], A[t + 1]) <= 0:
            feasible = False
            penalties.append(1e6)

    welfare = welfare_terms.sum()

    result = pd.DataFrame({
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

    penalty_value = np.sum(penalties) if len(penalties) > 0 else 0.0

    return {
        "path": result,
        "terminal": terminal,
        "welfare": welfare,
        "feasible": feasible,
        "penalty": penalty_value,
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

    # Negative welfare because scipy minimizes.
    return -sim["welfare"] + 1e6 * sim["penalty"]


def build_constraints(params):
    T = params["T"]

    constraints = []

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

    constraints.append({"type": "ineq", "fun": budget_share_constraint})
    constraints.append({"type": "ineq", "fun": max_investment_constraint})
    constraints.append({"type": "ineq", "fun": min_consumption_constraint})
    constraints.append({"type": "ineq", "fun": min_h_constraint})
    constraints.append({"type": "ineq", "fun": min_dai_constraint})

    return constraints


@st.cache_data(show_spinner=False)
def solve_dynamic_model(
    rho=0.97,
    gamma_utility=1.0,
    shock_year=None,
    shock_pct=0.0,
    strategy_hint="balanced",
):
    params = get_model_params()
    T = params["T"]

    if strategy_hint == "front_load":
        t = np.arange(T)
        inv_base = 0.37 - 0.012 * t
        inv_base = np.clip(inv_base, 0.23, 0.40)
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

    # Consumption share upper can be 1
    upper[0:T] = 0.95

    # Specific lower bounds
    lower[0:T] = params["min_consumption_share"]
    lower[4*T:5*T] = params["min_H_investment_share"]

    bounds = list(zip(lower, upper))

    constraints = build_constraints(params)

    callback_history = []

    def callback(xk):
        sim = simulate_path(xk, params=params, rho=rho, gamma_utility=gamma_utility)
        callback_history.append({
            "iteration": len(callback_history) + 1,
            "welfare": sim["welfare"],
            "penalty": sim["penalty"],
        })

    if not SCIPY_AVAILABLE:
        return None

    res = minimize(
        objective_slsqp,
        z0,
        args=(params, rho, gamma_utility, shock_year, shock_pct),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={
            "maxiter": 400,
            "ftol": 1e-7,
            "disp": False,
        }
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

    if strategy == "even":
        inv = np.ones(T) * 0.34
    elif strategy == "front_load":
        t = np.arange(T)
        inv = 0.42 - 0.018 * t
        inv = np.clip(inv, 0.22, 0.42)
    elif strategy == "back_load":
        t = np.arange(T)
        inv = 0.25 + 0.014 * t
        inv = np.clip(inv, 0.25, 0.40)
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
# 2. PHẦN 8.1 — BỐI CẢNH
# ---------------------------------------------------------
def show_context():
    st.header("8.1. Bối cảnh Việt Nam")

    st.markdown("""
    Bài 8 chuyển từ các bài toán phân bổ tĩnh sang **tối ưu hóa động liên thời gian**.
    Vấn đề chính sách không còn là “năm nay đầu tư bao nhiêu”, mà là **đầu tư theo quỹ đạo nào trong 10 năm**
    để vừa duy trì tiêu dùng hiện tại, vừa tích lũy vốn cho tăng trưởng dài hạn.
    """)

    st.markdown("""
    Trong bối cảnh Việt Nam đặt mục tiêu trở thành nước thu nhập trung bình cao vào năm 2030 và nước phát triển
    thu nhập cao vào năm 2045, mô hình này giúp minh họa đánh đổi giữa: **tiêu dùng hiện tại**, **đầu tư vật chất**,
    **hạ tầng số**, **AI** và **nhân lực số**. Đây là đánh đổi trung tâm của chiến lược phát triển dài hạn.
    """)

    timeline = pd.DataFrame({
        "Mốc": ["2026", "2030", "2035", "2045"],
        "Ý nghĩa chính sách": [
            "Khởi đầu quỹ đạo đầu tư số sau giai đoạn 2020–2025.",
            "Mục tiêu nước thu nhập trung bình cao; kinh tế số và AI phải tạo năng suất rõ hơn.",
            "Điểm cuối mô phỏng Bài 8; đánh giá tích lũy K, D, AI, H.",
            "Tầm nhìn nước phát triển thu nhập cao; yêu cầu nền tảng nhân lực và công nghệ bền vững.",
        ],
        "Trọng tâm mô hình": [
            "Điều kiện ban đầu K, D, AI, H, A.",
            "Hiệu ứng tích lũy và độ trễ đầu tư.",
            "So sánh chiến lược tối ưu, front-load, trải đều.",
            "Diễn giải chính sách dài hạn.",
        ]
    })

    st.dataframe(timeline, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giai đoạn", "2026–2035")
    c2.metric("Trạng thái", "K, D, AI, H, A")
    c3.metric("Điều khiển", "I_K, I_D, I_AI, I_H, C")
    c4.metric("Phương pháp", "SLSQP")

    st.subheader("Ảnh 8.1 — Logic tối ưu động trong chiến lược phát triển Việt Nam")

    labels = [
        "Sản lượng Y_t",
        "Tiêu dùng C_t",
        "Đầu tư K",
        "Đầu tư D",
        "Đầu tư AI",
        "Đầu tư H",
        "Tích lũy vốn",
        "TFP A_t",
        "Sản lượng Y_{t+1}",
        "Phúc lợi liên thời gian",
    ]

    fig_flow = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.3),
            label=labels,
        ),
        link=dict(
            source=[0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8],
            target=[1, 2, 3, 4, 5, 6, 6, 6, 6, 8, 8, 9],
            value=[45, 15, 8, 7, 10, 15, 8, 7, 10, 25, 20, 45],
        )
    )])

    fig_flow.update_layout(
        title="Từ sản lượng hiện tại đến tích lũy vốn, TFP và phúc lợi tương lai",
        height=520
    )

    st.plotly_chart(fig_flow, use_container_width=True)

    st.info(
        "Thông điệp bối cảnh: đầu tư AI và số hóa có thể làm tăng năng suất dài hạn, nhưng nếu đầu tư quá nhiều ngay hiện tại, "
        "tiêu dùng ngắn hạn bị giảm. Tối ưu động giúp định lượng đánh đổi này."
    )


# ---------------------------------------------------------
# 3. PHẦN 8.2 — MÔ HÌNH TOÁN HỌC
# ---------------------------------------------------------
def show_math_model():
    st.header("8.2. Mô hình toán học")

    st.markdown("""
    Mô hình tối ưu động gồm ba khối: **hàm mục tiêu phúc lợi**, **hàm sản xuất Cobb-Douglas mở rộng**,
    và **phương trình chuyển trạng thái** của K, D, AI, H, A.
    """)

    st.subheader("Bước 1 — Hàm mục tiêu phúc lợi liên thời gian")

    st.latex(r"""
    \max \sum_{t=2026}^{2035} \rho^{t-2026} U(C_t)
    """)

    st.latex(r"""
    U(C_t)=\ln(C_t)
    """)

    st.latex(r"""
    U(C_t)=\frac{C_t^{1-\gamma}-1}{1-\gamma},\quad \gamma=1.5
    """)

    st.markdown("""
    `C_t` là tiêu dùng tại năm `t`.  
    `ρ` là hệ số chiết khấu. Nếu `ρ` cao, mô hình coi trọng tương lai hơn.  
    Hàm log nhấn mạnh tiêu dùng ổn định; hàm CRRA cho phép mô phỏng mức độ ác cảm rủi ro cao hơn.
    """)

    st.subheader("Bước 2 — Hàm sản xuất Cobb-Douglas mở rộng")

    st.latex(r"""
    Y_t =
    A_t K_t^{0.33} L_t^{0.42} D_t^{0.10} AI_t^{0.08} H_t^{0.07}
    """)

    st.markdown("""
    Sản lượng phụ thuộc vào vốn vật chất, lao động, hạ tầng số, năng lực AI và vốn nhân lực.
    Các số mũ là độ co giãn sản lượng theo từng yếu tố.
    """)

    st.subheader("Bước 3 — Động học tích lũy vốn")

    st.latex(r"""
    K_{t+1}=(1-\delta_K)K_t+I_{K,t}
    """)

    st.latex(r"""
    D_{t+1}=(1-\delta_D)D_t+I_{D,t}
    """)

    st.latex(r"""
    AI_{t+1}=(1-\delta_{AI})AI_t+I_{AI,t}
    """)

    st.latex(r"""
    H_{t+1}=H_t+\theta_H I_{H,t}-\mu H_t
    """)

    st.markdown("""
    Vốn vật chất, số hóa và AI bị khấu hao theo thời gian.  
    Nhân lực số tăng nhờ đầu tư đào tạo nhưng giảm do chảy máu chất xám hoặc hao hụt kỹ năng.
    """)

    st.subheader("Bước 4 — Cơ chế TFP nội sinh")

    st.latex(r"""
    A_{t+1}=A_t(1+\phi_1D_t+\phi_2AI_t+\phi_3H_t)
    """)

    st.markdown("""
    D, AI và H không chỉ tác động trực tiếp đến sản lượng mà còn làm tăng TFP. Đây là cơ chế lan tỏa dài hạn
    của chuyển đổi số, AI và nhân lực chất lượng cao.
    """)

    st.subheader("Bước 5 — Ràng buộc nguồn lực hàng năm")

    st.latex(r"""
    C_t+I_{K,t}+I_{D,t}+I_{AI,t}+I_{H,t}\leq Y_t
    """)

    st.markdown("""
    Tổng tiêu dùng và đầu tư không được vượt sản lượng năm đó. Đây là ràng buộc ngân sách vĩ mô cơ bản.
    """)

    params = get_model_params()

    param_df = pd.DataFrame({
        "Nhóm": [
            "Khấu hao", "Khấu hao", "Khấu hao",
            "Nhân lực", "Nhân lực",
            "TFP", "TFP", "TFP",
            "Chiết khấu"
        ],
        "Ký hiệu": [
            "δ_K", "δ_D", "δ_AI",
            "θ_H", "μ",
            "φ_1", "φ_2", "φ_3",
            "ρ"
        ],
        "Giá trị": [
            params["delta_K"], params["delta_D"], params["delta_AI"],
            params["theta_H"], params["mu_brain_drain"],
            params["phi_D"], params["phi_AI"], params["phi_H"],
            params["rho"]
        ],
        "Ý nghĩa": [
            "Khấu hao vốn vật chất",
            "Khấu hao hạ tầng số",
            "Khấu hao vốn AI",
            "Hiệu quả chuyển đầu tư H thành vốn nhân lực",
            "Suy giảm nhân lực do brain drain/hao hụt kỹ năng",
            "Tác động D đến TFP",
            "Tác động AI đến TFP",
            "Tác động H đến TFP",
            "Mức coi trọng tương lai"
        ]
    })

    st.subheader("Bảng tham số mô hình")

    st.dataframe(param_df, use_container_width=True)

    initial_df = pd.DataFrame({
        "Biến trạng thái ban đầu 2026": ["K0", "L0", "D0", "AI0", "H0", "A0"],
        "Giá trị": [
            params["K0"],
            params["L0"],
            params["D0"],
            params["AI0"],
            params["H0"],
            params["A0"],
        ],
        "Đơn vị": [
            "nghìn tỷ VND",
            "triệu lao động",
            "% GDP / chỉ số",
            "nghìn DN số",
            "% lao động có kỹ năng",
            "chỉ số TFP",
        ]
    })

    st.subheader("Điều kiện ban đầu")

    st.dataframe(initial_df, use_container_width=True)

    st.success(
        "Tư duy mô hình: đầu tư hiện tại làm giảm tiêu dùng hiện tại nhưng tăng K, D, AI, H và A trong tương lai. "
        "Vì vậy nghiệm tối ưu phụ thuộc mạnh vào hệ số chiết khấu ρ và hiệu quả tích lũy của từng loại vốn."
    )


# ---------------------------------------------------------
# 4. PHẦN 8.3 — GIẢI LẬP TRÌNH
# ---------------------------------------------------------
def show_programming_solution():
    st.header("8.3. Giải yêu cầu lập trình")

    st.markdown("""
    Vì hàm Cobb-Douglas và động học TFP tạo bài toán phi tuyến, module này dùng **scipy.optimize.minimize**
    với thuật toán **SLSQP**. Biến quyết định là tỷ trọng tiêu dùng và tỷ trọng đầu tư vào K, D, AI, H trong từng năm.
    """)

    if not SCIPY_AVAILABLE:
        st.error("Chưa cài scipy. Hãy thêm `scipy` vào requirements.txt.")
        return None

    st.subheader("Thiết lập tham số mô phỏng")

    c1, c2, c3 = st.columns(3)

    rho = c1.slider(
        "ρ - hệ số chiết khấu",
        min_value=0.85,
        max_value=0.99,
        value=0.97,
        step=0.01,
        key="bai8_rho"
    )

    utility_type = c2.selectbox(
        "Hàm thỏa dụng",
        ["Log utility", "CRRA gamma = 1.5"],
        key="bai8_utility"
    )

    gamma_utility = 1.0 if utility_type == "Log utility" else 1.5

    strategy_hint = c3.selectbox(
        "Gợi ý nghiệm khởi tạo",
        ["balanced", "front_load", "even"],
        format_func=lambda x: {
            "balanced": "Cân bằng",
            "front_load": "Front-load",
            "even": "Trải đều"
        }[x],
        key="bai8_strategy_hint"
    )

    # -----------------------------------------------------
    # Câu 8.3.1
    # -----------------------------------------------------
    st.subheader("Câu 8.3.1 — Giải tối ưu động bằng SLSQP")

    with st.spinner("Đang giải tối ưu động bằng SLSQP..."):
        opt = solve_dynamic_model(
            rho=rho,
            gamma_utility=gamma_utility,
            shock_year=None,
            shock_pct=0.0,
            strategy_hint=strategy_hint,
        )

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
        fig_cb = px.line(
            opt["callback"],
            x="iteration",
            y="welfare",
            markers=True,
            title="Ảnh 8.2 — Quá trình hội tụ của SLSQP"
        )
        fig_cb.update_layout(height=420)
        st.plotly_chart(fig_cb, use_container_width=True)

    # -----------------------------------------------------
    # Câu 8.3.2
    # -----------------------------------------------------
    st.subheader("Câu 8.3.2 — Quỹ đạo tối ưu của K, D, AI, H, Y, C")

    state_long = path.melt(
        id_vars="year",
        value_vars=["K", "D", "AI", "H", "A_TFP"],
        var_name="Biến trạng thái",
        value_name="Giá trị"
    )

    fig_state = px.line(
        state_long,
        x="year",
        y="Giá trị",
        color="Biến trạng thái",
        markers=True,
        title="Ảnh 8.3 — Quỹ đạo trạng thái K, D, AI, H và TFP"
    )
    fig_state.update_layout(height=520)
    st.plotly_chart(fig_state, use_container_width=True)

    yc_long = path.melt(
        id_vars="year",
        value_vars=["Y", "C", "Total_investment"],
        var_name="Biến",
        value_name="Giá trị"
    )

    fig_yc = px.line(
        yc_long,
        x="year",
        y="Giá trị",
        color="Biến",
        markers=True,
        title="Ảnh 8.4 — Sản lượng, tiêu dùng và tổng đầu tư"
    )
    fig_yc.update_layout(height=520)
    st.plotly_chart(fig_yc, use_container_width=True)

    inv_share_long = path.melt(
        id_vars="year",
        value_vars=["IK_share", "ID_share", "IAI_share", "IH_share"],
        var_name="Tỷ trọng đầu tư",
        value_name="Tỷ trọng trong Y"
    )

    fig_inv = px.area(
        inv_share_long,
        x="year",
        y="Tỷ trọng trong Y",
        color="Tỷ trọng đầu tư",
        title="Ảnh 8.5 — Cơ cấu tỷ trọng đầu tư tối ưu theo thời gian"
    )
    fig_inv.update_layout(height=520)
    st.plotly_chart(fig_inv, use_container_width=True)

    # -----------------------------------------------------
    # Câu 8.3.3
    # -----------------------------------------------------
    st.subheader("Câu 8.3.3 — Cú sốc năm 2028: Y giảm 8% so với kế hoạch")

    with st.spinner("Đang giải lại mô hình với cú sốc 2028..."):
        shock = solve_dynamic_model(
            rho=rho,
            gamma_utility=gamma_utility,
            shock_year=2028,
            shock_pct=0.08,
            strategy_hint=strategy_hint,
        )

    if shock is not None:
        shock_path = shock["path"]

        shock_compare = path[["year", "Y", "C", "Total_investment", "I_K", "I_D", "I_AI", "I_H"]].copy()
        shock_compare = shock_compare.rename(columns={
            "Y": "Y_base",
            "C": "C_base",
            "Total_investment": "Investment_base",
            "I_K": "IK_base",
            "I_D": "ID_base",
            "I_AI": "IAI_base",
            "I_H": "IH_base",
        })

        shock_tmp = shock_path[["year", "Y", "C", "Total_investment", "I_K", "I_D", "I_AI", "I_H"]].copy()
        shock_tmp = shock_tmp.rename(columns={
            "Y": "Y_shock",
            "C": "C_shock",
            "Total_investment": "Investment_shock",
            "I_K": "IK_shock",
            "I_D": "ID_shock",
            "I_AI": "IAI_shock",
            "I_H": "IH_shock",
        })

        shock_compare = shock_compare.merge(shock_tmp, on="year")

        st.dataframe(shock_compare.round(3), use_container_width=True)

        shock_long = shock_compare.melt(
            id_vars="year",
            value_vars=["Y_base", "Y_shock", "C_base", "C_shock", "Investment_base", "Investment_shock"],
            var_name="Kịch bản",
            value_name="Giá trị"
        )

        fig_shock = px.line(
            shock_long,
            x="year",
            y="Giá trị",
            color="Kịch bản",
            markers=True,
            title="Ảnh 8.6 — So sánh quỹ đạo trước và sau cú sốc 2028"
        )
        fig_shock.update_layout(height=540)
        st.plotly_chart(fig_shock, use_container_width=True)

        st.info(
            "Diễn giải: khi Y năm 2028 giảm 8%, mô hình phải điều chỉnh đồng thời tiêu dùng và đầu tư. "
            "Nếu vẫn muốn duy trì tích lũy dài hạn, tiêu dùng ngắn hạn có thể giảm; nếu ưu tiên ổn định tiêu dùng, đầu tư tương lai sẽ bị chậm lại."
        )

    # -----------------------------------------------------
    # Câu 8.3.4
    # -----------------------------------------------------
    st.subheader("Câu 8.3.4 — So sánh chiến lược trải đều và front-load")

    even = simulate_fixed_strategy("even", rho=rho, gamma_utility=gamma_utility)
    front = simulate_fixed_strategy("front_load", rho=rho, gamma_utility=gamma_utility)
    back = simulate_fixed_strategy("back_load", rho=rho, gamma_utility=gamma_utility)

    strategy_df = pd.DataFrame([
        {
            "Chiến lược": "Tối ưu SLSQP",
            "Welfare": opt["welfare"],
            "Y_2035": path["Y"].iloc[-1],
            "C_2035": path["C"].iloc[-1],
            "K_2035": path["K"].iloc[-1],
            "AI_2035": path["AI"].iloc[-1],
            "H_2035": path["H"].iloc[-1],
        },
        {
            "Chiến lược": "Đầu tư trải đều",
            "Welfare": even["welfare"],
            "Y_2035": even["path"]["Y"].iloc[-1],
            "C_2035": even["path"]["C"].iloc[-1],
            "K_2035": even["path"]["K"].iloc[-1],
            "AI_2035": even["path"]["AI"].iloc[-1],
            "H_2035": even["path"]["H"].iloc[-1],
        },
        {
            "Chiến lược": "Front-load",
            "Welfare": front["welfare"],
            "Y_2035": front["path"]["Y"].iloc[-1],
            "C_2035": front["path"]["C"].iloc[-1],
            "K_2035": front["path"]["K"].iloc[-1],
            "AI_2035": front["path"]["AI"].iloc[-1],
            "H_2035": front["path"]["H"].iloc[-1],
        },
        {
            "Chiến lược": "Back-load",
            "Welfare": back["welfare"],
            "Y_2035": back["path"]["Y"].iloc[-1],
            "C_2035": back["path"]["C"].iloc[-1],
            "K_2035": back["path"]["K"].iloc[-1],
            "AI_2035": back["path"]["AI"].iloc[-1],
            "H_2035": back["path"]["H"].iloc[-1],
        },
    ])

    st.dataframe(strategy_df.round(3), use_container_width=True)

    fig_strategy = px.bar(
        strategy_df.sort_values("Welfare", ascending=False),
        x="Chiến lược",
        y="Welfare",
        text="Welfare",
        title="Ảnh 8.7 — So sánh welfare giữa các chiến lược đầu tư"
    )
    fig_strategy.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_strategy.update_layout(height=460)
    st.plotly_chart(fig_strategy, use_container_width=True)

    strategy_paths = []
    for name, obj in [
        ("Tối ưu SLSQP", {"path": path}),
        ("Trải đều", even),
        ("Front-load", front),
        ("Back-load", back),
    ]:
        tmp = obj["path"][["year", "Y", "C", "Total_investment"]].copy()
        tmp["Chiến lược"] = name
        strategy_paths.append(tmp)

    strategy_paths = pd.concat(strategy_paths, ignore_index=True)

    fig_strategy_line = px.line(
        strategy_paths,
        x="year",
        y="Y",
        color="Chiến lược",
        markers=True,
        title="Ảnh 8.8 — So sánh quỹ đạo sản lượng Y giữa các chiến lược"
    )
    fig_strategy_line.update_layout(height=500)
    st.plotly_chart(fig_strategy_line, use_container_width=True)

    best_strategy = strategy_df.sort_values("Welfare", ascending=False).iloc[0]

    st.success(
        f"Chiến lược có welfare cao nhất trong mô phỏng là **{best_strategy['Chiến lược']}**. "
        "Nếu front-load thắng, mô hình hàm ý đầu tư sớm tạo hiệu ứng tích lũy và TFP sớm hơn. "
        "Nếu trải đều hoặc tối ưu SLSQP thắng, mô hình hàm ý cần cân bằng giữa tiêu dùng hiện tại và đầu tư dài hạn."
    )

    return {
        "opt": opt,
        "shock": shock,
        "even": even,
        "front": front,
        "back": back,
        "strategy_df": strategy_df,
    }


# ---------------------------------------------------------
# 5. PHẦN 8.4 — THẢO LUẬN CHÍNH SÁCH
# ---------------------------------------------------------
def show_policy_discussion():
    st.header("8.4. Câu hỏi thảo luận chính sách")

    if not SCIPY_AVAILABLE:
        st.error("Cần cài scipy để chạy phần thảo luận chính sách.")
        return

    opt_097 = solve_dynamic_model(rho=0.97, gamma_utility=1.0, strategy_hint="balanced")
    opt_090 = solve_dynamic_model(rho=0.90, gamma_utility=1.0, strategy_hint="balanced")

    if opt_097 is None or opt_090 is None:
        st.error("Không thể giải mô hình để thảo luận.")
        return

    path = opt_097["path"]
    path090 = opt_090["path"]

    # -----------------------------------------------------
    # Câu a
    # -----------------------------------------------------
    st.subheader("a) Quỹ đạo tối ưu có front-loaded hay back-loaded không?")

    invest_early = path[path["year"].between(2026, 2028)]["Total_investment"].sum()
    invest_late = path[path["year"].between(2033, 2035)]["Total_investment"].sum()

    pattern = "front-loaded" if invest_early > invest_late else "back-loaded"

    c1, c2, c3 = st.columns(3)
    c1.metric("Đầu tư 2026–2028", f"{invest_early:,.2f}")
    c2.metric("Đầu tư 2033–2035", f"{invest_late:,.2f}")
    c3.metric("Kết luận", pattern)

    fig_a = px.bar(
        path,
        x="year",
        y="Total_investment",
        text="Total_investment",
        title="Minh chứng câu a — Tổng đầu tư theo năm"
    )
    fig_a.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_a.update_layout(height=480)
    st.plotly_chart(fig_a, use_container_width=True)

    st.success(
        f"Trả lời: quỹ đạo hiện tại có xu hướng **{pattern}**. "
        "Nếu đầu tư tập trung sớm, nguyên nhân là K, D, AI, H cần thời gian tích lũy và tạo tác động lan tỏa qua TFP. "
        "Nếu đầu tư dồn về cuối, mô hình đang ưu tiên tiêu dùng hiện tại hoặc ràng buộc làm giảm khả năng đầu tư sớm."
    )

    st.info(
        "Liên hệ Việt Nam: với mục tiêu 2030 và 2045, đầu tư vào hạ tầng số, AI và nhân lực thường cần đi trước "
        "để tạo năng lực sản xuất mới, nhưng front-load quá mạnh có thể làm giảm tiêu dùng và áp lực ngân sách ngắn hạn."
    )

    # -----------------------------------------------------
    # Câu b
    # -----------------------------------------------------
    st.subheader("b) Tỷ lệ đầu tư AI/đầu tư H có ổn định không?")

    ratio_df = path[["year", "I_AI", "I_H"]].copy()
    ratio_df["AI_H_ratio"] = ratio_df["I_AI"] / (ratio_df["I_H"] + 1e-9)

    ratio_std = ratio_df["AI_H_ratio"].std()
    ratio_mean = ratio_df["AI_H_ratio"].mean()

    c4, c5 = st.columns(2)
    c4.metric("AI/H trung bình", f"{ratio_mean:.3f}")
    c5.metric("Độ lệch chuẩn AI/H", f"{ratio_std:.3f}")

    fig_b = px.line(
        ratio_df,
        x="year",
        y="AI_H_ratio",
        markers=True,
        title="Minh chứng câu b — Tỷ lệ đầu tư AI/H theo thời gian"
    )
    fig_b.update_layout(height=460)
    st.plotly_chart(fig_b, use_container_width=True)

    inv_ai_h = path.melt(
        id_vars="year",
        value_vars=["I_AI", "I_H"],
        var_name="Loại đầu tư",
        value_name="Giá trị"
    )

    fig_b2 = px.area(
        inv_ai_h,
        x="year",
        y="Giá trị",
        color="Loại đầu tư",
        title="So sánh đầu tư AI và đầu tư nhân lực số H"
    )
    fig_b2.update_layout(height=500)
    st.plotly_chart(fig_b2, use_container_width=True)

    st.success(
        "Trả lời: nếu tỷ lệ AI/H ổn định, mô hình hàm ý AI và nhân lực nên đi đồng thời. "
        "Nếu AI/H cao ở đầu kỳ, mô hình ưu tiên mở rộng năng lực công nghệ trước; nếu H đi trước, mô hình nhấn mạnh hấp thụ công nghệ và giảm rủi ro thiếu kỹ sư AI."
    )

    st.info(
        "Hàm ý chính sách: Việt Nam không nên chỉ đầu tư thiết bị, trung tâm dữ liệu hoặc thuật toán AI mà thiếu đào tạo nhân lực. "
        "Nhân lực số là điều kiện để hấp thụ công nghệ, vận hành AI an toàn và tránh phụ thuộc bên ngoài."
    )

    # -----------------------------------------------------
    # Câu c
    # -----------------------------------------------------
    st.subheader("c) Nếu ρ giảm từ 0,97 xuống 0,90 thì kết quả thay đổi thế nào?")

    compare_rho = pd.DataFrame([
        {
            "ρ": 0.97,
            "Welfare": opt_097["welfare"],
            "Y_2035": path["Y"].iloc[-1],
            "C_2026": path["C"].iloc[0],
            "C_2035": path["C"].iloc[-1],
            "Investment_2026_2028": path[path["year"].between(2026, 2028)]["Total_investment"].sum(),
            "Investment_2033_2035": path[path["year"].between(2033, 2035)]["Total_investment"].sum(),
            "AI_2035": path["AI"].iloc[-1],
            "H_2035": path["H"].iloc[-1],
        },
        {
            "ρ": 0.90,
            "Welfare": opt_090["welfare"],
            "Y_2035": path090["Y"].iloc[-1],
            "C_2026": path090["C"].iloc[0],
            "C_2035": path090["C"].iloc[-1],
            "Investment_2026_2028": path090[path090["year"].between(2026, 2028)]["Total_investment"].sum(),
            "Investment_2033_2035": path090[path090["year"].between(2033, 2035)]["Total_investment"].sum(),
            "AI_2035": path090["AI"].iloc[-1],
            "H_2035": path090["H"].iloc[-1],
        }
    ])

    st.dataframe(compare_rho.round(3), use_container_width=True)

    rho_paths = pd.concat([
        path[["year", "C", "Total_investment", "Y", "AI", "H"]].assign(rho="ρ = 0.97"),
        path090[["year", "C", "Total_investment", "Y", "AI", "H"]].assign(rho="ρ = 0.90"),
    ])

    fig_c = px.line(
        rho_paths,
        x="year",
        y="Total_investment",
        color="rho",
        markers=True,
        title="Minh chứng câu c — Đầu tư thay đổi khi hệ số chiết khấu giảm"
    )
    fig_c.update_layout(height=480)
    st.plotly_chart(fig_c, use_container_width=True)

    fig_c2 = px.line(
        rho_paths,
        x="year",
        y="C",
        color="rho",
        markers=True,
        title="Tiêu dùng thay đổi khi ρ giảm"
    )
    fig_c2.update_layout(height=480)
    st.plotly_chart(fig_c2, use_container_width=True)

    st.warning(
        "Trả lời: khi ρ giảm từ 0,97 xuống 0,90, mô hình coi trọng hiện tại hơn tương lai. "
        "Do đó, xu hướng thường là tiêu dùng hiện tại tăng tương đối và đầu tư dài hạn giảm tương đối. "
        "Đây là một lý do kinh tế giải thích vì sao chính phủ hoặc nhà hoạch định chính sách có thể dưới đầu tư vào R&D, AI và nhân lực: "
        "lợi ích của các khoản đầu tư này đến muộn, trong khi chi phí ngân sách phát sinh ngay."
    )

    st.markdown("""
    **Kết luận chính sách của Bài 8:**  
    Tối ưu động cho thấy phát triển AI và kinh tế số không thể chỉ nhìn theo từng năm ngân sách.
    Việt Nam cần một quỹ đạo đầu tư dài hạn, trong đó tiêu dùng hiện tại, tích lũy vốn, hạ tầng số, AI và nhân lực được cân bằng.
    Nếu tầm nhìn chính sách ngắn hạn, nền kinh tế dễ dưới đầu tư vào các yếu tố tạo năng suất dài hạn như R&D, AI và nhân lực số.
    """)


# ---------------------------------------------------------
# 6. HÀM RENDER CHÍNH
# ---------------------------------------------------------
def render():
    st.title("⏳ Bài 8 — Tối ưu động phân bổ liên thời gian 2026–2035")

    st.markdown("""
    Bài 8 mô phỏng chiến lược phân bổ nguồn lực liên thời gian của Việt Nam bằng mô hình tối ưu động.
    Mục tiêu là tìm quỹ đạo tiêu dùng và đầu tư vào **K, D, AI, H** sao cho tổng phúc lợi xã hội 2026–2035 cao nhất.
    """)

    tabs = st.tabs([
        "8.1 Bối cảnh",
        "8.2 Mô hình toán học",
        "8.3 Giải lập trình",
        "8.4 Chính sách",
    ])

    with tabs[0]:
        show_context()

    with tabs[1]:
        show_math_model()

    with tabs[2]:
        show_programming_solution()

    with tabs[3]:
        show_policy_discussion()

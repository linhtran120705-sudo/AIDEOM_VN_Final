import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


try:
    from ai_agent import render_ai_agent
    AI_AGENT_AVAILABLE = True
except Exception:
    render_ai_agent = None
    AI_AGENT_AVAILABLE = False


# =========================================================
# BÀI 7 — TỐI ƯU ĐA MỤC TIÊU PARETO VỚI NSGA-II
# =========================================================


# ---------------------------------------------------------
# 1. DỮ LIỆU NỀN TỪ BÀI 4 + THAM SỐ BÀI 7
# ---------------------------------------------------------
def get_base_data():
    regions = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]

    region_names = {
        "NMM": "Trung du miền núi phía Bắc",
        "RRD": "Đồng bằng sông Hồng",
        "NCC": "Bắc Trung Bộ + DH Trung Bộ",
        "CH": "Tây Nguyên",
        "SE": "Đông Nam Bộ",
        "MD": "Đồng bằng sông Cửu Long",
    }

    items = ["I", "D", "AI", "H"]

    item_names = {
        "I": "Hạ tầng số",
        "D": "Dữ liệu/CĐS doanh nghiệp",
        "AI": "Trí tuệ nhân tạo",
        "H": "Nhân lực số",
    }

    beta = np.array([
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ])

    e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    sigma = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

    param_df = pd.DataFrame({
        "Mã vùng": regions,
        "Vùng": [region_names[r] for r in regions],
        "eᵣ - CO₂/tỷ": e,
        "ρᵣ - rủi ro/AI": rho,
        "σᵣ - giảm rủi ro/H": sigma,
    })

    beta_df = pd.DataFrame(
        beta,
        index=[region_names[r] for r in regions],
        columns=[item_names[j] for j in items]
    )

    return regions, region_names, items, item_names, beta, e, rho, sigma, param_df, beta_df


def approximate_inequality(region_budget):
    """
    f2: xấp xỉ bất bình đẳng phân bổ vùng bằng MAD chuẩn hóa.
    Giá trị càng thấp càng bao trùm.
    """

    mean_budget = np.mean(region_budget)
    if mean_budget <= 1e-9:
        return 0.0

    mad = np.mean(np.abs(region_budget - mean_budget))
    return mad / mean_budget


def compute_objectives_from_matrix(X, beta, e, rho, sigma):
    """
    X shape = (6, 4), thứ tự cột: I, D, AI, H.
    Trả về:
    growth_gain: cần maximize.
    inequality: cần minimize.
    emission: cần minimize.
    security_risk: cần minimize.
    """

    growth_gain = float((beta * X).sum())

    region_budget = X.sum(axis=1)
    inequality = float(approximate_inequality(region_budget))

    emission = float((e * (X[:, 0] + X[:, 2])).sum())

    security_risk = float((rho * X[:, 2]).sum() - (sigma * X[:, 3]).sum())

    return growth_gain, inequality, emission, security_risk


def normalize_for_topsis(values, benefit_flags):
    arr = values.astype(float).copy()
    norm = np.zeros_like(arr)

    for j in range(arr.shape[1]):
        col = arr[:, j]
        cmin, cmax = col.min(), col.max()

        if abs(cmax - cmin) < 1e-12:
            norm[:, j] = 1.0
        else:
            if benefit_flags[j]:
                norm[:, j] = (col - cmin) / (cmax - cmin)
            else:
                norm[:, j] = (cmax - col) / (cmax - cmin)

    return norm


def topsis_on_pareto(df, weights):
    """
    Dùng TOPSIS để chọn nghiệm thỏa hiệp.
    Tiêu chí:
    - growth_gain: benefit
    - inequality: cost
    - emission: cost
    - security_risk: cost
    """

    criteria = ["growth_gain", "inequality", "emission", "security_risk"]
    benefit_flags = np.array([True, False, False, False])

    X = df[criteria].values.astype(float)
    X_norm = normalize_for_topsis(X, benefit_flags)

    w = np.array(weights, dtype=float)
    w = w / w.sum()

    V = X_norm * w

    A_star = V.max(axis=0)
    A_neg = V.min(axis=0)

    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    score = S_neg / (S_star + S_neg + 1e-12)

    out = df.copy()
    out["topsis_score"] = score
    out["topsis_rank"] = out["topsis_score"].rank(ascending=False, method="dense").astype(int)

    return out.sort_values("topsis_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------
# 2. ĐỊNH NGHĨA PROBLEM CHO PYMOO
# ---------------------------------------------------------
if PYMOO_AVAILABLE:
    class VietnamDigitalParetoProblem(ElementwiseProblem):
        def __init__(
            self,
            total_budget=50000,
            min_region=5000,
            max_region=12000,
            min_h_total=12000,
            min_d_total=8000,
            max_item_per_region=12000,
        ):
            self.regions, self.region_names, self.items, self.item_names, self.beta, self.e, self.rho, self.sigma, _, _ = get_base_data()

            self.total_budget = total_budget
            self.min_region = min_region
            self.max_region = max_region
            self.min_h_total = min_h_total
            self.min_d_total = min_d_total
            self.max_item_per_region = max_item_per_region

            # 24 biến: 6 vùng × 4 hạng mục
            # 4 mục tiêu
            # Ràng buộc:
            # C1 ngân sách tổng
            # C2 sàn vùng 6
            # C3 trần vùng 6
            # C4 sàn H toàn quốc
            # C5 sàn D toàn quốc
            super().__init__(
                n_var=24,
                n_obj=4,
                n_ieq_constr=15,
                xl=np.zeros(24),
                xu=np.ones(24) * max_item_per_region
            )

        def _evaluate(self, x, out, *args, **kwargs):
            X = x.reshape(6, 4)

            growth_gain, inequality, emission, security_risk = compute_objectives_from_matrix(
                X, self.beta, self.e, self.rho, self.sigma
            )

            # pymoo mặc định minimize.
            # f1 tăng trưởng cần maximize nên đổi dấu.
            f1 = -growth_gain
            f2 = inequality
            f3 = emission
            f4 = security_risk

            region_budget = X.sum(axis=1)

            G = []

            # C1: tổng ngân sách <= total_budget
            G.append(X.sum() - self.total_budget)

            # C2: mỗi vùng >= min_region => min_region - sum <= 0
            for r in range(6):
                G.append(self.min_region - region_budget[r])

            # C3: mỗi vùng <= max_region
            for r in range(6):
                G.append(region_budget[r] - self.max_region)

            # C4: tổng H >= min_h_total
            G.append(self.min_h_total - X[:, 3].sum())

            # C5: tổng D >= min_d_total
            G.append(self.min_d_total - X[:, 1].sum())

            out["F"] = [f1, f2, f3, f4]
            out["G"] = G


# ---------------------------------------------------------
# 3. CHẠY NSGA-II
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_nsga2(total_budget, min_region, max_region, min_h_total, min_d_total, pop_size, n_gen, seed):
    if not PYMOO_AVAILABLE:
        return None, None

    problem = VietnamDigitalParetoProblem(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        min_d_total=min_d_total,
        max_item_per_region=max_region,
    )

    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False,
        save_history=False,
    )

    X = res.X
    F = res.F

    if X is None or F is None:
        return pd.DataFrame(), pd.DataFrame()

    records = []

    regions, region_names, items, item_names, beta, e, rho, sigma, param_df, beta_df = get_base_data()

    for idx in range(len(X)):
        mat = X[idx].reshape(6, 4)

        growth_gain, inequality, emission, security_risk = compute_objectives_from_matrix(
            mat, beta, e, rho, sigma
        )

        records.append({
            "solution_id": idx,
            "growth_gain": growth_gain,
            "inequality": inequality,
            "emission": emission,
            "security_risk": security_risk,
            "total_budget_used": mat.sum(),
            "H_total": mat[:, 3].sum(),
            "D_total": mat[:, 1].sum(),
            "I_total": mat[:, 0].sum(),
            "AI_total": mat[:, 2].sum(),
        })

    pareto_df = pd.DataFrame(records)

    alloc_rows = []
    for idx in range(len(X)):
        mat = X[idx].reshape(6, 4)
        for r_idx, r in enumerate(regions):
            for j_idx, j in enumerate(items):
                alloc_rows.append({
                    "solution_id": idx,
                    "Mã vùng": r,
                    "Vùng": region_names[r],
                    "Mã hạng mục": j,
                    "Hạng mục": item_names[j],
                    "Ngân sách, tỷ VND": mat[r_idx, j_idx],
                })

    allocation_df = pd.DataFrame(alloc_rows)

    return pareto_df, allocation_df


# ---------------------------------------------------------
# 4. BỐI CẢNH
# ---------------------------------------------------------
def show_context():
    st.header("7.1. Bối cảnh Việt Nam")

    st.markdown("""
    Trong hoạch định chính sách, Việt Nam không chỉ cần một nghiệm tối ưu duy nhất.
    Khi phát triển kinh tế số và AI, bốn mục tiêu thường xung đột với nhau:
    **tăng trưởng nhanh**, **bao trùm xã hội**, **môi trường xanh**, và **an ninh dữ liệu**.
    """)

    st.markdown("""
    Bài 7 sử dụng **NSGA-II** để tìm tập nghiệm Pareto. Mỗi nghiệm là một phương án phân bổ ngân sách số
    cho 6 vùng và 4 hạng mục đầu tư. Không nghiệm nào trong tập Pareto có thể cải thiện một mục tiêu
    mà không làm xấu đi ít nhất một mục tiêu khác.
    """)

    st.markdown("""
    Liên hệ thực tiễn Việt Nam: định hướng chuyển đổi số theo QĐ 749/QĐ-TTg, chiến lược AI theo QĐ 127/QĐ-TTg,
    tinh thần khoa học - công nghệ và đổi mới sáng tạo theo Nghị quyết 57-NQ/TW, cùng cam kết Net Zero 2050 tại COP26
    đều cho thấy chính sách không thể chỉ tối đa hóa GDP, mà phải cân bằng nhiều mục tiêu.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Biến quyết định", "24", "6 vùng × 4 hạng mục")
    c2.metric("Mục tiêu", "4", "tăng trưởng, bao trùm, CO₂, an ninh")
    c3.metric("Phương pháp", "NSGA-II")
    c4.metric("Đầu ra", "Pareto set")

    policy_df = pd.DataFrame({
        "Mục tiêu chính sách": [
            "Tăng trưởng GDP",
            "Bao trùm vùng miền",
            "Môi trường - Net Zero",
            "An ninh dữ liệu",
        ],
        "Ý nghĩa trong mô hình": [
            "Tối đa hóa GDP gain kỳ vọng từ đầu tư số.",
            "Giảm bất bình đẳng phân bổ ngân sách giữa các vùng.",
            "Giảm phát thải gián tiếp từ hạ tầng số và AI.",
            "Giảm rủi ro dữ liệu ròng bằng kết hợp AI và nhân lực số.",
        ],
        "Liên hệ Việt Nam": [
            "Tăng trưởng kinh tế số, năng suất và năng lực cạnh tranh.",
            "Không để vùng yếu bị bỏ lại trong chuyển đổi số.",
            "Cam kết Net Zero 2050 tại COP26.",
            "Chủ quyền số, an toàn dữ liệu và năng lực quản trị AI.",
        ],
    })

    st.dataframe(policy_df, use_container_width=True)

    st.subheader("Ảnh 7.1 — Bản đồ đánh đổi chính sách trong kỷ nguyên AI")

    labels = [
        "Ngân sách kinh tế số",
        "Tăng trưởng GDP",
        "Bao trùm xã hội",
        "Môi trường",
        "An ninh dữ liệu",
        "Tập nghiệm Pareto",
        "Lựa chọn chính sách cuối cùng",
    ]

    fig_flow = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.3),
            label=labels,
        ),
        link=dict(
            source=[0, 0, 0, 0, 1, 2, 3, 4, 5],
            target=[1, 2, 3, 4, 5, 5, 5, 5, 6],
            value=[25, 20, 15, 15, 15, 15, 15, 15, 25],
        )
    )])
    fig_flow.update_layout(
        title="Từ ngân sách số đến tập nghiệm Pareto và lựa chọn chính sách",
        height=520,
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    st.info(
        "Thông điệp bối cảnh: NSGA-II không thay nhà hoạch định chính sách ra quyết định. "
        "Nó tạo ra bản đồ các phương án khả thi để thấy rõ phải đánh đổi gì nếu ưu tiên tăng trưởng, bao trùm, môi trường hoặc an ninh."
    )


# ---------------------------------------------------------
# 5. MÔ HÌNH TOÁN HỌC
# ---------------------------------------------------------
def show_math_model():
    st.header("7.2. Mô hình toán học đa mục tiêu")

    st.markdown("""
    Biến quyết định là mức ngân sách phân bổ cho từng hạng mục đầu tư `j` tại từng vùng `r`.
    Mỗi nghiệm là một ma trận 6×4.
    """)

    st.subheader("Bước 1 — Biến quyết định")

    st.latex(r"""
    x_{j,r} \geq 0,\quad
    j \in \{I,D,AI,H\},\quad r \in \{1,\ldots,6\}
    """)

    st.markdown("""
    `I`: hạ tầng số; `D`: dữ liệu/chuyển đổi số doanh nghiệp; `AI`: trí tuệ nhân tạo; `H`: nhân lực số.
    Tổng cộng có `6 × 4 = 24` biến quyết định.
    """)

    st.subheader("Bước 2 — Mục tiêu 1: tối đa hóa tăng trưởng GDP")

    st.latex(r"""
    \max f_1(x) = \sum_r \sum_j \beta_{j,r}x_{j,r}
    """)

    st.markdown("""
    `βⱼ,ᵣ` là hệ số tác động biên. Mục tiêu này ưu tiên hiệu quả kinh tế và tăng trưởng nhanh.
    """)

    st.subheader("Bước 3 — Mục tiêu 2: giảm bất bình đẳng phân bổ vùng")

    st.latex(r"""
    \min f_2(x) =
    \frac{1}{\bar{B}}
    \cdot
    \frac{1}{R}
    \sum_r |B_r - \bar{B}|
    """)

    st.markdown("""
    `Bᵣ = Σⱼxⱼ,ᵣ` là tổng ngân sách vùng `r`.  
    `f₂` là độ lệch tuyệt đối trung bình chuẩn hóa. Giá trị càng thấp nghĩa là phân bổ càng bao trùm.
    """)

    st.subheader("Bước 4 — Mục tiêu 3: giảm phát thải gián tiếp")

    st.latex(r"""
    \min f_3(x) = \sum_r e_r(x_{I,r}+x_{AI,r})
    """)

    st.markdown("""
    `eᵣ` là cường độ phát thải của vùng. Hạ tầng số và AI được giả định tiêu thụ nhiều năng lượng hơn,
    nên tạo phát thải gián tiếp lớn hơn.
    """)

    st.subheader("Bước 5 — Mục tiêu 4: giảm rủi ro an ninh dữ liệu ròng")

    st.latex(r"""
    \min f_4(x) =
    \sum_r \rho_r x_{AI,r}
    -
    \sum_r \sigma_r x_{H,r}
    """)

    st.markdown("""
    `ρᵣ` phản ánh rủi ro dữ liệu phát sinh khi mở rộng AI.  
    `σᵣ` phản ánh khả năng giảm rủi ro nhờ đầu tư nhân lực số, quản trị dữ liệu và an ninh mạng.
    """)

    st.subheader("Bước 6 — Dạng tổng quát")

    st.latex(r"""
    \max f_1(x),\quad
    \min f_2(x),\quad
    \min f_3(x),\quad
    \min f_4(x)
    """)

    constraint_df = pd.DataFrame({
        "Nhóm ràng buộc": [
            "Ngân sách tổng",
            "Sàn ngân sách mỗi vùng",
            "Trần ngân sách mỗi vùng",
            "Sàn nhân lực số",
            "Sàn dữ liệu/chuyển đổi số",
            "Không âm",
        ],
        "Dạng mô hình": [
            "ΣᵣΣⱼxⱼ,ᵣ ≤ B",
            "Σⱼxⱼ,ᵣ ≥ Lᵣ",
            "Σⱼxⱼ,ᵣ ≤ Uᵣ",
            "Σᵣxᴴ,ᵣ ≥ H_min",
            "Σᵣxᴰ,ᵣ ≥ D_min",
            "xⱼ,ᵣ ≥ 0",
        ],
        "Ý nghĩa": [
            "Không vượt ngân sách số quốc gia.",
            "Không để vùng yếu bị bỏ lại.",
            "Tránh tập trung quá mức vào một vùng.",
            "Đảm bảo năng lực hấp thụ và quản trị AI.",
            "Đảm bảo nền tảng dữ liệu cho chuyển đổi số.",
            "Không thể phân bổ âm.",
        ]
    })

    st.dataframe(constraint_df, use_container_width=True)

    st.success(
        "Điểm cốt lõi: bài toán này không có một nghiệm tối ưu tuyệt đối. Kết quả đúng là một tập nghiệm Pareto "
        "để nhà hoạch định chính sách chọn nghiệm thỏa hiệp."
    )


# ---------------------------------------------------------
# 6. THAM SỐ
# ---------------------------------------------------------
def show_parameters():
    st.header("7.3. Bảng tham số bổ sung")

    regions, region_names, items, item_names, beta, e, rho, sigma, param_df, beta_df = get_base_data()

    st.markdown("""
    Bảng dưới đây bổ sung ba nhóm tham số cho mô hình đa mục tiêu:
    `eᵣ` cho phát thải, `ρᵣ` cho rủi ro AI, và `σᵣ` cho khả năng giảm rủi ro nhờ nhân lực số.
    """)

    st.dataframe(param_df, use_container_width=True)

    st.subheader("Ảnh 7.2 — Heatmap hệ số β tăng trưởng từ Bài 4")

    fig_beta = px.imshow(
        beta_df,
        text_auto=".2f",
        aspect="auto",
        title="Hệ số tác động biên βⱼ,ᵣ theo vùng và hạng mục"
    )
    fig_beta.update_layout(height=540)
    st.plotly_chart(fig_beta, use_container_width=True)

    st.subheader("Ảnh 7.3 — Tham số môi trường và an ninh dữ liệu theo vùng")

    param_long = param_df.melt(
        id_vars=["Mã vùng", "Vùng"],
        value_vars=["eᵣ - CO₂/tỷ", "ρᵣ - rủi ro/AI", "σᵣ - giảm rủi ro/H"],
        var_name="Tham số",
        value_name="Giá trị"
    )

    fig_param = px.bar(
        param_long,
        x="Vùng",
        y="Giá trị",
        color="Tham số",
        barmode="group",
        title="So sánh eᵣ, ρᵣ, σᵣ giữa các vùng"
    )
    fig_param.update_layout(height=520)
    st.plotly_chart(fig_param, use_container_width=True)

    st.subheader("Ảnh 7.4 — Quan hệ giữa rủi ro AI và khả năng giảm rủi ro bằng nhân lực")

    fig_security = px.scatter(
        param_df,
        x="ρᵣ - rủi ro/AI",
        y="σᵣ - giảm rủi ro/H",
        size="eᵣ - CO₂/tỷ",
        color="Vùng",
        hover_name="Vùng",
        title="Vùng nào có rủi ro AI cao và cần đầu tư nhân lực quản trị số?"
    )
    fig_security.update_layout(height=500)
    st.plotly_chart(fig_security, use_container_width=True)

    st.info(
        "Cách đọc: Đông Nam Bộ có rủi ro AI cao hơn, nên nếu đầu tư mạnh AI cần đi kèm nhân lực số và quản trị dữ liệu. "
        "Tây Nguyên có cường độ phát thải thấp hơn nhưng năng lực AI thấp, phù hợp hơn với đầu tư nền tảng và bao trùm."
    )


# ---------------------------------------------------------
# 7. GIẢI LẬP TRÌNH
# ---------------------------------------------------------
def show_programming_solution():
    st.header("7.4. Giải yêu cầu lập trình")

    st.markdown("""
    Phần này chạy NSGA-II để tạo tập nghiệm Pareto. Mỗi nghiệm là một phương án phân bổ ngân sách số.
    Sau đó dùng TOPSIS để chọn một nghiệm thỏa hiệp theo trọng số chính sách.
    """)

    st.subheader("Thiết lập tham số NSGA-II và ràng buộc")

    c1, c2, c3, c4 = st.columns(4)

    total_budget = c1.number_input(
        "Ngân sách tổng, tỷ VND",
        min_value=30000,
        max_value=100000,
        value=50000,
        step=5000,
        key="bai7_total_budget"
    )

    min_region = c2.number_input(
        "Sàn mỗi vùng, tỷ VND",
        min_value=0,
        max_value=10000,
        value=5000,
        step=1000,
        key="bai7_min_region"
    )

    max_region = c3.number_input(
        "Trần mỗi vùng, tỷ VND",
        min_value=8000,
        max_value=30000,
        value=12000,
        step=1000,
        key="bai7_max_region"
    )

    min_h_total = c4.number_input(
        "Sàn H toàn quốc, tỷ VND",
        min_value=0,
        max_value=30000,
        value=12000,
        step=1000,
        key="bai7_min_h"
    )

    c5, c6, c7 = st.columns(3)

    min_d_total = c5.number_input(
        "Sàn D toàn quốc, tỷ VND",
        min_value=0,
        max_value=30000,
        value=8000,
        step=1000,
        key="bai7_min_d"
    )

    pop_size = c6.slider(
        "pop_size",
        min_value=40,
        max_value=200,
        value=100,
        step=20,
        key="bai7_pop_size"
    )

    n_gen = c7.slider(
        "n_gen",
        min_value=50,
        max_value=300,
        value=200,
        step=50,
        key="bai7_n_gen"
    )

    seed = st.number_input(
        "Seed tái lập kết quả",
        min_value=1,
        max_value=9999,
        value=42,
        step=1,
        key="bai7_seed"
    )

    if not PYMOO_AVAILABLE:
        st.error(
            "Chưa cài `pymoo`. Hãy thêm `pymoo` vào requirements.txt để chạy NSGA-II."
        )
        return None

    if min_region * 6 > total_budget:
        st.error("Mô hình không khả thi: sàn mỗi vùng × 6 lớn hơn ngân sách tổng.")
        return None

    if min_h_total + min_d_total > total_budget:
        st.warning(
            "Tổng sàn H và D khá lớn so với ngân sách. Mô hình vẫn có thể chạy nhưng không gian khả thi bị thu hẹp."
        )

    with st.spinner("Đang chạy NSGA-II. Lần đầu có thể mất vài chục giây..."):
        pareto_df, allocation_df = run_nsga2(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_h_total=min_h_total,
            min_d_total=min_d_total,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
        )

    if pareto_df is None or pareto_df.empty:
        st.error("NSGA-II chưa tìm được nghiệm khả thi. Hãy nới ràng buộc hoặc tăng số thế hệ.")
        return None

    # -----------------------------------------------------
    # 7.4.1 Kết quả Pareto
    # -----------------------------------------------------
    st.subheader("Câu 7.4.1 — Cài đặt Problem và chạy NSGA-II")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số nghiệm Pareto", f"{len(pareto_df)}")
    m2.metric("Tăng trưởng max", f"{pareto_df['growth_gain'].max():,.0f}", "tỷ VND")
    m3.metric("Bao trùm tốt nhất", f"{pareto_df['inequality'].min():.3f}")
    m4.metric("CO₂ thấp nhất", f"{pareto_df['emission'].min():,.0f}")

    st.dataframe(pareto_df.round(3), use_container_width=True)

    # -----------------------------------------------------
    # 7.4.2 Biểu đồ Pareto
    # -----------------------------------------------------
    st.subheader("Câu 7.4.2 — Trích xuất tập Pareto và trực quan hóa")

    fig_3d = px.scatter_3d(
        pareto_df,
        x="growth_gain",
        y="inequality",
        z="emission",
        color="security_risk",
        size="AI_total",
        hover_data=["solution_id", "total_budget_used", "H_total", "D_total"],
        title="Ảnh 7.5 — Đường biên Pareto 3D: tăng trưởng - bao trùm - phát thải"
    )
    fig_3d.update_layout(height=720)
    st.plotly_chart(fig_3d, use_container_width=True)

    parallel_df = pareto_df[[
        "growth_gain", "inequality", "emission", "security_risk",
        "I_total", "D_total", "AI_total", "H_total"
    ]].copy()

    fig_parallel = px.parallel_coordinates(
        parallel_df,
        dimensions=[
            "growth_gain", "inequality", "emission", "security_risk",
            "I_total", "D_total", "AI_total", "H_total"
        ],
        color="growth_gain",
        title="Ảnh 7.6 — Parallel coordinates của tập nghiệm Pareto"
    )
    fig_parallel.update_layout(height=620)
    st.plotly_chart(fig_parallel, use_container_width=True)

    # -----------------------------------------------------
    # 7.4.3 TOPSIS chọn nghiệm thỏa hiệp
    # -----------------------------------------------------
    st.subheader("Câu 7.4.3 — Chọn nghiệm thỏa hiệp bằng TOPSIS")

    st.markdown("""
    Trọng số chính sách mặc định:
    **0,40 tăng trưởng; 0,25 bao trùm; 0,20 môi trường; 0,15 an ninh dữ liệu**.
    """)

    w1, w2, w3, w4 = st.columns(4)

    growth_w = w1.slider("w tăng trưởng", 0.05, 0.70, 0.40, 0.05, key="bai7_w_growth")
    inclusion_w = w2.slider("w bao trùm", 0.05, 0.60, 0.25, 0.05, key="bai7_w_inclusion")
    env_w = w3.slider("w môi trường", 0.05, 0.60, 0.20, 0.05, key="bai7_w_env")
    security_w = w4.slider("w an ninh", 0.05, 0.60, 0.15, 0.05, key="bai7_w_security")

    weights = np.array([growth_w, inclusion_w, env_w, security_w])
    weights = weights / weights.sum()

    ranked = topsis_on_pareto(pareto_df, weights)
    compromise = ranked.iloc[0]

    c8, c9, c10, c11 = st.columns(4)
    c8.metric("Nghiệm thỏa hiệp", f"#{int(compromise['solution_id'])}")
    c9.metric("TOPSIS score", f"{compromise['topsis_score']:.3f}")
    c10.metric("Growth gain", f"{compromise['growth_gain']:,.0f}")
    c11.metric("Inequality", f"{compromise['inequality']:.3f}")

    st.dataframe(ranked.head(10).round(3), use_container_width=True)

    comp_alloc = allocation_df[allocation_df["solution_id"] == compromise["solution_id"]].copy()

    alloc_matrix = comp_alloc.pivot(
        index="Vùng",
        columns="Hạng mục",
        values="Ngân sách, tỷ VND"
    )

    st.markdown("#### Ma trận phân bổ của nghiệm thỏa hiệp")

    st.dataframe(alloc_matrix.round(1), use_container_width=True)

    fig_alloc = px.imshow(
        alloc_matrix,
        text_auto=".0f",
        aspect="auto",
        title="Ảnh 7.7 — Heatmap phân bổ ngân sách của nghiệm thỏa hiệp TOPSIS"
    )
    fig_alloc.update_layout(height=560)
    st.plotly_chart(fig_alloc, use_container_width=True)

    # -----------------------------------------------------
    # 7.4.4 Chi phí cơ hội
    # -----------------------------------------------------
    st.subheader("Câu 7.4.4 — Chi phí cơ hội của nghiệm tăng trưởng cao nhất")

    max_growth = pareto_df.sort_values("growth_gain", ascending=False).iloc[0]

    opportunity = pd.DataFrame([
        {
            "Nghiệm": "Tăng trưởng cao nhất",
            "solution_id": int(max_growth["solution_id"]),
            "growth_gain": max_growth["growth_gain"],
            "inequality": max_growth["inequality"],
            "emission": max_growth["emission"],
            "security_risk": max_growth["security_risk"],
        },
        {
            "Nghiệm": "Thỏa hiệp TOPSIS",
            "solution_id": int(compromise["solution_id"]),
            "growth_gain": compromise["growth_gain"],
            "inequality": compromise["inequality"],
            "emission": compromise["emission"],
            "security_risk": compromise["security_risk"],
        },
    ])

    st.dataframe(opportunity.round(3), use_container_width=True)

    growth_premium = (max_growth["growth_gain"] / compromise["growth_gain"] - 1) * 100
    inclusion_cost = (max_growth["inequality"] / compromise["inequality"] - 1) * 100 if compromise["inequality"] > 0 else np.nan
    env_cost = (max_growth["emission"] / compromise["emission"] - 1) * 100 if compromise["emission"] > 0 else np.nan
    security_cost = (
        (max_growth["security_risk"] - compromise["security_risk"]) /
        (abs(compromise["security_risk"]) + 1e-9) * 100
    )

    c12, c13, c14, c15 = st.columns(4)
    c12.metric("Tăng trưởng cao hơn", f"{growth_premium:.2f}%")
    c13.metric("Bao trùm xấu hơn", f"{inclusion_cost:.2f}%")
    c14.metric("Phát thải cao hơn", f"{env_cost:.2f}%")
    c15.metric("Rủi ro an ninh đổi", f"{security_cost:.2f}%")

    fig_opp = px.bar(
        opportunity.melt(
            id_vars=["Nghiệm", "solution_id"],
            value_vars=["growth_gain", "inequality", "emission", "security_risk"],
            var_name="Mục tiêu",
            value_name="Giá trị"
        ),
        x="Mục tiêu",
        y="Giá trị",
        color="Nghiệm",
        barmode="group",
        title="Ảnh 7.8 — So sánh nghiệm tăng trưởng cao nhất và nghiệm thỏa hiệp"
    )
    fig_opp.update_layout(height=520)
    st.plotly_chart(fig_opp, use_container_width=True)

    st.success(
        f"Nghiệm tăng trưởng cao nhất tăng GDP gain thêm khoảng **{growth_premium:.2f}%** so với nghiệm thỏa hiệp, "
        f"nhưng làm chỉ số bất bình đẳng phân bổ xấu hơn khoảng **{inclusion_cost:.2f}%** "
        f"và phát thải cao hơn khoảng **{env_cost:.2f}%**. "
        "Đây chính là chi phí cơ hội của việc ưu tiên tăng trưởng tuyệt đối."
    )

    return {
        "pareto_df": pareto_df,
        "allocation_df": allocation_df,
        "ranked": ranked,
        "compromise": compromise,
        "max_growth": max_growth,
        "opportunity": opportunity,
    }


# ---------------------------------------------------------
# 8. THẢO LUẬN CHÍNH SÁCH
# ---------------------------------------------------------
def show_policy_discussion():
    st.header("7.5. Câu hỏi thảo luận chính sách")

    if not PYMOO_AVAILABLE:
        st.error("Cần cài `pymoo` để chạy phần thảo luận chính sách.")
        return

    pareto_df, allocation_df = run_nsga2(
        total_budget=50000,
        min_region=5000,
        max_region=12000,
        min_h_total=12000,
        min_d_total=8000,
        pop_size=100,
        n_gen=200,
        seed=42,
    )

    if pareto_df is None or pareto_df.empty:
        st.error("Chưa có tập nghiệm Pareto để thảo luận.")
        return

    ranked = topsis_on_pareto(pareto_df, [0.40, 0.25, 0.20, 0.15])
    compromise = ranked.iloc[0]
    max_growth = pareto_df.sort_values("growth_gain", ascending=False).iloc[0]

    # -----------------------------------------------------
    # Câu a
    # -----------------------------------------------------
    st.subheader("a) Đánh đổi giữa tăng trưởng và bao trùm có rõ không?")

    corr_growth_inclusion = pareto_df["growth_gain"].corr(pareto_df["inequality"])

    c1, c2 = st.columns(2)
    c1.metric("Tương quan growth - inequality", f"{corr_growth_inclusion:.3f}")
    c2.metric(
        "Diễn giải",
        "Đánh đổi rõ" if corr_growth_inclusion > 0.3 else "Đánh đổi yếu/trung bình"
    )

    fig_a = px.scatter(
        pareto_df,
        x="growth_gain",
        y="inequality",
        color="emission",
        size="AI_total",
        hover_data=["solution_id", "security_risk"],
        title="Minh chứng câu a — Đánh đổi giữa tăng trưởng và bao trùm"
    )
    fig_a.update_layout(
        xaxis_title="GDP gain, tỷ VND",
        yaxis_title="Bất bình đẳng phân bổ vùng, thấp hơn là tốt",
        height=520
    )
    st.plotly_chart(fig_a, use_container_width=True)

    st.success(
        f"Tương quan giữa tăng trưởng và bất bình đẳng là **{corr_growth_inclusion:.3f}**. "
        "Nếu tương quan dương, các nghiệm tăng trưởng cao thường đi kèm phân bổ kém bao trùm hơn. "
        "Điều này phản ánh thực tế Việt Nam: vốn số và AI có xu hướng sinh lợi cao hơn ở vùng đã có nền tảng tốt, "
        "nhưng nếu chỉ chạy theo hiệu quả ngắn hạn, khoảng cách số vùng miền có thể tăng."
    )

    # -----------------------------------------------------
    # Câu b
    # -----------------------------------------------------
    st.subheader("b) Trọng số 0,40; 0,25; 0,20; 0,15 có phù hợp không?")

    weight_scenarios = pd.DataFrame({
        "Kịch bản": [
            "Mặc định",
            "COP26 xanh hóa",
            "AI-centric",
            "Bao trùm vùng",
        ],
        "w_growth": [0.40, 0.30, 0.35, 0.30],
        "w_inclusion": [0.25, 0.20, 0.15, 0.40],
        "w_environment": [0.20, 0.35, 0.15, 0.15],
        "w_security": [0.15, 0.15, 0.35, 0.15],
    })

    scenario_rows = []

    for _, row in weight_scenarios.iterrows():
        w = [row["w_growth"], row["w_inclusion"], row["w_environment"], row["w_security"]]
        rank_tmp = topsis_on_pareto(pareto_df, w)
        best = rank_tmp.iloc[0]

        scenario_rows.append({
            "Kịch bản": row["Kịch bản"],
            "solution_id": int(best["solution_id"]),
            "TOPSIS score": best["topsis_score"],
            "growth_gain": best["growth_gain"],
            "inequality": best["inequality"],
            "emission": best["emission"],
            "security_risk": best["security_risk"],
        })

    scenario_result = pd.DataFrame(scenario_rows)

    st.dataframe(weight_scenarios, use_container_width=True)
    st.dataframe(scenario_result.round(3), use_container_width=True)

    fig_b = px.parallel_coordinates(
        scenario_result,
        dimensions=["growth_gain", "inequality", "emission", "security_risk"],
        color="growth_gain",
        title="Minh chứng câu b — Nghiệm thỏa hiệp thay đổi theo ưu tiên chính sách"
    )
    fig_b.update_layout(height=560)
    st.plotly_chart(fig_b, use_container_width=True)

    st.info(
        "Trọng số mặc định 0,40 cho tăng trưởng phản ánh ưu tiên phát triển kinh tế và năng suất. "
        "Tuy nhiên, nếu nhấn mạnh cam kết COP26, nên tăng trọng số môi trường lên khoảng 0,30–0,35. "
        "Nếu bám sát Quyết định 127/QĐ-TTg về AI, có thể tăng trọng số an ninh dữ liệu và năng lực AI, "
        "vì trung tâm AI quốc gia đòi hỏi quản trị dữ liệu, hạ tầng tính toán và nhân lực chất lượng cao."
    )

    # -----------------------------------------------------
    # Câu c
    # -----------------------------------------------------
    st.subheader("c) NSGA-II khác gì LP đơn mục tiêu? Có thay thế quyết định chính trị không?")

    comparison_df = pd.DataFrame({
        "Tiêu chí so sánh": [
            "Đầu ra",
            "Cách nhìn chính sách",
            "Xử lý mục tiêu xung đột",
            "Vai trò của người ra quyết định",
            "Ứng dụng phù hợp",
        ],
        "LP đơn mục tiêu": [
            "Một nghiệm tối ưu duy nhất.",
            "Tối đa/min hóa một mục tiêu chính.",
            "Các mục tiêu khác phải chuyển thành ràng buộc hoặc trọng số.",
            "Chọn trước mục tiêu và chấp nhận nghiệm cuối cùng.",
            "Phân bổ ngân sách khi ưu tiên rõ ràng.",
        ],
        "NSGA-II đa mục tiêu": [
            "Một tập nghiệm Pareto.",
            "Cho thấy bản đồ đánh đổi giữa nhiều mục tiêu.",
            "Giữ đồng thời nhiều mục tiêu xung đột.",
            "Chọn nghiệm thỏa hiệp sau khi thấy đánh đổi.",
            "Chính sách công có tăng trưởng, công bằng, môi trường, an ninh cùng lúc.",
        ],
    })

    st.dataframe(comparison_df, use_container_width=True)

    fig_c = px.scatter_3d(
        pareto_df,
        x="growth_gain",
        y="inequality",
        z="emission",
        color="security_risk",
        title="Minh chứng câu c — NSGA-II tạo không gian lựa chọn thay vì một nghiệm duy nhất"
    )
    fig_c.update_layout(height=680)
    st.plotly_chart(fig_c, use_container_width=True)

    st.success(
        "NSGA-II khác LP đơn mục tiêu ở chỗ nó không ép chính sách vào một hàm mục tiêu duy nhất. "
        "Nó cho thấy nhiều phương án Pareto, mỗi phương án có cấu trúc đánh đổi khác nhau."
    )

    st.warning(
        "NSGA-II không thay thế quyết định chính trị. Nó chỉ cung cấp bằng chứng định lượng. "
        "Lựa chọn cuối cùng vẫn phải xét ưu tiên quốc gia, công bằng vùng miền, an sinh xã hội, cam kết khí hậu, "
        "an ninh dữ liệu và tính khả thi thể chế."
    )

    st.markdown("""
    **Kết luận chính sách của Bài 7:**  
    Trong phát triển AI và kinh tế số, Việt Nam cần chuyển từ tư duy “một nghiệm tối ưu” sang tư duy “tập phương án đánh đổi”.
    NSGA-II giúp minh bạch hóa đánh đổi, còn TOPSIS giúp chọn một nghiệm thỏa hiệp theo ưu tiên chính sách.
    """)



# ---------------------------------------------------------
# 9. AI ANALYST — PHÂN TÍCH PARETO VÀ ĐÁNH ĐỔI CHÍNH SÁCH
# ---------------------------------------------------------
def build_pareto_policy_intelligence(pareto_df, allocation_df):
    """
    Tạo lớp phân tích bổ sung cho Bài 7:
    - Chọn nghiệm thỏa hiệp bằng TOPSIS.
    - So sánh các nghiệm cực trị: tăng trưởng, bao trùm, môi trường, an ninh.
    - Tạo ma trận hành động vùng cho nghiệm thỏa hiệp.
    - Tạo các kịch bản trọng số để AI Agent có dữ liệu phân tích sâu hơn.
    """

    if pareto_df is None or pareto_df.empty:
        return None

    default_weights = [0.40, 0.25, 0.20, 0.15]
    ranked = topsis_on_pareto(pareto_df, default_weights)
    compromise = ranked.iloc[0]

    max_growth = pareto_df.sort_values("growth_gain", ascending=False).iloc[0]
    best_inclusion = pareto_df.sort_values("inequality", ascending=True).iloc[0]
    best_green = pareto_df.sort_values("emission", ascending=True).iloc[0]
    best_security = pareto_df.sort_values("security_risk", ascending=True).iloc[0]

    key_solutions = pd.DataFrame([
        {
            "Loại nghiệm": "Thỏa hiệp TOPSIS",
            "solution_id": int(compromise["solution_id"]),
            "growth_gain": float(compromise["growth_gain"]),
            "inequality": float(compromise["inequality"]),
            "emission": float(compromise["emission"]),
            "security_risk": float(compromise["security_risk"]),
            "AI_total": float(compromise["AI_total"]),
            "H_total": float(compromise["H_total"]),
            "D_total": float(compromise["D_total"]),
            "I_total": float(compromise["I_total"]),
            "topsis_score": float(compromise["topsis_score"]),
        },
        {
            "Loại nghiệm": "Tăng trưởng cao nhất",
            "solution_id": int(max_growth["solution_id"]),
            "growth_gain": float(max_growth["growth_gain"]),
            "inequality": float(max_growth["inequality"]),
            "emission": float(max_growth["emission"]),
            "security_risk": float(max_growth["security_risk"]),
            "AI_total": float(max_growth["AI_total"]),
            "H_total": float(max_growth["H_total"]),
            "D_total": float(max_growth["D_total"]),
            "I_total": float(max_growth["I_total"]),
            "topsis_score": np.nan,
        },
        {
            "Loại nghiệm": "Bao trùm tốt nhất",
            "solution_id": int(best_inclusion["solution_id"]),
            "growth_gain": float(best_inclusion["growth_gain"]),
            "inequality": float(best_inclusion["inequality"]),
            "emission": float(best_inclusion["emission"]),
            "security_risk": float(best_inclusion["security_risk"]),
            "AI_total": float(best_inclusion["AI_total"]),
            "H_total": float(best_inclusion["H_total"]),
            "D_total": float(best_inclusion["D_total"]),
            "I_total": float(best_inclusion["I_total"]),
            "topsis_score": np.nan,
        },
        {
            "Loại nghiệm": "Phát thải thấp nhất",
            "solution_id": int(best_green["solution_id"]),
            "growth_gain": float(best_green["growth_gain"]),
            "inequality": float(best_green["inequality"]),
            "emission": float(best_green["emission"]),
            "security_risk": float(best_green["security_risk"]),
            "AI_total": float(best_green["AI_total"]),
            "H_total": float(best_green["H_total"]),
            "D_total": float(best_green["D_total"]),
            "I_total": float(best_green["I_total"]),
            "topsis_score": np.nan,
        },
        {
            "Loại nghiệm": "An ninh tốt nhất",
            "solution_id": int(best_security["solution_id"]),
            "growth_gain": float(best_security["growth_gain"]),
            "inequality": float(best_security["inequality"]),
            "emission": float(best_security["emission"]),
            "security_risk": float(best_security["security_risk"]),
            "AI_total": float(best_security["AI_total"]),
            "H_total": float(best_security["H_total"]),
            "D_total": float(best_security["D_total"]),
            "I_total": float(best_security["I_total"]),
            "topsis_score": np.nan,
        },
    ])

    weight_scenarios = pd.DataFrame({
        "Kịch bản": [
            "Mặc định cân bằng",
            "Tăng trưởng nhanh",
            "COP26 xanh hóa",
            "An ninh dữ liệu",
            "Bao trùm vùng miền",
        ],
        "w_growth": [0.40, 0.60, 0.30, 0.30, 0.30],
        "w_inclusion": [0.25, 0.15, 0.20, 0.15, 0.40],
        "w_environment": [0.20, 0.10, 0.35, 0.15, 0.15],
        "w_security": [0.15, 0.15, 0.15, 0.40, 0.15],
    })

    scenario_rows = []
    for _, row in weight_scenarios.iterrows():
        weights = [row["w_growth"], row["w_inclusion"], row["w_environment"], row["w_security"]]
        tmp_ranked = topsis_on_pareto(pareto_df, weights)
        best = tmp_ranked.iloc[0]
        scenario_rows.append({
            "Kịch bản": row["Kịch bản"],
            "solution_id": int(best["solution_id"]),
            "TOPSIS score": float(best["topsis_score"]),
            "growth_gain": float(best["growth_gain"]),
            "inequality": float(best["inequality"]),
            "emission": float(best["emission"]),
            "security_risk": float(best["security_risk"]),
            "AI_total": float(best["AI_total"]),
            "H_total": float(best["H_total"]),
        })
    scenario_result = pd.DataFrame(scenario_rows)

    comp_id = int(compromise["solution_id"])
    comp_alloc = allocation_df[allocation_df["solution_id"] == comp_id].copy()

    alloc_matrix = comp_alloc.pivot(
        index="Vùng",
        columns="Hạng mục",
        values="Ngân sách, tỷ VND"
    )

    region_profile = alloc_matrix.copy()
    region_profile["Tổng ngân sách vùng"] = region_profile.sum(axis=1)

    for col in ["Trí tuệ nhân tạo", "Nhân lực số", "Dữ liệu/CĐS doanh nghiệp", "Hạ tầng số"]:
        if col not in region_profile.columns:
            region_profile[col] = 0.0

    region_profile["Tỷ trọng AI"] = region_profile["Trí tuệ nhân tạo"] / region_profile["Tổng ngân sách vùng"].replace(0, np.nan)
    region_profile["Tỷ trọng H"] = region_profile["Nhân lực số"] / region_profile["Tổng ngân sách vùng"].replace(0, np.nan)
    region_profile["Tỷ trọng D"] = region_profile["Dữ liệu/CĐS doanh nghiệp"] / region_profile["Tổng ngân sách vùng"].replace(0, np.nan)
    region_profile["Tỷ trọng I"] = region_profile["Hạ tầng số"] / region_profile["Tổng ngân sách vùng"].replace(0, np.nan)

    def classify_region(row):
        ai_share = row["Tỷ trọng AI"]
        h_share = row["Tỷ trọng H"]
        d_share = row["Tỷ trọng D"]
        i_share = row["Tỷ trọng I"]

        if ai_share >= 0.30 and h_share >= 0.20:
            return "Tăng tốc AI kèm nhân lực quản trị"
        if h_share + i_share >= 0.55:
            return "Xây nền hạ tầng và nhân lực trước"
        if d_share >= 0.30:
            return "Ưu tiên dữ liệu/CĐS doanh nghiệp"
        if ai_share >= 0.30:
            return "Mở rộng AI nhưng cần kiểm soát rủi ro"
        return "Phân bổ cân bằng, theo dõi thêm"

    region_profile["Nhóm hành động chính sách"] = region_profile.apply(classify_region, axis=1)
    region_profile = region_profile.reset_index()

    corr_growth_inclusion = pareto_df["growth_gain"].corr(pareto_df["inequality"])
    corr_growth_emission = pareto_df["growth_gain"].corr(pareto_df["emission"])
    corr_ai_security = pareto_df["AI_total"].corr(pareto_df["security_risk"])

    growth_premium = (max_growth["growth_gain"] / compromise["growth_gain"] - 1) * 100 if compromise["growth_gain"] != 0 else np.nan
    inclusion_cost = (max_growth["inequality"] / compromise["inequality"] - 1) * 100 if compromise["inequality"] > 0 else np.nan
    emission_cost = (max_growth["emission"] / compromise["emission"] - 1) * 100 if compromise["emission"] > 0 else np.nan

    unique_solutions_by_scenario = scenario_result["solution_id"].nunique()

    metrics = {
        "So_nghiem_Pareto": int(len(pareto_df)),
        "Nghiem_thoa_hiep_solution_id": int(compromise["solution_id"]),
        "TOPSIS_score_thoa_hiep": float(compromise["topsis_score"]),
        "Growth_gain_thoa_hiep": float(compromise["growth_gain"]),
        "Inequality_thoa_hiep": float(compromise["inequality"]),
        "Emission_thoa_hiep": float(compromise["emission"]),
        "Security_risk_thoa_hiep": float(compromise["security_risk"]),
        "AI_total_thoa_hiep": float(compromise["AI_total"]),
        "H_total_thoa_hiep": float(compromise["H_total"]),
        "Max_growth_solution_id": int(max_growth["solution_id"]),
        "Growth_premium_vs_compromise_pct": float(growth_premium),
        "Inclusion_cost_of_max_growth_pct": float(inclusion_cost),
        "Emission_cost_of_max_growth_pct": float(emission_cost),
        "Corr_growth_inequality": float(corr_growth_inclusion),
        "Corr_growth_emission": float(corr_growth_emission),
        "Corr_AI_security_risk": float(corr_ai_security),
        "Unique_solutions_when_weights_change": int(unique_solutions_by_scenario),
    }

    return {
        "ranked": ranked,
        "compromise": compromise,
        "max_growth": max_growth,
        "key_solutions": key_solutions,
        "scenario_result": scenario_result,
        "allocation_matrix": alloc_matrix,
        "region_profile": region_profile,
        "metrics": metrics,
    }


def show_ai_policy_analysis():
    st.header("🤖 AI Analyst — Phân tích Pareto, TOPSIS và đánh đổi chính sách")

    st.markdown("""
    Tab này bổ sung một lớp phân tích tự động cho Bài 7. Thay vì chỉ hiển thị tập nghiệm Pareto,
    phần này tóm tắt các nghiệm cực trị, chọn nghiệm thỏa hiệp, phân tích độ nhạy theo trọng số chính sách
    và tạo dữ liệu đầu vào cho AI Agent diễn giải kết quả.
    """)

    if not PYMOO_AVAILABLE:
        st.error("Chưa cài `pymoo`, nên chưa thể tạo tập nghiệm Pareto cho AI Analyst.")
        st.info("Hãy thêm `pymoo` vào `requirements.txt`, sau đó chạy lại ứng dụng.")
        return

    with st.spinner("Đang tạo dữ liệu Pareto mặc định cho AI Analyst..."):
        pareto_df, allocation_df = run_nsga2(
            total_budget=50000,
            min_region=5000,
            max_region=12000,
            min_h_total=12000,
            min_d_total=8000,
            pop_size=100,
            n_gen=200,
            seed=42,
        )

    if pareto_df is None or pareto_df.empty:
        st.error("Chưa tìm được nghiệm Pareto khả thi để phân tích AI.")
        st.info("Có thể nới ràng buộc, tăng số thế hệ NSGA-II hoặc kiểm tra lại thư viện `pymoo`.")
        return

    intelligence = build_pareto_policy_intelligence(pareto_df, allocation_df)

    if intelligence is None:
        st.error("Không tạo được policy intelligence cho Bài 7.")
        return

    metrics = intelligence["metrics"]
    key_solutions = intelligence["key_solutions"]
    scenario_result = intelligence["scenario_result"]
    alloc_matrix = intelligence["allocation_matrix"]
    region_profile = intelligence["region_profile"]

    st.subheader("1. Policy intelligence của tập nghiệm Pareto")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số nghiệm Pareto", f"{metrics['So_nghiem_Pareto']}")
    c2.metric("Nghiệm thỏa hiệp", f"#{metrics['Nghiem_thoa_hiep_solution_id']}")
    c3.metric("TOPSIS score", f"{metrics['TOPSIS_score_thoa_hiep']:.3f}")
    c4.metric("Growth gain", f"{metrics['Growth_gain_thoa_hiep']:,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Tăng trưởng đổi thêm nếu chọn max-growth", f"{metrics['Growth_premium_vs_compromise_pct']:.2f}%")
    c6.metric("Bao trùm xấu hơn", f"{metrics['Inclusion_cost_of_max_growth_pct']:.2f}%")
    c7.metric("Phát thải tăng thêm", f"{metrics['Emission_cost_of_max_growth_pct']:.2f}%")

    st.dataframe(key_solutions.round(3), use_container_width=True)

    fig_key = px.scatter(
        key_solutions,
        x="growth_gain",
        y="inequality",
        size="emission",
        color="Loại nghiệm",
        hover_data=["solution_id", "security_risk", "AI_total", "H_total"],
        title="Các nghiệm đại diện: tăng trưởng, bao trùm, môi trường, an ninh và thỏa hiệp"
    )
    fig_key.update_layout(height=520)
    st.plotly_chart(fig_key, use_container_width=True)

    st.subheader("2. Độ nhạy khi thay đổi trọng số chính sách")

    st.dataframe(scenario_result.round(3), use_container_width=True)

    fig_scenario = px.parallel_coordinates(
        scenario_result,
        dimensions=["growth_gain", "inequality", "emission", "security_risk", "AI_total", "H_total"],
        color="growth_gain",
        title="Nghiệm thỏa hiệp thay đổi khi ưu tiên chính sách thay đổi"
    )
    fig_scenario.update_layout(height=560)
    st.plotly_chart(fig_scenario, use_container_width=True)

    if metrics["Unique_solutions_when_weights_change"] == 1:
        st.success(
            "Các kịch bản trọng số vẫn chọn cùng một nghiệm thỏa hiệp. Điều này cho thấy nghiệm đề xuất khá vững."
        )
    else:
        st.warning(
            f"Các kịch bản trọng số chọn {metrics['Unique_solutions_when_weights_change']} nghiệm khác nhau. "
            "Điều này cho thấy quyết định cuối cùng nhạy với ưu tiên chính sách."
        )

    st.subheader("3. Ma trận hành động vùng cho nghiệm thỏa hiệp")

    st.dataframe(region_profile.round(3), use_container_width=True)

    fig_region = px.scatter(
        region_profile,
        x="Tỷ trọng AI",
        y="Tỷ trọng H",
        size="Tổng ngân sách vùng",
        color="Nhóm hành động chính sách",
        hover_name="Vùng",
        title="Ma trận hành động: AI share × H share × quy mô ngân sách vùng"
    )
    fig_region.update_layout(height=520)
    st.plotly_chart(fig_region, use_container_width=True)

    st.markdown("#### Ma trận phân bổ ngân sách của nghiệm thỏa hiệp")
    st.dataframe(alloc_matrix.round(1), use_container_width=True)

    fig_alloc = px.imshow(
        alloc_matrix,
        text_auto=".0f",
        aspect="auto",
        title="Heatmap phân bổ ngân sách dùng làm đầu vào cho AI Analyst"
    )
    fig_alloc.update_layout(height=540)
    st.plotly_chart(fig_alloc, use_container_width=True)

    ai_result_table = key_solutions[[
        "Loại nghiệm",
        "solution_id",
        "growth_gain",
        "inequality",
        "emission",
        "security_risk",
        "AI_total",
        "H_total",
        "D_total",
        "I_total",
    ]].copy()

    policy_questions = (
        "Nghiệm thỏa hiệp TOPSIS có cân bằng hợp lý giữa tăng trưởng, bao trùm, môi trường và an ninh dữ liệu không? "
        "Nếu chọn nghiệm tăng trưởng cao nhất thì chi phí cơ hội về bao trùm vùng miền và phát thải là gì? "
        "Các vùng nào nên tăng tốc AI, vùng nào nên ưu tiên nhân lực số và hạ tầng trước? "
        "Khi thay đổi trọng số theo COP26, AI-centric hoặc bao trùm vùng miền, nghiệm được chọn có thay đổi mạnh không? "
        "NSGA-II nên được dùng như bằng chứng hỗ trợ quyết định chính trị hay có thể thay thế quyết định chính trị?"
    )

    st.subheader("4. AI Agent phân tích kết quả Bài 7")

    if AI_AGENT_AVAILABLE:
        render_ai_agent(
            bai_name="Bài 7 — NSGA-II Pareto cho tối ưu đa mục tiêu kinh tế số và AI",
            model_goal=(
                "Tạo tập nghiệm Pareto cho bài toán phân bổ ngân sách kinh tế số Việt Nam, "
                "cân bằng bốn mục tiêu: tăng trưởng GDP, bao trùm vùng miền, phát thải môi trường "
                "và rủi ro an ninh dữ liệu. Sau đó dùng TOPSIS để chọn nghiệm thỏa hiệp."
            ),
            metrics=metrics,
            result_table=ai_result_table.round(3),
            policy_questions=policy_questions,
            key_suffix="bai7",
        )
    else:
        st.error("Chưa tìm thấy `ai_agent.py`, nên chưa thể hiển thị AI Agent.")
        st.info("Hãy đặt `ai_agent.py` cùng cấp với `app.py`, rồi chạy lại Streamlit.")


# ---------------------------------------------------------
# 9. RENDER
# ---------------------------------------------------------
def render():
    st.title("🌐 Bài 7 — NSGA-II Pareto cho tối ưu đa mục tiêu")

    st.markdown("""
    Bài 7 mô phỏng bài toán phân bổ ngân sách kinh tế số Việt Nam bằng **tối ưu đa mục tiêu**.
    Mục tiêu là tìm tập nghiệm Pareto giữa tăng trưởng, bao trùm, môi trường và an ninh dữ liệu.
    """)

    tabs = st.tabs([
        "7.1 Bối cảnh",
        "7.2 Mô hình toán học",
        "7.3 Tham số",
        "7.4 Giải lập trình",
        "7.5 Chính sách",
        "🤖 AI Analyst",
    ])

    with tabs[0]:
        show_context()

    with tabs[1]:
        show_math_model()

    with tabs[2]:
        show_parameters()

    with tabs[3]:
        show_programming_solution()

    with tabs[4]:
        show_policy_discussion()

    with tabs[5]:
        show_ai_policy_analysis()


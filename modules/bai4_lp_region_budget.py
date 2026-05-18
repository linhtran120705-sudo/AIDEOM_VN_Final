from pathlib import Path

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

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


# =========================================================
# BÀI 4 — LP PHÂN BỔ NGÂN SÁCH SỐ THEO NGÀNH - VÙNG
# =========================================================


# ---------------------------------------------------------
# 1. DỮ LIỆU GỐC
# ---------------------------------------------------------
def get_region_item_data():
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
        "D": "CĐS doanh nghiệp",
        "AI": "Năng lực AI",
        "H": "Nhân lực số",
    }

    beta = {
        ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
        ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
        ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
        ("CH",  "I"): 1.20, ("CH",  "D"): 0.75, ("CH",  "AI"): 0.45, ("CH",  "H"): 1.35,
        ("SE",  "I"): 0.90, ("SE",  "D"): 1.30, ("SE",  "AI"): 1.55, ("SE",  "H"): 1.00,
        ("MD",  "I"): 1.10, ("MD",  "D"): 0.85, ("MD",  "AI"): 0.65, ("MD",  "H"): 1.25,
    }

    D0 = {
        "NMM": 38,
        "RRD": 78,
        "NCC": 55,
        "CH": 32,
        "SE": 82,
        "MD": 48,
    }

    beta_df = pd.DataFrame(
        [[beta[(r, j)] for j in items] for r in regions],
        index=[region_names[r] for r in regions],
        columns=[item_names[j] for j in items],
    )

    digital_df = pd.DataFrame({
        "Mã vùng": regions,
        "Vùng": [region_names[r] for r in regions],
        "Digital Index ban đầu Dᵣ": [D0[r] for r in regions],
    })

    return regions, region_names, items, item_names, beta, D0, beta_df, digital_df


def get_beta_long():
    regions, region_names, items, item_names, beta, D0, beta_df, digital_df = get_region_item_data()

    rows = []
    for r in regions:
        for j in items:
            rows.append({
                "Mã vùng": r,
                "Vùng": region_names[r],
                "Hạng mục": item_names[j],
                "Mã hạng mục": j,
                "β tác động biên": beta[(r, j)],
                "Digital Index Dᵣ": D0[r],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 2. GIẢI MÔ HÌNH BẰNG PULP
# ---------------------------------------------------------
def solve_pulp_model(
    total_budget=50000,
    min_region=5000,
    max_region=12000,
    min_h_total=12000,
    gamma=0.002,
    lam=0.7,
    enforce_fairness=True,
    enforce_region_cap=True,
):
    if not PULP_AVAILABLE:
        return None

    regions, region_names, items, item_names, beta, D0, beta_df, digital_df = get_region_item_data()

    model = pulp.LpProblem("VN_Digital_Budget_Region_LP", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", (regions, items), lowBound=0)
    M = pulp.LpVariable("Dmax_after_investment", lowBound=0)

    # Hàm mục tiêu
    model += pulp.lpSum(beta[(r, j)] * x[r][j] for r in regions for j in items), "GDP_gain"

    # C1. Ngân sách tổng
    model += pulp.lpSum(x[r][j] for r in regions for j in items) <= total_budget, "C1_Total_budget"

    # C2, C3. Sàn và trần vùng
    for r in regions:
        model += pulp.lpSum(x[r][j] for j in items) >= min_region, f"C2_Min_region_{r}"
        if enforce_region_cap:
            model += pulp.lpSum(x[r][j] for j in items) <= max_region, f"C3_Max_region_{r}"

    # C4. Sàn nhân lực số toàn quốc
    model += pulp.lpSum(x[r]["H"] for r in regions) >= min_h_total, "C4_Min_total_H"

    # C5. Công bằng vùng bằng biến phụ M
    if enforce_fairness:
        for r in regions:
            model += D0[r] + gamma * x[r]["D"] <= M, f"C5a_Define_max_D_{r}"
        for r in regions:
            model += D0[r] + gamma * x[r]["D"] >= lam * M, f"C5b_Fairness_{r}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return {
            "status": status,
            "objective": np.nan,
            "allocation_matrix": None,
            "allocation_long": None,
            "region_summary": None,
            "item_summary": None,
            "shadow_table": None,
            "fairness_table": None,
        }

    allocation_matrix = pd.DataFrame(
        [[pulp.value(x[r][j]) for j in items] for r in regions],
        index=[region_names[r] for r in regions],
        columns=[item_names[j] for j in items],
    )

    allocation_long = allocation_matrix.reset_index().melt(
        id_vars="index",
        var_name="Hạng mục",
        value_name="Ngân sách phân bổ, tỷ VND",
    ).rename(columns={"index": "Vùng"})

    region_summary = pd.DataFrame({
        "Mã vùng": regions,
        "Vùng": [region_names[r] for r in regions],
        "Tổng ngân sách, tỷ VND": [sum(pulp.value(x[r][j]) for j in items) for r in regions],
        "Digital Index ban đầu": [D0[r] for r in regions],
        "Đầu tư D, tỷ VND": [pulp.value(x[r]["D"]) for r in regions],
        "Digital Index sau đầu tư": [D0[r] + gamma * pulp.value(x[r]["D"]) for r in regions],
    })

    region_summary["Tỷ trọng ngân sách, %"] = (
        region_summary["Tổng ngân sách, tỷ VND"] /
        region_summary["Tổng ngân sách, tỷ VND"].sum() * 100
    )

    item_summary = pd.DataFrame({
        "Hạng mục": [item_names[j] for j in items],
        "Tổng ngân sách, tỷ VND": [
            sum(pulp.value(x[r][j]) for r in regions) for j in items
        ],
    })

    item_summary["Tỷ trọng, %"] = (
        item_summary["Tổng ngân sách, tỷ VND"] /
        item_summary["Tổng ngân sách, tỷ VND"].sum() * 100
    )

    shadow_rows = []
    for name, constraint in model.constraints.items():
        shadow_rows.append({
            "Ràng buộc": name,
            "Shadow price": constraint.pi,
            "Slack": constraint.slack,
        })

    shadow_table = pd.DataFrame(shadow_rows)

    fairness_table = region_summary[[
        "Vùng",
        "Digital Index ban đầu",
        "Đầu tư D, tỷ VND",
        "Digital Index sau đầu tư",
    ]].copy()

    fairness_table["Ngưỡng công bằng λ·max"] = (
        lam * fairness_table["Digital Index sau đầu tư"].max()
    )

    fairness_table["Đạt công bằng?"] = (
        fairness_table["Digital Index sau đầu tư"] >= fairness_table["Ngưỡng công bằng λ·max"]
    )

    return {
        "status": status,
        "objective": pulp.value(model.objective),
        "allocation_matrix": allocation_matrix,
        "allocation_long": allocation_long,
        "region_summary": region_summary,
        "item_summary": item_summary,
        "shadow_table": shadow_table,
        "fairness_table": fairness_table,
    }


# ---------------------------------------------------------
# 3. GIẢI MÔ HÌNH BẰNG CVXPY
# ---------------------------------------------------------
def solve_cvxpy_model(
    total_budget=50000,
    min_region=5000,
    max_region=12000,
    min_h_total=12000,
    gamma=0.002,
    lam=0.7,
    enforce_fairness=True,
    enforce_region_cap=True,
):
    if not CVXPY_AVAILABLE:
        return None

    regions, region_names, items, item_names, beta, D0, beta_df, digital_df = get_region_item_data()

    n_r = len(regions)
    n_j = len(items)

    beta_matrix = np.array([[beta[(r, j)] for j in items] for r in regions])
    D0_vector = np.array([D0[r] for r in regions], dtype=float)

    x = cp.Variable((n_r, n_j), nonneg=True)
    M = cp.Variable(nonneg=True)

    objective = cp.Maximize(cp.sum(cp.multiply(beta_matrix, x)))

    constraints = []

    # C1
    constraints.append(cp.sum(x) <= total_budget)

    # C2, C3
    for r_idx in range(n_r):
        constraints.append(cp.sum(x[r_idx, :]) >= min_region)
        if enforce_region_cap:
            constraints.append(cp.sum(x[r_idx, :]) <= max_region)

    # C4: H là cột cuối cùng
    constraints.append(cp.sum(x[:, 3]) >= min_h_total)

    # C5
    if enforce_fairness:
        for r_idx in range(n_r):
            constraints.append(D0_vector[r_idx] + gamma * x[r_idx, 1] <= M)
        for r_idx in range(n_r):
            constraints.append(D0_vector[r_idx] + gamma * x[r_idx, 1] >= lam * M)

    problem = cp.Problem(objective, constraints)

    solver_used = None
    for solver in ["CLARABEL", "SCIPY", "ECOS", "SCS"]:
        try:
            if solver in cp.installed_solvers():
                problem.solve(solver=solver)
                solver_used = solver
                break
        except Exception:
            continue

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return {
            "status": problem.status,
            "solver": solver_used,
            "objective": np.nan,
            "allocation_matrix": None,
        }

    allocation_matrix = pd.DataFrame(
        x.value,
        index=[region_names[r] for r in regions],
        columns=[item_names[j] for j in items],
    )

    return {
        "status": problem.status,
        "solver": solver_used,
        "objective": problem.value,
        "allocation_matrix": allocation_matrix,
    }


# ---------------------------------------------------------
# 4. KIỂM TRA RÀNG BUỘC
# ---------------------------------------------------------
def check_constraints(result, total_budget, min_region, max_region, min_h_total, gamma, lam):
    if result is None or result["allocation_matrix"] is None:
        return pd.DataFrame()

    allocation = result["allocation_matrix"]
    regions, region_names, items, item_names, beta, D0, beta_df, digital_df = get_region_item_data()

    total_used = allocation.values.sum()
    region_totals = allocation.sum(axis=1)
    total_h = allocation["Nhân lực số"].sum()

    digital_after = []
    for r in regions:
        region_name = region_names[r]
        digital_after.append(D0[r] + gamma * allocation.loc[region_name, "CĐS doanh nghiệp"])

    max_digital_after = max(digital_after)

    checks = []

    checks.append({
        "Nhóm ràng buộc": "C1 Ngân sách tổng",
        "Giá trị kiểm tra": total_used,
        "Ngưỡng": f"≤ {total_budget}",
        "Đạt?": total_used <= total_budget + 1e-5,
    })

    checks.append({
        "Nhóm ràng buộc": "C4 Sàn nhân lực số",
        "Giá trị kiểm tra": total_h,
        "Ngưỡng": f"≥ {min_h_total}",
        "Đạt?": total_h >= min_h_total - 1e-5,
    })

    for region_name, value in region_totals.items():
        checks.append({
            "Nhóm ràng buộc": f"C2 Sàn vùng - {region_name}",
            "Giá trị kiểm tra": value,
            "Ngưỡng": f"≥ {min_region}",
            "Đạt?": value >= min_region - 1e-5,
        })

        checks.append({
            "Nhóm ràng buộc": f"C3 Trần vùng - {region_name}",
            "Giá trị kiểm tra": value,
            "Ngưỡng": f"≤ {max_region}",
            "Đạt?": value <= max_region + 1e-5,
        })

    for idx, r in enumerate(regions):
        region_name = region_names[r]
        checks.append({
            "Nhóm ràng buộc": f"C5 Công bằng - {region_name}",
            "Giá trị kiểm tra": digital_after[idx],
            "Ngưỡng": f"≥ {lam * max_digital_after:.3f}",
            "Đạt?": digital_after[idx] >= lam * max_digital_after - 1e-5,
        })

    return pd.DataFrame(checks)


# ---------------------------------------------------------
# 5. PHẦN 4.1 — BỐI CẢNH
# ---------------------------------------------------------
def show_context():
    st.header("4.1. Bối cảnh Việt Nam")

    st.markdown("""
    Bài 4 chuyển từ bài toán ngân sách đơn giản sang bài toán phân bổ **theo vùng kinh tế - xã hội**.
    Việt Nam có 6 vùng với mức độ sẵn sàng số khác nhau. Nếu chỉ tối đa hóa GDP gain, vốn có thể chảy về
    các vùng có hệ số sinh lợi cao. Vì vậy, mô hình cần thêm ràng buộc công bằng vùng miền để tránh tập trung
    quá mức vào một vài vùng phát triển hơn.
    """)

    regions, region_names, items, item_names, beta, D0, beta_df, digital_df = get_region_item_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ngân sách", "50.000", "tỷ VND")
    c2.metric("Số vùng", "6")
    c3.metric("Hạng mục đầu tư", "4")
    c4.metric("Biến quyết định", "24", "xⱼ,ᵣ")

    st.subheader("Ảnh 4.1 — Chênh lệch Digital Index ban đầu giữa 6 vùng")

    fig_digital = px.bar(
        digital_df.sort_values("Digital Index ban đầu Dᵣ", ascending=False),
        x="Vùng",
        y="Digital Index ban đầu Dᵣ",
        text="Digital Index ban đầu Dᵣ",
        title="Digital Index ban đầu Dᵣ: nền tảng cho ràng buộc công bằng vùng",
    )
    fig_digital.update_traces(textposition="outside")
    fig_digital.update_layout(height=480)
    st.plotly_chart(fig_digital, use_container_width=True)

    st.dataframe(digital_df, use_container_width=True)

    st.subheader("Ảnh 4.2 — Luồng quyết định phân bổ ngân sách số quốc gia")

    labels = [
        "Ngân sách số quốc gia\n50.000 tỷ VND",
        "6 vùng kinh tế - xã hội",
        "4 hạng mục\nI, D, AI, H",
        "Tối đa hóa GDP gain",
        "Ràng buộc công bằng vùng",
        "Phân bổ tối ưu xⱼ,ᵣ",
    ]

    fig_flow = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.3),
            label=labels,
        ),
        link=dict(
            source=[0, 1, 2, 2, 3, 4],
            target=[1, 2, 3, 4, 5, 5],
            value=[50, 50, 25, 25, 30, 20],
        )
    )])

    fig_flow.update_layout(
        title="Ảnh 4.2 — Từ ngân sách quốc gia đến phân bổ tối ưu theo vùng - hạng mục",
        height=520,
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    st.info(
        "Thông điệp bối cảnh: mô hình không chỉ trả lời nên đầu tư vào đâu để GDP tăng cao nhất, "
        "mà còn kiểm tra chi phí của việc bảo đảm công bằng số giữa các vùng."
    )


# ---------------------------------------------------------
# 6. PHẦN 4.2 — MÔ HÌNH TOÁN HỌC
# ---------------------------------------------------------
def show_math_model():
    st.header("4.2. Mô hình toán học đầy đủ")

    st.markdown("""
    Bài toán là một mô hình quy hoạch tuyến tính. Biến quyết định là mức ngân sách phân bổ cho từng cặp
    **vùng r** và **hạng mục đầu tư j**.
    """)

    st.subheader("Bước 1 — Biến quyết định")

    st.latex(r"""
    x_{j,r} \in \mathbb{R}^{+}, \quad
    j \in \{I,D,AI,H\}, \quad r \in \{1,\ldots,6\}
    """)

    variable_table = pd.DataFrame({
        "Ký hiệu": ["I", "D", "AI", "H", "r"],
        "Ý nghĩa": [
            "Hạ tầng số",
            "Chuyển đổi số doanh nghiệp",
            "Năng lực AI",
            "Nhân lực số",
            "Vùng kinh tế - xã hội",
        ],
    })

    st.dataframe(variable_table, use_container_width=True)

    st.subheader("Bước 2 — Hàm mục tiêu")

    st.latex(r"""
    \max Z = \sum_{r}\sum_{j} \beta_{j,r}x_{j,r}
    """)

    st.markdown("""
    Trong đó, `βⱼ,ᵣ` là hệ số tác động biên của 1 tỷ VND đầu tư vào hạng mục `j` tại vùng `r`.
    Mô hình chọn phương án phân bổ làm cho tổng GDP gain kỳ vọng `Z` lớn nhất.
    """)

    st.subheader("Bước 3 — Các nhóm ràng buộc")

    constraint_table = pd.DataFrame({
        "Mã": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "Công thức": [
            "ΣᵣΣⱼ xⱼ,ᵣ ≤ 50.000",
            "Σⱼ xⱼ,ᵣ ≥ 5.000, ∀r",
            "Σⱼ xⱼ,ᵣ ≤ 12.000, ∀r",
            "Σᵣ xᴴ,ᵣ ≥ 12.000",
            "Dᵣ + γxᴰ,ᵣ ≥ λ·maxᵣ(Dᵣ + γxᴰ,ᵣ)",
            "xⱼ,ᵣ ≥ 0",
        ],
        "Ý nghĩa chính sách": [
            "Không vượt ngân sách kinh tế số quốc gia.",
            "Mỗi vùng có sàn ngân sách để tránh bị bỏ lại.",
            "Mỗi vùng có trần ngân sách để tránh tập trung quá mức.",
            "Nhân lực số phải được ưu tiên tối thiểu ở cấp quốc gia.",
            "Sau đầu tư, vùng yếu không được cách quá xa vùng mạnh nhất.",
            "Không thể phân bổ ngân sách âm.",
        ],
    })

    st.dataframe(constraint_table, use_container_width=True)

    st.latex(r"""
    D_r + \gamma x_{D,r} \leq M,\quad
    D_r + \gamma x_{D,r} \geq \lambda M
    """)

    st.markdown("""
    Để đưa ràng buộc `max` vào mô hình tuyến tính, ta dùng biến phụ `M`.
    `M` đại diện cho mức Digital Index sau đầu tư cao nhất. Khi đó, mọi vùng phải đạt ít nhất `λM`.
    """)

    st.success(
        "Điểm mạnh của mô hình: vừa tối đa hóa hiệu quả kinh tế, vừa lượng hóa được chi phí của công bằng vùng miền."
    )


# ---------------------------------------------------------
# 7. PHẦN 4.3 — BẢNG HỆ SỐ β
# ---------------------------------------------------------
def show_beta_data():
    st.header("4.3. Bảng hệ số tác động biên βⱼ,ᵣ")

    regions, region_names, items, item_names, beta, D0, beta_df, digital_df = get_region_item_data()
    beta_long = get_beta_long()

    st.markdown("""
    Hệ số `βⱼ,ᵣ` phản ánh mức GDP gain kỳ vọng khi đầu tư vào một hạng mục cụ thể tại một vùng cụ thể.
    Vùng có sẵn nền tảng số cao thường có hệ số AI và D cao hơn; vùng còn yếu thường có hệ số nhân lực số H
    hoặc hạ tầng I cao hơn do dư địa cải thiện lớn.
    """)

    st.dataframe(beta_df, use_container_width=True)

    st.subheader("Ảnh 4.3 — Heatmap hệ số tác động biên βⱼ,ᵣ")

    fig_beta = px.imshow(
        beta_df,
        text_auto=".2f",
        aspect="auto",
        title="Heatmap βⱼ,ᵣ: hạng mục nào hiệu quả nhất ở từng vùng?",
    )
    fig_beta.update_layout(height=560)
    st.plotly_chart(fig_beta, use_container_width=True)

    st.subheader("Ảnh 4.4 — Hệ số tác động theo từng vùng")

    fig_group = px.bar(
        beta_long,
        x="Vùng",
        y="β tác động biên",
        color="Hạng mục",
        barmode="group",
        title="So sánh β theo vùng và hạng mục",
    )
    fig_group.update_layout(height=560)
    st.plotly_chart(fig_group, use_container_width=True)

    st.subheader("Ảnh 4.5 — Digital Index ban đầu và logic ưu tiên đầu tư")

    fig_scatter = px.scatter(
        beta_long,
        x="Digital Index Dᵣ",
        y="β tác động biên",
        color="Hạng mục",
        size="β tác động biên",
        hover_name="Vùng",
        title="Digital Index ban đầu và hệ số tác động biên",
    )
    fig_scatter.update_layout(height=520)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.info(
        "Cách đọc: Đông Nam Bộ có β_AI cao nhất nên nếu chỉ tối đa hóa GDP gain, vốn dễ chảy vào AI tại vùng này. "
        "Ngược lại, Tây Nguyên có β_AI thấp nhưng β_H và β_I cao, nên mô hình có xu hướng ưu tiên nhân lực số và hạ tầng trước."
    )


# ---------------------------------------------------------
# 8. PHẦN 4.4 — GIẢI BÀI TOÁN
# ---------------------------------------------------------
def show_programming_solution():
    st.header("4.4. Giải bài toán lập trình")

    st.markdown("""
    Phần này giải mô hình bằng **PuLP/CBC**, giải lại bằng **CVXPY**, kiểm tra ràng buộc,
    vẽ heatmap phân bổ tối ưu và so sánh với mô hình không có ràng buộc công bằng.
    """)

    st.subheader("Thiết lập tham số mô hình")

    c1, c2, c3, c4 = st.columns(4)
    total_budget = c1.number_input(
        "Ngân sách tổng, tỷ VND",
        min_value=30000,
        max_value=100000,
        value=50000,
        step=5000,
        key="bai4_total_budget",
    )
    min_region = c2.number_input(
        "Sàn mỗi vùng, tỷ VND",
        min_value=0,
        max_value=10000,
        value=5000,
        step=1000,
        key="bai4_min_region",
    )
    max_region = c3.number_input(
       "Trần mỗi vùng, tỷ VND",
        min_value=6000,
        max_value=30000,
        value=13000,
        step=1000,
        key="bai4_max_region",
    )
    min_h_total = c4.number_input(
        "Sàn H toàn quốc, tỷ VND",
        min_value=0,
        max_value=30000,
        value=12000,
        step=1000,
        key="bai4_min_h",
    )

    c5, c6 = st.columns(2)
    gamma = c5.number_input(
        "γ - hiệu quả đầu tư D",
        min_value=0.0005,
        max_value=0.01,
        value=0.002,
        step=0.0005,
        format="%.4f",
        key="bai4_gamma",
    )
    lam = c6.slider(
        "λ - mức công bằng vùng",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        key="bai4_lambda",
    )
# -----------------------------------------------------
# Kiểm tra nhanh khả thi của ràng buộc công bằng C5
# -----------------------------------------------------
regions, region_names, items, item_names, beta, D0, beta_df, digital_df = get_region_item_data()

d_max_initial = max(D0.values())
d_min_initial = min(D0.values())

required_d_for_weakest = (lam * d_max_initial - d_min_initial) / gamma
max_possible_d_investment = max_region

if required_d_for_weakest > max_possible_d_investment:
    st.error(
        f"Ràng buộc hiện tại có nguy cơ KHÔNG KHẢ THI. "
        f"Vùng yếu nhất cần ít nhất {required_d_for_weakest:,.0f} tỷ VND đầu tư D "
        f"để đạt λ = {lam:.2f}, nhưng trần mỗi vùng chỉ là {max_region:,.0f} tỷ VND."
    )

    suggested_lambda = (d_min_initial + gamma * max_region) / d_max_initial

    st.warning(
        f"Gợi ý sửa: giảm λ xuống tối đa khoảng {suggested_lambda:.2f}, "
        f"hoặc tăng trần mỗi vùng lên ít nhất {required_d_for_weakest:,.0f} tỷ VND."
    )

    st.info(
        "Bạn vẫn có thể tiếp tục thử nghiệm mô hình, nhưng nếu PuLP báo Infeasible thì nguyên nhân chính "
        "là ràng buộc công bằng C5 quá chặt so với trần ngân sách mỗi vùng."
    )
    # -----------------------------------------------------
    # 4.4.1 PuLP
    # -----------------------------------------------------
    st.subheader("Câu 4.4.1 — Giải bằng PuLP/CBC")

    if not PULP_AVAILABLE:
        st.error("Chưa cài PuLP. Hãy thêm `pulp` vào requirements.txt.")
        return

    pulp_result = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=True,
        enforce_region_cap=True,
    )

    if pulp_result["status"] != "Optimal":
        st.error(f"Mô hình PuLP không tối ưu. Trạng thái: {pulp_result['status']}")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Trạng thái PuLP", pulp_result["status"])
    m2.metric("Z* GDP gain", f"{pulp_result['objective']:,.2f}", "tỷ VND")
    m3.metric("Tổng ngân sách dùng", f"{pulp_result['allocation_matrix'].values.sum():,.0f}", "tỷ VND")

    st.markdown("#### Ma trận phân bổ tối ưu 6×4, đơn vị: tỷ VND")
    st.dataframe(pulp_result["allocation_matrix"].round(2), use_container_width=True)

    st.markdown("#### Kiểm tra ràng buộc")
    checks = check_constraints(
        pulp_result,
        total_budget,
        min_region,
        max_region,
        min_h_total,
        gamma,
        lam,
    )
    st.dataframe(checks.round(3), use_container_width=True)

    # -----------------------------------------------------
    # 4.4.2 CVXPY
    # -----------------------------------------------------
    st.subheader("Câu 4.4.2 — Giải lại bằng CVXPY và so sánh với PuLP")

    if not CVXPY_AVAILABLE:
        st.warning("Chưa cài CVXPY. Hãy thêm `cvxpy` vào requirements.txt nếu muốn giải bằng CVXPY.")
        cvxpy_result = None
    else:
        cvxpy_result = solve_cvxpy_model(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_h_total=min_h_total,
            gamma=gamma,
            lam=lam,
            enforce_fairness=True,
            enforce_region_cap=True,
        )

        if cvxpy_result["allocation_matrix"] is not None:
            diff_obj = abs(pulp_result["objective"] - cvxpy_result["objective"])
            max_diff_x = abs(
                pulp_result["allocation_matrix"] - cvxpy_result["allocation_matrix"]
            ).values.max()

            c7, c8, c9 = st.columns(3)
            c7.metric("Trạng thái CVXPY", cvxpy_result["status"])
            c8.metric("Solver", str(cvxpy_result["solver"]))
            c9.metric("Chênh lệch Z*", f"{diff_obj:,.6f}")

            st.dataframe(cvxpy_result["allocation_matrix"].round(2), use_container_width=True)

            if max_diff_x < 1e-2:
                st.success(
                    "PuLP và CVXPY cho kết quả gần như giống nhau. Đây là kiểm tra tái lập tốt cho mô hình."
                )
            else:
                st.warning(
                    f"PuLP và CVXPY có chênh lệch phân bổ tối đa khoảng {max_diff_x:.4f}. "
                    "Điều này có thể do nghiệm tối ưu không duy nhất hoặc sai khác số học giữa solver."
                )
        else:
            st.warning(f"CVXPY chưa tìm được nghiệm tối ưu. Trạng thái: {cvxpy_result['status']}")

    # -----------------------------------------------------
    # 4.4.3 Heatmap phân bổ
    # -----------------------------------------------------
    st.subheader("Câu 4.4.3 — Heatmap phân bổ tối ưu")

    fig_heat = px.imshow(
        pulp_result["allocation_matrix"],
        text_auto=".0f",
        aspect="auto",
        title="Ảnh 4.6 — Heatmap phân bổ ngân sách tối ưu xⱼ,ᵣ",
    )
    fig_heat.update_layout(height=560)
    st.plotly_chart(fig_heat, use_container_width=True)

    fig_region = px.bar(
        pulp_result["region_summary"].sort_values("Tổng ngân sách, tỷ VND", ascending=False),
        x="Vùng",
        y="Tổng ngân sách, tỷ VND",
        text="Tổng ngân sách, tỷ VND",
        title="Ảnh 4.7 — Tổng ngân sách theo vùng",
    )
    fig_region.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_region.update_layout(height=480)
    st.plotly_chart(fig_region, use_container_width=True)

    fig_item = px.pie(
        pulp_result["item_summary"],
        names="Hạng mục",
        values="Tổng ngân sách, tỷ VND",
        title="Ảnh 4.8 — Cơ cấu ngân sách theo hạng mục đầu tư",
        hole=0.42,
    )
    fig_item.update_layout(height=480)
    st.plotly_chart(fig_item, use_container_width=True)

    top_region = pulp_result["region_summary"].sort_values("Tổng ngân sách, tỷ VND", ascending=False).iloc[0]
    top_item_each_region = pulp_result["allocation_matrix"].idxmax(axis=1).reset_index()
    top_item_each_region.columns = ["Vùng", "Hạng mục được ưu tiên nhất"]

    st.dataframe(top_item_each_region, use_container_width=True)

    st.info(
        f"Vùng nhận ngân sách lớn nhất là **{top_region['Vùng']}** với khoảng "
        f"**{top_region['Tổng ngân sách, tỷ VND']:,.0f} tỷ VND**. "
        "Bảng trên cho biết hạng mục được ưu tiên nhất ở từng vùng."
    )

    # -----------------------------------------------------
    # 4.4.4 So sánh với mô hình không có công bằng
    # -----------------------------------------------------
    st.subheader("Câu 4.4.4 — Chi phí kinh tế của ràng buộc công bằng vùng miền")

    no_fair_result = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=False,
        enforce_region_cap=True,
    )

    comparison_rows = [
        {
            "Kịch bản": "Có ràng buộc công bằng C5",
            "Z* GDP gain, tỷ VND": pulp_result["objective"],
        },
        {
            "Kịch bản": "Bỏ ràng buộc công bằng C5",
            "Z* GDP gain, tỷ VND": no_fair_result["objective"],
        },
    ]

    compare_fair = pd.DataFrame(comparison_rows)
    cost_fairness = no_fair_result["objective"] - pulp_result["objective"]
    cost_fairness_pct = cost_fairness / no_fair_result["objective"] * 100

    st.dataframe(compare_fair.round(2), use_container_width=True)

    c10, c11 = st.columns(2)
    c10.metric("Chi phí công bằng", f"{cost_fairness:,.2f}", "tỷ VND GDP gain")
    c11.metric("Tỷ lệ giảm Z*", f"{cost_fairness_pct:.2f}%")

    fig_compare = px.bar(
        compare_fair,
        x="Kịch bản",
        y="Z* GDP gain, tỷ VND",
        text="Z* GDP gain, tỷ VND",
        title="Ảnh 4.9 — So sánh Z* khi có và không có ràng buộc công bằng",
    )
    fig_compare.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_compare.update_layout(height=460)
    st.plotly_chart(fig_compare, use_container_width=True)

    st.success(
        f"Chi phí kinh tế của công bằng vùng miền là khoảng **{cost_fairness:,.2f} tỷ VND GDP gain**, "
        f"tương đương **{cost_fairness_pct:.2f}%** so với mô hình không có C5. "
        "Đây là đánh đổi định lượng giữa hiệu quả kinh tế và công bằng vùng."
    )

    return {
        "full": pulp_result,
        "no_fair": no_fair_result,
        "cvxpy": cvxpy_result,
        "checks": checks,
        "cost_fairness": cost_fairness,
        "cost_fairness_pct": cost_fairness_pct,
    }


# ---------------------------------------------------------
# 9. PHẦN 4.5 — THẢO LUẬN CHÍNH SÁCH
# ---------------------------------------------------------
def show_policy_discussion():
    st.header("4.5. Câu hỏi thảo luận chính sách")

    if not PULP_AVAILABLE:
        st.error("Cần cài PuLP để tạo phần thảo luận chính sách.")
        return

    total_budget = 50000
    min_region = 5000
    max_region = 12000
    min_h_total = 12000
    gamma = 0.002
    lam = 0.7

    full = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=True,
        enforce_region_cap=True,
    )

    no_fair = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=False,
        enforce_region_cap=True,
    )

    no_cap = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=True,
        enforce_region_cap=False,
    )

    if full["status"] != "Optimal":
        st.error("Mô hình gốc không tối ưu nên không thể thảo luận.")
        return

    # -----------------------------------------------------
    # Câu a
    # -----------------------------------------------------
    st.subheader("a) Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào?")

    no_fair_region = no_fair["region_summary"].sort_values("Tổng ngân sách, tỷ VND", ascending=False)
    st.dataframe(no_fair_region.round(2), use_container_width=True)

    fig_a = px.bar(
        no_fair_region,
        x="Vùng",
        y="Tổng ngân sách, tỷ VND",
        text="Tổng ngân sách, tỷ VND",
        title="Minh chứng câu a — Phân bổ theo vùng khi bỏ ràng buộc công bằng C5",
    )
    fig_a.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_a.update_layout(height=480)
    st.plotly_chart(fig_a, use_container_width=True)

    top_region = no_fair_region.iloc[0]

    st.success(
        f"Trả lời: nếu bỏ ràng buộc công bằng, vốn có xu hướng chảy nhiều nhất về **{top_region['Vùng']}**, "
        f"với khoảng **{top_region['Tổng ngân sách, tỷ VND']:,.0f} tỷ VND**. "
        "Nguyên nhân là mô hình tối đa hóa GDP gain nên ưu tiên các vùng/hạng mục có hệ số β cao, "
        "đặc biệt các vùng đã có nền tảng số tốt và khả năng hấp thụ AI cao."
    )

    st.warning(
        "Hậu quả dài hạn: nếu vốn chỉ tập trung vào vùng mạnh, khoảng cách số giữa các vùng có thể nới rộng, "
        "vùng yếu khó nâng năng lực hấp thụ công nghệ, và mục tiêu phát triển bao trùm bị suy giảm."
    )

    # -----------------------------------------------------
    # Câu b
    # -----------------------------------------------------
    st.subheader("b) Ràng buộc trần ngân sách mỗi vùng có phải chính sách phân quyền không?")

    cap_compare = pd.DataFrame({
        "Kịch bản": [
            "Có trần vùng C3",
            "Bỏ trần vùng C3",
        ],
        "Z* GDP gain, tỷ VND": [
            full["objective"],
            no_cap["objective"],
        ],
    })

    reduction = no_cap["objective"] - full["objective"]
    reduction_pct = reduction / no_cap["objective"] * 100

    st.dataframe(cap_compare.round(2), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Mức giảm Z* do C3", f"{reduction:,.2f}", "tỷ VND")
    c2.metric("Tỷ lệ giảm", f"{reduction_pct:.2f}%")

    fig_b = px.bar(
        cap_compare,
        x="Kịch bản",
        y="Z* GDP gain, tỷ VND",
        text="Z* GDP gain, tỷ VND",
        title="Minh chứng câu b — Chi phí của trần ngân sách vùng C3",
    )
    fig_b.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_b.update_layout(height=440)
    st.plotly_chart(fig_b, use_container_width=True)

    st.info(
        f"Trả lời: C3 có thể xem là một dạng chính sách phân quyền vì nó giới hạn mức tập trung ngân sách vào một vùng. "
        f"Trong mô hình, C3 làm Z* giảm khoảng **{reduction:,.2f} tỷ VND**, tương đương **{reduction_pct:.2f}%**. "
        "Nếu mức giảm nhỏ, đánh đổi này có thể chấp nhận được vì đổi lại ngân sách được phân bổ đều hơn, "
        "giảm rủi ro tập trung và tăng tính chính danh của chính sách vùng."
    )

    # -----------------------------------------------------
    # Câu c
    # -----------------------------------------------------
    st.subheader("c) Tây Nguyên nên đầu tư AI hay H và I trước?")

    ch_alloc = full["allocation_matrix"].loc["Tây Nguyên"].reset_index()
    ch_alloc.columns = ["Hạng mục", "Ngân sách, tỷ VND"]

    regions, region_names, items, item_names, beta, D0, beta_df, digital_df = get_region_item_data()

    ch_beta = pd.DataFrame({
        "Hạng mục": [item_names[j] for j in items],
        "β của Tây Nguyên": [beta[("CH", j)] for j in items],
    })

    ch_compare = ch_alloc.merge(ch_beta, on="Hạng mục")
    ch_compare["GDP gain kỳ vọng"] = ch_compare["Ngân sách, tỷ VND"] * ch_compare["β của Tây Nguyên"]

    st.dataframe(ch_compare.round(2), use_container_width=True)

    fig_c = go.Figure()
    fig_c.add_trace(go.Bar(
        x=ch_compare["Hạng mục"],
        y=ch_compare["Ngân sách, tỷ VND"],
        name="Ngân sách",
    ))
    fig_c.add_trace(go.Scatter(
        x=ch_compare["Hạng mục"],
        y=ch_compare["β của Tây Nguyên"],
        name="β Tây Nguyên",
        mode="lines+markers",
        yaxis="y2",
    ))

    fig_c.update_layout(
        title="Minh chứng câu c — Tây Nguyên: ngân sách tối ưu và hệ số β",
        yaxis=dict(title="Ngân sách, tỷ VND"),
        yaxis2=dict(title="β", overlaying="y", side="right"),
        height=460,
    )
    st.plotly_chart(fig_c, use_container_width=True)

    ai_beta = beta[("CH", "AI")]
    h_beta = beta[("CH", "H")]
    i_beta = beta[("CH", "I")]

    st.success(
        f"Trả lời: Tây Nguyên không nên ưu tiên AI trước nếu xét theo hệ số tác động hiện tại. "
        f"β_AI của Tây Nguyên chỉ **{ai_beta:.2f}**, thấp hơn β_H = **{h_beta:.2f}** và β_I = **{i_beta:.2f}**. "
        "Mô hình vì vậy có xu hướng ưu tiên **nhân lực số H** và **hạ tầng số I** trước, "
        "vì đây là hai nền tảng giúp vùng nâng năng lực hấp thụ công nghệ trước khi mở rộng đầu tư AI."
    )

    st.markdown("""
    **Kết luận chính sách của Bài 4:**  
    Bài toán cho thấy phát triển kinh tế số không nên chỉ chạy theo nơi có hiệu quả ngắn hạn cao nhất.
    Với vùng còn yếu như Tây Nguyên hoặc Trung du miền núi phía Bắc, chính sách hợp lý hơn là đầu tư nền tảng:
    hạ tầng số, chuyển đổi số doanh nghiệp và nhân lực số. Khi nền tảng tăng lên, đầu tư AI mới có khả năng phát huy hiệu quả.
    """)


# ---------------------------------------------------------
# 10. HÀM RENDER CHÍNH
# ---------------------------------------------------------
def render():
    st.title("🧭 Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng")

    st.markdown("""
    Bài 4 xây dựng mô hình **quy hoạch tuyến tính cỡ vừa** để phân bổ 50.000 tỷ VND ngân sách kinh tế số
    cho 6 vùng kinh tế - xã hội và 4 hạng mục đầu tư. Mục tiêu là tối đa hóa GDP gain nhưng vẫn bảo đảm
    công bằng vùng miền.
    """)

    tabs = st.tabs([
        "4.1 Bối cảnh",
        "4.2 Mô hình toán học",
        "4.3 Dữ liệu β",
        "4.4 Giải lập trình",
        "4.5 Chính sách",
    ])

    with tabs[0]:
        show_context()

    with tabs[1]:
        show_math_model()

    with tabs[2]:
        show_beta_data()

    with tabs[3]:
        show_programming_solution()

    with tabs[4]:
        show_policy_discussion()

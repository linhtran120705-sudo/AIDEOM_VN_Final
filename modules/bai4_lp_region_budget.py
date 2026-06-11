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

try:
    from ai_agent import render_ai_agent
    AI_AGENT_AVAILABLE = True
except Exception:
    render_ai_agent = None
    AI_AGENT_AVAILABLE = False


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
    max_region=13000,
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
    max_region=13000,
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
    beta_matrix = np.array([[beta[(r, j)] for j in items] for r in regions])
    D0_vector = np.array([D0[r] for r in regions], dtype=float)

    x = cp.Variable((n_r, len(items)), nonneg=True)
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

    # C4: H là cột thứ 4, index = 3
    constraints.append(cp.sum(x[:, 3]) >= min_h_total)

    # C5: D là cột thứ 2, index = 1
    if enforce_fairness:
        for r_idx in range(n_r):
            constraints.append(D0_vector[r_idx] + gamma * x[r_idx, 1] <= M)
        for r_idx in range(n_r):
            constraints.append(D0_vector[r_idx] + gamma * x[r_idx, 1] >= lam * M)

    problem = cp.Problem(objective, constraints)

    solver_used = None
    installed = cp.installed_solvers()

    for solver in ["CLARABEL", "SCIPY", "ECOS", "SCS"]:
        try:
            if solver in installed:
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
# 5. KIỂM TRA KHẢ THI NHANH C5
# ---------------------------------------------------------
def quick_feasibility_check(max_region, gamma, lam):
    regions, region_names, items, item_names, beta, D0, beta_df, digital_df = get_region_item_data()

    d_max_initial = max(D0.values())
    d_min_initial = min(D0.values())

    required_d_for_weakest = max(0, (lam * d_max_initial - d_min_initial) / gamma)
    suggested_lambda = (d_min_initial + gamma * max_region) / d_max_initial

    is_warning = required_d_for_weakest > max_region

    return {
        "is_warning": is_warning,
        "required_d_for_weakest": required_d_for_weakest,
        "suggested_lambda": suggested_lambda,
        "d_max_initial": d_max_initial,
        "d_min_initial": d_min_initial,
    }


# ---------------------------------------------------------
# 6. PHẦN 4.1 — BỐI CẢNH
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
# 7. PHẦN 4.2 — MÔ HÌNH TOÁN HỌC
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
            "Σⱼ xⱼ,ᵣ ≤ 13.000, ∀r",
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
# 8. PHẦN 4.3 — BẢNG HỆ SỐ β
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
# 9. PHẦN 4.4 — GIẢI BÀI TOÁN
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
    feasibility = quick_feasibility_check(max_region=max_region, gamma=gamma, lam=lam)

    if feasibility["is_warning"]:
        st.error(
            f"Ràng buộc hiện tại có nguy cơ KHÔNG KHẢ THI. "
            f"Vùng yếu nhất cần ít nhất {feasibility['required_d_for_weakest']:,.0f} tỷ VND đầu tư D "
            f"để đạt λ = {lam:.2f}, nhưng trần mỗi vùng chỉ là {max_region:,.0f} tỷ VND."
        )

        st.warning(
            f"Gợi ý sửa: giảm λ xuống tối đa khoảng {feasibility['suggested_lambda']:.2f}, "
            f"hoặc tăng trần mỗi vùng lên ít nhất {feasibility['required_d_for_weakest']:,.0f} tỷ VND."
        )

        st.info(
            "Bạn vẫn có thể tiếp tục thử nghiệm mô hình. Nếu PuLP báo Infeasible thì nguyên nhân chính "
            "là ràng buộc công bằng C5 quá chặt so với trần ngân sách mỗi vùng."
        )
    else:
        st.success(
            "Kiểm tra nhanh C5: bộ tham số hiện tại có khả năng khả thi. "
            "Có thể tiếp tục giải mô hình bằng PuLP/CBC."
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

    if pulp_result is None:
        st.error("Không thể chạy PuLP. Hãy kiểm tra thư viện `pulp` trong requirements.txt.")
        return

    if pulp_result["status"] != "Optimal":
        st.error(f"Mô hình PuLP không tối ưu. Trạng thái: {pulp_result['status']}")

        st.markdown("""
        **Cách xử lý nhanh:**

        - Tăng `Trần mỗi vùng` lên 13.000 hoặc 14.000 tỷ VND.
        - Hoặc giảm `λ - mức công bằng vùng` xuống 0.65.
        - Hoặc tăng `γ - hiệu quả đầu tư D` nếu giả định đầu tư chuyển đổi số tạo tác động mạnh hơn.
        """)

        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Trạng thái PuLP", pulp_result["status"])
    m2.metric("Z* GDP gain", f"{pulp_result['objective']:,.2f}", "tỷ VND")
    m3.metric(
        "Tổng ngân sách dùng",
        f"{pulp_result['allocation_matrix'].values.sum():,.0f}",
        "tỷ VND"
    )

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

    cvxpy_result = None

    if not CVXPY_AVAILABLE:
        st.warning("Chưa cài CVXPY. Hãy thêm `cvxpy` vào requirements.txt nếu muốn giải bằng CVXPY.")
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

        if cvxpy_result is not None and cvxpy_result["allocation_matrix"] is not None:
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
            st.warning("CVXPY chưa tìm được nghiệm tối ưu hoặc solver không khả dụng.")

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

    top_region = pulp_result["region_summary"].sort_values(
        "Tổng ngân sách, tỷ VND",
        ascending=False
    ).iloc[0]

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

    if no_fair_result is None or no_fair_result["status"] != "Optimal":
        st.warning("Mô hình không có ràng buộc công bằng chưa tối ưu, không thể so sánh chi phí công bằng.")
        return

    compare_fair = pd.DataFrame([
        {
            "Kịch bản": "Có ràng buộc công bằng C5",
            "Z* GDP gain, tỷ VND": pulp_result["objective"],
        },
        {
            "Kịch bản": "Bỏ ràng buộc công bằng C5",
            "Z* GDP gain, tỷ VND": no_fair_result["objective"],
        },
    ])

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
# 10. PHẦN 4.5 — THẢO LUẬN CHÍNH SÁCH
# ---------------------------------------------------------
def show_policy_discussion():
    st.header("4.5. Câu hỏi thảo luận chính sách")

    if not PULP_AVAILABLE:
        st.error("Cần cài PuLP để tạo phần thảo luận chính sách.")
        return

    total_budget = 50000
    min_region = 5000
    max_region = 13000
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

    if full is None or full["status"] != "Optimal":
        st.error("Mô hình gốc không tối ưu nên không thể thảo luận.")
        st.info("Gợi ý: dùng max_region = 13000 hoặc giảm λ xuống 0.65 để mô hình khả thi hơn.")
        return

    if no_fair is None or no_fair["status"] != "Optimal":
        st.error("Mô hình bỏ công bằng không tối ưu nên không thể thảo luận câu a.")
        return

    if no_cap is None or no_cap["status"] != "Optimal":
        st.error("Mô hình bỏ trần vùng không tối ưu nên không thể thảo luận câu b.")
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
# 11. PHẦN AI ANALYST — PHÂN TÍCH CHÍNH SÁCH NÂNG CAO
# ---------------------------------------------------------
def build_bai4_policy_intelligence(full_result, no_fair_result=None, no_cap_result=None):
    """
    Tạo bảng policy intelligence cho Bài 4.
    Bảng này không thay đổi nghiệm LP, chỉ diễn giải nghiệm theo các lát cắt chính sách mới:
    hiệu quả, công bằng vùng, năng lực hấp thụ AI, rủi ro tập trung và chi phí đánh đổi.
    """
    if full_result is None or full_result.get("status") != "Optimal":
        return pd.DataFrame()

    region_summary = full_result["region_summary"].copy()
    item_summary = full_result["item_summary"].copy()
    allocation_matrix = full_result["allocation_matrix"].copy()

    total_budget_used = allocation_matrix.values.sum()
    objective = full_result["objective"]
    efficiency_ratio = objective / total_budget_used if total_budget_used else np.nan

    top_region = region_summary.sort_values("Tổng ngân sách, tỷ VND", ascending=False).iloc[0]
    lowest_initial_region = region_summary.sort_values("Digital Index ban đầu", ascending=True).iloc[0]
    highest_initial_region = region_summary.sort_values("Digital Index ban đầu", ascending=False).iloc[0]

    digital_gap_before = (
        region_summary["Digital Index ban đầu"].max() -
        region_summary["Digital Index ban đầu"].min()
    )
    digital_gap_after = (
        region_summary["Digital Index sau đầu tư"].max() -
        region_summary["Digital Index sau đầu tư"].min()
    )
    digital_gap_change = digital_gap_after - digital_gap_before

    item_share = item_summary.set_index("Hạng mục")["Tỷ trọng, %"].to_dict()
    ai_share = item_share.get("Năng lực AI", 0.0)
    h_share = item_share.get("Nhân lực số", 0.0)
    i_share = item_share.get("Hạ tầng số", 0.0)
    d_share = item_share.get("CĐS doanh nghiệp", 0.0)

    cost_fairness = np.nan
    cost_fairness_pct = np.nan
    if no_fair_result is not None and no_fair_result.get("status") == "Optimal":
        cost_fairness = no_fair_result["objective"] - full_result["objective"]
        cost_fairness_pct = cost_fairness / no_fair_result["objective"] * 100

    cost_cap = np.nan
    cost_cap_pct = np.nan
    if no_cap_result is not None and no_cap_result.get("status") == "Optimal":
        cost_cap = no_cap_result["objective"] - full_result["objective"]
        cost_cap_pct = cost_cap / no_cap_result["objective"] * 100

    concentration_hhi = ((region_summary["Tỷ trọng ngân sách, %"] / 100) ** 2).sum()

    rows = [
        {
            "Lớp phân tích": "Hiệu quả tổng thể",
            "Chỉ báo": "Z*/ngân sách sử dụng",
            "Giá trị": efficiency_ratio,
            "Diễn giải chính sách": (
                "Đo hiệu quả chuyển hóa 1 tỷ VND ngân sách số thành GDP gain kỳ vọng. "
                "Chỉ báo này giúp so sánh chất lượng phân bổ, không chỉ nhìn tổng Z*."
            ),
        },
        {
            "Lớp phân tích": "Công bằng vùng",
            "Chỉ báo": "Thay đổi chênh lệch Digital Index sau đầu tư",
            "Giá trị": digital_gap_change,
            "Diễn giải chính sách": (
                "Nếu giá trị âm, khoảng cách số giữa vùng mạnh và vùng yếu được thu hẹp. "
                "Nếu dương, cần xem lại λ, γ hoặc trần/sàn vùng."
            ),
        },
        {
            "Lớp phân tích": "Chi phí công bằng",
            "Chỉ báo": "Mất mát Z* do ràng buộc công bằng C5, %",
            "Giá trị": cost_fairness_pct,
            "Diễn giải chính sách": (
                "Lượng hóa giá phải trả khi yêu cầu vùng yếu không bị bỏ lại quá xa. "
                "Đây là đánh đổi giữa hiệu quả GDP ngắn hạn và phát triển bao trùm."
            ),
        },
        {
            "Lớp phân tích": "Phân quyền vùng",
            "Chỉ báo": "Mất mát Z* do trần ngân sách vùng C3, %",
            "Giá trị": cost_cap_pct,
            "Diễn giải chính sách": (
                "Cho biết chi phí của việc hạn chế tập trung vốn vào một số vùng mạnh. "
                "Nếu chi phí thấp, C3 là công cụ phân quyền hợp lý."
            ),
        },
        {
            "Lớp phân tích": "Rủi ro tập trung",
            "Chỉ báo": "HHI cơ cấu ngân sách vùng",
            "Giá trị": concentration_hhi,
            "Diễn giải chính sách": (
                "HHI càng cao nghĩa là ngân sách càng tập trung vào ít vùng. "
                "Chỉ báo này bổ sung cho việc nhìn đơn thuần vùng nhận ngân sách lớn nhất."
            ),
        },
        {
            "Lớp phân tích": "Năng lực hấp thụ AI",
            "Chỉ báo": "Tỷ trọng ngân sách cho AI, %",
            "Giá trị": ai_share,
            "Diễn giải chính sách": (
                "Nếu tỷ trọng AI cao ở vùng đã sẵn sàng số, mô hình nhấn mạnh hiệu quả ngắn hạn. "
                "Nếu tỷ trọng H/I/D cao ở vùng yếu, mô hình ưu tiên xây nền trước khi mở rộng AI."
            ),
        },
        {
            "Lớp phân tích": "Nền tảng triển khai",
            "Chỉ báo": "Tỷ trọng I + D + H, %",
            "Giá trị": i_share + d_share + h_share,
            "Diễn giải chính sách": (
                "Đo mức ưu tiên cho hạ tầng số, chuyển đổi số doanh nghiệp và nhân lực số. "
                "Đây là nhóm điều kiện nền để AI đi vào thực thi chính sách và đời sống."
            ),
        },
        {
            "Lớp phân tích": "Vùng cần chú ý",
            "Chỉ báo": f"Vùng Digital Index thấp nhất: {lowest_initial_region['Vùng']}",
            "Giá trị": lowest_initial_region["Digital Index ban đầu"],
            "Diễn giải chính sách": (
                "Vùng có nền tảng số ban đầu thấp cần được đọc như đối tượng ưu tiên năng lực nền, "
                "không nên đánh giá chỉ bằng hệ số AI hiện tại."
            ),
        },
        {
            "Lớp phân tích": "Vùng có sức kéo",
            "Chỉ báo": f"Vùng Digital Index cao nhất: {highest_initial_region['Vùng']}",
            "Giá trị": highest_initial_region["Digital Index ban đầu"],
            "Diễn giải chính sách": (
                "Vùng có nền tảng số cao thường có khả năng hấp thụ AI và CĐS tốt hơn, "
                "nhưng cần tránh để vốn tập trung quá mức gây nới rộng khoảng cách số."
            ),
        },
        {
            "Lớp phân tích": "Trung tâm ngân sách",
            "Chỉ báo": f"Vùng nhận ngân sách lớn nhất: {top_region['Vùng']}",
            "Giá trị": top_region["Tổng ngân sách, tỷ VND"],
            "Diễn giải chính sách": (
                "Vùng nhận vốn lớn nhất là điểm cần kiểm tra năng lực giải ngân, tác động lan tỏa "
                "và khả năng phối hợp vùng, không chỉ xét hiệu quả biên β."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_bai4_region_action_matrix(full_result):
    """
    Phân loại hành động chính sách cho từng vùng dựa trên:
    - mức Digital Index ban đầu,
    - khoản ngân sách được phân bổ,
    - hạng mục được ưu tiên nhất,
    - mức cải thiện Digital Index sau đầu tư.
    """
    if full_result is None or full_result.get("status") != "Optimal":
        return pd.DataFrame()

    region_summary = full_result["region_summary"].copy()
    allocation = full_result["allocation_matrix"].copy()

    rows = []
    for _, row in region_summary.iterrows():
        region = row["Vùng"]
        top_item = allocation.loc[region].idxmax()
        top_value = allocation.loc[region].max()
        digital_initial = row["Digital Index ban đầu"]
        digital_after = row["Digital Index sau đầu tư"]
        improvement = digital_after - digital_initial
        total_budget = row["Tổng ngân sách, tỷ VND"]

        if digital_initial < 45 and top_item in ["Hạ tầng số", "Nhân lực số", "CĐS doanh nghiệp"]:
            action = "Xây nền năng lực số trước"
            recommendation = (
                "Ưu tiên hạ tầng, nhân lực và chuyển đổi số doanh nghiệp để nâng năng lực hấp thụ công nghệ. "
                "AI nên triển khai thí điểm sau khi điều kiện nền cải thiện."
            )
        elif digital_initial >= 70 and top_item in ["Năng lực AI", "CĐS doanh nghiệp"]:
            action = "Tăng tốc AI và CĐS nâng cao"
            recommendation = (
                "Có thể triển khai AI, dữ liệu và tự động hóa ở quy mô lớn hơn, đồng thời yêu cầu lan tỏa sang vùng yếu hơn."
            )
        elif total_budget >= region_summary["Tổng ngân sách, tỷ VND"].median():
            action = "Cân bằng hiệu quả và công bằng"
            recommendation = (
                "Vùng nhận ngân sách tương đối lớn, cần gắn phân bổ vốn với KPI về giải ngân, tăng Digital Index và lan tỏa vùng."
            )
        else:
            action = "Theo dõi và hỗ trợ mục tiêu"
            recommendation = (
                "Duy trì mức đầu tư tối thiểu, chọn dự án có khả năng lan tỏa rõ, tránh phân tán nguồn lực vào quá nhiều mục tiêu."
            )

        rows.append({
            "Vùng": region,
            "Digital Index ban đầu": digital_initial,
            "Digital Index sau đầu tư": digital_after,
            "Cải thiện Digital Index": improvement,
            "Tổng ngân sách, tỷ VND": total_budget,
            "Hạng mục ưu tiên nhất": top_item,
            "Ngân sách hạng mục ưu tiên, tỷ VND": top_value,
            "Nhóm hành động": action,
            "Khuyến nghị thực thi": recommendation,
        })

    return pd.DataFrame(rows)


def show_ai_policy_analysis():
    """
    Tab AI Analyst cho Bài 4.
    Phần này gọi render_ai_agent từ ai_agent.py, đồng thời thêm lớp phân tích mới:
    policy intelligence, ma trận hành động vùng và bảng chỉ báo đánh đổi hiệu quả - công bằng.
    """
    st.header("🤖 AI Analyst — Phân tích phân bổ ngân sách số theo vùng")

    st.markdown("""
    Phần này bổ sung tác nhân AI cho Bài 4. AI không thay đổi nghiệm tối ưu của mô hình,
    mà đọc kết quả LP theo các lớp chính sách: **hiệu quả kinh tế, công bằng vùng, năng lực hấp thụ AI,
    chi phí của ràng buộc và khuyến nghị thực thi theo từng vùng**.
    """)

    if not AI_AGENT_AVAILABLE:
        st.error(
            "Chưa import được `render_ai_agent` từ `ai_agent.py`. "
            "Hãy kiểm tra file `ai_agent.py` có nằm cùng cấp với `app.py` không."
        )
        return

    if not PULP_AVAILABLE:
        st.error("Cần cài PuLP để tạo kết quả đầu vào cho AI Analyst. Hãy thêm `pulp` vào requirements.txt.")
        return

    total_budget = 50000
    min_region = 5000
    max_region = 13000
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

    if full is None or full.get("status") != "Optimal":
        st.error("Mô hình gốc của Bài 4 chưa tối ưu nên chưa thể tạo phân tích AI.")
        st.info("Gợi ý: dùng max_region = 13.000 hoặc giảm λ xuống 0,65 để mô hình khả thi hơn.")
        return

    policy_intel = build_bai4_policy_intelligence(full, no_fair, no_cap)
    action_matrix = build_bai4_region_action_matrix(full)

    st.subheader("Bảng 4.AI.1 — Policy intelligence cho nghiệm tối ưu")
    st.dataframe(policy_intel.round(4), use_container_width=True)

    st.subheader("Bảng 4.AI.2 — Ma trận hành động theo vùng")
    st.dataframe(action_matrix.round(3), use_container_width=True)

    fig_action = px.scatter(
        action_matrix,
        x="Digital Index ban đầu",
        y="Tổng ngân sách, tỷ VND",
        size="Ngân sách hạng mục ưu tiên, tỷ VND",
        color="Nhóm hành động",
        hover_name="Vùng",
        title="Ảnh 4.AI.1 — Ma trận hành động: nền tảng số ban đầu × ngân sách tối ưu",
    )
    fig_action.update_layout(height=520)
    st.plotly_chart(fig_action, use_container_width=True)

    item_summary = full["item_summary"].copy()
    region_summary = full["region_summary"].copy()

    top_region = region_summary.sort_values("Tổng ngân sách, tỷ VND", ascending=False).iloc[0]
    weakest_region = region_summary.sort_values("Digital Index ban đầu", ascending=True).iloc[0]
    strongest_region = region_summary.sort_values("Digital Index ban đầu", ascending=False).iloc[0]

    no_fair_objective = np.nan
    cost_fairness = np.nan
    cost_fairness_pct = np.nan
    if no_fair is not None and no_fair.get("status") == "Optimal":
        no_fair_objective = no_fair["objective"]
        cost_fairness = no_fair["objective"] - full["objective"]
        cost_fairness_pct = cost_fairness / no_fair["objective"] * 100

    no_cap_objective = np.nan
    cost_cap = np.nan
    cost_cap_pct = np.nan
    if no_cap is not None and no_cap.get("status") == "Optimal":
        no_cap_objective = no_cap["objective"]
        cost_cap = no_cap["objective"] - full["objective"]
        cost_cap_pct = cost_cap / no_cap["objective"] * 100

    digital_gap_before = region_summary["Digital Index ban đầu"].max() - region_summary["Digital Index ban đầu"].min()
    digital_gap_after = region_summary["Digital Index sau đầu tư"].max() - region_summary["Digital Index sau đầu tư"].min()

    metrics = {
        "Tong_ngan_sach_ty_VND": float(total_budget),
        "Trang_thai_mo_hinh": str(full["status"]),
        "Z_co_cong_bang_C5": float(full["objective"]),
        "Z_khong_cong_bang_C5": float(no_fair_objective) if not pd.isna(no_fair_objective) else None,
        "Chi_phi_cong_bang_ty_VND": float(cost_fairness) if not pd.isna(cost_fairness) else None,
        "Chi_phi_cong_bang_pct": float(cost_fairness_pct) if not pd.isna(cost_fairness_pct) else None,
        "Z_bo_tran_vung_C3": float(no_cap_objective) if not pd.isna(no_cap_objective) else None,
        "Chi_phi_tran_vung_C3_pct": float(cost_cap_pct) if not pd.isna(cost_cap_pct) else None,
        "Vung_nhan_ngan_sach_lon_nhat": str(top_region["Vùng"]),
        "Ngan_sach_vung_lon_nhat_ty_VND": float(top_region["Tổng ngân sách, tỷ VND"]),
        "Vung_Digital_Index_thap_nhat": str(weakest_region["Vùng"]),
        "Digital_Index_thap_nhat": float(weakest_region["Digital Index ban đầu"]),
        "Vung_Digital_Index_cao_nhat": str(strongest_region["Vùng"]),
        "Digital_Index_cao_nhat": float(strongest_region["Digital Index ban đầu"]),
        "Khoang_cach_Digital_Index_truoc_dau_tu": float(digital_gap_before),
        "Khoang_cach_Digital_Index_sau_dau_tu": float(digital_gap_after),
        "Tong_dau_tu_AI_ty_VND": float(item_summary.loc[item_summary["Hạng mục"] == "Năng lực AI", "Tổng ngân sách, tỷ VND"].iloc[0]),
        "Tong_dau_tu_H_ty_VND": float(item_summary.loc[item_summary["Hạng mục"] == "Nhân lực số", "Tổng ngân sách, tỷ VND"].iloc[0]),
        "Tong_dau_tu_I_ty_VND": float(item_summary.loc[item_summary["Hạng mục"] == "Hạ tầng số", "Tổng ngân sách, tỷ VND"].iloc[0]),
        "Tong_dau_tu_D_ty_VND": float(item_summary.loc[item_summary["Hạng mục"] == "CĐS doanh nghiệp", "Tổng ngân sách, tỷ VND"].iloc[0]),
        "lambda_cong_bang": float(lam),
        "gamma_hieu_qua_dau_tu_D": float(gamma),
    }

    ai_result_table = action_matrix[[
        "Vùng",
        "Digital Index ban đầu",
        "Digital Index sau đầu tư",
        "Tổng ngân sách, tỷ VND",
        "Hạng mục ưu tiên nhất",
        "Nhóm hành động",
        "Khuyến nghị thực thi",
    ]].copy()

    policy_questions = (
        "Nếu bỏ ràng buộc công bằng C5, vốn sẽ tập trung vào vùng nào và rủi ro dài hạn là gì? "
        "Ràng buộc trần vùng C3 có thể xem là chính sách phân quyền hay không, và chi phí đánh đổi là bao nhiêu? "
        "Với vùng nền tảng số yếu như Tây Nguyên hoặc Trung du miền núi phía Bắc, nên đầu tư AI ngay hay ưu tiên H/I/D trước? "
        "Kết quả mô hình gợi ý cơ chế phối hợp trung ương - địa phương như thế nào để vừa hiệu quả vừa bao trùm?"
    )

    render_ai_agent(
        bai_name="Bài 4 — Phân bổ ngân sách số theo ngành - vùng bằng quy hoạch tuyến tính",
        model_goal=(
            "Tối ưu hóa phân bổ 50.000 tỷ VND ngân sách kinh tế số cho 6 vùng kinh tế - xã hội "
            "và 4 hạng mục đầu tư I, D, AI, H nhằm tối đa hóa GDP gain, đồng thời bảo đảm sàn vùng, "
            "trần vùng, sàn nhân lực số và ràng buộc công bằng Digital Index giữa các vùng."
        ),
        metrics=metrics,
        result_table=ai_result_table,
        policy_questions=policy_questions,
        key_suffix="bai4",
    )


# ---------------------------------------------------------
# 12. HÀM RENDER CHÍNH
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
        "🤖 AI Analyst",
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

    with tabs[5]:
        show_ai_policy_analysis()

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
# BÀI 5 — QUY HOẠCH NGUYÊN HỖN HỢP MIP
# LỰA CHỌN DỰ ÁN CHUYỂN ĐỔI SỐ
# =========================================================


# ---------------------------------------------------------
# 1. DỮ LIỆU 15 DỰ ÁN
# ---------------------------------------------------------
def get_project_data():
    df = pd.DataFrame({
        "id": list(range(1, 16)),
        "code": [f"P{i}" for i in range(1, 16)],
        "project_name": [
            "Trung tâm dữ liệu quốc gia Hòa Lạc",
            "Trung tâm dữ liệu quốc gia phía Nam",
            "Hệ thống 5G phủ sóng toàn quốc",
            "Hệ thống định danh điện tử VNeID 2.0",
            "Cổng dịch vụ công quốc gia v3",
            "Y tế số quốc gia",
            "Giáo dục số K-12 toàn quốc",
            "Trung tâm AI quốc gia + supercomputing",
            "Sandbox tài chính số",
            "Logistics thông minh + cảng biển số",
            "Nông nghiệp số ĐBSCL",
            "Đào tạo 50.000 kỹ sư AI/bán dẫn",
            "Khu công nghiệp bán dẫn Bắc Ninh - Bắc Giang",
            "An ninh mạng quốc gia SOC",
            "Open Data + dữ liệu mở quốc gia",
        ],
        "field": [
            "Hạ tầng",
            "Hạ tầng",
            "Hạ tầng",
            "Chính phủ số",
            "Chính phủ số",
            "Y tế số",
            "Giáo dục",
            "AI",
            "Tài chính số",
            "Logistics",
            "Nông nghiệp",
            "Nhân lực",
            "Bán dẫn",
            "An ninh",
            "Dữ liệu",
        ],
        "cost_total": [
            12000, 11500, 18000, 4500, 3200,
            5800, 6500, 15000, 2500, 7200,
            4800, 8500, 20000, 3800, 1500
        ],
        "benefit_npv": [
            21500, 20800, 32500, 9200, 6800,
            11400, 12200, 28500, 5800, 13800,
            8500, 16200, 35000, 7500, 3800
        ],
        "cost_year_1_2": [
            8500, 7500, 12000, 3500, 2500,
            4000, 4500, 9000, 1800, 5000,
            3500, 5500, 13000, 2800, 1200
        ],
        "cost_year_3_5": [
            3500, 4000, 6000, 1000, 700,
            1800, 2000, 6000, 700, 2200,
            1300, 3000, 7000, 1000, 300
        ],
    })

    df["benefit_cost_ratio"] = df["benefit_npv"] / df["cost_total"]

    def probability(row):
        if row["field"] == "Hạ tầng":
            return 0.85
        if row["field"] == "Chính phủ số":
            return 0.75
        if row["field"] in ["AI", "Bán dẫn"]:
            return 0.65
        return 0.80

    df["completion_probability"] = df.apply(probability, axis=1)
    df["expected_benefit"] = df["benefit_npv"] * df["completion_probability"]

    return df


# ---------------------------------------------------------
# 2. GIẢI MÔ HÌNH MIP BẰNG PULP
# ---------------------------------------------------------
def solve_project_mip(
    total_budget=80000,
    early_budget=40000,
    min_projects=7,
    max_projects=11,
    force_p14=True,
    keep_data_center_exclusion=True,
    force_both_data_centers=False,
    risk_adjusted=False,
    synergy_bonus=0,
):
    if not PULP_AVAILABLE:
        return None

    df = get_project_data()
    ids = df["id"].tolist()

    cost = dict(zip(df["id"], df["cost_total"]))
    early = dict(zip(df["id"], df["cost_year_1_2"]))

    if risk_adjusted:
        benefit = dict(zip(df["id"], df["expected_benefit"]))
        objective_name = "Expected_benefit"
    else:
        benefit = dict(zip(df["id"], df["benefit_npv"]))
        objective_name = "NPV_benefit"

    model = pulp.LpProblem("VN_Project_Selection_MIP", pulp.LpMaximize)

    y = pulp.LpVariable.dicts("y", ids, lowBound=0, upBound=1, cat="Binary")

    # Biến phụ cho hiệu ứng cộng hưởng P8 và P13
    z_8_13 = pulp.LpVariable("z_synergy_P8_P13", lowBound=0, upBound=1, cat="Binary")

    # Hàm mục tiêu
    model += (
        pulp.lpSum(benefit[i] * y[i] for i in ids) + synergy_bonus * z_8_13,
        objective_name
    )

    # C1. Ngân sách tổng 5 năm
    model += pulp.lpSum(cost[i] * y[i] for i in ids) <= total_budget, "C1_Total_budget_5_years"

    # C2. Ngân sách năm 1-2
    model += pulp.lpSum(early[i] * y[i] for i in ids) <= early_budget, "C2_Budget_year_1_2"

    # C3. Loại trừ trung tâm dữ liệu
    if keep_data_center_exclusion:
        model += y[1] + y[2] <= 1, "C3_Data_center_exclusion"

    # C4. AI quốc gia cần đào tạo kỹ sư
    model += y[8] <= y[12], "C4_AI_requires_training"

    # C5. Bán dẫn cần đào tạo kỹ sư
    model += y[13] <= y[12], "C5_Semiconductor_requires_training"

    # C6. Cân đối lĩnh vực
    model += y[4] + y[5] >= 1, "C6_At_least_one_digital_government"
    if force_p14:
        model += y[14] >= 1, "C6_Cybersecurity_required"

    # C7. Số lượng dự án
    model += pulp.lpSum(y[i] for i in ids) >= min_projects, "C7_Min_number_projects"
    model += pulp.lpSum(y[i] for i in ids) <= max_projects, "C7_Max_number_projects"

    # Kịch bản Quốc hội yêu cầu cả P1 và P2
    if force_both_data_centers:
        model += y[1] >= 1, "Scenario_Force_P1"
        model += y[2] >= 1, "Scenario_Force_P2"

    # Hiệu ứng cộng hưởng: z = 1 nếu cả P8 và P13 cùng được chọn
    model += z_8_13 <= y[8], "Synergy_upper_P8"
    model += z_8_13 <= y[13], "Synergy_upper_P13"
    model += z_8_13 >= y[8] + y[13] - 1, "Synergy_lower_P8_P13"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return {
            "status": status,
            "objective": np.nan,
            "selected_df": pd.DataFrame(),
            "full_df": df,
            "summary": {},
            "constraint_check": pd.DataFrame(),
            "synergy_active": None,
        }

    selected_ids = [i for i in ids if pulp.value(y[i]) > 0.5]
    selected_df = df[df["id"].isin(selected_ids)].copy()

    selected_df["selected"] = 1
    df_result = df.copy()
    df_result["selected"] = df_result["id"].apply(lambda i: 1 if i in selected_ids else 0)

    total_cost = selected_df["cost_total"].sum()
    total_early = selected_df["cost_year_1_2"].sum()

    if risk_adjusted:
        total_benefit = selected_df["expected_benefit"].sum()
    else:
        total_benefit = selected_df["benefit_npv"].sum()

    objective_value = pulp.value(model.objective)
    marginal_npv_ratio = objective_value / total_cost if total_cost > 0 else np.nan

    summary = {
        "status": status,
        "objective": objective_value,
        "total_cost": total_cost,
        "total_early_cost": total_early,
        "total_benefit_without_synergy": total_benefit,
        "npv_per_cost": marginal_npv_ratio,
        "number_selected": len(selected_ids),
        "selected_codes": ", ".join(selected_df["code"].tolist()),
        "risk_adjusted": risk_adjusted,
        "synergy_bonus": synergy_bonus,
        "synergy_active": pulp.value(z_8_13),
    }

    checks = pd.DataFrame({
        "Ràng buộc": [
            "C1 Ngân sách tổng 5 năm",
            "C2 Ngân sách năm 1-2",
            "C3 P1 và P2 không cùng chọn",
            "C4 Nếu chọn P8 thì phải chọn P12",
            "C5 Nếu chọn P13 thì phải chọn P12",
            "C6 Ít nhất một dự án chính phủ số",
            "C6 P14 an ninh mạng bắt buộc",
            "C7 Số lượng dự án tối thiểu",
            "C7 Số lượng dự án tối đa",
        ],
        "Giá trị kiểm tra": [
            total_cost,
            total_early,
            int(1 in selected_ids) + int(2 in selected_ids),
            f"P8={int(8 in selected_ids)}, P12={int(12 in selected_ids)}",
            f"P13={int(13 in selected_ids)}, P12={int(12 in selected_ids)}",
            int(4 in selected_ids) + int(5 in selected_ids),
            int(14 in selected_ids),
            len(selected_ids),
            len(selected_ids),
        ],
        "Ngưỡng": [
            f"≤ {total_budget}",
            f"≤ {early_budget}",
            "≤ 1" if keep_data_center_exclusion else "không áp dụng",
            "P8 ≤ P12",
            "P13 ≤ P12",
            "≥ 1",
            "≥ 1" if force_p14 else "không bắt buộc",
            f"≥ {min_projects}",
            f"≤ {max_projects}",
        ],
        "Đạt?": [
            total_cost <= total_budget,
            total_early <= early_budget,
            (int(1 in selected_ids) + int(2 in selected_ids) <= 1) if keep_data_center_exclusion else True,
            (8 not in selected_ids) or (12 in selected_ids),
            (13 not in selected_ids) or (12 in selected_ids),
            (4 in selected_ids) or (5 in selected_ids),
            (14 in selected_ids) if force_p14 else True,
            len(selected_ids) >= min_projects,
            len(selected_ids) <= max_projects,
        ]
    })

    return {
        "status": status,
        "objective": objective_value,
        "selected_df": selected_df,
        "full_df": df_result,
        "summary": summary,
        "constraint_check": checks,
        "synergy_active": pulp.value(z_8_13),
    }


# ---------------------------------------------------------
# 3. PHẦN 5.1 — BỐI CẢNH VIỆT NAM
# ---------------------------------------------------------
def show_context():
    st.header("5.1. Bối cảnh Việt Nam")

    st.markdown("""
    Bài 5 đặt ra một bài toán lựa chọn danh mục dự án chuyển đổi số quốc gia giai đoạn 2026–2030.
    Nhà nước có nhiều dự án ứng cử, nhưng ngân sách có hạn, ngân sách giai đoạn đầu cũng có hạn,
    và một số dự án có quan hệ ràng buộc lẫn nhau. Vì vậy, bài toán không chỉ là chọn dự án có lợi ích cao,
    mà là chọn **tập dự án tối ưu** dưới các ràng buộc chính sách.
    """)

    st.markdown("""
    Về thực tiễn Việt Nam, bài toán này gắn với tinh thần **QĐ 749/QĐ-TTg** về chuyển đổi số quốc gia
    và định hướng khoa học - công nghệ, đổi mới sáng tạo, chuyển đổi số theo **Nghị quyết 57-NQ/TW**.
    Trọng tâm chính sách là: đầu tư đủ mạnh vào hạ tầng số, dữ liệu, AI, nhân lực, an ninh mạng,
    nhưng vẫn kiểm soát ngân sách và rủi ro triển khai.
    """)

    df = get_project_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số dự án ứng cử", "15")
    c2.metric("Ngân sách 5 năm", "80.000", "tỷ VND")
    c3.metric("Ngân sách năm 1-2", "40.000", "tỷ VND")
    c4.metric("Số dự án cần chọn", "7–11")

    field_summary = df.groupby("field", as_index=False).agg(
        total_cost=("cost_total", "sum"),
        total_benefit=("benefit_npv", "sum"),
        n_projects=("id", "count")
    )
    field_summary["benefit_cost_ratio"] = field_summary["total_benefit"] / field_summary["total_cost"]

    st.subheader("Ảnh 5.1 — Cơ cấu danh mục dự án theo lĩnh vực")

    fig_field = px.treemap(
        df,
        path=["field", "code"],
        values="cost_total",
        color="benefit_cost_ratio",
        hover_data=["project_name", "benefit_npv", "benefit_cost_ratio"],
        title="Danh mục 15 dự án: quy mô chi phí và tỷ suất lợi ích/chi phí"
    )
    fig_field.update_layout(height=560)
    st.plotly_chart(fig_field, use_container_width=True)

    st.subheader("Ảnh 5.2 — Chi phí và lợi ích theo lĩnh vực")

    fig_field_bar = go.Figure()
    fig_field_bar.add_trace(go.Bar(
        x=field_summary["field"],
        y=field_summary["total_cost"],
        name="Tổng chi phí"
    ))
    fig_field_bar.add_trace(go.Bar(
        x=field_summary["field"],
        y=field_summary["total_benefit"],
        name="Tổng lợi ích NPV"
    ))
    fig_field_bar.update_layout(
        title="Tổng chi phí và NPV kỳ vọng theo lĩnh vực",
        xaxis_title="Lĩnh vực",
        yaxis_title="Tỷ VND",
        barmode="group",
        height=500
    )
    st.plotly_chart(fig_field_bar, use_container_width=True)

    st.info(
        "Thông điệp bối cảnh: nếu chỉ nhìn lợi ích tuyệt đối, các dự án lớn như 5G, AI, bán dẫn rất hấp dẫn. "
        "Nhưng MIP buộc ta xét đồng thời chi phí, ngân sách năm đầu, ràng buộc tiên quyết, số lượng dự án và an ninh mạng."
    )


# ---------------------------------------------------------
# 4. PHẦN 5.2 — DANH MỤC 15 DỰ ÁN
# ---------------------------------------------------------
def show_project_data():
    st.header("5.2. Danh mục 15 dự án ứng cử")

    df = get_project_data()

    display = df[[
        "code", "project_name", "field", "cost_total", "benefit_npv",
        "cost_year_1_2", "cost_year_3_5", "benefit_cost_ratio",
        "completion_probability", "expected_benefit"
    ]].copy()

    display.columns = [
        "Mã", "Tên dự án", "Lĩnh vực", "Chi phí 5 năm, tỷ VND", "NPV lợi ích, tỷ VND",
        "Chi phí năm 1-2", "Chi phí năm 3-5", "NPV/Chi phí",
        "Xác suất đúng tiến độ", "Lợi ích kỳ vọng có rủi ro"
    ]

    st.dataframe(display.round(3), use_container_width=True)

    st.subheader("Ảnh 5.3 — Bản đồ chi phí - lợi ích của 15 dự án")

    fig_scatter = px.scatter(
        df,
        x="cost_total",
        y="benefit_npv",
        size="benefit_cost_ratio",
        color="field",
        hover_name="code",
        hover_data=["project_name", "benefit_cost_ratio"],
        title="Dự án nào đắt, dự án nào tạo lợi ích lớn?"
    )
    fig_scatter.update_layout(
        xaxis_title="Chi phí 5 năm, tỷ VND",
        yaxis_title="NPV lợi ích, tỷ VND",
        height=540
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Ảnh 5.4 — Xếp hạng tỷ suất lợi ích/chi phí")

    ratio_df = df.sort_values("benefit_cost_ratio", ascending=False)

    fig_ratio = px.bar(
        ratio_df,
        x="benefit_cost_ratio",
        y="code",
        color="field",
        orientation="h",
        text="benefit_cost_ratio",
        hover_data=["project_name"],
        title="Tỷ suất NPV/Chi phí của từng dự án"
    )
    fig_ratio.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_ratio.update_layout(height=560, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_ratio, use_container_width=True)

    st.subheader("Ảnh 5.5 — Áp lực ngân sách năm 1-2 so với cả giai đoạn")

    cost_long = df.melt(
        id_vars=["code", "project_name", "field"],
        value_vars=["cost_year_1_2", "cost_year_3_5"],
        var_name="Giai đoạn",
        value_name="Chi phí, tỷ VND"
    )

    cost_long["Giai đoạn"] = cost_long["Giai đoạn"].replace({
        "cost_year_1_2": "Năm 1-2",
        "cost_year_3_5": "Năm 3-5",
    })

    fig_stack = px.bar(
        cost_long,
        x="code",
        y="Chi phí, tỷ VND",
        color="Giai đoạn",
        hover_data=["project_name"],
        title="Cơ cấu chi phí theo giai đoạn triển khai"
    )
    fig_stack.update_layout(height=520)
    st.plotly_chart(fig_stack, use_container_width=True)

    st.info(
        "Cách đọc dữ liệu: một dự án có tỷ suất cao chưa chắc được chọn nếu nó không phù hợp với ràng buộc "
        "ngân sách năm đầu, ràng buộc tiên quyết hoặc giới hạn số lượng dự án."
    )


# ---------------------------------------------------------
# 5. PHẦN 5.3 — MÔ HÌNH TOÁN HỌC
# ---------------------------------------------------------
def show_math_model():
    st.header("5.3. Mô hình toán học")

    st.markdown("""
    Bài toán là một mô hình **quy hoạch nguyên hỗn hợp MIP** dạng knapsack tổng quát hóa.
    Khác với LP, biến quyết định ở đây là biến nhị phân: dự án được chọn hoặc không được chọn.
    """)

    st.subheader("Bước 1 — Biến quyết định")

    st.latex(r"""
    y_i \in \{0,1\},\quad i=1,\ldots,15
    """)

    st.markdown("""
    Trong đó:
    - `yᵢ = 1` nếu dự án `i` được chọn.
    - `yᵢ = 0` nếu dự án `i` không được chọn.
    """)

    st.subheader("Bước 2 — Hàm mục tiêu")

    st.latex(r"""
    \max Z = \sum_i B_i y_i
    """)

    st.markdown("""
    `Bᵢ` là lợi ích NPV kỳ vọng của dự án `i`. Mục tiêu là chọn tập dự án làm tổng lợi ích NPV lớn nhất.
    """)

    st.subheader("Bước 3 — Hệ ràng buộc")

    constraint_df = pd.DataFrame({
        "Mã": ["C1", "C2", "C3", "C4", "C5", "C6a", "C6b", "C7"],
        "Công thức": [
            "Σ Cᵢyᵢ ≤ 80.000",
            "Σ C₁ᵢyᵢ ≤ 40.000",
            "y₁ + y₂ ≤ 1",
            "y₈ ≤ y₁₂",
            "y₁₃ ≤ y₁₂",
            "y₄ + y₅ ≥ 1",
            "y₁₄ ≥ 1",
            "7 ≤ Σyᵢ ≤ 11",
        ],
        "Ý nghĩa": [
            "Tổng chi phí 5 năm không vượt ngân sách.",
            "Chi phí giai đoạn năm 1-2 không vượt trần giải ngân sớm.",
            "Chỉ chọn một trung tâm dữ liệu quốc gia.",
            "Nếu chọn trung tâm AI quốc gia thì phải chọn đào tạo kỹ sư.",
            "Nếu chọn khu bán dẫn thì phải chọn đào tạo kỹ sư.",
            "Ít nhất một dự án chính phủ số.",
            "An ninh mạng quốc gia là dự án bắt buộc.",
            "Danh mục không quá ít và không quá dàn trải.",
        ],
    })

    st.dataframe(constraint_df, use_container_width=True)

    st.subheader("Bước 4 — Mở rộng có rủi ro")

    st.latex(r"""
    \max E[Z] = \sum_i p_i B_i y_i
    """)

    st.markdown("""
    Với mô hình rủi ro, `pᵢ` là xác suất hoàn thành đúng tiến độ. Khi đó, lợi ích được điều chỉnh theo rủi ro triển khai.
    Đây là cách làm thực tế hơn vì các dự án AI, bán dẫn hoặc hạ tầng lớn thường có rủi ro tiến độ cao.
    """)

    st.subheader("Bước 5 — Mở rộng hiệu ứng cộng hưởng")

    st.latex(r"""
    z_{8,13} \leq y_8,\quad z_{8,13} \leq y_{13},\quad
    z_{8,13} \geq y_8 + y_{13} - 1
    """)

    st.markdown("""
    Biến phụ `z₈,₁₃` bằng 1 nếu cả P8 và P13 cùng được chọn. Khi đó có thể cộng thêm bonus vào hàm mục tiêu
    để mô hình hóa hiệu ứng cộng hưởng giữa AI quốc gia và bán dẫn.
    """)

    st.success(
        "Tư duy mô hình: MIP giúp chọn danh mục dự án tối ưu trong điều kiện ngân sách, tiến độ, tiên quyết công nghệ "
        "và yêu cầu chính sách cùng tồn tại."
    )


# ---------------------------------------------------------
# 6. PHẦN 5.4 — GIẢI LẬP TRÌNH
# ---------------------------------------------------------
def show_programming_solution():
    st.header("5.4. Giải yêu cầu lập trình")

    st.markdown("""
    Phần này giải bài toán bằng PuLP/CBC, sau đó phân tích ba kịch bản:
    nới ngân sách lên 100.000 tỷ, yêu cầu chọn cả P1 và P2, và điều chỉnh rủi ro tiến độ.
    """)

    st.subheader("Thiết lập tham số mô hình")

    c1, c2, c3, c4 = st.columns(4)

    total_budget = c1.number_input(
        "Ngân sách tổng 5 năm, tỷ VND",
        min_value=50000,
        max_value=120000,
        value=80000,
        step=5000,
        key="bai5_total_budget",
    )

    early_budget = c2.number_input(
        "Ngân sách năm 1-2, tỷ VND",
        min_value=25000,
        max_value=70000,
        value=40000,
        step=2500,
        key="bai5_early_budget",
    )

    min_projects = c3.number_input(
        "Số dự án tối thiểu",
        min_value=1,
        max_value=15,
        value=7,
        step=1,
        key="bai5_min_projects",
    )

    max_projects = c4.number_input(
        "Số dự án tối đa",
        min_value=1,
        max_value=15,
        value=11,
        step=1,
        key="bai5_max_projects",
    )

    if min_projects > max_projects:
        st.error("Số dự án tối thiểu không được lớn hơn số dự án tối đa.")
        return

    # -----------------------------------------------------
    # 5.4.1 Bài toán gốc
    # -----------------------------------------------------
    st.subheader("Câu 5.4.1 — Giải mô hình gốc bằng PuLP/CBC")

    if not PULP_AVAILABLE:
        st.error("Chưa cài PuLP. Hãy thêm `pulp` vào requirements.txt.")
        return

    base = solve_project_mip(
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p14=True,
        keep_data_center_exclusion=True,
        force_both_data_centers=False,
        risk_adjusted=False,
        synergy_bonus=0,
    )

    if base is None or base["status"] != "Optimal":
        st.error(f"Mô hình gốc không tối ưu. Trạng thái: {base['status'] if base else 'Không chạy được'}")
        return

    s = base["summary"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trạng thái", base["status"])
    m2.metric("Z* NPV", f"{s['objective']:,.0f}", "tỷ VND")
    m3.metric("Tổng chi phí", f"{s['total_cost']:,.0f}", "tỷ VND")
    m4.metric("NPV/Chi phí", f"{s['npv_per_cost']:.2f}")

    st.markdown("#### Danh mục dự án được chọn")

    selected_display = base["selected_df"][[
        "code", "project_name", "field", "cost_total",
        "benefit_npv", "cost_year_1_2", "benefit_cost_ratio"
    ]].copy()

    selected_display.columns = [
        "Mã", "Tên dự án", "Lĩnh vực", "Chi phí 5 năm",
        "NPV lợi ích", "Chi phí năm 1-2", "NPV/Chi phí"
    ]

    st.dataframe(selected_display.round(3), use_container_width=True)

    st.markdown("#### Kiểm tra ràng buộc")

    st.dataframe(base["constraint_check"], use_container_width=True)

    fig_selected = px.bar(
        base["full_df"],
        x="code",
        y="benefit_npv",
        color=base["full_df"]["selected"].map({1: "Được chọn", 0: "Không chọn"}),
        hover_data=["project_name", "field", "cost_total"],
        title="Ảnh 5.6 — Dự án được chọn trong nghiệm tối ưu"
    )
    fig_selected.update_layout(
        xaxis_title="Mã dự án",
        yaxis_title="NPV lợi ích, tỷ VND",
        height=500,
        legend_title_text="Trạng thái"
    )
    st.plotly_chart(fig_selected, use_container_width=True)

    selected_field = base["selected_df"].groupby("field", as_index=False).agg(
        total_cost=("cost_total", "sum"),
        total_benefit=("benefit_npv", "sum"),
        n_projects=("id", "count")
    )

    fig_field = px.pie(
        selected_field,
        names="field",
        values="total_cost",
        title="Ảnh 5.7 — Cơ cấu chi phí danh mục tối ưu theo lĩnh vực",
        hole=0.42
    )
    fig_field.update_layout(height=480)
    st.plotly_chart(fig_field, use_container_width=True)

    # -----------------------------------------------------
    # 5.4.2 Nới ngân sách lên 100.000 tỷ
    # -----------------------------------------------------
    st.subheader("Câu 5.4.2 — Phân tích kịch bản nới ngân sách lên 100.000 tỷ")

    budget_100 = solve_project_mip(
        total_budget=100000,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p14=True,
        keep_data_center_exclusion=True,
        risk_adjusted=False,
    )

    scenario_rows = []

    for name, res in [
        ("Ngân sách gốc", base),
        ("Nới ngân sách 100.000", budget_100)
    ]:
        if res is not None and res["status"] == "Optimal":
            scenario_rows.append({
                "Kịch bản": name,
                "Z* NPV, tỷ VND": res["summary"]["objective"],
                "Tổng chi phí, tỷ VND": res["summary"]["total_cost"],
                "Chi phí năm 1-2": res["summary"]["total_early_cost"],
                "Số dự án": res["summary"]["number_selected"],
                "Danh mục chọn": res["summary"]["selected_codes"],
            })
        else:
            scenario_rows.append({
                "Kịch bản": name,
                "Z* NPV, tỷ VND": np.nan,
                "Tổng chi phí, tỷ VND": np.nan,
                "Chi phí năm 1-2": np.nan,
                "Số dự án": np.nan,
                "Danh mục chọn": "Không tối ưu",
            })

    scenario_df = pd.DataFrame(scenario_rows)
    st.dataframe(scenario_df, use_container_width=True)

    fig_budget = px.bar(
        scenario_df,
        x="Kịch bản",
        y="Z* NPV, tỷ VND",
        text="Z* NPV, tỷ VND",
        title="Ảnh 5.8 — Z* thay đổi khi nới ngân sách tổng"
    )
    fig_budget.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_budget.update_layout(height=450)
    st.plotly_chart(fig_budget, use_container_width=True)

    if budget_100 is not None and budget_100["status"] == "Optimal":
        base_set = set(base["selected_df"]["code"])
        new_set = set(budget_100["selected_df"]["code"])

        added = sorted(list(new_set - base_set))
        removed = sorted(list(base_set - new_set))

        st.info(
            f"Khi tăng ngân sách lên 100.000 tỷ VND, các dự án thêm mới là: "
            f"**{', '.join(added) if added else 'không có'}**. "
            f"Các dự án bị thay ra là: **{', '.join(removed) if removed else 'không có'}**."
        )

    # -----------------------------------------------------
    # 5.4.3 Quốc hội yêu cầu phải có cả P1 và P2
    # -----------------------------------------------------
    st.subheader("Câu 5.4.3 — Yêu cầu phải có cả P1 và P2")

    strict_both = solve_project_mip(
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p14=True,
        keep_data_center_exclusion=True,
        force_both_data_centers=True,
        risk_adjusted=False,
    )

    relaxed_both = solve_project_mip(
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p14=True,
        keep_data_center_exclusion=False,
        force_both_data_centers=True,
        risk_adjusted=False,
    )

    both_rows = []

    both_rows.append({
        "Kịch bản": "Giữ C3: y1 + y2 ≤ 1 và ép chọn P1, P2",
        "Trạng thái": strict_both["status"] if strict_both else "Không chạy",
        "Z* NPV": strict_both["summary"]["objective"] if strict_both and strict_both["status"] == "Optimal" else np.nan,
        "Ghi chú": "Mâu thuẫn logic nếu vừa loại trừ vừa bắt buộc cả hai."
    })

    both_rows.append({
        "Kịch bản": "Thay C3 bằng yêu cầu redundancy: chọn cả P1 và P2",
        "Trạng thái": relaxed_both["status"] if relaxed_both else "Không chạy",
        "Z* NPV": relaxed_both["summary"]["objective"] if relaxed_both and relaxed_both["status"] == "Optimal" else np.nan,
        "Ghi chú": "Kiểm tra khả thi nếu chính sách thay đổi."
    })

    both_df = pd.DataFrame(both_rows)
    st.dataframe(both_df, use_container_width=True)

    if relaxed_both is not None and relaxed_both["status"] == "Optimal":
        delta_z = relaxed_both["summary"]["objective"] - base["summary"]["objective"]
        st.success(
            f"Nếu thay ràng buộc loại trừ bằng yêu cầu redundancy, bài toán vẫn khả thi. "
            f"Z* thay đổi {delta_z:,.0f} tỷ VND so với mô hình gốc."
        )
        st.dataframe(
            relaxed_both["selected_df"][["code", "project_name", "field", "cost_total", "benefit_npv"]],
            use_container_width=True
        )
    else:
        st.warning(
            "Nếu vẫn giữ C3 y1 + y2 ≤ 1 mà lại bắt buộc chọn cả P1 và P2, mô hình chắc chắn không khả thi "
            "vì hai yêu cầu mâu thuẫn trực tiếp."
        )

    # -----------------------------------------------------
    # 5.4.4 Mở rộng rủi ro tiến độ
    # -----------------------------------------------------
    st.subheader("Câu 5.4.4 — Mô hình có rủi ro tiến độ")

    risk_case = solve_project_mip(
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p14=True,
        keep_data_center_exclusion=True,
        force_both_data_centers=False,
        risk_adjusted=True,
        synergy_bonus=0,
    )

    risk_rows = []

    for name, res in [
        ("Không điều chỉnh rủi ro", base),
        ("Điều chỉnh theo xác suất hoàn thành", risk_case)
    ]:
        if res is not None and res["status"] == "Optimal":
            risk_rows.append({
                "Kịch bản": name,
                "Giá trị mục tiêu": res["summary"]["objective"],
                "Tổng chi phí": res["summary"]["total_cost"],
                "Số dự án": res["summary"]["number_selected"],
                "Danh mục chọn": res["summary"]["selected_codes"],
            })

    risk_df = pd.DataFrame(risk_rows)
    st.dataframe(risk_df, use_container_width=True)

    if risk_case is not None and risk_case["status"] == "Optimal":
        base_set = set(base["selected_df"]["code"])
        risk_set = set(risk_case["selected_df"]["code"])

        added_risk = sorted(list(risk_set - base_set))
        removed_risk = sorted(list(base_set - risk_set))

        st.info(
            f"Khi tính rủi ro tiến độ, dự án thêm vào: **{', '.join(added_risk) if added_risk else 'không có'}**; "
            f"dự án bị loại ra: **{', '.join(removed_risk) if removed_risk else 'không có'}**. "
            "Điều này cho thấy danh mục tối ưu có thể thay đổi khi xét khả năng thực thi thay vì chỉ xét NPV danh nghĩa."
        )

        risk_plot = risk_case["full_df"].copy()
        risk_plot["selected_label"] = risk_plot["selected"].map({1: "Được chọn", 0: "Không chọn"})

        fig_risk = px.scatter(
            risk_plot,
            x="benefit_npv",
            y="expected_benefit",
            size="cost_total",
            color="selected_label",
            hover_name="code",
            hover_data=["project_name", "field", "completion_probability"],
            title="Ảnh 5.9 — NPV danh nghĩa và lợi ích kỳ vọng sau điều chỉnh rủi ro"
        )
        fig_risk.update_layout(
            xaxis_title="NPV danh nghĩa, tỷ VND",
            yaxis_title="Lợi ích kỳ vọng có rủi ro, tỷ VND",
            height=520,
            legend_title_text="Trạng thái"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    return {
        "base": base,
        "budget_100": budget_100,
        "strict_both": strict_both,
        "relaxed_both": relaxed_both,
        "risk_case": risk_case,
    }


# ---------------------------------------------------------
# 7. PHẦN 5.5 — THẢO LUẬN CHÍNH SÁCH
# ---------------------------------------------------------
def show_policy_discussion():
    st.header("5.5. Câu hỏi thảo luận chính sách")

    if not PULP_AVAILABLE:
        st.error("Cần cài PuLP để chạy phần thảo luận chính sách.")
        return

    base = solve_project_mip(
        total_budget=80000,
        early_budget=40000,
        min_projects=7,
        max_projects=11,
        force_p14=True,
        keep_data_center_exclusion=True,
        risk_adjusted=False,
    )

    no_p14 = solve_project_mip(
        total_budget=80000,
        early_budget=40000,
        min_projects=7,
        max_projects=11,
        force_p14=False,
        keep_data_center_exclusion=True,
        risk_adjusted=False,
    )

    synergy_case = solve_project_mip(
        total_budget=80000,
        early_budget=40000,
        min_projects=7,
        max_projects=11,
        force_p14=True,
        keep_data_center_exclusion=True,
        risk_adjusted=False,
        synergy_bonus=5000,
    )

    if base is None or base["status"] != "Optimal":
        st.error("Mô hình gốc không tối ưu nên không thể thảo luận.")
        return

    df = get_project_data()
    selected_codes = set(base["selected_df"]["code"])

    # -----------------------------------------------------
    # Câu a
    # -----------------------------------------------------
    st.subheader("a) Vì sao mô hình có thể bỏ qua P15 dù tỷ suất lợi ích/chi phí rất cao?")

    p15 = df[df["code"] == "P15"].iloc[0]
    p15_selected = "P15" in selected_codes

    p15_compare = df[["code", "project_name", "field", "cost_total", "benefit_npv", "benefit_cost_ratio"]].copy()
    p15_compare["selected"] = p15_compare["code"].apply(lambda x: "Được chọn" if x in selected_codes else "Không chọn")
    p15_compare = p15_compare.sort_values("benefit_cost_ratio", ascending=False)

    st.dataframe(p15_compare.round(3), use_container_width=True)

    fig_p15 = px.bar(
        p15_compare,
        x="benefit_cost_ratio",
        y="code",
        color="selected",
        orientation="h",
        hover_data=["project_name", "field"],
        title="Minh chứng câu a — P15 có tỷ suất cao nhưng quyết định chọn phụ thuộc toàn bộ danh mục"
    )
    fig_p15.update_traces(texttemplate="%{x:.2f}", textposition="outside")
    fig_p15.update_layout(height=560, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_p15, use_container_width=True)

    if p15_selected:
        st.success(
            f"Trong nghiệm hiện tại, P15 **được chọn**. Điều này hợp lý vì P15 có chi phí thấp "
            f"({p15['cost_total']:,.0f} tỷ VND) và tỷ suất NPV/chi phí cao ({p15['benefit_cost_ratio']:.2f}). "
            "Tuy nhiên, câu hỏi vẫn có ý nghĩa chính sách: nếu trong một kịch bản khác P15 bị loại, nguyên nhân không phải vì P15 kém hiệu quả, "
            "mà vì mô hình MIP tối ưu cả danh mục dưới ràng buộc ngân sách, số lượng dự án, tiên quyết và an ninh mạng."
        )
    else:
        st.warning(
            f"P15 không được chọn dù có tỷ suất NPV/chi phí cao ({p15['benefit_cost_ratio']:.2f}). "
            "Lý do là MIP không xếp hạng dự án đơn lẻ theo tỷ suất, mà chọn tổ hợp dự án tạo tổng NPV cao nhất "
            "trong các ràng buộc ngân sách, số lượng dự án, dự án bắt buộc và quan hệ tiên quyết. "
            "Về chính sách, đây chưa chắc là kết quả mong muốn vì dữ liệu mở là nền tảng cho AI, chính phủ số và đổi mới sáng tạo."
        )

    st.info(
        "Liên hệ Việt Nam: với định hướng QĐ 749/QĐ-TTg và Nghị quyết 57-NQ/TW, Open Data không chỉ có NPV trực tiếp "
        "mà còn tạo hạ tầng dữ liệu cho AI và dịch vụ công. Vì vậy, nếu mô hình bỏ qua P15, nhà hoạch định chính sách có thể "
        "thêm ràng buộc bắt buộc chọn P15 hoặc mô hình hóa lợi ích lan tỏa của dữ liệu mở."
    )

    # -----------------------------------------------------
    # Câu b
    # -----------------------------------------------------
    st.subheader("b) Ràng buộc bắt buộc P14 có làm giảm Z* không? Có hợp lý không?")

    if no_p14 is None or no_p14["status"] != "Optimal":
        st.error("Mô hình không bắt buộc P14 không tối ưu, không thể so sánh.")
    else:
        compare_p14 = pd.DataFrame([
            {
                "Kịch bản": "Có bắt buộc P14",
                "Z* NPV": base["summary"]["objective"],
                "Tổng chi phí": base["summary"]["total_cost"],
                "Danh mục": base["summary"]["selected_codes"],
            },
            {
                "Kịch bản": "Không bắt buộc P14",
                "Z* NPV": no_p14["summary"]["objective"],
                "Tổng chi phí": no_p14["summary"]["total_cost"],
                "Danh mục": no_p14["summary"]["selected_codes"],
            },
        ])

        st.dataframe(compare_p14, use_container_width=True)

        delta_p14 = no_p14["summary"]["objective"] - base["summary"]["objective"]

        fig_p14 = px.bar(
            compare_p14,
            x="Kịch bản",
            y="Z* NPV",
            text="Z* NPV",
            title="Minh chứng câu b — Tác động của ràng buộc bắt buộc P14"
        )
        fig_p14.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_p14.update_layout(height=450)
        st.plotly_chart(fig_p14, use_container_width=True)

        if abs(delta_p14) < 1e-6:
            st.success(
                "Ràng buộc bắt buộc P14 **không làm giảm Z*** trong nghiệm hiện tại, vì P14 vẫn được mô hình chọn ngay cả khi không bắt buộc. "
                "Điều này cho thấy P14 vừa có ý nghĩa an ninh, vừa đủ hiệu quả về NPV trong danh mục."
            )
        else:
            st.warning(
                f"Ràng buộc bắt buộc P14 làm Z* giảm khoảng **{delta_p14:,.0f} tỷ VND** so với trường hợp không bắt buộc. "
                "Đây là chi phí cơ hội của an ninh mạng."
            )

        st.info(
            "Về chính sách, bắt buộc P14 vẫn hợp lý. Trong chuyển đổi số quốc gia, an ninh mạng là điều kiện nền tảng: "
            "nếu thiếu SOC quốc gia, các dự án dữ liệu, định danh số, y tế số, AI và dịch vụ công đều có rủi ro hệ thống. "
            "Do đó, P14 là dạng ràng buộc an toàn quốc gia, không nên chỉ đánh giá bằng NPV tài chính."
        )

    # -----------------------------------------------------
    # Câu c
    # -----------------------------------------------------
    st.subheader("c) Mô hình hóa hiệu ứng cộng hưởng giữa P8 và P13 như thế nào?")

    compare_synergy = pd.DataFrame([
        {
            "Kịch bản": "Mô hình gốc",
            "Z* NPV": base["summary"]["objective"],
            "Synergy bonus": 0,
            "P8 và P13 cùng chọn?": "Có" if base["summary"]["synergy_active"] == 1 else "Không",
            "Danh mục": base["summary"]["selected_codes"],
        },
        {
            "Kịch bản": "Có bonus cộng hưởng P8-P13",
            "Z* NPV": synergy_case["summary"]["objective"] if synergy_case and synergy_case["status"] == "Optimal" else np.nan,
            "Synergy bonus": 5000,
            "P8 và P13 cùng chọn?": "Có" if synergy_case and synergy_case["summary"]["synergy_active"] == 1 else "Không",
            "Danh mục": synergy_case["summary"]["selected_codes"] if synergy_case and synergy_case["status"] == "Optimal" else "Không tối ưu",
        },
    ])

    st.dataframe(compare_synergy, use_container_width=True)

    fig_syn = px.bar(
        compare_synergy,
        x="Kịch bản",
        y="Z* NPV",
        text="Z* NPV",
        title="Minh chứng câu c — Tác động của hiệu ứng cộng hưởng P8 và P13"
    )
    fig_syn.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_syn.update_layout(height=450)
    st.plotly_chart(fig_syn, use_container_width=True)

    st.latex(r"""
    z_{8,13} \leq y_8,\quad z_{8,13} \leq y_{13},\quad
    z_{8,13} \geq y_8 + y_{13} - 1
    """)

    st.latex(r"""
    \max Z = \sum_i B_i y_i + Bonus_{8,13}z_{8,13}
    """)

    st.success(
        "Cách mô hình hóa: thêm biến nhị phân phụ z₈,₁₃. Biến này bằng 1 khi cả P8 và P13 cùng được chọn, "
        "sau đó cộng thêm một khoản bonus vào hàm mục tiêu. Cách này phản ánh thực tế rằng AI quốc gia và bán dẫn "
        "không hoàn toàn độc lập: AI cần năng lực tính toán và chip, còn bán dẫn hưởng lợi từ hệ sinh thái AI."
    )

    st.markdown("""
    **Kết luận chính sách của Bài 5:**  
    MIP giúp lựa chọn danh mục dự án không chỉ theo lợi ích riêng lẻ, mà theo cấu trúc hệ thống. 
    Với Việt Nam, một danh mục chuyển đổi số tốt cần cân bằng giữa hạ tầng, dữ liệu, AI, nhân lực, an ninh mạng và khả năng giải ngân.
    Đây chính là đánh đổi chính sách giữa **tăng trưởng nhanh**, **an toàn hệ thống**, **năng lực thực thi** và **nền tảng dài hạn**.
    """)


# ---------------------------------------------------------
# 8. HÀM RENDER CHÍNH
# ---------------------------------------------------------
def render():
    st.title("🧩 Bài 5 — MIP lựa chọn dự án chuyển đổi số")

    st.markdown("""
    Bài 5 sử dụng **quy hoạch nguyên hỗn hợp — Mixed Integer Programming (MIP)** để lựa chọn danh mục dự án
    chuyển đổi số quốc gia. Mỗi dự án là một quyết định nhị phân: chọn hoặc không chọn.
    """)

    tabs = st.tabs([
        "5.1 Bối cảnh",
        "5.2 Dữ liệu dự án",
        "5.3 Mô hình toán học",
        "5.4 Giải lập trình",
        "5.5 Chính sách",
    ])

    with tabs[0]:
        show_context()

    with tabs[1]:
        show_project_data()

    with tabs[2]:
        show_math_model()

    with tabs[3]:
        show_programming_solution()

    with tabs[4]:
        show_policy_discussion()

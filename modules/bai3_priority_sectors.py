import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# BÀI 3 — TÍNH CHỈ SỐ ƯU TIÊN NGÀNH PRIORITY_i
# =========================================================


# ---------------------------------------------------------
# 1. ĐỌC VÀ TẠO DỮ LIỆU
# ---------------------------------------------------------
@st.cache_data
def load_sector_data():
    """
    Đọc file vietnam_sectors_2024.csv nếu có.
    Nếu không có file, dùng dữ liệu mặc định theo đề bài.
    """

    possible_paths = [
        "vietnam_sectors_2024.csv",
        "data/vietnam_sectors_2024.csv",
        "datasets/vietnam_sectors_2024.csv",
        "vietnam_sectors_2024(1).csv",
    ]

    csv_path = None
    for path in possible_paths:
        if Path(path).exists():
            csv_path = path
            break

    if csv_path:
        df = pd.read_csv(csv_path)

        # Chuẩn hóa tên cột nếu file CSV có tên cột khác nhau
        rename_map = {
            "sector": "sector_name_vi",
            "sector_name": "sector_name_vi",
            "growth": "growth_rate_2024_pct",
            "growth_pct": "growth_rate_2024_pct",
            "productivity": "productivity_million_VND_per_worker",
            "labor_productivity_million_VND": "productivity_million_VND_per_worker",
            "spillover": "spillover_coef_0_1",
            "export": "export_billion_USD",
            "employment": "labor_million",
            "ai_readiness": "ai_readiness_0_100",
            "risk": "automation_risk_pct",
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        required_cols = [
            "sector_name_vi",
            "growth_rate_2024_pct",
            "productivity_million_VND_per_worker",
            "spillover_coef_0_1",
            "export_billion_USD",
            "labor_million",
            "ai_readiness_0_100",
            "automation_risk_pct",
        ]

        if all(col in df.columns for col in required_cols):
            return df[required_cols].copy()

    # Dữ liệu mặc định theo đề bài
    df = pd.DataFrame({
        "sector_name_vi": [
            "Nông-Lâm-Thủy sản",
            "CN chế biến chế tạo",
            "Xây dựng",
            "Khai khoáng",
            "Bán buôn-bán lẻ",
            "Tài chính-Ngân hàng",
            "Logistics-Vận tải",
            "CNTT-Truyền thông",
            "Giáo dục-Đào tạo",
            "Y tế",
        ],
        "growth_rate_2024_pct": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
        "productivity_million_VND_per_worker": [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
        "spillover_coef_0_1": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
        "export_billion_USD": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
        "labor_million": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
        "ai_readiness_0_100": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
        "automation_risk_pct": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18],
    })

    return df


# ---------------------------------------------------------
# 2. HÀM CHUẨN HÓA VÀ TÍNH PRIORITY
# ---------------------------------------------------------
def norm_good(series):
    denominator = series.max() - series.min()
    if denominator == 0:
        return series * 0
    return (series - series.min()) / denominator


def norm_bad(series):
    denominator = series.max() - series.min()
    if denominator == 0:
        return series * 0
    return (series.max() - series) / denominator


def normalize_sector_data(df):
    """
    Chuẩn hóa 7 tiêu chí:
    - 6 tiêu chí tốt: càng cao càng tốt
    - Risk là tiêu chí xấu: càng thấp càng tốt, nên đảo chiều
    """

    cols_good = {
        "growth_rate_2024_pct": "Growth_norm",
        "productivity_million_VND_per_worker": "Productivity_norm",
        "spillover_coef_0_1": "Spillover_norm",
        "export_billion_USD": "Export_norm",
        "labor_million": "Employment_norm",
        "ai_readiness_0_100": "AIReadiness_norm",
    }

    col_bad = "automation_risk_pct"

    norm_df = df[["sector_name_vi"]].copy()

    for raw_col, norm_col in cols_good.items():
        norm_df[norm_col] = norm_good(df[raw_col])

    norm_df["Risk_norm_bad_reversed"] = norm_bad(df[col_bad])

    return norm_df


def get_default_weights():
    """
    Trọng số mặc định theo đề:
    a1=0.15; a2=0.15; a3=0.20; a4=0.15; a5=0.10; a6=0.20; a7=0.15
    """

    return {
        "Growth_norm": 0.15,
        "Productivity_norm": 0.15,
        "Spillover_norm": 0.20,
        "Export_norm": 0.15,
        "Employment_norm": 0.10,
        "AIReadiness_norm": 0.20,
        "Risk_norm_bad_reversed": 0.15,
    }


def calculate_priority(df, weights=None):
    if weights is None:
        weights = get_default_weights()

    norm_df = normalize_sector_data(df)

    result = df.copy()
    for col in norm_df.columns:
        if col != "sector_name_vi":
            result[col] = norm_df[col]

    result["Priority"] = 0
    for col, w in weights.items():
        result["Priority"] += result[col] * w

    result["Rank"] = result["Priority"].rank(ascending=False, method="dense").astype(int)
    result = result.sort_values("Priority", ascending=False).reset_index(drop=True)

    return result


def normalize_weights(weight_dict):
    total = sum(weight_dict.values())
    if total == 0:
        return weight_dict
    return {k: v / total for k, v in weight_dict.items()}


# ---------------------------------------------------------
# 3. PHÂN TÍCH ĐỘ NHẠY AI READINESS
# ---------------------------------------------------------
def ai_weight_sensitivity(df):
    base_weights = get_default_weights()
    ai_values = np.arange(0.05, 0.401, 0.05)

    rows = []
    rank_matrix = []

    for ai_w in ai_values:
        temp = base_weights.copy()
        temp["AIReadiness_norm"] = ai_w
        temp = normalize_weights(temp)

        result = calculate_priority(df, temp)

        top3 = result.head(3)["sector_name_vi"].tolist()

        rows.append({
            "Trọng số AI Readiness": round(ai_w, 2),
            "Top 1": top3[0],
            "Top 2": top3[1],
            "Top 3": top3[2],
            "Top-3": " | ".join(top3),
        })

        for _, row in result.iterrows():
            rank_matrix.append({
                "Trọng số AI Readiness": round(ai_w, 2),
                "Ngành": row["sector_name_vi"],
                "Xếp hạng": int(row["Rank"]),
                "Priority": row["Priority"],
            })

    return pd.DataFrame(rows), pd.DataFrame(rank_matrix)


# ---------------------------------------------------------
# 4. SO SÁNH HAI BỘ TRỌNG SỐ CHÍNH SÁCH
# ---------------------------------------------------------
def compare_policy_orientations(df):
    """
    Hai bộ trọng số:
    1. Định hướng tăng trưởng: ưu tiên Growth, Productivity, Export, AI.
    2. Định hướng bao trùm: ưu tiên Employment, Spillover, giảm Risk.
    """

    growth_weights = {
        "Growth_norm": 0.25,
        "Productivity_norm": 0.25,
        "Spillover_norm": 0.10,
        "Export_norm": 0.25,
        "Employment_norm": 0.05,
        "AIReadiness_norm": 0.10,
        "Risk_norm_bad_reversed": 0.00,
    }

    inclusive_weights = {
        "Growth_norm": 0.10,
        "Productivity_norm": 0.10,
        "Spillover_norm": 0.25,
        "Export_norm": 0.05,
        "Employment_norm": 0.25,
        "AIReadiness_norm": 0.10,
        "Risk_norm_bad_reversed": 0.15,
    }

    growth_result = calculate_priority(df, growth_weights)
    inclusive_result = calculate_priority(df, inclusive_weights)

    comparison = pd.DataFrame({
        "Ngành": df["sector_name_vi"],
    })

    comparison = comparison.merge(
        growth_result[["sector_name_vi", "Priority", "Rank"]],
        left_on="Ngành",
        right_on="sector_name_vi",
        how="left"
    ).drop(columns=["sector_name_vi"])

    comparison = comparison.rename(columns={
        "Priority": "Priority - Định hướng tăng trưởng",
        "Rank": "Rank - Định hướng tăng trưởng",
    })

    comparison = comparison.merge(
        inclusive_result[["sector_name_vi", "Priority", "Rank"]],
        left_on="Ngành",
        right_on="sector_name_vi",
        how="left"
    ).drop(columns=["sector_name_vi"])

    comparison = comparison.rename(columns={
        "Priority": "Priority - Định hướng bao trùm",
        "Rank": "Rank - Định hướng bao trùm",
    })

    comparison["Chênh lệch thứ hạng"] = (
        comparison["Rank - Định hướng bao trùm"] -
        comparison["Rank - Định hướng tăng trưởng"]
    )

    return growth_result, inclusive_result, comparison


# ---------------------------------------------------------
# 5. PHẦN 3.1 — BỐI CẢNH VIỆT NAM
# ---------------------------------------------------------
def show_context(df):
    st.header("3.1. Bối cảnh Việt Nam")

    st.markdown("""
    Việt Nam không thể chuyển đổi số và ứng dụng AI đồng thời với cùng cường độ ở tất cả các ngành.
    Vì vậy, cần một **chỉ số ưu tiên định lượng** để xác định ngành nào nên được thúc đẩy trước,
    ngành nào cần đi sau hoặc cần chính sách hỗ trợ riêng.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Số ngành phân tích", "10 ngành")
    c2.metric("Số tiêu chí đánh giá", "7 tiêu chí")
    c3.metric("Mục tiêu", "Xếp hạng ưu tiên AI & CĐS")

    gdp_structure = pd.DataFrame({
        "Khu vực kinh tế": [
            "Nông-lâm-thủy sản",
            "Công nghiệp - xây dựng",
            "Dịch vụ",
            "Khác / thuế sản phẩm"
        ],
        "Tỷ trọng GDP 2024, %": [11.86, 37.64, 42.36, 8.14],
    })

    st.subheader("Ảnh 3.1 — Cơ cấu kinh tế Việt Nam 2024")

    fig_gdp = px.pie(
        gdp_structure,
        names="Khu vực kinh tế",
        values="Tỷ trọng GDP 2024, %",
        hole=0.42,
        title="Cơ cấu GDP 2024: nền kinh tế cần lựa chọn ngành ưu tiên chuyển đổi số"
    )
    fig_gdp.update_layout(height=480)
    st.plotly_chart(fig_gdp, use_container_width=True)

    st.markdown("""
    **Câu chuyện chính sách:** công nghiệp - xây dựng và dịch vụ chiếm tỷ trọng lớn trong GDP,
    nhưng không phải ngành nào cũng có cùng mức độ sẵn sàng AI, cùng khả năng lan tỏa hoặc cùng rủi ro tự động hóa.
    Vì vậy, bài toán cần chuyển từ cảm tính sang một chỉ số định lượng.
    """)

    st.subheader("Ảnh 3.2 — Bản đồ tư duy lựa chọn ngành ưu tiên AI")

    labels = [
        "10 ngành kinh tế Việt Nam",
        "Tăng trưởng",
        "Năng suất",
        "Lan tỏa",
        "Xuất khẩu",
        "Việc làm",
        "AI Readiness",
        "Rủi ro tự động hóa",
        "Priorityᵢ",
        "Xếp hạng ưu tiên"
    ]

    source = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    target = [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 9]
    value = [1, 1, 1, 1, 1, 1, 1, 0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15, 1]

    fig_flow = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            label=labels,
            line=dict(color="black", width=0.3),
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
        )
    )])

    fig_flow.update_layout(
        title="Ảnh 3.2 — Từ dữ liệu ngành đến chỉ số Priorityᵢ",
        height=520
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    st.info(
        "Thông điệp bối cảnh: ngành được ưu tiên không nhất thiết là ngành có một chỉ tiêu cao nhất. "
        "Ngành ưu tiên phải cân bằng giữa tăng trưởng, năng suất, lan tỏa, xuất khẩu, việc làm, mức sẵn sàng AI "
        "và rủi ro tự động hóa."
    )


# ---------------------------------------------------------
# 6. PHẦN 3.2 — MÔ HÌNH TOÁN HỌC
# ---------------------------------------------------------
def show_math_model():
    st.header("3.2. Mô hình toán học")

    st.markdown("""
    Mô hình xây dựng một điểm số tổng hợp cho từng ngành. Mỗi ngành được đánh giá bằng 7 tiêu chí.
    Trước khi cộng điểm, các tiêu chí phải được chuẩn hóa về cùng thang đo `[0, 1]`.
    """)

    st.subheader("Bước 1 — Công thức chỉ số ưu tiên ngành")

    st.latex(r"""
    Priority_i =
    a_1 Growth_i
    + a_2 Productivity_i
    + a_3 Spillover_i
    + a_4 Export_i
    + a_5 Employment_i
    + a_6 AIReadiness_i
    + a_7 RiskReversed_i
    """)

    st.markdown("""
    Trong code, rủi ro tự động hóa được **đảo chiều** thành `RiskReversed`.
    Nghĩa là ngành có rủi ro thấp sẽ nhận điểm cao hơn, vì phù hợp hơn với mục tiêu chuyển đổi số bao trùm.
    """)

    st.subheader("Bước 2 — Chuẩn hóa tiêu chí tốt")

    st.latex(r"""
    \tilde{x}_i =
    \frac{x_i - \min(x)}
    {\max(x) - \min(x)}
    """)

    st.markdown("""
    Công thức này dùng cho các tiêu chí càng cao càng tốt như tăng trưởng, năng suất, lan tỏa, xuất khẩu,
    việc làm và AI Readiness.
    """)

    st.subheader("Bước 3 — Chuẩn hóa tiêu chí xấu")

    st.latex(r"""
    \tilde{x}^{risk}_i =
    \frac{\max(x) - x_i}
    {\max(x) - \min(x)}
    """)

    st.markdown("""
    Công thức này dùng cho `Risk`. Nếu ngành có rủi ro tự động hóa thấp, điểm chuẩn hóa sau đảo chiều sẽ cao.
    """)

    weights = get_default_weights()

    weight_table = pd.DataFrame({
        "Tiêu chí chuẩn hóa": list(weights.keys()),
        "Trọng số mặc định": list(weights.values()),
        "Diễn giải": [
            "Tốc độ tăng trưởng ngành",
            "Năng suất lao động",
            "Khả năng lan tỏa sang ngành khác",
            "Năng lực xuất khẩu",
            "Quy mô việc làm",
            "Mức độ sẵn sàng AI",
            "Rủi ro tự động hóa sau đảo chiều",
        ]
    })

    st.subheader("Bước 4 — Bộ trọng số mặc định")

    st.dataframe(weight_table, use_container_width=True)

    fig_w = px.bar(
        weight_table,
        x="Tiêu chí chuẩn hóa",
        y="Trọng số mặc định",
        text="Trọng số mặc định",
        title="Ảnh 3.3 — Trọng số mặc định trong chỉ số Priorityᵢ"
    )
    fig_w.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_w.update_layout(height=450)
    st.plotly_chart(fig_w, use_container_width=True)

    st.success(
        "Tư duy mô hình: Priorityᵢ không phải là một chỉ tiêu đơn lẻ, mà là chỉ số tổng hợp. "
        "Kết quả phụ thuộc vào dữ liệu chuẩn hóa và bộ trọng số chính sách."
    )


# ---------------------------------------------------------
# 7. PHẦN 3.3 — DỮ LIỆU 10 NGÀNH
# ---------------------------------------------------------
def show_data(df):
    st.header("3.3. Dữ liệu 10 ngành Việt Nam 2024")

    display_df = df.copy()
    display_df.columns = [
        "Ngành",
        "Tăng trưởng 2024, %",
        "Năng suất, triệu VND/LĐ",
        "Lan tỏa, 0-1",
        "Xuất khẩu, tỷ USD",
        "Việc làm, triệu LĐ",
        "AI Readiness, 0-100",
        "Rủi ro tự động hóa, %",
    ]

    st.dataframe(display_df, use_container_width=True)

    st.subheader("Ảnh 3.4 — So sánh tăng trưởng và AI Readiness")

    fig_scatter = px.scatter(
        df,
        x="ai_readiness_0_100",
        y="growth_rate_2024_pct",
        size="export_billion_USD",
        color="automation_risk_pct",
        hover_name="sector_name_vi",
        title="Ngành tăng trưởng cao chưa chắc đã có AI Readiness cao",
        labels={
            "ai_readiness_0_100": "AI Readiness",
            "growth_rate_2024_pct": "Tăng trưởng 2024, %",
            "export_billion_USD": "Xuất khẩu, tỷ USD",
            "automation_risk_pct": "Rủi ro tự động hóa, %",
        }
    )
    fig_scatter.update_layout(height=520)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Ảnh 3.5 — Hồ sơ dữ liệu của 10 ngành")

    radar_cols = [
        "growth_rate_2024_pct",
        "productivity_million_VND_per_worker",
        "spillover_coef_0_1",
        "export_billion_USD",
        "labor_million",
        "ai_readiness_0_100",
    ]

    radar_norm = df[["sector_name_vi"]].copy()
    for col in radar_cols:
        radar_norm[col] = norm_good(df[col])

    radar_long = radar_norm.melt(
        id_vars="sector_name_vi",
        var_name="Tiêu chí",
        value_name="Điểm chuẩn hóa"
    )

    fig_profile = px.line_polar(
        radar_long,
        r="Điểm chuẩn hóa",
        theta="Tiêu chí",
        color="sector_name_vi",
        line_close=True,
        title="Ảnh 3.5 — Radar profile: mỗi ngành mạnh/yếu ở tiêu chí nào"
    )
    fig_profile.update_layout(height=620)
    st.plotly_chart(fig_profile, use_container_width=True)

    st.subheader("Ảnh 3.6 — Heatmap dữ liệu gốc sau chuẩn hóa")

    norm_df = normalize_sector_data(df)
    heat = norm_df.set_index("sector_name_vi")

    fig_heat = px.imshow(
        heat,
        text_auto=".2f",
        aspect="auto",
        title="Ma trận chuẩn hóa min-max của 10 ngành"
    )
    fig_heat.update_layout(height=580)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info(
        "Cách đọc dữ liệu: một ngành có thể rất mạnh ở năng suất nhưng yếu ở tăng trưởng, việc làm hoặc rủi ro. "
        "Do đó, cần tính Priorityᵢ thay vì chỉ nhìn một cột riêng lẻ."
    )


# ---------------------------------------------------------
# 8. PHẦN 3.4 — GIẢI YÊU CẦU LẬP TRÌNH
# ---------------------------------------------------------
def show_programming_solution(df):
    st.header("3.4. Giải yêu cầu lập trình")

    st.markdown("""
    Phần này giải 4 yêu cầu: chuẩn hóa dữ liệu, tính Priorityᵢ, phân tích độ nhạy trọng số AI Readiness,
    và so sánh hai bộ trọng số chính sách.
    """)

    # -----------------------------------------------------
    # Câu 3.4.1
    # -----------------------------------------------------
    st.subheader("Câu 3.4.1 — Chuẩn hóa min-max 7 tiêu chí")

    norm_df = normalize_sector_data(df)

    st.dataframe(norm_df.round(3), use_container_width=True)

    fig_norm = px.imshow(
        norm_df.set_index("sector_name_vi"),
        text_auto=".2f",
        aspect="auto",
        title="Ảnh 3.7 — Ma trận chuẩn hóa dùng để tính Priorityᵢ"
    )
    fig_norm.update_layout(height=580)
    st.plotly_chart(fig_norm, use_container_width=True)

    st.info(
        "Risk được đảo chiều: ngành có rủi ro tự động hóa càng thấp thì điểm Risk sau chuẩn hóa càng cao. "
        "Điều này giúp chỉ số Priorityᵢ tránh ưu tiên quá mức các ngành có nguy cơ thay thế lao động cao."
    )

    # -----------------------------------------------------
    # Câu 3.4.2
    # -----------------------------------------------------
    st.subheader("Câu 3.4.2 — Tính Priorityᵢ và xếp hạng 10 ngành")

    result = calculate_priority(df)

    result_display = result[[
        "sector_name_vi",
        "Priority",
        "Rank",
        "growth_rate_2024_pct",
        "productivity_million_VND_per_worker",
        "spillover_coef_0_1",
        "export_billion_USD",
        "labor_million",
        "ai_readiness_0_100",
        "automation_risk_pct",
    ]].copy()

    result_display.columns = [
        "Ngành",
        "Priorityᵢ",
        "Xếp hạng",
        "Tăng trưởng, %",
        "Năng suất",
        "Lan tỏa",
        "Xuất khẩu",
        "Việc làm",
        "AI Readiness",
        "Rủi ro TĐH, %",
    ]

    st.dataframe(result_display.round(3), use_container_width=True)

    top3 = result.head(3)

    c1, c2, c3 = st.columns(3)
    c1.metric("Top 1", top3.iloc[0]["sector_name_vi"], f"{top3.iloc[0]['Priority']:.3f}")
    c2.metric("Top 2", top3.iloc[1]["sector_name_vi"], f"{top3.iloc[1]['Priority']:.3f}")
    c3.metric("Top 3", top3.iloc[2]["sector_name_vi"], f"{top3.iloc[2]['Priority']:.3f}")

    fig_rank = px.bar(
        result,
        x="Priority",
        y="sector_name_vi",
        orientation="h",
        text="Priority",
        title="Ảnh 3.8 — Xếp hạng ngành theo chỉ số Priorityᵢ"
    )
    fig_rank.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_rank.update_layout(height=560, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_rank, use_container_width=True)

    # -----------------------------------------------------
    # Phân rã điểm Priority theo tiêu chí
    # -----------------------------------------------------
    st.markdown("#### Phân rã điểm Priorityᵢ theo từng tiêu chí")

    weights = get_default_weights()
    contrib = result[["sector_name_vi"]].copy()

    for col, w in weights.items():
        contrib[col] = result[col] * w

    contrib_long = contrib.melt(
        id_vars="sector_name_vi",
        var_name="Tiêu chí",
        value_name="Đóng góp vào Priority"
    )

    fig_contrib = px.bar(
        contrib_long,
        x="Đóng góp vào Priority",
        y="sector_name_vi",
        color="Tiêu chí",
        orientation="h",
        title="Ảnh 3.9 — Vì sao mỗi ngành có điểm Priorityᵢ cao/thấp?"
    )
    fig_contrib.update_layout(height=620, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_contrib, use_container_width=True)

    # -----------------------------------------------------
    # Câu 3.4.3
    # -----------------------------------------------------
    st.subheader("Câu 3.4.3 — Độ nhạy khi thay đổi trọng số AI Readiness")

    sens_table, rank_matrix = ai_weight_sensitivity(df)

    st.dataframe(sens_table, use_container_width=True)

    fig_sens_heat = px.imshow(
        rank_matrix.pivot(
            index="Ngành",
            columns="Trọng số AI Readiness",
            values="Xếp hạng"
        ),
        text_auto=True,
        aspect="auto",
        title="Ảnh 3.10 — Heatmap thứ hạng khi thay đổi trọng số AI Readiness"
    )
    fig_sens_heat.update_layout(height=620)
    st.plotly_chart(fig_sens_heat, use_container_width=True)

    fig_sens_line = px.line(
        rank_matrix,
        x="Trọng số AI Readiness",
        y="Priority",
        color="Ngành",
        markers=True,
        title="Ảnh 3.11 — Điểm Priority thay đổi khi trọng số AI Readiness tăng"
    )
    fig_sens_line.update_layout(height=560)
    st.plotly_chart(fig_sens_line, use_container_width=True)

    top3_unique = sens_table["Top-3"].nunique()

    if top3_unique == 1:
        st.success(
            "Kết quả độ nhạy: Top-3 ổn định khi thay đổi trọng số AI Readiness từ 0,05 đến 0,40. "
            "Điều này cho thấy kết quả xếp hạng khá vững."
        )
    else:
        st.warning(
            f"Kết quả độ nhạy: Top-3 thay đổi giữa {top3_unique} cấu hình khác nhau. "
            "Điều này cho thấy xếp hạng phụ thuộc đáng kể vào mức ưu tiên chính sách dành cho AI Readiness."
        )

    # -----------------------------------------------------
    # Câu 3.4.4
    # -----------------------------------------------------
    st.subheader("Câu 3.4.4 — So sánh định hướng tăng trưởng và định hướng bao trùm")

    growth_result, inclusive_result, comparison = compare_policy_orientations(df)

    st.markdown("#### Top-3 theo định hướng tăng trưởng")

    st.dataframe(
        growth_result.head(3)[["sector_name_vi", "Priority", "Rank"]].round(3),
        use_container_width=True
    )

    st.markdown("#### Top-3 theo định hướng bao trùm")

    st.dataframe(
        inclusive_result.head(3)[["sector_name_vi", "Priority", "Rank"]].round(3),
        use_container_width=True
    )

    st.markdown("#### Bảng so sánh toàn bộ 10 ngành")

    st.dataframe(comparison.round(3), use_container_width=True)

    comparison_long = comparison.melt(
        id_vars="Ngành",
        value_vars=[
            "Rank - Định hướng tăng trưởng",
            "Rank - Định hướng bao trùm",
        ],
        var_name="Bộ trọng số",
        value_name="Xếp hạng"
    )

    fig_compare = px.line(
        comparison_long,
        x="Bộ trọng số",
        y="Xếp hạng",
        color="Ngành",
        markers=True,
        title="Ảnh 3.12 — Thứ hạng ngành thay đổi theo định hướng chính sách"
    )
    fig_compare.update_layout(height=560, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_compare, use_container_width=True)

    st.info(
        "Diễn giải: bộ trọng số tăng trưởng thường ưu tiên ngành có năng suất, xuất khẩu và tăng trưởng cao. "
        "Bộ trọng số bao trùm lại ưu tiên ngành tạo nhiều việc làm, có lan tỏa xã hội và rủi ro tự động hóa thấp hơn."
    )

    return {
        "default_result": result,
        "norm_df": norm_df,
        "sensitivity": sens_table,
        "rank_matrix": rank_matrix,
        "growth_result": growth_result,
        "inclusive_result": inclusive_result,
        "comparison": comparison,
    }


# ---------------------------------------------------------
# 9. PHẦN 3.5 — THẢO LUẬN CHÍNH SÁCH
# ---------------------------------------------------------
def show_policy_discussion(df):
    st.header("3.5. Câu hỏi thảo luận chính sách")

    result = calculate_priority(df)
    growth_result, inclusive_result, comparison = compare_policy_orientations(df)

    top3 = result.head(3)
    khai_khoang = result[result["sector_name_vi"] == "Khai khoáng"].iloc[0]

    # -----------------------------------------------------
    # Câu a
    # -----------------------------------------------------
    st.subheader("a) Ba ngành nào nên được ưu tiên đẩy mạnh chuyển đổi số và AI trước?")

    top3_table = top3[[
        "sector_name_vi",
        "Priority",
        "Rank",
        "growth_rate_2024_pct",
        "spillover_coef_0_1",
        "export_billion_USD",
        "ai_readiness_0_100",
        "automation_risk_pct",
    ]].copy()

    top3_table.columns = [
        "Ngành",
        "Priorityᵢ",
        "Xếp hạng",
        "Tăng trưởng, %",
        "Lan tỏa",
        "Xuất khẩu, tỷ USD",
        "AI Readiness",
        "Rủi ro TĐH, %",
    ]

    st.dataframe(top3_table.round(3), use_container_width=True)

    fig_top3 = px.bar(
        top3,
        x="sector_name_vi",
        y="Priority",
        text="Priority",
        title="Minh chứng câu a — Top-3 ngành ưu tiên theo Priorityᵢ"
    )
    fig_top3.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_top3.update_layout(height=430)
    st.plotly_chart(fig_top3, use_container_width=True)

    top3_names = ", ".join(top3["sector_name_vi"].tolist())

    st.success(
        f"Trả lời: Theo bộ trọng số mặc định, ba ngành nên ưu tiên là **{top3_names}**. "
        "Các ngành này có điểm Priorityᵢ cao vì kết hợp được nhiều yếu tố: tăng trưởng, lan tỏa, xuất khẩu, "
        "AI Readiness hoặc vai trò trong chuỗi giá trị. Kết quả này phù hợp với tinh thần ưu tiên khoa học - công nghệ, "
        "đổi mới sáng tạo và chuyển đổi số, vì mô hình không chỉ chọn ngành lớn mà chọn ngành có khả năng tạo tác động lan tỏa."
    )

    st.info(
        "Lưu ý khi diễn giải với Nghị quyết 57-NQ/TW: nếu bài làm cần trích dẫn chính thức, bạn nên bổ sung nguồn văn bản "
        "trong phần thuyết minh. Trong mô hình này, sự phù hợp được hiểu ở góc độ định lượng: ưu tiên ngành có AI Readiness, "
        "lan tỏa và năng lực tạo giá trị cao."
    )

    # -----------------------------------------------------
    # Câu b
    # -----------------------------------------------------
    st.subheader("b) Vì sao Khai khoáng có năng suất rất cao nhưng không nằm trong nhóm ưu tiên?")

    khai_table = pd.DataFrame({
        "Tiêu chí": [
            "Priorityᵢ",
            "Xếp hạng",
            "Tăng trưởng, %",
            "Năng suất, triệu VND/LĐ",
            "Lan tỏa",
            "Xuất khẩu, tỷ USD",
            "Việc làm, triệu LĐ",
            "AI Readiness",
            "Rủi ro tự động hóa, %",
        ],
        "Giá trị của Khai khoáng": [
            khai_khoang["Priority"],
            khai_khoang["Rank"],
            khai_khoang["growth_rate_2024_pct"],
            khai_khoang["productivity_million_VND_per_worker"],
            khai_khoang["spillover_coef_0_1"],
            khai_khoang["export_billion_USD"],
            khai_khoang["labor_million"],
            khai_khoang["ai_readiness_0_100"],
            khai_khoang["automation_risk_pct"],
        ]
    })

    st.dataframe(khai_table.round(3), use_container_width=True)

    selected = result[result["sector_name_vi"].isin(["Khai khoáng"] + top3["sector_name_vi"].tolist())]

    compare_cols = [
        "sector_name_vi",
        "Productivity_norm",
        "Growth_norm",
        "Spillover_norm",
        "Export_norm",
        "Employment_norm",
        "AIReadiness_norm",
        "Risk_norm_bad_reversed",
    ]

    compare_long = selected[compare_cols].melt(
        id_vars="sector_name_vi",
        var_name="Tiêu chí",
        value_name="Điểm chuẩn hóa"
    )

    fig_khai = px.bar(
        compare_long,
        x="Tiêu chí",
        y="Điểm chuẩn hóa",
        color="sector_name_vi",
        barmode="group",
        title="Minh chứng câu b — Khai khoáng mạnh về năng suất nhưng yếu ở nhiều tiêu chí khác"
    )
    fig_khai.update_layout(height=540)
    st.plotly_chart(fig_khai, use_container_width=True)

    st.warning(
        f"Trả lời: Khai khoáng có năng suất rất cao, đạt **{khai_khoang['productivity_million_VND_per_worker']:.1f} triệu VND/LĐ**, "
        f"nhưng chỉ xếp hạng **{int(khai_khoang['Rank'])}** theo Priorityᵢ. Nguyên nhân là ngành này tăng trưởng âm "
        f"(**{khai_khoang['growth_rate_2024_pct']:.2f}%**), lan tỏa thấp, việc làm thấp, AI Readiness chưa cao và rủi ro tự động hóa cao "
        f"(**{khai_khoang['automation_risk_pct']:.0f}%**). Vì vậy, nếu chỉ nhìn năng suất sẽ dễ kết luận sai; chỉ số tổng hợp cho thấy "
        "Khai khoáng không phải lựa chọn ưu tiên hàng đầu cho chiến lược chuyển đổi số và AI theo hướng lan tỏa rộng."
    )

    # -----------------------------------------------------
    # Câu c
    # -----------------------------------------------------
    st.subheader("c) Bộ trọng số nên do ai quyết định?")

    st.markdown("""
    Câu hỏi này không chỉ là kỹ thuật, mà là vấn đề **governance**. Trọng số quyết định ngành nào được ưu tiên,
    nên nếu chỉ để một nhóm kỹ thuật quyết định thì kết quả có thể thiếu tính chính danh chính sách.
    """)

    governance = pd.DataFrame({
        "Chủ thể": [
            "Chuyên gia kỹ thuật",
            "Hội đồng chính sách",
            "Đối thoại công khai với doanh nghiệp, địa phương, người lao động",
        ],
        "Vai trò phù hợp": [
            "Thiết kế mô hình, kiểm tra dữ liệu, chuẩn hóa, phân tích độ nhạy.",
            "Chọn bộ trọng số chính thức, cân bằng mục tiêu tăng trưởng, bao trùm và an sinh.",
            "Phản biện tác động xã hội, rủi ro việc làm, khả năng thực thi và tính công bằng.",
        ],
        "Rủi ro nếu đứng một mình": [
            "Có thể quá thiên về chỉ tiêu đo được, thiếu góc nhìn xã hội.",
            "Có thể bị ảnh hưởng bởi ưu tiên nhiệm kỳ hoặc lợi ích bộ ngành.",
            "Có thể kéo dài quá trình ra quyết định nếu thiếu khung kỹ thuật rõ ràng.",
        ]
    })

    st.dataframe(governance, use_container_width=True)

    st.markdown("#### Minh chứng: cùng dữ liệu nhưng đổi trọng số thì thứ hạng thay đổi")

    compare_top = pd.DataFrame({
        "Định hướng": ["Mặc định", "Tăng trưởng", "Bao trùm"],
        "Top 1": [
            result.iloc[0]["sector_name_vi"],
            growth_result.iloc[0]["sector_name_vi"],
            inclusive_result.iloc[0]["sector_name_vi"],
        ],
        "Top 2": [
            result.iloc[1]["sector_name_vi"],
            growth_result.iloc[1]["sector_name_vi"],
            inclusive_result.iloc[1]["sector_name_vi"],
        ],
        "Top 3": [
            result.iloc[2]["sector_name_vi"],
            growth_result.iloc[2]["sector_name_vi"],
            inclusive_result.iloc[2]["sector_name_vi"],
        ],
    })

    st.dataframe(compare_top, use_container_width=True)

    compare_long = comparison.melt(
        id_vars="Ngành",
        value_vars=[
            "Rank - Định hướng tăng trưởng",
            "Rank - Định hướng bao trùm"
        ],
        var_name="Bộ trọng số",
        value_name="Xếp hạng"
    )

    fig_gov = px.line(
        compare_long,
        x="Bộ trọng số",
        y="Xếp hạng",
        color="Ngành",
        markers=True,
        title="Minh chứng câu c — Trọng số là lựa chọn chính sách, không chỉ là kỹ thuật"
    )
    fig_gov.update_layout(height=560, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_gov, use_container_width=True)

    st.success(
        "Trả lời: Bộ trọng số nên được quyết định theo mô hình ba lớp. "
        "Chuyên gia kỹ thuật xây dựng mô hình và kiểm định độ nhạy; hội đồng chính sách lựa chọn trọng số phù hợp mục tiêu phát triển; "
        "còn doanh nghiệp, địa phương và người lao động cần tham gia phản biện để bảo đảm tính chính danh. "
        "Cách làm này tốt hơn việc giao hoàn toàn cho một nhóm, vì trọng số ảnh hưởng trực tiếp đến phân bổ nguồn lực và lợi ích giữa các ngành."
    )


# ---------------------------------------------------------
# 10. HÀM RENDER CHÍNH
# ---------------------------------------------------------
def render():
    st.title("📊 Bài 3 — Tính chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành Việt Nam")

    st.markdown("""
    Bài 3 sử dụng phương pháp **ra quyết định đa tiêu chí đơn giản** để xếp hạng 10 ngành kinh tế Việt Nam.
    Mục tiêu là xác định ngành nào nên được ưu tiên đẩy mạnh chuyển đổi số và ứng dụng AI trước.
    """)

    df = load_sector_data()

    tabs = st.tabs([
        "3.1 Bối cảnh",
        "3.2 Mô hình",
        "3.3 Dữ liệu",
        "3.4 Lập trình",
        "3.5 Chính sách",
    ])

    with tabs[0]:
        show_context(df)

    with tabs[1]:
        show_math_model()

    with tabs[2]:
        show_data(df)

    with tabs[3]:
        show_programming_solution(df)

    with tabs[4]:
        show_policy_discussion(df)

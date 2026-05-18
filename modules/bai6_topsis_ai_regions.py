import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# BÀI 6 — TOPSIS XẾP HẠNG 6 VÙNG KINH TẾ VIỆT NAM
# THEO MỨC ĐỘ ƯU TIÊN ĐẦU TƯ AI
# =========================================================


# ---------------------------------------------------------
# 1. DỮ LIỆU GỐC
# ---------------------------------------------------------
def get_region_data():
    df = pd.DataFrame({
        "region_code": ["NMM", "RRD", "NCC", "CH", "SE", "MD"],
        "region_name_vi": [
            "Trung du miền núi phía Bắc",
            "Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ",
            "Tây Nguyên",
            "Đông Nam Bộ",
            "Đồng bằng sông Cửu Long",
        ],
        "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
        "fdi_registered_billion_USD": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
        "digital_index_0_100": [38, 78, 55, 32, 82, 48],
        "ai_readiness_0_100": [22, 68, 40, 18, 75, 30],
        "trained_labor_pct": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
        "rd_intensity_pct": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
        "internet_penetration_pct": [72, 92, 84, 68, 94, 78],
        "gini_coef": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
    })

    return df


def get_criteria_info():
    criteria = [
        "grdp_per_capita_million_VND",
        "fdi_registered_billion_USD",
        "digital_index_0_100",
        "ai_readiness_0_100",
        "trained_labor_pct",
        "rd_intensity_pct",
        "internet_penetration_pct",
        "gini_coef",
    ]

    labels = {
        "grdp_per_capita_million_VND": "GRDP/người",
        "fdi_registered_billion_USD": "FDI",
        "digital_index_0_100": "Digital Index",
        "ai_readiness_0_100": "AI Readiness",
        "trained_labor_pct": "LĐ qua đào tạo",
        "rd_intensity_pct": "R&D/GRDP",
        "internet_penetration_pct": "Internet",
        "gini_coef": "Gini",
    }

    units = {
        "grdp_per_capita_million_VND": "triệu VND/người",
        "fdi_registered_billion_USD": "tỷ USD",
        "digital_index_0_100": "0–100",
        "ai_readiness_0_100": "0–100",
        "trained_labor_pct": "%",
        "rd_intensity_pct": "%",
        "internet_penetration_pct": "%",
        "gini_coef": "hệ số",
    }

    is_benefit = np.array([True, True, True, True, True, True, True, False])

    expert_weights = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

    return criteria, labels, units, is_benefit, expert_weights


# ---------------------------------------------------------
# 2. HÀM TOPSIS, ENTROPY, AHP
# ---------------------------------------------------------
def vector_normalize(X):
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1, denom)
    return X / denom


def minmax_for_entropy(X, is_benefit):
    X_adj = X.copy().astype(float)

    for j in range(X.shape[1]):
        col = X[:, j].astype(float)
        col_min = col.min()
        col_max = col.max()
        denom = col_max - col_min

        if denom == 0:
            X_adj[:, j] = 1.0
        else:
            if is_benefit[j]:
                X_adj[:, j] = (col - col_min) / denom
            else:
                X_adj[:, j] = (col_max - col) / denom

    return X_adj + 1e-12


def entropy_weights(X, is_benefit):
    X_adj = minmax_for_entropy(X, is_benefit)
    P = X_adj / X_adj.sum(axis=0)
    k = 1.0 / np.log(X.shape[0])
    E = -k * np.sum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E

    if d.sum() == 0:
        return np.ones(X.shape[1]) / X.shape[1]

    return d / d.sum()


def topsis(df, weights, is_benefit):
    criteria, labels, units, _, _ = get_criteria_info()

    X = df[criteria].values.astype(float)

    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    R = vector_normalize(X)
    V = R * weights

    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))

    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    C_star = S_neg / (S_star + S_neg)

    result = df.copy()
    result["S_plus"] = S_star
    result["S_minus"] = S_neg
    result["TOPSIS_score"] = C_star
    result["Rank"] = result["TOPSIS_score"].rank(ascending=False, method="dense").astype(int)

    result = result.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)

    matrices = {
        "X": pd.DataFrame(X, columns=[labels[c] for c in criteria], index=df["region_name_vi"]),
        "R": pd.DataFrame(R, columns=[labels[c] for c in criteria], index=df["region_name_vi"]),
        "V": pd.DataFrame(V, columns=[labels[c] for c in criteria], index=df["region_name_vi"]),
        "A_star": pd.Series(A_star, index=[labels[c] for c in criteria]),
        "A_neg": pd.Series(A_neg, index=[labels[c] for c in criteria]),
    }

    return result, matrices


def ahp_weights_simple():
    """
    AHP đơn giản: dùng ma trận so sánh cặp được thiết kế nhất quán tương đối.
    Các tiêu chí về AI Readiness, Digital Index, R&D, lao động đào tạo được ưu tiên cao.
    """

    criteria, labels, units, is_benefit, expert_weights = get_criteria_info()

    # Vector ưu tiên giả định theo định hướng chiến lược AI
    base_priority = np.array([0.09, 0.08, 0.16, 0.24, 0.14, 0.15, 0.06, 0.08])
    base_priority = base_priority / base_priority.sum()

    # Ma trận so sánh cặp nhất quán: a_ij = w_i / w_j
    pairwise = base_priority[:, None] / base_priority[None, :]

    eigvals, eigvecs = np.linalg.eig(pairwise)
    max_idx = np.argmax(eigvals.real)
    weights = eigvecs[:, max_idx].real
    weights = np.abs(weights)
    weights = weights / weights.sum()

    n = len(criteria)
    lambda_max = eigvals[max_idx].real
    ci = (lambda_max - n) / (n - 1)

    # RI theo Saaty, n=8
    ri = 1.41
    cr = ci / ri if ri != 0 else 0

    pairwise_df = pd.DataFrame(
        pairwise,
        index=[labels[c] for c in criteria],
        columns=[labels[c] for c in criteria],
    )

    weight_df = pd.DataFrame({
        "Tiêu chí": [labels[c] for c in criteria],
        "AHP weight": weights,
    })

    return weights, pairwise_df, weight_df, ci, cr


def ai_weight_sensitivity(df):
    criteria, labels, units, is_benefit, expert_weights = get_criteria_info()

    ai_index = criteria.index("ai_readiness_0_100")
    ai_values = np.arange(0.10, 0.401, 0.05)

    rows = []
    rank_rows = []

    base_other = expert_weights.copy()

    for ai_w in ai_values:
        temp = base_other.copy()
        old_ai = temp[ai_index]
        temp[ai_index] = ai_w

        # Giữ tỷ trọng AI theo giá trị mới, các tiêu chí còn lại co giãn để tổng = 1
        other_sum_old = 1 - old_ai
        other_sum_new = 1 - ai_w

        for j in range(len(temp)):
            if j != ai_index:
                temp[j] = base_other[j] / other_sum_old * other_sum_new

        temp = temp / temp.sum()

        result, _ = topsis(df, temp, is_benefit)
        top3 = result.head(3)["region_name_vi"].tolist()

        rows.append({
            "w_AI": round(ai_w, 2),
            "Top 1": top3[0],
            "Top 2": top3[1],
            "Top 3": top3[2],
            "Top-3": " | ".join(top3),
        })

        for _, row in result.iterrows():
            rank_rows.append({
                "w_AI": round(ai_w, 2),
                "Vùng": row["region_name_vi"],
                "TOPSIS_score": row["TOPSIS_score"],
                "Rank": row["Rank"],
            })

    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


def compare_rankings(expert_result, entropy_result, ahp_result):
    comparison = expert_result[["region_name_vi", "TOPSIS_score", "Rank"]].copy()
    comparison = comparison.rename(columns={
        "TOPSIS_score": "Score chuyên gia",
        "Rank": "Rank chuyên gia",
    })

    comparison = comparison.merge(
        entropy_result[["region_name_vi", "TOPSIS_score", "Rank"]],
        on="region_name_vi",
        how="left"
    )

    comparison = comparison.rename(columns={
        "TOPSIS_score": "Score Entropy",
        "Rank": "Rank Entropy",
    })

    comparison = comparison.merge(
        ahp_result[["region_name_vi", "TOPSIS_score", "Rank"]],
        on="region_name_vi",
        how="left"
    )

    comparison = comparison.rename(columns={
        "TOPSIS_score": "Score AHP",
        "Rank": "Rank AHP",
    })

    comparison["Chênh lệch Rank Entropy - Chuyên gia"] = (
        comparison["Rank Entropy"] - comparison["Rank chuyên gia"]
    )

    comparison["Mức thay đổi tuyệt đối"] = comparison[
        "Chênh lệch Rank Entropy - Chuyên gia"
    ].abs()

    return comparison.sort_values("Rank chuyên gia")


# ---------------------------------------------------------
# 3. PHẦN 6.1 — BỐI CẢNH
# ---------------------------------------------------------
def show_context():
    st.header("6.1. Bối cảnh Việt Nam")

    st.markdown("""
    Việt Nam đặt mục tiêu phát triển mạnh nghiên cứu, ứng dụng và thương mại hóa AI đến năm 2030.
    Trong điều kiện ngân sách có hạn, không thể triển khai đồng loạt các trung tâm AI và sandbox dữ liệu
    ở tất cả các vùng cùng lúc. Vì vậy, cần một phương pháp định lượng để xác định vùng nào có mức độ
    sẵn sàng AI cao hơn và nên được ưu tiên trước.
    """)

    st.markdown("""
    Bài 6 sử dụng **TOPSIS** để xếp hạng 6 vùng kinh tế - xã hội theo mức độ ưu tiên đầu tư AI.
    TOPSIS phù hợp vì phương pháp này không chỉ nhìn vào một tiêu chí riêng lẻ như AI Readiness,
    mà đánh giá đồng thời nhiều tiêu chí: GRDP/người, FDI, Digital Index, AI Readiness, lao động qua đào tạo,
    R&D, Internet và Gini.
    """)

    df = get_region_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số vùng phân tích", "6")
    c2.metric("Số tiêu chí", "8")
    c3.metric("Tiêu chí lợi ích", "7")
    c4.metric("Tiêu chí chi phí", "1", "Gini")

    st.subheader("Ảnh 6.1 — Bức tranh sẵn sàng AI của 6 vùng")

    fig_scatter = px.scatter(
        df,
        x="digital_index_0_100",
        y="ai_readiness_0_100",
        size="fdi_registered_billion_USD",
        color="grdp_per_capita_million_VND",
        hover_name="region_name_vi",
        title="Digital Index, AI Readiness và FDI của 6 vùng",
        labels={
            "digital_index_0_100": "Digital Index",
            "ai_readiness_0_100": "AI Readiness",
            "fdi_registered_billion_USD": "FDI, tỷ USD",
            "grdp_per_capita_million_VND": "GRDP/người",
        }
    )
    fig_scatter.update_layout(height=520)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Ảnh 6.2 — Luồng đánh giá TOPSIS cho trung tâm AI vùng")

    labels = [
        "6 vùng kinh tế - xã hội",
        "8 tiêu chí đầu vào",
        "Chuẩn hóa vector",
        "Gắn trọng số",
        "Lý tưởng tốt A*",
        "Lý tưởng xấu A-",
        "Khoảng cách S+, S-",
        "Điểm C*",
        "Xếp hạng ưu tiên AI",
    ]

    fig_flow = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.3),
            label=labels,
        ),
        link=dict(
            source=[0, 1, 2, 3, 3, 4, 5, 6, 7],
            target=[1, 2, 3, 4, 5, 6, 6, 7, 8],
            value=[6, 8, 8, 4, 4, 4, 4, 6, 6],
        )
    )])
    fig_flow.update_layout(
        title="Ảnh 6.2 — Từ dữ liệu vùng đến xếp hạng ưu tiên AI bằng TOPSIS",
        height=520
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    st.info(
        "Thông điệp bối cảnh: vùng dẫn đầu về AI không nhất thiết chỉ là vùng có AI Readiness cao nhất. "
        "Một vùng được ưu tiên cần có nền tảng kinh tế, FDI, số hóa, nhân lực, R&D, Internet và mức cân bằng xã hội tương đối phù hợp."
    )


# ---------------------------------------------------------
# 4. PHẦN 6.2 — LÝ THUYẾT TOPSIS
# ---------------------------------------------------------
def show_topsis_theory():
    st.header("6.2. Lý thuyết TOPSIS")

    st.markdown("""
    TOPSIS là phương pháp ra quyết định đa tiêu chí. Ý tưởng cốt lõi là:
    phương án tốt nhất nên **gần lời giải lý tưởng dương** và **xa lời giải lý tưởng âm**.
    """)

    st.subheader("Bước 1 — Chuẩn hóa ma trận quyết định")

    st.latex(r"""
    r_{ij} = \frac{x_{ij}}{\sqrt{\sum_i x_{ij}^{2}}}
    """)

    st.markdown("""
    `xᵢⱼ` là giá trị gốc của vùng `i` theo tiêu chí `j`.  
    `rᵢⱼ` là giá trị sau chuẩn hóa vector.  
    Mục đích là đưa các tiêu chí khác đơn vị như tỷ USD, %, điểm số, hệ số Gini về cùng thang so sánh.
    """)

    st.subheader("Bước 2 — Ma trận chuẩn hóa có trọng số")

    st.latex(r"""
    v_{ij} = w_j \cdot r_{ij}
    """)

    st.markdown("""
    `wⱼ` là trọng số của tiêu chí `j`. Nếu AI Readiness quan trọng hơn, `w_AI` sẽ lớn hơn.
    Đây là điểm thể hiện quan điểm chính sách trong mô hình.
    """)

    st.subheader("Bước 3 — Lời giải lý tưởng dương và âm")

    st.latex(r"""
    A^{*} =
    \left\{
    \max_i v_{ij} \text{ nếu } j \text{ là tiêu chí lợi ích};
    \min_i v_{ij} \text{ nếu } j \text{ là tiêu chí chi phí}
    \right\}
    """)

    st.latex(r"""
    A^{-} =
    \left\{
    \min_i v_{ij} \text{ nếu } j \text{ là tiêu chí lợi ích};
    \max_i v_{ij} \text{ nếu } j \text{ là tiêu chí chi phí}
    \right\}
    """)

    st.markdown("""
    Với tiêu chí lợi ích như AI Readiness, Digital Index, Internet, giá trị càng cao càng tốt.  
    Với tiêu chí chi phí như Gini, giá trị càng thấp càng tốt vì phản ánh mức bất bình đẳng thấp hơn.
    """)

    st.subheader("Bước 4 — Khoảng cách Euclide đến A* và A-")

    st.latex(r"""
    S_i^{*} = \sqrt{\sum_j (v_{ij} - v_j^{*})^2}
    """)

    st.latex(r"""
    S_i^{-} = \sqrt{\sum_j (v_{ij} - v_j^{-})^2}
    """)

    st.markdown("""
    `Sᵢ*` càng nhỏ thì vùng `i` càng gần phương án lý tưởng tốt.  
    `Sᵢ⁻` càng lớn thì vùng `i` càng xa phương án lý tưởng xấu.
    """)

    st.subheader("Bước 5 — Hệ số gần gũi tương đối")

    st.latex(r"""
    C_i^{*} =
    \frac{S_i^{-}}{S_i^{*} + S_i^{-}},
    \quad 0 \leq C_i^{*} \leq 1
    """)

    st.markdown("""
    `Cᵢ*` càng cao thì vùng càng được ưu tiên.  
    Kết quả TOPSIS được xếp hạng theo `Cᵢ*` giảm dần.
    """)

    criteria, labels, units, is_benefit, expert_weights = get_criteria_info()

    theory_table = pd.DataFrame({
        "Tiêu chí": [labels[c] for c in criteria],
        "Đơn vị": [units[c] for c in criteria],
        "Loại tiêu chí": ["Lợi ích" if b else "Chi phí" for b in is_benefit],
        "Trọng số chuyên gia": expert_weights,
        "Ý nghĩa chính sách": [
            "Nền tảng kinh tế và sức mua vùng",
            "Khả năng thu hút vốn và công nghệ quốc tế",
            "Mức độ trưởng thành chuyển đổi số",
            "Mức sẵn sàng triển khai AI",
            "Nền tảng nhân lực cho AI",
            "Năng lực đổi mới sáng tạo",
            "Hạ tầng kết nối số",
            "Cân bằng xã hội; càng thấp càng tốt",
        ]
    })

    st.subheader("Bảng chú giải tiêu chí và trọng số chuyên gia")

    st.dataframe(theory_table, use_container_width=True)

    fig_w = px.bar(
        theory_table,
        x="Tiêu chí",
        y="Trọng số chuyên gia",
        color="Loại tiêu chí",
        text="Trọng số chuyên gia",
        title="Ảnh 6.3 — Trọng số chuyên gia dùng trong TOPSIS"
    )
    fig_w.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_w.update_layout(height=460)
    st.plotly_chart(fig_w, use_container_width=True)

    st.success(
        "Điểm mạnh của TOPSIS là minh bạch: có thể nhìn rõ dữ liệu đầu vào, trọng số, điểm C*, khoảng cách đến lý tưởng tốt/xấu "
        "và kiểm tra độ nhạy khi thay đổi trọng số chính sách."
    )


# ---------------------------------------------------------
# 5. PHẦN 6.3 — DỮ LIỆU
# ---------------------------------------------------------
def show_data():
    st.header("6.3. Dữ liệu 6 vùng kinh tế - xã hội Việt Nam")

    df = get_region_data()
    criteria, labels, units, is_benefit, expert_weights = get_criteria_info()

    display = df.copy()
    display = display.rename(columns={
        "region_name_vi": "Vùng",
        "grdp_per_capita_million_VND": "GRDP/người, tr.VND",
        "fdi_registered_billion_USD": "FDI, tỷ USD",
        "digital_index_0_100": "Digital Index",
        "ai_readiness_0_100": "AI Readiness",
        "trained_labor_pct": "LĐ qua đào tạo, %",
        "rd_intensity_pct": "R&D/GRDP, %",
        "internet_penetration_pct": "Internet, %",
        "gini_coef": "Gini",
    })

    st.dataframe(
        display[[
            "Vùng", "GRDP/người, tr.VND", "FDI, tỷ USD", "Digital Index",
            "AI Readiness", "LĐ qua đào tạo, %", "R&D/GRDP, %",
            "Internet, %", "Gini"
        ]],
        use_container_width=True
    )

    st.subheader("Ảnh 6.4 — Heatmap dữ liệu gốc theo vùng")

    heat_df = df.set_index("region_name_vi")[criteria]
    heat_df.columns = [labels[c] for c in criteria]

    fig_heat = px.imshow(
        heat_df,
        text_auto=True,
        aspect="auto",
        title="Dữ liệu gốc: mỗi vùng mạnh/yếu ở tiêu chí nào?"
    )
    fig_heat.update_layout(height=560)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Ảnh 6.5 — Radar profile chuẩn hóa min-max của từng vùng")

    X = df[criteria].values.astype(float)
    X_adj = minmax_for_entropy(X, is_benefit)

    radar_df = pd.DataFrame(X_adj, columns=[labels[c] for c in criteria])
    radar_df["Vùng"] = df["region_name_vi"]

    radar_long = radar_df.melt(
        id_vars="Vùng",
        var_name="Tiêu chí",
        value_name="Điểm chuẩn hóa 0-1"
    )

    fig_radar = px.line_polar(
        radar_long,
        r="Điểm chuẩn hóa 0-1",
        theta="Tiêu chí",
        color="Vùng",
        line_close=True,
        title="Radar profile: hồ sơ sẵn sàng AI theo vùng"
    )
    fig_radar.update_layout(height=640)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Ảnh 6.6 — So sánh AI Readiness và Internet penetration")

    fig_corr = px.scatter(
        df,
        x="internet_penetration_pct",
        y="ai_readiness_0_100",
        size="trained_labor_pct",
        color="digital_index_0_100",
        hover_name="region_name_vi",
        title="AI Readiness có thể đi cùng hạ tầng Internet và nhân lực số",
        labels={
            "internet_penetration_pct": "Internet, %",
            "ai_readiness_0_100": "AI Readiness",
            "trained_labor_pct": "LĐ qua đào tạo, %",
            "digital_index_0_100": "Digital Index",
        }
    )
    fig_corr.update_layout(height=520)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info(
        "Dữ liệu cho thấy Đông Nam Bộ và Đồng bằng sông Hồng có nền tảng AI mạnh hơn. "
        "Tây Nguyên và Trung du miền núi phía Bắc có điểm thấp hơn ở AI Readiness, Digital Index và R&D, "
        "nhưng vẫn cần được cân nhắc trong chính sách bao trùm."
    )


# ---------------------------------------------------------
# 6. PHẦN 6.4 — GIẢI LẬP TRÌNH
# ---------------------------------------------------------
def show_programming_solution():
    st.header("6.4. Giải yêu cầu lập trình")

    df = get_region_data()
    criteria, labels, units, is_benefit, expert_weights = get_criteria_info()

    # -----------------------------------------------------
    # 6.4.1 TOPSIS với trọng số chuyên gia
    # -----------------------------------------------------
    st.subheader("Câu 6.4.1 — TOPSIS với trọng số chuyên gia")

    expert_result, expert_matrices = topsis(df, expert_weights, is_benefit)

    display_expert = expert_result[[
        "region_name_vi", "TOPSIS_score", "S_plus", "S_minus", "Rank",
        "ai_readiness_0_100", "digital_index_0_100", "trained_labor_pct",
        "rd_intensity_pct", "gini_coef"
    ]].copy()

    display_expert.columns = [
        "Vùng", "Cᵢ* TOPSIS", "Sᵢ* đến A+", "Sᵢ- đến A-",
        "Xếp hạng", "AI Readiness", "Digital Index", "LĐ ĐT",
        "R&D/GRDP", "Gini"
    ]

    st.dataframe(display_expert.round(4), use_container_width=True)

    top1 = expert_result.iloc[0]
    top3 = expert_result.head(3)

    c1, c2, c3 = st.columns(3)
    c1.metric("Top 1", top1["region_name_vi"], f"C* = {top1['TOPSIS_score']:.3f}")
    c2.metric("Top 2", top3.iloc[1]["region_name_vi"], f"C* = {top3.iloc[1]['TOPSIS_score']:.3f}")
    c3.metric("Top 3", top3.iloc[2]["region_name_vi"], f"C* = {top3.iloc[2]['TOPSIS_score']:.3f}")

    fig_rank = px.bar(
        expert_result,
        x="TOPSIS_score",
        y="region_name_vi",
        orientation="h",
        text="TOPSIS_score",
        title="Ảnh 6.7 — Xếp hạng vùng theo TOPSIS với trọng số chuyên gia"
    )
    fig_rank.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_rank.update_layout(height=480, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("#### Ma trận chuẩn hóa có trọng số V")

    st.dataframe(expert_matrices["V"].round(4), use_container_width=True)

    fig_v = px.imshow(
        expert_matrices["V"],
        text_auto=".3f",
        aspect="auto",
        title="Ảnh 6.8 — Ma trận chuẩn hóa có trọng số V"
    )
    fig_v.update_layout(height=560)
    st.plotly_chart(fig_v, use_container_width=True)

    # -----------------------------------------------------
    # 6.4.2 Entropy weights
    # -----------------------------------------------------
    st.subheader("Câu 6.4.2 — TOPSIS với trọng số Entropy khách quan")

    X = df[criteria].values.astype(float)
    ent_w = entropy_weights(X, is_benefit)

    entropy_result, entropy_matrices = topsis(df, ent_w, is_benefit)

    weight_compare = pd.DataFrame({
        "Tiêu chí": [labels[c] for c in criteria],
        "Trọng số chuyên gia": expert_weights,
        "Trọng số Entropy": ent_w,
        "Chênh lệch Entropy - chuyên gia": ent_w - expert_weights,
    })

    st.dataframe(weight_compare.round(4), use_container_width=True)

    fig_weight_compare = px.bar(
        weight_compare.melt(
            id_vars="Tiêu chí",
            value_vars=["Trọng số chuyên gia", "Trọng số Entropy"],
            var_name="Bộ trọng số",
            value_name="Trọng số"
        ),
        x="Tiêu chí",
        y="Trọng số",
        color="Bộ trọng số",
        barmode="group",
        title="Ảnh 6.9 — So sánh trọng số chuyên gia và Entropy"
    )
    fig_weight_compare.update_layout(height=500)
    st.plotly_chart(fig_weight_compare, use_container_width=True)

    st.markdown("#### Kết quả TOPSIS dùng Entropy")

    display_entropy = entropy_result[["region_name_vi", "TOPSIS_score", "Rank"]].copy()
    display_entropy.columns = ["Vùng", "Cᵢ* Entropy", "Rank Entropy"]

    st.dataframe(display_entropy.round(4), use_container_width=True)

    # -----------------------------------------------------
    # 6.4.3 Độ nhạy AI weight
    # -----------------------------------------------------
    st.subheader("Câu 6.4.3 — Phân tích độ nhạy khi thay đổi w_AI")

    sens_table, rank_matrix = ai_weight_sensitivity(df)

    st.dataframe(sens_table, use_container_width=True)

    heat_rank = rank_matrix.pivot(index="Vùng", columns="w_AI", values="Rank")

    fig_sens_heat = px.imshow(
        heat_rank,
        text_auto=True,
        aspect="auto",
        title="Ảnh 6.10 — Heatmap thứ hạng khi thay đổi trọng số AI Readiness"
    )
    fig_sens_heat.update_layout(height=560)
    st.plotly_chart(fig_sens_heat, use_container_width=True)

    fig_sens_line = px.line(
        rank_matrix,
        x="w_AI",
        y="TOPSIS_score",
        color="Vùng",
        markers=True,
        title="Ảnh 6.11 — Điểm TOPSIS thay đổi khi tăng trọng số AI Readiness"
    )
    fig_sens_line.update_layout(height=540)
    st.plotly_chart(fig_sens_line, use_container_width=True)

    top3_unique = sens_table["Top-3"].nunique()

    if top3_unique == 1:
        st.success(
            "Top-3 ổn định khi thay đổi w_AI từ 0.10 đến 0.40. "
            "Điều này cho thấy kết quả ưu tiên đầu tư AI có độ vững cao."
        )
    else:
        st.warning(
            f"Top-3 thay đổi qua {top3_unique} cấu hình trọng số. "
            "Điều này cho thấy kết quả xếp hạng nhạy với quan điểm chính sách về AI Readiness."
        )

    # -----------------------------------------------------
    # 6.4.4 AHP đơn giản
    # -----------------------------------------------------
    st.subheader("Câu 6.4.4 — Mở rộng AHP đơn giản và so sánh với TOPSIS")

    ahp_w, pairwise_df, ahp_weight_df, ci, cr = ahp_weights_simple()
    ahp_result, ahp_matrices = topsis(df, ahp_w, is_benefit)

    st.markdown("#### Trọng số AHP")

    c4, c5 = st.columns(2)
    c4.metric("Consistency Index CI", f"{ci:.6f}")
    c5.metric("Consistency Ratio CR", f"{cr:.6f}")

    st.dataframe(ahp_weight_df.round(4), use_container_width=True)

    fig_ahp_w = px.bar(
        ahp_weight_df,
        x="Tiêu chí",
        y="AHP weight",
        text="AHP weight",
        title="Ảnh 6.12 — Trọng số AHP đơn giản"
    )
    fig_ahp_w.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_ahp_w.update_layout(height=460)
    st.plotly_chart(fig_ahp_w, use_container_width=True)

    st.markdown("#### Bảng so sánh xếp hạng chuyên gia, Entropy và AHP")

    comparison = compare_rankings(expert_result, entropy_result, ahp_result)

    st.dataframe(comparison.round(4), use_container_width=True)

    compare_long = comparison.melt(
        id_vars="region_name_vi",
        value_vars=["Rank chuyên gia", "Rank Entropy", "Rank AHP"],
        var_name="Phương pháp",
        value_name="Xếp hạng"
    )

    fig_compare = px.line(
        compare_long,
        x="Phương pháp",
        y="Xếp hạng",
        color="region_name_vi",
        markers=True,
        title="Ảnh 6.13 — Thứ hạng vùng thay đổi theo phương pháp trọng số"
    )
    fig_compare.update_layout(height=560, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_compare, use_container_width=True)

    st.info(
        "Entropy phản ánh mức độ phân tán khách quan của dữ liệu. AHP phản ánh đánh giá ưu tiên chiến lược. "
        "Nếu các phương pháp cho kết quả gần nhau, khuyến nghị chính sách có độ tin cậy cao hơn."
    )

    return {
        "expert_result": expert_result,
        "entropy_result": entropy_result,
        "ahp_result": ahp_result,
        "comparison": comparison,
        "sensitivity": sens_table,
        "rank_matrix": rank_matrix,
        "entropy_weights": ent_w,
        "ahp_weights": ahp_w,
    }


# ---------------------------------------------------------
# 7. PHẦN 6.5 — THẢO LUẬN CHÍNH SÁCH
# ---------------------------------------------------------
def show_policy_discussion():
    st.header("6.5. Câu hỏi thảo luận chính sách")

    df = get_region_data()
    criteria, labels, units, is_benefit, expert_weights = get_criteria_info()

    expert_result, _ = topsis(df, expert_weights, is_benefit)

    X = df[criteria].values.astype(float)
    ent_w = entropy_weights(X, is_benefit)
    entropy_result, _ = topsis(df, ent_w, is_benefit)

    ahp_w, pairwise_df, ahp_weight_df, ci, cr = ahp_weights_simple()
    ahp_result, _ = topsis(df, ahp_w, is_benefit)

    comparison = compare_rankings(expert_result, entropy_result, ahp_result)

    # -----------------------------------------------------
    # Câu a
    # -----------------------------------------------------
    st.subheader("a) Vùng nào dẫn đầu theo TOPSIS với trọng số chuyên gia?")

    top_region = expert_result.iloc[0]
    top3 = expert_result.head(3)

    st.dataframe(
        expert_result[["region_name_vi", "TOPSIS_score", "S_plus", "S_minus", "Rank"]].round(4),
        use_container_width=True
    )

    fig_a = px.bar(
        expert_result,
        x="TOPSIS_score",
        y="region_name_vi",
        orientation="h",
        text="TOPSIS_score",
        title="Minh chứng câu a — Vùng dẫn đầu theo TOPSIS chuyên gia"
    )
    fig_a.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_a.update_layout(height=480, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_a, use_container_width=True)

    st.success(
        f"Trả lời: vùng dẫn đầu theo TOPSIS với trọng số chuyên gia là **{top_region['region_name_vi']}**, "
        f"với điểm C* = **{top_region['TOPSIS_score']:.3f}**. "
        "Đây là ứng viên mạnh để triển khai trung tâm AI hoặc sandbox dữ liệu đầu tiên vì vùng này gần phương án lý tưởng tốt "
        "về nền tảng kinh tế, số hóa, AI Readiness, nhân lực và R&D."
    )

    st.info(
        "Tuy nhiên, quyết định triển khai trung tâm AI quốc gia không nên chỉ dựa vào TOPSIS. "
        "Theo tinh thần Quyết định 127/QĐ-TTg về chiến lược AI đến 2030, lựa chọn địa điểm còn cần xét kết nối vùng, "
        "an ninh dữ liệu, năng lực đại học - viện nghiên cứu, hạ tầng điện toán, đất đai, chi phí vận hành và vai trò lan tỏa quốc gia."
    )

        # -----------------------------------------------------
    # Câu b
    # -----------------------------------------------------
    st.subheader("b) Khi dùng Entropy, vùng nào thay đổi xếp hạng lớn nhất? Vì sao?")

    st.markdown("""
    Phần này so sánh xếp hạng theo **trọng số chuyên gia** và **trọng số Entropy**.
    Nếu mức thay đổi bằng 0 ở tất cả các vùng, điều đó cho thấy kết quả TOPSIS khá ổn định:
    Entropy làm thay đổi trọng số tiêu chí nhưng chưa đủ làm đảo thứ hạng vùng.
    """)

    # Bảng so sánh rank
    comparison_display = comparison[[
        "region_name_vi",
        "Score chuyên gia",
        "Rank chuyên gia",
        "Score Entropy",
        "Rank Entropy",
        "Chênh lệch Rank Entropy - Chuyên gia",
        "Mức thay đổi tuyệt đối"
    ]].copy()

    comparison_display.columns = [
        "Vùng",
        "Điểm TOPSIS - Chuyên gia",
        "Rank chuyên gia",
        "Điểm TOPSIS - Entropy",
        "Rank Entropy",
        "Chênh lệch rank",
        "Mức thay đổi tuyệt đối"
    ]

    st.dataframe(comparison_display.round(4), use_container_width=True)

    max_change = comparison["Mức thay đổi tuyệt đối"].max()

    if max_change == 0:
        st.success(
            "Kết quả: không có vùng nào thay đổi xếp hạng khi chuyển từ trọng số chuyên gia sang trọng số Entropy. "
            "Điều này cho thấy thứ hạng TOPSIS của 6 vùng tương đối vững, không phụ thuộc mạnh vào cách xác định trọng số."
        )

        # Biểu đồ thay thế: so sánh điểm TOPSIS thay vì vẽ cột toàn số 0
        score_compare = comparison[[
            "region_name_vi",
            "Score chuyên gia",
            "Score Entropy"
        ]].copy()

        score_long = score_compare.melt(
            id_vars="region_name_vi",
            value_vars=["Score chuyên gia", "Score Entropy"],
            var_name="Bộ trọng số",
            value_name="Điểm TOPSIS"
        )

        fig_b = px.bar(
            score_long,
            x="region_name_vi",
            y="Điểm TOPSIS",
            color="Bộ trọng số",
            barmode="group",
            text="Điểm TOPSIS",
            title="Minh chứng câu b — Điểm TOPSIS thay đổi nhưng thứ hạng không đổi"
        )

        fig_b.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_b.update_layout(
            height=500,
            xaxis_title="Vùng",
            yaxis_title="Điểm TOPSIS C*",
            legend_title_text="Bộ trọng số"
        )

        st.plotly_chart(fig_b, use_container_width=True)

        # Biểu đồ slope rank: thể hiện trực quan rank giữ nguyên
        rank_long = comparison[[
            "region_name_vi",
            "Rank chuyên gia",
            "Rank Entropy"
        ]].copy()

        rank_long = rank_long.melt(
            id_vars="region_name_vi",
            value_vars=["Rank chuyên gia", "Rank Entropy"],
            var_name="Phương pháp",
            value_name="Xếp hạng"
        )

        fig_rank_stable = px.line(
            rank_long,
            x="Phương pháp",
            y="Xếp hạng",
            color="region_name_vi",
            markers=True,
            title="Minh chứng bổ sung — Thứ hạng giữ ổn định giữa hai phương pháp trọng số"
        )

        fig_rank_stable.update_layout(
            height=520,
            yaxis=dict(autorange="reversed", dtick=1),
            xaxis_title="Phương pháp trọng số",
            yaxis_title="Xếp hạng TOPSIS",
            legend_title_text="Vùng"
        )

        st.plotly_chart(fig_rank_stable, use_container_width=True)

        st.info(
            "Diễn giải chính sách: khi cả trọng số chuyên gia và Entropy đều cho cùng thứ hạng, "
            "khuyến nghị lựa chọn vùng ưu tiên đầu tư AI có độ tin cậy cao hơn. "
            "Tuy nhiên, vẫn cần thận trọng vì Entropy chỉ phản ánh độ phân tán dữ liệu, không thay thế được ưu tiên chiến lược "
            "như cân bằng vùng, an ninh dữ liệu và vai trò lan tỏa quốc gia."
        )

    else:
        max_change_row = comparison.sort_values("Mức thay đổi tuyệt đối", ascending=False).iloc[0]

        fig_b = px.bar(
            comparison.sort_values("Mức thay đổi tuyệt đối", ascending=False),
            x="Mức thay đổi tuyệt đối",
            y="region_name_vi",
            orientation="h",
            text="Mức thay đổi tuyệt đối",
            title="Minh chứng câu b — Vùng thay đổi thứ hạng mạnh nhất khi dùng Entropy"
        )

        fig_b.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_b.update_layout(
            height=480,
            yaxis=dict(autorange="reversed"),
            xaxis_title="Số bậc thay đổi",
            yaxis_title="Vùng"
        )

        st.plotly_chart(fig_b, use_container_width=True)

        st.success(
            f"Trả lời: vùng thay đổi xếp hạng lớn nhất là **{max_change_row['region_name_vi']}**, "
            f"với mức thay đổi **{max_change_row['Mức thay đổi tuyệt đối']:.0f} bậc**. "
            "Nguyên nhân là Entropy tự gán trọng số cao hơn cho các tiêu chí có độ phân tán dữ liệu lớn. "
            "Vùng nào mạnh hoặc yếu rõ ở các tiêu chí được Entropy tăng trọng số sẽ thay đổi thứ hạng nhiều hơn."
        )

    # Bảng trọng số để giải thích vì sao Entropy có thể làm thay đổi hoặc không làm thay đổi rank
    weight_df = pd.DataFrame({
        "Tiêu chí": [labels[c] for c in criteria],
        "Trọng số chuyên gia": expert_weights,
        "Trọng số Entropy": ent_w,
        "Chênh lệch": ent_w - expert_weights,
    })

    st.markdown("#### Bảng giải thích trọng số")

    st.dataframe(weight_df.round(4), use_container_width=True)

    fig_weight_b = px.bar(
        weight_df.melt(
            id_vars="Tiêu chí",
            value_vars=["Trọng số chuyên gia", "Trọng số Entropy"],
            var_name="Bộ trọng số",
            value_name="Trọng số"
        ),
        x="Tiêu chí",
        y="Trọng số",
        color="Bộ trọng số",
        barmode="group",
        title="Vì sao kết quả Entropy có thể khác hoặc giống trọng số chuyên gia?"
    )

    fig_weight_b.update_layout(
        height=500,
        xaxis_title="Tiêu chí",
        yaxis_title="Trọng số",
        legend_title_text="Bộ trọng số"
    )

    st.plotly_chart(fig_weight_b, use_container_width=True)
    # Câu c
    # -----------------------------------------------------
    st.subheader("c) Tương quan giữa AI Readiness và Internet penetration ảnh hưởng thế nào?")

    corr_value = df["ai_readiness_0_100"].corr(df["internet_penetration_pct"])

    c1, c2 = st.columns(2)
    c1.metric("Tương quan AI Readiness - Internet", f"{corr_value:.3f}")
    c2.metric("Diễn giải", "Cao" if abs(corr_value) >= 0.7 else "Trung bình/thấp")

    fig_c = px.scatter(
        df,
        x="internet_penetration_pct",
        y="ai_readiness_0_100",
        text="region_code",
        size="digital_index_0_100",
        color="region_name_vi",
        title="Minh chứng câu c — Tương quan giữa Internet và AI Readiness",
        labels={
            "internet_penetration_pct": "Internet penetration, %",
            "ai_readiness_0_100": "AI Readiness",
            "digital_index_0_100": "Digital Index",
        }
    )
    fig_c.update_traces(textposition="top center")
    fig_c.update_layout(height=520)
    st.plotly_chart(fig_c, use_container_width=True)

    corr_matrix = df[[
        "digital_index_0_100",
        "ai_readiness_0_100",
        "trained_labor_pct",
        "rd_intensity_pct",
        "internet_penetration_pct",
    ]].corr()

    corr_matrix.columns = ["Digital", "AI", "LĐ ĐT", "R&D", "Internet"]
    corr_matrix.index = ["Digital", "AI", "LĐ ĐT", "R&D", "Internet"]

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Ma trận tương quan giữa các tiêu chí công nghệ"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.warning(
        "Trả lời: nếu AI Readiness và Internet penetration tương quan rất cao, TOPSIS có thể vô tình đếm hai lần cùng một năng lực nền tảng. "
        "Khi đó các vùng đã mạnh về hạ tầng số sẽ được cộng điểm lặp lại, làm kết quả nghiêng nhiều hơn về vùng phát triển."
    )

    st.info(
        "Cách xử lý: kiểm tra ma trận tương quan; gộp các tiêu chí tương quan cao thành một chỉ số hạ tầng số tổng hợp; "
        "giảm trọng số một trong hai tiêu chí; dùng PCA/factor analysis; hoặc chạy phân tích độ nhạy bỏ từng tiêu chí để xem xếp hạng có đổi mạnh không."
    )

    # -----------------------------------------------------
    # Câu d
    # -----------------------------------------------------
    st.subheader("d) Nếu chọn 3 trung tâm AI lớn, nên chọn vùng nào? Có cần điều chỉnh địa - chính trị không?")

    final_top3 = expert_result.head(3)[["region_name_vi", "TOPSIS_score", "Rank"]].copy()

    st.dataframe(final_top3.round(4), use_container_width=True)

    fig_d = px.bar(
        final_top3,
        x="region_name_vi",
        y="TOPSIS_score",
        text="TOPSIS_score",
        title="Minh chứng câu d — Ba vùng đề xuất cho trung tâm AI lớn theo TOPSIS"
    )
    fig_d.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_d.update_layout(height=460)
    st.plotly_chart(fig_d, use_container_width=True)

    top3_names = ", ".join(final_top3["region_name_vi"].tolist())

    st.success(
        f"Trả lời: nếu chỉ dựa trên TOPSIS với trọng số chuyên gia, có thể chọn 3 vùng: **{top3_names}**. "
        "Đây là các vùng có nền tảng tương đối tốt về kinh tế, FDI, số hóa, AI Readiness, nhân lực và R&D."
    )

    st.warning(
        "Tuy nhiên, với mục tiêu quốc gia theo Quyết định 127/QĐ-TTg là đưa Việt Nam trở thành trung tâm AI của ASEAN, "
        "việc chọn 3 trung tâm AI không nên chỉ tối đa hóa điểm sẵn sàng. Cần điều chỉnh thêm tiêu chí địa - chính trị "
        "và cân bằng vùng: miền Bắc, miền Nam, miền Trung/Tây Nguyên hoặc ĐBSCL có thể cần một vai trò chiến lược để bảo đảm lan tỏa quốc gia."
    )

    st.markdown("""
    **Kết luận chính sách của Bài 6:**  
    TOPSIS giúp đưa ra xếp hạng minh bạch và có thể kiểm tra nhạy cảm. Tuy nhiên, kết quả định lượng nên là
    đầu vào cho quyết định chính sách, không phải quyết định cuối cùng. Với chiến lược AI quốc gia, Việt Nam cần
    cân bằng giữa ba mục tiêu: **hiệu quả triển khai nhanh**, **lan tỏa vùng miền**, và **an ninh - chủ quyền dữ liệu**.
    """)


# ---------------------------------------------------------
# 8. HÀM RENDER CHÍNH
# ---------------------------------------------------------
def render():
    st.title("🤖 Bài 6 — TOPSIS xếp hạng 6 vùng ưu tiên đầu tư AI")

    st.markdown("""
    Bài 6 sử dụng **TOPSIS** và **Entropy weight** để xếp hạng 6 vùng kinh tế - xã hội Việt Nam
    theo mức độ ưu tiên đầu tư AI. Module này trình bày theo cấu trúc:
    **bối cảnh → lý thuyết → dữ liệu → tính toán → thảo luận chính sách**.
    """)

    tabs = st.tabs([
        "6.1 Bối cảnh",
        "6.2 Lý thuyết TOPSIS",
        "6.3 Dữ liệu",
        "6.4 Giải lập trình",
        "6.5 Chính sách",
    ])

    with tabs[0]:
        show_context()

    with tabs[1]:
        show_topsis_theory()

    with tabs[2]:
        show_data()

    with tabs[3]:
        show_programming_solution()

    with tabs[4]:
        show_policy_discussion()

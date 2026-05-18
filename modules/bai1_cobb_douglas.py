import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# BÀI 1 — HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG VỚI AI & SỐ HÓA
# Module Streamlit dùng cho app.py
# =========================================================


# -----------------------------
# 1. CẤU HÌNH GIAO DIỆN
# -----------------------------
COLOR_MAIN = "#16a34a"
COLOR_AI = "#2563eb"
COLOR_DIGITAL = "#9333ea"
COLOR_WARNING = "#f97316"


# -----------------------------
# 2. HÀM ĐỌC DỮ LIỆU
# -----------------------------
@st.cache_data
def load_macro_data():
    """
    Đọc file vietnam_macro_2020_2025.csv nếu có trong GitHub.
    Nếu không có, dùng dữ liệu mặc định theo đề bài.
    """

    possible_paths = [
        "vietnam_macro_2020_2025.csv",
        "data/vietnam_macro_2020_2025.csv",
        "datasets/vietnam_macro_2020_2025.csv",
        "vietnam_macro_2020_2025(1).csv",
    ]

    csv_path = None
    for path in possible_paths:
        if Path(path).exists():
            csv_path = path
            break

    if csv_path:
        macro = pd.read_csv(csv_path)
    else:
        macro = pd.DataFrame({
            "year": [2020, 2021, 2022, 2023, 2024, 2025],
            "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
            "GDP_growth_pct": [2.91, 2.58, 8.12, 5.05, 7.09, 8.02],
            "digital_economy_share_GDP_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
            "labor_productivity_million_VND": [151.2, 171.3, 188.1, 199.3, 221.9, 245.0],
        })

    # Dữ liệu đầu vào theo bảng 1.3 của đề bài.
    model_input = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "K": [16500, 17800, 19600, 21300, 23500, 25900],
        "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    })

    df = pd.merge(macro, model_input, on="year", how="right")

    if "GDP_trillion_VND" in df.columns:
        df["Y"] = df["GDP_trillion_VND"]
    else:
        df["Y"] = [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6]

    return df


# -----------------------------
# 3. HÀM TÍNH TOÁN CHÍNH
# -----------------------------
def calculate_model(df, alpha, beta, gamma, delta, theta):
    df = df.copy()

    df["A_TFP"] = df["Y"] / (
        df["K"] ** alpha *
        df["L"] ** beta *
        df["D"] ** gamma *
        df["AI"] ** delta *
        df["H"] ** theta
    )

    A_bar = df["A_TFP"].mean()

    df["Y_hat"] = A_bar * (
        df["K"] ** alpha *
        df["L"] ** beta *
        df["D"] ** gamma *
        df["AI"] ** delta *
        df["H"] ** theta
    )

    df["APE_pct"] = np.abs(df["Y"] - df["Y_hat"]) / df["Y"] * 100
    mape = df["APE_pct"].mean()

    return df, A_bar, mape


def calculate_growth_accounting(df, alpha, beta, gamma, delta, theta):
    """
    Phân rã tăng trưởng theo dạng log:
    ΔlnY = ΔlnA + αΔlnK + βΔlnL + γΔlnD + δΔlnAI + θΔlnH
    """

    ga = pd.DataFrame()
    ga["year"] = df["year"].iloc[1:].values

    ga["GDP_growth_log_pct"] = np.diff(np.log(df["Y"])) * 100
    ga["K_contribution_pct_point"] = alpha * np.diff(np.log(df["K"])) * 100
    ga["L_contribution_pct_point"] = beta * np.diff(np.log(df["L"])) * 100
    ga["D_contribution_pct_point"] = gamma * np.diff(np.log(df["D"])) * 100
    ga["AI_contribution_pct_point"] = delta * np.diff(np.log(df["AI"])) * 100
    ga["H_contribution_pct_point"] = theta * np.diff(np.log(df["H"])) * 100
    ga["TFP_contribution_pct_point"] = np.diff(np.log(df["A_TFP"])) * 100

    contribution_cols = [
        "K_contribution_pct_point",
        "L_contribution_pct_point",
        "D_contribution_pct_point",
        "AI_contribution_pct_point",
        "H_contribution_pct_point",
        "TFP_contribution_pct_point",
    ]

    avg_growth = ga["GDP_growth_log_pct"].mean()

    summary = pd.DataFrame({
        "Yếu tố": ["Vốn vật chất K", "Lao động L", "Số hóa D", "Năng lực AI", "Nhân lực số H", "TFP A"],
        "Đóng góp bình quân, điểm %": [ga[c].mean() for c in contribution_cols],
    })

    summary["Tỷ trọng trong tăng trưởng GDP, %"] = (
        summary["Đóng góp bình quân, điểm %"] / avg_growth * 100
    )

    return ga, summary, avg_growth


def simulate_2030(df, alpha, beta, gamma, delta, theta):
    """
    Kịch bản 2030 theo đề:
    - D = 30%
    - AI = 100 nghìn DN số
    - H = 35%
    - K và L tăng 6%/năm
    - TFP tăng 1,2%/năm
    """

    base = df[df["year"] == 2025].iloc[0]

    years = list(range(2026, 2031))
    scenario = []

    for n, year in enumerate(years, start=1):
        K_t = base["K"] * (1.06 ** n)
        L_t = base["L"] * (1.06 ** n)

        # Nội suy tuyến tính để người xem thấy quá trình đi từ 2025 đến mục tiêu 2030
        D_t = base["D"] + (30.0 - base["D"]) * n / 5
        AI_t = base["AI"] + (100.0 - base["AI"]) * n / 5
        H_t = base["H"] + (35.0 - base["H"]) * n / 5

        A_t = base["A_TFP"] * (1.012 ** n)

        Y_t = A_t * (
            K_t ** alpha *
            L_t ** beta *
            D_t ** gamma *
            AI_t ** delta *
            H_t ** theta
        )

        scenario.append({
            "year": year,
            "K": K_t,
            "L": L_t,
            "D": D_t,
            "AI": AI_t,
            "H": H_t,
            "A_TFP": A_t,
            "Y_forecast": Y_t,
        })

    return pd.DataFrame(scenario)


def cagr(start, end, years):
    return (end / start) ** (1 / years) - 1


# -----------------------------
# 4. CÁC KHỐI HIỂN THỊ
# -----------------------------
def show_context(df):
    st.header("1.1. Bối cảnh Việt Nam 2020–2025")

    st.markdown("""
    Bài toán đặt ra là: nếu nền kinh tế Việt Nam được mô hình hóa bằng hàm sản xuất Cobb-Douglas mở rộng,
    trong đó ngoài **vốn K** và **lao động L** còn có thêm **số hóa D**, **năng lực AI** và **nhân lực số H**,
    thì mô hình có giải thích tốt biến động GDP thực tế hay không.
    """)

    gdp_2020 = df.loc[df["year"] == 2020, "Y"].iloc[0]
    gdp_2025 = df.loc[df["year"] == 2025, "Y"].iloc[0]
    d_2020 = df.loc[df["year"] == 2020, "D"].iloc[0]
    d_2025 = df.loc[df["year"] == 2025, "D"].iloc[0]
    ai_2020 = df.loc[df["year"] == 2020, "AI"].iloc[0]
    ai_2025 = df.loc[df["year"] == 2025, "AI"].iloc[0]
    h_2020 = df.loc[df["year"] == 2020, "H"].iloc[0]
    h_2025 = df.loc[df["year"] == 2025, "H"].iloc[0]

    gdp_cagr = cagr(gdp_2020, gdp_2025, 5) * 100
    d_cagr = cagr(d_2020, d_2025, 5) * 100
    ai_cagr = cagr(ai_2020, ai_2025, 5) * 100
    h_cagr = cagr(h_2020, h_2025, 5) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP 2025", f"{gdp_2025:,.1f}", "nghìn tỷ VND")
    c2.metric("CAGR GDP 2020–2025", f"{gdp_cagr:.2f}%/năm")
    c3.metric("Kinh tế số/GDP 2025", f"{d_2025:.1f}%")
    c4.metric("DN công nghệ số 2025", f"{ai_2025:.1f} nghìn")

    context_table = pd.DataFrame({
        "Chỉ tiêu": [
            "GDP",
            "Kinh tế số/GDP D",
            "Năng lực AI / DN công nghệ số",
            "Nhân lực số H",
        ],
        "2020": [gdp_2020, d_2020, ai_2020, h_2020],
        "2025": [gdp_2025, d_2025, ai_2025, h_2025],
        "Tăng trưởng kép/năm": [
            f"{gdp_cagr:.2f}%",
            f"{d_cagr:.2f}%",
            f"{ai_cagr:.2f}%",
            f"{h_cagr:.2f}%",
        ],
    })

    st.dataframe(context_table, use_container_width=True)

    long_df = df.melt(
        id_vars="year",
        value_vars=["Y", "K", "D", "AI", "H"],
        var_name="Chỉ tiêu",
        value_name="Giá trị"
    )

    fig = px.line(
        long_df,
        x="year",
        y="Giá trị",
        color="Chỉ tiêu",
        markers=True,
        title="Bức tranh tăng trưởng và chuyển đổi số Việt Nam 2020–2025"
    )
    fig.update_layout(height=480, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Ý nghĩa bối cảnh: GDP tăng nhanh sau giai đoạn 2020–2021, trong khi D, AI và H đều tăng. "
        "Vì vậy, Bài 1 không chỉ kiểm tra GDP theo vốn và lao động truyền thống, mà còn lượng hóa vai trò "
        "của chuyển đổi số, AI và nhân lực số trong tăng trưởng."
    )


def show_math_model():
    st.header("1.2. Mô hình toán học")

    st.markdown("### Bước 1 — Hàm sản xuất Cobb-Douglas mở rộng")

    st.latex(r"""
    Y_t = A_t \cdot K_t^{\alpha} \cdot L_t^{\beta}
    \cdot D_t^{\gamma} \cdot AI_t^{\delta} \cdot H_t^{\theta}
    """)

    st.markdown("""
    Trong mô hình này, GDP không chỉ được tạo ra từ **vốn vật chất K** và **lao động L**,
    mà còn chịu tác động của ba yếu tố mới: **số hóa D**, **năng lực AI** và **nhân lực số H**.
    """)

    st.markdown("### Bước 2 — Điều kiện lợi suất không đổi theo quy mô")

    st.latex(r"""
    \alpha + \beta + \gamma + \delta + \theta = 1
    """)

    st.markdown("""
    Điều kiện này có nghĩa là nếu tất cả đầu vào cùng tăng theo một tỷ lệ, sản lượng kỳ vọng cũng tăng theo tỷ lệ tương ứng.
    Trong code, ta cho phép người dùng chỉnh α, β, γ, δ và để θ tự động tính theo phần còn lại.
    """)

    st.markdown("### Bước 3 — Giải ngược TFP")

    st.latex(r"""
    A_t = \frac{Y_t}
    {K_t^{\alpha} \cdot L_t^{\beta} \cdot D_t^{\gamma} \cdot AI_t^{\delta} \cdot H_t^{\theta}}
    """)

    st.markdown("""
    TFP là phần sản lượng không được giải thích trực tiếp bởi các đầu vào quan sát được.
    Nếu A_t tăng, điều đó thường hàm ý chất lượng tăng trưởng, công nghệ, quản trị hoặc hiệu quả phân bổ nguồn lực được cải thiện.
    """)

    st.markdown("### Bước 4 — Log hóa để phân rã tăng trưởng")

    st.latex(r"""
    \ln Y_t = \ln A_t + \alpha \ln K_t + \beta \ln L_t
    + \gamma \ln D_t + \delta \ln AI_t + \theta \ln H_t
    """)

    st.markdown("### Bước 5 — Phương trình growth accounting")

    st.latex(r"""
    \Delta \ln Y_t =
    \Delta \ln A_t
    + \alpha \Delta \ln K_t
    + \beta \Delta \ln L_t
    + \gamma \Delta \ln D_t
    + \delta \Delta \ln AI_t
    + \theta \Delta \ln H_t
    """)

    st.success(
        "Tư duy chính của Bài 1: từ dữ liệu GDP thực tế, ta tính ngược TFP, sau đó kiểm tra mô hình dự báo GDP "
        "và phân rã xem tăng trưởng đến từ K, L, D, AI, H hay TFP."
    )


def show_data(df):
    st.header("1.3. Dữ liệu Việt Nam 2020–2025")

    display_df = df[["year", "Y", "K", "L", "D", "AI", "H"]].copy()
    display_df.columns = [
        "Năm",
        "Y - GDP, nghìn tỷ VND",
        "K - Vốn tích lũy, nghìn tỷ",
        "L - Lao động, triệu người",
        "D - Kinh tế số/GDP, %",
        "AI - DN công nghệ số, nghìn",
        "H - LĐ qua đào tạo, %",
    ]

    st.dataframe(display_df, use_container_width=True)

    st.markdown("### Chuẩn hóa chỉ số để so sánh xu hướng")

    norm_df = df[["year", "Y", "K", "L", "D", "AI", "H"]].copy()
    for col in ["Y", "K", "L", "D", "AI", "H"]:
        norm_df[col] = norm_df[col] / norm_df[col].iloc[0] * 100

    norm_long = norm_df.melt(
        id_vars="year",
        value_vars=["Y", "K", "L", "D", "AI", "H"],
        var_name="Biến",
        value_name="Chỉ số, 2020 = 100"
    )

    fig = px.line(
        norm_long,
        x="year",
        y="Chỉ số, 2020 = 100",
        color="Biến",
        markers=True,
        title="So sánh tốc độ tăng của các biến đầu vào, 2020 = 100"
    )
    fig.update_layout(height=480, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Biểu đồ chuẩn hóa giúp thấy biến nào tăng nhanh hơn so với gốc 2020. "
        "D và AI tăng nhanh hơn lao động L, cho thấy chuyển đổi số là nguồn động lực mới cần được đưa vào mô hình."
    )


def show_parameters():
    st.header("Thiết lập tham số mô hình")

    st.markdown("""
    Hệ số đề xuất của đề bài:  
    **α = 0,33; β = 0,42; γ = 0,10; δ = 0,08; θ = 0,07**.
    """)

    c1, c2, c3, c4 = st.columns(4)

    alpha = c1.slider("α - Vốn vật chất K", 0.10, 0.60, 0.33, 0.01)
    beta = c2.slider("β - Lao động L", 0.10, 0.60, 0.42, 0.01)
    gamma = c3.slider("γ - Số hóa D", 0.00, 0.30, 0.10, 0.01)
    delta = c4.slider("δ - Năng lực AI", 0.00, 0.30, 0.08, 0.01)

    theta = 1 - alpha - beta - gamma - delta

    c5, c6 = st.columns([1, 3])
    c5.metric("θ - Nhân lực số H", f"{theta:.2f}")
    c6.progress(max(0, min(1, alpha + beta + gamma + delta + theta)))

    if theta < 0:
        st.error("Tổng α + β + γ + δ đang lớn hơn 1. Cần giảm một hoặc nhiều hệ số.")
        st.stop()

    if theta > 0.30:
        st.warning("θ đang khá lớn. Nên kiểm tra lại giả định vì nhân lực số có thể bị gán vai trò quá cao.")

    param_table = pd.DataFrame({
        "Tham số": ["α", "β", "γ", "δ", "θ"],
        "Yếu tố": ["Vốn K", "Lao động L", "Số hóa D", "AI", "Nhân lực số H"],
        "Giá trị": [alpha, beta, gamma, delta, theta],
        "Ý nghĩa": [
            "GDP co giãn theo vốn vật chất",
            "GDP co giãn theo lao động",
            "GDP co giãn theo mức độ số hóa",
            "GDP co giãn theo năng lực AI",
            "GDP co giãn theo nhân lực số",
        ],
    })

    st.dataframe(param_table, use_container_width=True)

    return alpha, beta, gamma, delta, theta


def show_requirement_14(df, alpha, beta, gamma, delta, theta):
    st.header("1.4. Giải yêu cầu lập trình")

    df_model, A_bar, mape = calculate_model(df, alpha, beta, gamma, delta, theta)
    ga, ga_summary, avg_growth = calculate_growth_accounting(df_model, alpha, beta, gamma, delta, theta)
    scenario_2030 = simulate_2030(df_model, alpha, beta, gamma, delta, theta)

    # -------------------------
    # Câu 1.4.1
    # -------------------------
    st.subheader("Câu 1.4.1 — Ước lượng TFP A_t")

    c1, c2, c3 = st.columns(3)
    c1.metric("TFP 2020", f"{df_model['A_TFP'].iloc[0]:.3f}")
    c2.metric("TFP 2025", f"{df_model['A_TFP'].iloc[-1]:.3f}")
    c3.metric(
        "Thay đổi TFP 2020–2025",
        f"{(df_model['A_TFP'].iloc[-1] / df_model['A_TFP'].iloc[0] - 1) * 100:.2f}%"
    )

    st.dataframe(
        df_model[["year", "Y", "K", "L", "D", "AI", "H", "A_TFP"]].round(3),
        use_container_width=True
    )

    fig_a = px.line(
        df_model,
        x="year",
        y="A_TFP",
        markers=True,
        title="Xu hướng năng suất nhân tố tổng hợp A_t, 2020–2025"
    )
    fig_a.update_traces(line=dict(width=4), marker=dict(size=10))
    fig_a.update_layout(height=430)
    st.plotly_chart(fig_a, use_container_width=True)

    if df_model["A_TFP"].iloc[-1] > df_model["A_TFP"].iloc[0]:
        st.success(
            "Bình luận: TFP có xu hướng tăng trong toàn kỳ 2020–2025. "
            "Điều này hàm ý tăng trưởng không chỉ đến từ mở rộng vốn và lao động, "
            "mà còn có dấu hiệu cải thiện về hiệu quả, công nghệ và chất lượng tổ chức sản xuất."
        )
    else:
        st.warning(
            "Bình luận: TFP giảm trong toàn kỳ. Điều này cho thấy tăng trưởng có thể đang phụ thuộc nhiều vào mở rộng đầu vào, "
            "chưa phản ánh rõ cải thiện về hiệu quả hay chất lượng tăng trưởng."
        )

    # -------------------------
    # Câu 1.4.2
    # -------------------------
    st.subheader("Câu 1.4.2 — So sánh GDP thực tế và GDP dự báo")

    c4, c5 = st.columns(2)
    c4.metric("A trung bình 2020–2025", f"{A_bar:.3f}")
    c5.metric("MAPE", f"{mape:.2f}%")

    forecast_table = df_model[["year", "Y", "Y_hat", "APE_pct"]].copy()
    forecast_table.columns = ["Năm", "GDP thực tế", "GDP dự báo", "Sai số tuyệt đối, %"]
    st.dataframe(forecast_table.round(2), use_container_width=True)

    fig_y = go.Figure()
    fig_y.add_trace(go.Scatter(
        x=df_model["year"],
        y=df_model["Y"],
        mode="lines+markers",
        name="GDP thực tế",
        line=dict(width=4)
    ))
    fig_y.add_trace(go.Scatter(
        x=df_model["year"],
        y=df_model["Y_hat"],
        mode="lines+markers",
        name="GDP dự báo",
        line=dict(width=4, dash="dash")
    ))
    fig_y.update_layout(
        title="So sánh GDP thực tế và GDP dự báo bằng A trung bình",
        xaxis_title="Năm",
        yaxis_title="GDP, nghìn tỷ VND",
        height=460
    )
    st.plotly_chart(fig_y, use_container_width=True)

    if mape < 5:
        st.success(
            f"MAPE = {mape:.2f}%, mô hình có mức sai số thấp trong bộ dữ liệu nhỏ 2020–2025. "
            "Điều này cho thấy Cobb-Douglas mở rộng có khả năng mô phỏng tương đối tốt xu hướng GDP."
        )
    elif mape < 10:
        st.info(
            f"MAPE = {mape:.2f}%, mô hình ở mức chấp nhận được cho bài tập mô phỏng chính sách. "
            "Tuy nhiên, cần thận trọng vì số quan sát chỉ có 6 năm."
        )
    else:
        st.warning(
            f"MAPE = {mape:.2f}%, sai số còn khá cao. Nên kiểm tra lại hệ số co giãn hoặc cách đo các biến D, AI, H."
        )

    # -------------------------
    # Câu 1.4.3
    # -------------------------
    st.subheader("Câu 1.4.3 — Phân rã tăng trưởng 2020–2025")

    st.markdown(f"""
    Tăng trưởng GDP bình quân theo log trong giai đoạn 2020–2025 là khoảng **{avg_growth:.2f}%/năm**.
    Bảng dưới đây cho biết mỗi yếu tố đóng góp bao nhiêu điểm phần trăm và chiếm bao nhiêu phần trăm trong tăng trưởng.
    """)

    st.dataframe(ga.round(3), use_container_width=True)
    st.dataframe(ga_summary.round(2), use_container_width=True)

    fig_contrib = px.bar(
        ga_summary,
        x="Yếu tố",
        y="Đóng góp bình quân, điểm %",
        text="Đóng góp bình quân, điểm %",
        title="Đóng góp bình quân của từng yếu tố vào tăng trưởng GDP"
    )
    fig_contrib.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_contrib.update_layout(height=460)
    st.plotly_chart(fig_contrib, use_container_width=True)

    fig_share = px.pie(
        ga_summary,
        names="Yếu tố",
        values="Tỷ trọng trong tăng trưởng GDP, %",
        title="Cơ cấu đóng góp vào tăng trưởng GDP bình quân 2020–2025"
    )
    fig_share.update_layout(height=460)
    st.plotly_chart(fig_share, use_container_width=True)

    # -------------------------
    # Câu 1.4.4
    # -------------------------
    st.subheader("Câu 1.4.4 — Mô phỏng kịch bản GDP Việt Nam đến năm 2030")

    y_2025 = df_model.loc[df_model["year"] == 2025, "Y"].iloc[0]
    y_2030 = scenario_2030.loc[scenario_2030["year"] == 2030, "Y_forecast"].iloc[0]
    growth_2030 = (y_2030 / y_2025 - 1) * 100

    c6, c7, c8 = st.columns(3)
    c6.metric("GDP thực tế 2025", f"{y_2025:,.1f}", "nghìn tỷ VND")
    c7.metric("GDP dự báo 2030", f"{y_2030:,.1f}", "nghìn tỷ VND")
    c8.metric("Tăng so với 2025", f"{growth_2030:.2f}%")

    st.dataframe(scenario_2030.round(2), use_container_width=True)

    combined = pd.concat([
        df_model[["year", "Y"]].rename(columns={"Y": "GDP"}),
        scenario_2030[["year", "Y_forecast"]].rename(columns={"Y_forecast": "GDP"})
    ], ignore_index=True)

    combined["Loại dữ liệu"] = np.where(combined["year"] <= 2025, "Thực tế", "Dự báo kịch bản")

    fig_2030 = px.line(
        combined,
        x="year",
        y="GDP",
        color="Loại dữ liệu",
        markers=True,
        title="GDP thực tế 2020–2025 và dự báo kịch bản đến 2030"
    )
    fig_2030.update_traces(line=dict(width=4), marker=dict(size=9))
    fig_2030.update_layout(height=480)
    st.plotly_chart(fig_2030, use_container_width=True)

    st.info(
        "Diễn giải: kịch bản 2030 giả định Việt Nam tăng mạnh tỷ trọng kinh tế số, mở rộng số doanh nghiệp công nghệ số, "
        "nâng tỷ lệ nhân lực số, đồng thời duy trì tăng trưởng vốn, lao động và TFP. "
        "Do đó, GDP dự báo phản ánh cả mở rộng đầu vào và cải thiện chất lượng tăng trưởng."
    )

    return df_model, ga_summary, scenario_2030, mape


def show_policy_discussion(df_model, ga_summary, scenario_2030):
    st.header("1.5. Câu hỏi thảo luận chính sách")

    # Tính minh chứng
    tfp_2020 = df_model["A_TFP"].iloc[0]
    tfp_2025 = df_model["A_TFP"].iloc[-1]
    tfp_change = (tfp_2025 / tfp_2020 - 1) * 100
    tfp_cagr = cagr(tfp_2020, tfp_2025, 5) * 100

    new_factors = ga_summary[ga_summary["Yếu tố"].isin(["Số hóa D", "Năng lực AI", "Nhân lực số H"])].copy()
    top_new = new_factors.sort_values("Đóng góp bình quân, điểm %", ascending=False).iloc[0]

    d_2020 = df_model.loc[df_model["year"] == 2020, "D"].iloc[0]
    d_2025 = df_model.loc[df_model["year"] == 2025, "D"].iloc[0]
    d_2030_target = 30.0

    d_cagr_past = cagr(d_2020, d_2025, 5) * 100
    d_cagr_need = cagr(d_2025, d_2030_target, 5) * 100

    y_2030 = scenario_2030.loc[scenario_2030["year"] == 2030, "Y_forecast"].iloc[0]

    # -------------------------
    # Câu a
    # -------------------------
    st.subheader("a) TFP tăng hay giảm? Điều đó nói gì về chất lượng tăng trưởng?")

    c1, c2, c3 = st.columns(3)
    c1.metric("TFP 2020", f"{tfp_2020:.3f}")
    c2.metric("TFP 2025", f"{tfp_2025:.3f}")
    c3.metric("CAGR TFP", f"{tfp_cagr:.2f}%/năm")

    fig_tfp = px.area(
        df_model,
        x="year",
        y="A_TFP",
        title="Minh chứng cho câu a: xu hướng TFP A_t"
    )
    fig_tfp.update_layout(height=400)
    st.plotly_chart(fig_tfp, use_container_width=True)

    if tfp_change > 0:
        st.success(
            f"Trả lời: TFP tăng khoảng {tfp_change:.2f}% trong giai đoạn 2020–2025, tương đương {tfp_cagr:.2f}%/năm. "
            "Điều này cho thấy chất lượng tăng trưởng có cải thiện: GDP không chỉ tăng nhờ mở rộng vốn và lao động, "
            "mà còn nhờ hiệu quả tổng hợp, công nghệ, quản trị và khả năng hấp thụ chuyển đổi số."
        )
    else:
        st.warning(
            f"Trả lời: TFP giảm khoảng {abs(tfp_change):.2f}% trong giai đoạn 2020–2025. "
            "Điều này hàm ý tăng trưởng còn phụ thuộc nhiều vào mở rộng đầu vào, trong khi hiệu quả tổng hợp chưa cải thiện rõ."
        )

    # -------------------------
    # Câu b
    # -------------------------
    st.subheader("b) Trong D, AI, H, yếu tố nào đóng góp nhiều nhất? Vì sao?")

    st.dataframe(new_factors.round(2), use_container_width=True)

    fig_new = px.bar(
        new_factors.sort_values("Đóng góp bình quân, điểm %", ascending=False),
        x="Yếu tố",
        y="Đóng góp bình quân, điểm %",
        text="Đóng góp bình quân, điểm %",
        title="Minh chứng cho câu b: đóng góp của ba yếu tố mới D, AI, H"
    )
    fig_new.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_new.update_layout(height=420)
    st.plotly_chart(fig_new, use_container_width=True)

    st.info(
        f"Trả lời: Trong ba yếu tố mới, **{top_new['Yếu tố']}** đóng góp lớn nhất, "
        f"với khoảng **{top_new['Đóng góp bình quân, điểm %']:.2f} điểm %/năm**. "
        "Lý do là đóng góp tăng trưởng phụ thuộc đồng thời vào hai thành phần: "
        "tốc độ tăng của biến đầu vào và hệ số co giãn của biến đó trong hàm sản xuất. "
        "Một biến có hệ số không quá lớn nhưng tăng nhanh vẫn có thể tạo đóng góp đáng kể."
    )

    # -------------------------
    # Câu c
    # -------------------------
    st.subheader("c) Mục tiêu kinh tế số đạt 30% GDP vào 2030 có khả thi không?")

    c4, c5, c6 = st.columns(3)
    c4.metric("D năm 2025", f"{d_2025:.1f}% GDP")
    c5.metric("Mục tiêu D năm 2030", "30.0% GDP")
    c6.metric("Tốc độ D cần đạt", f"{d_cagr_need:.2f}%/năm")

    d_path = pd.DataFrame({
        "Giai đoạn": ["Đã đạt 2020–2025", "Cần đạt 2025–2030"],
        "CAGR D, %/năm": [d_cagr_past, d_cagr_need],
    })

    fig_d = px.bar(
        d_path,
        x="Giai đoạn",
        y="CAGR D, %/năm",
        text="CAGR D, %/năm",
        title="Minh chứng cho câu c: tốc độ tăng D quá khứ và tốc độ cần để đạt 30% GDP"
    )
    fig_d.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_d.update_layout(height=420)
    st.plotly_chart(fig_d, use_container_width=True)

    if d_cagr_need <= d_cagr_past:
        feasibility = "khá khả thi về mặt tốc độ số học"
    else:
        feasibility = "khả thi nhưng cần nỗ lực cao hơn tốc độ đã đạt trong giai đoạn trước"

    st.success(
        f"Trả lời: Từ mức {d_2025:.1f}% năm 2025 lên 30% năm 2030, kinh tế số cần tăng khoảng "
        f"{d_cagr_need:.2f}%/năm. So với tốc độ đã đạt giai đoạn 2020–2025 là {d_cagr_past:.2f}%/năm, "
        f"mục tiêu này **{feasibility}**. Theo mô hình, nếu đi kèm K và L tăng 6%/năm, TFP tăng 1,2%/năm, "
        f"AI đạt 100 nghìn DN và H đạt 35%, GDP năm 2030 có thể đạt khoảng **{y_2030:,.1f} nghìn tỷ VND**."
    )

    st.markdown("""
    **Các ràng buộc chính sách cần đi kèm:**

    1. **Ràng buộc nhân lực số:** tăng D nhưng thiếu H sẽ làm hiệu quả chuyển đổi số thấp.
    2. **Ràng buộc hạ tầng dữ liệu:** AI cần dữ liệu, cloud, trung tâm dữ liệu và kết nối số.
    3. **Ràng buộc hấp thụ của doanh nghiệp:** doanh nghiệp nhỏ và vừa phải có khả năng ứng dụng công nghệ.
    4. **Ràng buộc thể chế:** cần tiêu chuẩn dữ liệu, an toàn thông tin, sandbox và khung pháp lý cho AI.
    5. **Ràng buộc vùng miền:** nếu D chỉ tăng ở các đô thị lớn, tăng trưởng có thể cao nhưng thiếu bao trùm.
    """)


# -----------------------------
# 5. HÀM RENDER CHÍNH
# -----------------------------
def render():
    st.title("🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa")

    st.markdown("""
    Module này trình bày Bài 1 theo đúng logic của một bài phân tích mô hình ra quyết định:
    **bối cảnh → mô hình → dữ liệu → tính toán → mô phỏng → thảo luận chính sách**.
    """)

    df = load_macro_data()

    tabs = st.tabs([
        "1.1 Bối cảnh",
        "1.2 Mô hình",
        "1.3 Dữ liệu",
        "1.4 Tính toán",
        "1.5 Chính sách",
    ])

    with tabs[0]:
        show_context(df)

    with tabs[1]:
        show_math_model()

    with tabs[2]:
        show_data(df)

    with tabs[3]:
        alpha, beta, gamma, delta, theta = show_parameters()
        df_model, ga_summary, scenario_2030, mape = show_requirement_14(
            df, alpha, beta, gamma, delta, theta
        )

    with tabs[4]:
        alpha, beta, gamma, delta, theta = show_parameters()
        df_model, A_bar, mape = calculate_model(df, alpha, beta, gamma, delta, theta)
        ga, ga_summary, avg_growth = calculate_growth_accounting(
            df_model, alpha, beta, gamma, delta, theta
        )
        scenario_2030 = simulate_2030(df_model, alpha, beta, gamma, delta, theta)
        show_policy_discussion(df_model, ga_summary, scenario_2030)

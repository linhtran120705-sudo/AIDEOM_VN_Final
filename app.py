import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="VN AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide"
)

st.sidebar.title("🇻🇳 VN AIDEOM-VN")
st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI")

menu = st.sidebar.radio(
    "Chọn bài",
    [
        "🏠 Trang chủ",
        "🌱 Bài 1 — Cobb-Douglas + AI",
        "💰 Bài 2 — LP ngân sách số",
        "📊 Bài 3 — Priority 10 ngành",
        "🗺️ Bài 4 — LP ngành-vùng",
        "🎯 Bài 5 — MIP 15 dự án",
        "🏆 Bài 6 — TOPSIS 6 vùng",
        "🌐 Bài 7 — NSGA-II Pareto",
        "⏳ Bài 8 — Động 2026-2035",
        "👷 Bài 9 — Lao động & AI",
        "🎲 Bài 10 — Stochastic SP",
        "🤖 Bài 11 — Q-learning RL",
        "🧠 Bài 12 — AIDEOM tích hợp",
    ]
)

if menu == "🏠 Trang chủ":
    st.title("VN AIDEOM-VN")
    st.subheader("AI-Driven Decision Optimization Model for Vietnam")

    st.write(
        "Web app giải 12 bài mô hình ra quyết định phát triển kinh tế Việt Nam "
        "trong kỉ nguyên AI, sử dụng Python, Streamlit và Gemini API."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GDP 2025", "514,0 tỷ USD", "+8,02%")
    col2.metric("Kinh tế số/GDP", "≈19,5%", "+1,2 điểm %")
    col3.metric("FDI 2025", "27,6 tỷ USD", "+8,9%")
    col4.metric("GDP/người 2025", "5.026 USD", "+6,9%")

elif menu == "🌱 Bài 1 — Cobb-Douglas + AI":
    st.title("🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa")

    st.markdown("""
    Mô hình:

    $Y_t = A_t K_t^\\alpha L_t^\\beta D_t^\\gamma AI_t^\\delta H_t^\\theta$
    """)

    df = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "Y": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
        "K": [16500, 17800, 19600, 21300, 23500, 25900],
        "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
        "D": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
        "AI": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
        "H": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    })

    st.subheader("1. Dữ liệu đầu vào")
    st.dataframe(df)

    st.subheader("2. Chỉnh tham số")

    col1, col2, col3, col4 = st.columns(4)

    alpha = col1.slider("α - Vốn K", 0.10, 0.60, 0.33, 0.01)
    beta = col2.slider("β - Lao động L", 0.10, 0.60, 0.42, 0.01)
    gamma = col3.slider("γ - Số hóa D", 0.00, 0.30, 0.10, 0.01)
    delta = col4.slider("δ - AI", 0.00, 0.30, 0.08, 0.01)

    theta = 1 - alpha - beta - gamma - delta
    st.metric("θ - Nhân lực số H tự động", f"{theta:.2f}")

    if theta < 0:
        st.error("Tổng hệ số đang lớn hơn 1. Cần giảm alpha, beta, gamma hoặc delta.")
        st.stop()

    df["A_TFP"] = df["Y"] / (
        df["K"] ** alpha *
        df["L"] ** beta *
        df["D"] ** gamma *
        df["AI"] ** delta *
        df["H"] ** theta
    )

    st.subheader("3. TFP A_t")
    st.dataframe(df[["year", "A_TFP"]].round(3))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["year"], df["A_TFP"], marker="o")
    ax.set_title("Xu hướng TFP A_t, 2020-2025")
    ax.set_xlabel("Năm")
    ax.set_ylabel("A_t")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    A_bar = df["A_TFP"].mean()

    df["Y_hat"] = A_bar * (
        df["K"] ** alpha *
        df["L"] ** beta *
        df["D"] ** gamma *
        df["AI"] ** delta *
        df["H"] ** theta
    )

    df["APE_pct"] = abs(df["Y"] - df["Y_hat"]) / df["Y"] * 100
    mape = df["APE_pct"].mean()

    st.subheader("4. GDP thực tế và GDP dự báo")
    c1, c2 = st.columns(2)
    c1.metric("A trung bình", f"{A_bar:.3f}")
    c2.metric("MAPE", f"{mape:.2f}%")

    st.dataframe(df[["year", "Y", "Y_hat", "APE_pct"]].round(2))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(df["year"], df["Y"], marker="o", label="GDP thực tế")
    ax2.plot(df["year"], df["Y_hat"], marker="o", linestyle="--", label="GDP dự báo")
    ax2.set_title("So sánh GDP thực tế và GDP dự báo")
    ax2.set_xlabel("Năm")
    ax2.set_ylabel("GDP, nghìn tỷ VND")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

elif menu == "💰 Bài 2 — LP ngân sách số":
    from scipy.optimize import linprog

    st.title("💰 Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách số")

    st.markdown("""
    Bài toán phân bổ ngân sách cho 4 hạng mục đầu tư số:

    - **x1**: Hạ tầng số  
    - **x2**: AI và dữ liệu  
    - **x3**: Nhân lực số  
    - **x4**: R&D công nghệ  

    Hàm mục tiêu:

    **Max Z = 0.85x1 + 1.20x2 + 0.95x3 + 1.35x4**
    """)

    st.subheader("1. Chỉnh tham số ngân sách")

    B = st.slider("Tổng ngân sách B, nghìn tỷ VND", 80, 150, 100, 5)

    col1, col2, col3, col4 = st.columns(4)
    min_x1 = col1.slider("x1 tối thiểu - Hạ tầng số", 0, 50, 25, 1)
    min_x2 = col2.slider("x2 tối thiểu - AI & dữ liệu", 0, 50, 15, 1)
    min_x3 = col3.slider("x3 tối thiểu - Nhân lực số", 0, 50, 20, 1)
    min_x4 = col4.slider("x4 tối thiểu - R&D", 0, 50, 10, 1)

    strategic_share = st.slider(
        "Tỷ trọng tối thiểu của AI + R&D trong tổng ngân sách",
        0.10, 0.60, 0.35, 0.01
    )

    st.subheader("2. Giải bài toán tối ưu")

    # Tối đa hóa Z được chuyển thành minimize -Z
    c = [-0.85, -1.20, -0.95, -1.35]

    s = strategic_share

    # Ràng buộc:
    # x1 + x2 + x3 + x4 <= B
    # x1 >= min_x1 -> -x1 <= -min_x1
    # x2 >= min_x2 -> -x2 <= -min_x2
    # x3 >= min_x3 -> -x3 <= -min_x3
    # x4 >= min_x4 -> -x4 <= -min_x4
    # x2 + x4 >= s(x1+x2+x3+x4)
    # tương đương: s*x1 + (s-1)*x2 + s*x3 + (s-1)*x4 <= 0
    A_ub = [
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [s, s - 1, s, s - 1]
    ]

    b_ub = [
        B,
        -min_x1,
        -min_x2,
        -min_x3,
        -min_x4,
        0
    ]

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * 4,
        method="highs"
    )

    if res.success:
        x1, x2, x3, x4 = res.x
        Z = -res.fun

        st.success("Bài toán có nghiệm tối ưu.")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Z* - GDP gain tối ưu", f"{Z:.2f}")
        metric2.metric("Ngân sách sử dụng", f"{sum(res.x):.2f}")
        metric3.metric("Tỷ trọng AI + R&D", f"{(x2 + x4) / sum(res.x) * 100:.2f}%")

        result_df = pd.DataFrame({
            "Biến": ["x1", "x2", "x3", "x4"],
            "Hạng mục": ["Hạ tầng số", "AI & dữ liệu", "Nhân lực số", "R&D công nghệ"],
            "Phân bổ tối ưu": [x1, x2, x3, x4],
            "Hệ số tác động GDP": [0.85, 1.20, 0.95, 1.35]
        })

        st.subheader("3. Bảng phân bổ tối ưu")
        st.dataframe(result_df.round(2))

        st.subheader("4. Biểu đồ phân bổ ngân sách tối ưu")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(result_df["Hạng mục"], result_df["Phân bổ tối ưu"])
        ax.set_title("Phân bổ ngân sách tối ưu cho 4 hạng mục")
        ax.set_ylabel("Nghìn tỷ VND")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)

        st.subheader("5. Phân tích độ nhạy theo ngân sách")

        budget_list = [100, 120, 140]
        z_list = []
        status_list = []

        for B_test in budget_list:
            b_test = [
                B_test,
                -min_x1,
                -min_x2,
                -min_x3,
                -min_x4,
                0
            ]

            res_test = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_test,
                bounds=[(0, None)] * 4,
                method="highs"
            )

            if res_test.success:
                z_list.append(-res_test.fun)
                status_list.append("Tối ưu")
            else:
                z_list.append(np.nan)
                status_list.append("Không khả thi")

        sensitivity_df = pd.DataFrame({
            "Ngân sách B": budget_list,
            "Z*": z_list,
            "Trạng thái": status_list
        })

        st.dataframe(sensitivity_df.round(2))

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(sensitivity_df["Ngân sách B"], sensitivity_df["Z*"], marker="o")
        ax2.set_title("Đường cong Z*(B)")
        ax2.set_xlabel("Ngân sách, nghìn tỷ VND")
        ax2.set_ylabel("GDP gain tối ưu")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        st.subheader("6. Nhận xét tự động")

        max_row = result_df.loc[result_df["Phân bổ tối ưu"].idxmax()]
        st.info(
            f"Hạng mục được phân bổ nhiều nhất là **{max_row['Hạng mục']}** "
            f"với {max_row['Phân bổ tối ưu']:.2f} nghìn tỷ VND."
        )

        if x4 >= max(x1, x2, x3):
            st.success("R&D được ưu tiên nhiều do có hệ số tác động GDP cao nhất.")
        else:
            st.write(
                "Kết quả phân bổ không chỉ phụ thuộc vào hệ số tác động, "
                "mà còn phụ thuộc vào các ràng buộc tối thiểu và tỷ trọng công nghệ chiến lược."
            )

    else:
        st.error("Bài toán không khả thi với bộ tham số hiện tại.")
        st.write("Gợi ý: Hãy tăng tổng ngân sách hoặc giảm các mức tối thiểu của x1, x2, x3, x4.")
else:
    st.warning("Module này sẽ được bổ sung ở bước tiếp theo.")

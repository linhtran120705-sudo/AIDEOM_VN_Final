import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def solve_lp_budget(B, min_x1, min_x2, min_x3, min_x4, strategic_share):
    """
    Giải bài toán LP Bài 2:
    Max Z = 0.85x1 + 1.20x2 + 0.95x3 + 1.35x4

    Ràng buộc:
    x1 + x2 + x3 + x4 <= B
    x1 >= min_x1
    x2 >= min_x2
    x3 >= min_x3
    x4 >= min_x4
    x2 + x4 >= strategic_share * (x1 + x2 + x3 + x4)
    xi >= 0
    """

    # linprog là bài toán minimize, nên đổi Max Z thành Min -Z
    c = [-0.85, -1.20, -0.95, -1.35]

    s = strategic_share

    # x2 + x4 >= s(x1+x2+x3+x4)
    # Chuyển về dạng A_ub x <= b_ub:
    # s*x1 + (s-1)*x2 + s*x3 + (s-1)*x4 <= 0
    A_ub = [
        [1, 1, 1, 1],          # tổng ngân sách
        [-1, 0, 0, 0],         # x1 >= min_x1
        [0, -1, 0, 0],         # x2 >= min_x2
        [0, 0, -1, 0],         # x3 >= min_x3
        [0, 0, 0, -1],         # x4 >= min_x4
        [s, s - 1, s, s - 1]   # tỷ trọng AI + R&D
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

    return res


def render():
    st.title("💰 Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách số")

    st.markdown("""
    ### 1. Mô hình bài toán

    Bài toán phân bổ ngân sách cho 4 hạng mục đầu tư số:

    - **x1**: Hạ tầng số  
    - **x2**: AI và dữ liệu  
    - **x3**: Nhân lực số  
    - **x4**: R&D công nghệ  

    Hàm mục tiêu:

    **Max Z = 0.85x1 + 1.20x2 + 0.95x3 + 1.35x4**

    Các ràng buộc chính:

    - Tổng ngân sách không vượt quá B  
    - Mỗi hạng mục có mức đầu tư tối thiểu  
    - AI + R&D phải chiếm tối thiểu một tỷ trọng nhất định trong tổng ngân sách  
    """)

    st.divider()

    st.subheader("2. Chỉnh tham số đầu vào")

    B = st.slider(
        "Tổng ngân sách B, nghìn tỷ VND",
        min_value=80,
        max_value=150,
        value=100,
        step=5
    )

    col1, col2, col3, col4 = st.columns(4)

    min_x1 = col1.slider(
        "x1 tối thiểu - Hạ tầng số",
        min_value=0,
        max_value=60,
        value=25,
        step=1
    )

    min_x2 = col2.slider(
        "x2 tối thiểu - AI & dữ liệu",
        min_value=0,
        max_value=60,
        value=15,
        step=1
    )

    min_x3 = col3.slider(
        "x3 tối thiểu - Nhân lực số",
        min_value=0,
        max_value=60,
        value=20,
        step=1
    )

    min_x4 = col4.slider(
        "x4 tối thiểu - R&D",
        min_value=0,
        max_value=60,
        value=10,
        step=1
    )

    strategic_share = st.slider(
        "Tỷ trọng tối thiểu của AI + R&D trong tổng ngân sách",
        min_value=0.10,
        max_value=0.60,
        value=0.35,
        step=0.01
    )

    st.info(
        f"Tổng mức tối thiểu hiện tại = "
        f"{min_x1 + min_x2 + min_x3 + min_x4:.0f} nghìn tỷ VND."
    )

    st.divider()

    st.subheader("3. Kết quả tối ưu")

    res = solve_lp_budget(
        B=B,
        min_x1=min_x1,
        min_x2=min_x2,
        min_x3=min_x3,
        min_x4=min_x4,
        strategic_share=strategic_share
    )

    if not res.success:
        st.error("Bài toán không khả thi với bộ tham số hiện tại.")
        st.write("Gợi ý: Hãy tăng tổng ngân sách hoặc giảm các mức tối thiểu.")
        return

    x1, x2, x3, x4 = res.x
    Z = -res.fun

    total_budget_used = x1 + x2 + x3 + x4
    strategic_ratio = (x2 + x4) / total_budget_used if total_budget_used > 0 else 0

    col_a, col_b, col_c = st.columns(3)

    col_a.metric("Z* - GDP gain tối ưu", f"{Z:.2f}")
    col_b.metric("Ngân sách sử dụng", f"{total_budget_used:.2f}")
    col_c.metric("Tỷ trọng AI + R&D", f"{strategic_ratio * 100:.2f}%")

    result_df = pd.DataFrame({
        "Biến": ["x1", "x2", "x3", "x4"],
        "Hạng mục": [
            "Hạ tầng số",
            "AI & dữ liệu",
            "Nhân lực số",
            "R&D công nghệ"
        ],
        "Phân bổ tối ưu": [x1, x2, x3, x4],
        "Hệ số tác động GDP": [0.85, 1.20, 0.95, 1.35],
        "Đóng góp vào Z": [
            0.85 * x1,
            1.20 * x2,
            0.95 * x3,
            1.35 * x4
        ]
    })

    st.dataframe(result_df.round(2), use_container_width=True)

    st.divider()

    st.subheader("4. Biểu đồ phân bổ ngân sách tối ưu")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(result_df["Hạng mục"], result_df["Phân bổ tối ưu"])
    ax.set_title("Phân bổ ngân sách tối ưu cho 4 hạng mục")
    ax.set_ylabel("Nghìn tỷ VND")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

    st.subheader("5. Biểu đồ đóng góp vào GDP gain")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(result_df["Hạng mục"], result_df["Đóng góp vào Z"])
    ax2.set_title("Đóng góp của từng hạng mục vào Z*")
    ax2.set_ylabel("GDP gain kỳ vọng")
    ax2.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig2)

    st.divider()

    st.subheader("6. Phân tích độ nhạy theo ngân sách")

    budget_list = [100, 120, 140]
    z_list = []
    status_list = []

    for B_test in budget_list:
        res_test = solve_lp_budget(
            B=B_test,
            min_x1=min_x1,
            min_x2=min_x2,
            min_x3=min_x3,
            min_x4=min_x4,
            strategic_share=strategic_share
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

    st.dataframe(sensitivity_df.round(2), use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(sensitivity_df["Ngân sách B"], sensitivity_df["Z*"], marker="o")
    ax3.set_title("Đường cong Z*(B)")
    ax3.set_xlabel("Ngân sách, nghìn tỷ VND")
    ax3.set_ylabel("GDP gain tối ưu")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

    st.divider()

    st.subheader("7. Kịch bản ưu tiên nhân lực số")

    st.markdown("""
    Theo yêu cầu mở rộng của bài, Chính phủ có thể muốn tăng mức tối thiểu của nhân lực số lên **x3 ≥ 30**.
    """)

    res_human = solve_lp_budget(
        B=B,
        min_x1=min_x1,
        min_x2=min_x2,
        min_x3=30,
        min_x4=min_x4,
        strategic_share=strategic_share
    )

    if res_human.success:
        Z_human = -res_human.fun
        delta_Z = Z_human - Z

        col_h1, col_h2 = st.columns(2)
        col_h1.metric("Z* khi x3 ≥ 30", f"{Z_human:.2f}")
        col_h2.metric("Thay đổi so với gốc", f"{delta_Z:.2f}")

        human_df = pd.DataFrame({
            "Biến": ["x1", "x2", "x3", "x4"],
            "Hạng mục": [
                "Hạ tầng số",
                "AI & dữ liệu",
                "Nhân lực số",
                "R&D công nghệ"
            ],
            "Phân bổ khi x3 ≥ 30": res_human.x
        })

        st.dataframe(human_df.round(2), use_container_width=True)
    else:
        st.warning("Kịch bản x3 ≥ 30 không khả thi với ngân sách và ràng buộc hiện tại.")

    st.divider()

    st.subheader("8. Nhận xét tự động")

    max_row = result_df.loc[result_df["Phân bổ tối ưu"].idxmax()]

    st.info(
        f"Hạng mục được phân bổ nhiều nhất là **{max_row['Hạng mục']}**, "
        f"với {max_row['Phân bổ tối ưu']:.2f} nghìn tỷ VND."
    )

    if x4 >= max(x1, x2, x3):
        st.success(
            "R&D được ưu tiên mạnh do có hệ số tác động GDP cao nhất trong hàm mục tiêu."
        )
    elif x2 >= max(x1, x3, x4):
        st.success(
            "AI & dữ liệu được phân bổ cao, phản ánh vai trò của công nghệ chiến lược trong tăng trưởng số."
        )
    else:
        st.write(
            "Kết quả phân bổ không chỉ phụ thuộc vào hệ số tác động GDP, "
            "mà còn phụ thuộc vào các ràng buộc tối thiểu và tỷ trọng công nghệ chiến lược."
        )

    st.warning(
        "Lưu ý: Đây là mô hình tuyến tính đơn giản. Kết quả nên được hiểu như một mô phỏng chính sách, "
        "không phải khuyến nghị ngân sách cuối cùng."
    )

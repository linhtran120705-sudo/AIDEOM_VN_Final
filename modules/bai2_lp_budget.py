import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.optimize import linprog

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# =========================================================
# BÀI 2 — PHÂN BỔ NGÂN SÁCH ĐƠN GIẢN THEO 4 HẠNG MỤC ĐẦU TƯ SỐ
# =========================================================


# ---------------------------------------------------------
# 1. DỮ LIỆU GỐC CỦA BÀI TOÁN
# ---------------------------------------------------------
def get_base_data():
    items = pd.DataFrame({
        "Ký hiệu": ["x₁", "x₂", "x₃", "x₄"],
        "Hạng mục": [
            "Hạ tầng số",
            "AI và dữ liệu",
            "Nhân lực số",
            "R&D công nghệ"
        ],
        "Tên ngắn": ["I", "AI", "H", "R&D"],
        "Hệ số tác động GDP": [0.85, 1.20, 0.95, 1.35],
        "Sàn đầu tư, nghìn tỷ VND": [25, 15, 20, 10],
        "Diễn giải": [
            "Đầu tư vào hạ tầng số, mạng, nền tảng kết nối, trung tâm dữ liệu cơ bản.",
            "Đầu tư vào AI, dữ liệu lớn, nền tảng phân tích và tự động hóa.",
            "Đào tạo, nâng kỹ năng số, kỹ sư AI, nhân lực công nghệ.",
            "Nghiên cứu, đổi mới sáng tạo, công nghệ lõi và năng lực dài hạn."
        ]
    })

    budget_default = 100
    strategic_share = 0.35

    return items, budget_default, strategic_share


# ---------------------------------------------------------
# 2. HÀM GIẢI BÀI TOÁN BẰNG SCIPY
# ---------------------------------------------------------
def solve_with_scipy(budget=100, min_h=20):
    """
    Bài toán gốc:
    Max Z = 0.85x1 + 1.20x2 + 0.95x3 + 1.35x4

    Đưa về dạng minimize:
    Min -Z
    """

    c = [-0.85, -1.20, -0.95, -1.35]

    A_ub = [
        [1, 1, 1, 1],                 # x1 + x2 + x3 + x4 <= budget
        [-1, 0, 0, 0],                # x1 >= 25
        [0, -1, 0, 0],                # x2 >= 15
        [0, 0, -1, 0],                # x3 >= min_h
        [0, 0, 0, -1],                # x4 >= 10
        [0.35, -0.65, 0.35, -0.65],   # x2 + x4 >= 35% tổng ngân sách dùng
    ]

    b_ub = [
        budget,
        -25,
        -15,
        -min_h,
        -10,
        0
    ]

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * 4,
        method="highs"
    )

    if res.success:
        x = res.x
        z = -res.fun

        result = pd.DataFrame({
            "Ký hiệu": ["x₁", "x₂", "x₃", "x₄"],
            "Hạng mục": ["Hạ tầng số", "AI và dữ liệu", "Nhân lực số", "R&D công nghệ"],
            "Phân bổ tối ưu, nghìn tỷ VND": x,
            "Hệ số tác động GDP": [0.85, 1.20, 0.95, 1.35],
            "GDP kỳ vọng tạo thêm": x * np.array([0.85, 1.20, 0.95, 1.35])
        })

        result["Tỷ trọng ngân sách, %"] = result["Phân bổ tối ưu, nghìn tỷ VND"] / result["Phân bổ tối ưu, nghìn tỷ VND"].sum() * 100

        return {
            "success": True,
            "x": x,
            "z": z,
            "table": result,
            "message": res.message
        }

    return {
        "success": False,
        "x": None,
        "z": None,
        "table": None,
        "message": res.message
    }


# ---------------------------------------------------------
# 3. HÀM GIẢI BÀI TOÁN BẰNG PULP
# ---------------------------------------------------------
def solve_with_pulp(budget=100, min_h=20):
    if not PULP_AVAILABLE:
        return None

    model = pulp.LpProblem("VN_Digital_Budget_LP", pulp.LpMaximize)

    x1 = pulp.LpVariable("x1_Ha_tang_so", lowBound=0)
    x2 = pulp.LpVariable("x2_AI_va_du_lieu", lowBound=0)
    x3 = pulp.LpVariable("x3_Nhan_luc_so", lowBound=0)
    x4 = pulp.LpVariable("x4_RD_cong_nghe", lowBound=0)

    # Hàm mục tiêu
    model += 0.85 * x1 + 1.20 * x2 + 0.95 * x3 + 1.35 * x4, "GDP_gain"

    # Ràng buộc
    model += x1 + x2 + x3 + x4 <= budget, "C1_Ngan_sach_tong"
    model += x1 >= 25, "C2_San_ha_tang_so"
    model += x2 >= 15, "C3_San_AI_du_lieu"
    model += x3 >= min_h, "C4_San_nhan_luc_so"
    model += x4 >= 10, "C5_San_RD_cong_nghe"
    model += x2 + x4 >= 0.35 * (x1 + x2 + x3 + x4), "C6_Ty_trong_cong_nghe_chien_luoc"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    values = {
        "x₁ - Hạ tầng số": pulp.value(x1),
        "x₂ - AI và dữ liệu": pulp.value(x2),
        "x₃ - Nhân lực số": pulp.value(x3),
        "x₄ - R&D công nghệ": pulp.value(x4),
    }

    allocation = pd.DataFrame({
        "Hạng mục": list(values.keys()),
        "Phân bổ, nghìn tỷ VND": list(values.values()),
        "Hệ số tác động GDP": [0.85, 1.20, 0.95, 1.35],
    })

    allocation["GDP kỳ vọng tạo thêm"] = (
        allocation["Phân bổ, nghìn tỷ VND"] * allocation["Hệ số tác động GDP"]
    )

    shadow_rows = []
    for name, constraint in model.constraints.items():
        shadow_rows.append({
            "Ràng buộc": name,
            "Shadow price / Dual value": constraint.pi,
            "Slack": constraint.slack,
            "Diễn giải nhanh": interpret_shadow_price(name, constraint.pi, constraint.slack)
        })

    shadow_table = pd.DataFrame(shadow_rows)

    return {
        "status": status,
        "objective": pulp.value(model.objective),
        "allocation": allocation,
        "shadow_table": shadow_table
    }


def interpret_shadow_price(name, pi, slack):
    if abs(pi) < 1e-7:
        return "Ràng buộc không làm thay đổi giá trị tối ưu tại nghiệm hiện tại."
    if "Ngan_sach_tong" in name:
        return f"Nếu tăng ngân sách thêm 1 nghìn tỷ VND, GDP kỳ vọng có thể tăng khoảng {pi:.2f} nghìn tỷ VND."
    if "San" in name:
        return "Ràng buộc sàn đang ảnh hưởng đến nghiệm tối ưu; nếu nới/làm chặt ràng buộc, Z* có thể thay đổi."
    if "Ty_trong" in name:
        return "Ràng buộc tỷ trọng công nghệ chiến lược có tác động đến cấu trúc phân bổ AI + R&D."
    return "Ràng buộc có tác động biên đến nghiệm tối ưu."


# ---------------------------------------------------------
# 4. PHÂN TÍCH ĐỘ NHẠY
# ---------------------------------------------------------
def sensitivity_budget(budgets, min_h=20):
    rows = []

    for b in budgets:
        res = solve_with_scipy(budget=b, min_h=min_h)

        if res["success"]:
            x = res["x"]
            rows.append({
                "Ngân sách B, nghìn tỷ VND": b,
                "Z* - GDP kỳ vọng tăng thêm": res["z"],
                "x₁ Hạ tầng số": x[0],
                "x₂ AI và dữ liệu": x[1],
                "x₃ Nhân lực số": x[2],
                "x₄ R&D công nghệ": x[3],
                "AI + R&D, %": (x[1] + x[3]) / x.sum() * 100
            })
        else:
            rows.append({
                "Ngân sách B, nghìn tỷ VND": b,
                "Z* - GDP kỳ vọng tăng thêm": np.nan,
                "x₁ Hạ tầng số": np.nan,
                "x₂ AI và dữ liệu": np.nan,
                "x₃ Nhân lực số": np.nan,
                "x₄ R&D công nghệ": np.nan,
                "AI + R&D, %": np.nan
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 5. PHẦN 2.1 — BỐI CẢNH VIỆT NAM
# ---------------------------------------------------------
def show_context():
    st.header("2.1. Bối cảnh Việt Nam")

    st.markdown("""
    Bài 2 đặt Việt Nam vào một tình huống ra quyết định ngân sách: Nhà nước có **100.000 tỷ VND**
    để phân bổ cho 4 hạng mục đầu tư số trong năm 2026. Mục tiêu không chỉ là chi hết ngân sách,
    mà là phân bổ sao cho **GDP kỳ vọng tăng thêm lớn nhất**, đồng thời vẫn bảo đảm các ngưỡng tối thiểu
    cho hạ tầng số, AI, nhân lực số và R&D.
    """)

    items, budget, strategic_share = get_base_data()

    c1, c2, c3 = st.columns(3)
    c1.metric("Ngân sách giả định", "100", "nghìn tỷ VND")
    c2.metric("Số hạng mục đầu tư", "4", "I, AI, H, R&D")
    c3.metric("Tỷ trọng AI + R&D tối thiểu", "35%", "công nghệ chiến lược")

    st.subheader("Ảnh minh họa: Bản đồ tư duy phân bổ ngân sách số 2026")

    flow = go.Figure()

    flow.add_trace(go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.4),
            label=[
                "Ngân sách số 2026\n100 nghìn tỷ VND",
                "Hạ tầng số\nx₁",
                "AI và dữ liệu\nx₂",
                "Nhân lực số\nx₃",
                "R&D công nghệ\nx₄",
                "Tăng GDP kỳ vọng\nZ*"
            ],
        ),
        link=dict(
            source=[0, 0, 0, 0, 1, 2, 3, 4],
            target=[1, 2, 3, 4, 5, 5, 5, 5],
            value=[25, 15, 20, 10, 21.25, 18, 19, 13.5],
        )
    ))

    flow.update_layout(
        title_text="Ảnh 2.1 — Dòng ngân sách số và tác động kỳ vọng đến GDP",
        height=500
    )

    st.plotly_chart(flow, use_container_width=True)

    st.markdown("""
    **Ý nghĩa của ảnh minh họa:** ngân sách được chia thành 4 “kênh đầu tư”. Mỗi kênh có hệ số tác động GDP khác nhau.
    Vì vậy, bài toán LP sẽ tự động tìm cách phân bổ nhiều hơn cho hạng mục có hiệu quả biên cao, nhưng vẫn phải tôn trọng
    các ràng buộc tối thiểu và tỷ trọng chiến lược.
    """)

    st.subheader("Bảng bối cảnh 4 hạng mục đầu tư")

    st.dataframe(items, use_container_width=True)

    fig_coef = px.bar(
        items,
        x="Hạng mục",
        y="Hệ số tác động GDP",
        text="Hệ số tác động GDP",
        title="Ảnh 2.2 — So sánh hệ số tác động GDP của 4 hạng mục đầu tư số"
    )
    fig_coef.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_coef.update_layout(height=450)
    st.plotly_chart(fig_coef, use_container_width=True)

    fig_floor = px.pie(
        items,
        names="Hạng mục",
        values="Sàn đầu tư, nghìn tỷ VND",
        title="Ảnh 2.3 — Cơ cấu ngân sách tối thiểu theo ràng buộc chính sách"
    )
    fig_floor.update_layout(height=450)
    st.plotly_chart(fig_floor, use_container_width=True)

    st.info(
        "Thông điệp bối cảnh: R&D có hệ số tác động cao nhất, nhưng không thể dồn toàn bộ ngân sách vào R&D "
        "vì chính sách yêu cầu vẫn phải bảo đảm hạ tầng số, AI-dữ liệu và nhân lực số. Đây chính là lý do cần dùng "
        "quy hoạch tuyến tính để hỗ trợ ra quyết định."
    )


# ---------------------------------------------------------
# 6. PHẦN 2.2 + 2.3 — MÔ HÌNH TOÁN HỌC
# ---------------------------------------------------------
def show_math_model():
    st.header("2.2. Mô hình toán học và diễn giải hệ số mục tiêu")

    st.subheader("Bước 1 — Xác định biến quyết định")

    st.markdown("""
    Mỗi biến quyết định biểu thị số ngân sách phân bổ cho một hạng mục đầu tư số.
    Đơn vị của biến là **nghìn tỷ VND**.
    """)

    variables = pd.DataFrame({
        "Biến": ["x₁", "x₂", "x₃", "x₄"],
        "Ý nghĩa": [
            "Đầu tư hạ tầng số",
            "Đầu tư AI và dữ liệu",
            "Đầu tư nhân lực số",
            "Đầu tư R&D công nghệ"
        ],
        "Đơn vị": ["nghìn tỷ VND"] * 4
    })

    st.dataframe(variables, use_container_width=True)

    st.subheader("Bước 2 — Hàm mục tiêu")

    st.latex(r"""
    \max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4
    """)

    st.markdown("""
    Hàm mục tiêu cho biết GDP kỳ vọng tăng thêm từ mỗi phương án phân bổ ngân sách.
    Hệ số càng lớn nghĩa là cùng 1 nghìn tỷ VND đầu tư, hạng mục đó tạo ra GDP kỳ vọng cao hơn.
    """)

    coef_table = pd.DataFrame({
        "Hạng mục": ["Hạ tầng số", "AI và dữ liệu", "Nhân lực số", "R&D công nghệ"],
        "Biến": ["x₁", "x₂", "x₃", "x₄"],
        "Hệ số": [0.85, 1.20, 0.95, 1.35],
        "Diễn giải": [
            "1 nghìn tỷ VND vào hạ tầng số tạo 0,85 nghìn tỷ VND GDP kỳ vọng.",
            "1 nghìn tỷ VND vào AI và dữ liệu tạo 1,20 nghìn tỷ VND GDP kỳ vọng.",
            "1 nghìn tỷ VND vào nhân lực số tạo 0,95 nghìn tỷ VND GDP kỳ vọng.",
            "1 nghìn tỷ VND vào R&D tạo 1,35 nghìn tỷ VND GDP kỳ vọng."
        ]
    })

    st.dataframe(coef_table, use_container_width=True)

    fig = px.bar(
        coef_table,
        x="Hạng mục",
        y="Hệ số",
        text="Hệ số",
        title="Ảnh 2.4 — R&D có hệ số tác động GDP cao nhất"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bước 3 — Hệ ràng buộc")

    st.latex(r"""
    x_1 + x_2 + x_3 + x_4 \leq 100
    """)

    st.latex(r"""
    x_1 \geq 25,\quad x_2 \geq 15,\quad x_3 \geq 20,\quad x_4 \geq 10
    """)

    st.latex(r"""
    x_2 + x_4 \geq 0.35(x_1+x_2+x_3+x_4)
    """)

    st.latex(r"""
    x_1,x_2,x_3,x_4 \geq 0
    """)

    constraints = pd.DataFrame({
        "Ràng buộc": [
            "Ngân sách tổng",
            "Sàn hạ tầng số",
            "Sàn AI và dữ liệu",
            "Sàn nhân lực số",
            "Sàn R&D công nghệ",
            "Tỷ trọng công nghệ chiến lược",
            "Không âm"
        ],
        "Công thức": [
            "x₁ + x₂ + x₃ + x₄ ≤ 100",
            "x₁ ≥ 25",
            "x₂ ≥ 15",
            "x₃ ≥ 20",
            "x₄ ≥ 10",
            "x₂ + x₄ ≥ 0,35(x₁+x₂+x₃+x₄)",
            "x₁,x₂,x₃,x₄ ≥ 0"
        ],
        "Ý nghĩa chính sách": [
            "Tổng chi không vượt quá 100.000 tỷ VND.",
            "Bảo đảm nền tảng hạ tầng số tối thiểu.",
            "Không bỏ qua AI và dữ liệu.",
            "Đáp ứng yêu cầu phát triển kỹ năng và nhân lực số.",
            "Duy trì năng lực đổi mới công nghệ dài hạn.",
            "AI và R&D phải chiếm ít nhất 35% tổng ngân sách được sử dụng.",
            "Không thể đầu tư âm."
        ]
    })

    st.dataframe(constraints, use_container_width=True)

    st.success(
        "Tư duy mô hình: bài toán không đơn thuần chọn hạng mục có hệ số cao nhất. "
        "Nó phải cân bằng giữa hiệu quả GDP kỳ vọng và các yêu cầu chính sách tối thiểu."
    )


# ---------------------------------------------------------
# 7. PHẦN 2.4 — GIẢI BÀI TOÁN LẬP TRÌNH
# ---------------------------------------------------------
def show_programming_solution():
    st.header("2.4. Giải bài toán lập trình")

    st.markdown("""
    Phần này giải lần lượt 4 yêu cầu: dùng `scipy.optimize.linprog`, dùng `PuLP`,
    phân tích độ nhạy ngân sách và kiểm tra kịch bản ưu tiên nhân lực số.
    """)

    st.subheader("Thiết lập kịch bản")

    c1, c2 = st.columns(2)
    budget = c1.slider(
        "Ngân sách tổng B, nghìn tỷ VND",
        min_value=80,
        max_value=160,
        value=100,
        step=10,
        key="bai2_budget_slider"
    )
    min_h = c2.slider(
        "Sàn nhân lực số x₃, nghìn tỷ VND",
        min_value=20,
        max_value=50,
        value=20,
        step=5,
        key="bai2_min_h_slider"
    )

    # -----------------------------------------------------
    # 2.4.1 Scipy
    # -----------------------------------------------------
    st.subheader("Câu 2.4.1 — Giải bằng scipy.optimize.linprog")

    scipy_result = solve_with_scipy(budget=budget, min_h=min_h)

    if not scipy_result["success"]:
        st.error(f"Bài toán không giải được: {scipy_result['message']}")
        return None

    x = scipy_result["x"]
    z = scipy_result["z"]
    result_table = scipy_result["table"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Trạng thái", "Tối ưu")
    m2.metric("Z* - GDP kỳ vọng", f"{z:,.2f}", "nghìn tỷ VND")
    m3.metric("Tổng ngân sách dùng", f"{x.sum():,.2f}", "nghìn tỷ VND")

    st.dataframe(result_table.round(2), use_container_width=True)

    fig_alloc = px.bar(
        result_table,
        x="Hạng mục",
        y="Phân bổ tối ưu, nghìn tỷ VND",
        text="Phân bổ tối ưu, nghìn tỷ VND",
        title="Ảnh 2.5 — Phân bổ ngân sách tối ưu theo scipy"
    )
    fig_alloc.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_alloc.update_layout(height=450)
    st.plotly_chart(fig_alloc, use_container_width=True)

    fig_gdp = px.bar(
        result_table,
        x="Hạng mục",
        y="GDP kỳ vọng tạo thêm",
        text="GDP kỳ vọng tạo thêm",
        title="Ảnh 2.6 — Đóng góp GDP kỳ vọng theo từng hạng mục"
    )
    fig_gdp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_gdp.update_layout(height=450)
    st.plotly_chart(fig_gdp, use_container_width=True)

    ai_rd_share = (x[1] + x[3]) / x.sum() * 100

    st.info(
        f"Kết quả scipy cho thấy ngân sách tối ưu là: hạ tầng số {x[0]:.2f}, "
        f"AI và dữ liệu {x[1]:.2f}, nhân lực số {x[2]:.2f}, R&D {x[3]:.2f} nghìn tỷ VND. "
        f"Tỷ trọng AI + R&D đạt {ai_rd_share:.2f}%."
    )

    # -----------------------------------------------------
    # 2.4.2 PuLP + shadow price
    # -----------------------------------------------------
    st.subheader("Câu 2.4.2 — Giải bằng PuLP và phân tích shadow price")

    if not PULP_AVAILABLE:
        st.warning(
            "Môi trường hiện tại chưa cài PuLP. Hãy thêm `pulp` vào file requirements.txt rồi deploy lại."
        )
    else:
        pulp_result = solve_with_pulp(budget=budget, min_h=min_h)

        p1, p2 = st.columns(2)
        p1.metric("Trạng thái PuLP", pulp_result["status"])
        p2.metric("Z* theo PuLP", f"{pulp_result['objective']:,.2f}")

        st.dataframe(pulp_result["allocation"].round(2), use_container_width=True)

        st.markdown("#### Bảng giá đối ngẫu / Shadow price")

        st.dataframe(pulp_result["shadow_table"].round(4), use_container_width=True)

        budget_shadow = pulp_result["shadow_table"][
            pulp_result["shadow_table"]["Ràng buộc"] == "C1_Ngan_sach_tong"
        ]

        if not budget_shadow.empty:
            pi_budget = budget_shadow["Shadow price / Dual value"].iloc[0]
            st.success(
                f"Shadow price của ràng buộc ngân sách tổng là khoảng **{pi_budget:.2f}**. "
                f"Diễn giải: nếu tăng thêm 1 nghìn tỷ VND ngân sách, GDP kỳ vọng tối ưu có thể tăng thêm khoảng "
                f"**{pi_budget:.2f} nghìn tỷ VND**, trong phạm vi nghiệm tối ưu hiện tại còn ổn định."
            )

    # -----------------------------------------------------
    # 2.4.3 Độ nhạy ngân sách
    # -----------------------------------------------------
    st.subheader("Câu 2.4.3 — Phân tích độ nhạy ngân sách B = 100, 120, 140")

    sens = sensitivity_budget([100, 120, 140], min_h=20)

    st.dataframe(sens.round(2), use_container_width=True)

    fig_sens = px.line(
        sens,
        x="Ngân sách B, nghìn tỷ VND",
        y="Z* - GDP kỳ vọng tăng thêm",
        markers=True,
        title="Ảnh 2.7 — Đường cong giá trị tối ưu Z*(B) khi tăng ngân sách"
    )
    fig_sens.update_traces(line=dict(width=4), marker=dict(size=10))
    fig_sens.update_layout(height=450)
    st.plotly_chart(fig_sens, use_container_width=True)

    sens_long = sens.melt(
        id_vars=["Ngân sách B, nghìn tỷ VND"],
        value_vars=["x₁ Hạ tầng số", "x₂ AI và dữ liệu", "x₃ Nhân lực số", "x₄ R&D công nghệ"],
        var_name="Hạng mục",
        value_name="Phân bổ, nghìn tỷ VND"
    )

    fig_stack = px.bar(
        sens_long,
        x="Ngân sách B, nghìn tỷ VND",
        y="Phân bổ, nghìn tỷ VND",
        color="Hạng mục",
        title="Ảnh 2.8 — Cơ cấu phân bổ thay đổi khi ngân sách tăng"
    )
    fig_stack.update_layout(height=480)
    st.plotly_chart(fig_stack, use_container_width=True)

    st.info(
        "Diễn giải độ nhạy: khi ngân sách tăng, mô hình thường ưu tiên rót thêm vào hạng mục có hệ số tác động cao, "
        "đặc biệt là R&D, sau khi đã bảo đảm các mức sàn chính sách."
    )

    # -----------------------------------------------------
    # 2.4.4 Ưu tiên nhân lực số
    # -----------------------------------------------------
    st.subheader("Câu 2.4.4 — Kịch bản ưu tiên nhân lực số: x₃ ≥ 30")

    base_case = solve_with_scipy(budget=100, min_h=20)
    human_case = solve_with_scipy(budget=100, min_h=30)

    compare_rows = []

    if base_case["success"]:
        compare_rows.append({
            "Kịch bản": "Gốc: x₃ ≥ 20",
            "Trạng thái": "Khả thi",
            "Z*": base_case["z"],
            "x₁": base_case["x"][0],
            "x₂": base_case["x"][1],
            "x₃": base_case["x"][2],
            "x₄": base_case["x"][3],
        })

    if human_case["success"]:
        compare_rows.append({
            "Kịch bản": "Ưu tiên nhân lực: x₃ ≥ 30",
            "Trạng thái": "Khả thi",
            "Z*": human_case["z"],
            "x₁": human_case["x"][0],
            "x₂": human_case["x"][1],
            "x₃": human_case["x"][2],
            "x₄": human_case["x"][3],
        })
    else:
        compare_rows.append({
            "Kịch bản": "Ưu tiên nhân lực: x₃ ≥ 30",
            "Trạng thái": "Không khả thi",
            "Z*": np.nan,
            "x₁": np.nan,
            "x₂": np.nan,
            "x₃": np.nan,
            "x₄": np.nan,
        })

    compare = pd.DataFrame(compare_rows)
    st.dataframe(compare.round(2), use_container_width=True)

    if human_case["success"]:
        delta_z = human_case["z"] - base_case["z"]
        st.success(
            f"Kịch bản x₃ ≥ 30 vẫn khả thi. Z* thay đổi từ {base_case['z']:.2f} lên "
            f"{human_case['z']:.2f}, tức thay đổi {delta_z:.2f} nghìn tỷ VND. "
            "Nếu Z* giảm, điều đó phản ánh chi phí cơ hội của việc ưu tiên nhân lực số so với phân bổ thuần túy theo hiệu quả GDP ngắn hạn."
        )
    else:
        st.error(
            "Kịch bản x₃ ≥ 30 không khả thi với các ràng buộc hiện tại. Cần tăng ngân sách hoặc nới một số ràng buộc khác."
        )

    return {
        "scipy": scipy_result,
        "sensitivity": sens,
        "base_case": base_case,
        "human_case": human_case
    }


# ---------------------------------------------------------
# 8. PHẦN 2.5 — THẢO LUẬN CHÍNH SÁCH
# ---------------------------------------------------------
def show_policy_discussion():
    st.header("2.5. Câu hỏi thảo luận chính sách")

    base_case = solve_with_scipy(budget=100, min_h=20)
    sens = sensitivity_budget([100, 120, 140], min_h=20)

    if not base_case["success"]:
        st.error("Không thể tạo phần thảo luận vì bài toán gốc không khả thi.")
        return

    x = base_case["x"]
    z = base_case["z"]
    ai_rd_share = (x[1] + x[3]) / x.sum() * 100

    if PULP_AVAILABLE:
        pulp_result = solve_with_pulp(budget=100, min_h=20)
        budget_shadow_table = pulp_result["shadow_table"][
            pulp_result["shadow_table"]["Ràng buộc"] == "C1_Ngan_sach_tong"
        ]
        if not budget_shadow_table.empty:
            shadow_budget = budget_shadow_table["Shadow price / Dual value"].iloc[0]
        else:
            shadow_budget = np.nan
    else:
        # Xấp xỉ shadow price bằng chênh lệch Z* khi tăng ngân sách từ 100 lên 120
        z100 = sens.loc[sens["Ngân sách B, nghìn tỷ VND"] == 100, "Z* - GDP kỳ vọng tăng thêm"].iloc[0]
        z120 = sens.loc[sens["Ngân sách B, nghìn tỷ VND"] == 120, "Z* - GDP kỳ vọng tăng thêm"].iloc[0]
        shadow_budget = (z120 - z100) / 20

    # -----------------------------------------------------
    # Câu a
    # -----------------------------------------------------
    st.subheader("a) Khi ngân sách tổng tăng thêm 1 tỷ VND, GDP kỳ vọng tăng thêm bao nhiêu?")

    c1, c2, c3 = st.columns(3)
    c1.metric("Z* tại B = 100", f"{z:.2f}", "nghìn tỷ VND")
    c2.metric("Shadow price ngân sách", f"{shadow_budget:.2f}")
    c3.metric("Tổng ngân sách dùng", f"{x.sum():.2f}", "nghìn tỷ VND")

    fig_shadow = px.line(
        sens,
        x="Ngân sách B, nghìn tỷ VND",
        y="Z* - GDP kỳ vọng tăng thêm",
        markers=True,
        title="Minh chứng câu a — Z* tăng khi ngân sách tổng tăng"
    )
    fig_shadow.update_traces(line=dict(width=4), marker=dict(size=10))
    fig_shadow.update_layout(height=430)
    st.plotly_chart(fig_shadow, use_container_width=True)

    st.info(
        f"Trả lời: Theo nghiệm tối ưu, nếu tăng thêm 1 nghìn tỷ VND ngân sách, GDP kỳ vọng tăng thêm khoảng "
        f"**{shadow_budget:.2f} nghìn tỷ VND** trong phạm vi nghiệm còn ổn định. "
        "Đây có thể xem là cận trên ngắn hạn của lợi ích biên từ vốn công trong mô hình, nhưng không nên hiểu là con số chắc chắn ngoài thực tế "
        "vì mô hình chưa tính độ trễ, rủi ro triển khai, thất thoát, năng lực hấp thụ và tác động xã hội."
    )

    # -----------------------------------------------------
    # Câu b
    # -----------------------------------------------------
    st.subheader("b) Vì sao R&D có hệ số tác động cao nhất nhưng ràng buộc tối thiểu thấp nhất?")

    items, _, _ = get_base_data()

    fig_rd = go.Figure()

    fig_rd.add_trace(go.Bar(
        x=items["Hạng mục"],
        y=items["Hệ số tác động GDP"],
        name="Hệ số tác động GDP"
    ))

    fig_rd.add_trace(go.Scatter(
        x=items["Hạng mục"],
        y=items["Sàn đầu tư, nghìn tỷ VND"],
        name="Sàn đầu tư",
        mode="lines+markers",
        yaxis="y2"
    ))

    fig_rd.update_layout(
        title="Minh chứng câu b — R&D có hệ số cao nhưng sàn chính sách thấp",
        yaxis=dict(title="Hệ số tác động GDP"),
        yaxis2=dict(title="Sàn đầu tư, nghìn tỷ VND", overlaying="y", side="right"),
        height=450
    )

    st.plotly_chart(fig_rd, use_container_width=True)

    rd_coef = items.loc[items["Hạng mục"] == "R&D công nghệ", "Hệ số tác động GDP"].iloc[0]
    rd_floor = items.loc[items["Hạng mục"] == "R&D công nghệ", "Sàn đầu tư, nghìn tỷ VND"].iloc[0]

    st.success(
        f"Trả lời: R&D có hệ số cao nhất, bằng **{rd_coef:.2f}**, nhưng sàn tối thiểu chỉ **{rd_floor:.0f} nghìn tỷ VND** "
        "vì R&D thường có độ trễ dài, rủi ro cao và đòi hỏi năng lực hấp thụ công nghệ. "
        "Trong quản lý ngân sách, Nhà nước vẫn phải ưu tiên nền tảng bắt buộc như hạ tầng số và nhân lực số trước. "
        "Do đó, sàn thấp không có nghĩa R&D kém quan trọng; nó phản ánh sự thận trọng chính sách và giới hạn năng lực triển khai."
    )

    # -----------------------------------------------------
    # Câu c
    # -----------------------------------------------------
    st.subheader("c) Tỷ lệ 35% AI + R&D có khả thi không?")

    allocation = base_case["table"].copy()

    fig_share = px.pie(
        allocation,
        names="Hạng mục",
        values="Phân bổ tối ưu, nghìn tỷ VND",
        title="Minh chứng câu c — Cơ cấu phân bổ tối ưu theo mô hình"
    )
    fig_share.update_layout(height=450)
    st.plotly_chart(fig_share, use_container_width=True)

    c4, c5, c6 = st.columns(3)
    c4.metric("AI + R&D theo nghiệm tối ưu", f"{ai_rd_share:.2f}%")
    c5.metric("Yêu cầu tối thiểu", "35.00%")
    c6.metric("Mức vượt yêu cầu", f"{ai_rd_share - 35:.2f} điểm %")

    if ai_rd_share >= 35:
        st.success(
            f"Trả lời: Trong mô hình, tỷ lệ AI + R&D đạt **{ai_rd_share:.2f}%**, cao hơn mức tối thiểu 35%, "
            "nên xét về mặt toán học là khả thi. Tuy nhiên, trong thực tiễn quản lý ngân sách, khả thi hay không còn phụ thuộc "
            "vào áp lực chi cho hạ tầng giao thông, an sinh xã hội, y tế, giáo dục và khả năng giải ngân dự án công nghệ."
        )
    else:
        st.warning(
            f"Trả lời: Trong nghiệm hiện tại, tỷ lệ AI + R&D chỉ đạt **{ai_rd_share:.2f}%**, thấp hơn yêu cầu 35%, "
            "nên cần điều chỉnh lại ngân sách hoặc tăng ràng buộc dành cho công nghệ chiến lược."
        )

    st.markdown("""
    **Kết luận chính sách ngắn gọn:**

    - Mô hình LP cho thấy phân bổ tối ưu có xu hướng ưu tiên hạng mục có hiệu quả biên cao.
    - R&D và AI là nhóm tạo tác động GDP lớn, nhưng cần đi cùng hạ tầng số và nhân lực số.
    - Shadow price giúp cơ quan quản lý hiểu “giá trị biên” của ngân sách, nhưng không thay thế được đánh giá định tính về rủi ro, công bằng và năng lực triển khai.
    """)


# ---------------------------------------------------------
# 9. HÀM RENDER CHÍNH
# ---------------------------------------------------------
def render():
    st.title("💰 Bài 2 — Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số")

    st.markdown("""
    Bài 2 sử dụng **quy hoạch tuyến tính — Linear Programming (LP)** để hỗ trợ quyết định phân bổ ngân sách số.
    Trọng tâm của bài là tìm phương án phân bổ tối ưu giữa **hạ tầng số, AI và dữ liệu, nhân lực số, R&D công nghệ**
    nhằm tối đa hóa GDP kỳ vọng, đồng thời vẫn tuân thủ các ràng buộc chính sách.
    """)

    tabs = st.tabs([
        "2.1 Bối cảnh",
        "2.2 Mô hình toán học",
        "2.4 Giải lập trình",
        "2.5 Thảo luận chính sách"
    ])

    with tabs[0]:
        show_context()

    with tabs[1]:
        show_math_model()

    with tabs[2]:
        show_programming_solution()

    with tabs[3]:
        show_policy_discussion()

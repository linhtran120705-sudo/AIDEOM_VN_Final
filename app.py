import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules import bai1_cobb_douglas
from modules import bai2_lp_budget
from modules import bai3_priority_sectors
from modules import bai4_lp_region_budget
from modules import bai5_mip_project_selection
from modules import bai6_topsis_ai_regions
from modules import bai7_nsga2_pareto
from modules import bai8_dynamic_optimization
from modules import bai9_ai_labor_market
from modules import bai10_stochastic_programming
from modules import bai11_qlearning_policy
from modules import bai12_aideom_vn_integrated



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
    st.markdown(
        """
        <style>
        .hero-card {
            background: linear-gradient(135deg, rgba(14,165,233,0.22), rgba(139,92,246,0.20), rgba(34,197,94,0.13));
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 24px;
            padding: 30px 34px;
            margin-bottom: 22px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.22);
        }
        .hero-title {
            font-size: 42px;
            font-weight: 900;
            line-height: 1.15;
            margin-bottom: 10px;
        }
        .hero-subtitle {
            font-size: 20px;
            font-weight: 650;
            opacity: 0.92;
            margin-bottom: 12px;
        }
        .hero-note {
            font-size: 16px;
            line-height: 1.65;
            opacity: 0.88;
        }
        .badge {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.13);
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 14px;
            font-weight: 650;
        }
        .section-card {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 16px;
        }
        .small-muted {
            font-size: 14px;
            opacity: 0.78;
            line-height: 1.55;
        }
        .big-number {
            font-size: 30px;
            font-weight: 850;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # HERO SECTION
    # =========================
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🇻🇳 VN AIDEOM-VN</div>
            <div class="hero-subtitle">AI-Driven Decision Optimization Model for Vietnam</div>
            <div class="hero-note">
                Dashboard mô phỏng 12 bài toán ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI.
                Hệ thống kết hợp <b>Python</b>, <b>tối ưu hóa</b>, <b>học tăng cường</b>, 
                <b>mô phỏng chính sách</b> và <b>AI Analyst</b> để chuyển bài toán kinh tế thành mô hình định lượng có thể kiểm chứng.
            </div>
            <br>
            <span class="badge">🐍 Python</span>
            <span class="badge">📊 Streamlit Dashboard</span>
            <span class="badge">🧮 Optimization</span>
            <span class="badge">🤖 Reinforcement Learning</span>
            <span class="badge">🧠 Gemini AI Analyst</span>
            <span class="badge">🇻🇳 Vietnam 2020–2025 Data</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # QUICK MACRO METRICS
    # =========================
    st.subheader("📌 Bức tranh kinh tế Việt Nam tham chiếu nhanh 2024–2025")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GDP 2025", "514,0 tỷ USD", "+8,02%")
    col2.metric("Kinh tế số/GDP", "≈19,5%", "+1,2 điểm %")
    col3.metric("FDI giải ngân 2025", "27,6 tỷ USD", "+8,9%")
    col4.metric("GDP/người 2025", "5.026 USD", "+6,9%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("GDP 2025", "12.847,6 nghìn tỷ VND")
    col6.metric("DN công nghệ số", "80,1 nghìn")
    col7.metric("GII 2025", "Hạng 44/139")
    col8.metric("KH-CN/GDP", "≈2,49%")

    st.caption(
        "Các chỉ tiêu trên lấy theo bảng số liệu tham chiếu nhanh trong đề bài. "
        "Số liệu được làm tròn nhằm phục vụ mô phỏng và giảng dạy."
    )

    # =========================
    # WHY THIS APP
    # =========================
    st.divider()
    st.subheader("🎯 Mục tiêu của web app")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="section-card">
                <div class="big-number">①</div>
                <h4>Chuyển chính sách thành mô hình</h4>
                <p class="small-muted">
                Mỗi bài biến một vấn đề phát triển kinh tế Việt Nam thành mô hình toán học:
                hàm sản xuất, LP, MIP, TOPSIS, Pareto, stochastic programming và Q-learning.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="section-card">
                <div class="big-number">②</div>
                <h4>Chạy thử kịch bản tương tác</h4>
                <p class="small-muted">
                Người dùng có thể chỉnh tham số, ngân sách, trọng số, ràng buộc và cú sốc;
                sau đó xem bảng kết quả, biểu đồ và thay đổi chính sách tương ứng.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="section-card">
                <div class="big-number">③</div>
                <h4>Giải nghĩa kết quả bằng AI</h4>
                <p class="small-muted">
                Tác nhân AI Analyst hỗ trợ diễn giải kết quả theo góc nhìn chính sách công:
                tăng trưởng, bao trùm, rủi ro, công bằng vùng và trách nhiệm giải trình.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================
    # 12 EXERCISES MAP
    # =========================
    st.divider()
    st.subheader("🗺️ Bản đồ 12 bài theo 4 cấp độ")

    level_df = pd.DataFrame({
        "Cấp độ": [
            "🟢 DỄ", "🟢 DỄ", "🟢 DỄ",
            "🟡 TRUNG BÌNH", "🟡 TRUNG BÌNH", "🟡 TRUNG BÌNH",
            "🟠 KHÁ KHÓ", "🟠 KHÁ KHÓ", "🟠 KHÁ KHÓ",
            "🔴 KHÓ", "🔴 KHÓ", "🔴 KHÓ"
        ],
        "Bài": [
            "Bài 1", "Bài 2", "Bài 3",
            "Bài 4", "Bài 5", "Bài 6",
            "Bài 7", "Bài 8", "Bài 9",
            "Bài 10", "Bài 11", "Bài 12"
        ],
        "Tên bài": [
            "Cobb-Douglas + AI",
            "LP ngân sách số",
            "Priority 10 ngành",
            "LP ngành-vùng",
            "MIP 15 dự án",
            "TOPSIS 6 vùng",
            "NSGA-II Pareto",
            "Tối ưu động 2026–2035",
            "Lao động & AI",
            "Stochastic Programming",
            "Q-learning RL",
            "AIDEOM tích hợp"
        ],
        "Trọng tâm": [
            "TFP, growth accounting, GDP 2030",
            "Phân bổ 100 nghìn tỷ cho I/AI/H/R&D",
            "Xếp hạng ngành ưu tiên chuyển đổi số",
            "Công bằng vùng miền trong phân bổ ngân sách",
            "Chọn danh mục dự án số tối ưu",
            "Xếp hạng vùng sẵn sàng AI",
            "Đánh đổi tăng trưởng - bao trùm - môi trường - dữ liệu",
            "Lộ trình đầu tư nhiều năm",
            "Tác động AI đến lao động và kỹ năng",
            "Tối ưu trong điều kiện bất định",
            "Chính sách thích nghi bằng Q-learning",
            "Dashboard tích hợp và khuyến nghị chính sách"
        ],
        "Công cụ chính": [
            "numpy, pandas",
            "scipy.optimize, PuLP",
            "pandas, min-max, heatmap",
            "PuLP, CVXPY",
            "PuLP/CBC",
            "TOPSIS, Entropy",
            "pymoo",
            "cvxpy, numpy",
            "mô phỏng kịch bản",
            "pyomo / scipy",
            "gymnasium, stable-baselines3",
            "streamlit, AI Analyst"
        ]
    })

    st.dataframe(level_df, use_container_width=True, hide_index=True)

    level_count = level_df.groupby("Cấp độ", as_index=False).agg(Số_bài=("Bài", "count"))
    fig_level = px.bar(
        level_count,
        x="Cấp độ",
        y="Số_bài",
        text="Số_bài",
        title="Phân bổ 12 bài theo 4 cấp độ năng lực"
    )
    fig_level.update_traces(textposition="outside")
    fig_level.update_layout(height=400, yaxis_title="Số bài")
    st.plotly_chart(fig_level, use_container_width=True)

    # =========================
    # HIGHLIGHT RESULTS
    # =========================
    st.divider()
    st.subheader("🌟 Một số kết quả nổi bật có thể khám phá trong web")

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown(
            """
            <div class="section-card">
                <h4>🌱 Bài 1 — Tăng trưởng & TFP</h4>
                <p class="small-muted">
                Ước lượng năng suất nhân tố tổng hợp A<sub>t</sub>, so sánh GDP thực tế với GDP dự báo,
                phân rã tăng trưởng 2020–2025 và mô phỏng GDP 2030 khi kinh tế số đạt 30% GDP.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with h2:
        st.markdown(
            """
            <div class="section-card">
                <h4>💰 Bài 2 — Ngân sách số</h4>
                <p class="small-muted">
                Giải LP phân bổ ngân sách cho hạ tầng số, AI & dữ liệu, nhân lực số và R&D.
                Người dùng có thể tăng ngân sách 100 → 120 → 140 để quan sát Z*(B).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with h3:
        st.markdown(
            """
            <div class="section-card">
                <h4>🧠 Bài 11 — Q-learning</h4>
                <p class="small-muted">
                Mô phỏng chính sách kinh tế thích nghi: agent học cách chọn chính sách theo trạng thái
                GDP, số hóa, năng lực AI và rủi ro thất nghiệp.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Static dashboard highlights based on assignment-defined scenarios
    highlight_df = pd.DataFrame({
        "Nhóm kết quả": [
            "Growth accounting",
            "LP ngân sách số",
            "Priority ngành",
            "TOPSIS vùng",
            "Q-learning"
        ],
        "Câu hỏi chính sách": [
            "TFP và số hóa đóng góp thế nào vào tăng trưởng?",
            "Nên phân bổ ngân sách số vào hạng mục nào?",
            "Ngành nào nên ưu tiên chuyển đổi số và AI trước?",
            "Vùng nào sẵn sàng triển khai trung tâm AI?",
            "Chính sách có nên cố định hay thích nghi theo trạng thái?"
        ],
        "Giá trị hiển thị trên web": [
            "A_t, MAPE, GDP 2030, đóng góp K/L/D/AI/H",
            "Z*, shadow price, Z*(B), kịch bản x3 ≥ 30",
            "Priority_i, top-3 ngành, sensitivity theo AI readiness",
            "TOPSIS score, Entropy weights, top-3 vùng",
            "Q-table, policy map, reward, DQN demo"
        ]
    })

    st.dataframe(highlight_df, use_container_width=True, hide_index=True)

    # =========================
    # DATA SOURCES AND SAFETY NOTE
    # =========================
    st.divider()
    st.subheader("📁 Dữ liệu và nguyên tắc sử dụng")

    d1, d2 = st.columns([1.2, 1])

    with d1:
        st.markdown(
            """
            <div class="section-card">
                <h4>📦 Bộ dữ liệu đi kèm</h4>
                <ul>
                    <li><b>vietnam_macro_2020_2025.csv</b>: GDP, FDI, xuất nhập khẩu, lạm phát, năng suất lao động, kinh tế số/GDP.</li>
                    <li><b>vietnam_sectors_2024.csv</b>: 10 ngành, tăng trưởng, xuất khẩu, lao động, AI readiness, rủi ro tự động hóa.</li>
                    <li><b>vietnam_regions_2024.csv</b>: 6 vùng, GRDP/người, FDI, digital index, AI readiness, Gini, R&D.</li>
                    <li><b>vietnam_priorities.csv</b>: tham số chính sách do sinh viên tự tạo.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d2:
        st.markdown(
            """
            <div class="section-card">
                <h4>⚠️ Lưu ý học thuật</h4>
                <p class="small-muted">
                Các số liệu trong dashboard phục vụ mô phỏng và học tập.
                Khi viết luận văn hoặc bài báo, cần truy xuất số liệu gốc từ nguồn chính thức
                và kiểm tra lại đơn vị đo lường, thời điểm công bố, phương pháp tính.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =========================
    # HOW TO USE
    # =========================
    st.divider()
    st.subheader("🚀 Cách sử dụng nhanh")

    st.markdown(
        """
        <div class="section-card">
            <ol>
                <li>Chọn một bài ở thanh menu bên trái.</li>
                <li>Đọc phần bối cảnh và mô hình toán học.</li>
                <li>Điều chỉnh tham số bằng slider hoặc ô nhập liệu.</li>
                <li>Xem bảng kết quả, biểu đồ và phần nhận xét tự động.</li>
                <li>Dùng AI Analyst để hỗ trợ diễn giải kết quả theo góc nhìn chính sách.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        "Gợi ý: nên bắt đầu từ Bài 1 → Bài 2 → Bài 3 để hiểu logic tăng trưởng, tối ưu ngân sách và xếp hạng ưu tiên trước khi sang các bài khó hơn."
    )
elif menu == "🌱 Bài 1 — Cobb-Douglas + AI":
    bai1_cobb_douglas.render()

elif menu == "💰 Bài 2 — LP ngân sách số":
    bai2_lp_budget.render()

elif menu == "📊 Bài 3 — Priority 10 ngành":
    bai3_priority_sectors.render()

elif menu == "🗺️ Bài 4 — LP ngành-vùng":
    bai4_lp_region_budget.render()

elif menu == "🎯 Bài 5 — MIP 15 dự án":
    bai5_mip_project_selection.render()

elif menu == "🏆 Bài 6 — TOPSIS 6 vùng":
    bai6_topsis_ai_regions.render()

elif menu == "🌐 Bài 7 — NSGA-II Pareto":
    bai7_nsga2_pareto.render()

elif menu == "⏳ Bài 8 — Động 2026-2035":
    bai8_dynamic_optimization.render()

elif menu == "👷 Bài 9 — Lao động & AI":
    bai9_ai_labor_market.render()

elif menu == "🎲 Bài 10 — Stochastic SP":
    bai10_stochastic_programming.render()

elif menu == "🤖 Bài 11 — Q-learning RL":
    bai11_qlearning_policy.render()

elif menu == "🧠 Bài 12 — AIDEOM tích hợp":
    bai12_aideom_vn_integrated.render()


else:
    st.warning("Module này sẽ được bổ sung ở bước tiếp theo.")

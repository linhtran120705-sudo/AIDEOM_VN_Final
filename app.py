import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules import bai1_cobb_douglas
from modules import bai2_lp_budget
from modules import bai3_priority_sectors


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
    bai1_cobb_douglas.render()

elif menu == "💰 Bài 2 — LP ngân sách số":
    bai2_lp_budget.render()

elif choice == "📊 Bài 3 — Priority 10 ngành":
    bai3_priority_sectors.render()




else:
    st.warning("Module này sẽ được bổ sung ở bước tiếp theo.")

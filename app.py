import base64
from pathlib import Path

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

# =========================================================
# GLOBAL THEME — ẢNH NỀN TOÀN WEB
# =========================================================
APP_DIR = Path(__file__).resolve().parent
BACKGROUND_IMAGE = APP_DIR / "assets" / "vn_aideom_background.png"
HOMEPAGE_IMAGE = APP_DIR / "assets" / "homepage_intro.jpg"
BACKGROUND_VIDEO = APP_DIR / "assets" / "background_video.mp4"


def _image_to_base64(image_path: Path) -> str:
    """
    Chuyển ảnh nền thành base64 để Streamlit hiển thị ổn định
    khi đưa project lên GitHub/Streamlit Cloud.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return ""


def apply_global_aideom_theme() -> None:
    """
    Giao diện chung cho toàn bộ dashboard:
    - Ảnh nền AI phát triển ở Việt Nam
    - Lớp phủ tối để dễ đọc chữ
    - Card trong suốt kiểu glassmorphism
    - Sidebar và tab đồng bộ với tinh thần AIDEOM-VN
    """
    bg_base64 = _image_to_base64(BACKGROUND_IMAGE)

    if bg_base64:
        background_css = f"""
        background-image:
            linear-gradient(120deg, rgba(2, 6, 23, 0.48), rgba(15, 23, 42, 0.34), rgba(2, 44, 34, 0.40)),
            url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        """
    else:
        background_css = """
        background:
            radial-gradient(circle at 20% 20%, rgba(239, 68, 68, 0.28), transparent 30%),
            radial-gradient(circle at 78% 28%, rgba(14, 165, 233, 0.30), transparent 34%),
            radial-gradient(circle at 55% 85%, rgba(34, 197, 94, 0.22), transparent 32%),
            linear-gradient(135deg, #020617, #0f172a 45%, #052e2b);
        """

    st.markdown(
        f"""
        <style>
        /* Nền toàn bộ app */
        .stApp {{
            {background_css}
            color: #f8fafc;
        }}

        /* Làm header mặc định trong suốt */
        [data-testid="stHeader"] {{
            background: rgba(2, 6, 23, 0.00);
        }}

        /* Vùng nội dung chính có lớp phủ nhẹ để chữ dễ đọc */
        .block-container {{
            padding-top: 2.0rem;
            padding-bottom: 3.0rem;
            background: linear-gradient(180deg, rgba(2, 6, 23, 0.58), rgba(15, 23, 42, 0.44));
            border-radius: 22px;
        }}

        /* Sidebar kiểu kính mờ */
        [data-testid="stSidebar"] > div:first-child {{
            background:
                linear-gradient(180deg, rgba(2, 6, 23, 0.94), rgba(15, 23, 42, 0.88)),
                radial-gradient(circle at 20% 10%, rgba(239, 68, 68, 0.25), transparent 35%),
                radial-gradient(circle at 80% 35%, rgba(14, 165, 233, 0.20), transparent 35%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(16px);
        }}

        /* Chữ trong sidebar */
        [data-testid="stSidebar"] * {{
            color: #e5e7eb;
        }}

        /* Card chung cho hero và section */
        .hero-card, .section-card, .aideom-glass-card {{
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.78), rgba(30, 41, 59, 0.56));
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 24px;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.32);
            backdrop-filter: blur(14px);
        }}

        /* Hero riêng cho trang chủ */
        .hero-card {{
            background:
                linear-gradient(135deg, rgba(220, 38, 38, 0.26), rgba(14, 165, 233, 0.22), rgba(34, 197, 94, 0.14)),
                linear-gradient(135deg, rgba(15, 23, 42, 0.76), rgba(2, 6, 23, 0.46));
        }}

        .hero-title {{
            color: #ffffff;
            text-shadow: 0 3px 18px rgba(0, 0, 0, 0.35);
        }}

        .hero-subtitle {{
            color: #bae6fd;
        }}

        .hero-note, .small-muted {{
            color: #e5e7eb;
        }}

        .badge {{
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #f8fafc;
            backdrop-filter: blur(8px);
        }}

        /* Metric card */
        [data-testid="stMetric"] {{
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.78), rgba(15, 118, 110, 0.18));
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(12px);
        }}

        [data-testid="stMetricLabel"] {{
            color: #cbd5e1;
        }}

        [data-testid="stMetricValue"] {{
            color: #ffffff;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            background: rgba(15, 23, 42, 0.58);
            border-radius: 999px;
            margin-right: 8px;
            border: 1px solid rgba(255, 255, 255, 0.10);
        }}

        button[data-baseweb="tab"] p {{
            color: #e5e7eb;
            font-weight: 650;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.88), rgba(34, 197, 94, 0.72));
            border: 1px solid rgba(255, 255, 255, 0.32);
        }}

        /* Bảng dữ liệu */
        [data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
        }}

        /* Plotly chart container */
        [data-testid="stPlotlyChart"] {{
            background: rgba(15, 23, 42, 0.56);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 20px;
            padding: 10px;
            backdrop-filter: blur(10px);
        }}

        /* Expanders, info/warning boxes */
        [data-testid="stExpander"] {{
            background: rgba(15, 23, 42, 0.62);
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }}

        /* Đảm bảo các khối markdown dễ đọc */
        h1, h2, h3, h4, h5, h6, p, li, label, span {{
            text-shadow: 0 1px 8px rgba(0, 0, 0, 0.20);
        }}

        /* Footer/scrollbar tinh gọn */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(148, 163, 184, 0.45);
            border-radius: 999px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(15, 23, 42, 0.24);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



# =========================================================
# GLOBAL VIDEO BACKGROUND — VIDEO NỀN ĐỘNG TOÀN WEB
# =========================================================
def render_global_video_background() -> None:
    """
    Hiển thị video nền chạy ẩn phía sau toàn bộ web.
    Video phải nằm tại: assets/background_video.mp4
    """
    video_base64 = _image_to_base64(BACKGROUND_VIDEO)

    if not video_base64:
        st.sidebar.warning("Thiếu video nền: assets/background_video.mp4")
        return

    st.sidebar.caption("✅ Đã nạp video nền: assets/background_video.mp4")

    video_html = f"""<style>
html, body {{
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
}}
.stApp {{
    background: transparent !important;
}}
[data-testid="stAppViewContainer"] {{
    background: transparent !important;
}}
[data-testid="stAppViewContainer"] > .main {{
    background: transparent !important;
    position: relative !important;
    z-index: 2 !important;
}}
.block-container {{
    position: relative !important;
    z-index: 3 !important;
    background: linear-gradient(180deg, rgba(2, 6, 23, 0.18), rgba(2, 6, 23, 0.08)) !important;
    border-radius: 22px !important;
}}
[data-testid="stHeader"] {{
    position: relative !important;
    z-index: 10 !important;
    background: rgba(2, 6, 23, 0.00) !important;
}}
[data-testid="stSidebar"] {{
    position: relative !important;
    z-index: 20 !important;
}}
#aideom-bg-video-layer {{
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    overflow: hidden !important;
    z-index: 0 !important;
    pointer-events: none !important;
}}
#aideom-bg-video-layer video {{
    width: 100vw !important;
    height: 100vh !important;
    object-fit: cover !important;
    opacity: 0.82 !important;
    filter: brightness(0.82) contrast(1.18) saturate(1.18) !important;
}}
#aideom-bg-video-overlay {{
    position: fixed !important;
    inset: 0 !important;
    z-index: 1 !important;
    pointer-events: none !important;
    background:
        linear-gradient(120deg, rgba(2, 6, 23, 0.70), rgba(15, 23, 42, 0.48), rgba(2, 44, 34, 0.58)),
        radial-gradient(circle at 18% 18%, rgba(185, 28, 28, 0.24), transparent 32%),
        radial-gradient(circle at 84% 20%, rgba(14, 165, 233, 0.20), transparent 34%),
        radial-gradient(circle at 50% 88%, rgba(34, 197, 94, 0.14), transparent 36%) !important;
}}
</style>
<div id="aideom-bg-video-layer">
<video autoplay muted loop playsinline preload="auto">
<source src="data:video/mp4;base64,{video_base64}" type="video/mp4" />
</video>
</div>
<div id="aideom-bg-video-overlay"></div>"""

    st.markdown(video_html, unsafe_allow_html=True)


# =========================================================
# COLOR TUNING — NỀN NHẠT HƠN, NỘI DUNG ĐẬM HƠN
# =========================================================
def apply_color_readability_tuning() -> None:
    """
    Tinh chỉnh màu tổng thể:
    - Video/nền phía sau sáng và nhẹ hơn để nhìn rõ chuyển động.
    - Khối nội dung chính, card, metric và sidebar đậm hơn để chữ rõ, dễ đọc.
    """
    st.markdown(
        """
        <style>
        /* Chỉ hạ độ sáng video nền phía sau; giữ nguyên toàn bộ nội dung phía trên */
        #aideom-bg-video-layer video {
            opacity: 0.66 !important;
            filter: brightness(0.58) contrast(1.02) saturate(0.92) !important;
        }

        #aideom-bg-video-overlay {
            background:
                linear-gradient(120deg, rgba(2, 6, 23, 0.36), rgba(15, 23, 42, 0.24), rgba(2, 44, 34, 0.30)),
                radial-gradient(circle at 18% 18%, rgba(185, 28, 28, 0.16), transparent 34%),
                radial-gradient(circle at 84% 20%, rgba(14, 165, 233, 0.15), transparent 36%),
                radial-gradient(circle at 50% 88%, rgba(34, 197, 94, 0.10), transparent 38%) !important;
        }

        /* Vùng nội dung chính đậm hơn để tách khỏi nền */
        .block-container {
            background: linear-gradient(180deg, rgba(2, 6, 23, 0.66), rgba(15, 23, 42, 0.50)) !important;
            border: 1px solid rgba(255, 255, 255, 0.13) !important;
            box-shadow: 0 22px 62px rgba(0, 0, 0, 0.34) !important;
            border-radius: 24px !important;
        }

        /* Card nội dung chính rõ và đậm hơn */
        .hero-card, .section-card, .aideom-glass-card {
            background: linear-gradient(135deg, rgba(2, 6, 23, 0.92), rgba(15, 23, 42, 0.82)) !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            box-shadow: 0 22px 56px rgba(0, 0, 0, 0.42) !important;
        }

        .hero-card {
            background:
                radial-gradient(circle at 18% 12%, rgba(220, 38, 38, 0.26), transparent 32%),
                radial-gradient(circle at 82% 24%, rgba(14, 165, 233, 0.28), transparent 36%),
                radial-gradient(circle at 48% 110%, rgba(34, 197, 94, 0.16), transparent 40%),
                linear-gradient(135deg, rgba(2, 6, 23, 0.94), rgba(15, 23, 42, 0.78)) !important;
        }

        [data-testid="stMetric"],
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"],
        [data-testid="stExpander"] {
            background: linear-gradient(135deg, rgba(2, 6, 23, 0.88), rgba(15, 23, 42, 0.78)) !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.34) !important;
        }

        /* Chữ trong nội dung chính đậm và sáng hơn */
        .block-container h1,
        .block-container h2,
        .block-container h3,
        .block-container h4,
        .block-container h5,
        .block-container h6,
        .hero-title,
        .homepage-visual-title {
            color: #ffffff !important;
            font-weight: 900 !important;
            text-shadow: 0 3px 16px rgba(0, 0, 0, 0.55) !important;
        }

        .block-container p,
        .block-container li,
        .block-container label,
        .hero-note,
        .small-muted,
        .homepage-visual-caption {
            color: #f1f5f9 !important;
            font-weight: 600 !important;
            opacity: 1 !important;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.48) !important;
        }

        .hero-subtitle {
            color: #dff7ff !important;
            font-weight: 850 !important;
        }

        .badge {
            background: rgba(15, 23, 42, 0.80) !important;
            border: 1px solid rgba(255, 255, 255, 0.26) !important;
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        /* Sidebar giữ đậm để menu nổi bật */
        [data-testid="stSidebar"] > div:first-child {
            background:
                linear-gradient(180deg, rgba(2, 6, 23, 0.96), rgba(15, 23, 42, 0.91)),
                radial-gradient(circle at 20% 10%, rgba(239, 68, 68, 0.22), transparent 35%),
                radial-gradient(circle at 80% 35%, rgba(14, 165, 233, 0.18), transparent 35%) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# HOME PAGE VISUAL — ẢNH MỞ ĐẦU TRANG CHỦ
# =========================================================
def render_homepage_intro_visual() -> None:
    """
    Hiển thị ảnh mở đầu bằng st.image.
    Cách này tránh lỗi Streamlit in thừa HTML <div> ra giao diện.
    Ảnh cần nằm tại: assets/homepage_intro.jpg
    """
    if not HOMEPAGE_IMAGE.exists():
        st.warning("Chưa tìm thấy ảnh mở đầu. Hãy upload ảnh đúng đường dẫn: assets/homepage_intro.jpg")
        return

    st.image(str(HOMEPAGE_IMAGE), use_container_width=True)


apply_global_aideom_theme()
render_global_video_background()
apply_color_readability_tuning()

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
        @keyframes aideomPulse {
            0%, 100% { transform: scale(1); opacity: 0.72; }
            50% { transform: scale(1.035); opacity: 1; }
        }
        @keyframes aideomFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        @keyframes aideomShimmer {
            0% { transform: translateX(-130%); opacity: 0; }
            18% { opacity: 0.92; }
            55% { opacity: 0.42; }
            100% { transform: translateX(130%); opacity: 0; }
        }
        @keyframes aideomScan {
            0% { transform: translateY(-110%); opacity: 0; }
            20% { opacity: 0.32; }
            100% { transform: translateY(110%); opacity: 0; }
        }
        @keyframes aideomGlowBorder {
            0%, 100% { filter: hue-rotate(0deg); opacity: 0.55; }
            50% { filter: hue-rotate(28deg); opacity: 0.92; }
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 18% 12%, rgba(220, 38, 38, 0.28), transparent 32%),
                radial-gradient(circle at 82% 24%, rgba(14, 165, 233, 0.32), transparent 36%),
                radial-gradient(circle at 48% 110%, rgba(34, 197, 94, 0.20), transparent 40%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.86), rgba(15, 23, 42, 0.50));
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 28px;
            padding: 34px 38px;
            margin-bottom: 24px;
            box-shadow: 0 22px 60px rgba(0,0,0,0.34), 0 0 38px rgba(14,165,233,0.13);
            backdrop-filter: blur(16px);
        }
        .hero-card::before {
            content: "";
            position: absolute;
            inset: -2px;
            background: linear-gradient(115deg, transparent 0%, rgba(255,255,255,0.24) 45%, transparent 70%);
            animation: aideomShimmer 7.2s ease-in-out infinite;
            pointer-events: none;
        }
        .hero-card::after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            right: -80px;
            top: -90px;
            background: radial-gradient(circle, rgba(14,165,233,0.36), transparent 66%);
            animation: aideomPulse 5.8s ease-in-out infinite;
            pointer-events: none;
        }
        .hero-title {
            position: relative;
            z-index: 2;
            font-size: 44px;
            font-weight: 950;
            line-height: 1.12;
            margin-bottom: 10px;
            letter-spacing: -0.8px;
        }
        .hero-subtitle {
            position: relative;
            z-index: 2;
            font-size: 21px;
            font-weight: 750;
            color: #bae6fd;
            opacity: 0.96;
            margin-bottom: 12px;
        }
        .hero-note {
            position: relative;
            z-index: 2;
            font-size: 16px;
            line-height: 1.72;
            opacity: 0.92;
            max-width: 1080px;
        }
        .badge {
            position: relative;
            z-index: 2;
            display: inline-block;
            padding: 8px 13px;
            border-radius: 999px;
            background: rgba(255,255,255,0.11);
            border: 1px solid rgba(255,255,255,0.18);
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 14px;
            font-weight: 750;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), 0 8px 18px rgba(0,0,0,0.14);
            backdrop-filter: blur(10px);
        }
        .homepage-visual-shell {
            position: relative;
            overflow: hidden;
            border-radius: 32px;
            margin: 4px 0 30px 0;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(2, 6, 23, 0.62);
            box-shadow: 0 28px 72px rgba(0,0,0,0.42), 0 0 50px rgba(14,165,233,0.18);
        }
        .homepage-visual-shell::before {
            content: "";
            position: absolute;
            inset: 0;
            padding: 1px;
            border-radius: 32px;
            background: linear-gradient(120deg, rgba(185,28,28,0.78), rgba(14,165,233,0.88), rgba(34,197,94,0.64), rgba(255,255,255,0.18));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            animation: aideomGlowBorder 6s ease-in-out infinite;
            z-index: 5;
            pointer-events: none;
        }
        .homepage-visual-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at 12% 18%, rgba(220,38,38,0.22), transparent 28%),
                radial-gradient(circle at 88% 16%, rgba(14,165,233,0.24), transparent 32%),
                linear-gradient(180deg, rgba(2,6,23,0.04), rgba(2,6,23,0.50));
            z-index: 2;
            pointer-events: none;
        }
        .homepage-intro-img {
            width: 100%;
            display: block;
            border-radius: 30px;
            filter: saturate(1.08) contrast(1.04) brightness(0.98);
            transform: scale(1.002);
        }
        .homepage-visual-glow {
            position: absolute;
            width: 360px;
            height: 360px;
            right: 4%;
            top: 8%;
            background: radial-gradient(circle, rgba(14,165,233,0.34), transparent 68%);
            filter: blur(8px);
            animation: aideomPulse 5.5s ease-in-out infinite;
            z-index: 3;
            pointer-events: none;
        }
        .homepage-scanline {
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, transparent 0%, rgba(125,211,252,0.20) 50%, transparent 100%);
            animation: aideomScan 6.5s linear infinite;
            z-index: 4;
            pointer-events: none;
        }
        .homepage-visual-overlay {
            position: absolute;
            left: 28px;
            bottom: 26px;
            width: min(650px, calc(100% - 56px));
            z-index: 6;
            padding: 18px 20px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(2,6,23,0.68), rgba(15,23,42,0.38));
            border: 1px solid rgba(255,255,255,0.18);
            backdrop-filter: blur(14px);
            box-shadow: 0 18px 42px rgba(0,0,0,0.26);
        }
        .homepage-live-badge {
            display: inline-block;
            padding: 7px 11px;
            border-radius: 999px;
            background: rgba(220,38,38,0.82);
            color: #ffffff;
            font-weight: 850;
            font-size: 12px;
            letter-spacing: 0.8px;
            margin-bottom: 10px;
            box-shadow: 0 0 22px rgba(220,38,38,0.34);
        }
        .homepage-visual-title {
            font-size: 27px;
            line-height: 1.15;
            font-weight: 950;
            color: #ffffff;
            text-shadow: 0 3px 18px rgba(0,0,0,0.40);
            margin-bottom: 6px;
        }
        .homepage-visual-caption {
            color: #dbeafe;
            font-size: 14.5px;
            line-height: 1.55;
        }
        .floating-chip {
            position: absolute;
            z-index: 6;
            padding: 9px 13px;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(15,23,42,0.74), rgba(14,165,233,0.26));
            border: 1px solid rgba(255,255,255,0.22);
            color: #f8fafc;
            font-size: 13px;
            font-weight: 850;
            backdrop-filter: blur(12px);
            box-shadow: 0 14px 28px rgba(0,0,0,0.22);
            animation: aideomFloat 4.6s ease-in-out infinite;
        }
        .chip-1 { right: 5%; top: 12%; animation-delay: 0s; }
        .chip-2 { right: 11%; bottom: 22%; animation-delay: 0.7s; }
        .chip-3 { left: 6%; top: 13%; animation-delay: 1.25s; }
        .section-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(15,23,42,0.68), rgba(30,41,59,0.44));
            border: 1px solid rgba(255,255,255,0.13);
            border-radius: 20px;
            padding: 19px 21px;
            margin-bottom: 16px;
            box-shadow: 0 16px 34px rgba(0,0,0,0.24);
            backdrop-filter: blur(12px);
            transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
        }
        .section-card:hover {
            transform: translateY(-4px);
            border-color: rgba(125,211,252,0.42);
            box-shadow: 0 24px 48px rgba(14,165,233,0.14), 0 18px 34px rgba(0,0,0,0.28);
        }
        .small-muted {
            font-size: 14px;
            opacity: 0.84;
            line-height: 1.60;
        }
        .big-number {
            font-size: 31px;
            font-weight: 900;
            color: #7dd3fc;
            text-shadow: 0 0 18px rgba(14,165,233,0.32);
        }
        @media (max-width: 760px) {
            .hero-title { font-size: 34px; }
            .homepage-visual-overlay { position: relative; left: auto; bottom: auto; width: auto; margin: 14px; }
            .floating-chip { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    apply_color_readability_tuning()

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
    # CINEMATIC HOMEPAGE IMAGE
    # =========================
    render_homepage_intro_visual()


    # =========================
    # INTRO VIDEO SECTION
    # =========================
    st.divider()
    st.subheader("🎬 Video giới thiệu AIDEOM-VN")

    video_path = APP_DIR / "assets" / "intro_video.mp4"

    if video_path.exists():
        st.video(str(video_path))
        st.caption("Video giới thiệu được tải từ thư mục assets/intro_video.mp4 trong repository.")
    else:
        st.warning("Chưa tìm thấy video. Hãy upload file MP4 đúng đường dẫn: assets/intro_video.mp4")
        st.info(
            "Lưu ý: không upload file .rar để phát video trực tiếp. "
            "Bạn cần giải nén file .rar, lấy file video .mp4 bên trong, rồi upload file .mp4 vào thư mục assets."
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

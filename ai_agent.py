import os
import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Đọc file .env khi chạy trên máy cá nhân.
# Trên Streamlit Cloud, app sẽ ưu tiên đọc st.secrets.
load_dotenv()


# =========================================================
# 1. HÀM LẤY CẤU HÌNH GEMINI VÀ THÔNG TIN CHỦ WEB
# =========================================================
def _get_secret_value(key: str) -> Optional[str]:
    """
    Lấy giá trị từ Streamlit Secrets một cách an toàn.
    Nếu chưa có secrets.toml hoặc chưa cấu hình trên Cloud thì không làm app bị lỗi.
    """
    try:
        value = st.secrets.get(key, None)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return None


def _get_gemini_key() -> Optional[str]:
    """
    Lấy Gemini API key theo thứ tự ưu tiên:
    1. Streamlit Cloud / .streamlit/secrets.toml: GEMINI_API_KEY
    2. File .env hoặc biến môi trường: GEMINI_API_KEY
    """
    return _get_secret_value("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")


def _get_gemini_model() -> str:
    """
    Lấy tên model Gemini.
    Có thể đổi trong Streamlit Secrets hoặc .env bằng biến GEMINI_MODEL.
    """
    return (
        _get_secret_value("GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL")
        or "gemini-2.5-flash"
    )


def _get_web_owner_contact() -> str:
    """
    Thông tin liên hệ chủ sở hữu web.
    Không bắt buộc. Có thể cấu hình trong Streamlit Secrets hoặc .env:
    WEB_OWNER_CONTACT = "email hoặc tên nhóm quản trị"
    """
    return (
        _get_secret_value("WEB_OWNER_CONTACT")
        or os.getenv("WEB_OWNER_CONTACT")
        or "chủ sở hữu website / nhóm phát triển dashboard"
    )


def _show_missing_api_key_warning() -> None:
    """
    Cảnh báo khi chưa có GEMINI_API_KEY.
    Hiển thị cho người dùng web để họ biết cần báo chủ sở hữu web.
    """
    owner_contact = _get_web_owner_contact()

    st.warning(
        """
        Chưa cấu hình **GEMINI_API_KEY**, nên tác nhân Gemini chưa thể chạy trực tiếp.
        Website vẫn hoạt động bằng chế độ **phân tích offline dự phòng**.
        """
    )

    st.info(
        f"""
        **Hướng xử lý:** vui lòng thông báo cho **{owner_contact}** bổ sung `GEMINI_API_KEY`
        trong **Streamlit Secrets** hoặc file `.env` để kích hoạt tác nhân Gemini.
        """
    )


def _show_quota_warning() -> None:
    """
    Cảnh báo khi API bị hết quota, hết token hoặc quá giới hạn request.
    """
    owner_contact = _get_web_owner_contact()

    st.error(
        """
        Gemini hiện chưa thể phản hồi vì API có thể đã **hết quota**, **hết token**,
        hoặc vượt giới hạn số lượt gọi trong thời gian ngắn.
        """
    )

    st.info(
        f"""
        **Hướng xử lý:** vui lòng thông báo cho **{owner_contact}** kiểm tra lại hạn mức Gemini API,
        billing/quota, hoặc thay API key/model khác trong **Streamlit Secrets**.
        Trong thời gian chờ xử lý, website sẽ tự chuyển sang chế độ phân tích offline.
        """
    )


def _show_model_overload_warning() -> None:
    """
    Cảnh báo khi model Gemini bị quá tải 503.
    """
    owner_contact = _get_web_owner_contact()

    st.warning(
        """
        Model Gemini đang quá tải tạm thời do nhu cầu sử dụng cao.
        Đây không phải lỗi dữ liệu hoặc lỗi giao diện dashboard.
        """
    )

    st.info(
        f"""
        Người dùng có thể thử lại sau vài phút. Nếu lỗi lặp lại nhiều lần,
        vui lòng thông báo cho **{owner_contact}** đổi `GEMINI_MODEL`
        sang model nhẹ hơn, ví dụ `gemini-2.5-flash`.
        """
    )


def _show_invalid_key_warning() -> None:
    """
    Cảnh báo khi API key sai, hết hạn hoặc chưa được cấp quyền.
    """
    owner_contact = _get_web_owner_contact()

    st.error(
        """
        Gemini API key có thể không hợp lệ, hết hạn, bị sai định dạng
        hoặc chưa được cấp quyền sử dụng model hiện tại.
        """
    )

    st.info(
        f"""
        **Hướng xử lý:** vui lòng thông báo cho **{owner_contact}** kiểm tra lại `GEMINI_API_KEY`
        trong Streamlit Secrets, bảo đảm key còn hiệu lực và không bị copy thiếu ký tự.
        """
    )


# =========================================================
# 2. HÀM CHUẨN HÓA DỮ LIỆU GỬI CHO AI
# =========================================================
def _safe_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Chuẩn hóa metrics để tránh lỗi khi chuyển kiểu numpy/pandas sang JSON.
    """
    clean: Dict[str, Any] = {}

    if not metrics:
        return clean

    for key, value in metrics.items():
        try:
            if hasattr(value, "item"):
                value = value.item()

            if isinstance(value, (int, float, str, bool)) or value is None:
                clean[key] = value
            else:
                clean[key] = str(value)
        except Exception:
            clean[key] = str(value)

    return clean


def _safe_table_preview(result_table: Any, max_rows: int = 8) -> str:
    """
    Chuyển bảng kết quả thành text ngắn để gửi cho Gemini.
    Dùng to_string để không cần thêm thư viện tabulate.
    """
    if result_table is None:
        return "Không có bảng kết quả."

    try:
        if isinstance(result_table, pd.DataFrame):
            return result_table.head(max_rows).to_string(index=False)
        if isinstance(result_table, pd.Series):
            return result_table.head(max_rows).to_string()
        return str(result_table)
    except Exception:
        return "Không đọc được bảng kết quả."


# =========================================================
# 3. TÁC NHÂN OFFLINE DỰ PHÒNG
# =========================================================
def offline_policy_analysis(
    bai_name: str,
    model_goal: str,
    metrics: Optional[Dict[str, Any]],
    policy_questions: str,
) -> str:
    """
    Tác nhân phân tích offline.
    Dùng khi chưa có Gemini API key, gọi Gemini lỗi, hết quota hoặc sai model.
    """
    metrics_clean = _safe_metrics(metrics)

    lines = []
    lines.append(f"### 🤖 Tác nhân AI phân tích kết quả — {bai_name}")
    lines.append("")
    lines.append("**Chế độ:** Phân tích tự động offline. Chưa gọi Gemini API.")
    lines.append("")

    lines.append("#### 1. Mục tiêu mô hình")
    lines.append(model_goal)
    lines.append("")

    lines.append("#### 2. Tóm tắt chỉ số chính")
    if metrics_clean:
        for key, value in metrics_clean.items():
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:,.4f}")
            else:
                lines.append(f"- **{key}**: {value}")
    else:
        lines.append("- Chưa truyền chỉ số định lượng vào tác nhân AI.")
    lines.append("")

    lines.append("#### 3. Diễn giải học thuật")
    lines.append(
        "Kết quả cần được hiểu như một mô phỏng định lượng phục vụ ra quyết định. "
        "Các chỉ số đầu ra cho biết cách giả định về tham số, nguồn lực và ràng buộc "
        "được chuyển hóa thành kết quả kinh tế. Vì vậy, ý nghĩa của mô hình không chỉ "
        "nằm ở con số cuối cùng, mà còn ở quan hệ giữa đầu vào, thuật toán và hàm mục tiêu."
    )
    lines.append("")

    lines.append("#### 4. Hàm ý chính sách")
    lines.append(
        "Về chính sách, kết quả nên được đọc theo ba lớp: hiệu quả kinh tế, tính khả thi "
        "khi triển khai và rủi ro phân bổ nguồn lực. Nếu một yếu tố đầu vào tạo đóng góp lớn, "
        "chính sách không nên chỉ tăng ngân sách cho yếu tố đó, mà cần kiểm tra năng lực hấp thụ, "
        "dữ liệu, nhân lực, thể chế và khả năng phối hợp giữa các bên liên quan."
    )
    lines.append("")

    lines.append("#### 5. Câu hỏi chính sách cần trả lời")
    lines.append(policy_questions if policy_questions else "Chưa nhập câu hỏi chính sách cụ thể.")
    lines.append("")

    lines.append("#### 6. Khuyến nghị sử dụng")
    lines.append(
        "Khi viết báo cáo, nên kết hợp phần phân tích này với bảng kết quả, biểu đồ, "
        "công thức mô hình và giả định tham số. Không nên rút ra kết luận chính sách "
        "tách rời khỏi dữ liệu đầu vào và điều kiện mô phỏng."
    )

    return "\n".join(lines)


# =========================================================
# 4. TÁC NHÂN GEMINI
# =========================================================
def gemini_policy_analysis(
    bai_name: str,
    model_goal: str,
    metrics: Optional[Dict[str, Any]],
    result_table: Any,
    policy_questions: str,
) -> Optional[str]:
    """
    Gọi Gemini để phân tích kết quả.
    Nếu lỗi thì trả về None để app chuyển sang chế độ offline.
    """
    api_key = _get_gemini_key()

    if not api_key:
        _show_missing_api_key_warning()
        return None

    try:
        from google import genai

        model_name = _get_gemini_model()
        client = genai.Client(api_key=api_key)

        metrics_clean = _safe_metrics(metrics)
        table_preview = _safe_table_preview(result_table)

        prompt = f"""
Bạn là một tác nhân AI phân tích chính sách công và mô hình ra quyết định.

Hãy viết bằng tiếng Việt, giọng học thuật, rõ ràng, chắc chắn, không lan man.
Không bịa số liệu ngoài dữ liệu được cung cấp.
Không nói chung chung kiểu "cần thêm nghiên cứu".
Phải bám trực tiếp vào kết quả mô hình, bảng kết quả và chỉ số đầu ra.

Tên bài:
{bai_name}

Mục tiêu mô hình:
{model_goal}

Các chỉ số chính:
{json.dumps(metrics_clean, ensure_ascii=False, indent=2)}

Bảng kết quả tóm tắt:
{table_preview}

Câu hỏi/chủ điểm chính sách cần trả lời:
{policy_questions}

Yêu cầu đầu ra:
1. Tóm tắt kết quả chính.
2. Giải thích thuật toán hoặc logic mô hình.
3. Phân tích ý nghĩa kinh tế của kết quả.
4. Trả lời trực tiếp các câu hỏi chính sách.
5. Nêu hàm ý chính sách cụ thể cho Việt Nam.
6. Nêu 2-3 lưu ý khi diễn giải kết quả.

Viết khoảng 500-800 từ.
"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        if response is not None and getattr(response, "text", None):
            return response.text

        st.warning(
            "Gemini không trả về nội dung phân tích. Website sẽ chuyển sang chế độ offline dự phòng."
        )
        return None

    except Exception as error:
        error_text = str(error)

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
            or "quota" in error_text.lower()
            or "rate limit" in error_text.lower()
            or "token" in error_text.lower()
        ):
            _show_quota_warning()

        elif (
            "503" in error_text
            or "UNAVAILABLE" in error_text
            or "high demand" in error_text.lower()
            or "overloaded" in error_text.lower()
        ):
            _show_model_overload_warning()

        elif (
            "401" in error_text
            or "403" in error_text
            or "API_KEY_INVALID" in error_text
            or "PERMISSION_DENIED" in error_text
            or "invalid api key" in error_text.lower()
        ):
            _show_invalid_key_warning()

        elif (
            "404" in error_text
            or "not found" in error_text.lower()
            or "model" in error_text.lower()
        ):
            st.error(
                "Không gọi được Gemini vì model hiện tại có thể không tồn tại "
                "hoặc chưa được tài khoản API key hỗ trợ."
            )
            st.info(
                "Chủ sở hữu web nên kiểm tra biến `GEMINI_MODEL` trong Streamlit Secrets. "
                "Có thể thử đổi về `gemini-2.5-flash`."
            )

        else:
            st.error("Lỗi khi gọi Gemini. Website sẽ chuyển sang chế độ offline dự phòng.")
            with st.expander("Xem chi tiết lỗi kỹ thuật"):
                st.code(error_text)

        return None


# =========================================================
# 5. KHỐI HIỂN THỊ TRÊN STREAMLIT
# =========================================================
def render_ai_agent(
    bai_name: str,
    model_goal: str,
    metrics: Optional[Dict[str, Any]] = None,
    result_table: Any = None,
    policy_questions: str = "",
    key_suffix: str = "default",
) -> None:
    """
    Hàm hiển thị tác nhân AI trong từng bài.
    Gọi hàm này trong tab AI Analyst của Bài 1, Bài 2, ..., Bài 12.
    """
    st.divider()
    st.subheader("🤖 Tác nhân AI phân tích kết quả")

    st.markdown(
        """
        Tác nhân AI đọc các chỉ số đầu ra của mô hình, giải thích thuật toán,
        phân tích ý nghĩa kinh tế và chuyển kết quả định lượng thành hàm ý chính sách.
        """
    )

    api_key = _get_gemini_key()
    gemini_ready = bool(api_key)

    if gemini_ready:
        st.success("Đã nhận GEMINI_API_KEY. Có thể gọi Gemini.")
    else:
        _show_missing_api_key_warning()

    default_index = 1 if gemini_ready else 0

    mode = st.radio(
        "Chọn chế độ phân tích",
        [
            "Tự động offline — luôn chạy được",
            "Gemini AI — nếu đã cấu hình API key",
        ],
        index=default_index,
        key=f"ai_mode_{key_suffix}",
        horizontal=True,
    )

    if st.button("Tạo phân tích AI", key=f"run_ai_{key_suffix}"):
        with st.spinner("AI đang phân tích kết quả mô hình..."):
            text = None

            if mode == "Gemini AI — nếu đã cấu hình API key":
                text = gemini_policy_analysis(
                    bai_name=bai_name,
                    model_goal=model_goal,
                    metrics=metrics,
                    result_table=result_table,
                    policy_questions=policy_questions,
                )

                if text is None:
                    st.warning(
                        "Gemini chưa thể tạo phân tích ở thời điểm này. "
                        "Website tự chuyển sang chế độ phân tích offline để không gián đoạn trải nghiệm người dùng."
                    )

            if text is None:
                text = offline_policy_analysis(
                    bai_name=bai_name,
                    model_goal=model_goal,
                    metrics=metrics,
                    policy_questions=policy_questions,
                )

        st.markdown(text)

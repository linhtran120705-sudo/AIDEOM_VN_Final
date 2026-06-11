import os
import json
import pandas as pd
import streamlit as st


def _get_gemini_key():
    """
    Lấy Gemini API key theo 2 cách:
    1. Trên Streamlit Cloud: st.secrets["GEMINI_API_KEY"]
    2. Trên máy cá nhân: biến môi trường GEMINI_API_KEY trong file .env
    """
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


def _safe_table_preview(result_table, max_rows=8):
    """
    Chuyển bảng kết quả thành text ngắn để gửi cho AI.
    Tránh gửi bảng quá dài gây tốn token.
    """
    if result_table is None:
        return "Không có bảng kết quả."

    try:
        if isinstance(result_table, pd.DataFrame):
            return result_table.head(max_rows).to_markdown(index=False)
        return str(result_table)
    except Exception:
        return "Không đọc được bảng kết quả."


def _safe_metrics(metrics):
    """
    Chuẩn hóa metrics để tránh lỗi kiểu dữ liệu numpy khi chuyển sang JSON.
    """
    clean = {}
    if not metrics:
        return clean

    for key, value in metrics.items():
        try:
            if hasattr(value, "item"):
                clean[key] = value.item()
            else:
                clean[key] = value
        except Exception:
            clean[key] = str(value)

    return clean


def offline_policy_analysis(bai_name, model_goal, metrics, policy_questions):
    """
    Tác nhân phân tích offline.
    Dùng khi chưa có Gemini API key hoặc Gemini lỗi.
    Mục tiêu: app vẫn chạy được trên GitHub/Streamlit, không bị sập.
    """
    metrics = _safe_metrics(metrics)

    lines = []
    lines.append(f"### 🤖 Tác nhân AI phân tích kết quả — {bai_name}")
    lines.append("")
    lines.append("**Chế độ:** Phân tích tự động offline. Chưa dùng Gemini API hoặc API key chưa được cấu hình.")
    lines.append("")
    lines.append("#### 1. Mục tiêu mô hình")
    lines.append(model_goal)
    lines.append("")
    lines.append("#### 2. Tóm tắt chỉ số chính")

    if metrics:
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:,.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- Chưa truyền chỉ số định lượng vào tác nhân AI.")

    lines.append("")
    lines.append("#### 3. Diễn giải học thuật")
    lines.append(
        "Kết quả cần được hiểu như một mô phỏng định lượng phục vụ ra quyết định. "
        "Các chỉ số đầu ra không chỉ cho biết nghiệm tối ưu hoặc xu hướng dự báo, "
        "mà còn phản ánh cách mô hình chuyển hóa giả định chính sách thành kết quả kinh tế."
    )

    lines.append("")
    lines.append("#### 4. Hàm ý chính sách")
    lines.append(
        "Về chính sách, kết quả nên được đọc theo ba lớp: "
        "thứ nhất là hiệu quả kinh tế, thứ hai là tính khả thi khi triển khai, "
        "và thứ ba là rủi ro phân bổ nguồn lực. "
        "Nếu một biến đầu vào tạo đóng góp lớn, chính sách không nên chỉ tăng ngân sách cho biến đó, "
        "mà cần kiểm tra năng lực hấp thụ, dữ liệu, nhân lực và thể chế đi kèm."
    )

    lines.append("")
    lines.append("#### 5. Câu hỏi chính sách cần trả lời")
    lines.append(policy_questions)

    lines.append("")
    lines.append("#### 6. Khuyến nghị sử dụng")
    lines.append(
        "Khi viết báo cáo, nên kết hợp phần này với bảng kết quả, biểu đồ và công thức mô hình. "
        "Không nên trình bày kết luận chính sách tách rời khỏi giả định tham số."
    )

    return "\n".join(lines)


def gemini_policy_analysis(bai_name, model_goal, metrics, result_table, policy_questions):
    """
    Gọi Gemini để phân tích kết quả.
    Nếu lỗi thì trả về None để app chuyển sang offline.
    """
    api_key = _get_gemini_key()
    if not api_key:
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)

        metrics_clean = _safe_metrics(metrics)
        table_preview = _safe_table_preview(result_table)

        prompt = f"""
Bạn là một tác nhân AI phân tích chính sách công và mô hình ra quyết định.

Hãy viết phân tích bằng tiếng Việt, giọng học thuật, rõ ràng, chắc chắn, không lan man.
Không bịa số liệu ngoài dữ liệu được cung cấp.
Không nói chung chung kiểu "cần thêm nghiên cứu".
Phải bám vào kết quả mô hình.

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

        response = model.generate_content(prompt)

        if response and hasattr(response, "text"):
            return response.text

        return None

    except Exception as e:
        return None


def render_ai_agent(
    bai_name,
    model_goal,
    metrics=None,
    result_table=None,
    policy_questions="",
    key_suffix="default"
):
    """
    Hàm hiển thị tác nhân AI trong từng bài.
    Dùng chung cho cả 12 bài.
    """
    st.divider()
    st.subheader("🤖 Tác nhân AI phân tích kết quả")

    st.markdown(
        """
        Tác nhân AI có nhiệm vụ đọc các chỉ số đầu ra của mô hình, giải thích thuật toán,
        phân tích ý nghĩa kinh tế và chuyển kết quả định lượng thành hàm ý chính sách.
        """
    )

    mode = st.radio(
        "Chọn chế độ phân tích",
        [
            "Tự động offline — luôn chạy được",
            "Gemini AI — nếu đã cấu hình API key"
        ],
        key=f"ai_mode_{key_suffix}",
        horizontal=True
    )

    if st.button("Tạo phân tích AI", key=f"run_ai_{key_suffix}"):
        with st.spinner("AI đang phân tích kết quả mô hình..."):
            if mode == "Gemini AI — nếu đã cấu hình API key":
                text = gemini_policy_analysis(
                    bai_name=bai_name,
                    model_goal=model_goal,
                    metrics=metrics,
                    result_table=result_table,
                    policy_questions=policy_questions
                )

                if text is None:
                    st.warning(
                        "Chưa gọi được Gemini. App tự chuyển sang chế độ phân tích offline để không bị lỗi."
                    )
                    text = offline_policy_analysis(
                        bai_name=bai_name,
                        model_goal=model_goal,
                        metrics=metrics,
                        policy_questions=policy_questions
                    )
            else:
                text = offline_policy_analysis(
                    bai_name=bai_name,
                    model_goal=model_goal,
                    metrics=metrics,
                    policy_questions=policy_questions
                )

        st.markdown(text)

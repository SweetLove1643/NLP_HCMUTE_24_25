import streamlit as st
import google.generativeai as genai
from datetime import datetime
import time

def start_generative_ai():
    # Cấu hình API key
    api_key = "AIzaSyBfKmUhANIOnClBJz_FwqzUO4s-HeWoYnI"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"Lỗi cấu hình API: {e}")
        st.stop()

    # Thiết lập giao diện với CSS
    st.markdown(
        """
        <style>
        .chat-container {
            background: linear-gradient(135deg, #ffe0b2 0%, #ffcc80 100%); /* Màu nền gradient cam nhạt */
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            max-height: 600px;
            overflow-y: auto;
            margin-bottom: 20px;
            border: 1px solid #ffb74d;
        }
        .chat-message {
            padding: 12px 18px;
            border-radius: 15px;
            margin: 10px 0;
            max-width: 75%;
            word-wrap: break-word;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 16px;
            line-height: 1.5;
        }
        .user-message {
            background-color: #ff8a65; /* Màu cam đậm cho người dùng */
            color: white;
            border-radius: 15px;
            padding: 12px 18px;
            margin-left: auto;
            text-align: right;
            border-bottom-right-radius: 0;
        }
        .bot-message {
            background-color: #ffe0b2; /* Màu nền nhẹ nhàng cho bot */
            color: #333;
            border-radius: 15px;
            margin-right: auto;
            padding: 12px 18px;
            text-align: left;
            border-bottom-left-radius: 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .timestamp {
            font-size: 0.85em;
            color: #9e9e9e;
            margin-top: 2px;
            font-style: italic;
        }
        .stChatInput {
            background-color: #00000000; /* Nền trong suốt */
            border: 2px solid #ffb74d;
            border-radius: 25px;
            padding: 10px 15px;
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background: linear-gradient(90deg, #ffcc80 0%, #ffb74d 100%); /* Gradient cam */
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #ffb74d 0%, #ffcc80 100%);
            transform: scale(1.05);
        }
        .chat-header {
            text-align: center;
            margin-bottom: 20px;
            color: #e65100;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Tiêu đề và mô tả
    st.markdown(
        """
        <div class="chat-header">
            <h1>💬 Chatbot Thông Minh</h1>
            <p style="color: #546e7a;">Hỏi bất cứ điều gì, nhận câu trả lời ngay lập tức!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Khởi tạo session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Container cho lịch sử trò chuyện
    with st.container():
        # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            timestamp = message.get("timestamp", datetime.now().strftime("%H:%M:%S"))
            
            avatar = "🧑‍💻" if role == "user" else "🤖"
            css_class = "user-message" if role == "user" else "bot-message"
            
            with st.chat_message(role, avatar=avatar):
                st.markdown(
                    f'<div class="chat-message {css_class}">{content}</div>'
                    f'<div class="timestamp">{timestamp}</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # Nút xóa lịch sử trò chuyện
    if st.session_state.messages:
        if st.button("🗑️ Xóa lịch sử trò chuyện"):
            st.session_state.messages = []
            st.session_state.chat = model.start_chat(history=[])
            st.success("Đã xóa lịch sử trò chuyện!")
            st.rerun()

    # Nhập câu hỏi
    prompt = st.chat_input("Nhập câu hỏi của bạn...")

    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(
                f'<div class="user-message">{prompt}</div>'
                f'<div class="timestamp">{datetime.now().strftime("%H:%M:%S")}</div>',
                unsafe_allow_html=True
            )

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            try:
                with st.spinner("🤖 Đang xử lý..."):
                    if not hasattr(st.session_state, 'chat'):
                        st.session_state.chat = model.start_chat(history=[])
                    response = st.session_state.chat.send_message(prompt, stream=True)
                    bot_response = ""
                    for chunk in response:
                        if chunk.text:
                            bot_response += chunk.text
                            message_placeholder.markdown(
                                f'<div class="bot-message">{bot_response}</div>',
                                unsafe_allow_html=True
                            )
                            time.sleep(0.05)
            except Exception as e:
                message_placeholder.markdown(f'<div class="bot-message">Lỗi: {e}</div>', unsafe_allow_html=True)
                bot_response = f"Lỗi: {e}"
            finally:
                st.session_state.messages.append({
                    "role": "bot",
                    "content": bot_response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                message_placeholder.markdown(
                    f'<div class="bot-message">{bot_response}</div>'
                    f'<div class="timestamp">{datetime.now().strftime("%H:%M:%S")}</div>',
                    unsafe_allow_html=True
                )
import streamlit as st
import google.generativeai as genai
import os

# Load API key from environment variable
# api_key = os.environ.get("AIzaSyBfKmUhANIOnClBJz_FwqzUO4s-HeWoYnI")
# if not api_key:
#     st.error("AIzaSyBfKmUhANIOnClBJz_FwqzUO4s-HeWoYnI environment variable not set. Please set it before running.")
#     st.stop()
api_key = "AIzaSyBfKmUhANIOnClBJz_FwqzUO4s-HeWoYnI"

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("Chatbot Tiếng Việt với Gemini API")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner("Đang xử lý..."):
                if not hasattr(st.session_state, 'chat'):
                    st.session_state.chat = model.start_chat(history=[])
                response = st.session_state.chat.send_message(prompt, stream=True)
                bot_response = ""
                for chunk in response:
                    if chunk.text:
                        bot_response += chunk.text
                        message_placeholder.markdown(bot_response)
        except Exception as e:
            message_placeholder.markdown(f"Lỗi: {e}")
        finally:
            st.session_state.messages.append({"role": "bot", "content": bot_response})
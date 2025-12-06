import streamlit as st
import time

def stream_text(text: str, delay: float = 0.02):
    """
    生成器：模拟打字机效果
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)

def render_chat_history():
    """
    渲染 Session State 中的历史消息
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def render_assistant_response(text: str):
    """
    使用流式效果渲染 AI 回复，并自动保存到历史
    """
    # 使用 write_stream 实现打字效果
    st.write_stream(stream_text(text))
    
    # 保存到历史
    st.session_state.messages.append({"role": "assistant", "content": text})
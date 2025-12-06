import uuid
import streamlit as st

# --- 导入业务逻辑 ---
from graph.research.workflow import research_app
from langchain_core.messages import HumanMessage
# --- 导入你的新组件 ---
from ui.components.chat_interface import render_chat_history, render_assistant_response
from ui.components.state_visualizer import render_research_status

st.set_page_config(page_title="Research Assistant", page_icon="🧠")
st.title("🧠 Deep Research Assistant")
st.caption("Powered by LangGraph: Retriever -> Router -> Web Search -> Writer")

# ==========================================
# 🌟 【必须补上这一段】 初始化 thread_id
# ==========================================
# 放在这里，确保在后面使用之前它一定存在
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ==========================================
# 初始化历史消息
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 1. 调用组件：渲染历史消息
# ==========================================
render_chat_history()

# ==========================================
# 2. 处理用户交互
# ==========================================
if prompt := st.chat_input("Ask about your papers..."):
    
    # 显示用户提问
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 运行 Agent
    with st.chat_message("assistant"):
        status_box = st.status("🤔 Agent is thinking...", expanded=True)
        final_answer = ""
        
        try:
            initial_state = {"question": prompt,
                             "messages": [HumanMessage(content=prompt)]
            }
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # --- 3. 调用组件：实时可视化状态 ---
            for event in research_app.stream(initial_state, config=config):
                for node_name, state_update in event.items():
                    
                    # ✨ 核心重构点：将复杂的节点状态逻辑委托给组件
                    render_research_status(status_box, node_name, state_update)
                    
                    # 只有 writer 节点会产生最终答案，我们需要捕获它
                    if node_name == "writer":
                        final_answer = state_update.get("answer", "")
            
            status_box.update(label="✅ Ready!", state="complete", expanded=False)
            
            # --- 4. 调用组件：打字机输出 & 自动保存历史 ---
            if final_answer:
                render_assistant_response(final_answer)
            else:
                st.error("No answer generated.")
            
        except Exception as e:
            status_box.update(label="❌ Error", state="error")
            st.error(f"System Error: {e}")
import uuid
import sqlite3
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path

# --- 导入业务逻辑 ---
from graph.research.workflow import research_app
from config.settings import settings
# --- 导入组件 ---
from ui.components.chat_interface import render_chat_history, render_assistant_response
from ui.components.state_visualizer import render_research_status

st.set_page_config(page_title="Research Assistant", page_icon="🧠")
st.title("🧠 Deep Research Assistant")

# ==========================================
# 0. 数据库辅助函数
# ==========================================
DB_PATH = "checkpoints.sqlite"

def get_history_threads():
    """读取所有历史对话 ID"""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';")
        if not cursor.fetchone(): return []
        
        query = """
            SELECT thread_id, MAX(checkpoint_id) as last_active 
            FROM checkpoints 
            GROUP BY thread_id 
            ORDER BY last_active DESC
        """
        cursor.execute(query)
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        return threads
    except Exception:
        return []

def delete_chat_history(thread_id: str):
    """删除指定 thread_id 的所有记录"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        tables = ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                pass 
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Failed to delete chat: {e}")
        return False

def clear_all_history():
    """清空所有历史记录"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        tables = ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Failed to clear history: {e}")
        return False

# ==========================================
# 1. 会话管理 (Sidebar History)
# ==========================================
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = str(uuid.uuid4())

# 用于存储上传的文件路径 (Session级别)
if "uploaded_ref_path" not in st.session_state:
    st.session_state.uploaded_ref_path = None

with st.sidebar:
    # --- 新建对话 ---
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.uploaded_ref_path = None # 重置上传
        st.rerun()
    
    st.divider()

    # --- 临时上传文献对比 ---
    with st.expander("📂 Context Upload", expanded=True):
        st.caption("Upload a paper to compare with knowledge base.")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="ref_uploader")
        
        if uploaded_file:
            # 保存文件
            settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            save_path = settings.UPLOAD_DIR / f"temp_{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_ref_path = str(save_path)
            st.success("File ready for analysis!")
        else:
            st.session_state.uploaded_ref_path = None

    st.divider()

    # --- 设置 ---
    with st.expander("⚙️ Settings", expanded=False):
        allow_web = st.toggle("🌐 Enable Web Search", value=True)
        st.caption("🔍 **Retrieval (Top-K)**")
        top_k_val = st.slider("Docs", 1, 10, 5)
        st.caption("🌡️ **Temperature**")
        temp_val = st.slider("Creativity", 0.0, 1.0, 0.5, 0.1)

    st.divider()

    # --- 历史记录标题区 ---
    col_h1, col_h2 = st.columns([0.2, 0.3])
    with col_h1:
        st.write("**🕒 History**")
    with col_h2:
        if st.button("🗑️ Clear All", help="Delete ALL history", key="clear_all_btn", use_container_width=True):
            if clear_all_history():
                st.session_state.current_thread_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()

    # --- 渲染历史列表 ---
    history_threads = get_history_threads()
    
    if not history_threads:
        st.caption("No history found.")
    
    for t_id in history_threads:
        # 获取标题
        label = f"Chat {t_id[:6]}.."
        try:
            sp_config = {"configurable": {"thread_id": t_id}}
            snapshot = research_app.get_state(sp_config)
            if snapshot.values and "messages" in snapshot.values:
                for m in snapshot.values["messages"]:
                    if isinstance(m, HumanMessage):
                        content = m.content.strip().replace("\n", " ")
                        label = (content[:15] + "..") if len(content) > 15 else content
                        break
        except Exception:
            pass

        # 渲染按钮
        col_chat, col_del = st.columns([0.75, 0.25])
        is_active = (t_id == st.session_state.current_thread_id)
        
        with col_chat:
            if st.button(
                f"{'📂' if is_active else '📄'} {label}", 
                key=f"load_{t_id}", 
                use_container_width=True,
                help=f"ID: {t_id}"
            ):
                st.session_state.current_thread_id = t_id
                st.session_state.messages = []
                st.session_state.uploaded_ref_path = None # 切换对话时清除临时文件
                st.rerun()
        
        with col_del:
            if st.button("✕", key=f"del_{t_id}", help="Delete this chat", use_container_width=True):
                if delete_chat_history(t_id):
                    if t_id == st.session_state.current_thread_id:
                        st.session_state.current_thread_id = str(uuid.uuid4())
                        st.session_state.messages = []
                    st.rerun()

# 获取当前 ID
thread_id = st.session_state.current_thread_id

# ==========================================
# 2. 从数据库恢复消息
# ==========================================
if not st.session_state.get("messages"):
    st.session_state.messages = []
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = research_app.get_state(config)
        if state_snapshot.values and "messages" in state_snapshot.values:
            for msg in state_snapshot.values["messages"]:
                if isinstance(msg, HumanMessage):
                    st.session_state.messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    st.session_state.messages.append({"role": "assistant", "content": msg.content})
    except Exception:
        pass

# ==========================================
# 3. 渲染聊天界面
# ==========================================
render_chat_history()

# ==========================================
# 4. 处理用户输入
# ==========================================
if prompt := st.chat_input("Ask about your papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box = st.status("🤔 Agent is thinking...", expanded=True)
        final_answer = ""
        try:
            # 注入上传文件路径
            initial_state = {
                "question": prompt,
                "messages": [HumanMessage(content=prompt)],
                "allow_web_search": allow_web,
                "top_k": top_k_val,
                "temperature": temp_val,
                "uploaded_file_path": st.session_state.uploaded_ref_path # 👈 传入文件路径
            }
            config = {"configurable": {"thread_id": thread_id}}
            
            for event in research_app.stream(initial_state, config=config):
                for node_name, state_update in event.items():
                    render_research_status(status_box, node_name, state_update)
                    if node_name == "writer":
                        final_answer = state_update.get("answer", "")
            
            status_box.update(label="✅ Ready!", state="complete", expanded=False)
            if final_answer:
                render_assistant_response(final_answer)
            else:
                st.error("No answer generated.")
        except Exception as e:
            status_box.update(label="❌ Error", state="error")
            st.error(f"System Error: {e}")
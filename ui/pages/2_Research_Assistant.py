import sys
import time  # 📝 新增: 用于控制打字速度
from pathlib import Path
import uuid
import streamlit as st

# 路径黑魔法
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from graph.research.workflow import research_app

st.set_page_config(page_title="Research Assistant", page_icon="🧠")
st.title("🧠 Deep Research Assistant")
st.caption("Powered by LangGraph: Retriever -> Router -> Web Search -> Writer")

# 📝 新增: 模拟流式输出的生成器函数
def stream_text(text):
    """
    将完整文本拆分为字符/单词流，模拟打字机效果
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02) # ⚡️ 调整这里的数字可以控制打字速度 (秒)

# 初始化历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("Ask about your papers..."):
    # 1. 显示用户提问
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 运行 Agent
    with st.chat_message("assistant"):
        # 📝 修改: 状态框不要 expanded=True，跑完自动收起，体验更好
        status_box = st.status("🤔 Agent is thinking...", expanded=True)
        final_answer = ""
        
        try:
            # 构造初始 State
            initial_state = {"question": prompt}
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
            # 流式运行 Graph (后端处理)
            for event in research_app.stream(initial_state, config=config):
                for node_name, state_update in event.items():
                    
                    if node_name == "retrieve":
                        docs = state_update.get("context", [])
                        status_box.info(f"🔍 **Retriever**: Found {len(docs)} local documents.")
                        
                    elif node_name == "router":
                        decision = state_update.get("router_decision")
                        if decision == "web_search":
                            status_box.warning("🚦 **Router**: Need external info. Switching to Web Search.")
                        else:
                            status_box.success("🚦 **Router**: Local knowledge is sufficient.")
                            
                    elif node_name == "web_search":
                        queries = state_update.get("search_queries", [])
                        status_box.write(f"🌍 **Web Search**: Searching for `{queries}`...")
                        
                    elif node_name == "writer":
                        status_box.write("✍️ **Writer**: Synthesizing answer...")
                        final_answer = state_update.get("answer", "")
            
            # 📝 修改: 任务完成后，把状态框收起来，变成一个这就绪的小勾
            status_box.update(label="✅ Ready!", state="complete", expanded=False)
            
            # 📝 核心修改: 使用 write_stream 实现打字机效果
            if final_answer:
                # 这里调用我们在上面定义的 stream_text 函数
                response_container = st.write_stream(stream_text(final_answer))
                
                # 存入历史 (注意：要存完整文本，而不是流对象)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
                st.error("No answer generated.")
            
        except Exception as e:
            status_box.update(label="❌ Error", state="error")
            st.error(f"System Error: {e}")
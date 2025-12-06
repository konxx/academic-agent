# app.py
import streamlit as st

# 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="AI Academic Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 AI Academic Research Expert")

st.markdown("""
### Welcome!
这是一个基于 **LangGraph + DeepSeek + Qdrant** 的自主学术研究系统。

请从左侧侧边栏选择功能：

* **📚 Knowledge Base (知识库)**: 
    * 上传 PDF 论文
    * AI 自动提取元数据 (GPT-4o Vision)
    * 自动联网补全信息 (Tavily)
    * 构建向量索引 (Qwen)
    
* **🧠 Research Assistant (研究助手)**: 
    * 基于已入库的知识回答问题
    * (开发中) 自动撰写学术综述
""")

st.info("💡 Tip: 请先在 'Knowledge Base' 页面上传论文。")
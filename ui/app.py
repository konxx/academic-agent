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
### 欢迎使用 Academic Agent (AI 学术研究助手) 🎓

**Academic Agent** 是一个基于 **LangGraph**、**DeepSeek** 和 **Qdrant** 构建的自主学术研究系统。
它结合了 **Visual RAG** 技术来深度理解论文，并提供多种智能体工具来辅助你的研究工作。

请从左侧侧边栏选择功能模块：

*   **📚 Knowledge Base (知识库)**: 
    *   **论文入库**: 上传 PDF 论文，使用视觉模型进行深度解析。
    *   **信息提取**: 自动提取元数据、摘要，并进行联网补全。
    *   **向量索引**: 构建语义索引，为后续任务打好基础。
    
*   **🧠 Research Assistant (研究助手)**: 
    *   **学术问答**: 针对你的知识库进行提问，获取精准答案。
    *   **综述生成**: (Beta) 自动生成文献综述草稿。

*   **🧬 Knowledge Clustering (知识聚类)**:
    *   **多维分析**: 输入多个研究主题，分析论文与各主题的关联度。
    *   **趋势发现**: 快速筛选特定领域的文献，发现研究热点。

*   **⚔️ Idea Debate (观点辩论)**:
    *   **对抗演练**: 设定一个研究目标，让 AI 互相对抗辩论。
    *   **方案优化**: 通过 "Builder vs Critic" 的多轮博弈，打磨出更严谨的研究方案。
""")

st.info("💡 提示: 初次使用请先前往 **'Knowledge Base'** 页面上传并处理论文。")
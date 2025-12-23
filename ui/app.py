# app.py
# Academic Agent 主入口页面

import streamlit as st

# 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="Academic Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    /* 主标题渐变效果 */
    .main-title {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* 副标题样式 */
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* 功能卡片容器 */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* 功能图标 */
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* 功能标题 */
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    /* 功能描述 */
    .feature-desc {
        color: #4a5568;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* 技术栈徽章 */
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    
    /* 分隔线 */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        border-radius: 2px;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === 主标题区域 ===
st.markdown('<h1 class="main-title">🎓 Academic Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于 LangGraph 的自主学术研究助手 | Powered by DeepSeek & Qdrant</p>', unsafe_allow_html=True)

# === 技术栈徽章 ===
tech_cols = st.columns(7)
techs = ["🦜 LangGraph", "🧠 DeepSeek", "🔮 Qwen-VL", "📊 Qdrant", "🔍 Tavily", "🎨 Streamlit", "📄 PyMuPDF"]
for i, tech in enumerate(techs):
    with tech_cols[i]:
        st.markdown(f'<span class="tech-badge">{tech}</span>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# === 功能模块卡片 ===
st.markdown("### 🚀 功能模块")
st.markdown("")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("## 📚 知识库 (Knowledge Base)")
        st.markdown("""
        **构建你的专属学术知识库**
        
        - 🖼️ **Visual RAG**: 视觉大模型直接"阅读"论文页面
        - 📝 **元数据提取**: 自动识别标题、作者、摘要、年份
        - 🌐 **联网补全**: Tavily 搜索补充引用量等外部信息
        - 🔢 **向量索引**: 存入 Qdrant 支持语义检索
        """)
        st.page_link("pages/1_Knowledge_Base.py", label="进入知识库 →", icon="📚")

with col2:
    with st.container(border=True):
        st.markdown("## 🧠 研究助手 (Research Assistant)")
        st.markdown("""
        **基于知识库的智能问答系统**
        
        - 💬 **学术问答**: 精准回答论文细节、方法论问题
        - 🔄 **智能路由**: 自动判断使用本地库还是联网搜索
        - 📖 **综述撰写**: (Beta) 自动规划并撰写综述草稿
        - 💾 **对话记忆**: SQLite 持久化存储对话历史
        """)
        st.page_link("pages/2_Research_Assistant.py", label="开始研究 →", icon="🧠")

col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.markdown("## 🧬 知识聚类 (Knowledge Clustering)")
        st.markdown("""
        **多维度分析论文与主题的关联**
        
        - 📊 **多维分析**: 定义多个研究主题进行关联计算
        - 📈 **可视化**: 直观展示论文在各方向上的得分
        - 🎯 **严格过滤**: 基于关键词匹配的精确筛选
        - 🔥 **趋势发现**: 快速发现研究热点
        """)
        st.page_link("pages/3_Knowledge_Clustering.py", label="分析聚类 →", icon="🧬")

with col4:
    with st.container(border=True):
        st.markdown("## ⚔️ 观点辩论 (Idea Debate)")
        st.markdown("""
        **AI 对抗演练，打磨研究方案**
        
        - 🏟️ **对抗竞技场**: Builder vs Critic 多轮辩论
        - 🛠️ **Builder**: 提出方案、完善细节、应对挑战
        - 🔍 **Critic**: 寻找漏洞、提出质疑、严格审视
        - 💡 **思维链优化**: 产出更加严谨的研究方案
        """)
        st.page_link("pages/4_Idea_Debate.py", label="开始辩论 →", icon="⚔️")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# === 快速开始提示 ===
st.markdown("### 💡 快速开始")

with st.container(border=True):
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown("#### Step 1️⃣")
        st.markdown("前往 **知识库** 页面上传 PDF 论文")
        
    with tip_col2:
        st.markdown("#### Step 2️⃣")
        st.markdown("系统自动解析、提取元数据并入库")
        
    with tip_col3:
        st.markdown("#### Step 3️⃣")
        st.markdown("使用 **研究助手** 进行学术问答")

# === 页脚 ===
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        <p>Made with ❤️ using LangGraph & Streamlit</p>
        <p>© 2025 Academic Agent | <a href="https://github.com/konxx/academic-agent" target="_blank">GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
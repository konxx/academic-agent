# 🎓 Academic Agent (AI 学术研究助手)

**Academic Agent** 是一个基于 **LangGraph**、**DeepSeek** 和 **Qdrant** 构建的自主学术研究助手。它结合了 **Visual RAG** 技术来深度理解 PDF 论文内容，并利用 **Autonomous Agents**（自主智能体）协助你进行深度研究、知识聚类和观点辩论。

## ✨ 核心功能

### 1. 📚 智能知识库 (Knowledge Base)
*   **Visual RAG**: 利用视觉大模型 (如 Qwen-VL) 直接“阅读”论文页面，精准提取图表、公式和布局信息，克服传统 OCR 的缺陷。
*   **元数据提取**: 自动识别论文标题、作者、摘要、发表年份等关键信息。
*   **联网信息补全**: 通过 Tavily 搜索补全论文的引用量、最新评价等外部信息。
*   **向量化索引**: 使用 Qwen/OpenAI 嵌入模型将知识存入 Qdrant 向量数据库，支持高效语义检索。

### 2. 🧠 研究助手 (Research Assistant)
*   **学术问答**: 基于已构建的知识库，准确回答关于论文细节、方法论、实验结果等问题。
*   **综述撰写**: (开发中) 能够根据多篇论文的内容，自动规划并撰写学术综述草稿。

### 3. 🧬 知识聚类 (Knowledge Clustering)
*   **多维语义分析**: 用户可定义多个研究主题（关键词），系统自动计算论文与各主题的关联强度。
*   **可视化展示**: 直观展示每篇论文在不同研究方向上的得分，帮助快速筛选特定领域的文献。
*   **严格模式**: 支持基于关键词匹配的严格过滤。

### 4. ⚔️ 观点辩论 (Idea Debate)
*   **自主对抗竞技场**: 设定一个研究目标，观察两个 AI 智能体（Builder 和 Critic）进行多轮辩论。
*   **Builder (构建者)**: 负责提出方案、完善细节，不断优化以应对挑战。
*   **Critic (批评者)**: 负责寻找漏洞、提出质疑，直到方案无懈可击。
*   **思维链优化**: 通过对抗过程，自动产出更加严谨、深思熟虑的研究方案。

## 🛠️ 技术栈

*   **编排框架**: [LangGraph](https://langchain-ai.github.io/langgraph/) (Stateful Agents)
*   **大语言模型 (LLM)**:
    *   **推理/规划**: DeepSeek Reasoner (R1)
    *   **提取/摘要**: Qwen-VL / DeepSeek Chat
*   **Embedding**: Qwen Text Embedding / OpenAI
*   **向量数据库**: [Qdrant](https://qdrant.tech/)
*   **联网搜索**: [Tavily](https://tavily.com/)
*   **用户界面**: [Streamlit](https://streamlit.io/)
*   **PDF 处理**: PyMuPDF

## 🚀 快速开始

### 前置要求
*   Python 3.10+
*   API Keys:
    *   **DeepSeek** (用于推理/对话)
    *   **DashScope/Aliyun** (用于 Embedding - 可选)
    *   **Qdrant** (用于向量存储)
    *   **Tavily** (用于联网搜索)

### 安装步骤

1.  **克隆项目**
    ```bash
    git clone https://github.com/konxx/academic-agent.git
    cd academic-agent
    ```

2.  **创建虚拟环境**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

### 配置

1.  复制环境变量示例文件：
    ```bash
    cp .env.example .env
    ```

2.  编辑 `.env` 文件，填入你的 API Key：
    ```ini
    # DeepSeek
    AGENT_API_KEY=sk-xxxx
    EXTRACTOR_API_KEY=sk-xxxx

    # Embeddings (DashScope or OpenAI)
    EMBEDDING_API_KEY=sk-xxxx

    # Qdrant
    QDRANT_URL=https://xyz.qdrant.tech
    QDRANT_API_KEY=th-xxxx

    # Tavily
    TAVILY_API_KEY=tvly-xxxx
    ```

### 运行应用

启动 Streamlit 前端界面：

```bash
streamlit run ui/app.py
```

访问浏览器中的地址 (通常是 `http://localhost:8501`) 即可开始使用。

## 📂 项目结构

```text
.
├── config/             # 配置文件 & Prompts
├── core/               # 核心逻辑 (LLM 封装, PDF 加载, Qdrant 客户端)
├── graph/              # LangGraph 工作流定义
│   ├── ingestion/      # 论文入库工作流
│   └── research/       # 研究/问答工作流
├── ui/                 # Streamlit 前端界面
│   ├── app.py          # 主入口
│   ├── pages/          # 功能页面 (知识库, 研究助手, 聚类, 辩论)
│   └── components/     # UI 组件
├── utils/              # 通用工具函数
├── .env.example        # 环境变量示例
├── pyproject.toml      # 项目元数据
└── requirements.txt    # Python 依赖列表
```

## 📄 许可证

本项目采用 MIT 许可证。

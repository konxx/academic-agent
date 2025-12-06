# 🎓 Academic Agent (AI Academic Research Assistant)

**Academic Agent** 是一个基于 **LangGraph**、**DeepSeek** 和 **Qdrant** 构建的自主学术研究助手。它结合了视觉 RAG（Visual RAG）技术，能够深入理解 PDF 论文内容，并利用联网搜索能力补充信息，最终协助用户进行学术问答和综述撰写。

## ✨ 核心功能

### 1. 📚 智能知识库 (Knowledge Base)
*   **PDF 深度解析**: 利用视觉大模型 (Visual LLM) 直接“阅读”论文页面，精准提取图表、公式和布局信息，克服传统 OCR 的缺陷。
*   **自动元数据提取**: 自动识别论文标题、作者、摘要、发表年份等关键信息。
*   **联网信息补全**: 通过 Tavily 搜索补全论文的引用量、最新评价等外部信息。
*   **向量化索引**: 使用 Qwen (通义千问) 嵌入模型将知识存入 Qdrant 向量数据库，支持高效语义检索。

### 2. 🧠 研究助手 (Research Assistant)
*   **学术问答**: 基于已构建的知识库，准确回答关于论文细节、方法论、实验结果等问题。
*   **综述撰写 (开发中)**: 能够根据多篇论文的内容，自动规划并撰写学术综述草稿。

## 🛠️ 技术栈

*   **核心框架**: [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/)
*   **大语言模型 (LLM)**:
    *   **Agent (思考/规划)**: DeepSeek Reasoner (R1)
    *   **Extractor (提取/摘要)**: Qwen (Qwen3-VL-Plus)
*   **Embedding**: Qwen (Text Embedding v4)
*   **向量数据库**: [Qdrant](https://qdrant.tech/) (Cloud)
*   **用户界面**: [Streamlit](https://streamlit.io/)
*   **工具**: PyMuPDF (PDF 处理), Tavily (联网搜索)

## 🚀 快速开始

### 前置要求
*   Python 3.10+
*   API Keys:
    *   **DeepSeek API Key**: 用于推理和文本生成。
    *   **DashScope API Key (阿里云)**: 用于 Qwen Embedding。
    *   **Qdrant Cloud URL & API Key**: 用于向量存储。
    *   **Tavily API Key**: 用于联网搜索。

### 安装步骤

1.  **克隆项目**
    ```bash
    git clone https://github.com/konxx/academic-agent.git
    cd academic-agent
    ```

2.  **创建虚拟环境 (推荐)**
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
    # DeepSeek (用于 Agent 和 提取)
    AGENT_API_KEY=sk-xxxx
    EXTRACTOR_API_KEY=sk-xxxx

    # DashScope (用于 Embedding)
    EMBEDDING_API_KEY=sk-xxxx

    # Qdrant (向量数据库)
    QDRANT_URL=https://xyz.qdrant.tech
    QDRANT_API_KEY=th-xxxx

    # Tavily (搜索)
    TAVILY_API_KEY=tvly-xxxx
    ```
    *(具体配置项请参考 `config/settings.py`)*

### 运行应用

启动 Streamlit 前端界面：

```bash
streamlit run ui/app.py
```

访问浏览器中的地址 (通常是 `http://localhost:8501`) 即可开始使用。

## 📂 项目结构

```text
.
├── config/             # 配置文件 (settings.py, prompts/)
├── core/               # 核心逻辑 (LLM, PDF Loader, Qdrant, Search)
├── graph/              # LangGraph 工作流定义 (Ingestion, Research)
├── ui/                 # Streamlit 前端界面 (app.py, pages/, components/)
├── utils/              # 通用工具函数
├── .env.example        # 环境变量示例
├── pyproject.toml      # 项目元数据
├── requirements.txt    # 依赖列表
└── README.md           # 项目文档
```

## 📄 许可证

本项目采用 MIT 许可证。

# Paper LangGraph

基于 LangGraph 构建的学术研究助手，支持 PDF 文档管理和智能研究功能。

## 功能特点

- 📚 **知识库管理**：上传、解析和存储 PDF 文档，构建向量知识库
- 🔍 **智能检索**：基于向量相似度的高效文档检索
- 🌐 **网络搜索**：自动判断是否需要联网获取最新信息
- 🤖 **研究助手**：结合本地知识库和网络信息，生成高质量研究报告
- 🎨 **直观界面**：基于 Streamlit 的易用 Web 界面
- 🔄 **循环工作流**：基于 LangGraph 的强大循环图能力，支持复杂决策流程

## 技术栈

### 核心框架
- `langchain` / `langchain-core` - 大语言模型应用开发框架
- `langgraph` - 构建循环图工作流

### 模型接口
- `langchain-openai` - 连接 OpenAI 协议模型（含 DeepSeek API）

### 向量数据库
- `qdrant-client` / `langchain-qdrant` - Qdrant 向量数据库集成

### 前端界面
- `streamlit` - Web 应用框架

### 文档处理
- `pypdf` - PDF 解析
- `tiktoken` - Token 计算和文本切分

### 搜索工具
- `tavily-python` - 专为 Agent 设计的搜索 API

## 快速开始

### 环境要求
- Python 3.9+
- 访问 DeepSeek API、DashScope API 和 Qdrant Cloud 的权限

### 安装步骤

1. 克隆仓库
```bash
git clone <repository-url>
cd paper_langgraph
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
```bash
cp .env.example .env
```

编辑 `.env` 文件，填写必要的 API 密钥和配置信息：
```
# Agent 模型配置
AGENT_API_KEY=your-deepseek-api-key

# Extractor 模型配置  
EXTRACTOR_API_KEY=your-deepseek-api-key

# Embedding 模型配置
EMBEDDING_API_KEY=your-dashscope-api-key

# 向量数据库配置
QDRANT_URL=your-qdrant-cloud-url
QDRANT_API_KEY=your-qdrant-api-key

# 搜索工具配置
TAVILY_API_KEY=your-tavily-api-key
```

4. 验证配置
```bash
python check_settings.py
```

5. 运行应用
```bash
streamlit run ui/app.py
```

## 使用指南

### 知识库功能

1. 在左侧导航栏选择 "Knowledge Base"
2. 点击 "Upload PDF" 按钮上传学术论文
3. 等待系统自动解析和构建向量索引
4. 在搜索框中输入关键词，检索相关文档

### 研究助手功能

1. 在左侧导航栏选择 "Research Assistant"
2. 输入研究问题，例如："大语言模型在药物发现中的应用进展"
3. 系统会自动：
   - 从本地知识库检索相关信息
   - 判断是否需要联网获取最新信息
   - 结合所有信息生成完整研究报告

## 项目结构

```
paper_langgraph/
├── config/                   # 配置文件目录
│   ├── prompts/             # 提示词模板（YAML格式）
│   └── settings.py          # 全局配置管理
├── core/                     # 核心功能模块
│   ├── llm.py              # LLM 模型连接
│   ├── pdf_loader.py       # PDF 文档加载
│   ├── qdrant.py           # 向量数据库操作
│   ├── search.py           # 搜索工具集成
│   └── text_splitter.py    # 文本切分
├── graph/                    # LangGraph 工作流
│   ├── ingestion/          # 文档摄入工作流
│   │   ├── nodes.py       # 节点逻辑
│   │   ├── state.py       # 状态定义
│   │   └── workflow.py    # 工作流构建
│   └── research/           # 研究助手工作流
│       ├── nodes.py       # 节点逻辑
│       ├── state.py       # 状态定义
│       └── workflow.py    # 工作流构建
├── ui/                       # Streamlit 界面
│   ├── components/         # 可复用组件
│   ├── pages/             # 页面组件
│   └── app.py             # 应用入口
├── utils/                    # 工具函数
│   └── logger.py          # 日志管理
├── .env.example              # 环境变量示例
├── check_settings.py         # 配置检查脚本
├── pyproject.toml           # Python 项目配置
├── requirements.txt         # 依赖列表
└── readme.md               # 项目说明文档
```

## 技术架构

### 研究助手工作流

1. **检索节点 (retrieve)**：从本地向量数据库检索相关文档
2. **决策节点 (router)**：判断是否需要联网搜索
3. **搜索节点 (web_search)**：如果需要，执行网络搜索获取最新信息
4. **撰写节点 (writer)**：结合本地和网络信息，生成最终研究报告

### 文档摄入工作流

1. **加载节点**：加载和解析 PDF 文档
2. **切分节点**：将文档切分为合适大小的片段
3. **嵌入节点**：生成文本嵌入向量
4. **存储节点**：将向量存储到 Qdrant 数据库

## 配置说明

### 模型配置

- **Agent 模型**：用于复杂逻辑推理、规划和反思，默认使用 DeepSeek Reasoner
- **Extractor 模型**：用于快速 PDF 信息提取和简单摘要，默认使用 DeepSeek Chat
- **Embedding 模型**：用于生成文本嵌入向量，默认使用阿里通义千问 embedding-v4

### 向量数据库配置

- **QDRANT_URL**：Qdrant Cloud 集群 URL
- **QDRANT_API_KEY**：Qdrant Cloud API Key
- **QDRANT_COLLECTION_NAME**：向量集合名称，默认为 "academic_knowledge"

### 搜索工具配置

- **TAVILY_API_KEY**：Tavily 搜索 API Key

## 开发与扩展

### 添加新功能

1. 在 `graph/` 目录下创建新的工作流模块
2. 定义状态类和节点逻辑
3. 构建工作流图
4. 在 UI 中添加相应的交互界面

### 测试

运行测试脚本：
```bash
pytest
```

## 许可证

MIT License

import json
import yaml
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from config.settings import settings
from core.llm import get_agent_llm, get_embeddings
from core.qdrant import qdrant_manager
from core.search import search_tool
from graph.research.state import ResearchState
from utils.logger import logger

# --- 辅助函数: 加载 Research Prompt ---
def load_prompts():
    prompt_path = settings.PROMPTS_DIR / "research.yaml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()

# ==========================================
# Node 1: 意图路由节点 (Router)
# ==========================================
def router_node(state: ResearchState) -> Dict[str, Any]:
    """
    分析用户意图：是只查本地知识库，还是需要联网？
    """
    logger.info("🚦 Processing Node: Router")
    question = state["question"]
    
    llm = get_agent_llm(temperature=0) # 决策需要稳定
    prompt_cfg = PROMPTS["router"]
    
    messages = [
        SystemMessage(content=prompt_cfg["system"]),
        HumanMessage(content=prompt_cfg["user"].format(question=question))
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.replace("```json", "").replace("```", "").strip()
        decision_json = json.loads(content)
        decision = decision_json.get("decision", "web_search") # 默认联网，比较稳妥
        
        logger.info(f"   👉 Decision: {decision}")
        return {"router_decision": decision}
        
    except Exception as e:
        logger.error(f"❌ Router failed: {e}. Fallback to web_search.")
        return {"router_decision": "web_search"}

# ==========================================
# Node 2: 本地检索节点 (Retriever)
# ==========================================
def retrieve_node(state: ResearchState) -> Dict[str, Any]:
    """
    从 Qdrant 检索相关文档
    """
    logger.info("🔍 Processing Node: Local Retriever")
    question = state["question"]
    
    try:
        client = qdrant_manager.client
        embedding_model = get_embeddings()
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=embedding_model
        )
        
        # 检索 Top 5 (根据之前的合成文档，这里检索到的已经是高质量 Summary 了)
        docs = vector_store.similarity_search(question, k=5)
        logger.info(f"   ✅ Retrieved {len(docs)} documents from Qdrant.")
        
        return {"context": docs} # 将结果存入 context
        
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        return {"context": []}

# ==========================================
# Node 3: 联网搜索节点 (Web Search)
# ==========================================
def web_search_node(state: ResearchState) -> Dict[str, Any]:
    """
    生成关键词 -> 联网搜索 -> 封装为 Document
    """
    logger.info("🌍 Processing Node: Web Search")
    question = state["question"]
    existing_context = state.get("context", [])
    
    llm = get_agent_llm()
    
    # 1. 生成搜索词
    prompt_cfg = PROMPTS["generate_search_query"]
    messages = [
        SystemMessage(content=prompt_cfg["system"]),
        HumanMessage(content=prompt_cfg["user"].format(question=question))
    ]
    query_res = llm.invoke(messages).content.strip()
    # 简单处理：假设 LLM 返回的是逗号分隔的关键词
    queries = [q.strip() for q in query_res.split(",")]
    
    logger.info(f"   🔍 Generated Queries: {queries}")
    
    # 2. 执行搜索 (只搜第一个词，或者并发搜)
    # 为了演示简单，我们只用第一个关键词去搜
    search_query = queries[0]
    search_result_str = search_tool.search(search_query)
    
    # 3. 将搜索结果封装成 Document 对象，以便和 Qdrant 结果格式统一
    web_doc = Document(
        page_content=search_result_str,
        metadata={"source": "web_search", "query": search_query}
    )
    
    # 4. 追加到现有 Context
    return {
        "context": existing_context + [web_doc], # 合并
        "search_queries": queries
    }

# ==========================================
# Node 4: 综述撰写节点 (Writer)
# ==========================================
def writer_node(state: ResearchState) -> Dict[str, Any]:
    """
    读取 Context -> 生成最终回答
    """
    logger.info("✍️ Processing Node: Writer")
    question = state["question"]
    context_docs = state.get("context", [])
    messages = state.get("messages", [])
    
    if not context_docs:
        return {"answer": "抱歉，我没有找到任何相关资料，无法回答您的问题。"}
    
    # 1. 格式化上下文
    context_str = ""
    for i, doc in enumerate(context_docs):
        source = doc.metadata.get("title", "Web Search")
        venue = doc.metadata.get("venue", "")
        year = doc.metadata.get("year", "")
        context_str += f"\n--- Reference {i+1} ---\n{doc.page_content}\n"
    
    # 2. 🌟 格式化历史消息 (核心修改)
    # 把最近的对话变成字符串，喂给模型
    history_str = ""
    #recent_history = messages[:-1][-10:] 
    #取剩下历史中的最后 10 条 (即最近 5 轮问答),如果你想保留 10 轮，就改成 [-20:]
    recent_history = messages[:-1] # 不包含当前最新的这条问题
    for msg in recent_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"
    
    # 3. 调用 LLM
    llm = get_agent_llm(temperature=0.7) 
    prompt_cfg = PROMPTS["write_review"]
    
    system_msg = prompt_cfg["system"].format(
        context=context_str,
        chat_history=history_str, # 👈 注入历史
        question=question
    )
    
    msg_payload = [
        SystemMessage(content=system_msg),
        HumanMessage(content=question)
    ]
    
    try:
        response = llm.invoke(msg_payload)
        logger.info("   ✅ Answer generated.")
        
        # 🌟 关键：返回 messages 以便 LangGraph 自动保存
        # 我们返回一个 AIMessage，add_messages 会自动把它追加到历史里
        return {
            "answer": response.content,
            "messages": [AIMessage(content=response.content)] 
        }
    except Exception as e:
        logger.error(f"❌ Writing failed: {e}")
        return {"answer": "Error generating answer."}
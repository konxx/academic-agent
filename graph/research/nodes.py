import json
import yaml
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from config.settings import settings
from core.llm import get_agent_llm, get_embeddings, get_extractor_llm
from core.qdrant import qdrant_manager
from core.search import search_tool
from core.pdf_loader import load_pdf_as_images
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

    if not state.get("allow_web_search", True):
        logger.info("   🚫 Web search disabled by user. Forcing local retrieval.")
        return {"router_decision": "retrieve"}
    
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
# Node 2: 本地检索节点 (Retriever) + 上传处理
# ==========================================
def retrieve_node(state: ResearchState) -> Dict[str, Any]:
    """
    1. 处理上传的 PDF (如果有) -> 转换为 Text
    2. 从 Qdrant 检索相关文档
    """
    logger.info("🔍 Processing Node: Retriever & Processor")
    question = state["question"]
    top_k = state.get("top_k", 5)
    uploaded_path = state.get("uploaded_file_path")
    
    context_docs = []

    # --- A. 处理临时上传的文件 ---
    if uploaded_path:
        try:
            logger.info(f"   📄 Processing Uploaded PDF: {uploaded_path}")
            # 1. 转图片
            images = load_pdf_as_images(uploaded_path, max_pages=100)
            
            # 2. 视觉模型提取摘要
            llm = get_extractor_llm()
            user_content = [
                {"type": "text", "text": "Please analyze these images of a research paper. Provide a comprehensive summary including: Title, Authors, Key Contributions, Methodology, Main Results, and Limitations. This summary will be used to compare with other papers."}
            ]
            for img_b64 in images:
                user_content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
            
            msg = [HumanMessage(content=user_content)]
            response = llm.invoke(msg)
            
            # 3. 封装为 Document
            upload_doc = Document(
                page_content=f"--- [UPLOADED TARGET PAPER] ---\n{response.content}",
                metadata={"title": "Uploaded User Paper", "source": "uploaded_file", "year": "Current"}
            )
            context_docs.append(upload_doc)
            logger.info("   ✅ Uploaded file processed and added to context.")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to process upload: {e}")

    # --- B. Qdrant 检索 ---
    try:
        client = qdrant_manager.client
        embedding_model = get_embeddings()
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=embedding_model
        )
        
        # 检索 Top K
        docs = vector_store.similarity_search(question, k=top_k)
        logger.info(f"   ✅ Retrieved {len(docs)} documents from DB.")
        
        context_docs.extend(docs)
        
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
    
    return {"context": context_docs}

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
# Node 4: 撰写节点 (Writer)
# ==========================================
def writer_node(state: ResearchState) -> Dict[str, Any]:
    """
    读取 Context -> 生成最终回答
    """
    logger.info("✍️ Processing Node: Writer")
    temperature = state.get("temperature", 0.5)
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
        # 标记上传的文件
        if doc.metadata.get("source") == "uploaded_file":
            source = "[User Uploaded PDF]"
            
        context_str += f"\n--- Reference {i+1} ({source}) ---\n{doc.page_content}\n"
    
    # 2. 格式化历史消息
    history_str = ""
    recent_history = messages[:-1] # 不包含当前最新的这条问题
    for msg in recent_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_str += f"{role}: {msg.content}\n"
    
    # 3. 调用 LLM
    llm = get_agent_llm(temperature=temperature) 
    prompt_cfg = PROMPTS["write_review"]
    system_msg = prompt_cfg["system"].format(
        context=context_str,
        chat_history=history_str, 
        question=question
    )
    
    msg_payload = [
        SystemMessage(content=system_msg),
        HumanMessage(content=question)
    ]
    
    try:
        response = llm.invoke(msg_payload)
        logger.info("   ✅ Answer generated.")
        
        # 增加参考文献
        ref_section = "\n\n---\n### 📚 References\n\n"
        for i , doc in enumerate(context_docs):
            meta = doc.metadata
            index = i+1
            
            if meta.get("source") == "uploaded_file":
                ref_section += f"**[{index}]** 📂 **User Uploaded PDF**: *Analyzed Content*\n\n"
            elif meta.get("source") == "web_search":
                query = meta.get("query", "General Search")
                ref_section += f"**[{index}]** 🌐 **Web Search**: *{query}* (Content from Tavily)\n\n"
            else:
                # 论文来源
                title = meta.get("title", "Unknown Title")
                venue = meta.get("venue", "Unknown Venue")
                year = meta.get("year", "N/A")
                authors = meta.get("authors", [])
                
                auth_str = "Unknown Authors"
                if isinstance(authors, list) and len(authors) > 0:
                    auth_str = ", ".join(authors[:2])
                    if len(authors) > 2: auth_str += " et al."
                
                ref_section += f"**[{index}]** 📄 **{title}**\n"
                ref_section += f"   - *{auth_str}* | {venue}, {year}\n\n"
                
        final_content = response.content + ref_section
        
        return {
            "answer": final_content,
            "messages": [AIMessage(content=final_content)] 
        }
    except Exception as e:
        logger.error(f"❌ Writing failed: {e}")
        return {"answer": "Error generating answer."}
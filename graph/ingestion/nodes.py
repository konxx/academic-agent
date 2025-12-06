import json
import yaml
from pathlib import Path
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import settings
from core.llm import get_extractor_llm, get_embeddings
from core.pdf_loader import load_pdf_as_images
from core.qdrant import qdrant_manager
from core.search import search_tool
from graph.ingestion.state import IngestionState
from utils.logger import logger

# --- 辅助函数: 加载 Prompt ---
def load_prompts():
    prompt_path = settings.PROMPTS_DIR / "ingestion.yaml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()

# ==========================================
# Node 1: 元数据提取节点
# ==========================================
def extract_metadata_node(state: IngestionState) -> Dict[str, Any]:
    logger.info(f"👁️ Processing Node: Visual Extraction for {state['pdf_path']}")
    
    # 1. 加载图片 (如果 state 里没有)
    images = state.get("page_images")
    if not images:
        # 调用新的图片加载器
        images = load_pdf_as_images(state["pdf_path"], max_pages=5)
    
    # 2. 准备视觉模型的输入
    llm = get_extractor_llm()
    prompt_cfg = PROMPTS["extract_metadata"]
    
    # --- 构造多模态消息 ---
    # User 消息包含两部分：文本指令 + 图片列表
    user_content = [
        {"type": "text", "text": prompt_cfg["user"]} # 这里不需要再 format {text} 了
    ]
    
    # 把 5 张图片依次加进去
    for img_b64 in images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                # 告诉模型这是 JPEG/PNG 图片
                "url": f"data:image/png;base64,{img_b64}"
            }
        })
        
    messages = [
        SystemMessage(content=prompt_cfg["system"]),
        HumanMessage(content=user_content) # LangChain 会自动处理这个列表
    ]
    
    # 3. 调用模型 (后面的逻辑和之前一样)
    try:
        logger.info("   📤 Sending images to Vision LLM...")
        response = llm.invoke(messages)
        content = response.content.replace("```json", "").replace("```", "").strip()
        metadata = json.loads(content)
        
        logger.info(f"   ✅ Visual Extraction Success: {metadata.get('title')}")

        # 4. 关键：Agent 自我检查 (Reflection)
        missing = []
        if not metadata.get("year"): missing.append("year")
        venue = metadata.get("venue", "").lower()
        if not venue or "arxiv" in venue or "preprint" in venue: missing.append("venue")
        
        if missing:
            logger.warning(f"   ⚠️ Missing/Incomplete fields (triggering search): {missing}")
        
        return {
            "page_images": images,
            "metadata": metadata,
            "missing_fields": missing,
            "retry_count": state.get("retry_count", 0)
        }
        
    except json.JSONDecodeError:
        logger.error("❌ Failed to parse JSON from LLM")
        return {"status": "failed", "error_msg": "JSON Parse Error"}
    except Exception as e:
        logger.error(f"❌ Extraction Error: {e}")
        return {"status": "failed", "error_msg": str(e)}

# ==========================================
# Node 2: 联网修复节点 (Agentic Loop)
# ==========================================
def web_fixer_node(state: IngestionState) -> Dict[str, Any]:
    """
    生成搜索词 -> 联网 -> 修正 Metadata
    """
    current_retries = state.get("retry_count", 0)
    logger.info(f"🌍 Processing Node: Web Search Fixer (Attempt {current_retries + 1})")
    
    metadata = state["metadata"]
    missing = state["missing_fields"]
    
    # 1. 生成搜索关键词 (简单起见，直接用 Python 拼接，也可以用 LLM 生成)
    # PROMPTS["generate_search_query"] 可以在这里用，但为了省 Token，直接拼也不错：
    query = f"{metadata['title']} paper conference year bibtex"
    
    # 2. 执行搜索
    search_results = search_tool.search(query)
    
    # 3. 调用 llm 根据搜索结果修复
    llm = get_extractor_llm()
    prompt_cfg = PROMPTS["fix_metadata"]
    
    messages = [
        SystemMessage(content=prompt_cfg["system"].format(
            current_venue=metadata.get("venue", "Unknown") # 👈 注入当前 venue
        )),
        HumanMessage(content=prompt_cfg["user"].format(
            title=metadata['title'],
            current_venue=metadata.get("venue", "Unknown"),
            missing_fields=missing,
            search_results=search_results
        ))
    ]
    
    # 4. 更新 Metadata
    try:
        response = llm.invoke(messages)
        fix_json = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        
        # 合并新旧数据
        if fix_json:
            metadata.update(fix_json)
            logger.info(f"   ✅ Fixed Metadata: {fix_json}")
        else:
            logger.info("   ❌ Could not find info from web.")
            
    except Exception as e:
        logger.error(f"   Web fix failed: {e}")
    
    # 5. 再次检查是否还缺字段 (决定是否继续 Loop)
    new_missing = []
    if not metadata.get("year"): new_missing.append("year")
    if not metadata.get("venue"): new_missing.append("venue")
    
    return {
        "metadata": metadata,
        "missing_fields": new_missing,
        "retry_count": current_retries + 1
    }

# ==========================================
# Node 3: 向量入库节点
# ==========================================
def ingest_to_qdrant_node(state: IngestionState) -> Dict[str, Any]:
    logger.info("💾 Processing Node: Ingest High-Quality Metadata to Qdrant")
    
    metadata = state["metadata"]
    
    # 1. 构造合成文档 (保持不变)
    content_parts = [
        f"Title: {metadata.get('title', 'Unknown')}",
        f"Year: {metadata.get('year', 'Unknown')}",
        f"Venue: {metadata.get('venue', 'Unknown')}", # 这里的 venue 应该是修正后的
        f"Authors: {', '.join(metadata.get('authors', []))}",
        "--- Abstract ---",
        metadata.get('abstract', 'No abstract extracted.'),
        "--- Core Introduction & Background ---",
        metadata.get('introduction_summary', 'No summary provided.')
    ]
    clean_text = "\n\n".join(content_parts)
    
    # 2. 🌟 核心修改：取消切片，直接封装成一个 Document
    # 之前的 RecursiveCharacterTextSplitter 把这个 clean_text 切成了几段
    # 导致数据库里出现了多条拥有相同 Metadata 的记录
    final_doc = Document(
        page_content=clean_text,
        metadata={
            **metadata,
            "source": str(state["pdf_path"]),
            "content_type": "ai_generated_summary"
        }
    )
    
    # 3. 写入 Qdrant
    try:
        qdrant_manager.ensure_collection_exists()
        client = qdrant_manager.client
        embedding_model = get_embeddings()
        
        from langchain_qdrant import QdrantVectorStore
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=embedding_model
        )
        
        # 直接添加这一个文档
        vector_store.add_documents([final_doc]) 
        logger.info(f"   ✅ Successfully ingested 1 single document (Length: {len(clean_text)}).")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"❌ Database Error: {e}")
        return {"status": "failed", "error_msg": str(e)}
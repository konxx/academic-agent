import streamlit as st
from typing import Dict, Any, List

def render_ingestion_status(status_container, node_name: str, state_update: Dict[str, Any], preview_data: Dict):
    """
    可视化入库流程的状态更新
    
    :param status_container: st.status 对象
    :param node_name: 当前完成的节点名
    :param state_update: 节点返回的状态增量
    :param preview_data: 用于收集预览数据的字典 (引用传递)
    """
    if node_name == "extract_metadata":
        meta = state_update.get("metadata", {})
        title = meta.get("title", "Unknown")
        missing = state_update.get("missing_fields", [])
        
        # 收集预览数据
        if state_update.get("page_images"):
            preview_data["images"] = state_update["page_images"]
        if meta:
            preview_data["metadata"] = meta
        
        status_container.write(f"**👁️ Visual Extraction**: Reading PDF...")
        if missing:
            status_container.warning(f"⚠️ Missing fields: `{missing}`. Searching Web...")
        else:
            status_container.info(f"✅ Extracted: **{title}**")
            
    elif node_name == "web_fixer":
        status_container.write("🌍 **Web Fixer**: Searching Internet...")
        # 更新可能的元数据修复
        if state_update.get("metadata"):
            preview_data["metadata"] = state_update["metadata"]
            
        if not state_update.get("missing_fields"):
            status_container.success("✅ Metadata fixed!")
        else:
            status_container.warning(f"⚠️ Retrying search... (Attempt {state_update.get('retry_count')})")
            
    elif node_name == "ingest_to_qdrant":
        status_container.write("💾 **Database**: Indexing...")


def render_research_status(status_container, node_name: str, state_update: Dict[str, Any]):
    """
    可视化研究助手流程的状态更新
    """
    if node_name == "retrieve":
        docs = state_update.get("context", [])
        status_container.info(f"🔍 **Retriever**: Found {len(docs)} local documents.")
        
    elif node_name == "router":
        decision = state_update.get("router_decision")
        if decision == "web_search":
            status_container.warning("🚦 **Router**: Need external info. Switching to Web Search.")
        else:
            status_container.success("🚦 **Router**: Local knowledge is sufficient.")
            
    elif node_name == "web_search":
        queries = state_update.get("search_queries", [])
        status_container.write(f"🌍 **Web Search**: Searching for `{queries}`...")
        
    elif node_name == "writer":
        status_container.write("✍️ **Writer**: Synthesizing answer...")
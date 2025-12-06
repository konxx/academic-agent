from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.ingestion.state import IngestionState
from graph.ingestion.nodes import (
    extract_metadata_node,
    web_fixer_node,
    ingest_to_qdrant_node
)
from utils.logger import logger

# ==========================================
# 1. 定义条件边逻辑 (Router Logic)
# ==========================================
def decide_next_step(state: IngestionState) -> str:
    """
    判断下一步去哪里：
    - 如果字段齐全 -> 入库
    - 如果缺失但重试次数超标 -> 放弃治疗，直接入库
    - 如果缺失且还有重试机会 -> 联网修复
    """
    missing = state.get("missing_fields", [])
    retry_count = state.get("retry_count", 0)
    MAX_RETRIES = 3  # 最大重试次数

    if not missing:
        logger.info("✅ Data is complete. Moving to Ingestion.")
        return "ingest_to_qdrant"
    
    if retry_count >= MAX_RETRIES:
        logger.warning(f"🛑 Max retries ({MAX_RETRIES}) reached. Proceeding with incomplete metadata.")
        return "ingest_to_qdrant"

    logger.info(f"🔍 Missing fields detected: {missing}. Route -> Web Fixer.")
    return "web_fixer"

# ==========================================
# 2. 构建图结构 (Graph Construction)
# ==========================================
def build_ingestion_graph():
    # 初始化图，指定 State 类型
    workflow = StateGraph(IngestionState)

    # A. 添加节点
    workflow.add_node("extract_metadata", extract_metadata_node)
    workflow.add_node("web_fixer", web_fixer_node)
    workflow.add_node("ingest_to_qdrant", ingest_to_qdrant_node)

    # B. 设置起点
    workflow.set_entry_point("extract_metadata")

    # C. 添加条件边 (Conditional Edges)
    # 从 extract_metadata 出来后，走 decide_next_step 函数判断
    workflow.add_conditional_edges(
        "extract_metadata",
        decide_next_step,
        {
            "web_fixer": "web_fixer",
            "ingest_to_qdrant": "ingest_to_qdrant"
        }
    )

    # D. 添加循环边 (Cyclic Edge)
    # web_fixer 跑完后，不要直接去入库，而是再判断一次（或者回到提取？）
    # 这里我们简化逻辑：web_fixer 跑完后再次检查条件
    workflow.add_conditional_edges(
        "web_fixer",
        decide_next_step,
        {
            "web_fixer": "web_fixer",       # 如果还没修好，且没超次，继续修
            "ingest_to_qdrant": "ingest_to_qdrant" # 修好了，或者超次了，去入库
        }
    )

    # E. 终点
    workflow.add_edge("ingest_to_qdrant", END)

    # F. 编译 (Compile)
    # checkpointer=MemorySaver() 允许我们在步骤之间保存状态 (用于 Debug 或人机交互)
    return workflow.compile(checkpointer=MemorySaver())

# 实例化 App 对象，供 UI 调用
ingestion_app = build_ingestion_graph()
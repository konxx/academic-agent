from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.research.state import ResearchState
from graph.research.nodes import (
    router_node,
    retrieve_node,
    web_search_node,
    writer_node
)
from utils.logger import logger

# ==========================================
# 1. 定义条件边逻辑
# ==========================================
def decide_to_web_search(state: ResearchState) -> str:
    """
    根据 Router 的决策决定下一步
    """
    decision = state.get("router_decision", "retrieve")
    
    if decision == "web_search":
        logger.info("👉 Routing to: Web Search")
        return "web_search"
    else:
        logger.info("👉 Routing to: Writer (Skipping Web)")
        return "writer"

# ==========================================
# 2. 构建 Research Graph
# ==========================================
def build_research_graph():
    workflow = StateGraph(ResearchState)

    # A. 添加节点
    workflow.add_node("retrieve", retrieve_node)     # 查本地
    workflow.add_node("router", router_node)         # 做决策
    workflow.add_node("web_search", web_search_node) # 查网络
    workflow.add_node("writer", writer_node)         # 写答案

    # B. 设置起点
    # 策略：无论如何先查本地库，哪怕Router最后决定联网，本地资料也是很好的补充
    workflow.set_entry_point("retrieve")

    # C. 连接节点
    # 1. Retrieve -> Router (查完本地，让大脑判断一下够不够，或者问题是否需要即时信息)
    workflow.add_edge("retrieve", "router")

    # 2. Router -> Conditional (去联网 OR 直接写)
    workflow.add_conditional_edges(
        "router",
        decide_to_web_search,
        {
            "web_search": "web_search",
            "writer": "writer"
        }
    )

    # 3. Web Search -> Writer (搜完网，去写)
    workflow.add_edge("web_search", "writer")

    # 4. Writer -> End (写完结束)
    workflow.add_edge("writer", END)

    # D. 编译
    return workflow.compile(checkpointer=MemorySaver())

# 实例化 App
research_app = build_research_graph()
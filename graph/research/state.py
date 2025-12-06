from typing import TypedDict, List
from langchain_core.documents import Document

class ResearchState(TypedDict):
    """
    Research 工作流的状态
    """
    # 1. 输入
    question: str
    
    # 2. 决策与中间变量
    router_decision: str       # "retrieve" (只查库) 或 "web_search" (查库+联网)
    search_queries: List[str]  # 生成的联网搜索关键词
    
    # 3. 核心上下文 (混合了 Qdrant 的文档和 Web 的搜索结果)
    context: List[Document]
    
    # 4. 输出
    answer: str
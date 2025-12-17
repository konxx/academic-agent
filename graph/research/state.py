from typing import TypedDict, List, Annotated, Optional
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage # 👈 引入 Message
from langgraph.graph.message import add_messages # 👈 引入 reducer

class ResearchState(TypedDict):
    # --- 核心修改：增加 messages 字段 ---
    # add_messages 会自动把新消息追加到历史列表里，而不是覆盖
    messages: Annotated[List[AnyMessage], add_messages] 
    
    question: str
    router_decision: str
    search_queries: List[str]
    context: List[Document]
    answer: str
    allow_web_search: bool
    top_k: int
    temperature: float
    uploaded_file_path: Optional[str]
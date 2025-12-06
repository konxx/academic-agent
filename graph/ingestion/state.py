from typing import TypedDict, List, Optional, Dict, Any

class IngestionState(TypedDict):
    pdf_path: str
    # raw_text: str  <-- 删除这个
    page_images: List[str] # <-- 改成存储图片 Base64 列表
    file_name: str
    metadata: Dict[str, Any]
    missing_fields: List[str]
    retry_count: int
    status: str
    error_msg: Optional[str]
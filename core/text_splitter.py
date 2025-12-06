# 文本切分器,若上传全部论文,则在nodes中调用
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    获取统一配置的文本切分器
    
    :param chunk_size: 每个切片的最大字符数
    :param chunk_overlap: 切片之间的重叠字符数 (保证语义连续性)
    :return: Splitter 实例
    """
    return RecursiveCharacterTextSplitter(
        # 常见的分隔符，优先级从上到下
        separators=["\n\n", "\n", "。", ".", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

def split_text(text: str, chunk_size: int = 1000) -> list[str]:
    """
    辅助函数：直接将字符串切分为列表 (用于调试或非 Document 场景)
    """
    splitter = get_text_splitter(chunk_size=chunk_size)
    return splitter.split_text(text)
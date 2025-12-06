import sys
from pathlib import Path

# 防止路径引用错误 (同 qdrant.py 的逻辑)
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config.settings import settings
from utils.logger import logger

def get_agent_llm(temperature: float = 0.5) -> ChatOpenAI:
    """
    获取 Agent 思考模型 (如 DeepSeek Reasoner / R1)
    用于: 任务规划、复杂逻辑判断、综述撰写
    """
    return ChatOpenAI(
        model=settings.AGENT_MODEL_NAME,
        api_key=settings.AGENT_API_KEY,
        base_url=settings.AGENT_BASE_URL,
        temperature=temperature,
        max_retries=2,
        # DeepSeek Reasoner 可能不支持 system prompt 或者有特殊行为，
        # 但通过 OpenAI 接口调用通常兼容
    )

def get_extractor_llm() -> ChatOpenAI:
    """
    获取提取模型 (如 DeepSeek Chat / V3)
    用于: PDF 解析、元数据提取、简单摘要
    特点: 温度为 0，追求稳定性和格式准确性
    """
    return ChatOpenAI(
        model=settings.EXTRACTOR_MODEL_NAME,
        api_key=settings.EXTRACTOR_API_KEY,
        base_url=settings.EXTRACTOR_BASE_URL,
        temperature=0,  # 严格模式
        max_retries=3,
    )

def get_embeddings() -> OpenAIEmbeddings:
    """
    获取向量模型 (Qwen / DashScope)
    """
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        openai_api_key=settings.EMBEDDING_API_KEY,
        openai_api_base=settings.EMBEDDING_BASE_URL,
        # ⚠️ 关键设置: 阿里模型 Tokenizer 可能与 OpenAI 不同，禁用客户端检查避免报错
        check_embedding_ctx_length=False, 
        dimensions=2048,
        chunk_size=10
    )

if __name__ == "__main__":
    # --- 测试脚本 ---
    # 运行: python -m core.llm
    
    print("-" * 50)
    print("🤖 Testing Model Connectivity...")

    # 1. 测试 Agent 模型
    try:
        print(f"1. Testing Agent ({settings.AGENT_MODEL_NAME})...")
        agent = get_agent_llm()
        res = agent.invoke("你好，你是谁？请简短回答。")
        print(f"   ✅ Agent Response: {res.content}")
    except Exception as e:
        print(f"   ❌ Agent Failed: {e}")

    # 2. 测试 Extractor 模型
    try:
        print(f"2. Testing Extractor ({settings.EXTRACTOR_MODEL_NAME})...")
        extractor = get_extractor_llm()
        res = extractor.invoke("提取这句话里的数字：'我有3个苹果'，只输出数字。")
        print(f"   ✅ Extractor Response: {res.content}")
    except Exception as e:
        print(f"   ❌ Extractor Failed: {e}")

    # 3. ⚠️ 重要：测试 Embedding 维度
    try:
        print(f"3. Testing Embedding ({settings.EMBEDDING_MODEL_NAME})...")
        emb_model = get_embeddings()
        vector = emb_model.embed_query("测试向量维度")
        dim = len(vector)
        print(f"   ✅ Embedding Success! Vector Dimension: 【 {dim} 】")
        
        # 提示用户
        print("-" * 50)
        if dim == 2048:
            print("💡 维度是 2048，与 Qdrant 默认设置匹配。无需修改。")
        elif dim == 1024:
            print("⚠️  注意！维度是 1024。请修改 core/qdrant.py 中的 vector_size=1024")
        else:
            print(f"⚠️  注意！维度是 {dim}。请确保 core/qdrant.py 与此一致。")
            
    except Exception as e:
        print(f"   ❌ Embedding Failed: {e}")
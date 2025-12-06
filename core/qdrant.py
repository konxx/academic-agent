import sys
from typing import Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# 将项目根目录加入路径，确保能导入 config
sys.path.append("..") 
from config.settings import settings
from utils.logger import logger  # 假设你之后会创建这个，现在先用 print 代替也可以

class QdrantManager:
    """
    Qdrant 数据库管理器
    负责连接管理、集合创建和状态检查
    """
    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    @property
    def client(self) -> QdrantClient:
        """
        获取 QdrantClient 单例 (Lazy Loading)
        """
        if self._client is None:
            try:
                # 区分本地模式和云端模式
                if settings.QDRANT_API_KEY:
                    logger.info(f"🔌 Connecting to Qdrant Cloud: {settings.QDRANT_URL}...")
                    self._client = QdrantClient(
                        url=settings.QDRANT_URL,
                        api_key=settings.QDRANT_API_KEY,
                    )
                else:
                    logger.info(f"🔌 Connecting to Local Qdrant: {settings.QDRANT_URL}...")
                    self._client = QdrantClient(url=settings.QDRANT_URL)
                
                # 测试连接
                self._client.get_collections()
                logger.info("✅ Qdrant Connection Successful!")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Qdrant: {e}")
                raise e
        return self._client

    def ensure_collection_exists(self, vector_size: int = 2048):
        """
        检查集合是否存在，不存在则创建
        :param vector_size: 向量维度。
                            Qwen-v4 = 2048
                            请务必确认你的 Embedding 模型输出维度！
        """
        client = self.client
        exists = client.collection_exists(self.collection_name)

        if not exists:
            logger.warning(f"⚠️ Collection '{self.collection_name}' not found. Creating...")
            try:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"✅ Collection '{self.collection_name}' created (size={vector_size})")
            except Exception as e:
                logger.error(f"❌ Failed to create collection: {e}")
                raise
        else:
            logger.info(f"✅ Collection '{self.collection_name}' exists.")

    def delete_collection(self):
        """危险操作：删除集合"""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"🗑️ Collection '{self.collection_name}' deleted.")

    def get_info(self):
        """获取集合统计信息"""
        return self.client.get_collection(self.collection_name)

# 实例化单例
qdrant_manager = QdrantManager()

if __name__ == "__main__":
    # --- 简单的测试脚本 ---
    # 在命令行运行: python -m core.qdrant
    
    # 为了测试，我们临时定义一个 logger
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("-" * 50)
    print("🚀 Testing Qdrant Connection...")
    
    try:
        # 1. 强制初始化连接
        client = qdrant_manager.client
        
        # 2. 检查或创建集合
        qdrant_manager.ensure_collection_exists(vector_size=2048)
        
        # 3. 获取信息
        info = qdrant_manager.get_info()
        print(f"📊 Collection Info: Status={info.status}, Vectors Count={info.vectors_count}")
        
        print("✅ Qdrant Module is Ready!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("💡 Hint: Check your QDRANT_URL and API_KEY in .env")
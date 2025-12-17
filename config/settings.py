import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field

# 定位项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """
    全局配置类
    自动读取 .env 文件中的值并进行类型验证
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略 .env 中多余的字段
        case_sensitive=True # 区分大小写 (推荐)
    )

    # ==========================
    # 1. 基础路径配置
    # ==========================
    @computed_field
    def DATA_DIR(self) -> Path:
        path = PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    def PROMPTS_DIR(self) -> Path:
        return PROJECT_ROOT / "config" / "prompts"

    @computed_field
    def UPLOAD_DIR(self) -> Path:
        path = self.DATA_DIR / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ==========================
    # 2. Agent 模型 (DeepSeek Reasoner / R1)
    # ==========================
    # 用于复杂的逻辑推理、规划和反思
    AGENT_BASE_URL: str = Field(default="https://api.deepseek.com/v1")
    AGENT_API_KEY: str = Field(..., description="DeepSeek API Key") # ... 代表必填
    AGENT_MODEL_NAME: str = Field(default="deepseek-reasoner")

    # ==========================
    # 3. Critic 模型 (Qwen3-Max)
    # ==========================
    # 用于代码质量评估、功能分析、安全考虑
    CRITIC_BASE_URL: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    CRITIC_API_KEY: str = Field(..., description="DashScope API Key")
    CRITIC_MODEL_NAME: str = Field(default="qwen3-max")
    
    # ==========================
    # 3. Extractor 模型 (Qwen)
    # ==========================
    # 用于快速的 PDF 信息提取、简单摘要
    EXTRACTOR_BASE_URL: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    EXTRACTOR_API_KEY: str = Field(..., description="DashScope API Key")
    EXTRACTOR_MODEL_NAME: str = Field(default="qwen3-vl-plus")

    # ==========================
    # 4. Embedding 模型 (Qwen)
    # ==========================
    # 阿里通义千问 embedding-v4
    EMBEDDING_BASE_URL: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    EMBEDDING_API_KEY: str = Field(..., description="DashScope API Key")
    EMBEDDING_MODEL_NAME: str = Field(default="text-embedding-v4")

    # ==========================
    # 5. 向量数据库 (Qdrant Cloud)
    # ==========================
    # 注意: Cloud 版本必须要有 API Key
    QDRANT_URL: str = Field(..., description="Qdrant Cloud Cluster URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant Cloud API Key") 
    QDRANT_COLLECTION_NAME: str = Field(default="academic_knowledge")

    # ==========================
    # 6. 搜索工具 (Tavily)
    # ==========================
    TAVILY_API_KEY: str = Field(..., description="Tavily API Key")

# 实例化并导出
settings = Settings()

if __name__ == "__main__":
    # 运行此文件测试配置加载是否成功
    print(f"✅ Loaded Settings from {PROJECT_ROOT}")
    print(f"   Agent Model: {settings.AGENT_MODEL_NAME}")
    print(f"   Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"   Qdrant URL: {settings.QDRANT_URL}")
    if not settings.QDRANT_API_KEY:
        print("⚠️  Warning: QDRANT_API_KEY is missing! Cloud connection will fail.")
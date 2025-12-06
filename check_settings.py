# check_settings.py
from config.settings import settings

print("--- 当前生效的配置 ---")
print(f"URL:   {settings.EXTRACTOR_BASE_URL}")
print(f"Key:   {settings.EXTRACTOR_API_KEY[:5]}******") # 隐去部分密钥
print(f"Model: {settings.EXTRACTOR_MODEL_NAME}")
from tavily import TavilyClient
from config.settings import settings
from utils.logger import logger

class SearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

    def search(self, query: str, max_results: int = 3) -> str:
        """
        执行联网搜索并返回拼接好的字符串结果
        """
        logger.info(f"🔍 Searching Web: {query}")
        try:
            # search_depth="advanced" 会深入抓取内容，适合找年份
            response = self.client.search(
                query=query, 
                search_depth="advanced", 
                max_results=max_results
            )
            
            results = []
            for res in response.get('results', []):
                snippet = res.get('content', '')
                url = res.get('url', '')
                results.append(f"来源: {url}\n内容: {snippet}")
            
            return "\n---\n".join(results)
        
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return ""

# 单例
search_tool = SearchTool()
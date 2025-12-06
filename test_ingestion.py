import uuid
from graph.ingestion.workflow import ingestion_app

# 模拟一个 PDF 路径 (请确保该路径下真实存在一个 PDF 文件)
# 你可以随便找个论文放进 data/uploads/ 目录
TEST_PDF_PATH = "data/uploads/2511.13720v1.pdf" 

if __name__ == "__main__":
    print("🚀 Starting Ingestion Workflow Test...")
    
    # 初始配置
    # thread_id 是 LangGraph 用来区分不同对话/任务线程的
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 初始输入 State
    initial_state = {
        "pdf_path": TEST_PDF_PATH,
        "retry_count": 0
    }
    
    # 运行图
    for event in ingestion_app.stream(initial_state, config=config):
        for key, value in event.items():
            print(f"\n👉 Node Finished: {key}")
            if key == "extract_metadata":
                meta = value.get("metadata", {})
                print(f"   Title: {meta.get('title')}")
                print(f"   Missing: {value.get('missing_fields')}")
            elif key == "web_fixer":
                print("   🌐 Web Fixer executed.")
            elif key == "ingest_to_qdrant":
                print("   💾 Ingested to DB.")
                
    print("\n✅ Workflow Finished!")
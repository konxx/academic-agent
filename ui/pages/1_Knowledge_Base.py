# ui/pages/1_Knowledge_Base.py
import sys
from pathlib import Path

# 获取项目根目录 (ui/app.py 的上一级的上一级)
# __file__ = ui/app.py -> parent = ui/ -> parent = academic-agent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import streamlit as st
import base64
import os
import shutil
import uuid
from pathlib import Path

# 引入我们的后端逻辑
from config.settings import settings
from graph.ingestion.workflow import ingestion_app
from core.qdrant import qdrant_manager

st.set_page_config(page_title="Knowledge Base", page_icon="📚")

# --- 侧边栏：显示数据库状态 ---
with st.sidebar:
    st.header("📊 Database Status")
    if st.button("Refresh Stats"):
        try:
            info = qdrant_manager.client.get_collection(settings.QDRANT_COLLECTION_NAME)
            st.metric("Total Papers", info.points_count)
            st.success("Connected to Qdrant Cloud")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

st.title("📚 Knowledge Base Ingestion")
st.caption("Upload raw PDFs -> AI Agent Extraction -> Vector Database")

# --- 1. 文件上传区域 ---
uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    # 保存文件到 data/uploads
    save_dir = settings.UPLOAD_DIR
    file_path = save_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File saved: `{uploaded_file.name}`")
    
    # --- 2. 启动 Agent 按钮 ---
    if st.button("🚀 Start AI Ingestion Agent", type="primary"):
        
        # 容器：用于显示动态日志
        status_container = st.status("🤖 Agent is working...", expanded=True)
        
        # --- 1. 定义临时变量用于存储预览数据 ---
        preview_data = {
            "images": None,
            "metadata": {}
        }
        try:
            initial_state = {
                "pdf_path": str(file_path),
                "retry_count": 0
            }
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
            for event in ingestion_app.stream(initial_state, config=config):
                
                for node_name, state_update in event.items():
                    
                    if node_name == "extract_metadata":
                        meta = state_update.get("metadata", {})
                        title = meta.get("title", "Unknown")
                        missing = state_update.get("missing_fields", [])
                        
                        # --- 2. 捕获数据 ---
                        # 只要有图片或元数据，就存下来
                        if state_update.get("page_images"):
                            preview_data["images"] = state_update["page_images"]
                        if meta:
                            preview_data["metadata"] = meta
                        
                        status_container.write(f"**👁️ Visual Extraction**: Reading PDF...")
                        if missing:
                            status_container.warning(f"⚠️ Missing fields: `{missing}`. Searching Web...")
                        else:
                            status_container.info(f"✅ Extracted: **{title}**")
                            
                    elif node_name == "web_fixer":
                        status_container.write("🌍 **Web Fixer**: Searching Internet...")
                        # 再次更新元数据 (因为可能被修正了)
                        if state_update.get("metadata"):
                            preview_data["metadata"] = state_update["metadata"]
                            
                        if not state_update.get("missing_fields"):
                            status_container.success("✅ Metadata fixed!")
                        else:
                            status_container.warning("⚠️ Retrying search...")
                            
                    elif node_name == "ingest_to_qdrant":
                        status_container.write("💾 **Database**: Indexing...")
            
            status_container.update(label="✅ Processing Complete!", state="complete", expanded=False)
            st.balloons()
            
            # --- 3. 安全地显示结果 ---
            st.divider()
            st.subheader("🎉 Ingestion Result")
            
            final_meta = preview_data["metadata"]
            final_images = preview_data["images"]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                # 只有当真的有图片时才显示，防止报错
                if final_images and len(final_images) > 0:
                    # 对图片进行 Base64 编码
                    image_data = base64.b64decode(final_images[0])
                    st.image(image_data, caption="Cover Page Preview", use_container_width=True)
                else:
                    st.warning("No preview image available")
                    
            with col2:
                st.markdown(f"**Title:** {final_meta.get('title', 'Unknown')}")
                st.markdown(f"**Venue:** {final_meta.get('venue', 'Unknown')}")
                st.markdown(f"**Year:** {final_meta.get('year', 'Unknown')}")
                with st.expander("Read Abstract"):
                    st.write(final_meta.get("abstract", "No abstract"))
                with st.expander("Read Introduction"):
                    st.write(final_meta.get("introduction", "No introduction"))
                with st.expander("Read Introduction Summary (Chinese)"):
                    st.write(final_meta.get("introduction_summary", "No summary"))
                    
        except Exception as e:
            status_container.update(label="❌ Error Occurred", state="error")
            st.error(f"An error occurred: {e}")
            # 打印详细堆栈以便调试 (可选)
            import traceback
            st.code(traceback.format_exc())
import base64
import uuid
import streamlit as st
import traceback

# --- 导入业务逻辑 ---
from graph.ingestion.workflow import ingestion_app

# --- 导入你的新组件 ---
# 注意：render_pdf_uploader 需要修改为返回列表 List[Path]
from ui.components.pdf_uploader import render_pdf_uploader
from ui.components.state_visualizer import render_ingestion_status

st.set_page_config(page_title="Knowledge Base", page_icon="📚")
st.title("📚 Knowledge Base Ingestion")
st.caption("Upload raw PDFs -> AI Agent Extraction -> Vector Database")

# ==========================================
# 1. 调用组件：文件上传 (支持多文件)
# ==========================================
# 这里接收的是一个列表，例如 [Path('.../a.pdf'), Path('.../b.pdf')]
file_paths = render_pdf_uploader()

if file_paths:
    st.info(f"📂 Ready to process {len(file_paths)} documents.")

    # ==========================================
    # 2. 启动 Agent 流程
    # ==========================================
    if st.button("🚀 Start AI Ingestion Agent", type="primary"):
        
        # 总进度条
        progress_bar = st.progress(0)
        total_files = len(file_paths)
        
        # 3. 循环遍历每个文件
        for i, file_path in enumerate(file_paths):
            # 为每个文件创建一个独立的展示区域
            st.divider()
            st.subheader(f"📄 Processing ({i+1}/{total_files}): `{file_path.name}`")
            
            status_container = st.status(f"🤖 Agent is analyzing {file_path.name}...", expanded=True)
            
            # 定义临时变量收集当前文件的结果数据
            preview_data = {
                "images": None,
                "metadata": {}
            }
            
            try:
                # 构造初始状态
                initial_state = {
                    "pdf_path": str(file_path),
                    "retry_count": 0
                }
                # 为每个文件生成独立的 thread_id，避免状态混淆
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                
                # --- Stream 运行图 ---
                for event in ingestion_app.stream(initial_state, config=config):
                    for node_name, state_update in event.items():
                        # 调用可视化组件更新状态
                        render_ingestion_status(
                            status_container, 
                            node_name, 
                            state_update, 
                            preview_data
                        )
                
                status_container.update(label=f"✅ {file_path.name} - Complete!", state="complete", expanded=False)
                
                # ==========================================
                # 4. 展示该文件的最终结果
                # ==========================================
                with st.expander(f"🎉 View Result: {file_path.name}", expanded=True):
                    final_meta = preview_data.get("metadata", {})
                    final_images = preview_data.get("images", [])
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if final_images and len(final_images) > 0:
                            try:
                                # Base64 解码显示封面
                                image_data = base64.b64decode(final_images[0])
                                st.image(image_data, caption="Cover Page", use_container_width=True)
                            except Exception:
                                st.warning("Image render failed")
                        else:
                            st.warning("No preview image available")
                            
                    with col2:
                        st.markdown(f"**Title:** {final_meta.get('title', 'Unknown')}")
                        st.markdown(f"**Venue:** {final_meta.get('venue', 'Unknown')}")
                        st.markdown(f"**Year:** {final_meta.get('year', 'Unknown')}")
                        st.markdown(f"**Authors:** {', '.join(final_meta.get('authors', []))}")
                        
                        if final_meta.get("introduction_summary"):
                            st.caption("**Introduction Summary:**")
                            st.info(final_meta.get("introduction_summary"))
            
            except Exception as e:
                status_container.update(label=f"❌ Error on {file_path.name}", state="error")
                st.error(f"An error occurred with {file_path.name}: {e}")
                st.code(traceback.format_exc())
            
            # 更新进度条
            progress_bar.progress((i + 1) / total_files)

        st.balloons()
        st.success(f"🎉 All {total_files} documents have been processed!")
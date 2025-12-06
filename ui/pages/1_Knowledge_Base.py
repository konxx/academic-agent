import base64
import uuid
import streamlit as st

# --- 导入业务逻辑 ---
from graph.ingestion.workflow import ingestion_app

# --- 导入你的新组件 ---
from ui.components.pdf_uploader import render_pdf_uploader
from ui.components.state_visualizer import render_ingestion_status

st.set_page_config(page_title="Knowledge Base", page_icon="📚")
st.title("📚 Knowledge Base Ingestion")
st.caption("Upload raw PDFs -> AI Agent Extraction -> Vector Database")

# ==========================================
# 1. 调用组件：文件上传
# ==========================================
# 组件内部处理了文件保存，直接返回路径
file_path = render_pdf_uploader()

if file_path:
    # ==========================================
    # 2. 启动 Agent 流程
    # ==========================================
    if st.button("🚀 Start AI Ingestion Agent", type="primary"):
        
        status_container = st.status("🤖 Agent is working...", expanded=True)
        
        # 定义临时变量收集数据 (用于最后展示)
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
            
            # --- 3. 调用组件：实时可视化状态 ---
            for event in ingestion_app.stream(initial_state, config=config):
                for node_name, state_update in event.items():
                    # ✨ 核心重构点：一行代码搞定复杂的 UI 状态更新
                    render_ingestion_status(
                        status_container, 
                        node_name, 
                        state_update, 
                        preview_data
                    )
            
            status_container.update(label="✅ Processing Complete!", state="complete", expanded=False)
            st.balloons()
            
            # ==========================================
            # 4. 展示最终结果 (这部分逻辑保留在页面层)
            # ==========================================
            st.divider()
            st.subheader("🎉 Ingestion Result")
            
            final_meta = preview_data["metadata"]
            final_images = preview_data["images"]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if final_images and len(final_images) > 0:
                    # Base64 解码显示
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
                with st.expander("Read Introduction Summary (Chinese)"):
                    st.write(final_meta.get("introduction_summary", "No summary"))
                    
        except Exception as e:
            status_container.update(label="❌ Error Occurred", state="error")
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())
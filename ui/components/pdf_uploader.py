import streamlit as st
from pathlib import Path
from config.settings import settings

def render_pdf_uploader() -> Path | None:
    """
    渲染 PDF 上传组件
    :return: 上传并保存成功后的文件绝对路径，如果未上传则返回 None
    """
    uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file:
        # 确保目录存在 (双重保险)
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # 拼接保存路径
        file_path = settings.UPLOAD_DIR / uploaded_file.name
        
        # 写入文件
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File saved: `{uploaded_file.name}`")
        return file_path
    
    return None
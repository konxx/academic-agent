import streamlit as st
from pathlib import Path
from typing import List
from config.settings import settings

def render_pdf_uploader() -> List[Path]: # 👈 返回值改为列表
    """
    渲染 PDF 上传组件 (支持多文件)
    :return: 上传并保存成功后的文件绝对路径列表
    """
    # 1. 开启 accept_multiple_files=True
    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True 
    )
    
    saved_paths = []
    
    if uploaded_files:
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # 2. 循环处理每个文件
        for uploaded_file in uploaded_files:
            file_path = settings.UPLOAD_DIR / uploaded_file.name
            
            # 写入文件
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)
            
        if saved_paths:
            st.success(f"Successfully uploaded {len(saved_paths)} files.")
            
    return saved_paths # 返回列表，如果没有文件则为空列表
import fitz  # PyMuPDF
import base64
from pathlib import Path
from typing import List
from utils.logger import logger

def load_pdf_as_images(file_path: str, max_pages: int = 5) -> List[str]:
    """
    将 PDF 的前 N 页转换为 Base64 编码的 PNG 图片列表。
    让视觉大模型直接“看”论文。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    logger.info(f"🖼️ Rendering first {max_pages} pages of {path.name} to images...")
    
    base64_images = []
    try:
        # 打开 PDF
        doc = fitz.open(path)
        read_limit = min(len(doc), max_pages)
        
        for i in range(read_limit):
            page = doc.load_page(i)
            # 设置渲染分辨率 (zoom=2 表示 2 倍清晰度，这对小字很重要)
            pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
            img_bytes = pix.tobytes("png")
            
            # 转为 Base64 字符串
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            base64_images.append(img_b64)
            
        doc.close()
        logger.info(f"✅ Successfully rendered {len(base64_images)} pages as images.")
        return base64_images
    
    except Exception as e:
        logger.error(f"❌ Error rendering PDF to images: {e}")
        raise RuntimeError(f"Error rendering PDF: {e}")
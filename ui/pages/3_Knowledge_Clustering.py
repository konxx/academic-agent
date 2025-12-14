import streamlit as st
import numpy as np
from typing import List, Dict, Any

# --- 导入核心模块 ---
from core.qdrant import qdrant_manager
from core.llm import get_embeddings
from utils.logger import logger

st.set_page_config(page_title="Topic Clustering", page_icon="🧬")

st.title("🧬 Knowledge Clustering")
st.caption("多维语义分析：分析论文与多个研究主题的关联强度")

# ==========================================
# 1. Session State (关键词管理)
# ==========================================
if "cluster_keywords" not in st.session_state:
    st.session_state.cluster_keywords = [""]

def add_keyword():
    st.session_state.cluster_keywords.append("")

def remove_keyword(index):
    if len(st.session_state.cluster_keywords) > 0:
        st.session_state.cluster_keywords.pop(index)

# ==========================================
# 2. 侧边栏：配置
# ==========================================
with st.sidebar:
    st.header("⚙️ Analysis Config")
    
    top_k = st.slider("Max Papers per Topic", 1, 20, 5, help="每个关键词最多找出几篇论文？")
    score_threshold = st.slider("Min Similarity", 0.0, 1.0, 0.2, help="过滤掉不相关的论文")
    
    st.divider()
    st.markdown("### 🕵️‍♂️ Strict Mode")
    strict_match = st.checkbox("Require Keyword Match", value=False, 
                               help="选中后，论文必须包含至少一个关键词的文本")

# ==========================================
# 3. 辅助函数：余弦相似度计算
# ==========================================
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# ==========================================
# 4. 主界面：定义关键词
# ==========================================
st.subheader("1. Define Research Topics")
st.info("输入多个关键词（如 'RAG', 'Agent', 'Evaluation'），我们将分析论文在这些维度上的得分。")

# 渲染输入框
for i, keyword in enumerate(st.session_state.cluster_keywords):
    col1, col2 = st.columns([5, 1])
    with col1:
        st.session_state.cluster_keywords[i] = st.text_input(
            f"Topic #{i+1}", 
            value=keyword, 
            key=f"kw_{i}",
            placeholder="Enter keyword..."
        )
    with col2:
        if st.button("🗑️", key=f"del_{i}"):
            remove_keyword(i)
            st.rerun()

if st.button("➕ Add Topic"):
    add_keyword()
    st.rerun()

st.divider()

# ==========================================
# 5. 执行分析 (聚合去重 + 全量打分)
# ==========================================
st.subheader("2. Consolidated Results")

if st.button("🚀 Start Multi-Topic Analysis", type="primary"):
    valid_keywords = [k.strip() for k in st.session_state.cluster_keywords if k.strip()]
    
    if not valid_keywords:
        st.warning("⚠️ Please enter at least one keyword.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            embedding_model = get_embeddings()
            client = qdrant_manager.client
            collection_name = qdrant_manager.collection_name
            
            # --- 第一步：预计算所有关键词的向量 ---
            status_text.text("🧠 Embedding keywords...")
            keyword_vectors = {}
            for kw in valid_keywords:
                keyword_vectors[kw] = embedding_model.embed_query(kw)
            
            # --- 第二步：搜索候选论文 (聚合) ---
            candidate_papers = {} # Map[id, {metadata, vector}]
            
            for idx, (kw, kw_vec) in enumerate(keyword_vectors.items()):
                status_text.text(f"🔍 Searching candidates for '{kw}'...")
                
                # 注意：必须开启 with_vectors=True 才能在本地进行多维打分
                search_result = client.query_points(
                    collection_name=collection_name,
                    query=kw_vec,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=True 
                )
                
                hits = getattr(search_result, 'points', search_result)
                
                for hit in hits:
                    # 使用 payload.get("source") 或 hit.id 作为唯一标识
                    paper_id = hit.id 
                    
                    if paper_id not in candidate_papers:
                        payload = hit.payload or {}
                        # 兼容 LangChain 的 metadata 嵌套结构
                        meta = payload.get("metadata", payload)
                        
                        candidate_papers[paper_id] = {
                            "metadata": meta,
                            "vector": hit.vector, # 获取向量
                            "source_id": hit.id
                        }
                
                progress_bar.progress((idx + 1) / (len(valid_keywords) + 1))

            # --- 第三步：交叉打分与过滤 ---
            status_text.text("📊 Calculating cross-topic scores...")
            final_results = []
            
            for pid, data in candidate_papers.items():
                meta = data["metadata"]
                paper_vec = data["vector"]
                
                # 1. 严格模式检查 (文本匹配)
                if strict_match:
                    combined_text = (
                        meta.get("title", "") + 
                        meta.get("abstract", "") + 
                        meta.get("introduction_summary", "")
                    ).lower()
                    
                    # 只要包含任意一个关键词即可保留 (或者你可以改为必须包含所有)
                    has_match = any(kw.lower() in combined_text for kw in valid_keywords)
                    if not has_match:
                        continue

                # 2. 计算该论文对 *所有* 关键词的得分
                scores = {}
                total_score = 0
                for kw, kw_vec in keyword_vectors.items():
                    # 如果 paper_vec 是 None (某些旧数据可能没存向量)，则无法计算
                    if paper_vec is None:
                        sim = 0.0
                    else:
                        sim = cosine_similarity(kw_vec, paper_vec)
                    
                    scores[kw] = sim
                    total_score += sim
                
                # 存入结果对象
                final_results.append({
                    "metadata": meta,
                    "scores": scores,
                    "avg_score": total_score / len(valid_keywords),
                    "max_score": max(scores.values()) if scores else 0
                })

            # --- 第四步：排序与展示 ---
            # 按最高匹配分排序
            final_results.sort(key=lambda x: x["max_score"], reverse=True)

            # 【新增】如果你想强制限制最终展示的总数量（例如只看全场最佳的 5 篇）
            # final_results = final_results[:5]  <-- 取消注释这行即可截断
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Found {len(final_results)} unique papers relevant to your topics.")
            
            if not final_results:
                st.warning("No papers met the criteria.")
            
            for item in final_results:
                meta = item["metadata"]
                scores = item["scores"]
                
                title = meta.get("title", "Unknown Title")
                venue = meta.get("venue", "Unknown Venue")
                year = meta.get("year", "N/A")
                authors = meta.get("authors", [])
                if isinstance(authors, list):
                    authors_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
                else:
                    authors_str = str(authors)
                
                # 外层容器
                with st.container(border=True):
                    # 标题行
                    st.markdown(f"### 📄 {title}")
                    st.caption(f"👤 {authors_str} | 🏛️ {venue} ({year})")
                    
                    st.divider()
                    
                    # 关键词得分展示区 (Grid Layout)
                    st.markdown("**Topic Relevance:**")
                    
                    # 动态列布局：每行显示 4 个得分
                    cols = st.columns(4)
                    for i, (kw, score) in enumerate(scores.items()):
                        col = cols[i % 4]
                        # 根据分数高低显示不同颜色
                        if score > 0.75:
                            color = "green"
                        elif score > 0.5:
                            color = "orange"
                        else:
                            color = "gray"
                            
                        col.markdown(f"**{kw}**: :{color}[`{score:.4f}`]")

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Clustering Error: {e}")
            if "with_vectors" in str(e):
                st.info("💡 Hint: Ensure your Qdrant instance allows retrieving vectors.")
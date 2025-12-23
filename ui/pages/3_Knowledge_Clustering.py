import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

# --- 核心模块导入 ---
from core.qdrant import qdrant_manager
from core.clustering import clustering_service
from utils.logger import logger

st.set_page_config(page_title="Knowledge Clustering", page_icon="🧬", layout="wide")

st.title("🧬 Knowledge Clustering")
st.caption("自动发现论文主题 + 交互式主题管理")

# ==========================================
# Session State 初始化
# ==========================================
if "clustering_result" not in st.session_state:
    st.session_state.clustering_result = None

if "cluster_names" not in st.session_state:
    st.session_state.cluster_names = {}

if "current_labels" not in st.session_state:
    st.session_state.current_labels = None


# ==========================================
# 辅助函数
# ==========================================
def create_scatter_plot(viz_data: Dict, n_dims: int = 2) -> go.Figure:
    """创建散点图可视化"""
    if n_dims == 2:
        fig = px.scatter(
            x=viz_data["x"],
            y=viz_data["y"],
            color=viz_data["cluster_names"],
            hover_name=viz_data["titles"],
            title="📊 论文聚类可视化 (2D)",
            labels={"x": "PC1", "y": "PC2", "color": "主题"},
        )
    else:
        fig = px.scatter_3d(
            x=viz_data["x"],
            y=viz_data["y"],
            z=viz_data["z"],
            color=viz_data["cluster_names"],
            hover_name=viz_data["titles"],
            title="📊 论文聚类可视化 (3D)",
            labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "主题"},
        )
    
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    return fig


# ==========================================
# 主界面：自动聚类
# ==========================================
st.subheader("1. 配置聚类参数")

col1, col2 = st.columns(2)

with col1:
    clustering_method = st.selectbox(
        "聚类算法",
        ["K-Means (推荐)", "DBSCAN (自动发现)"],
        help="K-Means 可指定主题数量，适合大论文库；DBSCAN 自动发现簇但可能不均匀"
    )

with col2:
    pca_components = st.slider(
        "PCA 降维维度",
        10, 100, 50,
        help="聚类前将向量降到多少维？较低维度可能发现更粗粒度的主题"
    )

st.markdown("### 📊 算法参数")

if "K-Means" in clustering_method:
    col_a, col_b = st.columns(2)
    with col_a:
        n_clusters = st.slider(
            "主题簇数量",
            3, 30, 10,
            help="将论文分为多少个主题？建议：50篇→5簇，200篇→10簇，500篇→15簇"
        )
    with col_b:
        st.info("💡 建议根据论文库大小设置 **8-15 个**主题簇")
else:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        eps_value = st.slider(
            "邻域半径 (eps)",
            0.1, 2.0, 0.5, 0.05,
            help="值越小，簇越多越细；值越大，簇越少越粗"
        )
    with col_b:
        min_samples = st.slider(
            "最小样本数",
            2, 10, 3,
            help="形成簇所需的最小邻居数"
        )
    with col_c:
        st.warning("⚠️ DBSCAN 对参数敏感，如结果不均匀请调整 eps")

generate_labels = st.checkbox(
    "🏷️ 使用 AI 生成主题标签",
    value=True,
    help="调用 LLM 为每个簇生成描述性标签"
)

st.divider()

if st.button("🚀 开始自动聚类", type="primary"):
    with st.spinner("正在分析论文库..."):
        try:
            progress = st.progress(0)
            status = st.empty()
            
            status.text("📚 正在获取论文数据...")
            papers = clustering_service.fetch_all_papers(limit=500)
            progress.progress(20)
            
            if len(papers) < 3:
                st.warning("⚠️ 论文库中论文数量不足，请先上传更多论文。")
            else:
                status.text("📉 正在进行降维...")
                vectors = np.array([p["vector"] for p in papers])
                reduced = clustering_service.reduce_dimensions(vectors, n_components=pca_components)
                progress.progress(40)
                
                status.text("🧬 正在执行聚类...")
                if "K-Means" in clustering_method:
                    labels, n_found = clustering_service.auto_cluster_kmeans(reduced, n_clusters)
                else:
                    labels, n_found = clustering_service.auto_cluster_hdbscan(
                        reduced,
                        min_cluster_size=min_samples, 
                        min_samples=min_samples,
                        eps=eps_value
                    )
                progress.progress(60)
                
                if generate_labels:
                    status.text("🏷️ 正在生成主题标签...")
                    grouped = clustering_service.group_papers_by_cluster(papers, labels)
                    cluster_names = clustering_service.generate_cluster_labels(grouped)
                else:
                    cluster_names = {i: f"Cluster {i}" for i in set(labels)}
                progress.progress(80)
                
                status.text("📊 正在准备可视化...")
                viz_data = clustering_service.prepare_visualization_data(
                    papers, labels, cluster_names, n_dims=3
                )
                progress.progress(100)
                
                st.session_state.clustering_result = {
                    "papers": papers,
                    "vectors": vectors,
                    "reduced": reduced,
                    "labels": labels,
                    "cluster_names": cluster_names,
                    "viz_data": viz_data
                }
                st.session_state.current_labels = labels
                st.session_state.cluster_names = cluster_names
                
                progress.empty()
                status.empty()
                
                st.success(f"✅ 聚类完成！发现 {n_found} 个主题簇，共 {len(papers)} 篇论文")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ 聚类失败: {str(e)}")
            logger.error(f"Clustering Error: {e}")

# --- 显示聚类结果 ---
if st.session_state.clustering_result:
    result = st.session_state.clustering_result
    papers = result["papers"]
    labels = st.session_state.current_labels
    cluster_names = st.session_state.cluster_names
    
    st.divider()
    st.subheader("2. 聚类结果可视化")
    
    viz_col1, viz_col2 = st.columns([2, 1])
    
    with viz_col1:
        viz_dims = st.radio("可视化维度", [2, 3], horizontal=True)
        viz_data = clustering_service.prepare_visualization_data(
            papers, labels, cluster_names, n_dims=viz_dims
        )
        fig = create_scatter_plot(viz_data, viz_dims)
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        st.markdown("### 📊 主题统计")
        grouped = clustering_service.group_papers_by_cluster(papers, labels)
        
        for cluster_id in sorted(grouped.keys()):
            if cluster_id == -1:
                continue
            cluster_papers = grouped[cluster_id]
            name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            st.metric(
                label=f"🏷️ {name}",
                value=f"{len(cluster_papers)} 篇"
            )
    
    st.divider()
    st.subheader("3. 交互式簇管理")
    
    manage_col1, manage_col2 = st.columns(2)
    
    with manage_col1:
        st.markdown("#### 🔗 合并簇")
        cluster_ids = [k for k in grouped.keys() if k != -1]
        clusters_to_merge = st.multiselect(
            "选择要合并的簇",
            options=cluster_ids,
            format_func=lambda x: f"{cluster_names.get(x, f'Cluster {x}')} ({len(grouped.get(x, []))}篇)"
        )
        
        if st.button("合并选中的簇") and len(clusters_to_merge) >= 2:
            new_labels = clustering_service.merge_clusters(labels, clusters_to_merge)
            st.session_state.current_labels = new_labels
            target = min(clusters_to_merge)
            merged_name = " + ".join([cluster_names.get(c, f"C{c}") for c in clusters_to_merge])
            st.session_state.cluster_names[target] = merged_name
            st.success("✅ 已合并簇")
            st.rerun()
    
    with manage_col2:
        st.markdown("#### ✂️ 拆分簇")
        cluster_to_split = st.selectbox(
            "选择要拆分的簇",
            options=cluster_ids,
            format_func=lambda x: f"{cluster_names.get(x, f'Cluster {x}')} ({len(grouped.get(x, []))}篇)"
        )
        n_splits = st.slider("拆分数量", 2, 5, 2)
        
        if st.button("拆分该簇"):
            new_labels = clustering_service.split_cluster(
                result["reduced"],
                labels,
                cluster_to_split,
                n_splits
            )
            st.session_state.current_labels = new_labels
            st.success("✅ 已拆分簇")
            st.rerun()
    
    st.divider()
    st.markdown("#### ✏️ 重命名簇")
    
    rename_col1, rename_col2 = st.columns([1, 2])
    with rename_col1:
        cluster_to_rename = st.selectbox(
            "选择簇",
            options=cluster_ids,
            format_func=lambda x: f"{cluster_names.get(x, f'Cluster {x}')}",
            key="rename_cluster"
        )
    
    current_name = cluster_names.get(cluster_to_rename, "")
    
    with rename_col2:
        new_name = st.text_input(
            "新名称",
            value=current_name,
            key=f"rename_input_{cluster_to_rename}"
        )
        if st.button("更新名称"):
            st.session_state.cluster_names[cluster_to_rename] = new_name
            st.success("✅ 已更新簇名称")
            st.rerun()
    
    st.divider()
    st.subheader("4. 论文详情")
    
    selected_cluster = st.selectbox(
        "选择主题查看论文",
        options=cluster_ids,
        format_func=lambda x: f"{cluster_names.get(x, f'Cluster {x}')} ({len(grouped.get(x, []))}篇)"
    )
    
    if selected_cluster is not None:
        cluster_papers = grouped.get(selected_cluster, [])
        
        for paper in cluster_papers:
            meta = paper["metadata"]
            title = meta.get("title", "Unknown Title")
            venue = meta.get("venue", "Unknown")
            year = meta.get("year", "N/A")
            authors = meta.get("authors", [])
            
            if isinstance(authors, list):
                authors_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
            else:
                authors_str = str(authors)
            
            with st.container(border=True):
                st.markdown(f"### 📄 {title}")
                st.caption(f"👤 {authors_str} | 🏛️ {venue} ({year})")
                
                if abstract := meta.get("abstract"):
                    with st.expander("查看摘要"):
                        st.write(abstract)
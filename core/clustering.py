"""
聚类服务模块
提供自动聚类、降维、簇管理和可视化功能
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# 确保能导入项目模块
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# 尝试导入 hdbscan，如果失败则使用 DBSCAN 作为回退
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from core.qdrant import qdrant_manager
from core.llm import get_critic_llm
from utils.logger import logger


class ClusteringService:
    """知识聚类服务 - 提供自动聚类和交互式簇管理功能"""

    def __init__(self):
        self.client = qdrant_manager.client
        self.collection_name = qdrant_manager.collection_name
        self._cache: Dict[str, Any] = {}

    def fetch_all_papers(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        从 Qdrant 获取所有论文的向量和元数据
        
        :param limit: 最大获取数量
        :return: 包含 id, vector, metadata 的论文列表
        """
        try:
            # 使用 scroll 获取所有数据
            papers = []
            offset = None
            
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True
                )
                
                points, offset = result
                
                if not points:
                    break
                
                for point in points:
                    payload = point.payload or {}
                    # 兼容 LangChain 的 metadata 嵌套结构
                    meta = payload.get("metadata", payload)
                    
                    papers.append({
                        "id": point.id,
                        "vector": point.vector,
                        "metadata": meta
                    })
                
                if len(papers) >= limit or offset is None:
                    break
            
            logger.info(f"📚 Fetched {len(papers)} papers from Qdrant")
            return papers
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch papers: {e}")
            raise

    def reduce_dimensions(
        self, 
        vectors: np.ndarray, 
        n_components: int = 50,
        for_visualization: bool = False
    ) -> np.ndarray:
        """
        使用 PCA 进行降维
        
        :param vectors: 原始向量矩阵 (n_samples, n_features)
        :param n_components: 目标维度
        :param for_visualization: 如果为 True，则降到 2D/3D
        :return: 降维后的向量
        """
        if for_visualization:
            n_components = min(3, n_components)
        
        # 确保 n_components 不超过样本数和特征数
        n_samples, n_features = vectors.shape
        n_components = min(n_components, n_samples, n_features)
        
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(vectors)
        
        explained_var = sum(pca.explained_variance_ratio_) * 100
        logger.info(f"📉 PCA: {n_features}D -> {n_components}D (explained variance: {explained_var:.1f}%)")
        
        return reduced

    def auto_cluster_hdbscan(
        self, 
        vectors: np.ndarray,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        eps: float = 0.5
    ) -> Tuple[np.ndarray, int]:
        """
        使用 HDBSCAN 进行自动聚类（如果不可用则回退到 DBSCAN）
        
        :param vectors: 向量矩阵
        :param min_cluster_size: 最小簇大小
        :param min_samples: 核心样本最小邻居数
        :param eps: DBSCAN 的邻域半径（仅在使用 DBSCAN 时生效）
        :return: (簇标签数组, 簇数量)
        """
        if HDBSCAN_AVAILABLE:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(vectors)
            algo_name = "HDBSCAN"
        else:
            # 回退到 DBSCAN
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(vectors)
            algo_name = "DBSCAN (fallback)"
        
        # 统计簇数量 (-1 表示噪声点)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"🧬 {algo_name}: Found {n_clusters} clusters, {n_noise} noise points")
        
        return labels, n_clusters

    def auto_cluster_kmeans(
        self, 
        vectors: np.ndarray,
        n_clusters: int = 5
    ) -> Tuple[np.ndarray, int]:
        """
        使用 K-Means 进行聚类（需指定簇数量）
        
        :param vectors: 向量矩阵
        :param n_clusters: 簇数量
        :return: (簇标签数组, 簇数量)
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        logger.info(f"🧬 K-Means: Created {n_clusters} clusters")
        
        return labels, n_clusters

    def generate_cluster_labels(
        self, 
        papers_by_cluster: Dict[int, List[Dict]],
        max_papers_per_cluster: int = 5
    ) -> Dict[int, str]:
        """
        使用 LLM 为每个簇生成带评分的关键词标签
        
        :param papers_by_cluster: 按簇分组的论文 {cluster_id: [papers]}
        :param max_papers_per_cluster: 每个簇用于生成标签的最大论文数
        :return: {cluster_id: topic_label}
        """
        llm = get_critic_llm()
        cluster_labels = {}
        
        for cluster_id, papers in papers_by_cluster.items():
            if cluster_id == -1:  # 跳过噪声点
                cluster_labels[-1] = "🔇 Noise / Uncategorized"
                continue
            
            # 取前 N 篇论文的标题和摘要
            sample_papers = papers[:max_papers_per_cluster]
            paper_info = "\n".join([
                f"- Title: {p['metadata'].get('title', 'Unknown')}\n  Abstract: {p['metadata'].get('abstract', '')[:300]}..."
                for p in sample_papers
            ])
            
            prompt = f"""Analyze the following academic papers and generate keyword tags with relevance scores.

Papers in this cluster ({len(papers)} total):
{paper_info}

Task:
1. Identify 3-5 keywords that best describe the common theme of these papers
2. Score each keyword from 0.0 to 1.0 based on how well it represents ALL papers in this cluster
3. Format: keyword1 (score), keyword2 (score), keyword3 (score)

Example output:
Federated Learning (0.95), Privacy Preservation (0.82), Gradient Compression (0.71)

Requirements:
- Each keyword should be 1-3 words
- Scores should reflect relevance: 0.9+ = core theme, 0.7-0.9 = important, 0.5-0.7 = related
- Use English academic terminology
- Output ONLY the formatted keywords with scores, nothing else

Keywords with scores:"""

            try:
                response = llm.invoke(prompt)
                label = response.content.strip().strip('"\'')
                cluster_labels[cluster_id] = label
                logger.info(f"🏷️ Cluster {cluster_id}: {label}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate label for cluster {cluster_id}: {e}")
                cluster_labels[cluster_id] = f"Cluster {cluster_id}"
        
        return cluster_labels

    def merge_clusters(
        self, 
        labels: np.ndarray, 
        cluster_ids_to_merge: List[int]
    ) -> np.ndarray:
        """
        合并指定的簇
        
        :param labels: 当前簇标签
        :param cluster_ids_to_merge: 要合并的簇 ID 列表
        :return: 更新后的标签
        """
        if len(cluster_ids_to_merge) < 2:
            return labels
        
        new_labels = labels.copy()
        target_cluster = min(cluster_ids_to_merge)  # 合并到最小 ID
        
        for cluster_id in cluster_ids_to_merge:
            new_labels[labels == cluster_id] = target_cluster
        
        logger.info(f"🔗 Merged clusters {cluster_ids_to_merge} -> {target_cluster}")
        return new_labels

    def split_cluster(
        self, 
        vectors: np.ndarray, 
        labels: np.ndarray, 
        cluster_id: int,
        n_splits: int = 2
    ) -> np.ndarray:
        """
        将指定簇拆分为多个子簇
        
        :param vectors: 原始向量
        :param labels: 当前簇标签
        :param cluster_id: 要拆分的簇 ID
        :param n_splits: 拆分数量
        :return: 更新后的标签
        """
        new_labels = labels.copy()
        mask = labels == cluster_id
        
        if mask.sum() < n_splits:
            logger.warning(f"⚠️ Cluster {cluster_id} has too few points to split")
            return labels
        
        # 对该簇内的点进行 K-Means
        cluster_vectors = vectors[mask]
        kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(cluster_vectors)
        
        # 分配新的簇 ID
        max_label = labels.max()
        new_cluster_ids = [cluster_id] + [max_label + i + 1 for i in range(n_splits - 1)]
        
        indices = np.where(mask)[0]
        for i, idx in enumerate(indices):
            new_labels[idx] = new_cluster_ids[sub_labels[i]]
        
        logger.info(f"✂️ Split cluster {cluster_id} into {new_cluster_ids}")
        return new_labels

    def group_papers_by_cluster(
        self, 
        papers: List[Dict], 
        labels: np.ndarray
    ) -> Dict[int, List[Dict]]:
        """
        按簇分组论文
        
        :param papers: 论文列表
        :param labels: 簇标签
        :return: {cluster_id: [papers]}
        """
        grouped = {}
        for i, paper in enumerate(papers):
            label = int(labels[i])
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(paper)
        
        return grouped

    def prepare_visualization_data(
        self, 
        papers: List[Dict], 
        labels: np.ndarray,
        cluster_names: Dict[int, str],
        n_dims: int = 2
    ) -> Dict[str, Any]:
        """
        准备可视化数据
        
        :param papers: 论文列表
        :param labels: 簇标签
        :param cluster_names: 簇名称映射
        :param n_dims: 可视化维度 (2 或 3)
        :return: 可用于 Plotly 的数据结构
        """
        vectors = np.array([p["vector"] for p in papers])
        
        # 降维到 2D 或 3D
        reduced = self.reduce_dimensions(vectors, n_components=n_dims, for_visualization=True)
        
        viz_data = {
            "x": reduced[:, 0].tolist(),
            "y": reduced[:, 1].tolist(),
            "z": reduced[:, 2].tolist() if n_dims >= 3 else None,
            "labels": labels.tolist(),
            "cluster_names": [cluster_names.get(int(l), f"Cluster {l}") for l in labels],
            "titles": [p["metadata"].get("title", "Unknown") for p in papers],
            "ids": [p["id"] for p in papers]
        }
        
        return viz_data


# 单例实例
clustering_service = ClusteringService()


if __name__ == "__main__":
    # 测试脚本
    print("-" * 50)
    print("🧬 Testing Clustering Service...")
    
    try:
        # 1. 获取论文
        papers = clustering_service.fetch_all_papers(limit=50)
        print(f"✅ Fetched {len(papers)} papers")
        
        if len(papers) < 3:
            print("⚠️ Not enough papers for clustering test")
        else:
            # 2. 准备向量
            vectors = np.array([p["vector"] for p in papers])
            print(f"📐 Vector shape: {vectors.shape}")
            
            # 3. 降维
            reduced = clustering_service.reduce_dimensions(vectors, n_components=50)
            print(f"📉 Reduced shape: {reduced.shape}")
            
            # 4. 聚类
            labels, n_clusters = clustering_service.auto_cluster_hdbscan(
                reduced, 
                min_cluster_size=2,
                min_samples=1
            )
            print(f"🧬 Found {n_clusters} clusters")
            
            # 5. 分组
            grouped = clustering_service.group_papers_by_cluster(papers, labels)
            for cluster_id, cluster_papers in grouped.items():
                print(f"   Cluster {cluster_id}: {len(cluster_papers)} papers")
        
        print("✅ Clustering Service is Ready!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

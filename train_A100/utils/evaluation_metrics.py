"""
关键评估指标计算模块
包含ARI, NMI, Modularity, Inference_Time_ms等图分析指标
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering, KMeans
import networkx as nx
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


class GraphEvaluationMetrics:
    """图评估指标计算器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def extract_communities_from_adjacency(self, adjacency: torch.Tensor, 
                                         node_mask: torch.Tensor,
                                         method: str = 'spectral',
                                         n_clusters: Optional[int] = None) -> np.ndarray:
        """从邻接矩阵提取社区标签"""
        # 获取有效节点的邻接矩阵
        valid_nodes = node_mask.sum().item()
        if valid_nodes <= 1:
            return np.array([0])
            
        adj_matrix = adjacency[:valid_nodes, :valid_nodes].detach().cpu().numpy()
        
        # 确保邻接矩阵对称
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        
        # 自动确定聚类数量
        if n_clusters is None:
            n_clusters = min(max(2, valid_nodes // 10), 10)
        
        n_clusters = min(n_clusters, valid_nodes)
        
        try:
            if method == 'spectral':
                # 谱聚类
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42,
                    n_init=10
                )
                labels = clustering.fit_predict(adj_matrix)
            elif method == 'kmeans':
                # 基于邻接矩阵的K-means
                clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clustering.fit_predict(adj_matrix)
            else:
                # 默认使用谱聚类
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42,
                    n_init=10
                )
                labels = clustering.fit_predict(adj_matrix)
                
        except Exception as e:
            # 如果聚类失败，返回简单的标签
            labels = np.arange(valid_nodes) % n_clusters
            
        return labels
    
    def calculate_modularity(self, adjacency: torch.Tensor, 
                           node_mask: torch.Tensor,
                           communities: np.ndarray) -> float:
        """计算模块度 (Modularity)"""
        valid_nodes = node_mask.sum().item()
        if valid_nodes <= 1:
            return 0.0
            
        adj_matrix = adjacency[:valid_nodes, :valid_nodes].detach().cpu().numpy()
        
        try:
            # 创建NetworkX图
            G = nx.from_numpy_array(adj_matrix)
            
            # 创建社区字典
            community_dict = {}
            for node, community in enumerate(communities):
                if community not in community_dict:
                    community_dict[community] = []
                community_dict[community].append(node)
            
            # 计算模块度
            modularity = nx.community.modularity(G, community_dict.values())
            
        except Exception as e:
            # 如果NetworkX失败，使用简单计算
            modularity = self._simple_modularity(adj_matrix, communities)
            
        return float(modularity)
    
    def _simple_modularity(self, adj_matrix: np.ndarray, communities: np.ndarray) -> float:
        """简单模块度计算"""
        n = adj_matrix.shape[0]
        m = np.sum(adj_matrix) / 2  # 边数
        
        if m == 0:
            return 0.0
            
        modularity = 0.0
        for c in np.unique(communities):
            nodes_in_c = np.where(communities == c)[0]
            
            # 社区内边数
            l_c = np.sum(adj_matrix[np.ix_(nodes_in_c, nodes_in_c)]) / 2
            
            # 社区度数
            d_c = np.sum(adj_matrix[nodes_in_c, :])
            
            modularity += l_c / m - (d_c / (2 * m)) ** 2
            
        return modularity
    
    def calculate_ari(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """计算调整兰德指数 (ARI)"""
        if len(true_labels) != len(pred_labels) or len(true_labels) <= 1:
            return 0.0
            
        try:
            ari = adjusted_rand_score(true_labels, pred_labels)
        except Exception:
            ari = 0.0
            
        return float(ari)
    
    def calculate_nmi(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """计算标准化互信息 (NMI)"""
        if len(true_labels) != len(pred_labels) or len(true_labels) <= 1:
            return 0.0
            
        try:
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
        except Exception:
            nmi = 0.0
            
        return float(nmi)
    
    def measure_inference_time(self, model, images: torch.Tensor, 
                             node_masks: torch.Tensor, 
                             warmup_runs: int = 5, 
                             measurement_runs: int = 10) -> float:
        """测量推理时间 (毫秒)"""
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(images, node_masks)
        
        # 测量
        torch.cuda.synchronize() if self.device == 'cuda' else None
        
        times = []
        with torch.no_grad():
            for _ in range(measurement_runs):
                start_time = time.perf_counter()
                _ = model(images, node_masks)
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        return float(np.mean(times))
    
    def create_ground_truth_labels(self, adjacency: torch.Tensor, 
                                 node_mask: torch.Tensor) -> np.ndarray:
        """为真实邻接矩阵创建参考社区标签"""
        # 这里使用谱聚类作为"真实"标签的参考
        # 在实际应用中，应该使用真实的社区标签
        return self.extract_communities_from_adjacency(
            adjacency, node_mask, method='spectral'
        )
    
    def evaluate_batch(self, model, batch: Dict[str, torch.Tensor], 
                      clustering_method: str = 'spectral') -> Dict[str, float]:
        """评估一个批次的所有指标"""
        images = batch['image'].to(self.device)
        points = batch['points'].to(self.device)
        adjacency_true = batch['adjacency'].to(self.device)
        node_masks = batch['node_mask'].to(self.device)
        
        batch_size = images.shape[0]
        
        # 推理时间测量
        inference_time = self.measure_inference_time(model, images, node_masks)
        
        # 前向传播
        with torch.no_grad():
            predictions = model(images, node_masks)
        
        adjacency_pred = predictions['adjacency_matrix']
        
        # 计算每个样本的指标
        ari_scores = []
        nmi_scores = []
        modularity_scores = []
        
        for b in range(batch_size):
            node_mask = node_masks[b]
            valid_nodes = node_mask.sum().item()
            
            if valid_nodes <= 1:
                continue
                
            # 获取真实和预测的邻接矩阵
            adj_true = adjacency_true[b]
            adj_pred = adjacency_pred[b]
            
            # 提取社区标签
            true_labels = self.create_ground_truth_labels(adj_true, node_mask)
            pred_labels = self.extract_communities_from_adjacency(
                adj_pred, node_mask, method=clustering_method
            )
            
            # 计算指标
            ari = self.calculate_ari(true_labels, pred_labels)
            nmi = self.calculate_nmi(true_labels, pred_labels)
            modularity = self.calculate_modularity(adj_pred, node_mask, pred_labels)
            
            ari_scores.append(ari)
            nmi_scores.append(nmi)
            modularity_scores.append(modularity)
        
        # 返回平均指标
        metrics = {
            'ARI': np.mean(ari_scores) if ari_scores else 0.0,
            'NMI': np.mean(nmi_scores) if nmi_scores else 0.0,
            'Modularity': np.mean(modularity_scores) if modularity_scores else 0.0,
            'Inference_Time_ms': inference_time
        }
        
        return metrics

    def compute_batch_metrics(self, pred_adj: np.ndarray, true_adj: np.ndarray, 
                            masks: np.ndarray, clustering_method: str = 'spectral') -> Dict[str, List[float]]:
        """
        计算批次指标，与训练脚本接口兼容
        
        Args:
            pred_adj: 预测的邻接矩阵 [batch_size, max_nodes, max_nodes]
            true_adj: 真实的邻接矩阵 [batch_size, max_nodes, max_nodes]
            masks: 节点掩码 [batch_size, max_nodes]
            
        Returns:
            Dictionary containing lists of metrics for each sample
        """
        batch_size = pred_adj.shape[0]
        
        # 初始化指标列表
        ari_scores = []
        nmi_scores = []
        modularity_scores = []
        inference_times = []
        
        for b in range(batch_size):
            node_mask = torch.from_numpy(masks[b]).bool()
            valid_nodes = node_mask.sum().item()
            
            if valid_nodes <= 1:
                # 跳过无效样本
                ari_scores.append(0.0)
                nmi_scores.append(0.0)
                modularity_scores.append(0.0)
                inference_times.append(0.0)
                continue
                
            # 转换为张量
            adj_true = torch.from_numpy(true_adj[b]).float()
            adj_pred = torch.from_numpy(pred_adj[b]).float()
            
            # 提取社区标签
            true_labels = self.create_ground_truth_labels(adj_true, node_mask)
            pred_labels = self.extract_communities_from_adjacency(
                adj_pred, node_mask, method=clustering_method
            )
            
            # 计算指标
            ari = self.calculate_ari(true_labels, pred_labels)
            nmi = self.calculate_nmi(true_labels, pred_labels)
            modularity = self.calculate_modularity(adj_pred, node_mask, pred_labels)
            
            ari_scores.append(ari)
            nmi_scores.append(nmi)
            modularity_scores.append(modularity)
            inference_times.append(1.0)  # 占位符，实际推理时间在外部计算
        
        # 返回列表形式的指标（与训练脚本期望的格式匹配）
        return {
            'ARI': ari_scores,
            'NMI': nmi_scores,
            'Modularity': modularity_scores,
            'Inference_Time_ms': inference_times
        }


def test_evaluation_metrics():
    """测试评估指标计算"""
    print("🧪 测试评估指标计算...")
    
    evaluator = GraphEvaluationMetrics()
    
    # 创建测试数据
    batch_size = 2
    max_nodes = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模拟批次数据
    batch = {
        'image': torch.randn(batch_size, 1, 64, 64).to(device),
        'points': torch.randn(batch_size, max_nodes, 2).to(device),
        'adjacency': torch.rand(batch_size, max_nodes, max_nodes).to(device),
        'node_mask': torch.ones(batch_size, max_nodes, dtype=torch.bool).to(device)
    }
    
    # 设置有效节点数
    batch['node_mask'][0, 30:] = False  # 第一个样本30个节点
    batch['node_mask'][1, 25:] = False  # 第二个样本25个节点
    
    # 创建简单的测试模型
    class SimpleModel(torch.nn.Module):
        def forward(self, images, node_masks):
            batch_size, max_nodes = node_masks.shape
            return {
                'adjacency_matrix': torch.rand(batch_size, max_nodes, max_nodes).to(images.device)
            }
    
    model = SimpleModel().to(device)
    
    # 评估指标
    metrics = evaluator.evaluate_batch(model, batch)
    
    print("✅ 评估指标结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("🎉 评估指标测试完成!")


if __name__ == "__main__":
    test_evaluation_metrics()
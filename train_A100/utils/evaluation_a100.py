"""
A100-Optimized Evaluation Framework for Image-to-Graph Models
Comprehensive evaluation utilities optimized for A100 GPU training and inference.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.feature_extraction import image as sk_image
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import json
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import A100-optimized models
from models.model_a_gnn_a100 import ModelA_GNN_A100
from models.model_b_similarity_a100 import ModelB_Similarity_A100


class A100GraphMetrics:
    """A100-optimized graph quality metrics calculator."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def calculate_graph_metrics(self, adjacency_matrix: np.ndarray, 
                              true_labels: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive graph quality metrics."""
        metrics = {}
        
        # Basic graph properties
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Connectivity metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G) if G.number_of_nodes() > 1 else 0.0
        metrics['avg_clustering'] = nx.average_clustering(G) if G.number_of_nodes() > 2 else 0.0
        
        # Path-based metrics
        try:
            if nx.is_connected(G):
                metrics['avg_path_length'] = nx.average_shortest_path_length(G)
                metrics['diameter'] = nx.diameter(G)
            else:
                # For disconnected graphs, use largest component
                largest_cc = max(nx.connected_components(G), key=len) if G.number_of_nodes() > 0 else set()
                if len(largest_cc) > 1:
                    G_sub = G.subgraph(largest_cc)
                    metrics['avg_path_length'] = nx.average_shortest_path_length(G_sub)
                    metrics['diameter'] = nx.diameter(G_sub)
                else:
                    metrics['avg_path_length'] = 0.0
                    metrics['diameter'] = 0.0
        except:
            metrics['avg_path_length'] = 0.0
            metrics['diameter'] = 0.0
        
        # Centrality metrics
        if G.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            
            metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
            metrics['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
            metrics['avg_closeness_centrality'] = np.mean(list(closeness_centrality.values()))
        else:
            metrics['avg_degree_centrality'] = 0.0
            metrics['avg_betweenness_centrality'] = 0.0
            metrics['avg_closeness_centrality'] = 0.0
        
        # Community structure metrics
        if true_labels is not None and len(np.unique(true_labels)) > 1:
            try:
                # Create communities from true labels
                communities = []
                for label in np.unique(true_labels):
                    community = set(np.where(true_labels == label)[0])
                    if len(community) > 0:
                        communities.append(community)
                
                if len(communities) > 1:
                    metrics['modularity'] = nx.community.modularity(G, communities)
                else:
                    metrics['modularity'] = 0.0
            except:
                metrics['modularity'] = 0.0
        else:
            metrics['modularity'] = 0.0
        
        # Small-world metrics
        try:
            # Calculate small-world coefficient
            if G.number_of_nodes() > 10 and G.number_of_edges() > 0:
                random_G = nx.erdos_renyi_graph(G.number_of_nodes(), nx.density(G))
                
                C = nx.average_clustering(G)
                C_rand = nx.average_clustering(random_G)
                
                if nx.is_connected(G) and nx.is_connected(random_G):
                    L = nx.average_shortest_path_length(G)
                    L_rand = nx.average_shortest_path_length(random_G)
                    
                    if C_rand > 0 and L_rand > 0:
                        metrics['small_world_coefficient'] = (C / C_rand) / (L / L_rand)
                    else:
                        metrics['small_world_coefficient'] = 0.0
                else:
                    metrics['small_world_coefficient'] = 0.0
            else:
                metrics['small_world_coefficient'] = 0.0
        except:
            metrics['small_world_coefficient'] = 0.0
        
        return metrics
    
    def calculate_spectral_metrics(self, adjacency_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate spectral properties of the graph."""
        metrics = {}
        
        try:
            # Compute Laplacian matrix
            G = nx.from_numpy_array(adjacency_matrix)
            L = nx.normalized(nx.laplacian_matrix(G), nodelist=list(G.nodes())).toarray()
            
            # Eigenvalue decomposition
            eigenvalues = np.linalg.eigvals(L)
            eigenvalues = np.real(eigenvalues)  # Take real part
            eigenvalues = np.sort(eigenvalues)
            
            metrics['spectral_gap'] = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0
            metrics['algebraic_connectivity'] = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
            metrics['largest_eigenvalue'] = eigenvalues[-1] if len(eigenvalues) > 0 else 0.0
            
            # Spectral radius
            A = adjacency_matrix
            if A.size > 0:
                spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))
                metrics['spectral_radius'] = float(spectral_radius)
            else:
                metrics['spectral_radius'] = 0.0
            
        except Exception as e:
            # Handle edge cases
            metrics['spectral_gap'] = 0.0
            metrics['algebraic_connectivity'] = 0.0
            metrics['largest_eigenvalue'] = 0.0
            metrics['spectral_radius'] = 0.0
        
        return metrics


class A100ClusteringEvaluator:
    """A100-optimized clustering evaluation with advanced metrics."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.graph_metrics = A100GraphMetrics(device)
    
    def evaluate_clustering(self, adjacency_matrix: np.ndarray, true_labels: np.ndarray,
                          predicted_coords: np.ndarray = None) -> Dict[str, float]:
        """Comprehensive clustering evaluation."""
        metrics = {}
        
        n_samples = len(true_labels)
        n_true_clusters = len(np.unique(true_labels))
        
        if n_samples <= 1 or n_true_clusters <= 1:
            return self._get_zero_metrics()
        
        # Spectral clustering on predicted adjacency
        try:
            spectral = SpectralClustering(
                n_clusters=n_true_clusters,
                affinity='precomputed',
                random_state=42,
                n_init=10,
                assign_labels='discretize'
            )
            
            predicted_labels = spectral.fit_predict(adjacency_matrix)
            
            # Core clustering metrics
            metrics['ari'] = adjusted_rand_score(true_labels, predicted_labels)
            metrics['nmi'] = normalized_mutual_info_score(true_labels, predicted_labels)
            
            # Silhouette score (if coordinates available)
            if predicted_coords is not None and len(predicted_coords) == n_samples:
                try:
                    metrics['silhouette_score'] = silhouette_score(predicted_coords, predicted_labels)
                except:
                    metrics['silhouette_score'] = 0.0
            else:
                metrics['silhouette_score'] = 0.0
            
            # Graph-based clustering metrics
            graph_metrics = self.graph_metrics.calculate_graph_metrics(adjacency_matrix, true_labels)
            metrics.update(graph_metrics)
            
            # Spectral metrics
            spectral_metrics = self.graph_metrics.calculate_spectral_metrics(adjacency_matrix)
            metrics.update(spectral_metrics)
            
            # Additional clustering quality metrics
            metrics.update(self._calculate_advanced_clustering_metrics(
                adjacency_matrix, true_labels, predicted_labels, predicted_coords
            ))
            
        except Exception as e:
            print(f"Error in clustering evaluation: {e}")
            metrics = self._get_zero_metrics()
        
        return metrics
    
    def _calculate_advanced_clustering_metrics(self, adjacency_matrix: np.ndarray,
                                             true_labels: np.ndarray, 
                                             predicted_labels: np.ndarray,
                                             predicted_coords: np.ndarray = None) -> Dict[str, float]:
        """Calculate advanced clustering quality metrics."""
        metrics = {}
        
        # Cluster purity and completeness
        metrics['purity'] = self._calculate_purity(true_labels, predicted_labels)
        metrics['completeness'] = self._calculate_completeness(true_labels, predicted_labels)
        
        # V-measure (harmonic mean of homogeneity and completeness)
        from sklearn.metrics import v_measure_score
        metrics['v_measure'] = v_measure_score(true_labels, predicted_labels)
        
        # Fowlkes-Mallows score
        from sklearn.metrics import fowlkes_mallows_score
        metrics['fowlkes_mallows'] = fowlkes_mallows_score(true_labels, predicted_labels)
        
        # Calinski-Harabasz score (if coordinates available)
        if predicted_coords is not None:
            try:
                from sklearn.metrics import calinski_harabasz_score
                metrics['calinski_harabasz'] = calinski_harabasz_score(predicted_coords, predicted_labels)
            except:
                metrics['calinski_harabasz'] = 0.0
        else:
            metrics['calinski_harabasz'] = 0.0
        
        # Davies-Bouldin score (if coordinates available)
        if predicted_coords is not None:
            try:
                from sklearn.metrics import davies_bouldin_score
                metrics['davies_bouldin'] = davies_bouldin_score(predicted_coords, predicted_labels)
            except:
                metrics['davies_bouldin'] = 0.0
        else:
            metrics['davies_bouldin'] = 0.0
        
        # Graph-based cluster quality
        metrics['intra_cluster_density'] = self._calculate_intra_cluster_density(
            adjacency_matrix, predicted_labels
        )
        metrics['inter_cluster_density'] = self._calculate_inter_cluster_density(
            adjacency_matrix, predicted_labels
        )
        
        return metrics
    
    def _calculate_purity(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
        """Calculate clustering purity."""
        confusion_matrix = pd.crosstab(predicted_labels, true_labels)
        return confusion_matrix.max().sum() / len(true_labels)
    
    def _calculate_completeness(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
        """Calculate clustering completeness."""
        confusion_matrix = pd.crosstab(true_labels, predicted_labels)
        return confusion_matrix.max().sum() / len(true_labels)
    
    def _calculate_intra_cluster_density(self, adjacency_matrix: np.ndarray, 
                                       predicted_labels: np.ndarray) -> float:
        """Calculate average intra-cluster density."""
        densities = []
        
        for cluster_id in np.unique(predicted_labels):
            cluster_nodes = np.where(predicted_labels == cluster_id)[0]
            
            if len(cluster_nodes) > 1:
                subgraph = adjacency_matrix[np.ix_(cluster_nodes, cluster_nodes)]
                density = np.sum(subgraph) / (len(cluster_nodes) * (len(cluster_nodes) - 1))
                densities.append(density)
        
        return np.mean(densities) if densities else 0.0
    
    def _calculate_inter_cluster_density(self, adjacency_matrix: np.ndarray,
                                       predicted_labels: np.ndarray) -> float:
        """Calculate average inter-cluster density."""
        densities = []
        unique_clusters = np.unique(predicted_labels)
        
        for i, cluster1 in enumerate(unique_clusters):
            for cluster2 in unique_clusters[i+1:]:
                nodes1 = np.where(predicted_labels == cluster1)[0]
                nodes2 = np.where(predicted_labels == cluster2)[0]
                
                if len(nodes1) > 0 and len(nodes2) > 0:
                    inter_edges = adjacency_matrix[np.ix_(nodes1, nodes2)]
                    density = np.sum(inter_edges) / (len(nodes1) * len(nodes2))
                    densities.append(density)
        
        return np.mean(densities) if densities else 0.0
    
    def _get_zero_metrics(self) -> Dict[str, float]:
        """Return dictionary of zero metrics for failed cases."""
        return {
            'ari': 0.0, 'nmi': 0.0, 'silhouette_score': 0.0,
            'modularity': 0.0, 'purity': 0.0, 'completeness': 0.0,
            'v_measure': 0.0, 'fowlkes_mallows': 0.0,
            'calinski_harabasz': 0.0, 'davies_bouldin': 0.0,
            'intra_cluster_density': 0.0, 'inter_cluster_density': 0.0,
            'n_nodes': 0, 'n_edges': 0, 'density': 0.0,
            'avg_clustering': 0.0, 'avg_path_length': 0.0,
            'diameter': 0.0, 'spectral_gap': 0.0,
            'algebraic_connectivity': 0.0, 'spectral_radius': 0.0
        }


class A100ModelEvaluator:
    """A100-optimized comprehensive model evaluator."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.clustering_evaluator = A100ClusteringEvaluator(device)
    
    def evaluate_model_comprehensive(self, model: nn.Module, test_loader: torch.utils.data.DataLoader,
                                   test_data: List[Dict], n_samples: int = None) -> Dict[str, Any]:
        """Comprehensive model evaluation with all metrics."""
        model.eval()
        
        if n_samples is None:
            n_samples = len(test_data)
        
        results = {
            'per_sample_metrics': [],
            'aggregate_metrics': {},
            'timing_metrics': {},
            'model_info': {}
        }
        
        # Model information
        if hasattr(model, 'get_model_info'):
            results['model_info'] = model.get_model_info()
        else:
            results['model_info'] = {
                'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        
        # Per-sample evaluation
        sample_idx = 0
        inference_times = []
        all_metrics = defaultdict(list)
        
        print(f"Evaluating model on {n_samples} samples...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Model evaluation"):
                if sample_idx >= n_samples:
                    break
                
                images = batch['image'].to(self.device)
                node_masks = batch['node_mask'].to(self.device)
                batch_size = images.shape[0]
                
                # Process each sample in batch
                for b in range(batch_size):
                    if sample_idx >= n_samples or sample_idx >= len(test_data):
                        break
                    
                    sample_data = test_data[sample_idx]
                    n_actual = sample_data['n_points']
                    
                    # Single sample inference with timing
                    single_image = images[b:b+1]
                    single_mask = node_masks[b:b+1]
                    
                    start_time = time.time()
                    predictions = model(single_image, single_mask)
                    inference_time = (time.time() - start_time) * 1000  # ms
                    
                    inference_times.append(inference_time)
                    
                    # Extract predictions
                    pred_coords = predictions['predicted_coords'][0].cpu().numpy()[:n_actual]
                    pred_adjacency = predictions['adjacency_matrix'][0].cpu().numpy()[:n_actual, :n_actual]
                    pred_count = predictions['node_counts'][0].cpu().numpy()
                    
                    # Ground truth
                    true_coords = sample_data['points'][:n_actual]
                    true_adjacency = sample_data['adjacency'][:n_actual, :n_actual]
                    true_labels = sample_data['labels']
                    true_count = n_actual
                    
                    # Calculate comprehensive metrics
                    sample_metrics = self._evaluate_single_sample(
                        pred_coords, pred_adjacency, pred_count,
                        true_coords, true_adjacency, true_labels, true_count,
                        inference_time
                    )
                    
                    results['per_sample_metrics'].append(sample_metrics)
                    
                    # Accumulate for aggregate metrics
                    for key, value in sample_metrics.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            all_metrics[key].append(value)
                    
                    sample_idx += 1
                
                if sample_idx >= n_samples:
                    break
        
        # Calculate aggregate metrics
        results['aggregate_metrics'] = self._calculate_aggregate_metrics(all_metrics)
        
        # Timing metrics
        results['timing_metrics'] = {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'total_samples': len(inference_times)
        }
        
        return results
    
    def _evaluate_single_sample(self, pred_coords: np.ndarray, pred_adjacency: np.ndarray, pred_count: float,
                               true_coords: np.ndarray, true_adjacency: np.ndarray, 
                               true_labels: np.ndarray, true_count: int,
                               inference_time: float) -> Dict[str, float]:
        """Evaluate a single sample comprehensively."""
        metrics = {}
        
        # Basic prediction metrics
        metrics['inference_time'] = inference_time
        metrics['count_error'] = abs(pred_count - true_count)
        metrics['count_relative_error'] = abs(pred_count - true_count) / max(true_count, 1)
        
        # Coordinate metrics
        if len(pred_coords) > 0 and len(true_coords) > 0:
            coord_errors = np.linalg.norm(pred_coords - true_coords, axis=1)
            metrics['coord_mse'] = np.mean(coord_errors ** 2)
            metrics['coord_rmse'] = np.sqrt(metrics['coord_mse'])
            metrics['coord_mae'] = np.mean(coord_errors)
            metrics['coord_max_error'] = np.max(coord_errors)
        else:
            metrics['coord_mse'] = float('inf')
            metrics['coord_rmse'] = float('inf')
            metrics['coord_mae'] = float('inf')
            metrics['coord_max_error'] = float('inf')
        
        # Adjacency metrics
        adj_diff = np.abs(pred_adjacency - true_adjacency)
        metrics['adj_mae'] = np.mean(adj_diff)
        metrics['adj_mse'] = np.mean(adj_diff ** 2)
        metrics['adj_max_error'] = np.max(adj_diff)
        
        # Binary adjacency metrics (threshold at 0.5)
        pred_adj_binary = (pred_adjacency > 0.5).astype(int)
        true_adj_binary = (true_adjacency > 0.3).astype(int)  # Different thresholds
        
        metrics['adj_accuracy'] = np.mean(pred_adj_binary == true_adj_binary)
        metrics['adj_precision'] = self._safe_divide(
            np.sum((pred_adj_binary == 1) & (true_adj_binary == 1)),
            np.sum(pred_adj_binary == 1)
        )
        metrics['adj_recall'] = self._safe_divide(
            np.sum((pred_adj_binary == 1) & (true_adj_binary == 1)),
            np.sum(true_adj_binary == 1)
        )
        metrics['adj_f1'] = self._safe_divide(
            2 * metrics['adj_precision'] * metrics['adj_recall'],
            metrics['adj_precision'] + metrics['adj_recall']
        )
        
        # Clustering evaluation
        clustering_metrics = self.clustering_evaluator.evaluate_clustering(
            pred_adjacency, true_labels, pred_coords
        )
        metrics.update(clustering_metrics)
        
        return metrics
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safe division with zero handling."""
        return numerator / denominator if denominator > 0 else 0.0
    
    def _calculate_aggregate_metrics(self, all_metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate statistics for all metrics."""
        aggregate = {}
        
        for metric_name, values in all_metrics.items():
            if len(values) > 0:
                values = np.array(values)
                values = values[~np.isnan(values)]  # Remove NaN values
                
                if len(values) > 0:
                    aggregate[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75))
                    }
        
        return aggregate
    
    def evaluate_sklearn_baseline(self, test_data: List[Dict], n_samples: int = None) -> Dict[str, Any]:
        """Evaluate sklearn img_to_graph baseline."""
        if n_samples is None:
            n_samples = len(test_data)
        
        results = {
            'per_sample_metrics': [],
            'aggregate_metrics': {},
            'timing_metrics': {},
            'model_info': {'method': 'sklearn_img_to_graph', 'parameters': 0}
        }
        
        inference_times = []
        all_metrics = defaultdict(list)
        
        print(f"Evaluating sklearn baseline on {n_samples} samples...")
        
        for i in tqdm(range(min(n_samples, len(test_data))), desc="sklearn evaluation"):
            sample = test_data[i]
            
            start_time = time.time()
            
            # sklearn img_to_graph
            try:
                image = sample['image']
                graph = sk_image.img_to_graph(image, mask=None, return_as=np.ndarray)
                adjacency_matrix = graph.toarray()
                
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
                
                # Ground truth
                true_labels = sample['labels']
                n_actual = sample['n_points']
                true_coords = sample['points'][:n_actual]
                true_adjacency = sample['adjacency'][:n_actual, :n_actual]
                
                # Resize adjacency matrix to match true size
                if adjacency_matrix.shape[0] != n_actual:
                    # Simple resizing strategy
                    if adjacency_matrix.shape[0] > n_actual:
                        adjacency_matrix = adjacency_matrix[:n_actual, :n_actual]
                    else:
                        # Pad with zeros
                        padded = np.zeros((n_actual, n_actual))
                        size = min(adjacency_matrix.shape[0], n_actual)
                        padded[:size, :size] = adjacency_matrix[:size, :size]
                        adjacency_matrix = padded
                
                # Evaluate (no coordinate prediction for sklearn)
                sample_metrics = self._evaluate_sklearn_sample(
                    adjacency_matrix, true_labels, true_adjacency, inference_time
                )
                
                results['per_sample_metrics'].append(sample_metrics)
                
                # Accumulate for aggregate metrics
                for key, value in sample_metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        all_metrics[key].append(value)
                
            except Exception as e:
                print(f"Error in sklearn sample {i}: {e}")
                # Add zero metrics for failed samples
                sample_metrics = self.clustering_evaluator._get_zero_metrics()
                sample_metrics['inference_time'] = 1000.0  # Default high time
                results['per_sample_metrics'].append(sample_metrics)
        
        # Calculate aggregate metrics
        results['aggregate_metrics'] = self._calculate_aggregate_metrics(all_metrics)
        
        # Timing metrics  
        results['timing_metrics'] = {
            'avg_inference_time': np.mean(inference_times) if inference_times else 1000.0,
            'std_inference_time': np.std(inference_times) if inference_times else 0.0,
            'min_inference_time': np.min(inference_times) if inference_times else 1000.0,
            'max_inference_time': np.max(inference_times) if inference_times else 1000.0,
            'total_samples': len(inference_times)
        }
        
        return results
    
    def _evaluate_sklearn_sample(self, pred_adjacency: np.ndarray, true_labels: np.ndarray,
                                true_adjacency: np.ndarray, inference_time: float) -> Dict[str, float]:
        """Evaluate sklearn sample (no coordinate predictions)."""
        metrics = {}
        
        metrics['inference_time'] = inference_time
        
        # Adjacency metrics
        adj_diff = np.abs(pred_adjacency - true_adjacency)
        metrics['adj_mae'] = np.mean(adj_diff)
        metrics['adj_mse'] = np.mean(adj_diff ** 2)
        metrics['adj_max_error'] = np.max(adj_diff)
        
        # Clustering evaluation (no coordinates available)
        clustering_metrics = self.clustering_evaluator.evaluate_clustering(
            pred_adjacency, true_labels, predicted_coords=None
        )
        metrics.update(clustering_metrics)
        
        return metrics
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare multiple model evaluation results."""
        comparison = {
            'summary_table': [],
            'statistical_tests': {},
            'rankings': {}
        }
        
        # Create summary table
        for model_name, results in model_results.items():
            agg_metrics = results['aggregate_metrics']
            timing = results['timing_metrics']
            
            summary_row = {
                'Model': model_name,
                'ARI': f"{agg_metrics.get('ari', {}).get('mean', 0):.4f} ± {agg_metrics.get('ari', {}).get('std', 0):.4f}",
                'NMI': f"{agg_metrics.get('nmi', {}).get('mean', 0):.4f} ± {agg_metrics.get('nmi', {}).get('std', 0):.4f}",
                'Modularity': f"{agg_metrics.get('modularity', {}).get('mean', 0):.4f} ± {agg_metrics.get('modularity', {}).get('std', 0):.4f}",
                'Inference Time (ms)': f"{timing.get('avg_inference_time', 0):.2f} ± {timing.get('std_inference_time', 0):.2f}",
                'Coordinate RMSE': f"{agg_metrics.get('coord_rmse', {}).get('mean', float('inf')):.4f}" if 'coord_rmse' in agg_metrics else 'N/A',
                'Adjacency MAE': f"{agg_metrics.get('adj_mae', {}).get('mean', 0):.4f} ± {agg_metrics.get('adj_mae', {}).get('std', 0):.4f}",
                'Parameters': results['model_info'].get('parameters', 0)
            }
            comparison['summary_table'].append(summary_row)
        
        # Model rankings
        metrics_for_ranking = ['ari', 'nmi', 'modularity']
        
        for metric in metrics_for_ranking:
            ranking = []
            for model_name, results in model_results.items():
                mean_value = results['aggregate_metrics'].get(metric, {}).get('mean', 0)
                ranking.append((model_name, mean_value))
            
            ranking.sort(key=lambda x: x[1], reverse=True)  # Higher is better
            comparison['rankings'][metric] = ranking
        
        # Speed ranking (lower is better)
        speed_ranking = []
        for model_name, results in model_results.items():
            avg_time = results['timing_metrics'].get('avg_inference_time', float('inf'))
            speed_ranking.append((model_name, avg_time))
        
        speed_ranking.sort(key=lambda x: x[1])  # Lower is better
        comparison['rankings']['inference_speed'] = speed_ranking
        
        return comparison
    
    def save_evaluation_results(self, results: Dict[str, Any], save_path: str):
        """Save comprehensive evaluation results."""
        # Prepare results for JSON serialization
        json_results = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (list, np.ndarray)):
                        json_results[key][sub_key] = [float(v) if isinstance(v, (int, float, np.number)) else str(v) for v in sub_value]
                    elif isinstance(sub_value, (int, float, np.number)):
                        json_results[key][sub_key] = float(sub_value)
                    else:
                        json_results[key][sub_key] = str(sub_value)
            elif isinstance(value, (list, np.ndarray)):
                json_results[key] = [float(v) if isinstance(v, (int, float, np.number)) else str(v) for v in value]
            elif isinstance(value, (int, float, np.number)):
                json_results[key] = float(value)
            else:
                json_results[key] = str(value)
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Evaluation results saved to {save_path}")


def main():
    """Main evaluation function for testing."""
    print("A100-Optimized Evaluation Framework Test")
    
    # Create evaluator
    evaluator = A100ModelEvaluator()
    
    # Test graph metrics
    graph_metrics = A100GraphMetrics()
    
    # Create test adjacency matrix
    n_nodes = 10
    test_adj = np.random.rand(n_nodes, n_nodes)
    test_adj = (test_adj + test_adj.T) / 2  # Make symmetric
    test_adj[np.diag_indices(n_nodes)] = 0  # Remove self-loops
    
    test_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    
    # Test metrics calculation
    metrics = graph_metrics.calculate_graph_metrics(test_adj, test_labels)
    print("Graph metrics:", metrics)
    
    spectral_metrics = graph_metrics.calculate_spectral_metrics(test_adj)
    print("Spectral metrics:", spectral_metrics)
    
    # Test clustering evaluation
    clustering_eval = A100ClusteringEvaluator()
    test_coords = np.random.rand(n_nodes, 2)
    
    clustering_metrics = clustering_eval.evaluate_clustering(test_adj, test_labels, test_coords)
    print("Clustering metrics:", clustering_metrics)
    
    print("A100-Optimized Evaluation Framework test completed!")


if __name__ == "__main__":
    main()
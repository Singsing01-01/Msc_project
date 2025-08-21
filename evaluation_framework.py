import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
import networkx as nx
from typing import Dict, List, Tuple, Union, Optional
import time
import warnings
warnings.filterwarnings('ignore')


class GraphPostProcessor:
    def __init__(self, k_top_edges: int = 10):
        self.k_top_edges = k_top_edges
    
    def symmetrize(self, adjacency: np.ndarray) -> np.ndarray:
        return (adjacency + adjacency.T) / 2
    
    def sparsify(self, adjacency: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        if k is None:
            k = self.k_top_edges
        
        n_nodes = adjacency.shape[0]
        sparsified = np.zeros_like(adjacency)
        
        for i in range(n_nodes):
            if k >= n_nodes - 1:
                sparsified[i] = adjacency[i]
            else:
                top_k_indices = np.argpartition(adjacency[i], -k)[-k:]
                sparsified[i, top_k_indices] = adjacency[i, top_k_indices]
        
        return sparsified
    
    def normalize_rows(self, adjacency: np.ndarray) -> np.ndarray:
        row_sums = adjacency.sum(axis=1)
        row_sums[row_sums == 0] = 1
        return adjacency / row_sums[:, np.newaxis]
    
    def process(self, adjacency: np.ndarray, 
                symmetrize: bool = True,
                sparsify: bool = True,
                normalize: bool = True) -> np.ndarray:
        
        processed = adjacency.copy()
        
        if symmetrize:
            processed = self.symmetrize(processed)
        
        if sparsify:
            processed = self.sparsify(processed)
        
        if normalize:
            processed = self.normalize_rows(processed)
        
        np.fill_diagonal(processed, 0)
        
        return processed


class SpectralClusteringEvaluator:
    def __init__(self, 
                 n_clusters: int = 2,
                 random_state: int = 42,
                 eigen_solver: str = 'amg'):
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        try:
            self.spectral_clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                eigen_solver=eigen_solver,
                random_state=random_state
            )
            self.eigen_solver = eigen_solver
        except ImportError:
            print(f"Warning: {eigen_solver} solver not available, falling back to 'arpack'")
            self.spectral_clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                eigen_solver='arpack',
                random_state=random_state
            )
            self.eigen_solver = 'arpack'
    
    def cluster(self, adjacency: np.ndarray) -> np.ndarray:
        try:
            if adjacency.sum() == 0:
                return np.zeros(adjacency.shape[0], dtype=int)
            
            adjacency_safe = np.maximum(adjacency, 0)
            
            if np.all(adjacency_safe == 0):
                return np.zeros(adjacency.shape[0], dtype=int)
            
            labels = self.spectral_clustering.fit_predict(adjacency_safe)
            return labels
            
        except Exception as e:
            if "pyamg" in str(e) and self.eigen_solver == 'amg':
                print(f"AMG solver not available, retrying with arpack...")
                try:
                    fallback_clustering = SpectralClustering(
                        n_clusters=self.n_clusters,
                        affinity='precomputed',
                        eigen_solver='arpack',
                        random_state=self.random_state
                    )
                    labels = fallback_clustering.fit_predict(adjacency_safe)
                    return labels
                except Exception as e2:
                    print(f"Fallback spectral clustering also failed: {e2}")
                    return np.zeros(adjacency.shape[0], dtype=int)
            else:
                print(f"Spectral clustering failed: {e}")
                return np.zeros(adjacency.shape[0], dtype=int)
    
    def evaluate_clustering(self, 
                          predicted_labels: np.ndarray, 
                          true_labels: np.ndarray) -> Dict[str, float]:
        
        if len(predicted_labels) != len(true_labels):
            min_len = min(len(predicted_labels), len(true_labels))
            predicted_labels = predicted_labels[:min_len]
            true_labels = true_labels[:min_len]
        
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels, average_method='geometric')
        
        return {
            'ari': ari,
            'nmi': nmi
        }


class ModularityCalculator:
    def __init__(self):
        pass
    
    def calculate_modularity(self, adjacency: np.ndarray, labels: np.ndarray) -> float:
        try:
            if adjacency.sum() == 0:
                return 0.0
            
            G = nx.from_numpy_array(adjacency)
            
            if G.number_of_edges() == 0:
                return 0.0
            
            partition = {}
            for i, label in enumerate(labels):
                if label not in partition:
                    partition[label] = []
                partition[label].append(i)
            
            communities = list(partition.values())
            
            modularity = nx.community.modularity(G, communities)
            return modularity
            
        except Exception as e:
            print(f"Modularity calculation failed: {e}")
            return 0.0


class PerformanceProfiler:
    def __init__(self):
        self.times = {}
    
    def time_function(self, func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    def profile_inference(self, model, images: torch.Tensor, node_masks: torch.Tensor, 
                         num_runs: int = 10) -> Dict[str, float]:
        
        model.eval()
        inference_times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                predictions = model(images, node_masks)
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        return {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times)
        }


class ComprehensiveEvaluator:
    def __init__(self, 
                 n_clusters: int = 2,
                 k_top_edges: int = 10,
                 random_state: int = 42):
        
        self.post_processor = GraphPostProcessor(k_top_edges)
        self.spectral_evaluator = SpectralClusteringEvaluator(n_clusters, random_state)
        self.modularity_calculator = ModularityCalculator()
        self.profiler = PerformanceProfiler()
    
    def evaluate_single_sample(self, 
                              predicted_adjacency: np.ndarray,
                              true_labels: np.ndarray,
                              process_graph: bool = True) -> Dict[str, float]:
        
        if process_graph:
            processed_adjacency = self.post_processor.process(predicted_adjacency)
        else:
            processed_adjacency = predicted_adjacency
        
        predicted_labels = self.spectral_evaluator.cluster(processed_adjacency)
        
        clustering_metrics = self.spectral_evaluator.evaluate_clustering(
            predicted_labels, true_labels
        )
        
        modularity = self.modularity_calculator.calculate_modularity(
            processed_adjacency, predicted_labels
        )
        
        return {
            'ari': clustering_metrics['ari'],
            'nmi': clustering_metrics['nmi'],
            'modularity': modularity,
            'predicted_labels': predicted_labels
        }
    
    def evaluate_model(self, 
                      model,
                      dataloader,
                      device: torch.device,
                      max_samples: Optional[int] = None) -> Dict[str, Union[float, List[float]]]:
        
        model.eval()
        
        all_ari = []
        all_nmi = []
        all_modularity = []
        all_inference_times = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if max_samples and sample_count >= max_samples:
                    break
                
                images = batch['images'].to(device)
                node_masks = batch['node_masks'].to(device)
                true_labels = batch['labels']
                
                start_time = time.time()
                predictions = model(images, node_masks)
                inference_time = time.time() - start_time
                
                batch_size = images.shape[0]
                
                for b in range(batch_size):
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    mask = node_masks[b].cpu().numpy()
                    n_valid = mask.sum()
                    
                    if n_valid <= 1:
                        continue
                    
                    pred_adj = predictions['adjacency_matrix'][b].cpu().numpy()
                    pred_adj_valid = pred_adj[:n_valid, :n_valid]
                    
                    true_labels_valid = true_labels[b][:n_valid].numpy()
                    
                    sample_metrics = self.evaluate_single_sample(
                        pred_adj_valid, true_labels_valid
                    )
                    
                    all_ari.append(sample_metrics['ari'])
                    all_nmi.append(sample_metrics['nmi'])
                    all_modularity.append(sample_metrics['modularity'])
                    all_inference_times.append(inference_time / batch_size)
                    
                    sample_count += 1
        
        if not all_ari:
            return {
                'mean_ari': 0.0,
                'std_ari': 0.0,
                'mean_nmi': 0.0,
                'std_nmi': 0.0,
                'mean_modularity': 0.0,
                'std_modularity': 0.0,
                'mean_inference_time': 0.0,
                'std_inference_time': 0.0,
                'num_samples': 0,
                'all_ari': [],
                'all_nmi': [],
                'all_modularity': []
            }
        
        return {
            'mean_ari': np.mean(all_ari),
            'std_ari': np.std(all_ari),
            'mean_nmi': np.mean(all_nmi),
            'std_nmi': np.std(all_nmi),
            'mean_modularity': np.mean(all_modularity),
            'std_modularity': np.std(all_modularity),
            'mean_inference_time': np.mean(all_inference_times),
            'std_inference_time': np.std(all_inference_times),
            'num_samples': len(all_ari),
            'all_ari': all_ari,
            'all_nmi': all_nmi,
            'all_modularity': all_modularity
        }
    
    def compare_models(self, 
                      models: Dict[str, torch.nn.Module],
                      dataloader,
                      device: torch.device,
                      max_samples: Optional[int] = None) -> Dict[str, Dict]:
        
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            results[model_name] = self.evaluate_model(
                model, dataloader, device, max_samples
            )
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Dict]):
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  ARI: {metrics['mean_ari']:.4f} ± {metrics['std_ari']:.4f}")
            print(f"  NMI: {metrics['mean_nmi']:.4f} ± {metrics['std_nmi']:.4f}")
            print(f"  Modularity: {metrics['mean_modularity']:.4f} ± {metrics['std_modularity']:.4f}")
            print(f"  Inference Time: {metrics['mean_inference_time']*1000:.2f} ± {metrics['std_inference_time']*1000:.2f} ms")
            print(f"  Samples Evaluated: {metrics['num_samples']}")
        
        if len(results) > 1:
            print(f"\nBEST PERFORMANCE:")
            best_ari_model = max(results.keys(), key=lambda k: results[k]['mean_ari'])
            print(f"  Best ARI: {best_ari_model} ({results[best_ari_model]['mean_ari']:.4f})")
            
            best_nmi_model = max(results.keys(), key=lambda k: results[k]['mean_nmi'])
            print(f"  Best NMI: {best_nmi_model} ({results[best_nmi_model]['mean_nmi']:.4f})")
            
            fastest_model = min(results.keys(), key=lambda k: results[k]['mean_inference_time'])
            print(f"  Fastest: {fastest_model} ({results[fastest_model]['mean_inference_time']*1000:.2f} ms)")


def test_evaluation_framework():
    from model_b_similarity import ModelB_Similarity
    from data_augmentation import create_data_loaders
    import numpy as np
    
    print("Testing evaluation framework...")
    
    train_data = np.load('/Users/jeremyfang/Downloads/image_to_graph/data/train_data.npy', allow_pickle=True)
    test_data = np.load('/Users/jeremyfang/Downloads/image_to_graph/data/test_data.npy', allow_pickle=True)
    
    _, test_loader = create_data_loaders(
        train_data.tolist()[:2], 
        test_data.tolist()[:3], 
        batch_size=1, 
        use_augmentation=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelB_Similarity().to(device)
    
    evaluator = ComprehensiveEvaluator()
    
    results = evaluator.evaluate_model(model, test_loader, device, max_samples=3)
    
    print("Evaluation Results:")
    print(f"  Mean ARI: {results['mean_ari']:.4f}")
    print(f"  Mean NMI: {results['mean_nmi']:.4f}")
    print(f"  Mean Modularity: {results['mean_modularity']:.4f}")
    print(f"  Mean Inference Time: {results['mean_inference_time']*1000:.2f} ms")
    print(f"  Samples: {results['num_samples']}")
    
    print("Evaluation framework test completed!")


if __name__ == "__main__":
    test_evaluation_framework()
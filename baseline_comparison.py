import numpy as np
from sklearn.feature_extraction import image
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
import torch
from typing import Dict, List, Tuple, Optional
import time
from evaluation_framework import ComprehensiveEvaluator, GraphPostProcessor


class SklearnImgToGraphBaseline:
    def __init__(self, 
                 n_neighbors: int = 10,
                 threshold: float = 0.1,
                 include_self: bool = False):
        
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.include_self = include_self
    
    def extract_graph_from_image(self, img_array: np.ndarray) -> np.ndarray:
        if len(img_array.shape) == 3:
            img_array = img_array.squeeze()
        
        try:
            graph = image.img_to_graph(img_array)
            
            adjacency = graph.toarray()
            
            if not self.include_self:
                np.fill_diagonal(adjacency, 0)
            
            adjacency = (adjacency + adjacency.T) / 2
            
            adjacency[adjacency < self.threshold] = 0
            
            return adjacency
            
        except Exception as e:
            print(f"sklearn img_to_graph failed: {e}")
            n_pixels = img_array.size
            return np.eye(n_pixels) * 0.1
    
    def predict(self, images: np.ndarray) -> List[np.ndarray]:
        predictions = []
        
        for img in images:
            adj = self.extract_graph_from_image(img)
            predictions.append(adj)
        
        return predictions


class RBFSimilarityBaseline:
    def __init__(self, 
                 gamma: float = 0.5,
                 k_neighbors: int = 10,
                 threshold: float = 0.1):
        
        self.gamma = gamma
        self.k_neighbors = k_neighbors
        self.threshold = threshold
    
    def extract_points_from_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = image.squeeze()
        
        y_coords, x_coords = np.where(image > 0.1)
        
        if len(y_coords) == 0:
            return np.empty((0, 2))
        
        points = np.column_stack([x_coords, y_coords])
        
        if len(points) > 300:
            indices = np.random.choice(len(points), 300, replace=False)
            points = points[indices]
        
        return points
    
    def build_rbf_graph(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= 1:
            return np.zeros((len(points), len(points)))
        
        n_points = len(points)
        adjacency = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    dist = np.linalg.norm(points[i] - points[j])
                    adjacency[i, j] = np.exp(-self.gamma * dist**2)
        
        adjacency = (adjacency + adjacency.T) / 2
        np.fill_diagonal(adjacency, 0)
        
        for i in range(n_points):
            row = adjacency[i]
            if self.k_neighbors < n_points - 1:
                threshold_val = np.partition(row, -self.k_neighbors)[-self.k_neighbors]
                adjacency[i][row < threshold_val] = 0
        
        return adjacency
    
    def predict(self, images: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        predictions = []
        
        for img in images:
            points = self.extract_points_from_image(img)
            adj = self.build_rbf_graph(points)
            predictions.append((adj, points))
        
        return predictions


class KNNGraphBaseline:
    def __init__(self, 
                 n_neighbors: int = 10,
                 mode: str = 'connectivity',
                 include_self: bool = False):
        
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.include_self = include_self
    
    def extract_points_from_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = image.squeeze()
        
        y_coords, x_coords = np.where(image > 0.1)
        
        if len(y_coords) == 0:
            return np.empty((0, 2))
        
        points = np.column_stack([x_coords, y_coords])
        
        if len(points) > 300:
            indices = np.random.choice(len(points), 300, replace=False)
            points = points[indices]
        
        return points
    
    def build_knn_graph(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= 1:
            return np.zeros((len(points), len(points)))
        
        try:
            graph = kneighbors_graph(points, 
                                   n_neighbors=min(self.n_neighbors, len(points)-1),
                                   mode=self.mode,
                                   include_self=self.include_self)
            
            adjacency = graph.toarray()
            adjacency = (adjacency + adjacency.T) / 2
            
            return adjacency
            
        except Exception as e:
            print(f"KNN graph construction failed: {e}")
            return np.zeros((len(points), len(points)))
    
    def predict(self, images: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        predictions = []
        
        for img in images:
            points = self.extract_points_from_image(img)
            adj = self.build_knn_graph(points)
            predictions.append((adj, points))
        
        return predictions


class BaselineEvaluator:
    def __init__(self):
        self.evaluator = ComprehensiveEvaluator()
        self.post_processor = GraphPostProcessor()
    
    def evaluate_sklearn_baseline(self, 
                                dataloader,
                                max_samples: Optional[int] = None) -> Dict[str, float]:
        
        baseline = SklearnImgToGraphBaseline()
        
        all_ari = []
        all_nmi = []
        all_modularity = []
        all_inference_times = []
        
        sample_count = 0
        
        for batch in dataloader:
            if max_samples and sample_count >= max_samples:
                break
            
            images = batch['images'].numpy()
            true_labels = batch['labels'].numpy()
            
            batch_size = images.shape[0]
            
            for b in range(batch_size):
                if max_samples and sample_count >= max_samples:
                    break
                
                image = images[b, 0]
                labels = true_labels[b]
                
                valid_mask = labels >= 0
                labels_valid = labels[valid_mask]
                
                if len(labels_valid) <= 1:
                    continue
                
                start_time = time.time()
                adj_predictions = baseline.predict([image])
                inference_time = time.time() - start_time
                
                if len(adj_predictions) > 0:
                    adj = adj_predictions[0]
                    
                    if adj.shape[0] != len(labels_valid):
                        min_size = min(adj.shape[0], len(labels_valid))
                        adj = adj[:min_size, :min_size]
                        labels_valid = labels_valid[:min_size]
                    
                    if adj.shape[0] > 1:
                        sample_metrics = self.evaluator.evaluate_single_sample(
                            adj, labels_valid, process_graph=True
                        )
                        
                        all_ari.append(sample_metrics['ari'])
                        all_nmi.append(sample_metrics['nmi'])
                        all_modularity.append(sample_metrics['modularity'])
                        all_inference_times.append(inference_time)
                
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
                'num_samples': 0
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
            'num_samples': len(all_ari)
        }
    
    def evaluate_rbf_baseline(self, 
                            dataloader,
                            max_samples: Optional[int] = None) -> Dict[str, float]:
        
        baseline = RBFSimilarityBaseline()
        
        all_ari = []
        all_nmi = []
        all_modularity = []
        all_inference_times = []
        
        sample_count = 0
        
        for batch in dataloader:
            if max_samples and sample_count >= max_samples:
                break
            
            images = batch['images'].numpy()
            true_labels = batch['labels'].numpy()
            
            batch_size = images.shape[0]
            
            for b in range(batch_size):
                if max_samples and sample_count >= max_samples:
                    break
                
                image = images[b, 0]
                labels = true_labels[b]
                
                valid_mask = labels >= 0
                labels_valid = labels[valid_mask]
                
                if len(labels_valid) <= 1:
                    continue
                
                start_time = time.time()
                predictions = baseline.predict([image])
                inference_time = time.time() - start_time
                
                if len(predictions) > 0:
                    adj, points = predictions[0]
                    
                    if adj.shape[0] > 1 and adj.shape[0] <= len(labels_valid):
                        sample_labels = labels_valid[:adj.shape[0]]
                        
                        sample_metrics = self.evaluator.evaluate_single_sample(
                            adj, sample_labels, process_graph=True
                        )
                        
                        all_ari.append(sample_metrics['ari'])
                        all_nmi.append(sample_metrics['nmi'])
                        all_modularity.append(sample_metrics['modularity'])
                        all_inference_times.append(inference_time)
                
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
                'num_samples': 0
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
            'num_samples': len(all_ari)
        }
    
    def evaluate_knn_baseline(self, 
                            dataloader,
                            max_samples: Optional[int] = None) -> Dict[str, float]:
        
        baseline = KNNGraphBaseline()
        
        all_ari = []
        all_nmi = []
        all_modularity = []
        all_inference_times = []
        
        sample_count = 0
        
        for batch in dataloader:
            if max_samples and sample_count >= max_samples:
                break
            
            images = batch['images'].numpy()
            true_labels = batch['labels'].numpy()
            
            batch_size = images.shape[0]
            
            for b in range(batch_size):
                if max_samples and sample_count >= max_samples:
                    break
                
                image = images[b, 0]
                labels = true_labels[b]
                
                valid_mask = labels >= 0
                labels_valid = labels[valid_mask]
                
                if len(labels_valid) <= 1:
                    continue
                
                start_time = time.time()
                predictions = baseline.predict([image])
                inference_time = time.time() - start_time
                
                if len(predictions) > 0:
                    adj, points = predictions[0]
                    
                    if adj.shape[0] > 1 and adj.shape[0] <= len(labels_valid):
                        sample_labels = labels_valid[:adj.shape[0]]
                        
                        sample_metrics = self.evaluator.evaluate_single_sample(
                            adj, sample_labels, process_graph=True
                        )
                        
                        all_ari.append(sample_metrics['ari'])
                        all_nmi.append(sample_metrics['nmi'])
                        all_modularity.append(sample_metrics['modularity'])
                        all_inference_times.append(inference_time)
                
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
                'num_samples': 0
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
            'num_samples': len(all_ari)
        }
    
    def evaluate_all_baselines(self, 
                             dataloader,
                             max_samples: Optional[int] = None) -> Dict[str, Dict]:
        
        print("Evaluating baseline methods...")
        
        results = {}
        
        print("  Evaluating RBF Similarity baseline...")
        results['rbf_baseline'] = self.evaluate_rbf_baseline(dataloader, max_samples)
        
        print("  Evaluating KNN Graph baseline...")
        results['knn_baseline'] = self.evaluate_knn_baseline(dataloader, max_samples)
        
        try:
            print("  Evaluating sklearn img_to_graph baseline...")
            results['sklearn_baseline'] = self.evaluate_sklearn_baseline(dataloader, max_samples)
        except Exception as e:
            print(f"  sklearn baseline failed: {e}")
            results['sklearn_baseline'] = {
                'mean_ari': 0.0, 'std_ari': 0.0,
                'mean_nmi': 0.0, 'std_nmi': 0.0,
                'mean_modularity': 0.0, 'std_modularity': 0.0,
                'mean_inference_time': 0.0, 'std_inference_time': 0.0,
                'num_samples': 0
            }
        
        return results


def test_baseline_comparison():
    from data_augmentation import create_data_loaders
    import numpy as np
    
    print("Testing baseline comparison...")
    
    train_data = np.load('/Users/jeremyfang/Downloads/image_to_graph/data/train_data.npy', allow_pickle=True)
    test_data = np.load('/Users/jeremyfang/Downloads/image_to_graph/data/test_data.npy', allow_pickle=True)
    
    _, test_loader = create_data_loaders(
        train_data.tolist()[:2], 
        test_data.tolist()[:3], 
        batch_size=1, 
        use_augmentation=False
    )
    
    evaluator = BaselineEvaluator()
    results = evaluator.evaluate_all_baselines(test_loader, max_samples=3)
    
    print("\nBaseline Results:")
    for method_name, metrics in results.items():
        print(f"\n{method_name.upper()}:")
        print(f"  ARI: {metrics['mean_ari']:.4f} ± {metrics['std_ari']:.4f}")
        print(f"  NMI: {metrics['mean_nmi']:.4f} ± {metrics['std_nmi']:.4f}")
        print(f"  Modularity: {metrics['mean_modularity']:.4f} ± {metrics['std_modularity']:.4f}")
        print(f"  Inference Time: {metrics['mean_inference_time']*1000:.2f} ms")
        print(f"  Samples: {metrics['num_samples']}")
    
    print("Baseline comparison test completed!")


if __name__ == "__main__":
    test_baseline_comparison()
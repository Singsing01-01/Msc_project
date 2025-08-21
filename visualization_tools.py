import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import torch
from typing import Dict, List, Tuple, Optional, Union
import os
from evaluation_framework import ComprehensiveEvaluator, SpectralClusteringEvaluator
from baseline_comparison import BaselineEvaluator, RBFSimilarityBaseline, KNNGraphBaseline


class ModelVisualizationTools:
    def __init__(self, figsize: Tuple[int, int] = (16, 12), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        self.evaluator = ComprehensiveEvaluator()
        
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_model_prediction(self, 
                                 model,
                                 sample_data: Dict,
                                 device: torch.device,
                                 save_path: Optional[str] = None,
                                 show_plot: bool = False) -> plt.Figure:
        
        model.eval()
        
        image = sample_data['image']
        true_points = sample_data['points']
        true_labels = sample_data['labels']
        true_adjacency = sample_data['adjacency']
        
        if torch.is_tensor(image):
            image_tensor = image.unsqueeze(0).unsqueeze(0).to(device)
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        
        n_true_nodes = len(true_points)
        node_mask = torch.ones(1, n_true_nodes, dtype=torch.bool).to(device)
        
        with torch.no_grad():
            predictions = model(image_tensor, node_mask)
        
        pred_coords = predictions['predicted_coords'][0, :n_true_nodes].cpu().numpy()
        pred_adjacency = predictions['adjacency_matrix'][0, :n_true_nodes, :n_true_nodes].cpu().numpy()
        
        spectral_evaluator = SpectralClusteringEvaluator(n_clusters=2)
        pred_labels = spectral_evaluator.cluster(pred_adjacency)
        true_labels_cluster = spectral_evaluator.cluster(true_adjacency)
        
        fig, axes = plt.subplots(2, 4, figsize=self.figsize)
        
        axes[0, 0].imshow(image, cmap='gray', aspect='equal')
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        if 'points_pixel' in sample_data:
            pixel_coords = sample_data['points_pixel']
            axes[0, 1].imshow(image, cmap='gray', alpha=0.7, aspect='equal')
            scatter = axes[0, 1].scatter(pixel_coords[:, 0], pixel_coords[:, 1], 
                                       c=true_labels, cmap='viridis', s=30, alpha=0.8)
            axes[0, 1].set_title('True Node Positions', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(scatter, ax=axes[0, 1], fraction=0.046, pad=0.04)
        else:
            axes[0, 1].scatter(true_points[:, 0], true_points[:, 1], 
                             c=true_labels, cmap='viridis', s=30, alpha=0.8)
            axes[0, 1].set_title('True Node Distribution', fontsize=14, fontweight='bold')
            axes[0, 1].axis('equal')
            axes[0, 1].grid(True, alpha=0.3)
        
        pred_coords_scaled = (pred_coords - pred_coords.min(axis=0)) / (pred_coords.max(axis=0) - pred_coords.min(axis=0) + 1e-8)
        pred_coords_scaled *= (image.shape[0] - 1)
        
        axes[0, 2].imshow(image, cmap='gray', alpha=0.7, aspect='equal')
        scatter = axes[0, 2].scatter(pred_coords_scaled[:, 0], pred_coords_scaled[:, 1], 
                                   c=pred_labels, cmap='plasma', s=30, alpha=0.8)
        axes[0, 2].set_title('Predicted Node Positions', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(scatter, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        axes[0, 3].scatter(pred_coords[:, 0], pred_coords[:, 1], 
                         c=pred_labels, cmap='plasma', s=30, alpha=0.8)
        axes[0, 3].set_title('Predicted Node Distribution', fontsize=14, fontweight='bold')
        axes[0, 3].axis('equal')
        axes[0, 3].grid(True, alpha=0.3)
        
        im1 = axes[1, 0].imshow(true_adjacency, cmap='hot', interpolation='nearest', aspect='equal')
        axes[1, 0].set_title('True Adjacency Matrix', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Node Index')
        axes[1, 0].set_ylabel('Node Index')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        im2 = axes[1, 1].imshow(pred_adjacency, cmap='hot', interpolation='nearest', aspect='equal')
        axes[1, 1].set_title('Predicted Adjacency Matrix', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Node Index')
        axes[1, 1].set_ylabel('Node Index')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        diff_adjacency = np.abs(true_adjacency - pred_adjacency)
        im3 = axes[1, 2].imshow(diff_adjacency, cmap='Reds', interpolation='nearest', aspect='equal')
        axes[1, 2].set_title('Adjacency Difference', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Node Index')
        axes[1, 2].set_ylabel('Node Index')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        if len(np.unique(true_labels_cluster)) > 1 and len(np.unique(pred_labels)) > 1:
            axes[1, 3].scatter(true_points[:, 0], true_points[:, 1], 
                             c=true_labels_cluster, marker='o', s=50, alpha=0.6, label='True Clusters')
            axes[1, 3].scatter(pred_coords[:, 0], pred_coords[:, 1], 
                             c=pred_labels, marker='x', s=50, alpha=0.8, label='Predicted Clusters')
            axes[1, 3].set_title('Clustering Comparison', fontsize=14, fontweight='bold')
            axes[1, 3].legend()
            axes[1, 3].axis('equal')
            axes[1, 3].grid(True, alpha=0.3)
        else:
            axes[1, 3].text(0.5, 0.5, 'Insufficient clusters\nfor comparison', 
                           ha='center', va='center', transform=axes[1, 3].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 3].set_title('Clustering Comparison', fontsize=14, fontweight='bold')
            axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def compare_models_visualization(self,
                                   models: Dict[str, torch.nn.Module],
                                   sample_data: Dict,
                                   device: torch.device,
                                   save_path: Optional[str] = None,
                                   show_plot: bool = False) -> plt.Figure:
        
        n_models = len(models)
        fig, axes = plt.subplots(n_models + 1, 4, figsize=(16, 4 * (n_models + 1)))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        image = sample_data['image']
        true_points = sample_data['points']
        true_labels = sample_data['labels']
        true_adjacency = sample_data['adjacency']
        
        axes[0, 0].imshow(image, cmap='gray', aspect='equal')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].scatter(true_points[:, 0], true_points[:, 1], 
                         c=true_labels, cmap='viridis', s=30, alpha=0.8)
        axes[0, 1].set_title('True Node Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].axis('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        im_true = axes[0, 2].imshow(true_adjacency, cmap='hot', interpolation='nearest', aspect='equal')
        axes[0, 2].set_title('True Adjacency Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im_true, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        spectral_evaluator = SpectralClusteringEvaluator(n_clusters=2)
        true_labels_cluster = spectral_evaluator.cluster(true_adjacency)
        axes[0, 3].scatter(true_points[:, 0], true_points[:, 1], 
                         c=true_labels_cluster, cmap='viridis', s=30, alpha=0.8)
        axes[0, 3].set_title('True Clustering', fontsize=12, fontweight='bold')
        axes[0, 3].axis('equal')
        axes[0, 3].grid(True, alpha=0.3)
        
        for idx, (model_name, model) in enumerate(models.items(), 1):
            model.eval()
            
            if torch.is_tensor(image):
                image_tensor = image.unsqueeze(0).unsqueeze(0).to(device)
            else:
                image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
            
            n_true_nodes = len(true_points)
            node_mask = torch.ones(1, n_true_nodes, dtype=torch.bool).to(device)
            
            with torch.no_grad():
                predictions = model(image_tensor, node_mask)
            
            pred_coords = predictions['predicted_coords'][0, :n_true_nodes].cpu().numpy()
            pred_adjacency = predictions['adjacency_matrix'][0, :n_true_nodes, :n_true_nodes].cpu().numpy()
            
            pred_labels = spectral_evaluator.cluster(pred_adjacency)
            
            pred_coords_scaled = (pred_coords - pred_coords.min(axis=0)) / (pred_coords.max(axis=0) - pred_coords.min(axis=0) + 1e-8)
            pred_coords_scaled *= (image.shape[0] - 1)
            
            axes[idx, 0].imshow(image, cmap='gray', alpha=0.7, aspect='equal')
            axes[idx, 0].scatter(pred_coords_scaled[:, 0], pred_coords_scaled[:, 1], 
                               c=pred_labels, cmap='plasma', s=20, alpha=0.8)
            axes[idx, 0].set_title(f'{model_name} - Predicted Positions', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].scatter(pred_coords[:, 0], pred_coords[:, 1], 
                               c=pred_labels, cmap='plasma', s=30, alpha=0.8)
            axes[idx, 1].set_title(f'{model_name} - Node Distribution', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('equal')
            axes[idx, 1].grid(True, alpha=0.3)
            
            im_pred = axes[idx, 2].imshow(pred_adjacency, cmap='hot', interpolation='nearest', aspect='equal')
            axes[idx, 2].set_title(f'{model_name} - Adjacency Matrix', fontsize=12, fontweight='bold')
            plt.colorbar(im_pred, ax=axes[idx, 2], fraction=0.046, pad=0.04)
            
            axes[idx, 3].scatter(pred_coords[:, 0], pred_coords[:, 1], 
                               c=pred_labels, cmap='plasma', s=30, alpha=0.8)
            axes[idx, 3].set_title(f'{model_name} - Clustering', fontsize=12, fontweight='bold')
            axes[idx, 3].axis('equal')
            axes[idx, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_baseline_comparison(self,
                                    sample_data: Dict,
                                    save_path: Optional[str] = None,
                                    show_plot: bool = False) -> plt.Figure:
        
        fig, axes = plt.subplots(2, 4, figsize=self.figsize)
        
        image = sample_data['image']
        true_points = sample_data['points']
        true_labels = sample_data['labels']
        true_adjacency = sample_data['adjacency']
        
        axes[0, 0].imshow(image, cmap='gray', aspect='equal')
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].scatter(true_points[:, 0], true_points[:, 1], 
                         c=true_labels, cmap='viridis', s=30, alpha=0.8)
        axes[0, 1].set_title('True Node Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].axis('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        rbf_baseline = RBFSimilarityBaseline()
        rbf_predictions = rbf_baseline.predict([image])
        if rbf_predictions:
            rbf_adj, rbf_points = rbf_predictions[0]
            
            if len(rbf_points) > 0:
                spectral_evaluator = SpectralClusteringEvaluator(n_clusters=2)
                rbf_labels = spectral_evaluator.cluster(rbf_adj)
                
                axes[0, 2].scatter(rbf_points[:, 0], rbf_points[:, 1], 
                                 c=rbf_labels, cmap='plasma', s=30, alpha=0.8)
                axes[0, 2].set_title('RBF Baseline - Nodes', fontsize=14, fontweight='bold')
                axes[0, 2].axis('equal')
                axes[0, 2].grid(True, alpha=0.3)
                
                im_rbf = axes[0, 3].imshow(rbf_adj, cmap='hot', interpolation='nearest', aspect='equal')
                axes[0, 3].set_title('RBF Baseline - Adjacency', fontsize=14, fontweight='bold')
                plt.colorbar(im_rbf, ax=axes[0, 3], fraction=0.046, pad=0.04)
            else:
                axes[0, 2].text(0.5, 0.5, 'No points detected', ha='center', va='center', 
                              transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('RBF Baseline - Nodes', fontsize=14, fontweight='bold')
                axes[0, 3].text(0.5, 0.5, 'No adjacency', ha='center', va='center', 
                              transform=axes[0, 3].transAxes)
                axes[0, 3].set_title('RBF Baseline - Adjacency', fontsize=14, fontweight='bold')
        
        knn_baseline = KNNGraphBaseline()
        knn_predictions = knn_baseline.predict([image])
        if knn_predictions:
            knn_adj, knn_points = knn_predictions[0]
            
            if len(knn_points) > 0:
                spectral_evaluator = SpectralClusteringEvaluator(n_clusters=2)
                knn_labels = spectral_evaluator.cluster(knn_adj)
                
                axes[1, 0].scatter(knn_points[:, 0], knn_points[:, 1], 
                                 c=knn_labels, cmap='plasma', s=30, alpha=0.8)
                axes[1, 0].set_title('KNN Baseline - Nodes', fontsize=14, fontweight='bold')
                axes[1, 0].axis('equal')
                axes[1, 0].grid(True, alpha=0.3)
                
                im_knn = axes[1, 1].imshow(knn_adj, cmap='hot', interpolation='nearest', aspect='equal')
                axes[1, 1].set_title('KNN Baseline - Adjacency', fontsize=14, fontweight='bold')
                plt.colorbar(im_knn, ax=axes[1, 1], fraction=0.046, pad=0.04)
            else:
                axes[1, 0].text(0.5, 0.5, 'No points detected', ha='center', va='center', 
                              transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('KNN Baseline - Nodes', fontsize=14, fontweight='bold')
                axes[1, 1].text(0.5, 0.5, 'No adjacency', ha='center', va='center', 
                              transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('KNN Baseline - Adjacency', fontsize=14, fontweight='bold')
        
        im_true = axes[1, 2].imshow(true_adjacency, cmap='hot', interpolation='nearest', aspect='equal')
        axes[1, 2].set_title('True Adjacency Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im_true, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        spectral_evaluator = SpectralClusteringEvaluator(n_clusters=2)
        true_labels_cluster = spectral_evaluator.cluster(true_adjacency)
        axes[1, 3].scatter(true_points[:, 0], true_points[:, 1], 
                         c=true_labels_cluster, cmap='viridis', s=30, alpha=0.8)
        axes[1, 3].set_title('True Clustering', fontsize=14, fontweight='bold')
        axes[1, 3].axis('equal')
        axes[1, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def create_performance_comparison_plot(self,
                                         results: Dict[str, Dict],
                                         save_path: Optional[str] = None,
                                         show_plot: bool = False) -> plt.Figure:
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = list(results.keys())
        ari_means = [results[method]['mean_ari'] for method in methods]
        ari_stds = [results[method]['std_ari'] for method in methods]
        nmi_means = [results[method]['mean_nmi'] for method in methods]
        nmi_stds = [results[method]['std_nmi'] for method in methods]
        mod_means = [results[method]['mean_modularity'] for method in methods]
        mod_stds = [results[method]['std_modularity'] for method in methods]
        inf_means = [results[method]['mean_inference_time'] * 1000 for method in methods]
        inf_stds = [results[method]['std_inference_time'] * 1000 for method in methods]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        axes[0, 0].bar(methods, ari_means, yerr=ari_stds, capsize=5, color=colors, alpha=0.7)
        axes[0, 0].set_title('Adjusted Rand Index (ARI)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('ARI Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(methods, nmi_means, yerr=nmi_stds, capsize=5, color=colors, alpha=0.7)
        axes[0, 1].set_title('Normalized Mutual Information (NMI)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('NMI Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].bar(methods, mod_means, yerr=mod_stds, capsize=5, color=colors, alpha=0.7)
        axes[1, 0].set_title('Modularity', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Modularity Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(methods, inf_means, yerr=inf_stds, capsize=5, color=colors, alpha=0.7)
        axes[1, 1].set_title('Inference Time', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig


def test_visualization_tools():
    from model_b_similarity import ModelB_Similarity
    from data_augmentation import create_data_loaders
    import numpy as np
    
    print("Testing visualization tools...")
    
    train_data = np.load('/Users/jeremyfang/Downloads/image_to_graph/data/train_data.npy', allow_pickle=True)
    test_data = np.load('/Users/jeremyfang/Downloads/image_to_graph/data/test_data.npy', allow_pickle=True)
    
    _, test_loader = create_data_loaders(
        train_data.tolist()[:1], 
        test_data.tolist()[:1], 
        batch_size=1, 
        use_augmentation=False
    )
    
    sample_batch = next(iter(test_loader))
    sample_data = {
        'image': sample_batch['images'][0, 0].numpy(),
        'points': sample_batch['points'][0].numpy(),
        'labels': sample_batch['labels'][0].numpy(),
        'adjacency': sample_batch['adjacency'][0].numpy(),
        'points_pixel': sample_batch['points_pixel'][0].numpy()
    }
    
    valid_mask = sample_data['labels'] >= 0
    for key in ['points', 'labels', 'points_pixel']:
        sample_data[key] = sample_data[key][valid_mask]
    
    n_valid = valid_mask.sum()
    sample_data['adjacency'] = sample_data['adjacency'][:n_valid, :n_valid]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelB_Similarity().to(device)
    
    visualizer = ModelVisualizationTools()
    
    os.makedirs('/Users/jeremyfang/Downloads/image_to_graph/visualizations', exist_ok=True)
    
    print("Creating model prediction visualization...")
    visualizer.visualize_model_prediction(
        model, sample_data, device,
        save_path='/Users/jeremyfang/Downloads/image_to_graph/visualizations/model_prediction.png'
    )
    
    print("Creating baseline comparison visualization...")
    visualizer.visualize_baseline_comparison(
        sample_data,
        save_path='/Users/jeremyfang/Downloads/image_to_graph/visualizations/baseline_comparison.png'
    )
    
    print("Visualization tools test completed!")


if __name__ == "__main__":
    test_visualization_tools()
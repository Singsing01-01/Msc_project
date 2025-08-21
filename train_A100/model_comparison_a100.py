"""
A100-Optimized Model Comparison Script for Image-to-Graph Training
Comprehensive comparison of Model A vs Model B vs sklearn baseline with A100 optimizations.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import image as sk_image
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import networkx as nx
import time
import os
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import A100-optimized models and utilities
from models.model_a_gnn_a100 import ModelA_GNN_A100
from models.model_b_similarity_a100 import ModelB_Similarity_A100
from data.data_generation_a100 import A100DataGenerator
from training_pipeline_a100 import A100DataLoaderOptimized, A100TrainingConfig


class A100ModelEvaluator:
    """A100-optimized model evaluator with comprehensive metrics."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def load_trained_model(self, model_type: str, checkpoint_path: str):
        """Load a trained model from checkpoint."""
        if model_type == 'model_a':
            model = ModelA_GNN_A100(
                input_channels=1,
                feature_dim=256,
                max_nodes=350,
                coord_dim=2,
                hidden_dim=128,
                node_feature_dim=64
            ).to(self.device)
        elif model_type == 'model_b':
            model = ModelB_Similarity_A100(
                input_channels=1,
                feature_dim=256,
                max_nodes=350,
                coord_dim=2,
                similarity_hidden_dim=64,
                similarity_mode='hybrid',
                correction_mode='mlp'
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def evaluate_sklearn_baseline(self, test_data: List[Dict], n_samples: int = None) -> Dict:
        """Evaluate sklearn img_to_graph baseline."""
        print("Evaluating sklearn baseline...")
        
        if n_samples is None:
            n_samples = len(test_data)
        
        sklearn_results = {
            'ari_scores': [],
            'nmi_scores': [],
            'modularity_scores': [],
            'inference_times': [],
            'method': 'sklearn_img_to_graph'
        }
        
        for i in tqdm(range(min(n_samples, len(test_data))), desc="sklearn evaluation"):
            sample = test_data[i]
            
            start_time = time.time()
            
            # Convert image to graph using sklearn
            image = sample['image']
            
            # Create graph from image
            graph = sk_image.img_to_graph(image, mask=None, return_as=np.ndarray)
            
            # Get true labels
            true_labels = sample['labels']
            n_clusters = len(np.unique(true_labels))
            
            # Perform spectral clustering
            if graph.shape[0] > 1 and n_clusters > 1:
                try:
                    spectral = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity='precomputed',
                        random_state=42,
                        n_init=10
                    )
                    predicted_labels = spectral.fit_predict(graph.toarray())
                    
                    # Calculate metrics
                    ari = adjusted_rand_score(true_labels, predicted_labels)
                    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
                    
                    # Calculate modularity
                    G = nx.from_numpy_array(graph.toarray())
                    if len(G.nodes()) > 0 and len(G.edges()) > 0:
                        communities = [{i for i, label in enumerate(predicted_labels) if label == c} 
                                     for c in np.unique(predicted_labels)]
                        communities = [c for c in communities if len(c) > 0]
                        if communities:
                            modularity = nx.community.modularity(G, communities)
                        else:
                            modularity = 0.0
                    else:
                        modularity = 0.0
                    
                except Exception as e:
                    print(f"Error in sample {i}: {e}")
                    ari, nmi, modularity = 0.0, 0.0, 0.0
            else:
                ari, nmi, modularity = 0.0, 0.0, 0.0
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            sklearn_results['ari_scores'].append(ari)
            sklearn_results['nmi_scores'].append(nmi)
            sklearn_results['modularity_scores'].append(modularity)
            sklearn_results['inference_times'].append(inference_time)
        
        # Calculate averages
        sklearn_results['avg_ari'] = np.mean(sklearn_results['ari_scores'])
        sklearn_results['avg_nmi'] = np.mean(sklearn_results['nmi_scores'])
        sklearn_results['avg_modularity'] = np.mean(sklearn_results['modularity_scores'])
        sklearn_results['avg_inference_time'] = np.mean(sklearn_results['inference_times'])
        
        return sklearn_results
    
    def evaluate_deep_learning_model(self, model, model_name: str, test_loader, 
                                   test_data: List[Dict]) -> Dict:
        """Evaluate a deep learning model."""
        print(f"Evaluating {model_name}...")
        
        model.eval()
        results = {
            'ari_scores': [],
            'nmi_scores': [],
            'modularity_scores': [],
            'inference_times': [],
            'method': model_name,
            'coordinate_errors': [],
            'edge_accuracies': []
        }
        
        sample_idx = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"{model_name} evaluation"):
                images = batch['image'].to(self.device)
                node_masks = batch['node_mask'].to(self.device)
                true_points = batch['points'].cpu().numpy()
                true_adjacency = batch['adjacency'].cpu().numpy()
                
                batch_size = images.shape[0]
                
                for b in range(batch_size):
                    if sample_idx >= len(test_data):
                        break
                    
                    sample_data = test_data[sample_idx]
                    true_labels = sample_data['labels']
                    n_actual = sample_data['n_points']
                    
                    # Single sample inference
                    start_time = time.time()
                    
                    single_image = images[b:b+1]
                    single_mask = node_masks[b:b+1]
                    
                    predictions = model(single_image, single_mask)
                    
                    inference_time = (time.time() - start_time) * 1000  # ms
                    
                    # Extract predictions
                    pred_coords = predictions['predicted_coords'][0].cpu().numpy()
                    pred_adjacency = predictions['adjacency_matrix'][0].cpu().numpy()
                    
                    # Calculate coordinate error
                    if n_actual > 0:
                        coord_error = np.mean(np.linalg.norm(
                            pred_coords[:n_actual] - true_points[b][:n_actual], axis=1
                        ))
                        results['coordinate_errors'].append(coord_error)
                        
                        # Calculate edge accuracy
                        true_adj_binary = (true_adjacency[b][:n_actual, :n_actual] > 0.3).astype(int)
                        pred_adj_binary = (pred_adjacency[:n_actual, :n_actual] > 0.5).astype(int)
                        edge_accuracy = np.mean(true_adj_binary == pred_adj_binary)
                        results['edge_accuracies'].append(edge_accuracy)
                    
                    # Perform spectral clustering on predicted adjacency
                    try:
                        if n_actual > 1:
                            pred_adj_sample = pred_adjacency[:n_actual, :n_actual]
                            n_clusters = len(np.unique(true_labels))
                            
                            if n_clusters > 1:
                                spectral = SpectralClustering(
                                    n_clusters=n_clusters,
                                    affinity='precomputed',
                                    random_state=42,
                                    n_init=10
                                )
                                predicted_labels = spectral.fit_predict(pred_adj_sample)
                                
                                # Calculate metrics
                                ari = adjusted_rand_score(true_labels, predicted_labels)
                                nmi = normalized_mutual_info_score(true_labels, predicted_labels)
                                
                                # Calculate modularity
                                G = nx.from_numpy_array(pred_adj_sample)
                                if len(G.nodes()) > 0 and len(G.edges()) > 0:
                                    communities = [{i for i, label in enumerate(predicted_labels) if label == c} 
                                                 for c in np.unique(predicted_labels)]
                                    communities = [c for c in communities if len(c) > 0]
                                    if communities:
                                        modularity = nx.community.modularity(G, communities)
                                    else:
                                        modularity = 0.0
                                else:
                                    modularity = 0.0
                            else:
                                ari, nmi, modularity = 0.0, 0.0, 0.0
                        else:
                            ari, nmi, modularity = 0.0, 0.0, 0.0
                    except Exception as e:
                        print(f"Error in {model_name} sample {sample_idx}: {e}")
                        ari, nmi, modularity = 0.0, 0.0, 0.0
                    
                    results['ari_scores'].append(ari)
                    results['nmi_scores'].append(nmi)
                    results['modularity_scores'].append(modularity)
                    results['inference_times'].append(inference_time)
                    
                    sample_idx += 1
                
                if sample_idx >= len(test_data):
                    break
        
        # Calculate averages
        results['avg_ari'] = np.mean(results['ari_scores'])
        results['avg_nmi'] = np.mean(results['nmi_scores'])
        results['avg_modularity'] = np.mean(results['modularity_scores'])
        results['avg_inference_time'] = np.mean(results['inference_times'])
        results['avg_coordinate_error'] = np.mean(results['coordinate_errors'])
        results['avg_edge_accuracy'] = np.mean(results['edge_accuracies'])
        
        return results
    
    def create_comprehensive_comparison(self, results_dict: Dict, save_dir: str):
        """Create comprehensive comparison visualizations and reports."""
        
        # Create summary table
        summary_data = []
        for method, results in results_dict.items():
            summary_data.append({
                'Method': method,
                'ARI': f"{results['avg_ari']:.4f} ± {np.std(results['ari_scores']):.4f}",
                'NMI': f"{results['avg_nmi']:.4f} ± {np.std(results['nmi_scores']):.4f}",
                'Modularity': f"{results['avg_modularity']:.4f} ± {np.std(results['modularity_scores']):.4f}",
                'Inference Time (ms)': f"{results['avg_inference_time']:.2f} ± {np.std(results['inference_times']):.2f}",
                'Samples': len(results['ari_scores'])
            })
            
            # Add model-specific metrics
            if 'avg_coordinate_error' in results:
                summary_data[-1]['Coord Error'] = f"{results['avg_coordinate_error']:.4f}"
                summary_data[-1]['Edge Accuracy'] = f"{results['avg_edge_accuracy']:.4f}"
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(save_dir, 'model_comparison_summary.csv'), index=False)
        
        # Create detailed comparison plots
        self._plot_performance_comparison(results_dict, save_dir)
        self._plot_inference_time_comparison(results_dict, save_dir)
        self._plot_metric_distributions(results_dict, save_dir)
        
        # Create detailed report
        self._generate_detailed_report(results_dict, save_dir)
        
        print(f"Comprehensive comparison saved to {save_dir}")
        
        return summary_df
    
    def _plot_performance_comparison(self, results_dict: Dict, save_dir: str):
        """Plot performance comparison across metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = list(results_dict.keys())
        metrics = ['avg_ari', 'avg_nmi', 'avg_modularity', 'avg_inference_time']
        metric_names = ['ARI', 'NMI', 'Modularity', 'Inference Time (ms)']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            values = [results_dict[method][metric] for method in methods]
            errors = [np.std(results_dict[method][metric.replace('avg_', '') + 's']) for method in methods]
            
            bars = ax.bar(methods, values, yerr=errors, capsize=5, alpha=0.7)
            ax.set_title(f'{name} Comparison', fontsize=14)
            ax.set_ylabel(name)
            
            # Color bars based on performance
            if metric != 'avg_inference_time':  # Higher is better
                best_idx = np.argmax(values)
            else:  # Lower is better
                best_idx = np.argmin(values)
            
            for i, bar in enumerate(bars):
                if i == best_idx:
                    bar.set_color('green')
                    bar.set_alpha(0.8)
            
            # Add value labels on bars
            for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + error,
                       f'{value:.4f}' if metric != 'avg_inference_time' else f'{value:.2f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inference_time_comparison(self, results_dict: Dict, save_dir: str):
        """Plot detailed inference time comparison."""
        plt.figure(figsize=(12, 8))
        
        methods = list(results_dict.keys())
        inference_times = [results_dict[method]['avg_inference_time'] for method in methods]
        
        # Create bar plot with log scale
        bars = plt.bar(methods, inference_times, alpha=0.7)
        plt.yscale('log')
        plt.ylabel('Inference Time (ms) - Log Scale')
        plt.title('Inference Time Comparison (Log Scale)')
        
        # Color the fastest method
        fastest_idx = np.argmin(inference_times)
        bars[fastest_idx].set_color('green')
        bars[fastest_idx].set_alpha(0.8)
        
        # Add speedup annotations
        baseline_time = max(inference_times)  # Usually sklearn
        for i, (bar, time) in enumerate(zip(bars, inference_times)):
            speedup = baseline_time / time
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}ms\\n{speedup:.1f}x faster' if speedup > 1 else f'{time:.2f}ms',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_distributions(self, results_dict: Dict, save_dir: str):
        """Plot metric distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['ari_scores', 'nmi_scores', 'modularity_scores', 'inference_times']
        metric_names = ['ARI', 'NMI', 'Modularity', 'Inference Time (ms)']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            for method in results_dict.keys():
                values = results_dict[method][metric]
                ax.hist(values, alpha=0.6, label=method, bins=20)
            
            ax.set_title(f'{name} Distribution')
            ax.set_xlabel(name)
            ax.set_ylabel('Frequency')
            ax.legend()
            
            if metric == 'inference_times':
                ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metric_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_detailed_report(self, results_dict: Dict, save_dir: str):
        """Generate detailed text report."""
        report = []
        report.append("A100-Optimized Image-to-Graph Model Comparison Report")
        report.append("=" * 60)
        report.append("")
        
        # Overall summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        
        # Find best performing models
        best_ari_method = max(results_dict.keys(), key=lambda x: results_dict[x]['avg_ari'])
        best_nmi_method = max(results_dict.keys(), key=lambda x: results_dict[x]['avg_nmi'])
        best_modularity_method = max(results_dict.keys(), key=lambda x: results_dict[x]['avg_modularity'])
        fastest_method = min(results_dict.keys(), key=lambda x: results_dict[x]['avg_inference_time'])
        
        report.append(f"Best ARI: {best_ari_method} ({results_dict[best_ari_method]['avg_ari']:.4f})")
        report.append(f"Best NMI: {best_nmi_method} ({results_dict[best_nmi_method]['avg_nmi']:.4f})")
        report.append(f"Best Modularity: {best_modularity_method} ({results_dict[best_modularity_method]['avg_modularity']:.4f})")
        report.append(f"Fastest Inference: {fastest_method} ({results_dict[fastest_method]['avg_inference_time']:.2f}ms)")
        report.append("")
        
        # Detailed method comparison
        for method, results in results_dict.items():
            report.append(f"METHOD: {method.upper()}")
            report.append("-" * 30)
            report.append(f"Samples Evaluated: {len(results['ari_scores'])}")
            report.append(f"Average ARI: {results['avg_ari']:.4f} ± {np.std(results['ari_scores']):.4f}")
            report.append(f"Average NMI: {results['avg_nmi']:.4f} ± {np.std(results['nmi_scores']):.4f}")
            report.append(f"Average Modularity: {results['avg_modularity']:.4f} ± {np.std(results['modularity_scores']):.4f}")
            report.append(f"Average Inference Time: {results['avg_inference_time']:.2f} ± {np.std(results['inference_times']):.2f} ms")
            
            if 'avg_coordinate_error' in results:
                report.append(f"Average Coordinate Error: {results['avg_coordinate_error']:.4f}")
                report.append(f"Average Edge Accuracy: {results['avg_edge_accuracy']:.4f}")
            
            # Calculate speedup vs sklearn
            if 'sklearn' in results_dict and method != list(results_dict.keys())[0]:
                sklearn_time = results_dict[list(results_dict.keys())[0]]['avg_inference_time']
                speedup = sklearn_time / results['avg_inference_time']
                report.append(f"Speedup vs Baseline: {speedup:.1f}x")
            
            report.append("")
        
        # Performance analysis
        report.append("PERFORMANCE ANALYSIS")
        report.append("-" * 20)
        
        # Target achievement
        target_ari = 0.80
        for method, results in results_dict.items():
            if results['avg_ari'] >= target_ari:
                report.append(f"✓ {method} achieved target ARI ≥ {target_ari}")
            else:
                report.append(f"✗ {method} did not achieve target ARI ≥ {target_ari}")
        
        report.append("")
        
        # Save report
        with open(os.path.join(save_dir, 'detailed_comparison_report.txt'), 'w') as f:
            f.write('\\n'.join(report))


def main():
    """Main comparison function."""
    print("A100-Optimized Model Comparison Starting...")
    
    # Setup
    config = A100TrainingConfig()
    save_dir = "/Users/jeremyfang/Downloads/image_to_graph/train_A100/results/comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load test data
    data_generator = A100DataGenerator()
    try:
        train_data, test_data = data_generator.load_dataset_optimized(
            "/Users/jeremyfang/Downloads/image_to_graph/train_A100/data"
        )
        print(f"Loaded {len(test_data)} test samples")
    except:
        print("Error loading data. Please run data generation first.")
        return
    
    # Create evaluator
    evaluator = A100ModelEvaluator()
    
    # Create test data loader
    data_loader_manager = A100DataLoaderOptimized(config)
    _, test_loader = data_loader_manager.create_data_loaders(train_data, test_data)
    
    results = {}
    
    # Evaluate sklearn baseline
    print("\\n" + "="*50)
    print("Evaluating sklearn img_to_graph baseline")
    print("="*50)
    
    sklearn_results = evaluator.evaluate_sklearn_baseline(test_data, n_samples=50)
    results['sklearn_img_to_graph'] = sklearn_results
    
    # Check for trained models and evaluate
    models_dir = "/Users/jeremyfang/Downloads/image_to_graph/train_A100/results"
    
    # Model A
    model_a_path = os.path.join(models_dir, 'model_a_best_model.pth')
    if os.path.exists(model_a_path):
        print("\\n" + "="*50)
        print("Evaluating Model A (GNN)")
        print("="*50)
        
        model_a = evaluator.load_trained_model('model_a', model_a_path)
        model_a_results = evaluator.evaluate_deep_learning_model(
            model_a, 'Model A (GNN)', test_loader, test_data
        )
        results['Model A (GNN)'] = model_a_results
    else:
        print("Model A checkpoint not found. Skipping evaluation.")
    
    # Model B
    model_b_path = os.path.join(models_dir, 'model_b_best_model.pth')
    if os.path.exists(model_b_path):
        print("\\n" + "="*50)
        print("Evaluating Model B (Similarity)")
        print("="*50)
        
        model_b = evaluator.load_trained_model('model_b', model_b_path)
        model_b_results = evaluator.evaluate_deep_learning_model(
            model_b, 'Model B (Similarity)', test_loader, test_data
        )
        results['Model B (Similarity)'] = model_b_results
    else:
        print("Model B checkpoint not found. Skipping evaluation.")
    
    # Create comprehensive comparison
    if len(results) > 1:
        print("\\n" + "="*50)
        print("Creating Comprehensive Comparison")
        print("="*50)
        
        summary_df = evaluator.create_comprehensive_comparison(results, save_dir)
        
        print("\\nComparison Summary:")
        print(summary_df.to_string(index=False))
        
        # Save raw results
        with open(os.path.join(save_dir, 'raw_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for method, data in results.items():
                json_results[method] = {}
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        json_results[method][key] = [float(v) for v in value]
                    else:
                        json_results[method][key] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
            
            json.dump(json_results, f, indent=2)
        
        print(f"\\nDetailed results saved to: {save_dir}")
        
        # Print key findings
        print("\\n" + "="*50)
        print("KEY FINDINGS")
        print("="*50)
        
        for method, data in results.items():
            print(f"{method}:")
            print(f"  ARI: {data['avg_ari']:.4f}")
            print(f"  NMI: {data['avg_nmi']:.4f}")
            print(f"  Modularity: {data['avg_modularity']:.4f}")
            print(f"  Inference Time: {data['avg_inference_time']:.2f}ms")
            
            if method != 'sklearn_img_to_graph' and 'sklearn_img_to_graph' in results:
                sklearn_time = results['sklearn_img_to_graph']['avg_inference_time']
                speedup = sklearn_time / data['avg_inference_time']
                print(f"  Speedup vs sklearn: {speedup:.1f}x")
            print()
        
    else:
        print("Insufficient models found for comparison.")
    
    print("A100-Optimized Model Comparison Completed!")


if __name__ == "__main__":
    main()
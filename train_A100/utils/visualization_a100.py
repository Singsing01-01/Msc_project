"""
A100-Optimized Visualization Tools for Image-to-Graph Training Results
Advanced visualization utilities optimized for A100 training results and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.animation import FuncAnimation
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Union
import json
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class A100Visualizer:
    """A100-optimized visualization toolkit for comprehensive result analysis."""
    
    def __init__(self, save_dir: str, dpi: int = 300):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
        
        # Color schemes
        self.model_colors = {
            'Model A (GNN)': '#1f77b4',
            'Model B (Similarity)': '#ff7f0e', 
            'sklearn_img_to_graph': '#2ca02c'
        }
        
        self.metric_colors = {
            'ARI': '#d62728',
            'NMI': '#9467bd',
            'Modularity': '#8c564b',
            'Inference Time': '#e377c2'
        }
    
    def visualize_training_progress(self, history_dict: Dict, model_name: str):
        """Visualize training progress with enhanced plots."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'{model_name} Training Progress - A100 Optimized', fontsize=18, fontweight='bold')
        
        epochs = range(1, len(history_dict['train_total_loss']) + 1)
        
        # Total loss with trend analysis
        ax = axes[0, 0]
        train_losses = history_dict['train_total_loss']
        ax.plot(epochs, train_losses, 'o-', label='Training', linewidth=2, markersize=4, color='#1f77b4')
        if 'val_total_loss' in history_dict:
            val_losses = history_dict['val_total_loss']
            ax.plot(epochs, val_losses, 's-', label='Validation', linewidth=2, markersize=4, color='#ff7f0e')
            
            # Add overfitting detection
            if len(val_losses) > 5:
                min_val_idx = np.argmin(val_losses)
                ax.axvline(x=min_val_idx+1, color='red', linestyle='--', alpha=0.7, 
                          label=f'Best Val (Epoch {min_val_idx+1})')
        
        # Add exponential moving average
        if len(train_losses) > 3:
            ema_alpha = 0.3
            ema = [train_losses[0]]
            for loss in train_losses[1:]:
                ema.append(ema_alpha * loss + (1 - ema_alpha) * ema[-1])
            ax.plot(epochs, ema, '--', alpha=0.7, label='EMA', color='green')
        
        ax.set_title('Total Loss with Trend Analysis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization
        
        # Coordinate loss
        ax = axes[0, 1]
        ax.plot(epochs, history_dict['train_coord_loss'], 'o-', label='Training', linewidth=2, markersize=4)
        if 'val_coord_loss' in history_dict:
            ax.plot(epochs, history_dict['val_coord_loss'], 's-', label='Validation', linewidth=2, markersize=4)
        ax.set_title('Coordinate Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Edge loss
        ax = axes[0, 2]
        ax.plot(epochs, history_dict['train_edge_loss'], 'o-', label='Training', linewidth=2, markersize=4)
        if 'val_edge_loss' in history_dict:
            ax.plot(epochs, history_dict['val_edge_loss'], 's-', label='Validation', linewidth=2, markersize=4)
        ax.set_title('Edge Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Count loss
        ax = axes[1, 0]
        ax.plot(epochs, history_dict['train_count_loss'], 'o-', label='Training', linewidth=2, markersize=4)
        if 'val_count_loss' in history_dict:
            ax.plot(epochs, history_dict['val_count_loss'], 's-', label='Validation', linewidth=2, markersize=4)
        ax.set_title('Count Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Model-specific losses
        if 'train_regularization_loss' in history_dict:  # Model A
            ax = axes[1, 1]
            ax.plot(epochs, history_dict['train_regularization_loss'], 'o-', label='Training', linewidth=2, markersize=4)
            if 'val_regularization_loss' in history_dict:
                ax.plot(epochs, history_dict['val_regularization_loss'], 's-', label='Validation', linewidth=2, markersize=4)
            ax.set_title('Regularization Loss', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        elif 'train_similarity_loss' in history_dict:  # Model B
            ax = axes[1, 1]
            ax.plot(epochs, history_dict['train_similarity_loss'], 'o-', label='Training', linewidth=2, markersize=4)
            if 'val_similarity_loss' in history_dict:
                ax.plot(epochs, history_dict['val_similarity_loss'], 's-', label='Validation', linewidth=2, markersize=4)
            ax.set_title('Similarity Loss', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        # Learning rate schedule
        if 'learning_rates' in history_dict:
            ax = axes[1, 2]
            ax.semilogy(epochs, history_dict['learning_rates'], 'o-', linewidth=2, markersize=4, color='purple')
            ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate (log scale)')
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 2].axis('off')
        
        # Loss components stacked area chart
        ax = axes[2, 0]
        loss_components = []
        labels = []
        
        if 'train_coord_loss' in history_dict:
            loss_components.append(history_dict['train_coord_loss'])
            labels.append('Coordinate')
        if 'train_edge_loss' in history_dict:
            loss_components.append(history_dict['train_edge_loss'])
            labels.append('Edge')
        if 'train_count_loss' in history_dict:
            loss_components.append(history_dict['train_count_loss'])
            labels.append('Count')
        
        if loss_components:
            ax.stackplot(epochs, *loss_components, labels=labels, alpha=0.7)
            ax.set_title('Loss Components (Stacked)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            axes[2, 0].axis('off')
        
        # Training vs Validation comparison
        ax = axes[2, 1]
        if 'val_total_loss' in history_dict:
            train_losses = history_dict['train_total_loss']
            val_losses = history_dict['val_total_loss']
            
            # Calculate validation gap
            val_gap = [v - t for v, t in zip(val_losses, train_losses[:len(val_losses)])]
            
            ax.plot(epochs[:len(val_gap)], val_gap, 'o-', linewidth=2, markersize=4, color='red')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.set_title('Validation Gap (Val - Train)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Difference')
            ax.grid(True, alpha=0.3)
            
            # Add overfitting warning
            if len(val_gap) > 5 and np.mean(val_gap[-3:]) > 0.1:
                ax.text(0.7, 0.9, 'Overfitting\nDetected!', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                       fontsize=10, fontweight='bold', color='white')
        else:
            axes[2, 1].axis('off')
        
        # Loss convergence analysis
        ax = axes[2, 2]
        if len(train_losses) > 10:
            # Calculate loss derivatives (rate of change)
            loss_derivatives = np.diff(train_losses)
            ax.plot(epochs[1:], loss_derivatives, 'o-', linewidth=2, markersize=3, color='orange')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.set_title('Loss Rate of Change', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('ΔLoss/ΔEpoch')
            ax.grid(True, alpha=0.3)
            
            # Add convergence indicator
            recent_changes = loss_derivatives[-5:] if len(loss_derivatives) >= 5 else loss_derivatives
            if np.all(np.abs(recent_changes) < 0.001):
                ax.text(0.7, 0.9, 'Converged', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7),
                       fontsize=10, fontweight='bold', color='white')
        else:
            axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{model_name.lower().replace(" ", "_")}_training_progress.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def visualize_model_predictions(self, model, test_sample: Dict, model_name: str, device: str = 'cuda'):
        """Visualize model predictions vs ground truth."""
        model.eval()
        
        # Prepare input
        image = torch.from_numpy(test_sample['image']).unsqueeze(0).unsqueeze(0).float().to(device)
        n_points = test_sample['n_points']
        
        # Create node mask
        node_mask = torch.zeros(1, 350, dtype=torch.bool).to(device)
        node_mask[0, :n_points] = True
        
        # Get predictions
        with torch.no_grad():
            predictions = model(image, node_mask)
        
        # Extract data
        pred_coords = predictions['predicted_coords'][0].cpu().numpy()[:n_points]
        pred_adjacency = predictions['adjacency_matrix'][0].cpu().numpy()[:n_points, :n_points]
        
        true_coords = test_sample['points'][:n_points]
        true_adjacency = test_sample['adjacency'][:n_points, :n_points]
        true_labels = test_sample['labels']
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name} Predictions vs Ground Truth', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(test_sample['image'], cmap='gray')
        axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # True coordinates
        axes[0, 1].scatter(true_coords[:, 0], true_coords[:, 1], c=true_labels, 
                          cmap='viridis', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        axes[0, 1].set_title('True Coordinates', fontsize=12, fontweight='bold')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predicted coordinates
        axes[0, 2].scatter(pred_coords[:, 0], pred_coords[:, 1], c=true_labels, 
                          cmap='viridis', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        axes[0, 2].set_title('Predicted Coordinates', fontsize=12, fontweight='bold')
        axes[0, 2].set_aspect('equal')
        axes[0, 2].grid(True, alpha=0.3)
        
        # True adjacency matrix
        im1 = axes[1, 0].imshow(true_adjacency, cmap='hot', interpolation='nearest')
        axes[1, 0].set_title('True Adjacency Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Predicted adjacency matrix
        im2 = axes[1, 1].imshow(pred_adjacency, cmap='hot', interpolation='nearest')
        axes[1, 1].set_title('Predicted Adjacency Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Error heatmap
        error_map = np.abs(true_adjacency - pred_adjacency)
        im3 = axes[1, 2].imshow(error_map, cmap='Reds', interpolation='nearest')
        axes[1, 2].set_title('Absolute Error Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # Add metrics
        coord_error = np.mean(np.linalg.norm(pred_coords - true_coords, axis=1))
        adj_error = np.mean(np.abs(pred_adjacency - true_adjacency))
        
        fig.text(0.02, 0.02, f'Coordinate MSE: {coord_error:.4f}\\nAdjacency MAE: {adj_error:.4f}', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{model_name.lower().replace(" ", "_")}_predictions.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def visualize_performance_comparison(self, results_dict: Dict):
        """Create comprehensive performance comparison visualizations."""
        
        # Performance radar chart
        self._create_radar_chart(results_dict)
        
        # Metric distributions
        self._create_metric_distributions(results_dict)
        
        # Performance vs efficiency scatter
        self._create_efficiency_scatter(results_dict)
        
        # Timeline comparison
        self._create_timeline_comparison(results_dict)
    
    def _create_radar_chart(self, results_dict: Dict):
        """Create radar chart for performance comparison."""
        metrics = ['avg_ari', 'avg_nmi', 'avg_modularity']
        metric_names = ['ARI', 'NMI', 'Modularity']
        
        # Normalize metrics to 0-1 scale
        normalized_data = {}
        for metric in metrics:
            values = [results_dict[method][metric] for method in results_dict.keys()]
            max_val = max(values)
            min_val = min(values)
            normalized_data[metric] = {method: (results_dict[method][metric] - min_val) / (max_val - min_val) 
                                     for method in results_dict.keys()}
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for method in results_dict.keys():
            values = [normalized_data[metric][method] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, 
                   color=self.model_colors.get(method, '#333333'))
            ax.fill(angles, values, alpha=0.25, color=self.model_colors.get(method, '#333333'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Comparison Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_radar_chart.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_metric_distributions(self, results_dict: Dict):
        """Create detailed metric distribution plots."""
        metrics = ['ari_scores', 'nmi_scores', 'modularity_scores', 'inference_times']
        metric_names = ['ARI Scores', 'NMI Scores', 'Modularity Scores', 'Inference Times (ms)']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Metric Distributions Comparison', fontsize=16, fontweight='bold')
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Create violin plots
            data_for_violin = []
            labels_for_violin = []
            
            for method in results_dict.keys():
                values = results_dict[method][metric]
                data_for_violin.extend(values)
                labels_for_violin.extend([method] * len(values))
            
            # Create DataFrame for seaborn
            df = pd.DataFrame({'Method': labels_for_violin, 'Value': data_for_violin})
            
            # Violin plot
            sns.violinplot(data=df, x='Method', y='Value', ax=ax)
            
            # Add box plot overlay
            sns.boxplot(data=df, x='Method', y='Value', ax=ax, width=0.3, 
                       boxprops=dict(alpha=0.7), showfliers=False)
            
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            if metric == 'inference_times':
                ax.set_yscale('log')
                ax.set_ylabel('Time (ms) - Log Scale')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metric_distributions_detailed.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_efficiency_scatter(self, results_dict: Dict):
        """Create performance vs efficiency scatter plot."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Performance vs Efficiency Analysis', fontsize=16, fontweight='bold')
        
        metrics = ['avg_ari', 'avg_nmi', 'avg_modularity']
        metric_names = ['ARI', 'NMI', 'Modularity']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            x_values = [results_dict[method]['avg_inference_time'] for method in results_dict.keys()]
            y_values = [results_dict[method][metric] for method in results_dict.keys()]
            
            # Create scatter plot
            for i, method in enumerate(results_dict.keys()):
                ax.scatter(x_values[i], y_values[i], s=150, alpha=0.7, 
                          color=self.model_colors.get(method, '#333333'), 
                          edgecolors='black', linewidth=1)
                
                # Add method labels
                ax.annotate(method, (x_values[i], y_values[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, alpha=0.8)
            
            ax.set_xlabel('Inference Time (ms)')
            ax.set_ylabel(name)
            ax.set_title(f'{name} vs Inference Time', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            
            # Add efficiency frontier
            if len(x_values) > 1:
                from scipy.spatial import ConvexHull
                points = np.column_stack([x_values, y_values])
                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            ax.plot(points[simplex, 0], points[simplex, 1], 'k--', alpha=0.3)
                    except:
                        pass  # Skip if convex hull fails
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_vs_efficiency.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_timeline_comparison(self, results_dict: Dict):
        """Create timeline comparison of different metrics."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(results_dict.keys())
        n_methods = len(methods)
        
        # Create timeline data
        timeline_data = []
        for i, method in enumerate(methods):
            timeline_data.append({
                'method': method,
                'ari': results_dict[method]['avg_ari'],
                'nmi': results_dict[method]['avg_nmi'],
                'modularity': results_dict[method]['avg_modularity'],
                'inference_time': results_dict[method]['avg_inference_time'],
                'y_pos': i
            })
        
        # Plot timeline
        for i, data in enumerate(timeline_data):
            # Performance bar
            performance_score = (data['ari'] + data['nmi'] + data['modularity']) / 3
            ax.barh(i, performance_score, height=0.6, alpha=0.7, 
                   color=self.model_colors.get(data['method'], '#333333'))
            
            # Add method name
            ax.text(-0.02, i, data['method'], ha='right', va='center', fontweight='bold')
            
            # Add performance score
            ax.text(performance_score + 0.01, i, f'{performance_score:.3f}', 
                   va='center', fontsize=10)
            
            # Add inference time annotation
            ax.text(performance_score + 0.1, i, f'{data["inference_time"]:.1f}ms', 
                   va='center', fontsize=9, alpha=0.7)
        
        ax.set_xlim(-0.3, 1.2)
        ax.set_ylim(-0.5, n_methods - 0.5)
        ax.set_xlabel('Average Performance Score (ARI + NMI + Modularity) / 3')
        ax.set_title('Method Performance Timeline', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_timeline.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_interactive_training_loss_dashboard(self, training_histories: Dict):
        """Create interactive training loss dashboard with advanced analysis."""
        if not training_histories:
            return
            
        # Create comprehensive interactive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Loss Evolution', 'Loss Components Breakdown', 
                           'Loss Rate of Change', 'Training vs Validation Comparison',
                           'Loss Smoothness Analysis', 'Convergence Detection'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = {'Model A (GNN)': '#1f77b4', 'Model B (Similarity)': '#ff7f0e'}
        
        for model_name, history in training_histories.items():
            if 'train_total_loss' not in history:
                continue
                
            epochs = list(range(1, len(history['train_total_loss']) + 1))
            color = colors.get(model_name, '#333333')
            
            # 1. Training Loss Evolution with EMA
            train_losses = history['train_total_loss']
            fig.add_trace(
                go.Scatter(x=epochs, y=train_losses, name=f'{model_name} Train Loss',
                         mode='lines+markers', line=dict(color=color)),
                row=1, col=1
            )
            
            # Add exponential moving average
            if len(train_losses) > 3:
                ema_alpha = 0.3
                ema = [train_losses[0]]
                for loss in train_losses[1:]:
                    ema.append(ema_alpha * loss + (1 - ema_alpha) * ema[-1])
                    
                fig.add_trace(
                    go.Scatter(x=epochs, y=ema, name=f'{model_name} EMA',
                             mode='lines', line=dict(color=color, dash='dash')),
                    row=1, col=1
                )
            
            # Add validation loss if available
            if 'val_total_loss' in history:
                val_losses = history['val_total_loss']
                fig.add_trace(
                    go.Scatter(x=epochs[:len(val_losses)], y=val_losses, 
                             name=f'{model_name} Val Loss',
                             mode='lines+markers', line=dict(color=color, dash='dot')),
                    row=1, col=1, secondary_y=True
                )
            
            # 2. Loss Components Breakdown
            if 'train_coord_loss' in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['train_coord_loss'], 
                             name=f'{model_name} Coord Loss', 
                             stackgroup=model_name, mode='lines'),
                    row=1, col=2
                )
            if 'train_edge_loss' in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['train_edge_loss'], 
                             name=f'{model_name} Edge Loss', 
                             stackgroup=model_name, mode='lines'),
                    row=1, col=2
                )
            if 'train_count_loss' in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['train_count_loss'], 
                             name=f'{model_name} Count Loss', 
                             stackgroup=model_name, mode='lines'),
                    row=1, col=2
                )
            
            # 3. Loss Rate of Change
            if len(train_losses) > 1:
                loss_derivatives = np.diff(train_losses)
                fig.add_trace(
                    go.Scatter(x=epochs[1:], y=loss_derivatives, 
                             name=f'{model_name} Loss Δ',
                             mode='lines+markers', line=dict(color=color)),
                    row=2, col=1
                )
                
                # Add convergence threshold
                fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5,
                             annotation_text="Convergence Line", row=2, col=1)
            
            # 4. Training vs Validation Comparison
            if 'val_total_loss' in history:
                val_losses = history['val_total_loss']
                val_gap = [v - t for v, t in zip(val_losses, train_losses[:len(val_losses)])]
                
                fig.add_trace(
                    go.Scatter(x=epochs[:len(val_gap)], y=val_gap, 
                             name=f'{model_name} Val Gap',
                             mode='lines+markers', line=dict(color=color)),
                    row=2, col=2
                )
                
                # Add overfitting warning line
                fig.add_hline(y=0.1, line_dash="dash", line_color="orange", opacity=0.7,
                             annotation_text="Overfitting Warning", row=2, col=2)
            
            # 5. Loss Smoothness Analysis (rolling variance)
            if len(train_losses) > 10:
                window_size = min(10, len(train_losses) // 3)
                rolling_var = []
                for i in range(window_size, len(train_losses)):
                    window_data = train_losses[i-window_size:i]
                    rolling_var.append(np.var(window_data))
                
                fig.add_trace(
                    go.Scatter(x=epochs[window_size:], y=rolling_var, 
                             name=f'{model_name} Loss Variance',
                             mode='lines', line=dict(color=color)),
                    row=3, col=1
                )
            
            # 6. Convergence Detection
            if len(train_losses) > 5:
                # Calculate moving average of absolute changes
                window_size = 5
                convergence_metric = []
                for i in range(window_size, len(train_losses)):
                    recent_changes = [abs(train_losses[j] - train_losses[j-1]) 
                                    for j in range(i-window_size+1, i+1)]
                    convergence_metric.append(np.mean(recent_changes))
                
                fig.add_trace(
                    go.Scatter(x=epochs[window_size:], y=convergence_metric, 
                             name=f'{model_name} Convergence Metric',
                             mode='lines', line=dict(color=color)),
                    row=3, col=2
                )
                
                # Add convergence threshold
                fig.add_hline(y=0.001, line_dash="dash", line_color="green", opacity=0.7,
                             annotation_text="Converged", row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Advanced Training Loss Analysis Dashboard",
            showlegend=True,
            height=1200,
            hovermode='x unified'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Loss Components", row=1, col=2)
        fig.update_yaxes(title_text="Loss Change Rate", row=2, col=1)
        fig.update_yaxes(title_text="Validation Gap", row=2, col=2)
        fig.update_yaxes(title_text="Loss Variance", row=3, col=1)
        fig.update_yaxes(title_text="Convergence Metric", row=3, col=2)
        
        # Update x-axes
        for row in range(1, 4):
            for col in range(1, 3):
                fig.update_xaxes(title_text="Epoch", row=row, col=col)
        
        # Save interactive loss dashboard
        pyo.plot(fig, filename=os.path.join(self.save_dir, 'interactive_loss_dashboard.html'), 
                auto_open=False)
        
        print(f"Interactive training loss dashboard saved to {os.path.join(self.save_dir, 'interactive_loss_dashboard.html')}")
        
    def create_interactive_dashboard(self, results_dict: Dict, training_histories: Dict = None):
        """Create interactive Plotly dashboard."""
        
        # Performance comparison
        methods = list(results_dict.keys())
        metrics = ['avg_ari', 'avg_nmi', 'avg_modularity', 'avg_inference_time']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Inference Time Comparison', 
                           'Metric Distributions', 'Training Progress'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Performance metrics bar chart
        for i, metric in enumerate(['avg_ari', 'avg_nmi', 'avg_modularity']):
            values = [results_dict[method][metric] for method in methods]
            fig.add_trace(
                go.Bar(name=metric.replace('avg_', '').upper(), x=methods, y=values),
                row=1, col=1
            )
        
        # Inference time comparison
        inference_times = [results_dict[method]['avg_inference_time'] for method in methods]
        fig.add_trace(
            go.Bar(x=methods, y=inference_times, name='Inference Time'),
            row=1, col=2
        )
        
        # Metric distributions (box plots)
        for method in methods:
            fig.add_trace(
                go.Box(y=results_dict[method]['ari_scores'], name=f'{method} ARI'),
                row=2, col=1
            )
        
        # Training progress (if available)
        if training_histories:
            for model_name, history in training_histories.items():
                if 'train_total_loss' in history:
                    epochs = list(range(1, len(history['train_total_loss']) + 1))
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['train_total_loss'], 
                                 name=f'{model_name} Loss', mode='lines+markers'),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title_text="A100-Optimized Image-to-Graph Model Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save interactive plot
        pyo.plot(fig, filename=os.path.join(self.save_dir, 'interactive_dashboard.html'), 
                auto_open=False)
        
        # Create dedicated training loss dashboard
        if training_histories:
            self.create_interactive_training_loss_dashboard(training_histories)
        
        print(f"Interactive dashboard saved to {os.path.join(self.save_dir, 'interactive_dashboard.html')}")
    
    def create_model_architecture_comparison(self, model_info_dict: Dict):
        """Create model architecture comparison visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold')
        
        # Parameter comparison
        ax = axes[0]
        models = list(model_info_dict.keys())
        params = [info['parameters'] / 1e6 for info in model_info_dict.values()]  # Convert to millions
        
        bars = ax.bar(models, params, alpha=0.7, color=[self.model_colors.get(m, '#333333') for m in models])
        ax.set_ylabel('Parameters (Millions)')
        ax.set_title('Model Size Comparison', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{param:.2f}M', ha='center', va='bottom', fontsize=11)
        
        # Architecture details
        ax = axes[1]
        ax.axis('off')
        
        y_pos = 0.9
        for model, info in model_info_dict.items():
            ax.text(0.05, y_pos, f"{model}:", fontsize=14, fontweight='bold', 
                   color=self.model_colors.get(model, '#333333'))
            y_pos -= 0.1
            
            ax.text(0.1, y_pos, f"Parameters: {info['parameters']:,}", fontsize=10)
            y_pos -= 0.07
            
            if 'optimizations' in info:
                ax.text(0.1, y_pos, f"Optimizations: {', '.join(info['optimizations'])}", fontsize=10)
                y_pos -= 0.07
            
            y_pos -= 0.05
        
        ax.set_title('Architecture Details', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'model_architecture_comparison.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, results_dict: Dict, training_histories: Dict = None, 
                                    model_info_dict: Dict = None):
        """Generate comprehensive visualization report with enhanced training loss analysis."""
        print("Generating comprehensive visualization report...")
        
        # Performance comparison
        self.visualize_performance_comparison(results_dict)
        
        # Enhanced training progress for each model
        if training_histories:
            print("Creating enhanced training loss visualizations...")
            for model_name, history in training_histories.items():
                self.visualize_training_progress(history, model_name)
            
            # Create dedicated interactive training loss dashboard
            self.create_interactive_training_loss_dashboard(training_histories)
        
        # Model architecture comparison
        if model_info_dict:
            self.create_model_architecture_comparison(model_info_dict)
        
        # Interactive dashboard
        self.create_interactive_dashboard(results_dict, training_histories)
        
        # Create summary plot
        self._create_summary_visualization(results_dict)
        
        print(f"Comprehensive visualization report with enhanced training loss analysis saved to {self.save_dir}")
        print("Generated visualizations include:")
        print("  • Static training progress plots with 9 detailed loss analysis panels")
        print("  • Interactive training loss dashboard with 6 advanced analysis views")
        print("  • Performance comparison charts and efficiency analysis")
        print("  • Executive summary with key findings")
        if training_histories:
            print("  • Training loss features: EMA, convergence detection, overfitting warnings, loss variance analysis")
    
    def _create_summary_visualization(self, results_dict: Dict):
        """Create final summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('A100-Optimized Image-to-Graph Models - Executive Summary', 
                    fontsize=16, fontweight='bold')
        
        methods = list(results_dict.keys())
        
        # 1. Overall performance scores
        performance_scores = []
        for method in methods:
            score = (results_dict[method]['avg_ari'] + 
                    results_dict[method]['avg_nmi'] + 
                    results_dict[method]['avg_modularity']) / 3
            performance_scores.append(score)
        
        bars1 = ax1.bar(methods, performance_scores, alpha=0.8, 
                       color=[self.model_colors.get(m, '#333333') for m in methods])
        ax1.set_title('Overall Performance Score', fontweight='bold')
        ax1.set_ylabel('Average Score (ARI + NMI + Modularity) / 3')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars1, performance_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Speed comparison
        inference_times = [results_dict[method]['avg_inference_time'] for method in methods]
        bars2 = ax2.bar(methods, inference_times, alpha=0.8,
                       color=[self.model_colors.get(m, '#333333') for m in methods])
        ax2.set_title('Inference Speed Comparison', fontweight='bold')
        ax2.set_ylabel('Time (ms)')
        ax2.set_yscale('log')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add speedup annotations
        baseline_time = max(inference_times)
        for bar, time in zip(bars2, inference_times):
            speedup = baseline_time / time
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}ms\\n{speedup:.1f}x' if speedup > 1 else f'{time:.1f}ms',
                    ha='center', va='bottom', fontsize=9)
        
        # 3. Target achievement
        target_ari = 0.80
        ari_scores = [results_dict[method]['avg_ari'] for method in methods]
        colors = ['green' if score >= target_ari else 'red' for score in ari_scores]
        
        bars3 = ax3.bar(methods, ari_scores, alpha=0.8, color=colors)
        ax3.axhline(y=target_ari, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_title('Target Achievement (ARI ≥ 0.80)', fontweight='bold')
        ax3.set_ylabel('ARI Score')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars3, ari_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Efficiency frontier
        for i, method in enumerate(methods):
            ax4.scatter(inference_times[i], performance_scores[i], s=200, alpha=0.8,
                       color=self.model_colors.get(method, '#333333'), 
                       edgecolors='black', linewidth=2)
            ax4.annotate(method, (inference_times[i], performance_scores[i]),
                        xytext=(10, 10), textcoords='offset points', fontsize=10)
        
        ax4.set_xlabel('Inference Time (ms)')
        ax4.set_ylabel('Performance Score')
        ax4.set_title('Performance vs Speed Trade-off', fontweight='bold')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'executive_summary.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()


def main():
    """Main visualization function for testing."""
    # Create test visualizer
    visualizer = A100Visualizer("/Users/jeremyfang/Downloads/image_to_graph/train_A100/results/visualizations")
    
    # Test with sample data
    sample_results = {
        'Model A (GNN)': {
            'avg_ari': 0.847, 'avg_nmi': 0.792, 'avg_modularity': 0.673,
            'avg_inference_time': 23.0,
            'ari_scores': np.random.normal(0.847, 0.05, 50),
            'nmi_scores': np.random.normal(0.792, 0.04, 50),
            'modularity_scores': np.random.normal(0.673, 0.06, 50),
            'inference_times': np.random.normal(23.0, 3.0, 50)
        },
        'Model B (Similarity)': {
            'avg_ari': 0.834, 'avg_nmi': 0.781, 'avg_modularity': 0.651,
            'avg_inference_time': 5.8,
            'ari_scores': np.random.normal(0.834, 0.04, 50),
            'nmi_scores': np.random.normal(0.781, 0.03, 50),
            'modularity_scores': np.random.normal(0.651, 0.05, 50),
            'inference_times': np.random.normal(5.8, 1.2, 50)
        },
        'sklearn_img_to_graph': {
            'avg_ari': 0.342, 'avg_nmi': 0.287, 'avg_modularity': 0.124,
            'avg_inference_time': 4200.0,
            'ari_scores': np.random.normal(0.342, 0.1, 50),
            'nmi_scores': np.random.normal(0.287, 0.08, 50),
            'modularity_scores': np.random.normal(0.124, 0.03, 50),
            'inference_times': np.random.normal(4200.0, 500.0, 50)
        }
    }
    
    sample_model_info = {
        'Model A (GNN)': {
            'parameters': 4575614,
            'optimizations': ['A100_TensorCores', 'Vectorized_Operations', 'Enhanced_Architecture']
        },
        'Model B (Similarity)': {
            'parameters': 1159998,
            'optimizations': ['A100_TensorCores', 'Vectorized_Similarity', 'Enhanced_Architecture']
        }
    }
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report(sample_results, model_info_dict=sample_model_info)
    
    print("Visualization testing completed!")


if __name__ == "__main__":
    main()
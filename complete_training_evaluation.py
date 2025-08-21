import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction import image as sklearn_image
import networkx as nx
from typing import Dict, List, Tuple

from model_a_gnn import ModelA_GNN, ModelALoss
from model_b_similarity import ModelB_Similarity, ModelBLoss  
from data_generation import SyntheticDataGenerator
from data_augmentation import create_data_loaders

class CompleteTrainingEvaluation:
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        
    def create_datasets(self, train_size: int = 100, test_size: int = 50):
        """Create training and testing datasets"""
        print("Creating datasets...")
        generator = SyntheticDataGenerator(img_size=64, n_samples=300, noise=0.1)
        
        # Generate concentric circles for training, double moons for testing  
        train_data, test_data = generator.create_train_test_split(
            train_size=train_size, test_size=test_size
        )
        
        train_loader, test_loader = create_data_loaders(
            train_data, test_data, batch_size=4, use_augmentation=True
        )
        
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        return train_loader, test_loader, test_data
    
    def train_model(self, model, model_name: str, train_loader, test_loader, 
                   epochs: int = 50, lr: float = 0.001):
        """Train a single model with proper optimization"""
        print(f"\nTraining {model_name}...")
        
        # Initialize loss function and optimizer
        if 'ModelA' in model_name:
            criterion = ModelALoss(coord_weight=1.0, edge_weight=1.0, count_weight=0.1)
        else:
            criterion = ModelBLoss(coord_weight=1.0, edge_weight=1.0, count_weight=0.1)
            
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        model.train()
        train_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in train_loader:
                images = batch['images'].to(self.device)
                targets = {
                    'points': batch['points'].to(self.device),
                    'adjacency': batch['adjacency'].to(self.device),
                    'node_masks': batch['node_masks'].to(self.device)
                }
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(images, targets['node_masks'])
                
                # Compute loss
                loss_dict = criterion(predictions, targets)
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'{model_name}_best.pth')
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(f'{model_name}_best.pth'))
        print(f"{model_name} training completed. Best loss: {best_loss:.4f}")
        
        return train_losses
    
    def evaluate_sklearn_baseline(self, test_data: List) -> Dict:
        """Evaluate sklearn img_to_graph baseline"""
        print("\nEvaluating sklearn baseline...")
        
        results = {'ari': [], 'nmi': [], 'modularity': [], 'inference_time': []}
        
        for sample in test_data[:20]:  # Evaluate on subset for speed
            start_time = time.time()
            
            try:
                # Use sklearn img_to_graph  
                graph = sklearn_image.img_to_graph(sample['image'])
                
                # Spectral clustering
                true_labels = sample['labels']
                valid_mask = true_labels >= 0
                true_labels = true_labels[valid_mask]
                
                n_clusters = len(np.unique(true_labels))
                if n_clusters < 2:
                    n_clusters = 2
                    
                clustering = SpectralClustering(
                    n_clusters=n_clusters, 
                    eigen_solver='arpack',
                    random_state=42,
                    n_init=10
                )
                predicted_labels = clustering.fit_predict(graph.toarray())
                
                # Ensure same length
                if len(predicted_labels) != len(true_labels):
                    min_len = min(len(predicted_labels), len(true_labels))
                    predicted_labels = predicted_labels[:min_len]
                    true_labels = true_labels[:min_len]
                
                # Compute metrics
                ari = adjusted_rand_score(true_labels, predicted_labels)
                nmi = normalized_mutual_info_score(true_labels, predicted_labels)
                
                # Modularity
                try:
                    G = nx.from_scipy_sparse_array(graph)
                    communities = [set(np.where(predicted_labels == i)[0]) for i in range(n_clusters)]
                    modularity = nx.algorithms.community.modularity(G, communities)
                except:
                    modularity = 0.0
                
                inference_time = time.time() - start_time
                
                results['ari'].append(ari)
                results['nmi'].append(nmi)
                results['modularity'].append(modularity)
                results['inference_time'].append(inference_time)
                
            except Exception as e:
                print(f"Error in sklearn evaluation: {e}")
                continue
        
        # Aggregate results
        return {
            'mean_ari': np.mean(results['ari']),
            'std_ari': np.std(results['ari']),
            'mean_nmi': np.mean(results['nmi']),
            'std_nmi': np.std(results['nmi']),
            'mean_modularity': np.mean(results['modularity']),
            'std_modularity': np.std(results['modularity']),
            'mean_inference_time': np.mean(results['inference_time']),
            'std_inference_time': np.std(results['inference_time']),
            'num_samples': len(results['ari'])
        }
    
    def evaluate_model(self, model, model_name: str, test_data: List) -> Dict:
        """Evaluate a trained model"""
        print(f"\nEvaluating {model_name}...")
        
        model.eval()
        results = {'ari': [], 'nmi': [], 'modularity': [], 'inference_time': []}
        
        with torch.no_grad():
            for sample in test_data[:20]:  # Evaluate on subset
                start_time = time.time()
                
                try:
                    # Prepare input
                    image = torch.FloatTensor(sample['image']).unsqueeze(0).unsqueeze(0).to(self.device)
                    true_labels = sample['labels']
                    valid_mask = true_labels >= 0
                    true_labels = true_labels[valid_mask]
                    
                    # Create node mask
                    n_nodes = len(sample['points'])
                    node_mask = torch.zeros(1, 350, dtype=torch.bool).to(self.device)
                    node_mask[0, :n_nodes] = True
                    
                    # Model prediction
                    predictions = model(image, node_mask)
                    
                    # Extract results
                    pred_coords = predictions['predicted_coords'][0][:n_nodes].cpu().numpy()
                    adjacency = predictions['adjacency_matrix'][0][:n_nodes, :n_nodes].cpu().numpy()
                    
                    # Binarize adjacency matrix
                    adj_binary = (adjacency > 0.5).astype(int)
                    
                    # Spectral clustering
                    n_clusters = len(np.unique(true_labels))
                    if n_clusters < 2:
                        n_clusters = 2
                        
                    clustering = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity='precomputed',
                        random_state=42,
                        n_init=10
                    )
                    predicted_labels = clustering.fit_predict(adj_binary)
                    
                    # Compute metrics
                    ari = adjusted_rand_score(true_labels, predicted_labels)
                    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
                    
                    # Modularity
                    try:
                        G = nx.from_numpy_array(adj_binary)
                        communities = [set(np.where(predicted_labels == i)[0]) for i in range(n_clusters)]
                        modularity = nx.algorithms.community.modularity(G, communities)
                    except:
                        modularity = 0.0
                    
                    inference_time = time.time() - start_time
                    
                    results['ari'].append(ari)
                    results['nmi'].append(nmi)
                    results['modularity'].append(modularity)
                    results['inference_time'].append(inference_time)
                    
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    continue
        
        # Aggregate results
        return {
            'mean_ari': np.mean(results['ari']),
            'std_ari': np.std(results['ari']),
            'mean_nmi': np.mean(results['nmi']),
            'std_nmi': np.std(results['nmi']),
            'mean_modularity': np.mean(results['modularity']),
            'std_modularity': np.std(results['modularity']),
            'mean_inference_time': np.mean(results['inference_time']),
            'std_inference_time': np.std(results['inference_time']),
            'num_samples': len(results['ari'])
        }
    
    def run_complete_evaluation(self):
        """Run complete training and evaluation pipeline"""
        print("="*60)
        print("Complete Model Training and Evaluation")
        print("="*60)
        
        # 1. Create datasets
        train_loader, test_loader, test_data = self.create_datasets(train_size=40, test_size=20)
        
        # 2. Initialize models
        model_a = ModelA_GNN().to(self.device)
        model_b = ModelB_Similarity().to(self.device)
        
        print(f"\nModel A parameters: {model_a.count_parameters():,}")
        print(f"Model B parameters: {model_b.count_parameters():,}")
        
        # Verify parameter constraints
        if model_a.count_parameters() > 1_500_000:
            raise ValueError(f"Model A has {model_a.count_parameters():,} parameters, exceeds 1.5M limit!")
        if model_b.count_parameters() > 1_500_000:
            raise ValueError(f"Model B has {model_b.count_parameters():,} parameters, exceeds 1.5M limit!")
        
        # 3. Train models
        print("\n" + "="*40)
        print("TRAINING PHASE")
        print("="*40)
        
        train_losses_a = self.train_model(model_a, "ModelA_GNN", train_loader, test_loader, epochs=20)
        train_losses_b = self.train_model(model_b, "ModelB_Similarity", train_loader, test_loader, epochs=20)
        
        # 4. Evaluation phase
        print("\n" + "="*40)
        print("EVALUATION PHASE")
        print("="*40)
        
        # Evaluate sklearn baseline
        sklearn_results = self.evaluate_sklearn_baseline(test_data)
        
        # Evaluate trained models
        model_a_results = self.evaluate_model(model_a, "ModelA_GNN", test_data)
        model_b_results = self.evaluate_model(model_b, "ModelB_Similarity", test_data)
        
        # 5. Results analysis
        results = {
            'sklearn_baseline': sklearn_results,
            'ModelA_GNN': model_a_results,
            'ModelB_Similarity': model_b_results
        }
        
        self.analyze_results(results)
        self.save_results(results)
        
        return results
    
    def analyze_results(self, results: Dict):
        """Analyze and display results"""
        print("\n" + "="*60)
        print("RESULTS ANALYSIS")
        print("="*60)
        
        # Display results table
        print(f"\n{'Method':<20} {'ARI':<8} {'NMI':<8} {'Modularity':<12} {'Time(ms)':<10}")
        print("-" * 60)
        
        for method_name, metrics in results.items():
            print(f"{method_name:<20} {metrics['mean_ari']:<8.4f} {metrics['mean_nmi']:<8.4f} "
                  f"{metrics['mean_modularity']:<12.4f} {metrics['mean_inference_time']*1000:<10.2f}")
        
        # Check requirements
        print(f"\n{'='*60}")
        print("REQUIREMENT VERIFICATION")
        print("="*60)
        
        target_ari = 0.80
        sklearn_ari = results['sklearn_baseline']['mean_ari']
        model_a_ari = results['ModelA_GNN']['mean_ari']
        model_b_ari = results['ModelB_Similarity']['mean_ari']
        
        print(f"\n1. ARI >= 0.80 Target:")
        print(f"   ModelA ARI: {model_a_ari:.4f} ({'PASS' if model_a_ari >= target_ari else 'FAIL'})")
        print(f"   ModelB ARI: {model_b_ari:.4f} ({'PASS' if model_b_ari >= target_ari else 'FAIL'})")
        
        print(f"\n2. Outperform sklearn baseline:")
        print(f"   sklearn ARI: {sklearn_ari:.4f}")
        print(f"   ModelA vs sklearn: {'PASS' if model_a_ari > sklearn_ari else 'FAIL'}")
        print(f"   ModelB vs sklearn: {'PASS' if model_b_ari > sklearn_ari else 'FAIL'}")
        
        print(f"\n3. Parameter constraints (<1.5M):")
        model_a = ModelA_GNN()
        model_b = ModelB_Similarity()
        print(f"   ModelA: {model_a.count_parameters():,} ({'PASS' if model_a.count_parameters() < 1_500_000 else 'FAIL'})")
        print(f"   ModelB: {model_b.count_parameters():,} ({'PASS' if model_b.count_parameters() < 1_500_000 else 'FAIL'})")
        
        # Overall assessment
        a_passes_ari = model_a_ari >= target_ari
        b_passes_ari = model_b_ari >= target_ari
        a_beats_sklearn = model_a_ari > sklearn_ari
        b_beats_sklearn = model_b_ari > sklearn_ari
        
        print(f"\n{'='*60}")
        print("OVERALL ASSESSMENT")
        print("="*60)
        
        if a_passes_ari and b_passes_ari and a_beats_sklearn and b_beats_sklearn:
            print("SUCCESS: All requirements met!")
        else:
            print("REQUIREMENTS NOT FULLY MET:")
            if not a_passes_ari:
                print(f"- ModelA ARI {model_a_ari:.4f} < 0.80")
            if not b_passes_ari:
                print(f"- ModelB ARI {model_b_ari:.4f} < 0.80")
            if not a_beats_sklearn:
                print(f"- ModelA does not beat sklearn baseline")
            if not b_beats_sklearn:
                print(f"- ModelB does not beat sklearn baseline")
    
    def save_results(self, results: Dict):
        """Save results to files"""
        # Create results DataFrame
        df = pd.DataFrame({
            'Method': list(results.keys()),
            'ARI': [results[k]['mean_ari'] for k in results.keys()],
            'ARI_std': [results[k]['std_ari'] for k in results.keys()],
            'NMI': [results[k]['mean_nmi'] for k in results.keys()],
            'NMI_std': [results[k]['std_nmi'] for k in results.keys()],
            'Modularity': [results[k]['mean_modularity'] for k in results.keys()],
            'Modularity_std': [results[k]['std_modularity'] for k in results.keys()],
            'Inference_Time_ms': [results[k]['mean_inference_time']*1000 for k in results.keys()],
            'Samples': [results[k]['num_samples'] for k in results.keys()]
        })
        
        # Sort by ARI
        df = df.sort_values('ARI', ascending=False)
        
        # Save to CSV
        df.to_csv('trained_model_results.csv', index=False)
        print(f"\nResults saved to: trained_model_results.csv")
        
        # Create visualization
        self.create_results_visualization(df)
    
    def create_results_visualization(self, df):
        """Create results visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = df['Method'].values
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # ARI comparison
        bars1 = axes[0, 0].bar(methods, df['ARI'], color=colors, alpha=0.7)
        axes[0, 0].axhline(y=0.80, color='red', linestyle='--', label='Target ARI >= 0.80')
        axes[0, 0].set_title('Adjusted Rand Index (ARI)')
        axes[0, 0].set_ylabel('ARI Score')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars1, df['ARI']):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom')
        
        # NMI comparison
        bars2 = axes[0, 1].bar(methods, df['NMI'], color=colors, alpha=0.7)
        axes[0, 1].set_title('Normalized Mutual Information (NMI)')
        axes[0, 1].set_ylabel('NMI Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars2, df['NMI']):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom')
        
        # Modularity comparison
        bars3 = axes[1, 0].bar(methods, df['Modularity'], color=colors, alpha=0.7)
        axes[1, 0].set_title('Modularity')
        axes[1, 0].set_ylabel('Modularity Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars3, df['Modularity']):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom')
        
        # Inference time comparison
        bars4 = axes[1, 1].bar(methods, df['Inference_Time_ms'], color=colors, alpha=0.7)
        axes[1, 1].set_title('Inference Time')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for bar, time in zip(bars4, df['Inference_Time_ms']):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                           f'{time:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('trained_model_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: trained_model_comparison.png")

def main():
    evaluator = CompleteTrainingEvaluation()
    results = evaluator.run_complete_evaluation()
    return results

if __name__ == "__main__":
    results = main()
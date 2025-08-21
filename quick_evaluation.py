import torch
import numpy as np
import pandas as pd
import time
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction import image as sklearn_image
import networkx as nx

from model_a_gnn import ModelA_GNN
from model_b_similarity import ModelB_Similarity
from data_generation import SyntheticDataGenerator

def quick_evaluation():
    """Run quick evaluation without full training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Generate small test dataset
    generator = SyntheticDataGenerator(img_size=64, n_samples=300, noise=0.1)
    _, test_data = generator.create_train_test_split(train_size=5, test_size=10)
    
    # Initialize models
    model_a = ModelA_GNN().to(device)
    model_b = ModelB_Similarity().to(device)
    
    print(f"Model A parameters: {model_a.count_parameters():,}")
    print(f"Model B parameters: {model_b.count_parameters():,}")
    
    results = {}
    
    # Evaluate sklearn baseline
    print("\nEvaluating sklearn baseline...")
    sklearn_results = evaluate_sklearn_baseline(test_data)
    results['sklearn_baseline'] = sklearn_results
    
    # Evaluate untrained models (random initialization)
    print("\nEvaluating untrained Model A...")
    model_a_results = evaluate_untrained_model(model_a, "ModelA_GNN", test_data, device)
    results['ModelA_GNN_untrained'] = model_a_results
    
    print("\nEvaluating untrained Model B...")
    model_b_results = evaluate_untrained_model(model_b, "ModelB_Similarity", test_data, device)
    results['ModelB_Similarity_untrained'] = model_b_results
    
    # Create results DataFrame
    df = pd.DataFrame({
        'Method': list(results.keys()),
        'ARI': [results[k]['mean_ari'] for k in results.keys()],
        'NMI': [results[k]['mean_nmi'] for k in results.keys()],
        'Modularity': [results[k]['mean_modularity'] for k in results.keys()],
        'Inference_Time_ms': [results[k]['mean_inference_time']*1000 for k in results.keys()],
        'Meets_ARI_Target': [results[k]['mean_ari'] >= 0.80 for k in results.keys()],
        'Beats_Sklearn': [results[k]['mean_ari'] > results['sklearn_baseline']['mean_ari'] for k in results.keys()],
        'Description': [
            'sklearn img_to_graph + spectral clustering',
            'GNN-based end-to-end architecture (untrained)',
            'Similarity-based lightweight architecture (untrained)'
        ]
    })
    
    # Save results
    df.to_csv('real_model_results_untrained.csv', index=False, encoding='utf-8-sig')
    
    # Display results
    print("\n" + "="*60)
    print("REAL EVALUATION RESULTS (UNTRAINED MODELS)")
    print("="*60)
    print(df.to_string(index=False))
    
    print(f"\nResults saved to: real_model_results_untrained.csv")
    
    return results

def evaluate_sklearn_baseline(test_data):
    """Evaluate sklearn baseline"""
    results = {'ari': [], 'nmi': [], 'modularity': [], 'inference_time': []}
    
    for sample in test_data:
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
                n_init=3
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
    
    return {
        'mean_ari': np.mean(results['ari']),
        'mean_nmi': np.mean(results['nmi']),
        'mean_modularity': np.mean(results['modularity']),
        'mean_inference_time': np.mean(results['inference_time']),
        'num_samples': len(results['ari'])
    }

def evaluate_untrained_model(model, model_name, test_data, device):
    """Evaluate untrained model (random initialization)"""
    model.eval()
    results = {'ari': [], 'nmi': [], 'modularity': [], 'inference_time': []}
    
    with torch.no_grad():
        for sample in test_data:
            start_time = time.time()
            
            try:
                # Prepare input
                image = torch.FloatTensor(sample['image']).unsqueeze(0).unsqueeze(0).to(device)
                true_labels = sample['labels']
                valid_mask = true_labels >= 0
                true_labels = true_labels[valid_mask]
                
                # Create node mask
                n_nodes = len(sample['points'])
                node_mask = torch.zeros(1, 350, dtype=torch.bool).to(device)
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
                    n_init=3
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
    
    return {
        'mean_ari': np.mean(results['ari']),
        'mean_nmi': np.mean(results['nmi']),
        'mean_modularity': np.mean(results['modularity']),
        'mean_inference_time': np.mean(results['inference_time']),
        'num_samples': len(results['ari'])
    }

if __name__ == "__main__":
    results = quick_evaluation()
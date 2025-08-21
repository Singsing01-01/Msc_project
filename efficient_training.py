import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction import image as sklearn_image
import time

from model_a_gnn import ModelA_GNN, ModelALoss
from model_b_similarity import ModelB_Similarity, ModelBLoss
from data_generation import SyntheticDataGenerator

def create_simple_data():
    """Create simple synthetic data for quick training"""
    generator = SyntheticDataGenerator(img_size=64, n_samples=300, noise=0.1)
    
    # Generate simple training data - just a few samples
    train_samples = []
    for i in range(10):
        points, labels, adjacency = generator.generate_circles_data(n_samples=300)
        image = generator.points_to_image(points)
        # Create pixel coordinates directly
        points_pixel = (points - points.min()) / (points.max() - points.min()) * 63
        
        train_samples.append({
            'image': image,
            'points': points,
            'points_pixel': points_pixel.astype(np.float32),
            'labels': labels,
            'adjacency': adjacency
        })
    
    # Generate test data with moons
    test_samples = []
    for i in range(5):
        points, labels, adjacency = generator.generate_moons_data(n_samples=300)
        image = generator.points_to_image(points)
        # Create pixel coordinates directly
        points_pixel = (points - points.min()) / (points.max() - points.min()) * 63
        
        test_samples.append({
            'image': image,
            'points': points,
            'points_pixel': points_pixel.astype(np.float32),
            'labels': labels,
            'adjacency': adjacency
        })
    
    return train_samples, test_samples

def train_model_simple(model, model_name, train_samples, epochs=50):
    """Simple training loop"""
    print(f"\nTraining {model_name} for {epochs} epochs...")
    
    if 'ModelA' in model_name:
        criterion = ModelALoss(coord_weight=1.0, edge_weight=1.0, count_weight=0.1)
    else:
        criterion = ModelBLoss(coord_weight=1.0, edge_weight=1.0, count_weight=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    model.train()
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        total_loss = 0
        
        for sample in train_samples:
            # Prepare batch data
            image = torch.FloatTensor(sample['image']).unsqueeze(0).unsqueeze(0).to(device)
            n_nodes = len(sample['points'])
            
            # Create targets
            targets = {
                'points': torch.FloatTensor(sample['points']).unsqueeze(0).to(device),
                'adjacency': torch.FloatTensor(sample['adjacency']).unsqueeze(0).to(device),
                'node_masks': torch.zeros(1, 350, dtype=torch.bool).to(device)
            }
            targets['node_masks'][0, :n_nodes] = True
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(image, targets['node_masks'])
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_samples)
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    
    print(f"{model_name} training completed!")

def evaluate_sklearn_simple(test_samples):
    """Simple sklearn baseline evaluation"""
    ari_scores = []
    
    for sample in test_samples:
        try:
            # Use sklearn img_to_graph
            graph = sklearn_image.img_to_graph(sample['image'])
            
            # Spectral clustering
            true_labels = sample['labels']
            n_clusters = len(np.unique(true_labels))
            if n_clusters < 2:
                n_clusters = 2
            
            clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', 
                                          random_state=42, n_init=10)
            predicted_labels = clustering.fit_predict(graph.toarray())
            
            # Compute ARI
            ari = adjusted_rand_score(true_labels, predicted_labels)
            ari_scores.append(ari)
            
        except Exception as e:
            print(f"Error in sklearn evaluation: {e}")
            ari_scores.append(0.0)
    
    return np.mean(ari_scores)

def evaluate_model_simple(model, test_samples):
    """Simple model evaluation"""
    model.eval()
    device = next(model.parameters()).device
    ari_scores = []
    
    with torch.no_grad():
        for sample in test_samples:
            try:
                # Prepare input
                image = torch.FloatTensor(sample['image']).unsqueeze(0).unsqueeze(0).to(device)
                n_nodes = len(sample['points'])
                node_mask = torch.zeros(1, 350, dtype=torch.bool).to(device)
                node_mask[0, :n_nodes] = True
                
                # Model prediction
                predictions = model(image, node_mask)
                
                # Extract adjacency matrix
                adjacency = predictions['adjacency_matrix'][0][:n_nodes, :n_nodes].cpu().numpy()
                adj_binary = (adjacency > 0.5).astype(int)
                
                # Spectral clustering
                true_labels = sample['labels']
                n_clusters = len(np.unique(true_labels))
                if n_clusters < 2:
                    n_clusters = 2
                
                clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                              random_state=42, n_init=10)
                predicted_labels = clustering.fit_predict(adj_binary)
                
                # Compute ARI
                ari = adjusted_rand_score(true_labels, predicted_labels)
                ari_scores.append(ari)
                
            except Exception as e:
                print(f"Error in model evaluation: {e}")
                ari_scores.append(0.0)
    
    return np.mean(ari_scores)

def main():
    """Main training and evaluation"""
    print("="*60)
    print("Efficient Model Training and Evaluation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data
    print("Creating datasets...")
    train_samples, test_samples = create_simple_data()
    
    # Initialize models
    model_a = ModelA_GNN().to(device)
    model_b = ModelB_Similarity().to(device)
    
    print(f"\nModel A parameters: {model_a.count_parameters():,}")
    print(f"Model B parameters: {model_b.count_parameters():,}")
    
    # Check parameter constraints
    if model_a.count_parameters() > 1_500_000:
        print(f"WARNING: Model A exceeds 1.5M parameter limit!")
    if model_b.count_parameters() > 1_500_000:
        print(f"WARNING: Model B exceeds 1.5M parameter limit!")
    
    # Train models
    train_model_simple(model_a, "ModelA_GNN", train_samples, epochs=100)
    train_model_simple(model_b, "ModelB_Similarity", train_samples, epochs=100)
    
    # Evaluate models
    print("\n" + "="*40)
    print("EVALUATION")
    print("="*40)
    
    sklearn_ari = evaluate_sklearn_simple(test_samples)
    model_a_ari = evaluate_model_simple(model_a, test_samples)
    model_b_ari = evaluate_model_simple(model_b, test_samples)
    
    # Results
    print(f"\nResults:")
    print(f"sklearn baseline ARI: {sklearn_ari:.4f}")
    print(f"Model A ARI: {model_a_ari:.4f}")
    print(f"Model B ARI: {model_b_ari:.4f}")
    
    # Check requirements
    print(f"\n" + "="*40)
    print("REQUIREMENT CHECK")
    print("="*40)
    
    target_ari = 0.80
    print(f"Target ARI >= {target_ari}")
    print(f"Model A meets target: {'YES' if model_a_ari >= target_ari else 'NO'}")
    print(f"Model B meets target: {'YES' if model_b_ari >= target_ari else 'NO'}")
    print(f"Model A beats sklearn: {'YES' if model_a_ari > sklearn_ari else 'NO'}")
    print(f"Model B beats sklearn: {'YES' if model_b_ari > sklearn_ari else 'NO'}")
    
    # Save results
    results_df = pd.DataFrame({
        'Method': ['sklearn_baseline', 'ModelA_GNN', 'ModelB_Similarity'],
        'ARI': [sklearn_ari, model_a_ari, model_b_ari],
        'Meets_Target': [sklearn_ari >= target_ari, model_a_ari >= target_ari, model_b_ari >= target_ari],
        'Beats_Sklearn': [False, model_a_ari > sklearn_ari, model_b_ari > sklearn_ari]
    })
    
    results_df.to_csv('efficient_training_results.csv', index=False)
    print(f"\nResults saved to: efficient_training_results.csv")
    
    return {
        'sklearn_ari': sklearn_ari,
        'model_a_ari': model_a_ari,
        'model_b_ari': model_b_ari
    }

if __name__ == "__main__":
    results = main()
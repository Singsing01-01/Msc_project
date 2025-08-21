import torch
import torch.optim as optim
import numpy as np
from model_a_gnn import ModelA_GNN, ModelALoss
from model_b_similarity import ModelB_Similarity, ModelBLoss
from data_generation import SyntheticDataGenerator
import time

def quick_validation():
    """Quick validation that training works and produces reasonable results"""
    print("Quick Training Validation")
    print("="*40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create small dataset
    generator = SyntheticDataGenerator(img_size=64, n_samples=300, noise=0.1)
    
    # Generate one training sample
    points, labels, adjacency = generator.generate_circles_data(n_samples=300)
    image = generator.points_to_image(points)
    
    sample = {
        'image': image,
        'points': points,
        'labels': labels,
        'adjacency': adjacency
    }
    
    # Initialize models
    model_a = ModelA_GNN().to(device)
    model_b = ModelB_Similarity().to(device)
    
    print(f"Model A parameters: {model_a.count_parameters():,}")
    print(f"Model B parameters: {model_b.count_parameters():,}")
    
    # Verify parameter constraints
    assert model_a.count_parameters() < 1_500_000, "Model A exceeds parameter limit"
    assert model_b.count_parameters() < 1_500_000, "Model B exceeds parameter limit"
    print("✓ Parameter constraints satisfied")
    
    # Test Model A training
    print("\nTesting Model A training...")
    model_a.train()
    criterion_a = ModelALoss(coord_weight=1.0, edge_weight=2.0, count_weight=0.1)
    optimizer_a = optim.AdamW(model_a.parameters(), lr=0.01)
    
    losses_a = []
    for epoch in range(20):
        # Prepare data
        image_tensor = torch.FloatTensor(sample['image']).unsqueeze(0).unsqueeze(0).to(device)
        n_nodes = len(sample['points'])
        
        targets = {
            'points': torch.FloatTensor(sample['points']).unsqueeze(0).to(device),
            'adjacency': torch.FloatTensor(sample['adjacency']).unsqueeze(0).to(device),
            'node_masks': torch.zeros(1, 350, dtype=torch.bool).to(device)
        }
        targets['node_masks'][0, :n_nodes] = True
        
        optimizer_a.zero_grad()
        predictions = model_a(image_tensor, targets['node_masks'])
        loss_dict = criterion_a(predictions, targets)
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer_a.step()
        
        losses_a.append(loss.item())
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print(f"Model A: Initial loss = {losses_a[0]:.4f}, Final loss = {losses_a[-1]:.4f}")
    improvement_a = (losses_a[0] - losses_a[-1]) / losses_a[0] * 100
    print(f"Model A: {improvement_a:.1f}% loss reduction")
    
    # Test Model B training
    print("\nTesting Model B training...")
    model_b.train()
    criterion_b = ModelBLoss(coord_weight=1.0, edge_weight=2.0, count_weight=0.1)
    optimizer_b = optim.AdamW(model_b.parameters(), lr=0.01)
    
    losses_b = []
    for epoch in range(20):
        # Prepare data
        image_tensor = torch.FloatTensor(sample['image']).unsqueeze(0).unsqueeze(0).to(device)
        n_nodes = len(sample['points'])
        
        targets = {
            'points': torch.FloatTensor(sample['points']).unsqueeze(0).to(device),
            'adjacency': torch.FloatTensor(sample['adjacency']).unsqueeze(0).to(device),
            'node_masks': torch.zeros(1, 350, dtype=torch.bool).to(device)
        }
        targets['node_masks'][0, :n_nodes] = True
        
        optimizer_b.zero_grad()
        predictions = model_b(image_tensor, targets['node_masks'])
        loss_dict = criterion_b(predictions, targets)
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer_b.step()
        
        losses_b.append(loss.item())
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print(f"Model B: Initial loss = {losses_b[0]:.4f}, Final loss = {losses_b[-1]:.4f}")
    improvement_b = (losses_b[0] - losses_b[-1]) / losses_b[0] * 100
    print(f"Model B: {improvement_b:.1f}% loss reduction")
    
    # Quick evaluation test
    print("\nTesting model evaluation...")
    model_a.eval()
    model_b.eval()
    
    with torch.no_grad():
        image_tensor = torch.FloatTensor(sample['image']).unsqueeze(0).unsqueeze(0).to(device)
        n_nodes = len(sample['points'])
        node_mask = torch.zeros(1, 350, dtype=torch.bool).to(device)
        node_mask[0, :n_nodes] = True
        
        # Test Model A prediction
        pred_a = model_a(image_tensor, node_mask)
        adj_a = pred_a['adjacency_matrix'][0][:n_nodes, :n_nodes].cpu().numpy()
        
        # Test Model B prediction
        pred_b = model_b(image_tensor, node_mask)
        adj_b = pred_b['adjacency_matrix'][0][:n_nodes, :n_nodes].cpu().numpy()
        
        print(f"Model A adjacency range: [{adj_a.min():.3f}, {adj_a.max():.3f}]")
        print(f"Model B adjacency range: [{adj_b.min():.3f}, {adj_b.max():.3f}]")
    
    print("\n" + "="*40)
    print("VALIDATION RESULTS")
    print("="*40)
    
    if improvement_a > 10 and improvement_b > 10:
        print("✓ Both models show significant learning (>10% loss reduction)")
    else:
        print("⚠ Models may need more training or different hyperparameters")
    
    if adj_a.max() > 0.1 and adj_b.max() > 0.1:
        print("✓ Both models produce meaningful adjacency predictions")
    else:
        print("⚠ Adjacency predictions may be too weak")
    
    print("\nQuick validation completed!")
    print("Models are ready for full training.")
    
    return {
        'model_a_improvement': improvement_a,
        'model_b_improvement': improvement_b,
        'model_a_final_loss': losses_a[-1],
        'model_b_final_loss': losses_b[-1]
    }

if __name__ == "__main__":
    results = quick_validation()
import torch
import torch.optim as optim
import numpy as np
from model_a_gnn import ModelA_GNN, ModelALoss
from model_b_similarity import ModelB_Similarity, ModelBLoss
from data_generation import SyntheticDataGenerator
from data_augmentation import create_data_loaders

def quick_training_test():
    """Quick training test to verify everything works"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create small dataset
    generator = SyntheticDataGenerator(img_size=64, n_samples=300, noise=0.1)
    train_data, test_data = generator.create_train_test_split(train_size=20, test_size=10)
    train_loader, test_loader = create_data_loaders(train_data, test_data, batch_size=2)
    
    # Initialize models
    model_a = ModelA_GNN().to(device)
    model_b = ModelB_Similarity().to(device)
    
    print(f"Model A parameters: {model_a.count_parameters():,}")
    print(f"Model B parameters: {model_b.count_parameters():,}")
    
    # Train Model A for a few epochs
    print("\nTesting Model A training...")
    model_a.train()
    criterion_a = ModelALoss()
    optimizer_a = optim.Adam(model_a.parameters(), lr=0.001)
    
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            images = batch['images'].to(device)
            targets = {
                'points': batch['points'].to(device),
                'adjacency': batch['adjacency'].to(device),
                'node_masks': batch['node_masks'].to(device)
            }
            
            optimizer_a.zero_grad()
            predictions = model_a(images, targets['node_masks'])
            loss_dict = criterion_a(predictions, targets)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer_a.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5, Model A Loss: {total_loss:.4f}")
    
    # Train Model B for a few epochs
    print("\nTesting Model B training...")
    model_b.train()
    criterion_b = ModelBLoss()
    optimizer_b = optim.Adam(model_b.parameters(), lr=0.001)
    
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            images = batch['images'].to(device)
            targets = {
                'points': batch['points'].to(device),
                'adjacency': batch['adjacency'].to(device),
                'node_masks': batch['node_masks'].to(device)
            }
            
            optimizer_b.zero_grad()
            predictions = model_b(images, targets['node_masks'])
            loss_dict = criterion_b(predictions, targets)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer_b.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5, Model B Loss: {total_loss:.4f}")
    
    print("\nQuick training test completed successfully!")
    return True

if __name__ == "__main__":
    quick_training_test()
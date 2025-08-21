import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import os
from tqdm import tqdm

from model_a_gnn import ModelA_GNN, ModelALoss
from model_b_similarity import ModelB_Similarity, ModelBLoss
from data_augmentation import GraphDataAugmentation, create_data_loaders


class DynamicLossWeights:
    def __init__(self, 
                 coord_start: float = 1.0,
                 edge_start: float = 0.5, 
                 edge_end: float = 1.0,
                 count_weight: float = 0.1,
                 similarity_weight: float = 0.5):
        
        self.coord_weight = coord_start
        self.edge_start = edge_start
        self.edge_end = edge_end
        self.count_weight = count_weight
        self.similarity_weight = similarity_weight
    
    def get_weights(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        progress = epoch / total_epochs
        edge_weight = self.edge_start + (self.edge_end - self.edge_start) * progress
        
        return {
            'coord_weight': self.coord_weight,
            'edge_weight': edge_weight,
            'count_weight': self.count_weight,
            'similarity_weight': self.similarity_weight
        }


class PhysicalConstraintLoss(nn.Module):
    def __init__(self, weight: float = 0.1):
        super(PhysicalConstraintLoss, self).__init__()
        self.weight = weight
    
    def forward(self, predicted_coords: torch.Tensor, 
                target_coords: torch.Tensor, 
                node_masks: torch.Tensor) -> torch.Tensor:
        
        batch_size = predicted_coords.shape[0]
        pred_max_nodes = predicted_coords.shape[1]
        target_max_nodes = node_masks.shape[1]
        total_loss = 0.0
        
        for b in range(batch_size):
            if target_max_nodes <= pred_max_nodes:
                mask = node_masks[b]
                pred = predicted_coords[b][:target_max_nodes][mask]
                target = target_coords[b][mask]
            else:
                mask = node_masks[b][:pred_max_nodes]
                pred = predicted_coords[b][mask]
                target = target_coords[b][:pred_max_nodes][mask]
            
            if pred.shape[0] <= 1:
                continue
            
            pred_dists = torch.cdist(pred, pred)
            target_dists = torch.cdist(target, target)
            
            distance_loss = torch.mean((pred_dists - target_dists) ** 2)
            total_loss += distance_loss
        
        return self.weight * total_loss / batch_size


class ModelTrainer:
    def __init__(self, 
                 model: Union[ModelA_GNN, ModelB_Similarity],
                 model_type: str,
                 device: torch.device,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-5,
                 gradient_clip_value: float = 1.0):
        
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device
        self.gradient_clip_value = gradient_clip_value
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
        
        self.dynamic_weights = DynamicLossWeights()
        self.physical_constraint = PhysicalConstraintLoss()
        
        if model_type == 'model_a':
            self.base_loss_fn = ModelALoss()
        else:
            self.base_loss_fn = ModelBLoss()
        
        self.training_history = {
            'train_loss': [],
            'train_coord_loss': [],
            'train_edge_loss': [],
            'train_count_loss': [],
            'train_physical_loss': [],
            'learning_rates': []
        }
        
        if model_type == 'model_b':
            self.training_history['train_similarity_loss'] = []
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor],
                    epoch: int, total_epochs: int) -> Dict[str, torch.Tensor]:
        
        weights = self.dynamic_weights.get_weights(epoch, total_epochs)
        
        if self.model_type == 'model_a':
            self.base_loss_fn.coord_weight = weights['coord_weight']
            self.base_loss_fn.edge_weight = weights['edge_weight']
            self.base_loss_fn.count_weight = weights['count_weight']
        else:
            self.base_loss_fn.coord_weight = weights['coord_weight']
            self.base_loss_fn.edge_weight = weights['edge_weight']
            self.base_loss_fn.count_weight = weights['count_weight']
            self.base_loss_fn.similarity_weight = weights['similarity_weight']
        
        base_losses = self.base_loss_fn(predictions, targets)
        
        physical_loss = self.physical_constraint(
            predictions['predicted_coords'],
            targets['points'],
            targets['node_masks']
        )
        
        total_loss = base_losses['total_loss'] + physical_loss
        
        result = base_losses.copy()
        result['total_loss'] = total_loss
        result['physical_loss'] = physical_loss
        
        return result
    
    def train_epoch(self, dataloader, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'coord_loss': 0.0,
            'edge_loss': 0.0,
            'count_loss': 0.0,
            'physical_loss': 0.0
        }
        
        if self.model_type == 'model_b':
            epoch_losses['similarity_loss'] = 0.0
        
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}"):
            images = batch['images'].to(self.device)
            points = batch['points'].to(self.device)
            adjacency = batch['adjacency'].to(self.device)
            node_masks = batch['node_masks'].to(self.device)
            
            targets = {
                'points': points,
                'adjacency': adjacency,
                'node_masks': node_masks
            }
            
            self.optimizer.zero_grad()
            
            predictions = self.model(images, node_masks)
            
            losses = self.compute_loss(predictions, targets, epoch, total_epochs)
            
            losses['total_loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
            
            self.optimizer.step()
            
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            num_batches += 1
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'coord_loss': 0.0,
            'edge_loss': 0.0,
            'count_loss': 0.0,
            'physical_loss': 0.0
        }
        
        if self.model_type == 'model_b':
            val_losses['similarity_loss'] = 0.0
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                points = batch['points'].to(self.device)
                adjacency = batch['adjacency'].to(self.device)
                node_masks = batch['node_masks'].to(self.device)
                
                targets = {
                    'points': points,
                    'adjacency': adjacency,
                    'node_masks': node_masks
                }
                
                predictions = self.model(images, node_masks)
                losses = self.compute_loss(predictions, targets, 0, 1)
                
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
                
                num_batches += 1
        
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def train(self, 
              train_loader, 
              val_loader, 
              num_epochs: int = 50,
              save_dir: str = "./checkpoints",
              save_every: int = 10) -> Dict[str, List[float]]:
        
        os.makedirs(save_dir, exist_ok=True)
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        best_val_loss = float('inf')
        
        print(f"Starting training {self.model_type} for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_losses = self.train_epoch(train_loader, epoch, num_epochs)
            
            val_losses = self.validate(val_loader)
            
            scheduler.step()
            
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.training_history['train_loss'].append(train_losses['total_loss'])
            self.training_history['train_coord_loss'].append(train_losses['coord_loss'])
            self.training_history['train_edge_loss'].append(train_losses['edge_loss'])
            self.training_history['train_count_loss'].append(train_losses['count_loss'])
            self.training_history['train_physical_loss'].append(train_losses['physical_loss'])
            self.training_history['learning_rates'].append(current_lr)
            
            if self.model_type == 'model_b':
                self.training_history['train_similarity_loss'].append(train_losses['similarity_loss'])
            
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
            print(f"  Train - Total: {train_losses['total_loss']:.4f}, "
                  f"Coord: {train_losses['coord_loss']:.4f}, "
                  f"Edge: {train_losses['edge_loss']:.4f}")
            print(f"  Val   - Total: {val_losses['total_loss']:.4f}, "
                  f"Coord: {val_losses['coord_loss']:.4f}, "
                  f"Edge: {val_losses['edge_loss']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'training_history': self.training_history
                }, os.path.join(save_dir, f"{self.model_type}_best.pth"))
            
            if (epoch + 1) % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_losses['total_loss'],
                    'training_history': self.training_history
                }, os.path.join(save_dir, f"{self.model_type}_epoch_{epoch+1}.pth"))
        
        return self.training_history
    
    def save_model(self, filepath: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']


def test_training_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_data = np.load('/Users/jeremyfang/Downloads/image_to_graph/data/train_data.npy', allow_pickle=True)
    test_data = np.load('/Users/jeremyfang/Downloads/image_to_graph/data/test_data.npy', allow_pickle=True)
    
    train_loader, val_loader = create_data_loaders(
        train_data.tolist(), 
        test_data.tolist(), 
        batch_size=2, 
        use_augmentation=True
    )
    
    print("Testing Model B training...")
    model_b = ModelB_Similarity()
    trainer_b = ModelTrainer(model_b, 'model_b', device)
    
    history_b = trainer_b.train(train_loader, val_loader, num_epochs=3, save_dir="./test_checkpoints")
    
    print(f"Training completed. Final loss: {history_b['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    test_training_pipeline()
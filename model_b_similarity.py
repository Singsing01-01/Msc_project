import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class LightweightCNNEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, feature_dim: int = 256):
        super(LightweightCNNEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, feature_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        
        return x


class NodeDetector(nn.Module):
    def __init__(self, feature_dim: int = 256, max_nodes: int = 350, coord_dim: int = 2):
        super(NodeDetector, self).__init__()
        
        self.max_nodes = max_nodes
        self.coord_dim = coord_dim
        
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_coords = nn.Linear(64, max_nodes * coord_dim)
        self.fc_count = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        coords = self.fc_coords(x).view(-1, self.max_nodes, self.coord_dim)
        coords = torch.tanh(coords)
        
        node_count = torch.sigmoid(self.fc_count(x)) * self.max_nodes
        
        return coords, node_count


class SimilarityCalculator(nn.Module):
    def __init__(self, coord_dim: int = 2):
        super(SimilarityCalculator, self).__init__()
        self.coord_dim = coord_dim
        
    def forward(self, coords: torch.Tensor, node_masks: torch.Tensor) -> torch.Tensor:
        batch_size, max_nodes, _ = coords.shape
        
        node_masks_trimmed = node_masks[:, :max_nodes] if node_masks.shape[1] > max_nodes else node_masks
        
        if node_masks_trimmed.shape[1] < max_nodes:
            padding = torch.zeros(batch_size, max_nodes - node_masks_trimmed.shape[1], 
                                dtype=torch.bool, device=node_masks.device)
            node_masks_trimmed = torch.cat([node_masks_trimmed, padding], dim=1)
        
        normalized_coords = F.normalize(coords, dim=-1, eps=1e-8)
        
        cosine_sim = torch.bmm(normalized_coords, normalized_coords.transpose(-1, -2))
        
        mask_matrix = node_masks_trimmed.unsqueeze(-1) & node_masks_trimmed.unsqueeze(-2)
        cosine_sim = cosine_sim * mask_matrix.float()
        
        identity = torch.eye(max_nodes, device=coords.device).unsqueeze(0).expand(batch_size, -1, -1)
        cosine_sim = cosine_sim * (1 - identity)
        
        return cosine_sim


class SimilarityCorrector(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super(SimilarityCorrector, self).__init__()
        
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, similarity_matrix: torch.Tensor, node_masks: torch.Tensor) -> torch.Tensor:
        batch_size, max_nodes, _ = similarity_matrix.shape
        
        node_masks_trimmed = node_masks[:, :max_nodes] if node_masks.shape[1] > max_nodes else node_masks
        
        if node_masks_trimmed.shape[1] < max_nodes:
            padding = torch.zeros(batch_size, max_nodes - node_masks_trimmed.shape[1], 
                                dtype=torch.bool, device=node_masks.device)
            node_masks_trimmed = torch.cat([node_masks_trimmed, padding], dim=1)
        
        corrected_matrix = torch.zeros_like(similarity_matrix)
        
        for b in range(batch_size):
            mask = node_masks_trimmed[b]
            n_valid = mask.sum().item()
            
            if n_valid <= 1:
                continue
            
            valid_sim = similarity_matrix[b][mask][:, mask]
            valid_sim_flat = valid_sim.flatten().unsqueeze(-1)
            
            corrected = F.relu(self.fc1(valid_sim_flat))
            corrected = self.dropout(corrected)
            corrected = F.relu(self.fc2(corrected))
            corrected = self.dropout(corrected)
            corrected = torch.sigmoid(self.fc3(corrected))
            
            corrected_sim = corrected.view(n_valid, n_valid)
            
            valid_indices = torch.nonzero(mask).squeeze(1)
            corrected_matrix[b][valid_indices.unsqueeze(1), valid_indices.unsqueeze(0)] = corrected_sim
        
        corrected_matrix = (corrected_matrix + corrected_matrix.transpose(-1, -2)) / 2
        
        identity = torch.eye(max_nodes, device=similarity_matrix.device).unsqueeze(0).expand(batch_size, -1, -1)
        corrected_matrix = corrected_matrix * (1 - identity)
        
        return corrected_matrix


class ModelB_Similarity(nn.Module):
    def __init__(self, 
                 input_channels: int = 1,
                 feature_dim: int = 256,
                 max_nodes: int = 350,
                 coord_dim: int = 2,
                 similarity_hidden_dim: int = 32):
        
        super(ModelB_Similarity, self).__init__()
        
        self.max_nodes = max_nodes
        self.coord_dim = coord_dim
        
        self.cnn_encoder = LightweightCNNEncoder(input_channels, feature_dim)
        self.node_detector = NodeDetector(feature_dim, max_nodes, coord_dim)
        self.similarity_calculator = SimilarityCalculator(coord_dim)
        self.similarity_corrector = SimilarityCorrector(similarity_hidden_dim)
        
    def forward(self, images: torch.Tensor, node_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_features = self.cnn_encoder(images)
        
        predicted_coords, node_counts = self.node_detector(image_features)
        
        raw_similarity = self.similarity_calculator(predicted_coords, node_masks)
        
        corrected_adjacency = self.similarity_corrector(raw_similarity, node_masks)
        
        return {
            'predicted_coords': predicted_coords,
            'node_counts': node_counts,
            'adjacency_matrix': corrected_adjacency,
            'raw_similarity': raw_similarity,
            'image_features': image_features
        }
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelBLoss(nn.Module):
    def __init__(self, coord_weight: float = 1.0, edge_weight: float = 1.0, 
                 count_weight: float = 0.1, similarity_weight: float = 0.5):
        super(ModelBLoss, self).__init__()
        self.coord_weight = coord_weight
        self.edge_weight = edge_weight
        self.count_weight = count_weight
        self.similarity_weight = similarity_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        batch_size, pred_max_nodes, coord_dim = predictions['predicted_coords'].shape
        target_max_nodes = targets['node_masks'].shape[1]
        device = predictions['predicted_coords'].device
        
        if target_max_nodes < pred_max_nodes:
            padding_size = pred_max_nodes - target_max_nodes
            
            node_masks_padded = torch.cat([
                targets['node_masks'], 
                torch.zeros(batch_size, padding_size, dtype=torch.bool, device=device)
            ], dim=1)
            
            points_padded = torch.cat([
                targets['points'],
                torch.zeros(batch_size, padding_size, coord_dim, device=device)
            ], dim=1)
            
            adjacency_padded = torch.cat([
                torch.cat([
                    targets['adjacency'],
                    torch.zeros(batch_size, target_max_nodes, padding_size, device=device)
                ], dim=2),
                torch.zeros(batch_size, padding_size, pred_max_nodes, device=device)
            ], dim=1)
            
        elif target_max_nodes > pred_max_nodes:
            node_masks_padded = targets['node_masks'][:, :pred_max_nodes]
            points_padded = targets['points'][:, :pred_max_nodes]
            adjacency_padded = targets['adjacency'][:, :pred_max_nodes, :pred_max_nodes]
        else:
            node_masks_padded = targets['node_masks']
            points_padded = targets['points']
            adjacency_padded = targets['adjacency']
        
        mask_expanded = node_masks_padded.unsqueeze(-1).expand(-1, -1, coord_dim)
        coord_loss = self.mse_loss(
            predictions['predicted_coords'][mask_expanded].view(-1),
            points_padded[mask_expanded].view(-1)
        )
        
        adjacency_mask = node_masks_padded.unsqueeze(-1) & node_masks_padded.unsqueeze(-2)
        edge_loss = self.bce_loss(
            predictions['adjacency_matrix'][adjacency_mask],
            adjacency_padded[adjacency_mask]
        )
        
        actual_counts = targets['node_masks'].sum(dim=1).float()
        count_loss = self.mse_loss(predictions['node_counts'].squeeze(), actual_counts)
        
        similarity_loss = self.mse_loss(
            predictions['raw_similarity'][adjacency_mask],
            adjacency_padded[adjacency_mask]
        )
        
        total_loss = (self.coord_weight * coord_loss + 
                     self.edge_weight * edge_loss + 
                     self.count_weight * count_loss +
                     self.similarity_weight * similarity_loss)
        
        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'edge_loss': edge_loss,
            'count_loss': count_loss,
            'similarity_loss': similarity_loss
        }


def test_model_b():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ModelB_Similarity(
        input_channels=1,
        feature_dim=256,
        max_nodes=350,
        coord_dim=2,
        similarity_hidden_dim=32
    ).to(device)
    
    print(f"Model B parameters: {model.count_parameters():,}")
    
    batch_size = 2
    max_nodes = 350
    actual_nodes = 300
    images = torch.randn(batch_size, 1, 64, 64).to(device)
    points = torch.randn(batch_size, max_nodes, 2).to(device)
    adjacency = torch.rand(batch_size, max_nodes, max_nodes).to(device)
    node_masks = torch.zeros(batch_size, max_nodes, dtype=torch.bool).to(device)
    node_masks[:, :actual_nodes] = True
    
    targets = {
        'points': points,
        'adjacency': adjacency,
        'node_masks': node_masks
    }
    
    with torch.no_grad():
        predictions = model(images, node_masks)
    
    print("Model B forward pass successful!")
    print(f"Predicted coords shape: {predictions['predicted_coords'].shape}")
    print(f"Adjacency matrix shape: {predictions['adjacency_matrix'].shape}")
    print(f"Raw similarity shape: {predictions['raw_similarity'].shape}")
    
    loss_fn = ModelBLoss()
    losses = loss_fn(predictions, targets)
    
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Coord loss: {losses['coord_loss'].item():.4f}")
    print(f"Edge loss: {losses['edge_loss'].item():.4f}")
    print(f"Count loss: {losses['count_loss'].item():.4f}")
    print(f"Similarity loss: {losses['similarity_loss'].item():.4f}")
    
    print(f"\nParameter comparison:")
    print(f"Model A: ~4.58M parameters")
    print(f"Model B: {model.count_parameters()/1e6:.2f}M parameters")
    print(f"Efficiency gain: {4.58/(model.count_parameters()/1e6):.1f}x fewer parameters")


if __name__ == "__main__":
    test_model_b()
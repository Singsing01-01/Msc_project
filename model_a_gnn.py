import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple, Dict


class CNNEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, feature_dim: int = 256):
        super(CNNEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, feature_dim)
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(0.2)
    
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


class NodeRegressor(nn.Module):
    def __init__(self, feature_dim: int = 256, max_nodes: int = 350, coord_dim: int = 2):
        super(NodeRegressor, self).__init__()
        
        self.max_nodes = max_nodes
        self.coord_dim = coord_dim
        
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_coords = nn.Linear(64, max_nodes * coord_dim)
        self.fc_count = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        coords = self.fc_coords(x).view(-1, self.max_nodes, self.coord_dim)
        node_count = torch.sigmoid(self.fc_count(x)) * self.max_nodes
        
        return coords, node_count


class GraphBuilder(nn.Module):
    def __init__(self, k_nearest: int = 10):
        super(GraphBuilder, self).__init__()
        self.k_nearest = k_nearest
    
    def build_knn_graph(self, coords: torch.Tensor, node_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_nodes, _ = coords.shape
        edge_indices = []
        edge_weights = []
        
        for b in range(batch_size):
            # Ensure mask matches coordinate tensor dimensions
            current_mask = node_masks[b][:max_nodes] if node_masks[b].shape[0] > max_nodes else node_masks[b]
            if current_mask.shape[0] < max_nodes:
                # Pad mask if too short
                padding = torch.zeros(max_nodes - current_mask.shape[0], dtype=torch.bool, device=current_mask.device)
                current_mask = torch.cat([current_mask, padding])
            mask = current_mask
            valid_coords = coords[b][mask]
            n_valid = valid_coords.shape[0]
            
            if n_valid <= 1:
                continue
            
            dists = torch.cdist(valid_coords, valid_coords)
            # Avoid inplace operation that breaks gradient computation
            mask_diag = torch.eye(n_valid, device=dists.device).bool()
            dists = dists.masked_fill(mask_diag, float('inf'))
            
            _, topk_indices = torch.topk(dists, min(self.k_nearest, n_valid-1), 
                                       dim=1, largest=False)
            
            valid_indices = torch.nonzero(mask).squeeze(1)
            
            for i in range(n_valid):
                for j in topk_indices[i]:
                    edge_indices.append([b * max_nodes + valid_indices[i], b * max_nodes + valid_indices[j]])
                    weight = torch.exp(-0.5 * dists[i, j])
                    edge_weights.append(weight)
        
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=coords.device)
            edge_weight = torch.zeros(0, device=coords.device)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=coords.device).t()
            edge_weight = torch.stack(edge_weights)
        
        return edge_index, edge_weight


class GNNProcessor(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 32):
        super(GNNProcessor, self).__init__()
        
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, coords: torch.Tensor, edge_index: torch.Tensor, 
               edge_weight: torch.Tensor, node_masks: torch.Tensor) -> torch.Tensor:
        
        batch_size, max_nodes, coord_dim = coords.shape
        output_dim = self.gcn2.out_channels
        
        node_features = torch.zeros(batch_size, max_nodes, output_dim, device=coords.device)
        
        if edge_index.numel() == 0:
            return node_features
        
        for b in range(batch_size):
            # Ensure mask matches coordinate tensor dimensions
            current_mask = node_masks[b][:max_nodes] if node_masks[b].shape[0] > max_nodes else node_masks[b]
            if current_mask.shape[0] < max_nodes:
                # Pad mask if too short
                padding = torch.zeros(max_nodes - current_mask.shape[0], dtype=torch.bool, device=current_mask.device)
                current_mask = torch.cat([current_mask, padding])
            mask = current_mask
            valid_coords = coords[b][mask]
            n_valid = valid_coords.shape[0]
            
            if n_valid <= 1:
                continue
            
            batch_edge_mask = (edge_index[0] >= b * max_nodes) & (edge_index[0] < (b + 1) * max_nodes)
            batch_edge_index = edge_index[:, batch_edge_mask] - b * max_nodes
            batch_edge_weight = edge_weight[batch_edge_mask]
            
            if batch_edge_index.numel() == 0:
                continue
            
            x = F.relu(self.gcn1(valid_coords, batch_edge_index, batch_edge_weight))
            x = self.dropout(x)
            x = F.relu(self.gcn2(x, batch_edge_index, batch_edge_weight))
            
            node_features[b][mask] = x
        
        return node_features


class EdgePredictor(nn.Module):
    def __init__(self, node_feature_dim: int = 32):
        super(EdgePredictor, self).__init__()
        
        self.fc1 = nn.Linear(node_feature_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, node_features: torch.Tensor, node_masks: torch.Tensor) -> torch.Tensor:
        batch_size, max_nodes, feature_dim = node_features.shape
        
        adjacency = torch.zeros(batch_size, max_nodes, max_nodes, device=node_features.device)
        
        for b in range(batch_size):
            # Ensure mask matches feature tensor dimensions
            current_mask = node_masks[b][:max_nodes] if node_masks[b].shape[0] > max_nodes else node_masks[b]
            if current_mask.shape[0] < max_nodes:
                # Pad mask if too short
                padding = torch.zeros(max_nodes - current_mask.shape[0], dtype=torch.bool, device=current_mask.device)
                current_mask = torch.cat([current_mask, padding])
            valid_features = node_features[b][current_mask]
            n_valid = valid_features.shape[0]
            
            if n_valid <= 1:
                continue
            
            for i in range(n_valid):
                for j in range(n_valid):
                    if i != j:
                        combined = torch.cat([valid_features[i], valid_features[j]], dim=0)
                        
                        edge_score = F.relu(self.fc1(combined))
                        edge_score = self.dropout(edge_score)
                        edge_score = F.relu(self.fc2(edge_score))
                        edge_score = self.dropout(edge_score)
                        edge_score = torch.sigmoid(self.fc3(edge_score))
                        
                        valid_i = torch.nonzero(current_mask)[i].item()
                        valid_j = torch.nonzero(current_mask)[j].item()
                        adjacency[b, valid_i, valid_j] = edge_score
        
        adjacency = (adjacency + adjacency.transpose(-1, -2)) / 2
        
        return adjacency


class ModelA_GNN(nn.Module):
    def __init__(self, 
                 input_channels: int = 1,
                 feature_dim: int = 256,
                 max_nodes: int = 350,
                 coord_dim: int = 2,
                 hidden_dim: int = 64,
                 node_feature_dim: int = 32):
        
        super(ModelA_GNN, self).__init__()
        
        self.max_nodes = max_nodes
        self.coord_dim = coord_dim
        
        self.cnn_encoder = CNNEncoder(input_channels, feature_dim)
        self.node_regressor = NodeRegressor(feature_dim, max_nodes, coord_dim)
        self.graph_builder = GraphBuilder(k_nearest=10)
        self.gnn_processor = GNNProcessor(coord_dim, hidden_dim, node_feature_dim)
        self.edge_predictor = EdgePredictor(node_feature_dim)
        
    def forward(self, images: torch.Tensor, node_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_features = self.cnn_encoder(images)
        
        predicted_coords, node_counts = self.node_regressor(image_features)
        
        edge_index, edge_weight = self.graph_builder.build_knn_graph(predicted_coords, node_masks)
        
        node_features = self.gnn_processor(predicted_coords, edge_index, edge_weight, node_masks)
        
        adjacency_matrix = self.edge_predictor(node_features, node_masks)
        
        return {
            'predicted_coords': predicted_coords,
            'node_counts': node_counts,
            'adjacency_matrix': adjacency_matrix,
            'node_features': node_features,
            'image_features': image_features
        }
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelALoss(nn.Module):
    def __init__(self, coord_weight: float = 1.0, edge_weight: float = 1.0, 
                 count_weight: float = 0.1):
        super(ModelALoss, self).__init__()
        self.coord_weight = coord_weight
        self.edge_weight = edge_weight
        self.count_weight = count_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        batch_size = predictions['predicted_coords'].shape[0]
        device = predictions['predicted_coords'].device
        
        total_coord_loss = 0.0
        total_edge_loss = 0.0
        valid_samples = 0
        
        # Process each sample in the batch separately to handle variable sizes
        for b in range(batch_size):
            # Get actual number of nodes for this sample
            mask = targets['node_masks'][b]
            n_actual = mask.sum().item()
            
            if n_actual == 0:
                continue
                
            # Extract valid coordinates and adjacency
            pred_coords = predictions['predicted_coords'][b][:n_actual]  # [n_actual, 2]
            true_coords = targets['points'][b][:n_actual]  # [n_actual, 2]
            
            pred_adj = predictions['adjacency_matrix'][b][:n_actual, :n_actual]  # [n_actual, n_actual]
            true_adj = targets['adjacency'][b][:n_actual, :n_actual]  # [n_actual, n_actual]
            
            # Coordinate loss for this sample
            coord_loss_sample = self.mse_loss(pred_coords, true_coords)
            total_coord_loss += coord_loss_sample
            
            # Edge loss for this sample
            edge_loss_sample = self.bce_loss(pred_adj, true_adj)
            total_edge_loss += edge_loss_sample
            
            valid_samples += 1
        
        # Average losses across valid samples
        if valid_samples > 0:
            coord_loss = total_coord_loss / valid_samples
            edge_loss = total_edge_loss / valid_samples
        else:
            coord_loss = torch.tensor(0.0, device=device, requires_grad=True)
            edge_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Node count loss
        actual_counts = targets['node_masks'].sum(dim=1).float()
        predicted_counts = predictions['node_counts'].squeeze()
        if predicted_counts.dim() == 0:
            predicted_counts = predicted_counts.unsqueeze(0)
        count_loss = self.mse_loss(predicted_counts, actual_counts)
        
        total_loss = (self.coord_weight * coord_loss + 
                     self.edge_weight * edge_loss + 
                     self.count_weight * count_loss)
        
        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'edge_loss': edge_loss,
            'count_loss': count_loss
        }


def test_model_a():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ModelA_GNN(
        input_channels=1,
        feature_dim=256,
        max_nodes=350,
        coord_dim=2,
        hidden_dim=64,
        node_feature_dim=32
    ).to(device)
    
    print(f"Model A parameters: {model.count_parameters():,}")
    
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
    
    print("Model A forward pass successful!")
    print(f"Predicted coords shape: {predictions['predicted_coords'].shape}")
    print(f"Adjacency matrix shape: {predictions['adjacency_matrix'].shape}")
    print(f"Node features shape: {predictions['node_features'].shape}")
    
    loss_fn = ModelALoss()
    losses = loss_fn(predictions, targets)
    
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Coord loss: {losses['coord_loss'].item():.4f}")
    print(f"Edge loss: {losses['edge_loss'].item():.4f}")
    print(f"Count loss: {losses['count_loss'].item():.4f}")


if __name__ == "__main__":
    test_model_a()
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.spatial.distance import pdist


class GraphDataAugmentation:
    def __init__(self, 
                 rotation_range: float = 30.0,
                 scale_range: Tuple[float, float] = (0.7, 1.3),
                 translation_range: int = 15,
                 brightness_range: float = 0.2,
                 noise_std: float = 0.05,
                 img_size: int = 64):
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.brightness_range = brightness_range
        self.noise_std = noise_std
        self.img_size = img_size
    
    def random_geometric_transform(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        tx = np.random.uniform(-self.translation_range, self.translation_range)
        ty = np.random.uniform(-self.translation_range, self.translation_range)
        
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        center = np.mean(points, axis=0)
        centered_points = points - center
        
        transformed_points = scale * (centered_points @ rotation_matrix.T) + center + np.array([tx, ty])
        
        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] = scale * rotation_matrix
        transform_matrix[:2, 2] = np.array([tx, ty])
        
        return transformed_points, transform_matrix
    
    def apply_image_transform(self, image: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        M = transform_matrix[:2, :]
        transformed_image = cv2.warpAffine(image, M, (self.img_size, self.img_size), 
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return transformed_image
    
    def apply_photometric_transform(self, image: np.ndarray) -> np.ndarray:
        brightness_factor = np.random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
        image = image * brightness_factor
        
        noise = np.random.normal(0, self.noise_std, image.shape)
        image = image + noise
        
        return np.clip(image, 0, 1)
    
    def update_adjacency_matrix(self, 
                               original_points: np.ndarray, 
                               transformed_points: np.ndarray,
                               original_adjacency: np.ndarray,
                               preserve_topology: bool = True) -> np.ndarray:
        
        if preserve_topology:
            return original_adjacency
        
        gamma = 0.5
        n_points = transformed_points.shape[0]
        new_adjacency = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    dist = np.linalg.norm(transformed_points[i] - transformed_points[j])
                    new_adjacency[i, j] = np.exp(-gamma * dist**2)
        
        new_adjacency = (new_adjacency + new_adjacency.T) / 2
        np.fill_diagonal(new_adjacency, 0)
        
        return new_adjacency
    
    def normalize_points_to_image_space(self, points: np.ndarray) -> np.ndarray:
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        
        normalized = (points - min_vals) / (max_vals - min_vals + 1e-8)
        pixel_coords = normalized * (self.img_size - 1)
        
        return pixel_coords
    
    def augment_sample(self, data_dict: Dict, preserve_topology: bool = True) -> Dict:
        original_points = data_dict['points'].copy()
        original_image = data_dict['image'].copy()
        original_adjacency = data_dict['adjacency'].copy()
        
        transformed_points, transform_matrix = self.random_geometric_transform(original_points)
        
        transformed_image = self.apply_image_transform(original_image, transform_matrix)
        transformed_image = self.apply_photometric_transform(transformed_image)
        
        updated_adjacency = self.update_adjacency_matrix(
            original_points, transformed_points, original_adjacency, preserve_topology
        )
        
        pixel_coords = self.normalize_points_to_image_space(transformed_points)
        
        return {
            'image': transformed_image.astype(np.float32),
            'points': transformed_points.astype(np.float32),
            'points_pixel': pixel_coords.astype(np.float32),
            'labels': data_dict['labels'].copy(),
            'adjacency': updated_adjacency.astype(np.float32),
            'transform_matrix': transform_matrix.astype(np.float32)
        }


class GraphDataset(Dataset):
    def __init__(self, 
                 data_list: List[Dict], 
                 augmentation: Optional[GraphDataAugmentation] = None,
                 num_augmentations: int = 5):
        self.data_list = data_list
        self.augmentation = augmentation
        self.num_augmentations = num_augmentations
        self.use_augmentation = augmentation is not None
    
    def __len__(self):
        if self.use_augmentation:
            return len(self.data_list) * (self.num_augmentations + 1)
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.use_augmentation:
            original_idx = idx // (self.num_augmentations + 1)
            aug_idx = idx % (self.num_augmentations + 1)
            
            if aug_idx == 0:
                return self.data_list[original_idx]
            else:
                return self.augmentation.augment_sample(self.data_list[original_idx])
        else:
            return self.data_list[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    max_nodes = max(sample['points'].shape[0] for sample in batch)
    batch_size = len(batch)
    
    images = torch.stack([torch.from_numpy(sample['image']).unsqueeze(0) for sample in batch])
    
    points = torch.zeros(batch_size, max_nodes, 2)
    points_pixel = torch.zeros(batch_size, max_nodes, 2)
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes)
    labels = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_masks = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        n_nodes = sample['points'].shape[0]
        points[i, :n_nodes] = torch.from_numpy(sample['points'])
        points_pixel[i, :n_nodes] = torch.from_numpy(sample['points_pixel'])
        adjacency[i, :n_nodes, :n_nodes] = torch.from_numpy(sample['adjacency'])
        labels[i, :n_nodes] = torch.from_numpy(sample['labels'])
        node_masks[i, :n_nodes] = True
    
    return {
        'images': images,
        'points': points,
        'points_pixel': points_pixel,
        'adjacency': adjacency,
        'labels': labels,
        'node_masks': node_masks,
        'max_nodes': max_nodes
    }


def create_data_loaders(train_data: List[Dict], 
                       test_data: List[Dict],
                       batch_size: int = 4,
                       num_workers: int = 0,
                       use_augmentation: bool = True) -> Tuple[DataLoader, DataLoader]:
    
    augmentation = GraphDataAugmentation() if use_augmentation else None
    
    train_dataset = GraphDataset(train_data, augmentation=augmentation, num_augmentations=3)
    test_dataset = GraphDataset(test_data, augmentation=None)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader


def test_augmentation():
    from data_generation import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator(img_size=64, n_samples=50, noise=0.1)
    sample_data = generator.generate_dataset('circles', n_samples=50)
    
    augmentation = GraphDataAugmentation()
    
    print("Testing data augmentation...")
    print(f"Original image shape: {sample_data['image'].shape}")
    print(f"Original points shape: {sample_data['points'].shape}")
    print(f"Original adjacency shape: {sample_data['adjacency'].shape}")
    
    augmented_sample = augmentation.augment_sample(sample_data)
    
    print(f"Augmented image shape: {augmented_sample['image'].shape}")
    print(f"Augmented points shape: {augmented_sample['points'].shape}")
    print(f"Augmented adjacency shape: {augmented_sample['adjacency'].shape}")
    
    original_distances = pdist(sample_data['points'])
    augmented_distances = pdist(augmented_sample['points'])
    
    print(f"Original distance statistics: mean={np.mean(original_distances):.4f}, std={np.std(original_distances):.4f}")
    print(f"Augmented distance statistics: mean={np.mean(augmented_distances):.4f}, std={np.std(augmented_distances):.4f}")
    
    correlation = np.corrcoef(original_distances, augmented_distances)[0, 1]
    print(f"Distance correlation: {correlation:.4f}")
    
    print("Data augmentation test completed successfully!")


if __name__ == "__main__":
    test_augmentation()
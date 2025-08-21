"""
A100-Optimized Data Generation for Image-to-Graph Training
Optimized for high-performance GPU training with larger batch sizes and efficient memory usage.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import os
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
import time


class A100DataGenerator:
    """Optimized data generator for A100 GPU training."""
    
    def __init__(self, img_size: int = 64, n_samples: int = 300, noise: float = 0.1, 
                 num_workers: int = None):
        self.img_size = img_size
        self.n_samples = n_samples
        self.noise = noise
        self.scaler = StandardScaler()
        self.num_workers = num_workers or min(cpu_count(), 16)
    
    def generate_circles_data(self, n_samples: int = None, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate circles data with optional random state for reproducibility."""
        if n_samples is None:
            n_samples = self.n_samples
        
        if random_state is None:
            random_state = np.random.randint(0, 10000)
        
        X, y = make_circles(n_samples=n_samples, noise=self.noise, factor=0.6, 
                           random_state=random_state)
        X = self.scaler.fit_transform(X)
        return X, y, self._create_adjacency_matrix_vectorized(X)
    
    def generate_moons_data(self, n_samples: int = None, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate moons data with optional random state for reproducibility."""
        if n_samples is None:
            n_samples = self.n_samples
        
        if random_state is None:
            random_state = np.random.randint(0, 10000)
        
        X, y = make_moons(n_samples=n_samples, noise=self.noise, 
                         random_state=random_state)
        X = self.scaler.fit_transform(X)
        return X, y, self._create_adjacency_matrix_vectorized(X)
    
    def _create_adjacency_matrix_vectorized(self, X: np.ndarray, gamma: float = 0.5) -> np.ndarray:
        """Vectorized adjacency matrix computation for better performance."""
        # Use broadcasting to compute all pairwise distances at once
        X_expanded = X[:, np.newaxis, :]  # Shape: (n, 1, 2)
        X_t_expanded = X[np.newaxis, :, :]  # Shape: (1, n, 2)
        
        # Compute squared distances using broadcasting
        squared_dists = np.sum((X_expanded - X_t_expanded) ** 2, axis=2)
        
        # Apply RBF kernel
        adjacency = np.exp(-gamma * squared_dists)
        
        # Make symmetric and remove diagonal
        adjacency = (adjacency + adjacency.T) / 2
        np.fill_diagonal(adjacency, 0)
        
        return adjacency
    
    def points_to_image_optimized(self, X: np.ndarray) -> np.ndarray:
        """Optimized image generation with vectorized operations."""
        # Normalize coordinates to image space
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
        X_pixel = (X_scaled * (self.img_size - 1)).astype(int)
        
        # Create image using vectorized operations
        image = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Clip coordinates to valid range
        X_pixel = np.clip(X_pixel, 0, self.img_size - 1)
        
        # Set main points
        image[X_pixel[:, 1], X_pixel[:, 0]] = 1.0
        
        # Vectorized neighbor point setting
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_x = np.clip(X_pixel[:, 0] + dx, 0, self.img_size - 1)
                neighbor_y = np.clip(X_pixel[:, 1] + dy, 0, self.img_size - 1)
                image[neighbor_y, neighbor_x] = np.maximum(image[neighbor_y, neighbor_x], 0.7)
        
        return image
    
    def generate_single_sample(self, params: Dict) -> Dict:
        """Generate a single sample - used for parallel processing."""
        sample_idx, dataset_type, n_points, noise_level, random_state = params.values()
        
        # Create temporary generator with specific parameters
        temp_generator = A100DataGenerator(
            img_size=self.img_size, 
            n_samples=n_points, 
            noise=noise_level
        )
        
        if dataset_type == 'circles':
            X, y, adj = temp_generator.generate_circles_data(n_points, random_state)
        else:  # moons
            X, y, adj = temp_generator.generate_moons_data(n_points, random_state)
        
        image = temp_generator.points_to_image_optimized(X)
        
        # Normalize pixel coordinates
        X_pixel = (X - X.min()) / (X.max() - X.min() + 1e-8) * (self.img_size - 1)
        
        return {
            'image': image.astype(np.float32),
            'points': X.astype(np.float32),
            'points_pixel': X_pixel.astype(np.float32),
            'labels': y.astype(np.int64),
            'adjacency': adj.astype(np.float32),
            'n_points': n_points,
            'sample_idx': sample_idx
        }
    
    def create_train_test_split_parallel(self, train_size: int = 100, test_size: int = 50,
                                       min_points: int = 250, max_points: int = 351,
                                       min_noise: float = 0.05, max_noise: float = 0.15) -> Tuple[List[Dict], List[Dict]]:
        """Parallel data generation for faster processing."""
        print(f"Generating {train_size + test_size} samples using {self.num_workers} workers...")
        start_time = time.time()
        
        # Prepare parameters for parallel processing
        train_params = []
        for i in range(train_size):
            train_params.append({
                'sample_idx': i,
                'dataset_type': 'circles',
                'n_points': np.random.randint(min_points, max_points),
                'noise_level': np.random.uniform(min_noise, max_noise),
                'random_state': np.random.randint(0, 100000)
            })
        
        test_params = []
        for i in range(test_size):
            test_params.append({
                'sample_idx': i,
                'dataset_type': 'moons',
                'n_points': np.random.randint(min_points, max_points),
                'noise_level': np.random.uniform(min_noise, max_noise),
                'random_state': np.random.randint(0, 100000)
            })
        
        # Generate data in parallel
        with Pool(processes=self.num_workers) as pool:
            train_data = pool.map(self.generate_single_sample, train_params)
            test_data = pool.map(self.generate_single_sample, test_params)
        
        generation_time = time.time() - start_time
        print(f"Data generation completed in {generation_time:.2f} seconds")
        print(f"Average time per sample: {generation_time/(train_size + test_size)*1000:.2f} ms")
        
        return train_data, test_data
    
    def save_dataset_optimized(self, train_data: List[Dict], test_data: List[Dict], 
                             save_dir: str, compress: bool = True):
        """Optimized dataset saving with compression and metadata."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with compression for faster I/O
        if compress:
            with open(os.path.join(save_dir, 'train_data.pkl'), 'wb') as f:
                pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(save_dir, 'test_data.pkl'), 'wb') as f:
                pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            np.save(os.path.join(save_dir, 'train_data.npy'), train_data)
            np.save(os.path.join(save_dir, 'test_data.npy'), test_data)
        
        # Save metadata
        metadata = {
            'train_size': len(train_data),
            'test_size': len(test_data),
            'img_size': self.img_size,
            'point_range': [min([d['n_points'] for d in train_data]), 
                           max([d['n_points'] for d in train_data])],
            'compressed': compress,
            'generation_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {save_dir}")
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        print(f"Compression: {'Enabled' if compress else 'Disabled'}")
    
    def load_dataset_optimized(self, save_dir: str) -> Tuple[List[Dict], List[Dict]]:
        """Optimized dataset loading."""
        # Try compressed format first
        pkl_train = os.path.join(save_dir, 'train_data.pkl')
        pkl_test = os.path.join(save_dir, 'test_data.pkl')
        
        if os.path.exists(pkl_train) and os.path.exists(pkl_test):
            with open(pkl_train, 'rb') as f:
                train_data = pickle.load(f)
            with open(pkl_test, 'rb') as f:
                test_data = pickle.load(f)
        else:
            # Fallback to numpy format
            train_data = np.load(os.path.join(save_dir, 'train_data.npy'), allow_pickle=True)
            test_data = np.load(os.path.join(save_dir, 'test_data.npy'), allow_pickle=True)
        
        print(f"Loaded {len(train_data)} training and {len(test_data)} testing samples")
        return train_data, test_data
    
    def create_visualization_batch(self, data_batch: List[Dict], save_path: str = None, 
                                 max_samples: int = 9):
        """Create batch visualization for monitoring data quality."""
        plt.ioff()
        n_samples = min(len(data_batch), max_samples)
        n_cols = 3
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            data = data_batch[i]
            
            # Overlay image and points
            ax.imshow(data['image'], cmap='gray', alpha=0.7)
            points = data['points_pixel']
            labels = data['labels']
            ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', 
                      s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
            
            ax.set_title(f'Sample {i}: {data["n_points"]} points')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Batch visualization saved to {save_path}")
        plt.close(fig)


def main():
    """Main function for A100-optimized data generation."""
    print("A100-Optimized Data Generation Starting...")
    
    # A100-optimized parameters
    generator = A100DataGenerator(
        img_size=64, 
        n_samples=300, 
        noise=0.1,
        num_workers=16  # Optimized for A100 system
    )
    
    # Generate larger dataset for A100 training
    train_data, test_data = generator.create_train_test_split_parallel(
        train_size=100, 
        test_size=50,
        min_points=250,
        max_points=351
    )
    
    # Save with optimization
    save_dir = "/Users/jeremyfang/Downloads/image_to_graph/train_A100/data"
    generator.save_dataset_optimized(train_data, test_data, save_dir, compress=True)
    
    # Create batch visualization
    generator.create_visualization_batch(
        train_data[:9], 
        save_path=os.path.join(save_dir, "training_batch_preview.png")
    )
    generator.create_visualization_batch(
        test_data[:9], 
        save_path=os.path.join(save_dir, "testing_batch_preview.png")
    )
    
    # Print statistics
    print("\nDataset Statistics:")
    train_points = [d['n_points'] for d in train_data]
    test_points = [d['n_points'] for d in test_data]
    
    print(f"Training samples: {len(train_data)}")
    print(f"  - Point range: {min(train_points)}-{max(train_points)}")
    print(f"  - Average points: {np.mean(train_points):.1f}")
    
    print(f"Testing samples: {len(test_data)}")
    print(f"  - Point range: {min(test_points)}-{max(test_points)}")
    print(f"  - Average points: {np.mean(test_points):.1f}")
    
    # Verify data integrity
    sample_data = train_data[0]
    print(f"\nSample data verification:")
    print(f"  Image: {sample_data['image'].shape} {sample_data['image'].dtype}")
    print(f"  Points: {sample_data['points'].shape} {sample_data['points'].dtype}")
    print(f"  Adjacency: {sample_data['adjacency'].shape} {sample_data['adjacency'].dtype}")
    print(f"  Labels: {sample_data['labels'].shape} {sample_data['labels'].dtype}")
    
    print("\nA100-Optimized Data Generation Completed!")


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import os


class SyntheticDataGenerator:
    def __init__(self, img_size: int = 64, n_samples: int = 300, noise: float = 0.1):
        self.img_size = img_size
        self.n_samples = n_samples
        self.noise = noise
        self.scaler = StandardScaler()
    
    def generate_circles_data(self, n_samples: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_samples is None:
            n_samples = self.n_samples
        
        X, y = make_circles(n_samples=n_samples, noise=self.noise, factor=0.6, random_state=42)
        X = self.scaler.fit_transform(X)
        return X, y, self._create_adjacency_matrix(X)
    
    def generate_moons_data(self, n_samples: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_samples is None:
            n_samples = self.n_samples
        
        X, y = make_moons(n_samples=n_samples, noise=self.noise, random_state=42)
        X = self.scaler.fit_transform(X)
        return X, y, self._create_adjacency_matrix(X)
    
    def _create_adjacency_matrix(self, X: np.ndarray, gamma: float = 0.5) -> np.ndarray:
        n_points = X.shape[0]
        adjacency = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    dist = np.linalg.norm(X[i] - X[j])
                    adjacency[i, j] = np.exp(-gamma * dist**2)
        
        adjacency = (adjacency + adjacency.T) / 2
        np.fill_diagonal(adjacency, 0)
        
        return adjacency
    
    def points_to_image(self, X: np.ndarray) -> np.ndarray:
        X_scaled = (X - X.min()) / (X.max() - X.min())
        X_pixel = (X_scaled * (self.img_size - 1)).astype(int)
        
        image = np.zeros((self.img_size, self.img_size))
        for point in X_pixel:
            x, y = point
            x = np.clip(x, 0, self.img_size - 1)
            y = np.clip(y, 0, self.img_size - 1)
            image[y, x] = 1.0
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.img_size and 0 <= ny < self.img_size:
                        image[ny, nx] = max(image[ny, nx], 0.7)
        
        return image
    
    def generate_dataset(self, dataset_type: str, n_samples: int = None) -> Dict:
        if dataset_type == 'circles':
            X, y, adj = self.generate_circles_data(n_samples)
        elif dataset_type == 'moons':
            X, y, adj = self.generate_moons_data(n_samples)
        else:
            raise ValueError("dataset_type must be 'circles' or 'moons'")
        
        image = self.points_to_image(X)
        
        X_pixel = (X - X.min()) / (X.max() - X.min()) * (self.img_size - 1)
        
        return {
            'image': image.astype(np.float32),
            'points': X.astype(np.float32),
            'points_pixel': X_pixel.astype(np.float32),
            'labels': y.astype(np.int64),
            'adjacency': adj.astype(np.float32)
        }
    
    def create_train_test_split(self, train_size: int = 100, test_size: int = 50):
        train_data = []
        test_data = []
        
        for i in range(train_size):
            n_points = np.random.randint(250, 351)
            noise_level = np.random.uniform(0.05, 0.15)
            
            temp_generator = SyntheticDataGenerator(
                img_size=self.img_size, 
                n_samples=n_points, 
                noise=noise_level
            )
            data = temp_generator.generate_dataset('circles', n_points)
            train_data.append(data)
        
        for i in range(test_size):
            n_points = np.random.randint(250, 351)
            noise_level = np.random.uniform(0.05, 0.15)
            
            temp_generator = SyntheticDataGenerator(
                img_size=self.img_size, 
                n_samples=n_points, 
                noise=noise_level
            )
            data = temp_generator.generate_dataset('moons', n_points)
            test_data.append(data)
        
        return train_data, test_data
    
    def save_dataset(self, train_data: List[Dict], test_data: List[Dict], save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, 'train_data.npy'), train_data)
        np.save(os.path.join(save_dir, 'test_data.npy'), test_data)
        
        print(f"Dataset saved to {save_dir}")
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
    
    def visualize_sample(self, data_dict: Dict, save_path: str = None):
        plt.ioff()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(data_dict['image'], cmap='gray')
        axes[0, 0].set_title('Generated Image')
        axes[0, 0].axis('off')
        
        points = data_dict['points_pixel']
        labels = data_dict['labels']
        axes[0, 1].scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', s=20)
        axes[0, 1].set_title('Data Points')
        axes[0, 1].set_xlim(0, self.img_size)
        axes[0, 1].set_ylim(0, self.img_size)
        axes[0, 1].invert_yaxis()
        
        im = axes[1, 0].imshow(data_dict['adjacency'], cmap='hot', interpolation='nearest')
        axes[1, 0].set_title('Adjacency Matrix')
        plt.colorbar(im, ax=axes[1, 0])
        
        adj_thresholded = (data_dict['adjacency'] > 0.3).astype(int)
        axes[1, 1].imshow(adj_thresholded, cmap='binary', interpolation='nearest')
        axes[1, 1].set_title('Thresholded Adjacency (>0.3)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def main():
    generator = SyntheticDataGenerator(img_size=64, n_samples=300, noise=0.1)
    
    train_data, test_data = generator.create_train_test_split(train_size=100, test_size=50)
    
    save_dir = "/Users/jeremyfang/Downloads/image_to_graph/data"
    generator.save_dataset(train_data, test_data, save_dir)
    
    sample_data = train_data[0]
    generator.visualize_sample(sample_data, 
                             save_path="/Users/jeremyfang/Downloads/image_to_graph/sample_visualization.png")
    
    print(f"Sample data shapes:")
    print(f"Image: {sample_data['image'].shape}")
    print(f"Points: {sample_data['points'].shape}")
    print(f"Adjacency: {sample_data['adjacency'].shape}")
    print(f"Labels: {sample_data['labels'].shape}")


if __name__ == "__main__":
    main()
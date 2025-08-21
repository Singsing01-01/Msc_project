import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, List, Tuple

# 导入模型
from model_a_gnn import ModelA_GNN, ModelALoss
from model_b_similarity import ModelB_Similarity, ModelBLoss
from data_generation import SyntheticDataGenerator


class FastMoonsGeneralizationEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = SyntheticDataGenerator(img_size=64, n_samples=200, noise=0.1)
        
        # 简化的模型配置 - 减少参数数量
        self.model_config_a = {
            'input_channels': 1,
            'feature_dim': 128,  # 减少特征维度
            'max_nodes': 200,    # 减少最大节点数
            'coord_dim': 2,
            'hidden_dim': 32,    # 减少隐藏层维度
            'node_feature_dim': 32  # 保持与GNN输出一致
        }
        
        self.model_config_b = {
            'input_channels': 1,
            'feature_dim': 128,  # 减少特征维度
            'max_nodes': 200,    # 减少最大节点数
            'coord_dim': 2,
            'similarity_hidden_dim': 16  # 减少隐藏层维度
        }
        
        # 快速训练配置
        self.training_config = {
            'batch_size': 16,    # 增加批次大小
            'learning_rate': 0.002,  # 增加学习率
            'epochs': 20,        # 减少训练轮数
            'weight_decay': 1e-5
        }
        
        # 初始化模型
        self.model_a = ModelA_GNN(**self.model_config_a).to(self.device)
        self.model_b = ModelB_Similarity(**self.model_config_b).to(self.device)
        
        # 损失函数
        self.loss_a = ModelALoss(coord_weight=1.0, edge_weight=1.0, count_weight=0.1)
        self.loss_b = ModelBLoss(coord_weight=1.0, edge_weight=1.0, count_weight=0.1, similarity_weight=0.5)
        
        # 优化器
        self.optimizer_a = optim.Adam(self.model_a.parameters(), 
                                    lr=self.training_config['learning_rate'], 
                                    weight_decay=self.training_config['weight_decay'])
        self.optimizer_b = optim.Adam(self.model_b.parameters(), 
                                    lr=self.training_config['learning_rate'], 
                                    weight_decay=self.training_config['weight_decay'])
        
        # 结果存储
        self.results = {
            'model_a': {'train_losses': [], 'test_losses': [], 'test_metrics': []},
            'model_b': {'train_losses': [], 'test_losses': [], 'test_metrics': []}
        }
    
    def generate_moons_dataset(self, n_samples: int) -> List[Dict]:
        """生成双月数据集"""
        dataset = []
        
        for i in range(n_samples):
            n_points = np.random.randint(150, 201)  # 减少节点数量
            noise_level = np.random.uniform(0.05, 0.15)
            
            temp_generator = SyntheticDataGenerator(
                img_size=self.generator.img_size, 
                n_samples=n_points, 
                noise=noise_level
            )
            data = temp_generator.generate_dataset('moons', n_points)
            dataset.append(data)
        
        return dataset
    
    def prepare_batch(self, batch_data: List[Dict]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """准备批次数据"""
        batch_size = len(batch_data)
        max_nodes = self.model_config_a['max_nodes']
        
        # 准备图像数据
        images = torch.stack([torch.from_numpy(data['image']).unsqueeze(0) for data in batch_data])
        
        # 准备目标数据
        targets = {
            'points': torch.zeros(batch_size, max_nodes, 2),
            'node_masks': torch.zeros(batch_size, max_nodes, dtype=torch.bool),
            'adjacency': torch.zeros(batch_size, max_nodes, max_nodes)
        }
        
        for i, data in enumerate(batch_data):
            n_points = len(data['points'])
            n_valid = min(n_points, max_nodes)
            
            # 填充坐标
            targets['points'][i, :n_valid] = torch.from_numpy(data['points'][:n_valid])
            
            # 填充节点掩码
            targets['node_masks'][i, :n_valid] = True
            
            # 填充邻接矩阵
            adj = torch.from_numpy(data['adjacency'][:n_valid, :n_valid])
            targets['adjacency'][i, :n_valid, :n_valid] = adj
        
        return images, targets
    
    def train_model(self, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, 
                   train_loader: DataLoader, model_name: str) -> List[float]:
        """训练模型"""
        model.train()
        train_losses = []
        
        for epoch in range(self.training_config['epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data in train_loader:
                images, targets = self.prepare_batch(batch_data)
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                optimizer.zero_grad()
                
                predictions = model(images, targets['node_masks'])
                loss_dict = loss_fn(predictions, targets)
                total_loss = sum(loss_dict.values())
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:  # 更频繁的打印
                print(f"{model_name} Epoch {epoch+1}/{self.training_config['epochs']}, Loss: {avg_loss:.4f}")
        
        return train_losses
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, model_name: str) -> Tuple[List[float], List[Dict]]:
        """评估模型"""
        model.eval()
        test_losses = []
        test_metrics = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                images, targets = self.prepare_batch(batch_data)
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                predictions = model(images, targets['node_masks'])
                
                # 计算损失
                if model_name == 'model_a':
                    loss_dict = self.loss_a(predictions, targets)
                else:
                    loss_dict = self.loss_b(predictions, targets)
                
                total_loss = sum(loss_dict.values())
                test_losses.append(total_loss.item())
                
                # 计算指标
                metrics = self.calculate_metrics(predictions, targets)
                test_metrics.append(metrics)
        
        return test_losses, test_metrics
    
    def calculate_metrics(self, predictions: Dict[str, torch.Tensor], 
                         targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算评估指标"""
        batch_size = predictions['predicted_coords'].shape[0]
        metrics = {
            'coord_mse': 0.0,
            'adjacency_accuracy': 0.0,
            'node_count_mae': 0.0
        }
        
        for b in range(batch_size):
            # 坐标MSE
            mask = targets['node_masks'][b]
            if mask.sum() > 0:
                coord_mse = torch.mean((predictions['predicted_coords'][b][mask] - 
                                      targets['points'][b][mask]) ** 2)
                metrics['coord_mse'] += coord_mse.item()
            
            # 邻接矩阵准确率
            adj_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            if adj_mask.sum() > 0:
                pred_adj = (predictions['adjacency_matrix'][b] > 0.5).float()
                true_adj = targets['adjacency'][b]
                accuracy = torch.mean((pred_adj[adj_mask] == true_adj[adj_mask]).float())
                metrics['adjacency_accuracy'] += accuracy.item()
            
            # 节点数量MAE
            pred_count = predictions['node_counts'][b].item()
            true_count = mask.sum().item()
            metrics['node_count_mae'] += abs(pred_count - true_count)
        
        # 平均化
        metrics = {k: v / batch_size for k, v in metrics.items()}
        return metrics
    
    def run_generalization_test(self):
        """运行泛化能力测试"""
        print("开始快速双月数据集泛化能力测试...")
        print(f"使用设备: {self.device}")
        print(f"模型A参数数量: {self.model_a.count_parameters():,}")
        print(f"模型B参数数量: {self.model_b.count_parameters():,}")
        
        # 生成训练数据（圆形数据）
        print("\n生成训练数据（圆形数据集）...")
        train_data = []
        for i in range(50):  # 减少训练样本数量
            n_points = np.random.randint(150, 201)  # 减少节点数量
            noise_level = np.random.uniform(0.05, 0.15)
            
            temp_generator = SyntheticDataGenerator(
                img_size=self.generator.img_size, 
                n_samples=n_points, 
                noise=noise_level
            )
            data = temp_generator.generate_dataset('circles', n_points)
            train_data.append(data)
        
        # 生成测试数据（双月数据）
        print("生成测试数据（双月数据集）...")
        test_data = self.generate_moons_dataset(25)  # 减少测试样本数量
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=self.training_config['batch_size'], shuffle=True, collate_fn=lambda x: x)
        test_loader = DataLoader(test_data, batch_size=self.training_config['batch_size'], shuffle=False, collate_fn=lambda x: x)
        
        # 训练和评估模型A
        print("\n=== 训练模型A ===")
        start_time = time.time()
        self.results['model_a']['train_losses'] = self.train_model(
            self.model_a, self.optimizer_a, self.loss_a, train_loader, "Model A"
        )
        train_time_a = time.time() - start_time
        
        print(f"模型A训练完成，耗时: {train_time_a:.2f}秒")
        
        print("\n=== 评估模型A在双月数据集上的表现 ===")
        test_losses_a, test_metrics_a = self.evaluate_model(self.model_a, test_loader, "Model A")
        self.results['model_a']['test_losses'] = test_losses_a
        self.results['model_a']['test_metrics'] = test_metrics_a
        
        # 训练和评估模型B
        print("\n=== 训练模型B ===")
        start_time = time.time()
        self.results['model_b']['train_losses'] = self.train_model(
            self.model_b, self.optimizer_b, self.loss_b, train_loader, "Model B"
        )
        train_time_b = time.time() - start_time
        
        print(f"模型B训练完成，耗时: {train_time_b:.2f}秒")
        
        print("\n=== 评估模型B在双月数据集上的表现 ===")
        test_losses_b, test_metrics_b = self.evaluate_model(self.model_b, test_loader, "Model B")
        self.results['model_b']['test_losses'] = test_losses_b
        self.results['model_b']['test_metrics'] = test_metrics_b
        
        # 分析结果
        self.analyze_results()
        
        # 可视化结果
        self.visualize_results()
        
        # 保存结果
        self.save_results()
    
    def analyze_results(self):
        """分析结果"""
        print("\n" + "="*60)
        print("快速双月数据集泛化能力分析结果")
        print("="*60)
        
        # 计算平均指标
        for model_name in ['model_a', 'model_b']:
            metrics = self.results[model_name]['test_metrics']
            
            avg_metrics = {
                'coord_mse': np.mean([m['coord_mse'] for m in metrics]),
                'adjacency_accuracy': np.mean([m['adjacency_accuracy'] for m in metrics]),
                'node_count_mae': np.mean([m['node_count_mae'] for m in metrics])
            }
            
            std_metrics = {
                'coord_mse': np.std([m['coord_mse'] for m in metrics]),
                'adjacency_accuracy': np.std([m['adjacency_accuracy'] for m in metrics]),
                'node_count_mae': np.std([m['node_count_mae'] for m in metrics])
            }
            
            print(f"\n{model_name.upper()} 在双月数据集上的表现:")
            print(f"  坐标MSE: {avg_metrics['coord_mse']:.4f} ± {std_metrics['coord_mse']:.4f}")
            print(f"  邻接矩阵准确率: {avg_metrics['adjacency_accuracy']:.4f} ± {std_metrics['adjacency_accuracy']:.4f}")
            print(f"  节点数量MAE: {avg_metrics['node_count_mae']:.2f} ± {std_metrics['node_count_mae']:.2f}")
        
        # 比较两个模型
        print("\n模型比较:")
        model_a_adj_acc = np.mean([m['adjacency_accuracy'] for m in self.results['model_a']['test_metrics']])
        model_b_adj_acc = np.mean([m['adjacency_accuracy'] for m in self.results['model_b']['test_metrics']])
        
        if model_a_adj_acc > model_b_adj_acc:
            print(f"模型A在邻接矩阵预测上表现更好: {model_a_adj_acc:.4f} vs {model_b_adj_acc:.4f}")
        else:
            print(f"模型B在邻接矩阵预测上表现更好: {model_b_adj_acc:.4f} vs {model_a_adj_acc:.4f}")
    
    def visualize_results(self):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 训练损失
        axes[0, 0].plot(self.results['model_a']['train_losses'], label='Model A', color='blue')
        axes[0, 0].plot(self.results['model_b']['train_losses'], label='Model B', color='red')
        axes[0, 0].set_title('训练损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 测试损失分布
        axes[0, 1].hist(self.results['model_a']['test_losses'], alpha=0.7, label='Model A', color='blue', bins=10)
        axes[0, 1].hist(self.results['model_b']['test_losses'], alpha=0.7, label='Model B', color='red', bins=10)
        axes[0, 1].set_title('测试损失分布')
        axes[0, 1].set_xlabel('Test Loss')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 邻接矩阵准确率比较
        model_a_adj_acc = [m['adjacency_accuracy'] for m in self.results['model_a']['test_metrics']]
        model_b_adj_acc = [m['adjacency_accuracy'] for m in self.results['model_b']['test_metrics']]
        
        axes[1, 0].boxplot([model_a_adj_acc, model_b_adj_acc], labels=['Model A', 'Model B'])
        axes[1, 0].set_title('邻接矩阵准确率比较')
        axes[1, 0].set_ylabel('Adjacency Accuracy')
        axes[1, 0].grid(True)
        
        # 坐标MSE比较
        model_a_coord_mse = [m['coord_mse'] for m in self.results['model_a']['test_metrics']]
        model_b_coord_mse = [m['coord_mse'] for m in self.results['model_b']['test_metrics']]
        
        axes[1, 1].boxplot([model_a_coord_mse, model_b_coord_mse], labels=['Model A', 'Model B'])
        axes[1, 1].set_title('坐标MSE比较')
        axes[1, 1].set_ylabel('Coordinate MSE')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('fast_moons_generalization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """保存结果"""
        import json
        
        # 保存详细结果
        results_summary = {
            'model_a': {
                'avg_coord_mse': np.mean([m['coord_mse'] for m in self.results['model_a']['test_metrics']]),
                'avg_adjacency_accuracy': np.mean([m['adjacency_accuracy'] for m in self.results['model_a']['test_metrics']]),
                'avg_node_count_mae': np.mean([m['node_count_mae'] for m in self.results['model_a']['test_metrics']])
            },
            'model_b': {
                'avg_coord_mse': np.mean([m['coord_mse'] for m in self.results['model_b']['test_metrics']]),
                'avg_adjacency_accuracy': np.mean([m['adjacency_accuracy'] for m in self.results['model_b']['test_metrics']]),
                'avg_node_count_mae': np.mean([m['node_count_mae'] for m in self.results['model_b']['test_metrics']])
            }
        }
        
        with open('fast_moons_generalization_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n结果已保存到:")
        print(f"- fast_moons_generalization_results.png (可视化图表)")
        print(f"- fast_moons_generalization_results.json (详细数据)")


def main():
    """主函数"""
    evaluator = FastMoonsGeneralizationEvaluator()
    evaluator.run_generalization_test()


if __name__ == "__main__":
    main() 
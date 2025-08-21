import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入模型和评估框架
from model_a_gnn import ModelA_GNN, ModelALoss
from model_b_similarity import ModelB_Similarity, ModelBLoss
from data_generation import SyntheticDataGenerator
from evaluation_framework import GraphPostProcessor, SpectralClusteringEvaluator, ModularityCalculator


class LightweightCompleteTest:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 保持完整的模型配置，但使用适中的参数
        self.model_config_a = {
            'input_channels': 1,
            'feature_dim': 128,    # 适中的特征维度
            'max_nodes': 150,      # 适中的最大节点数
            'coord_dim': 2,
            'hidden_dim': 32,      # 适中的隐藏层维度
            'node_feature_dim': 32 # 适中的节点特征维度
        }
        
        self.model_config_b = {
            'input_channels': 1,
            'feature_dim': 128,    # 适中的特征维度
            'max_nodes': 150,      # 适中的最大节点数
            'coord_dim': 2,
            'similarity_hidden_dim': 16  # 适中的隐藏层维度
        }
        
        # 轻量级训练配置
        self.training_config = {
            'batch_size': 8,       # 小批次
            'learning_rate': 0.001, # 标准学习率
            'epochs': 10,          # 较少的训练轮数
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
        
        # 评估组件
        self.graph_processor = GraphPostProcessor(k_top_edges=8)
        self.spectral_evaluator = SpectralClusteringEvaluator(n_clusters=2, random_state=42)
        self.modularity_calculator = ModularityCalculator()
        
        # 结果存储
        self.results = {
            'model_a': {'metrics': [], 'inference_times': [], 'train_losses': []},
            'model_b': {'metrics': [], 'inference_times': [], 'train_losses': []}
        }
    
    def generate_dataset(self, n_samples: int, dataset_type: str) -> List[Dict]:
        """生成数据集"""
        dataset = []
        
        for i in range(n_samples):
            n_points = np.random.randint(100, 151)  # 适中的节点数量
            noise_level = np.random.uniform(0.05, 0.15)
            
            temp_generator = SyntheticDataGenerator(
                img_size=64, 
                n_samples=n_points, 
                noise=noise_level
            )
            data = temp_generator.generate_dataset(dataset_type, n_points)
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
            'adjacency': torch.zeros(batch_size, max_nodes, max_nodes),
            'labels': []
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
            
            # 存储真实标签
            targets['labels'].append(data['labels'][:n_valid])
        
        return images, targets
    
    def train_model(self, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, 
                   train_loader: DataLoader, model_name: str) -> List[float]:
        """训练模型"""
        model.train()
        train_losses = []
        
        total_batches = len(train_loader)
        print(f"开始训练{model_name}，共{self.training_config['epochs']}个epoch，每epoch {total_batches}个batch")
        
        for epoch in range(self.training_config['epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            print(f"\n{model_name} Epoch {epoch+1}/{self.training_config['epochs']}:")
            
            for batch_idx, batch_data in enumerate(train_loader):
                images, targets = self.prepare_batch(batch_data)
                images = images.to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
                
                optimizer.zero_grad()
                
                predictions = model(images, targets['node_masks'])
                loss_dict = loss_fn(predictions, targets)
                total_loss = sum(loss_dict.values())
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # 显示每个batch的进度
                if (batch_idx + 1) % 2 == 0 or batch_idx == total_batches - 1:
                    current_loss = total_loss.item()
                    progress = (batch_idx + 1) / total_batches * 100
                    print(f"  Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) - Loss: {current_loss:.4f}")
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            print(f"  Epoch {epoch+1} 完成 - 平均Loss: {avg_loss:.4f}")
            
            # 显示训练进度条
            progress_bar = "█" * int((epoch + 1) / self.training_config['epochs'] * 20) + "░" * (20 - int((epoch + 1) / self.training_config['epochs'] * 20))
            print(f"  训练进度: [{progress_bar}] {epoch+1}/{self.training_config['epochs']}")
        
        print(f"\n{model_name} 训练完成！")
        return train_losses
    
    def evaluate_model_on_moons(self, model: nn.Module, test_data: List[Dict], model_name: str):
        """在双月数据集上评估模型"""
        print(f"\n=== 评估{model_name}在双月数据集上的表现 ===")
        print(f"共{len(test_data)}个测试样本")
        
        all_metrics = []
        all_inference_times = []
        
        for i, data in enumerate(test_data):
            # 显示进度
            progress = (i + 1) / len(test_data) * 100
            progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
            print(f"评估进度: [{progress_bar}] {i+1}/{len(test_data)} ({progress:.1f}%)")
            
            # 准备数据
            image = torch.from_numpy(data['image']).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 64, 64]
            n_points = len(data['points'])
            n_valid = min(n_points, self.model_config_a['max_nodes'])
            
            node_mask = torch.zeros(self.model_config_a['max_nodes'], dtype=torch.bool)
            node_mask[:n_valid] = True
            node_mask = node_mask.to(self.device)
            
            true_labels = data['labels'][:n_valid]
            
            # 评估
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                predictions = model(image, node_mask.unsqueeze(0))
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000
                
                # 获取邻接矩阵
                adjacency = predictions['adjacency_matrix'][0].cpu().numpy()
                
                # 处理邻接矩阵
                processed_adjacency = self.graph_processor.process(adjacency)
                
                # 谱聚类
                predicted_labels = self.spectral_evaluator.cluster(processed_adjacency)
                
                # 确保标签长度匹配
                n_valid = min(len(predicted_labels), len(true_labels))
                predicted_labels = predicted_labels[:n_valid]
                true_labels = true_labels[:n_valid]
                
                # 计算指标
                clustering_metrics = self.spectral_evaluator.evaluate_clustering(predicted_labels, true_labels)
                modularity = self.modularity_calculator.calculate_modularity(processed_adjacency, predicted_labels)
                
                metrics = {
                    'ari': clustering_metrics['ari'],
                    'nmi': clustering_metrics['nmi'],
                    'modularity': modularity,
                    'inference_time_ms': inference_time
                }
                
                all_metrics.append(metrics)
                all_inference_times.append(inference_time)
                
                print(f"  样本{i+1} - ARI: {metrics['ari']:.3f}, NMI: {metrics['nmi']:.3f}, 时间: {inference_time:.1f}ms")
        
        # 计算平均指标
        avg_metrics = {
            'ari': np.mean([m['ari'] for m in all_metrics]),
            'nmi': np.mean([m['nmi'] for m in all_metrics]),
            'modularity': np.mean([m['modularity'] for m in all_metrics]),
            'inference_time_ms': np.mean(all_inference_times)
        }
        
        self.results[model_name]['metrics'] = all_metrics
        self.results[model_name]['inference_times'] = all_inference_times
        
        print(f"✅ {model_name} 评估完成")
        print(f"  平均ARI: {avg_metrics['ari']:.3f}")
        print(f"  平均NMI: {avg_metrics['nmi']:.3f}")
        print(f"  平均Modularity: {avg_metrics['modularity']:.3f}")
        print(f"  平均推理时间: {avg_metrics['inference_time_ms']:.1f}ms")
        
        return avg_metrics
    
    def run_lightweight_complete_test(self):
        """运行轻量级完整测试"""
        print("="*60)
        print("轻量级完整双月数据集泛化能力测试")
        print("="*60)
        print(f"使用设备: {self.device}")
        print(f"模型A参数数量: {self.model_a.count_parameters():,}")
        print(f"模型B参数数量: {self.model_b.count_parameters():,}")
        
        # 生成训练数据（圆形数据）
        print("\n📊 生成训练数据（圆形数据集）...")
        train_data = self.generate_dataset(20, 'circles')  # 20个训练样本
        print("✅ 训练数据生成完成")
        
        # 生成测试数据（双月数据）
        print("\n📊 生成测试数据（双月数据集）...")
        test_data = self.generate_dataset(12, 'moons')  # 12个测试样本
        print("✅ 测试数据生成完成")
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=self.training_config['batch_size'], 
                                shuffle=True, collate_fn=lambda x: x)
        
        print(f"\n🚀 开始轻量级训练...")
        print(f"训练配置: {self.training_config['epochs']} epochs, batch_size={self.training_config['batch_size']}")
        
        # 训练模型A
        print("\n" + "="*40)
        print("🔧 训练模型A (GNN-based)")
        print("="*40)
        start_time = time.time()
        train_losses_a = self.train_model(self.model_a, self.optimizer_a, self.loss_a, train_loader, "Model A")
        train_time_a = time.time() - start_time
        print(f"✅ 模型A训练完成，耗时: {train_time_a:.2f}秒")
        self.results['model_a']['train_losses'] = train_losses_a
        
        # 训练模型B
        print("\n" + "="*40)
        print("🔧 训练模型B (Similarity-based)")
        print("="*40)
        start_time = time.time()
        train_losses_b = self.train_model(self.model_b, self.optimizer_b, self.loss_b, train_loader, "Model B")
        train_time_b = time.time() - start_time
        print(f"✅ 模型B训练完成，耗时: {train_time_b:.2f}秒")
        self.results['model_b']['train_losses'] = train_losses_b
        
        print(f"\n📈 开始评估阶段...")
        
        # 评估模型A
        metrics_a = self.evaluate_model_on_moons(self.model_a, test_data, 'model_a')
        
        # 评估模型B
        metrics_b = self.evaluate_model_on_moons(self.model_b, test_data, 'model_b')
        
        # 分析结果
        self.analyze_results()
        
        # 保存结果
        self.save_results()
    
    def analyze_results(self):
        """分析结果"""
        print("\n" + "="*60)
        print("轻量级完整泛化能力分析结果")
        print("="*60)
        
        # 计算统计信息
        for model_name in ['model_a', 'model_b']:
            metrics = self.results[model_name]['metrics']
            
            ari_values = [m['ari'] for m in metrics]
            nmi_values = [m['nmi'] for m in metrics]
            modularity_values = [m['modularity'] for m in metrics]
            inference_times = [m['inference_time_ms'] for m in metrics]
            
            print(f"\n{model_name.upper()} 统计信息:")
            print(f"  ARI: {np.mean(ari_values):.3f} ± {np.std(ari_values):.3f}")
            print(f"  NMI: {np.mean(nmi_values):.3f} ± {np.std(nmi_values):.3f}")
            print(f"  Modularity: {np.mean(modularity_values):.3f} ± {np.std(modularity_values):.3f}")
            print(f"  推理时间: {np.mean(inference_times):.1f} ± {np.std(inference_times):.1f} ms")
        
        # 比较两个模型
        print("\n模型比较:")
        model_a_ari = np.mean([m['ari'] for m in self.results['model_a']['metrics']])
        model_b_ari = np.mean([m['ari'] for m in self.results['model_b']['metrics']])
        
        if model_a_ari > model_b_ari:
            print(f"✅ 模型A在ARI上表现更好: {model_a_ari:.3f} vs {model_b_ari:.3f}")
        else:
            print(f"✅ 模型B在ARI上表现更好: {model_b_ari:.3f} vs {model_a_ari:.3f}")
        
        # 泛化能力评估
        print("\n泛化能力评估:")
        if model_a_ari > 0.3:
            print(f"✅ 模型A在双月数据集上显示出良好的泛化能力 (ARI > 0.3)")
        elif model_a_ari > 0.2:
            print(f"⚠️  模型A在双月数据集上显示出一定的泛化能力 (0.2 < ARI ≤ 0.3)")
        else:
            print(f"❌ 模型A在双月数据集上泛化能力有限 (ARI ≤ 0.2)")
        
        if model_b_ari > 0.3:
            print(f"✅ 模型B在双月数据集上显示出良好的泛化能力 (ARI > 0.3)")
        elif model_b_ari > 0.2:
            print(f"⚠️  模型B在双月数据集上显示出一定的泛化能力 (0.2 < ARI ≤ 0.3)")
        else:
            print(f"❌ 模型B在双月数据集上泛化能力有限 (ARI ≤ 0.2)")
    
    def save_results(self):
        """保存结果"""
        # 创建结果DataFrame
        results_data = []
        
        for model_name in ['model_a', 'model_b']:
            metrics = self.results[model_name]['metrics']
            
            avg_ari = np.mean([m['ari'] for m in metrics])
            avg_nmi = np.mean([m['nmi'] for m in metrics])
            avg_modularity = np.mean([m['modularity'] for m in metrics])
            avg_inference_time = np.mean([m['inference_time_ms'] for m in metrics])
            
            # 判断是否达到目标
            meets_ari_target = avg_ari > 0.3
            beats_sklearn = avg_inference_time < 100
            
            results_data.append({
                'Method': model_name.replace('_', ' ').title(),
                'ARI': avg_ari,
                'NMI': avg_nmi,
                'Modularity': avg_modularity,
                'Inference_Time_ms': avg_inference_time,
                'Meets_ARI_Target': meets_ari_target,
                'Beats_Sklearn': beats_sklearn,
                'Description': f'Lightweight complete {model_name.replace("_", " ").title()} on moons dataset'
            })
        
        # 保存到CSV
        df = pd.DataFrame(results_data)
        df.to_csv('lightweight_complete_moons_results.csv', index=False)
        
        print(f"\n结果已保存到 lightweight_complete_moons_results.csv")
        print(df.to_string(index=False))


def main():
    """主函数"""
    evaluator = LightweightCompleteTest()
    evaluator.run_lightweight_complete_test()


if __name__ == "__main__":
    main() 
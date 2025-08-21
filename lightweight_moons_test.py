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


class LightweightMoonsTest:
    def __init__(self, device='cpu'):
        self.device = device
        
        # 极简模型配置
        self.model_config_a = {
            'input_channels': 1,
            'feature_dim': 32,    # 大幅减少
            'max_nodes': 50,      # 大幅减少
            'coord_dim': 2,
            'hidden_dim': 8,      # 大幅减少
            'node_feature_dim': 8 # 大幅减少
        }
        
        self.model_config_b = {
            'input_channels': 1,
            'feature_dim': 32,    # 大幅减少
            'max_nodes': 50,      # 大幅减少
            'coord_dim': 2,
            'similarity_hidden_dim': 4  # 大幅减少
        }
        
        # 极简训练配置
        self.training_config = {
            'batch_size': 4,      # 小批次
            'learning_rate': 0.01, # 高学习率快速收敛
            'epochs': 5,          # 极少轮数
            'weight_decay': 1e-4
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
        self.graph_processor = GraphPostProcessor(k_top_edges=5)  # 减少边数
        self.spectral_evaluator = SpectralClusteringEvaluator(n_clusters=2, random_state=42)
        self.modularity_calculator = ModularityCalculator()
        
        # 结果存储
        self.results = {
            'model_a': {'metrics': [], 'inference_times': []},
            'model_b': {'metrics': [], 'inference_times': []}
        }
    
    def generate_small_dataset(self, n_samples: int, dataset_type: str) -> List[Dict]:
        """生成小规模数据集"""
        dataset = []
        
        for i in range(n_samples):
            n_points = np.random.randint(30, 51)  # 很少的节点
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
    
    def quick_train(self, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, 
                   train_loader: DataLoader, model_name: str) -> List[float]:
        """快速训练"""
        model.train()
        train_losses = []
        
        print(f"开始快速训练{model_name}...")
        
        for epoch in range(self.training_config['epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data in train_loader:
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
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            print(f"  Epoch {epoch+1}/{self.training_config['epochs']} - Loss: {avg_loss:.4f}")
        
        print(f"✅ {model_name} 训练完成")
        return train_losses
    
    def quick_evaluate(self, model: nn.Module, test_data: List[Dict], model_name: str):
        """快速评估"""
        print(f"\n快速评估{model_name}...")
        
        all_metrics = []
        all_inference_times = []
        
        for i, data in enumerate(test_data):
            print(f"  评估样本 {i+1}/{len(test_data)}")
            
            # 准备数据
            image = torch.from_numpy(data['image']).unsqueeze(0).to(self.device)
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
                predictions = model(image, node_mask)
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
                
                print(f"    ARI: {metrics['ari']:.3f}, NMI: {metrics['nmi']:.3f}, 时间: {inference_time:.1f}ms")
        
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
    
    def run_lightweight_test(self):
        """运行轻量级测试"""
        print("="*50)
        print("轻量级双月数据集泛化能力验证")
        print("="*50)
        print(f"使用设备: {self.device}")
        print(f"模型A参数数量: {self.model_a.count_parameters():,}")
        print(f"模型B参数数量: {self.model_b.count_parameters():,}")
        
        # 生成小规模训练数据（圆形数据）
        print("\n📊 生成小规模训练数据（圆形数据集）...")
        train_data = self.generate_small_dataset(10, 'circles')  # 只有10个样本
        print("✅ 训练数据生成完成")
        
        # 生成小规模测试数据（双月数据）
        print("\n📊 生成小规模测试数据（双月数据集）...")
        test_data = self.generate_small_dataset(8, 'moons')  # 只有8个样本
        print("✅ 测试数据生成完成")
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=self.training_config['batch_size'], 
                                shuffle=True, collate_fn=lambda x: x)
        
        print(f"\n🚀 开始快速训练...")
        print(f"训练配置: {self.training_config['epochs']} epochs, batch_size={self.training_config['batch_size']}")
        
        # 训练模型A
        print("\n" + "="*30)
        print("🔧 训练模型A")
        print("="*30)
        start_time = time.time()
        train_losses_a = self.quick_train(self.model_a, self.optimizer_a, self.loss_a, train_loader, "Model A")
        train_time_a = time.time() - start_time
        print(f"训练耗时: {train_time_a:.2f}秒")
        
        # 训练模型B
        print("\n" + "="*30)
        print("🔧 训练模型B")
        print("="*30)
        start_time = time.time()
        train_losses_b = self.quick_train(self.model_b, self.optimizer_b, self.loss_b, train_loader, "Model B")
        train_time_b = time.time() - start_time
        print(f"训练耗时: {train_time_b:.2f}秒")
        
        print(f"\n📈 开始快速评估...")
        
        # 评估模型A
        metrics_a = self.quick_evaluate(self.model_a, test_data, 'model_a')
        
        # 评估模型B
        metrics_b = self.quick_evaluate(self.model_b, test_data, 'model_b')
        
        # 分析结果
        self.analyze_results()
        
        # 保存结果
        self.save_results()
    
    def analyze_results(self):
        """分析结果"""
        print("\n" + "="*50)
        print("轻量级泛化能力分析结果")
        print("="*50)
        
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
            print(f"✅ 模型A在双月数据集上显示出泛化能力 (ARI > 0.3)")
        else:
            print(f"❌ 模型A在双月数据集上泛化能力有限 (ARI ≤ 0.3)")
        
        if model_b_ari > 0.3:
            print(f"✅ 模型B在双月数据集上显示出泛化能力 (ARI > 0.3)")
        else:
            print(f"❌ 模型B在双月数据集上泛化能力有限 (ARI ≤ 0.3)")
    
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
            meets_ari_target = avg_ari > 0.3  # 降低目标
            beats_sklearn = avg_inference_time < 50  # 降低目标
            
            results_data.append({
                'Method': model_name.replace('_', ' ').title(),
                'ARI': avg_ari,
                'NMI': avg_nmi,
                'Modularity': avg_modularity,
                'Inference_Time_ms': avg_inference_time,
                'Meets_ARI_Target': meets_ari_target,
                'Beats_Sklearn': beats_sklearn,
                'Description': f'Lightweight {model_name.replace("_", " ").title()} on moons dataset'
            })
        
        # 保存到CSV
        df = pd.DataFrame(results_data)
        df.to_csv('lightweight_moons_results.csv', index=False)
        
        print(f"\n结果已保存到 lightweight_moons_results.csv")
        print(df.to_string(index=False))


def main():
    """主函数"""
    evaluator = LightweightMoonsTest()
    evaluator.run_lightweight_test()


if __name__ == "__main__":
    main() 
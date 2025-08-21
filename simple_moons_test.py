import torch
import torch.nn as nn
import torch.optim as optim
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


def simple_moons_test():
    """最简单的双月数据集泛化能力测试"""
    device = 'cpu'
    print("="*50)
    print("最简单的双月数据集泛化能力测试")
    print("="*50)
    print(f"使用设备: {device}")
    
    # 极简模型配置
    model_config_a = {
        'input_channels': 1,
        'feature_dim': 64,    # 使用标准大小
        'max_nodes': 30,      # 极少的节点
        'coord_dim': 2,
        'hidden_dim': 8,      # 极小的隐藏层
        'node_feature_dim': 8 # 极小的特征维度
    }
    
    model_config_b = {
        'input_channels': 1,
        'feature_dim': 64,    # 使用标准大小
        'max_nodes': 30,      # 极少的节点
        'coord_dim': 2,
        'similarity_hidden_dim': 4  # 极小的隐藏层
    }
    
    # 初始化模型
    print("初始化模型...")
    model_a = ModelA_GNN(**model_config_a).to(device)
    model_b = ModelB_Similarity(**model_config_b).to(device)
    
    print(f"模型A参数数量: {model_a.count_parameters():,}")
    print(f"模型B参数数量: {model_b.count_parameters():,}")
    
    # 生成极少量数据
    print("\n生成极少量数据...")
    generator = SyntheticDataGenerator(img_size=64, n_samples=30, noise=0.1)
    
    # 训练数据（圆形）
    train_data = []
    for i in range(5):  # 只有5个训练样本
        n_points = np.random.randint(20, 31)  # 20-30个节点
        noise_level = np.random.uniform(0.05, 0.15)
        
        temp_generator = SyntheticDataGenerator(
            img_size=generator.img_size, 
            n_samples=n_points, 
            noise=noise_level
        )
        data = temp_generator.generate_dataset('circles', n_points)
        train_data.append(data)
    
    # 测试数据（双月）
    test_data = []
    for i in range(3):  # 只有3个测试样本
        n_points = np.random.randint(20, 31)  # 20-30个节点
        noise_level = np.random.uniform(0.05, 0.15)
        
        temp_generator = SyntheticDataGenerator(
            img_size=generator.img_size, 
            n_samples=n_points, 
            noise=noise_level
        )
        data = temp_generator.generate_dataset('moons', n_points)
        test_data.append(data)
    
    print(f"训练样本: {len(train_data)}")
    print(f"测试样本: {len(test_data)}")
    
    # 准备训练数据
    print("\n准备训练数据...")
    batch_size = len(train_data)
    max_nodes = model_config_a['max_nodes']
    
    # 准备图像数据
    images = torch.stack([torch.from_numpy(data['image']).unsqueeze(0) for data in train_data])
    
    # 准备目标数据
    targets = {
        'points': torch.zeros(batch_size, max_nodes, 2),
        'node_masks': torch.zeros(batch_size, max_nodes, dtype=torch.bool),
        'adjacency': torch.zeros(batch_size, max_nodes, max_nodes)
    }
    
    for i, data in enumerate(train_data):
        n_points = len(data['points'])
        n_valid = min(n_points, max_nodes)
        
        # 填充坐标
        targets['points'][i, :n_valid] = torch.from_numpy(data['points'][:n_valid])
        
        # 填充节点掩码
        targets['node_masks'][i, :n_valid] = True
        
        # 填充邻接矩阵
        adj = torch.from_numpy(data['adjacency'][:n_valid, :n_valid])
        targets['adjacency'][i, :n_valid, :n_valid] = adj
    
    print("数据准备完成")
    
    # 快速训练模型A
    print("\n" + "="*30)
    print("快速训练模型A")
    print("="*30)
    
    model_a.train()
    optimizer_a = optim.Adam(model_a.parameters(), lr=0.01)
    loss_a = ModelALoss()
    
    start_time = time.time()
    for epoch in range(3):  # 只训练3个epoch
        optimizer_a.zero_grad()
        
        predictions = model_a(images, targets['node_masks'])
        loss_dict = loss_a(predictions, targets)
        total_loss = sum(loss_dict.values())
        
        total_loss.backward()
        optimizer_a.step()
        
        print(f"  Epoch {epoch+1}/3 - Loss: {total_loss.item():.4f}")
    
    train_time_a = time.time() - start_time
    print(f"模型A训练完成，耗时: {train_time_a:.2f}秒")
    
    # 快速训练模型B
    print("\n" + "="*30)
    print("快速训练模型B")
    print("="*30)
    
    model_b.train()
    optimizer_b = optim.Adam(model_b.parameters(), lr=0.01)
    loss_b = ModelBLoss()
    
    start_time = time.time()
    for epoch in range(3):  # 只训练3个epoch
        optimizer_b.zero_grad()
        
        predictions = model_b(images, targets['node_masks'])
        loss_dict = loss_b(predictions, targets)
        total_loss = sum(loss_dict.values())
        
        total_loss.backward()
        optimizer_b.step()
        
        print(f"  Epoch {epoch+1}/3 - Loss: {total_loss.item():.4f}")
    
    train_time_b = time.time() - start_time
    print(f"模型B训练完成，耗时: {train_time_b:.2f}秒")
    
    # 评估组件
    graph_processor = GraphPostProcessor(k_top_edges=3)
    spectral_evaluator = SpectralClusteringEvaluator(n_clusters=2, random_state=42)
    modularity_calculator = ModularityCalculator()
    
    # 评估模型A
    print("\n" + "="*30)
    print("评估模型A在双月数据集上")
    print("="*30)
    
    model_a.eval()
    results_a = []
    
    with torch.no_grad():
        for i, data in enumerate(test_data):
            print(f"  评估样本 {i+1}/{len(test_data)}")
            
            # 准备数据
            image = torch.from_numpy(data['image']).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 64, 64]
            print(f"    图像形状: {image.shape}")
            n_points = len(data['points'])
            n_valid = min(n_points, max_nodes)
            
            node_mask = torch.zeros(max_nodes, dtype=torch.bool)
            node_mask[:n_valid] = True
            node_mask = node_mask.to(device)
            print(f"    节点掩码形状: {node_mask.unsqueeze(0).shape}")
            
            true_labels = data['labels'][:n_valid]
            
            # 推理
            start_time = time.time()
            predictions = model_a(image, node_mask.unsqueeze(0))  # 添加batch维度
            inference_time = (time.time() - start_time) * 1000
            
            # 获取邻接矩阵
            adjacency = predictions['adjacency_matrix'][0].cpu().numpy()
            
            # 处理邻接矩阵
            processed_adjacency = graph_processor.process(adjacency)
            
            # 谱聚类
            predicted_labels = spectral_evaluator.cluster(processed_adjacency)
            
            # 确保标签长度匹配
            n_valid = min(len(predicted_labels), len(true_labels))
            predicted_labels = predicted_labels[:n_valid]
            true_labels = true_labels[:n_valid]
            
            # 计算指标
            clustering_metrics = spectral_evaluator.evaluate_clustering(predicted_labels, true_labels)
            modularity = modularity_calculator.calculate_modularity(processed_adjacency, predicted_labels)
            
            results_a.append({
                'ari': clustering_metrics['ari'],
                'nmi': clustering_metrics['nmi'],
                'modularity': modularity,
                'inference_time_ms': inference_time
            })
            
            print(f"    ARI: {clustering_metrics['ari']:.3f}, NMI: {clustering_metrics['nmi']:.3f}, 时间: {inference_time:.1f}ms")
    
    # 评估模型B
    print("\n" + "="*30)
    print("评估模型B在双月数据集上")
    print("="*30)
    
    model_b.eval()
    results_b = []
    
    with torch.no_grad():
        for i, data in enumerate(test_data):
            print(f"  评估样本 {i+1}/{len(test_data)}")
            
            # 准备数据
            image = torch.from_numpy(data['image']).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 64, 64]
            n_points = len(data['points'])
            n_valid = min(n_points, max_nodes)
            
            node_mask = torch.zeros(max_nodes, dtype=torch.bool)
            node_mask[:n_valid] = True
            node_mask = node_mask.to(device)
            
            true_labels = data['labels'][:n_valid]
            
            # 推理
            start_time = time.time()
            predictions = model_b(image, node_mask.unsqueeze(0))  # 添加batch维度
            inference_time = (time.time() - start_time) * 1000
            
            # 获取邻接矩阵
            adjacency = predictions['adjacency_matrix'][0].cpu().numpy()
            
            # 处理邻接矩阵
            processed_adjacency = graph_processor.process(adjacency)
            
            # 谱聚类
            predicted_labels = spectral_evaluator.cluster(processed_adjacency)
            
            # 确保标签长度匹配
            n_valid = min(len(predicted_labels), len(true_labels))
            predicted_labels = predicted_labels[:n_valid]
            true_labels = true_labels[:n_valid]
            
            # 计算指标
            clustering_metrics = spectral_evaluator.evaluate_clustering(predicted_labels, true_labels)
            modularity = modularity_calculator.calculate_modularity(processed_adjacency, predicted_labels)
            
            results_b.append({
                'ari': clustering_metrics['ari'],
                'nmi': clustering_metrics['nmi'],
                'modularity': modularity,
                'inference_time_ms': inference_time
            })
            
            print(f"    ARI: {clustering_metrics['ari']:.3f}, NMI: {clustering_metrics['nmi']:.3f}, 时间: {inference_time:.1f}ms")
    
    # 分析结果
    print("\n" + "="*50)
    print("泛化能力分析结果")
    print("="*50)
    
    # 计算平均指标
    avg_a = {
        'ari': np.mean([r['ari'] for r in results_a]),
        'nmi': np.mean([r['nmi'] for r in results_a]),
        'modularity': np.mean([r['modularity'] for r in results_a]),
        'inference_time_ms': np.mean([r['inference_time_ms'] for r in results_a])
    }
    
    avg_b = {
        'ari': np.mean([r['ari'] for r in results_b]),
        'nmi': np.mean([r['nmi'] for r in results_b]),
        'modularity': np.mean([r['modularity'] for r in results_b]),
        'inference_time_ms': np.mean([r['inference_time_ms'] for r in results_b])
    }
    
    print(f"\n模型A平均指标:")
    print(f"  ARI: {avg_a['ari']:.3f}")
    print(f"  NMI: {avg_a['nmi']:.3f}")
    print(f"  Modularity: {avg_a['modularity']:.3f}")
    print(f"  推理时间: {avg_a['inference_time_ms']:.1f}ms")
    
    print(f"\n模型B平均指标:")
    print(f"  ARI: {avg_b['ari']:.3f}")
    print(f"  NMI: {avg_b['nmi']:.3f}")
    print(f"  Modularity: {avg_b['modularity']:.3f}")
    print(f"  推理时间: {avg_b['inference_time_ms']:.1f}ms")
    
    # 比较结果
    print(f"\n模型比较:")
    if avg_a['ari'] > avg_b['ari']:
        print(f"✅ 模型A在ARI上表现更好: {avg_a['ari']:.3f} vs {avg_b['ari']:.3f}")
    else:
        print(f"✅ 模型B在ARI上表现更好: {avg_b['ari']:.3f} vs {avg_a['ari']:.3f}")
    
    # 泛化能力评估
    print(f"\n泛化能力评估:")
    if avg_a['ari'] > 0.2:
        print(f"✅ 模型A在双月数据集上显示出泛化能力 (ARI > 0.2)")
    else:
        print(f"❌ 模型A在双月数据集上泛化能力有限 (ARI ≤ 0.2)")
    
    if avg_b['ari'] > 0.2:
        print(f"✅ 模型B在双月数据集上显示出泛化能力 (ARI > 0.2)")
    else:
        print(f"❌ 模型B在双月数据集上泛化能力有限 (ARI ≤ 0.2)")
    
    # 保存结果
    results_data = [
        {
            'Method': 'Model A',
            'ARI': avg_a['ari'],
            'NMI': avg_a['nmi'],
            'Modularity': avg_a['modularity'],
            'Inference_Time_ms': avg_a['inference_time_ms'],
            'Meets_ARI_Target': avg_a['ari'] > 0.2,
            'Beats_Sklearn': avg_a['inference_time_ms'] < 100,
            'Description': 'Simple GNN-based model on moons dataset'
        },
        {
            'Method': 'Model B',
            'ARI': avg_b['ari'],
            'NMI': avg_b['nmi'],
            'Modularity': avg_b['modularity'],
            'Inference_Time_ms': avg_b['inference_time_ms'],
            'Meets_ARI_Target': avg_b['ari'] > 0.2,
            'Beats_Sklearn': avg_b['inference_time_ms'] < 100,
            'Description': 'Simple Similarity-based model on moons dataset'
        }
    ]
    
    df = pd.DataFrame(results_data)
    df.to_csv('simple_moons_results.csv', index=False)
    
    print(f"\n结果已保存到 simple_moons_results.csv")
    print(df.to_string(index=False))
    
    print(f"\n✅ 测试完成！")


if __name__ == "__main__":
    simple_moons_test() 
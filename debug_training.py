import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Tuple

# 导入模型
from model_a_gnn import ModelA_GNN, ModelALoss
from model_b_similarity import ModelB_Similarity, ModelBLoss
from data_generation import SyntheticDataGenerator


def debug_training():
    """调试训练过程"""
    device = 'cpu'
    print(f"使用设备: {device}")
    
    # 简化的模型配置
    model_config_a = {
        'input_channels': 1,
        'feature_dim': 64,  # 进一步减少
        'max_nodes': 100,   # 进一步减少
        'coord_dim': 2,
        'hidden_dim': 16,   # 进一步减少
        'node_feature_dim': 16
    }
    
    model_config_b = {
        'input_channels': 1,
        'feature_dim': 64,  # 进一步减少
        'max_nodes': 100,   # 进一步减少
        'coord_dim': 2,
        'similarity_hidden_dim': 8
    }
    
    # 初始化模型
    print("初始化模型...")
    model_a = ModelA_GNN(**model_config_a).to(device)
    model_b = ModelB_Similarity(**model_config_b).to(device)
    
    print(f"模型A参数数量: {model_a.count_parameters():,}")
    print(f"模型B参数数量: {model_b.count_parameters():,}")
    
    # 生成少量测试数据
    print("生成测试数据...")
    generator = SyntheticDataGenerator(img_size=64, n_samples=100, noise=0.1)
    
    train_data = []
    for i in range(5):  # 只生成5个样本
        print(f"生成样本 {i+1}/5")
        n_points = np.random.randint(80, 101)  # 减少节点数
        noise_level = np.random.uniform(0.05, 0.15)
        
        temp_generator = SyntheticDataGenerator(
            img_size=generator.img_size, 
            n_samples=n_points, 
            noise=noise_level
        )
        data = temp_generator.generate_dataset('circles', n_points)
        train_data.append(data)
    
    print("数据生成完成")
    
    # 测试数据准备
    print("测试数据准备...")
    batch_data = train_data[:2]  # 只取2个样本
    batch_size = len(batch_data)
    max_nodes = model_config_a['max_nodes']
    
    # 准备图像数据
    print("准备图像数据...")
    images = torch.stack([torch.from_numpy(data['image']).unsqueeze(0) for data in batch_data])
    print(f"图像形状: {images.shape}")
    
    # 准备目标数据
    print("准备目标数据...")
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
    
    print("目标数据准备完成")
    print(f"节点掩码形状: {targets['node_masks'].shape}")
    print(f"有效节点数: {targets['node_masks'].sum(dim=1)}")
    
    # 测试模型A前向传播
    print("\n测试模型A前向传播...")
    model_a.eval()
    with torch.no_grad():
        try:
            start_time = time.time()
            predictions_a = model_a(images, targets['node_masks'])
            end_time = time.time()
            print(f"模型A前向传播成功，耗时: {end_time - start_time:.2f}秒")
            print(f"预测形状: {predictions_a['adjacency_matrix'].shape}")
        except Exception as e:
            print(f"模型A前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 测试模型B前向传播
    print("\n测试模型B前向传播...")
    model_b.eval()
    with torch.no_grad():
        try:
            start_time = time.time()
            predictions_b = model_b(images, targets['node_masks'])
            end_time = time.time()
            print(f"模型B前向传播成功，耗时: {end_time - start_time:.2f}秒")
            print(f"预测形状: {predictions_b['adjacency_matrix'].shape}")
        except Exception as e:
            print(f"模型B前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 测试损失计算
    print("\n测试损失计算...")
    loss_a = ModelALoss()
    loss_b = ModelBLoss()
    
    try:
        start_time = time.time()
        loss_dict_a = loss_a(predictions_a, targets)
        end_time = time.time()
        print(f"模型A损失计算成功，耗时: {end_time - start_time:.2f}秒")
        print(f"损失值: {loss_dict_a}")
    except Exception as e:
        print(f"模型A损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        start_time = time.time()
        loss_dict_b = loss_b(predictions_b, targets)
        end_time = time.time()
        print(f"模型B损失计算成功，耗时: {end_time - start_time:.2f}秒")
        print(f"损失值: {loss_dict_b}")
    except Exception as e:
        print(f"模型B损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试一个训练步骤
    print("\n测试一个训练步骤...")
    model_a.train()
    optimizer_a = optim.Adam(model_a.parameters(), lr=0.001)
    
    try:
        start_time = time.time()
        
        optimizer_a.zero_grad()
        predictions = model_a(images, targets['node_masks'])
        loss_dict = loss_a(predictions, targets)
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        optimizer_a.step()
        
        end_time = time.time()
        print(f"训练步骤成功，耗时: {end_time - start_time:.2f}秒")
        print(f"损失值: {total_loss.item():.4f}")
        
    except Exception as e:
        print(f"训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    debug_training() 
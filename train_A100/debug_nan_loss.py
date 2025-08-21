#!/usr/bin/env python3
"""
调试NaN损失问题
"""

import torch
import torch.nn as nn
import numpy as np
from models.model_a_gnn_a100 import ModelA_GNN_A100, ModelA_A100_Loss
from data.data_generation_a100 import A100DataGenerator

def check_model_gradients(model):
    """检查模型梯度是否包含NaN或inf"""
    has_nan = False
    has_inf = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"❌ NaN gradient in {name}")
                has_nan = True
            if torch.isinf(param.grad).any():
                print(f"❌ Inf gradient in {name}")
                has_inf = True
    return has_nan, has_inf

def check_tensor_health(tensor, name):
    """检查张量健康状况"""
    if torch.isnan(tensor).any():
        print(f"❌ {name} contains NaN")
        return False
    if torch.isinf(tensor).any():
        print(f"❌ {name} contains Inf")
        return False
    if (tensor == 0).all():
        print(f"⚠️  {name} is all zeros")
    print(f"✅ {name} is healthy - range: [{tensor.min():.6f}, {tensor.max():.6f}]")
    return True

def debug_model_step():
    """逐步调试模型前向传播"""
    print("🔍 开始调试NaN损失问题...")
    
    # 设置设备
    device = torch.device('cpu')  # 使用CPU调试
    print(f"使用设备: {device}")
    
    # 创建模型
    model = ModelA_GNN_A100(
        input_channels=1,
        feature_dim=256,
        max_nodes=350,
        coord_dim=2,
        hidden_dim=128,
        node_feature_dim=64
    ).to(device)
    
    # 创建损失函数
    criterion = ModelA_A100_Loss(
        coord_weight=1.0,
        edge_weight=2.5,
        count_weight=0.1,
        regularization_weight=0.01
    ).to(device)
    
    print(f"模型参数数量: {model.count_parameters():,}")
    
    # 创建简单的测试数据
    batch_size = 2
    max_nodes = 350
    
    # 生成真实数据
    generator = A100DataGenerator()
    train_data, _ = generator.create_train_test_split_parallel(train_size=batch_size, test_size=1)
    
    print("\n📊 准备测试数据...")
    images_list = []
    points_list = []
    adjacency_list = []
    masks_list = []
    
    for i in range(batch_size):
        sample = train_data[i]
        
        # 准备图像
        image = torch.from_numpy(sample['image']).unsqueeze(0).float()
        images_list.append(image)
        
        # 准备点坐标和邻接矩阵
        points = sample['points']
        adjacency = sample['adjacency']
        n_points = len(points)
        
        # 填充到max_nodes
        points_padded = np.zeros((max_nodes, 2))
        points_padded[:n_points] = points
        
        adjacency_padded = np.zeros((max_nodes, max_nodes))
        adjacency_padded[:n_points, :n_points] = adjacency
        
        # 创建掩码
        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:n_points] = True
        
        points_list.append(torch.from_numpy(points_padded).float())
        adjacency_list.append(torch.from_numpy(adjacency_padded).float())
        masks_list.append(mask)
    
    # 堆叠成批次
    images = torch.stack(images_list).to(device)
    points = torch.stack(points_list).to(device)
    adjacency = torch.stack(adjacency_list).to(device)
    node_masks = torch.stack(masks_list).to(device)
    
    # 检查输入数据健康状况
    print("\n🔍 检查输入数据:")
    check_tensor_health(images, "images")
    check_tensor_health(points, "points")
    check_tensor_health(adjacency, "adjacency")
    print(f"node_masks shape: {node_masks.shape}, active nodes per batch: {node_masks.sum(dim=1)}")
    
    # 逐步前向传播
    print("\n🔄 开始逐步前向传播...")
    
    model.train()
    
    try:
        # 1. CNN编码器
        print("1. CNN编码器...")
        image_features = model.cnn_encoder(images)
        check_tensor_health(image_features, "image_features")
        
        # 2. 节点回归器
        print("2. 节点回归器...")
        predicted_coords, node_counts = model.node_regressor(image_features)
        check_tensor_health(predicted_coords, "predicted_coords")
        check_tensor_health(node_counts, "node_counts")
        
        # 3. 图构建器
        print("3. 图构建器...")
        edge_index, edge_weight = model.graph_builder.build_knn_graph_vectorized(
            predicted_coords, node_masks)
        print(f"edge_index shape: {edge_index.shape}")
        if edge_index.numel() > 0:
            check_tensor_health(edge_weight, "edge_weight")
        else:
            print("⚠️  没有生成边")
        
        # 4. GNN处理器
        print("4. GNN处理器...")
        node_features = model.gnn_processor(predicted_coords, edge_index, edge_weight, node_masks)
        check_tensor_health(node_features, "node_features")
        
        # 5. 边预测器
        print("5. 边预测器...")
        adjacency_matrix, adjacency_logits = model.edge_predictor(node_features, node_masks)
        check_tensor_health(adjacency_matrix, "adjacency_matrix")
        check_tensor_health(adjacency_logits, "adjacency_logits")
        
        print("✅ 前向传播完成")
        
        # 准备目标数据
        targets = {
            'points': points,
            'adjacency': adjacency,
            'node_masks': node_masks
        }
        
        predictions = {
            'predicted_coords': predicted_coords,
            'node_counts': node_counts,
            'adjacency_matrix': adjacency_matrix,
            'adjacency_logits': adjacency_logits,
            'node_features': node_features,
            'image_features': image_features,
            'edge_index': edge_index,
            'edge_weight': edge_weight
        }
        
        # 6. 损失计算
        print("6. 损失计算...")
        loss_dict = criterion(predictions, targets)
        
        print("损失详情:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value.item():.6f}")
            if torch.isnan(value):
                print(f"  ❌ {key} is NaN!")
            elif torch.isinf(value):
                print(f"  ❌ {key} is Inf!")
        
        # 7. 反向传播测试
        print("7. 反向传播测试...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        
        loss_dict['total_loss'].backward()
        
        # 检查梯度
        has_nan_grad, has_inf_grad = check_model_gradients(model)
        
        if not has_nan_grad and not has_inf_grad:
            print("✅ 梯度健康")
            optimizer.step()
            print("✅ 优化器步骤完成")
        else:
            print("❌ 梯度包含NaN或Inf")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n🎯 调试完成")

if __name__ == "__main__":
    debug_model_step()
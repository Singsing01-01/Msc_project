#!/usr/bin/env python3
"""
测试Model A修复后的兼容性
"""

import torch
import torch.nn as nn
from models.model_a_gnn_a100 import ModelA_GNN_A100, ModelA_A100_Loss

def test_model_fixes():
    print("🔧 测试A100混合精度修复...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # 创建测试数据
    batch_size = 2
    max_nodes = 350
    
    images = torch.randn(batch_size, 1, 64, 64).to(device)
    node_masks = torch.ones(batch_size, max_nodes, dtype=torch.bool).to(device)
    # 设置实际节点数
    node_masks[0, 10:] = False  # 第一个样本10个节点
    node_masks[1, 15:] = False  # 第二个样本15个节点
    
    # 创建目标数据
    targets = {
        'points': torch.randn(batch_size, max_nodes, 2).to(device),
        'adjacency': torch.rand(batch_size, max_nodes, max_nodes).to(device),
        'node_masks': node_masks
    }
    
    print("✅ 测试数据准备完成")
    
    # 测试前向传播
    try:
        print("🔄 测试前向传播...")
        model.eval()
        with torch.no_grad():
            predictions = model(images, node_masks)
        
        print("✅ 前向传播成功")
        print(f"  - 预测坐标形状: {predictions['predicted_coords'].shape}")
        print(f"  - 邻接矩阵形状: {predictions['adjacency_matrix'].shape}")
        print(f"  - 邻接logits形状: {predictions['adjacency_logits'].shape}")
        print(f"  - 节点计数形状: {predictions['node_counts'].shape}")
        
        # 检查输出值范围
        adj_min, adj_max = predictions['adjacency_matrix'].min(), predictions['adjacency_matrix'].max()
        print(f"  - 邻接矩阵值范围: [{adj_min:.4f}, {adj_max:.4f}]")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False
    
    # 测试损失计算
    try:
        print("🔄 测试损失计算...")
        model.train()
        loss_dict = criterion(predictions, targets)
        
        print("✅ 损失计算成功")
        for key, value in loss_dict.items():
            print(f"  - {key}: {value.item():.6f}")
        
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        return False
    
    # 测试混合精度训练
    try:
        print("🔄 测试混合精度训练...")
        
        # 使用新的autocast API
        try:
            from torch.amp import autocast, GradScaler
            scaler = GradScaler('cuda')
            autocast_device = 'cuda'
        except (ImportError, TypeError):
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            autocast_device = None
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        optimizer.zero_grad()
        
        if autocast_device:
            with autocast(autocast_device):
                predictions = model(images, node_masks)
                loss_dict = criterion(predictions, targets)
                total_loss = loss_dict['total_loss']
        else:
            with autocast():
                predictions = model(images, node_masks)
                loss_dict = criterion(predictions, targets)
                total_loss = loss_dict['total_loss']
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✅ 混合精度训练成功")
        print(f"  - 总损失: {total_loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ 混合精度训练失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！修复成功！")
    return True

if __name__ == "__main__":
    success = test_model_fixes()
    if success:
        print("\n✅ 现在可以安全运行完整训练:")
        print("   python training_pipeline_a100.py")
    else:
        print("\n❌ 修复未完成，请检查错误信息")
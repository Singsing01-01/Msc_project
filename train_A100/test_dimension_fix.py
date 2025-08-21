#!/usr/bin/env python3
"""
测试维度修复后的模型
"""

import torch
from models.model_a_gnn_a100 import ModelA_GNN_A100, ModelA_A100_Loss

def test_model_dimensions():
    """测试模型维度是否正确"""
    print("🔧 测试模型维度修复...")
    
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
    
    print(f"模型参数数量: {model.count_parameters():,}")
    
    # 创建测试数据
    batch_size = 2
    max_nodes = 350
    
    images = torch.randn(batch_size, 1, 64, 64).to(device)
    node_masks = torch.ones(batch_size, max_nodes, dtype=torch.bool).to(device)
    # 设置实际节点数
    node_masks[0, 10:] = False  # 第一个样本10个节点
    node_masks[1, 15:] = False  # 第二个样本15个节点
    
    print(f"输入形状:")
    print(f"  images: {images.shape}")
    print(f"  node_masks: {node_masks.shape}")
    print(f"  有效节点数: {node_masks.sum(dim=1)}")
    
    # 测试前向传播
    try:
        print("\n🔄 测试前向传播...")
        model.eval()
        with torch.no_grad():
            predictions = model(images, node_masks)
        
        print("✅ 前向传播成功!")
        print(f"输出形状:")
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        # 检查关键输出
        pred_coords = predictions['predicted_coords']
        adj_matrix = predictions['adjacency_matrix']
        adj_logits = predictions['adjacency_logits']
        node_features = predictions['node_features']
        
        print(f"\n📊 详细检查:")
        print(f"  预测坐标范围: [{pred_coords.min():.4f}, {pred_coords.max():.4f}]")
        print(f"  邻接矩阵范围: [{adj_matrix.min():.4f}, {adj_matrix.max():.4f}]")
        print(f"  邻接logits范围: [{adj_logits.min():.4f}, {adj_logits.max():.4f}]")
        print(f"  节点特征范围: [{node_features.min():.4f}, {node_features.max():.4f}]")
        
        # 检查是否有NaN或Inf
        def check_tensor_health(tensor, name):
            if torch.isnan(tensor).any():
                print(f"  ❌ {name} 包含 NaN")
                return False
            if torch.isinf(tensor).any():
                print(f"  ❌ {name} 包含 Inf")
                return False
            print(f"  ✅ {name} 健康")
            return True
        
        print(f"\n🏥 健康检查:")
        all_healthy = True
        all_healthy &= check_tensor_health(pred_coords, "预测坐标")
        all_healthy &= check_tensor_health(adj_matrix, "邻接矩阵")
        all_healthy &= check_tensor_health(adj_logits, "邻接logits")
        all_healthy &= check_tensor_health(node_features, "节点特征")
        
        if all_healthy:
            print("🎉 所有输出都健康!")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """测试损失计算"""
    print("\n🔧 测试损失计算...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型和损失函数
    model = ModelA_GNN_A100(
        input_channels=1,
        feature_dim=256,
        max_nodes=350,
        coord_dim=2,
        hidden_dim=128,
        node_feature_dim=64
    ).to(device)
    
    criterion = ModelA_A100_Loss(
        coord_weight=1.0,
        edge_weight=1.0,
        count_weight=0.1,
        regularization_weight=0.001
    ).to(device)
    
    # 创建测试数据
    batch_size = 2
    max_nodes = 350
    
    images = torch.randn(batch_size, 1, 64, 64).to(device)
    points = torch.randn(batch_size, max_nodes, 2).to(device)
    adjacency = torch.rand(batch_size, max_nodes, max_nodes).to(device)
    node_masks = torch.ones(batch_size, max_nodes, dtype=torch.bool).to(device)
    node_masks[0, 10:] = False
    node_masks[1, 15:] = False
    
    try:
        model.train()
        predictions = model(images, node_masks)
        
        targets = {
            'points': points,
            'adjacency': adjacency,
            'node_masks': node_masks
        }
        
        loss_dict = criterion(predictions, targets)
        
        print("✅ 损失计算成功!")
        print("📊 损失详情:")
        for key, value in loss_dict.items():
            if torch.isnan(value):
                print(f"  ❌ {key}: NaN")
            elif torch.isinf(value):
                print(f"  ❌ {key}: Inf")
            else:
                print(f"  ✅ {key}: {value.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 开始模型维度测试...")
    
    success1 = test_model_dimensions()
    success2 = test_loss_computation()
    
    if success1 and success2:
        print("\n🎉 所有测试通过! 模型架构正确!")
        print("✅ 现在可以安全运行完整训练:")
        print("   python training_pipeline_a100.py")
    else:
        print("\n❌ 测试失败，需要进一步调试")
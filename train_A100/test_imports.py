#!/usr/bin/env python3
"""
测试所有导入是否正常
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试关键模块导入"""
    print("🔧 测试模块导入...")
    
    try:
        # 测试模型导入
        from models.model_a_gnn_a100 import ModelA_GNN_A100, ModelA_A100_Loss
        print("✅ Model A 导入成功")
        
        from models.model_b_similarity_a100 import ModelB_Similarity_A100, ModelB_A100_Loss  
        print("✅ Model B 导入成功")
        
        # 测试工具导入
        from utils.evaluation_metrics import GraphEvaluationMetrics
        print("✅ 评估指标导入成功")
        
        # 测试数据生成器导入
        from data.data_generation_a100 import A100DataGenerator
        print("✅ 数据生成器导入成功")
        
        # 测试优化器导入
        try:
            from simple_aggressive_optimizer import apply_simple_aggressive_optimization
            print("✅ 简化激进优化器导入成功")
        except ImportError:
            print("⚠️ 简化激进优化器不可用")
        
        try:
            from super_aggressive_optimizer import apply_super_aggressive_optimization
            print("✅ 超级激进优化器导入成功")
        except ImportError:
            print("⚠️ 超级激进优化器不可用")
            
        try:
            from extreme_optimizer import ExtremeMetricLoss
            print("✅ 极端优化器导入成功")
        except ImportError:
            print("⚠️ 极端优化器不可用")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_torch_imports():
    """测试PyTorch相关导入"""
    print("\n🔧 测试PyTorch导入...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import torch.nn as nn
        print("✅ torch.nn")
        
        import torch.optim as optim
        print("✅ torch.optim")
        
        # 测试CUDA可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用 - {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CUDA 不可用，将使用CPU")
            
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False

def test_other_dependencies():
    """测试其他依赖"""
    print("\n🔧 测试其他依赖...")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        
        try:
            import torch_geometric
            print(f"✅ PyTorch Geometric {torch_geometric.__version__}")
        except ImportError:
            print("⚠️ PyTorch Geometric 不可用")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        import tqdm
        print("✅ tqdm")
        
        return True
        
    except ImportError as e:
        print(f"❌ 依赖导入失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始导入测试...\n")
    
    test1 = test_torch_imports()
    test2 = test_imports() 
    test3 = test_other_dependencies()
    
    if all([test1, test2, test3]):
        print("\n🎉 所有导入测试通过!")
        print("✅ 环境配置正确，可以开始训练")
        print("\n📋 可用命令:")
        print("   python training_pipeline_a100.py  # 完整训练")
        print("   python train_model_b_only.py      # 仅训练Model B")
    else:
        print("\n❌ 部分导入测试失败")
        print("🔧 请检查环境配置和依赖安装")
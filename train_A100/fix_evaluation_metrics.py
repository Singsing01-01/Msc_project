#!/usr/bin/env python3
"""
修复 evaluation_metrics.py 缺失的 compute_batch_metrics 方法
"""

import os
import sys

def fix_evaluation_metrics():
    """添加缺失的 compute_batch_metrics 方法"""
    
    # 找到 evaluation_metrics.py 文件
    eval_file = "utils/evaluation_metrics.py"
    
    if not os.path.exists(eval_file):
        print(f"❌ 找不到文件: {eval_file}")
        return False
    
    # 读取现有文件
    with open(eval_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有该方法
    if 'def compute_batch_metrics(' in content:
        print("✅ compute_batch_metrics 方法已存在")
        return True
    
    # 找到插入点（在 evaluate_batch 方法之后）
    method_to_add = '''
    def compute_batch_metrics(self, pred_adj, true_adj, masks, clustering_method='spectral'):
        """
        计算批次指标，与训练脚本接口兼容
        
        Args:
            pred_adj: 预测的邻接矩阵 [batch_size, max_nodes, max_nodes] (numpy array)
            true_adj: 真实的邻接矩阵 [batch_size, max_nodes, max_nodes] (numpy array)
            masks: 节点掩码 [batch_size, max_nodes] (numpy array)
            
        Returns:
            Dictionary containing lists of metrics for each sample
        """
        import torch
        import numpy as np
        
        batch_size = pred_adj.shape[0]
        
        # 初始化指标列表
        ari_scores = []
        nmi_scores = []
        modularity_scores = []
        inference_times = []
        
        for b in range(batch_size):
            node_mask = torch.from_numpy(masks[b]).bool()
            valid_nodes = node_mask.sum().item()
            
            if valid_nodes <= 1:
                # 跳过无效样本
                ari_scores.append(0.0)
                nmi_scores.append(0.0)
                modularity_scores.append(0.0)
                inference_times.append(0.0)
                continue
                
            # 转换为张量
            adj_true = torch.from_numpy(true_adj[b]).float()
            adj_pred = torch.from_numpy(pred_adj[b]).float()
            
            # 提取社区标签
            true_labels = self.create_ground_truth_labels(adj_true, node_mask)
            pred_labels = self.extract_communities_from_adjacency(
                adj_pred, node_mask, method=clustering_method
            )
            
            # 计算指标
            ari = self.calculate_ari(true_labels, pred_labels)
            nmi = self.calculate_nmi(true_labels, pred_labels)
            modularity = self.calculate_modularity(adj_pred, node_mask, pred_labels)
            
            ari_scores.append(ari)
            nmi_scores.append(nmi)
            modularity_scores.append(modularity)
            inference_times.append(1.0)  # 占位符，实际推理时间在外部计算
        
        # 返回列表形式的指标（与训练脚本期望的格式匹配）
        return {
            'ARI': ari_scores,
            'NMI': nmi_scores,
            'Modularity': modularity_scores,
            'Inference_Time_ms': inference_times
        }
'''
    
    # 在最后一个方法之后插入新方法
    # 找到 "def test_evaluation_metrics():" 之前插入
    if 'def test_evaluation_metrics():' in content:
        insert_point = content.find('def test_evaluation_metrics():')
        new_content = content[:insert_point] + method_to_add + "\n\n" + content[insert_point:]
    else:
        # 如果没有找到测试函数，在文件最后插入
        new_content = content + method_to_add
    
    # 写回文件
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ 成功添加 compute_batch_metrics 方法到 evaluation_metrics.py")
    return True

if __name__ == "__main__":
    print("🔧 修复 evaluation_metrics.py...")
    success = fix_evaluation_metrics()
    
    if success:
        print("✅ 修复完成！现在可以重新运行训练脚本")
        print("📝 运行: python train_model_b_only.py")
    else:
        print("❌ 修复失败，请手动检查文件")
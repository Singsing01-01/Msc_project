import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
import time

from model_a_gnn import ModelA_GNN
from model_b_similarity import ModelB_Similarity
from training_pipeline import ModelTrainer
from evaluation_framework import ComprehensiveEvaluator
from baseline_comparison import BaselineEvaluator
from data_augmentation import create_data_loaders
from data_generation import SyntheticDataGenerator
from visualization_tools import ModelVisualizationTools


class ModelComparisonAnalysis:
    def __init__(self, device: torch.device = None, output_dir: str = "./comparison_results"):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        
        self.evaluator = ComprehensiveEvaluator()
        self.baseline_evaluator = BaselineEvaluator()
        self.visualizer = ModelVisualizationTools()
        
        print(f"Model comparison analysis initialized")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def prepare_datasets(self, train_size: int = 100, test_size: int = 50):
        """准备训练和测试数据集"""
        print("="*60)
        print("数据集准备")
        print("="*60)
        
        generator = SyntheticDataGenerator(img_size=64, n_samples=300, noise=0.1)
        
        print("生成同心圆训练数据和双月测试数据...")
        train_data, test_data = generator.create_train_test_split(
            train_size=train_size, test_size=test_size
        )
        
        train_loader, test_loader = create_data_loaders(
            train_data, test_data, 
            batch_size=4, 
            use_augmentation=True
        )
        
        print(f"训练样本数: {len(train_data)}")
        print(f"测试样本数: {len(test_data)}")
        print(f"跨域验证: 同心圆 → 双月")
        
        return train_loader, test_loader, test_data
    
    def train_both_models(self, train_loader, test_loader, epochs: int = 30):
        """训练Model A和Model B"""
        print("\n" + "="*60)
        print("模型训练阶段")
        print("="*60)
        
        models = {}
        training_histories = {}
        
        # 训练Model A (GNN-based)
        print("\n[TRAINING] 训练Model A (GNN架构)...")
        print("架构: CNN编码器 + 节点回归 + GCN图处理 + 边预测器")
        
        try:
            model_a = ModelA_GNN().to(self.device)
            trainer_a = ModelTrainer(model_a, 'model_a', self.device)
            
            print(f"Model A 参数量: {model_a.count_parameters():,}")
            
            start_time = time.time()
            history_a = trainer_a.train(
                train_loader, test_loader, 
                num_epochs=epochs, 
                save_dir=os.path.join(self.output_dir, 'checkpoints')
            )
            training_time_a = time.time() - start_time
            
            models['Model_A_GNN'] = model_a
            training_histories['Model_A_GNN'] = history_a
            
            print(f"[SUCCESS] Model A 训练完成")
            print(f"   最终损失: {history_a['train_loss'][-1]:.4f}")
            print(f"   训练时间: {training_time_a/60:.2f} 分钟")
            
        except Exception as e:
            print(f"[ERROR] Model A 训练失败: {e}")
        
        # 训练Model B (Similarity-based)
        print("\n[TRAINING] 训练Model B (相似度架构)...")
        print("架构: 轻量CNN编码器 + 节点检测 + 余弦相似度 + MLP校正")
        
        try:
            model_b = ModelB_Similarity().to(self.device)
            trainer_b = ModelTrainer(model_b, 'model_b', self.device)
            
            print(f"Model B 参数量: {model_b.count_parameters():,}")
            print(f"参数效率: {model_a.count_parameters()/model_b.count_parameters():.1f}x 更少")
            
            start_time = time.time()
            history_b = trainer_b.train(
                train_loader, test_loader, 
                num_epochs=epochs, 
                save_dir=os.path.join(self.output_dir, 'checkpoints')
            )
            training_time_b = time.time() - start_time
            
            models['Model_B_Similarity'] = model_b
            training_histories['Model_B_Similarity'] = history_b
            
            print(f"[SUCCESS] Model B 训练完成")
            print(f"   最终损失: {history_b['train_loss'][-1]:.4f}")
            print(f"   训练时间: {training_time_b/60:.2f} 分钟")
            
        except Exception as e:
            print(f"[ERROR] Model B 训练失败: {e}")
        
        return models, training_histories
    
    def evaluate_all_methods(self, models: Dict, test_loader, max_samples: int = 30):
        """评估所有方法：Model A, Model B, sklearn baselines"""
        print("\n" + "="*60)
        print("综合评估阶段")
        print("="*60)
        
        all_results = {}
        
        # 评估深度学习模型
        print("\n[EVALUATION] 评估深度学习模型...")
        for model_name, model in models.items():
            print(f"\n正在评估 {model_name}...")
            
            model_results = self.evaluator.evaluate_model(
                model, test_loader, self.device, max_samples=max_samples
            )
            all_results[model_name] = model_results
            
            print(f"  [SUCCESS] {model_name} 评估完成:")
            print(f"    ARI: {model_results['mean_ari']:.4f} ± {model_results['std_ari']:.4f}")
            print(f"    NMI: {model_results['mean_nmi']:.4f} ± {model_results['std_nmi']:.4f}")
            print(f"    模块度: {model_results['mean_modularity']:.4f} ± {model_results['std_modularity']:.4f}")
            print(f"    推理时间: {model_results['mean_inference_time']*1000:.2f} ms")
            print(f"    样本数: {model_results['num_samples']}")
        
        # 评估基线方法
        print("\n[EVALUATION] 评估基线方法...")
        baseline_results = self.baseline_evaluator.evaluate_all_baselines(
            test_loader, max_samples=max_samples
        )
        
        # 重点关注sklearn img_to_graph
        if 'sklearn_baseline' in baseline_results:
            sklearn_results = baseline_results['sklearn_baseline']
            print(f"\n  [SUCCESS] sklearn img_to_graph 评估完成:")
            print(f"    ARI: {sklearn_results['mean_ari']:.4f} ± {sklearn_results['std_ari']:.4f}")
            print(f"    NMI: {sklearn_results['mean_nmi']:.4f} ± {sklearn_results['std_nmi']:.4f}")
            print(f"    模块度: {sklearn_results['mean_modularity']:.4f} ± {sklearn_results['std_modularity']:.4f}")
            print(f"    推理时间: {sklearn_results['mean_inference_time']*1000:.2f} ms")
            print(f"    样本数: {sklearn_results['num_samples']}")
        
        # 合并所有结果
        all_results.update(baseline_results)
        
        return all_results
    
    def create_detailed_comparison_table(self, results: Dict) -> pd.DataFrame:
        """创建详细的比较表格"""
        print("\n[REPORT] 生成详细比较报告...")
        
        comparison_data = []
        
        for method_name, metrics in results.items():
            # 确定方法类型
            if 'Model_A' in method_name:
                method_type = "深度学习模型 (GNN)"
                architecture = "CNN + GCN + 边预测器"
            elif 'Model_B' in method_name:
                method_type = "深度学习模型 (相似度)"
                architecture = "轻量CNN + 余弦相似度 + MLP"
            elif 'sklearn' in method_name:
                method_type = "传统基线"
                architecture = "sklearn img_to_graph"
            else:
                method_type = "传统基线"
                architecture = "图构建算法"
            
            comparison_data.append({
                '方法名称': method_name,
                '方法类型': method_type,
                '架构描述': architecture,
                '平均ARI': metrics['mean_ari'],
                'ARI标准差': metrics['std_ari'],
                '平均NMI': metrics['mean_nmi'],
                'NMI标准差': metrics['std_nmi'],
                '平均模块度': metrics['mean_modularity'],
                '模块度标准差': metrics['std_modularity'],
                '平均推理时间(ms)': metrics['mean_inference_time'] * 1000,
                '推理时间标准差(ms)': metrics['std_inference_time'] * 1000,
                '评估样本数': metrics['num_samples']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 按ARI排序
        df = df.sort_values('平均ARI', ascending=False)
        
        return df
    
    def analyze_model_vs_sklearn_performance(self, results: Dict):
        """专门分析Model A和Model B与sklearn的性能对比"""
        print("\n" + "="*60)
        print("Model A & Model B vs sklearn img_to_graph 性能分析")
        print("="*60)
        
        sklearn_results = results.get('sklearn_baseline', {})
        model_a_results = results.get('Model_A_GNN', {})
        model_b_results = results.get('Model_B_Similarity', {})
        
        if not sklearn_results:
            print("[ERROR] sklearn基线结果不可用")
            return
        
        sklearn_ari = sklearn_results['mean_ari']
        sklearn_nmi = sklearn_results['mean_nmi']
        sklearn_modularity = sklearn_results['mean_modularity']
        sklearn_time = sklearn_results['mean_inference_time'] * 1000
        
        print(f"\n[BASELINE] sklearn img_to_graph 基线性能:")
        print(f"   ARI: {sklearn_ari:.4f}")
        print(f"   NMI: {sklearn_nmi:.4f}")
        print(f"   模块度: {sklearn_modularity:.4f}")
        print(f"   推理时间: {sklearn_time:.2f} ms")
        
        # 分析Model A vs sklearn
        if model_a_results:
            model_a_ari = model_a_results['mean_ari']
            model_a_nmi = model_a_results['mean_nmi']
            model_a_modularity = model_a_results['mean_modularity']
            model_a_time = model_a_results['mean_inference_time'] * 1000
            
            print(f"\n[COMPARISON] Model A (GNN) vs sklearn 对比:")
            print(f"   ARI: {model_a_ari:.4f} vs {sklearn_ari:.4f} " + 
                  f"({'[提升]' if model_a_ari > sklearn_ari else '[下降]'} " +
                  f"{abs(model_a_ari - sklearn_ari):.4f})")
            print(f"   NMI: {model_a_nmi:.4f} vs {sklearn_nmi:.4f} " + 
                  f"({'[提升]' if model_a_nmi > sklearn_nmi else '[下降]'} " +
                  f"{abs(model_a_nmi - sklearn_nmi):.4f})")
            print(f"   模块度: {model_a_modularity:.4f} vs {sklearn_modularity:.4f} " + 
                  f"({'[提升]' if model_a_modularity > sklearn_modularity else '[下降]'} " +
                  f"{abs(model_a_modularity - sklearn_modularity):.4f})")
            print(f"   推理时间: {model_a_time:.2f} ms vs {sklearn_time:.2f} ms " + 
                  f"({'[更快]' if model_a_time < sklearn_time else '[更慢]'} " +
                  f"{abs(model_a_time - sklearn_time):.2f} ms)")
        
        # 分析Model B vs sklearn
        if model_b_results:
            model_b_ari = model_b_results['mean_ari']
            model_b_nmi = model_b_results['mean_nmi']
            model_b_modularity = model_b_results['mean_modularity']
            model_b_time = model_b_results['mean_inference_time'] * 1000
            
            print(f"\n[COMPARISON] Model B (相似度) vs sklearn 对比:")
            print(f"   ARI: {model_b_ari:.4f} vs {sklearn_ari:.4f} " + 
                  f"({'[提升]' if model_b_ari > sklearn_ari else '[下降]'} " +
                  f"{abs(model_b_ari - sklearn_ari):.4f})")
            print(f"   NMI: {model_b_nmi:.4f} vs {sklearn_nmi:.4f} " + 
                  f"({'[提升]' if model_b_nmi > sklearn_nmi else '[下降]'} " +
                  f"{abs(model_b_nmi - sklearn_nmi):.4f})")
            print(f"   模块度: {model_b_modularity:.4f} vs {sklearn_modularity:.4f} " + 
                  f"({'[提升]' if model_b_modularity > sklearn_modularity else '[下降]'} " +
                  f"{abs(model_b_modularity - sklearn_modularity):.4f})")
            print(f"   推理时间: {model_b_time:.2f} ms vs {sklearn_time:.2f} ms " + 
                  f"({'[更快]' if model_b_time < sklearn_time else '[更慢]'} " +
                  f"{abs(model_b_time - sklearn_time):.2f} ms)")
        
        # 综合分析
        print(f"\n[ANALYSIS] 综合性能分析:")
        target_ari = 0.80
        
        methods = []
        if model_a_results: methods.append(("Model A", model_a_ari))
        if model_b_results: methods.append(("Model B", model_b_ari))
        methods.append(("sklearn", sklearn_ari))
        
        best_method = max(methods, key=lambda x: x[1])
        print(f"   最佳ARI: {best_method[0]} ({best_method[1]:.4f})")
        print(f"   目标ARI: {target_ari}")
        
        if best_method[1] >= target_ari:
            print(f"   [SUCCESS] 达到目标！{best_method[0]} 超过ARI阈值")
        else:
            gap = target_ari - best_method[1]
            print(f"   [WARNING] 未达到目标，差距: {gap:.4f}")
    
    def create_comparison_visualizations(self, results: Dict, test_data: List):
        """创建对比可视化"""
        print("\n[VISUALIZATION] 生成对比可视化...")
        
        # 性能对比图
        self.visualizer.create_performance_comparison_plot(
            results,
            save_path=os.path.join(self.output_dir, 'visualizations', 'performance_comparison.png')
        )
        
        # 专门的Model vs sklearn对比图
        self.create_model_sklearn_comparison_plot(results)
        
        # 模型预测可视化
        if test_data:
            sample_data = test_data[0].copy()
            
            # 处理数据
            valid_mask = sample_data['labels'] >= 0
            for key in ['points', 'labels']:
                if key in sample_data:
                    sample_data[key] = sample_data[key][valid_mask]
            
            if 'points_pixel' in sample_data:
                sample_data['points_pixel'] = sample_data['points_pixel'][valid_mask]
            
            n_valid = valid_mask.sum()
            sample_data['adjacency'] = sample_data['adjacency'][:n_valid, :n_valid]
            
            # 基线对比可视化
            self.visualizer.visualize_baseline_comparison(
                sample_data,
                save_path=os.path.join(self.output_dir, 'visualizations', 'baseline_comparison.png')
            )
    
    def create_model_sklearn_comparison_plot(self, results: Dict):
        """创建专门的Model vs sklearn对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取数据
        methods = []
        ari_scores = []
        nmi_scores = []
        modularity_scores = []
        inference_times = []
        
        method_colors = {
            'Model_A_GNN': '#ff7f0e',
            'Model_B_Similarity': '#2ca02c', 
            'sklearn_baseline': '#1f77b4'
        }
        
        for method in ['Model_A_GNN', 'Model_B_Similarity', 'sklearn_baseline']:
            if method in results:
                methods.append(method.replace('_', ' '))
                ari_scores.append(results[method]['mean_ari'])
                nmi_scores.append(results[method]['mean_nmi'])
                modularity_scores.append(results[method]['mean_modularity'])
                inference_times.append(results[method]['mean_inference_time'] * 1000)
        
        colors = [method_colors.get(method.replace(' ', '_'), '#gray') for method in methods]
        
        # ARI对比
        bars1 = axes[0, 0].bar(methods, ari_scores, color=colors, alpha=0.7)
        axes[0, 0].set_title('调整兰德指数 (ARI) 对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('ARI 分数')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加目标线
        axes[0, 0].axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='目标 ARI ≥ 0.80')
        axes[0, 0].legend()
        
        # 添加数值标签
        for bar, score in zip(bars1, ari_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.4f}', ha='center', va='bottom')
        
        # NMI对比
        bars2 = axes[0, 1].bar(methods, nmi_scores, color=colors, alpha=0.7)
        axes[0, 1].set_title('归一化互信息 (NMI) 对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('NMI 分数')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, score in zip(bars2, nmi_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.4f}', ha='center', va='bottom')
        
        # 模块度对比
        bars3 = axes[1, 0].bar(methods, modularity_scores, color=colors, alpha=0.7)
        axes[1, 0].set_title('模块度对比', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('模块度分数')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, score in zip(bars3, modularity_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.4f}', ha='center', va='bottom')
        
        # 推理时间对比
        bars4 = axes[1, 1].bar(methods, inference_times, color=colors, alpha=0.7)
        axes[1, 1].set_title('推理时间对比', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('时间 (ms)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, time in zip(bars4, inference_times):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                           f'{time:.1f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'model_sklearn_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_complete_comparison(self, train_size: int = 80, test_size: int = 40, 
                              epochs: int = 25, max_eval_samples: int = 25):
        """运行完整的对比分析"""
        print("="*80)
        print("Model A & Model B vs sklearn img_to_graph 综合对比分析")
        print("="*80)
        
        start_time = time.time()
        
        # 1. 准备数据
        train_loader, test_loader, test_data = self.prepare_datasets(train_size, test_size)
        
        # 2. 训练模型
        models, training_histories = self.train_both_models(train_loader, test_loader, epochs)
        
        if not models:
            print("[ERROR] 没有成功训练的模型，无法进行对比分析")
            return None
        
        # 3. 评估所有方法
        results = self.evaluate_all_methods(models, test_loader, max_eval_samples)
        
        # 4. 创建详细对比表格
        comparison_df = self.create_detailed_comparison_table(results)
        
        # 5. 保存结果
        comparison_df.to_csv(
            os.path.join(self.output_dir, 'model_sklearn_comparison.csv'), 
            index=False, encoding='utf-8-sig'
        )
        
        # 6. 性能分析
        self.analyze_model_vs_sklearn_performance(results)
        
        # 7. 生成可视化
        self.create_comparison_visualizations(results, test_data)
        
        # 8. 总结报告
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("对比分析总结")
        print("="*60)
        print(f"总耗时: {total_time/60:.2f} 分钟")
        print(f"结果保存至: {self.output_dir}")
        
        print(f"\n[RANKING] 最终排名 (按ARI排序):")
        for i, row in comparison_df.iterrows():
            rank = i + 1
            name = row['方法名称']
            ari = row['平均ARI']
            nmi = row['平均NMI']
            modularity = row['平均模块度']
            print(f"   {rank}. {name}: ARI={ari:.4f}, NMI={nmi:.4f}, 模块度={modularity:.4f}")
        
        return comparison_df


def main():
    """主函数：运行完整的模型对比分析"""
    analyzer = ModelComparisonAnalysis(
        output_dir="/Users/jeremyfang/Downloads/image_to_graph/model_sklearn_comparison"
    )
    
    results = analyzer.run_complete_comparison(
        train_size=60,      # 训练样本数
        test_size=30,       # 测试样本数  
        epochs=20,          # 训练轮数
        max_eval_samples=20 # 最大评估样本数
    )
    
    return results


if __name__ == "__main__":
    main()
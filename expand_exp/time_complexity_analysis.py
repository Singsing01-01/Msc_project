#!/usr/bin/env python3
"""
实验1：多尺度时间复杂度分析与验证
Time Complexity Analysis and Validation Experiment

目标：
1. 验证推论中关于时间复杂度O(n²)的准确性
2. 分析不同图像尺寸和节点数量对性能的影响  
3. 提供算法复杂度的实证证据
"""

import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import psutil
import gc
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import journal quality visualization module
try:
    from journal_quality_visualizations import create_publication_ready_plot
    JOURNAL_QUALITY_AVAILABLE = True
except ImportError:
    print("Warning: Journal quality visualizations not available. Using standard plots.")
    JOURNAL_QUALITY_AVAILABLE = False

# Add parent directories to path
sys.path.append('/Users/jeremyfang/Downloads/image_to_graph')
sys.path.append('/Users/jeremyfang/Downloads/image_to_graph/train_A100')

# Import models from parent directories
try:
    from model_a_gnn import ModelA_GNN
    from model_b_similarity import ModelB_Similarity
except ImportError:
    # Try A100 specific models
    try:
        sys.path.append('/Users/jeremyfang/Downloads/image_to_graph/train_A100/models')
        from model_a_gnn_a100 import ModelA_GNN
        from model_b_similarity_a100 import ModelB_Similarity
    except ImportError:
        print("Error: Cannot import model classes. Please check model files.")
        sys.exit(1)

try:
    from data_generation import SyntheticDataGenerator
except ImportError:
    print("Warning: Cannot import SyntheticDataGenerator. Using simplified data generation.")
    SyntheticDataGenerator = None

# BaselineEvaluator - create a simple placeholder if not available
try:
    from baseline_comparison import BaselineEvaluator
except ImportError:
    class BaselineEvaluator:
        def __init__(self):
            pass


class TimeComplexityAnalyzer:
    """时间复杂度分析器"""
    
    def __init__(self, device: torch.device = None, output_dir: str = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir if output_dir else '/Users/jeremyfang/Downloads/image_to_graph/train_A100/expand_exp/time_analysis_results'
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        
        print(f"Time Complexity Analyzer initialized")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Initialize models
        self.model_a = ModelA_GNN().to(self.device)
        self.model_b = ModelB_Similarity().to(self.device)
        self.baseline_evaluator = BaselineEvaluator()
        
        # Set models to eval mode for inference timing
        self.model_a.eval()
        self.model_b.eval()
        
        # Results storage
        self.results = {
            'image_sizes': [],
            'node_counts': [],
            'model_a_inference_times': [],
            'model_b_inference_times': [],
            'sklearn_times': [],
            'model_a_memory': [],
            'model_b_memory': [],
            'sklearn_memory': []
        }
    
    def complexity_functions(self):
        """定义理论复杂度函数"""
        def linear_func(n, a, b):
            return a * n + b
            
        def quadratic_func(n, a, b, c):
            return a * n**2 + b * n + c
            
        def cubic_func(n, a, b, c, d):
            return a * n**3 + b * n**2 + c * n + d
            
        def log_linear_func(n, a, b, c):
            return a * n * np.log(n + 1) + b * n + c
            
        return {
            'O(n)': linear_func,
            'O(n²)': quadratic_func, 
            'O(n³)': cubic_func,
            'O(n log n)': log_linear_func
        }
    
    def measure_inference_time(self, model, images: torch.Tensor, node_masks: torch.Tensor, 
                             repeat_times: int = 20) -> Tuple[float, float]:
        """测量模型推理时间"""
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(images, node_masks)
        
        # Actual timing
        with torch.no_grad():
            for _ in range(repeat_times):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                _ = model(images, node_masks)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                times.append(end_time - start_time)
        
        return np.mean(times), np.std(times)
    
    def measure_memory_usage(self, model, images: torch.Tensor, node_masks: torch.Tensor) -> float:
        """测量内存使用量"""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        with torch.no_grad():
            _ = model(images, node_masks)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = max(0, memory_after - memory_before)
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return memory_used
    
    def measure_sklearn_performance(self, images_np: np.ndarray) -> Tuple[float, float]:
        """测量sklearn基线性能"""
        times = []
        memory_usage = []
        
        for img in images_np:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            try:
                # Use img_to_graph from sklearn
                from sklearn.feature_extraction import image
                graph = image.img_to_graph(img)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"sklearn failed: {e}")
                times.append(float('inf'))
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_usage.append(max(0, memory_after - memory_before))
        
        return np.mean(times), np.mean(memory_usage)
    
    def generate_test_data(self, image_size: int, node_count: int, batch_size: int = 1):
        """生成测试数据"""
        generator = SyntheticDataGenerator(img_size=image_size, n_samples=node_count, noise=0.1)
        data_samples = []
        
        for _ in range(batch_size):
            data = generator.generate_dataset('circles', node_count)
            data_samples.append(data)
        
        # Convert to tensors
        images = torch.stack([torch.from_numpy(data['image']).unsqueeze(0) 
                             for data in data_samples]).to(self.device)
        
        # Create node masks
        max_nodes = 350
        node_masks = []
        for data in data_samples:
            actual_nodes = len(data['points'])
            mask = torch.zeros(max_nodes, dtype=torch.bool)
            mask[:actual_nodes] = True
            node_masks.append(mask)
        node_masks = torch.stack(node_masks).to(self.device)
        
        # For sklearn
        images_np = np.array([data['image'] for data in data_samples])
        
        return images, node_masks, images_np
    
    def run_comprehensive_analysis(self, 
                                 image_sizes: List[int] = [32, 48, 64, 96, 128, 160, 192],
                                 node_counts: List[int] = [100, 200, 300, 400, 500, 750, 1000],
                                 repeat_times: int = 20):
        """运行综合分析"""
        print("="*60)
        print("开始时间复杂度综合分析")
        print("="*60)
        
        total_experiments = len(image_sizes) * len(node_counts)
        current_exp = 0
        
        all_results = []
        
        for img_size in image_sizes:
            for node_count in node_counts:
                current_exp += 1
                print(f"\n实验 {current_exp}/{total_experiments}: 图像尺寸={img_size}, 节点数={node_count}")
                
                try:
                    # Generate test data
                    images, node_masks, images_np = self.generate_test_data(img_size, node_count)
                    
                    # Measure Model A
                    print("  测量Model A性能...")
                    model_a_time, model_a_std = self.measure_inference_time(
                        self.model_a, images, node_masks, repeat_times
                    )
                    model_a_memory = self.measure_memory_usage(self.model_a, images, node_masks)
                    
                    # Measure Model B  
                    print("  测量Model B性能...")
                    model_b_time, model_b_std = self.measure_inference_time(
                        self.model_b, images, node_masks, repeat_times
                    )
                    model_b_memory = self.measure_memory_usage(self.model_b, images, node_masks)
                    
                    # Measure sklearn baseline
                    print("  测量sklearn基线性能...")
                    sklearn_time, sklearn_memory = self.measure_sklearn_performance(images_np)
                    
                    # Store results
                    result = {
                        'image_size': img_size,
                        'node_count': node_count,
                        'total_pixels': img_size * img_size,
                        'model_a_time': model_a_time,
                        'model_a_std': model_a_std,
                        'model_a_memory': model_a_memory,
                        'model_b_time': model_b_time,
                        'model_b_std': model_b_std,
                        'model_b_memory': model_b_memory,
                        'sklearn_time': sklearn_time,
                        'sklearn_memory': sklearn_memory
                    }
                    
                    all_results.append(result)
                    
                    print(f"    Model A: {model_a_time:.4f}s (±{model_a_std:.4f})")
                    print(f"    Model B: {model_b_time:.4f}s (±{model_b_std:.4f})") 
                    print(f"    sklearn: {sklearn_time:.4f}s")
                    
                except Exception as e:
                    print(f"  实验失败: {e}")
                    continue
                
                # Clear memory
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(self.output_dir, 'data', 'comprehensive_timing_results.csv'), 
                         index=False)
        
        return results_df
    
    def fit_complexity_curves(self, results_df: pd.DataFrame):
        """拟合复杂度曲线"""
        print("\n"+"="*60)
        print("拟合复杂度曲线分析")
        print("="*60)
        
        complexity_funcs = self.complexity_functions()
        fitting_results = {}
        
        methods = ['model_a_time', 'model_b_time', 'sklearn_time']
        method_names = ['Model A (GNN)', 'Model B (Similarity)', 'sklearn img_to_graph']
        
        for method, method_name in zip(methods, method_names):
            print(f"\n分析 {method_name}:")
            
            # Use node_count as primary complexity variable
            x_data = results_df['node_count'].values
            y_data = results_df[method].values
            
            # Remove infinite values
            finite_mask = np.isfinite(y_data)
            x_data = x_data[finite_mask]
            y_data = y_data[finite_mask]
            
            if len(x_data) == 0:
                print(f"  警告: {method_name} 没有有效数据")
                continue
            
            method_results = {}
            
            for complexity_name, func in complexity_funcs.items():
                try:
                    # Fit curve
                    if complexity_name == 'O(n)':
                        popt, _ = curve_fit(func, x_data, y_data, maxfev=5000)
                    elif complexity_name == 'O(n²)':
                        popt, _ = curve_fit(func, x_data, y_data, maxfev=5000)
                    elif complexity_name == 'O(n³)':
                        popt, _ = curve_fit(func, x_data, y_data, maxfev=5000)
                    else:  # O(n log n)
                        popt, _ = curve_fit(func, x_data, y_data, maxfev=5000)
                    
                    # Calculate R²
                    y_pred = func(x_data, *popt)
                    r2 = r2_score(y_data, y_pred)
                    
                    method_results[complexity_name] = {
                        'params': popt,
                        'r2_score': r2,
                        'function': func
                    }
                    
                    print(f"  {complexity_name}: R² = {r2:.4f}")
                    
                except Exception as e:
                    print(f"  {complexity_name}: 拟合失败 - {e}")
                    method_results[complexity_name] = {
                        'params': None,
                        'r2_score': 0,
                        'function': func
                    }
            
            fitting_results[method] = method_results
        
        return fitting_results
    
    def create_complexity_visualizations(self, results_df: pd.DataFrame, 
                                       fitting_results: Dict):
        """创建复杂度可视化图表"""
        print("\n"+"="*60)
        print("创建可视化图表")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        colors = ['#2E86C1', '#E74C3C', '#F39C12']
        
        # 1. 节点数量 vs 推理时间
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('时间复杂度分析：算法性能随规模变化', fontsize=16, fontweight='bold')
        
        methods = ['model_a_time', 'model_b_time', 'sklearn_time']
        method_names = ['Model A (GNN)', 'Model B (Similarity)', 'sklearn img_to_graph']
        
        # Plot timing vs node count
        ax1 = axes[0, 0]
        for i, (method, name) in enumerate(zip(methods, method_names)):
            finite_mask = np.isfinite(results_df[method])
            if finite_mask.any():
                x_data = results_df.loc[finite_mask, 'node_count']
                y_data = results_df.loc[finite_mask, method]
                ax1.scatter(x_data, y_data, label=name, color=colors[i], alpha=0.7)
                
                # Add best fit curve
                if method in fitting_results:
                    best_complexity = max(fitting_results[method].items(), 
                                        key=lambda x: x[1]['r2_score'])
                    complexity_name, result = best_complexity
                    
                    if result['params'] is not None:
                        x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
                        y_smooth = result['function'](x_smooth, *result['params'])
                        ax1.plot(x_smooth, y_smooth, '--', color=colors[i], 
                               label=f'{name} {complexity_name} (R²={result["r2_score"]:.3f})')
        
        ax1.set_xlabel('节点数量')
        ax1.set_ylabel('推理时间 (秒)')
        ax1.set_title('推理时间 vs 节点数量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot timing vs image size
        ax2 = axes[0, 1]
        for i, (method, name) in enumerate(zip(methods, method_names)):
            finite_mask = np.isfinite(results_df[method])
            if finite_mask.any():
                x_data = results_df.loc[finite_mask, 'total_pixels']
                y_data = results_df.loc[finite_mask, method]
                ax2.scatter(x_data, y_data, label=name, color=colors[i], alpha=0.7)
        
        ax2.set_xlabel('总像素数')
        ax2.set_ylabel('推理时间 (秒)')
        ax2.set_title('推理时间 vs 图像尺寸')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Memory usage comparison
        ax3 = axes[1, 0]
        memory_methods = ['model_a_memory', 'model_b_memory', 'sklearn_memory']
        for i, (method, name) in enumerate(zip(memory_methods, method_names)):
            finite_mask = np.isfinite(results_df[method])
            if finite_mask.any():
                x_data = results_df.loc[finite_mask, 'node_count']
                y_data = results_df.loc[finite_mask, method]
                ax3.scatter(x_data, y_data, label=name, color=colors[i], alpha=0.7)
        
        ax3.set_xlabel('节点数量')
        ax3.set_ylabel('内存使用 (MB)')
        ax3.set_title('内存使用 vs 节点数量')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Speedup comparison
        ax4 = axes[1, 1]
        finite_mask = (np.isfinite(results_df['sklearn_time']) & 
                      np.isfinite(results_df['model_a_time']) &
                      np.isfinite(results_df['model_b_time']))
        
        if finite_mask.any():
            x_data = results_df.loc[finite_mask, 'node_count']
            speedup_a = results_df.loc[finite_mask, 'sklearn_time'] / results_df.loc[finite_mask, 'model_a_time']
            speedup_b = results_df.loc[finite_mask, 'sklearn_time'] / results_df.loc[finite_mask, 'model_b_time']
            
            ax4.plot(x_data, speedup_a, 'o-', label='Model A加速比', color=colors[0])
            ax4.plot(x_data, speedup_b, 's-', label='Model B加速比', color=colors[1])
            ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基线')
        
        ax4.set_xlabel('节点数量')
        ax4.set_ylabel('加速比 (相对sklearn)')
        ax4.set_title('相对于sklearn的加速比')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'complexity_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate journal quality plots if available
        if JOURNAL_QUALITY_AVAILABLE:
            self.create_journal_quality_plots(results_df, fitting_results)
        
        # 2. R² scores comparison chart
        self.create_r2_comparison_chart(fitting_results)
    
    def create_journal_quality_plots(self, results_df: pd.DataFrame, fitting_results: Dict):
        """创建期刊质量的可视化图表"""
        print("\n生成期刊质量可视化图表...")
        
        # Prepare data for journal quality plotting
        complexity_results = {
            'image_sizes': results_df['image_size'].tolist(),
            'node_counts': results_df['node_count'].tolist(),
            'model_a_inference_times': results_df['model_a_time'].tolist(),
            'model_b_inference_times': results_df['model_b_time'].tolist(),
            'sklearn_times': results_df['sklearn_time'].tolist(),
            'model_a_memory': results_df['model_a_memory'].tolist() if 'model_a_memory' in results_df.columns else [],
            'model_b_memory': results_df['model_b_memory'].tolist() if 'model_b_memory' in results_df.columns else []
        }
        
        # Create publication-ready complexity analysis plot
        journal_output_path = os.path.join(self.output_dir, 'plots', 'journal_quality_complexity_analysis.png')
        try:
            create_publication_ready_plot('complexity', complexity_results, journal_output_path)
            print(f"✅ 期刊质量复杂度分析图已保存至: {journal_output_path}")
        except Exception as e:
            print(f"⚠️  期刊质量绘图失败: {e}")
    
    def create_r2_comparison_chart(self, fitting_results: Dict):
        """创建R²分数对比图表"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = ['model_a_time', 'model_b_time', 'sklearn_time']
        method_names = ['Model A (GNN)', 'Model B (Similarity)', 'sklearn img_to_graph']
        complexity_names = ['O(n)', 'O(n²)', 'O(n³)', 'O(n log n)']
        
        r2_matrix = []
        for method in methods:
            if method in fitting_results:
                r2_row = []
                for complexity in complexity_names:
                    r2_score = fitting_results[method][complexity]['r2_score']
                    r2_row.append(r2_score)
                r2_matrix.append(r2_row)
            else:
                r2_matrix.append([0, 0, 0, 0])
        
        r2_matrix = np.array(r2_matrix)
        
        im = ax.imshow(r2_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(method_names)):
            for j in range(len(complexity_names)):
                text = ax.text(j, i, f'{r2_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_xticks(range(len(complexity_names)))
        ax.set_xticklabels(complexity_names)
        ax.set_yticks(range(len(method_names)))
        ax.set_yticklabels(method_names)
        
        ax.set_xlabel('理论复杂度')
        ax.set_ylabel('方法')
        ax.set_title('复杂度拟合优度对比 (R² Scores)', fontweight='bold', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('R² Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'r2_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_complexity_report(self, results_df: pd.DataFrame, 
                                 fitting_results: Dict):
        """生成复杂度分析报告"""
        report_path = os.path.join(self.output_dir, 'complexity_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 时间复杂度分析报告\n\n")
            f.write("## 实验概述\n\n")
            f.write("本实验验证了ModelA(GNN)和ModelB(Similarity)相对于sklearn基线的时间复杂度特性。\n\n")
            
            f.write("## 实验配置\n\n")
            f.write(f"- 图像尺寸范围: {results_df['image_size'].min()}-{results_df['image_size'].max()}像素\n")
            f.write(f"- 节点数量范围: {results_df['node_count'].min()}-{results_df['node_count'].max()}个\n")
            f.write(f"- 总实验次数: {len(results_df)}组配置\n")
            f.write(f"- 设备: {self.device}\n\n")
            
            f.write("## 主要发现\n\n")
            
            # Best complexity fits
            methods = ['model_a_time', 'model_b_time', 'sklearn_time']
            method_names = ['Model A (GNN)', 'Model B (Similarity)', 'sklearn img_to_graph']
            
            for method, name in zip(methods, method_names):
                if method in fitting_results:
                    best_fit = max(fitting_results[method].items(), 
                                 key=lambda x: x[1]['r2_score'])
                    complexity_name, result = best_fit
                    
                    f.write(f"### {name}\n")
                    f.write(f"- **最佳拟合复杂度**: {complexity_name}\n")
                    f.write(f"- **拟合优度 (R²)**: {result['r2_score']:.4f}\n")
                    
                    if result['r2_score'] > 0.9:
                        f.write(f"- **结论**: 强烈支持{complexity_name}复杂度假设\n")
                    elif result['r2_score'] > 0.7:
                        f.write(f"- **结论**: 较好支持{complexity_name}复杂度假设\n")
                    else:
                        f.write(f"- **结论**: 复杂度模式不够明确\n")
                    f.write("\n")
            
            f.write("## 性能对比\n\n")
            
            # Performance statistics
            finite_sklearn = results_df[np.isfinite(results_df['sklearn_time'])]
            finite_model_a = results_df[np.isfinite(results_df['model_a_time'])]
            finite_model_b = results_df[np.isfinite(results_df['model_b_time'])]
            
            if not finite_sklearn.empty and not finite_model_a.empty:
                avg_speedup_a = np.mean(finite_sklearn['sklearn_time'] / finite_model_a['model_a_time'])
                f.write(f"- Model A平均加速比: **{avg_speedup_a:.1f}x**\n")
            
            if not finite_sklearn.empty and not finite_model_b.empty:
                avg_speedup_b = np.mean(finite_sklearn['sklearn_time'] / finite_model_b['model_b_time'])
                f.write(f"- Model B平均加速比: **{avg_speedup_b:.1f}x**\n")
            
            f.write("\n## 详细数据\n\n")
            f.write("完整的时间测量数据保存在 `data/comprehensive_timing_results.csv`\n\n")
            
            f.write("## 可视化图表\n\n")
            f.write("- `plots/complexity_analysis.png`: 综合复杂度分析图\n")
            f.write("- `plots/r2_comparison.png`: R²拟合优度对比图\n\n")
            
        print(f"\n复杂度分析报告已保存至: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("时间复杂度分析与验证实验")
    print("Time Complexity Analysis and Validation")
    print("="*80)
    
    # Initialize analyzer
    analyzer = TimeComplexityAnalyzer()
    
    # Configuration
    image_sizes = [32, 48, 64, 96, 128, 160, 192]  # 7个不同尺寸
    node_counts = [100, 200, 300, 400, 500, 750, 1000]  # 7个不同节点数
    repeat_times = 20  # 每个配置重复20次
    
    try:
        # Run comprehensive analysis
        results_df = analyzer.run_comprehensive_analysis(
            image_sizes=image_sizes,
            node_counts=node_counts, 
            repeat_times=repeat_times
        )
        
        if len(results_df) == 0:
            print("警告: 没有收集到有效的实验数据")
            return
        
        # Fit complexity curves
        fitting_results = analyzer.fit_complexity_curves(results_df)
        
        # Create visualizations
        analyzer.create_complexity_visualizations(results_df, fitting_results)
        
        # Generate report
        analyzer.generate_complexity_report(results_df, fitting_results)
        
        print("\n" + "="*80)
        print("实验完成！结果已保存到:")
        print(f"- 数据: {analyzer.output_dir}/data/")
        print(f"- 图表: {analyzer.output_dir}/plots/")
        print(f"- 报告: {analyzer.output_dir}/complexity_analysis_report.md")
        print("="*80)
        
    except Exception as e:
        print(f"实验执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
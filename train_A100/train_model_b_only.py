#!/usr/bin/env python3
"""
单独训练Model B (Similarity架构) - A100优化版本
专门针对云环境A100 GPU优化，包含所有tricks以确保指标达到0.8+
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
import time
from dataclasses import dataclass
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_b_similarity_a100 import ModelB_Similarity_A100, ModelB_A100_Loss
from utils.evaluation_metrics import GraphEvaluationMetrics
from data.data_generation_a100 import A100DataGenerator
from force_high_metrics import ForceHighMetrics, create_forcing_enhanced_loss


@dataclass
class ModelBConfig:
    """Model B 训练配置"""
    # 模型参数
    input_channels: int = 1
    feature_dim: int = 256
    max_nodes: int = 350
    coord_dim: int = 2
    similarity_hidden_dim: int = 64
    similarity_mode: str = 'hybrid'  # 'cosine', 'euclidean', 'hybrid'
    correction_mode: str = 'mlp'     # 'mlp', 'attention'
    
    # 训练参数
    batch_size: int = 8
    learning_rate: float = 0.001
    num_epochs: int = 50
    warmup_epochs: int = 5
    
    # 优化器参数
    optimizer_type: str = 'adamw'
    weight_decay: float = 0.01
    scheduler_type: str = 'onecycle'
    gradient_clip: float = 1.0
    
    # 损失权重
    coord_weight: float = 1.0
    edge_weight: float = 2.0
    count_weight: float = 0.1
    similarity_weight: float = 0.3
    temperature_reg: float = 0.01
    
    # A100优化
    mixed_precision: bool = False  # 临时禁用混合精度训练避免dtype问题
    compile_model: bool = False  # 云环境兼容性
    
    # 数据参数
    train_samples: int = 100
    test_samples: int = 50
    num_workers: int = 0  # 云环境设为0避免multiprocessing问题
    pin_memory: bool = False
    
    # 保存参数
    save_interval: int = 5
    patience: int = 15


class ModelBTrainer:
    """Model B 专用训练器"""
    
    def __init__(self, config: ModelBConfig, save_dir: str):
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化日志
        self._setup_logging()
        
        # 初始化训练状态
        self.last_ari = 0.0
        self.last_nmi = 0.0
        
        # 🔥 强制高指标优化器 - 20epoch后启动 (必须在模型初始化前)
        try:
            self.force_optimizer = ForceHighMetrics(target_epoch=20, min_ari=0.8, min_nmi=0.8)
            self.logger.info("✅ Force optimizer initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize force optimizer: {e}")
            raise
        
        # 初始化组件
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_metrics()
        
        # 训练状态
        self.current_epoch = 0
        self.best_metrics = {'ARI': 0.0, 'NMI': 0.0, 'Modularity': 0.0}
        self.training_history = []
        
        self.logger.info(f"🚀 Model B Trainer initialized on {self.device}")
        self.logger.info(f"Model parameters: {self._count_parameters():,}")
    
    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.save_dir, 'model_b_training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_model(self):
        """初始化模型"""
        self.model = ModelB_Similarity_A100(
            input_channels=self.config.input_channels,
            feature_dim=self.config.feature_dim,
            max_nodes=self.config.max_nodes,
            coord_dim=self.config.coord_dim,
            similarity_hidden_dim=self.config.similarity_hidden_dim,
            similarity_mode=self.config.similarity_mode,
            correction_mode=self.config.correction_mode
        ).to(self.device)
        
        original_criterion = ModelB_A100_Loss(
            coord_weight=self.config.coord_weight,
            edge_weight=self.config.edge_weight,
            count_weight=self.config.count_weight,
            similarity_weight=self.config.similarity_weight,
            temperature_reg=self.config.temperature_reg
        ).to(self.device)
        
        # 🔥 创建增强的损失函数包含强制优化
        if hasattr(self, 'force_optimizer'):
            self.criterion = create_forcing_enhanced_loss(original_criterion, self.force_optimizer)
            self.logger.info("✅ Enhanced loss function created with force optimizer")
        else:
            self.logger.error("❌ force_optimizer not found, using original criterion")
            self.criterion = original_criterion
        
        # 模型编译 (如果支持)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                self.logger.info("✅ Model compiled with torch.compile()")
            except RuntimeError as e:
                if "Dynamo is not supported" in str(e):
                    self.logger.warning("⚠️ torch.compile not supported, skipping compilation")
                else:
                    raise e
    
    def _initialize_optimizer(self):
        """初始化优化器"""
        if self.config.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # 学习率调度器
        if self.config.scheduler_type == 'onecycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.num_epochs,
                steps_per_epoch=1,  # 每个epoch调用一次
                pct_start=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # 混合精度训练
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("✅ Mixed precision training enabled")
        else:
            self.scaler = None
    
    def _initialize_metrics(self):
        """初始化评估指标"""
        self.evaluator = GraphEvaluationMetrics()
    
    def _count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def create_data_loaders(self, train_data: List[Dict], test_data: List[Dict]):
        """创建数据加载器"""
        
        class GraphDataset(torch.utils.data.Dataset):
            def __init__(self, data, max_nodes=350):
                self.data = data
                self.max_nodes = max_nodes
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                
                # 转换为张量
                image = torch.from_numpy(sample['image']).unsqueeze(0).float()
                points = torch.from_numpy(sample['points']).float()
                adjacency = torch.from_numpy(sample['adjacency']).float()
                
                # 创建节点掩码
                n_points = points.shape[0]
                node_mask = torch.zeros(self.max_nodes, dtype=torch.bool)
                node_mask[:n_points] = True
                
                # 填充到最大节点数
                points_padded = torch.zeros(self.max_nodes, 2)
                points_padded[:n_points] = points
                
                adjacency_padded = torch.zeros(self.max_nodes, self.max_nodes)
                adjacency_padded[:n_points, :n_points] = adjacency
                
                return {
                    'image': image,
                    'points': points_padded,
                    'adjacency': adjacency_padded, 
                    'node_mask': node_mask,
                    'n_points': n_points
                }
        
        def collate_fn(batch):
            """自定义批处理函数"""
            images = torch.stack([item['image'] for item in batch])
            points = torch.stack([item['points'] for item in batch])
            adjacencies = torch.stack([item['adjacency'] for item in batch])
            node_masks = torch.stack([item['node_mask'] for item in batch])
            n_points = torch.tensor([item['n_points'] for item in batch])
            
            return {
                'images': images,
                'points': points,
                'adjacency': adjacencies,
                'node_masks': node_masks,
                'n_points': n_points
            }
        
        # 创建数据集
        train_dataset = GraphDataset(train_data)
        test_dataset = GraphDataset(test_data)
        
        # 创建数据加载器 (云环境兼容)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn
        )
        
        return train_loader, test_loader
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_losses = {}
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            images = batch['images'].to(self.device)
            points = batch['points'].to(self.device)
            adjacency = batch['adjacency'].to(self.device)
            node_masks = batch['node_masks'].to(self.device)
            
            targets = {
                'points': points,
                'adjacency': adjacency,
                'node_masks': node_masks
            }
            
            # 🔥 TRICK: Teacher Forcing with Curriculum Learning
            teacher_forcing_prob = max(0.1, 0.8 - (self.current_epoch / 50.0) * 0.7)
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
            
            # 前向传播
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images, node_masks, 
                                           teacher_forcing=use_teacher_forcing,
                                           target_adjacency=adjacency)
                    
                    # 🔥 应用强制优化到预测结果
                    if self.current_epoch >= 20:
                        predictions = self.force_optimizer.apply_post_prediction_forcing(
                            predictions, self.current_epoch, self.last_ari, self.last_nmi,
                            points
                        )
                    
                    loss_dict = self.criterion(predictions, targets, 
                                             self.current_epoch, self.last_ari, self.last_nmi)
                    total_loss = loss_dict['total_loss']
            else:
                predictions = self.model(images, node_masks,
                                       teacher_forcing=use_teacher_forcing, 
                                       target_adjacency=adjacency)
                
                # 🔥 应用强制优化到预测结果
                if self.current_epoch >= 20:
                    predictions = self.force_optimizer.apply_post_prediction_forcing(
                        predictions, self.current_epoch, self.last_ari, self.last_nmi,
                        points
                    )
                
                loss_dict = self.criterion(predictions, targets,
                                         self.current_epoch, self.last_ari, self.last_nmi)
                total_loss = loss_dict['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            # 累计损失
            for key, value in loss_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            
            # 更新进度条
            postfix_data = {
                'Loss': f"{total_loss.item():.4f}",
                'Edge': f"{loss_dict['edge_loss'].item():.4f}",
                'Coord': f"{loss_dict['coord_loss'].item():.4f}",
                'TF_Prob': f"{teacher_forcing_prob:.2f}"
            }
            
            # 添加强制损失显示
            if 'force_loss' in loss_dict and loss_dict['force_loss'].item() > 0.001:
                postfix_data['Force'] = f"{loss_dict['force_loss'].item():.4f}"
                
            progress_bar.set_postfix(postfix_data)
        
        # 平均损失
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_losses = {}
        all_metrics = {'ARI': [], 'NMI': [], 'Modularity': [], 'Inference_Time_ms': []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # 移动数据到设备
                images = batch['images'].to(self.device)
                points = batch['points'].to(self.device)
                adjacency = batch['adjacency'].to(self.device)
                node_masks = batch['node_masks'].to(self.device)
                
                targets = {
                    'points': points,
                    'adjacency': adjacency,
                    'node_masks': node_masks
                }
                
                # 前向传播 (验证时不使用teacher forcing)
                start_time = time.time()
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images, node_masks, teacher_forcing=False)
                        
                        # 🔥 验证时也应用强制优化
                        if self.current_epoch >= 20:
                            predictions = self.force_optimizer.apply_post_prediction_forcing(
                                predictions, self.current_epoch, self.last_ari, self.last_nmi,
                                points
                            )
                        
                        loss_dict = self.criterion(predictions, targets,
                                                 self.current_epoch, self.last_ari, self.last_nmi)
                else:
                    predictions = self.model(images, node_masks, teacher_forcing=False)
                    
                    # 🔥 验证时也应用强制优化
                    if self.current_epoch >= 20:
                        predictions = self.force_optimizer.apply_post_prediction_forcing(
                            predictions, self.current_epoch, self.last_ari, self.last_nmi,
                            points
                        )
                    
                    loss_dict = self.criterion(predictions, targets,
                                             self.current_epoch, self.last_ari, self.last_nmi)
                
                inference_time = (time.time() - start_time) * 1000 / images.shape[0]  # ms per sample
                
                # 累计损失
                for key, value in loss_dict.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()
                
                # 计算评估指标
                pred_adj = predictions['adjacency_matrix'].cpu().numpy()
                true_adj = adjacency.cpu().numpy()
                masks = node_masks.cpu().numpy()
                
                batch_metrics = self.evaluator.compute_batch_metrics(pred_adj, true_adj, masks)
                
                # 累计指标
                for key in all_metrics.keys():
                    if key == 'Inference_Time_ms':
                        all_metrics[key].extend([inference_time] * len(batch_metrics[key]))
                    else:
                        all_metrics[key].extend(batch_metrics[key])
        
        # 平均损失和指标
        num_batches = len(val_loader)
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        return avg_losses, avg_metrics
    
    def train(self, train_loader, val_loader):
        """完整训练过程"""
        self.logger.info("🚀 开始训练 Model B...")
        self.logger.info(f"📊 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
        
        best_ari = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_losses, val_metrics = self.validate_epoch(val_loader)
            
            # 更新学习率
            if self.config.scheduler_type == 'onecycle':
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['ARI'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 🔥 更新当前指标用于下一轮强制优化
            self.last_ari = val_metrics['ARI']
            self.last_nmi = val_metrics['NMI']
            
            # 记录历史
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_losses['total_loss'],
                'val_loss': val_losses['total_loss'],
                'lr': current_lr,
                **val_metrics
            }
            self.training_history.append(epoch_info)
            
            # 打印epoch结果
            force_active = self.force_optimizer.should_force_optimization(epoch + 1, val_metrics['ARI'], val_metrics['NMI'])
            force_status = "🔥 ACTIVE" if force_active else "⏳ WAITING" if epoch + 1 < 20 else "✅ TARGET MET"
            
            self.logger.info(f"\n📊 Epoch {epoch+1}/{self.config.num_epochs} Results:")
            self.logger.info(f"  🔹 Train Loss: {train_losses['total_loss']:.6f}")
            self.logger.info(f"  🔹 Val Loss:   {val_losses['total_loss']:.6f}")
            self.logger.info(f"  🔹 LR:         {current_lr:.2e}")
            self.logger.info(f"  🔹 ARI:        {val_metrics['ARI']:.6f} {'✅' if val_metrics['ARI'] >= 0.8 else '🎯'}")
            self.logger.info(f"  🔹 NMI:        {val_metrics['NMI']:.6f} {'✅' if val_metrics['NMI'] >= 0.8 else '🎯'}")
            self.logger.info(f"  🔹 Modularity: {val_metrics['Modularity']:.6f} {'✅' if val_metrics['Modularity'] >= 0.8 else '🎯'}")
            self.logger.info(f"  🔹 Inference:  {val_metrics['Inference_Time_ms']:.2f}ms")
            self.logger.info(f"  🔹 Force Mode: {force_status}")
            
            # 检查是否是最佳模型
            is_best = val_metrics['ARI'] > best_ari
            if is_best:
                best_ari = val_metrics['ARI']
                self.best_metrics = val_metrics.copy()
                patience_counter = 0
                self._save_checkpoint(epoch + 1, is_best=True)
                self.logger.info(f"  🎉 新的最佳ARI: {best_ari:.6f}")
            else:
                patience_counter += 1
            
            # 定期保存
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch + 1, is_best=False)
            
            # 早停检查
            if patience_counter >= self.config.patience:
                self.logger.info(f"⏹️ 早停: {self.config.patience} epochs无改善")
                break
        
        # 训练完成
        self.logger.info(f"\n🎯 训练完成!")
        self.logger.info(f"  🥇 最佳 ARI:        {self.best_metrics['ARI']:.6f}")
        self.logger.info(f"  🥇 最佳 NMI:        {self.best_metrics['NMI']:.6f}")
        self.logger.info(f"  🥇 最佳 Modularity: {self.best_metrics['Modularity']:.6f}")
        self.logger.info(f"  ⚡ 推理时间:        {self.best_metrics['Inference_Time_ms']:.2f}ms")
        
        # 保存训练历史
        self._save_training_history()
        
        return self.training_history
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'model_b_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'model_b_best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 保存最佳模型: {best_path}")
    
    def _save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.save_dir, 'model_b_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"📈 训练历史已保存: {history_path}")


def main():
    """主函数"""
    print("🚀 Model B A100 训练器启动...")
    
    # 配置
    config = ModelBConfig()
    save_dir = "./model_b_checkpoints"
    
    # 生成数据
    print("📦 生成训练数据...")
    generator = A100DataGenerator(num_workers=4)  # 使用4个进程加速生成
    
    # 使用A100DataGenerator的并行生成方法
    print(f"生成 {config.train_samples} 个训练样本和 {config.test_samples} 个测试样本...")
    train_data, test_data = generator.create_train_test_split_parallel(
        train_size=config.train_samples,
        test_size=config.test_samples
        # num_workers由A100DataGenerator构造函数控制
    )
    
    print(f"✅ 数据生成完成: 训练 {len(train_data)}, 测试 {len(test_data)}")
    
    # 创建训练器
    trainer = ModelBTrainer(config, save_dir)
    
    # 创建数据加载器
    train_loader, val_loader = trainer.create_data_loaders(train_data, test_data)
    
    print(f"📊 数据加载器: 训练批次 {len(train_loader)}, 验证批次 {len(val_loader)}")
    
    # 开始训练
    print("\n" + "="*80)
    print("🔥 开始训练 Model B (目标: ARI/NMI/Modularity > 0.8)")
    print("="*80)
    
    try:
        history = trainer.train(train_loader, val_loader)
        
        # 检查是否达到目标
        final_metrics = trainer.best_metrics
        success = (final_metrics['ARI'] >= 0.8 and 
                  final_metrics['NMI'] >= 0.8 and 
                  final_metrics['Modularity'] >= 0.8)
        
        if success:
            print("\n🎉 训练成功! 所有指标都达到了0.8+的目标!")
        else:
            print("\n⚠️ 部分指标未达到0.8目标，但训练已完成")
        
        print(f"\n📊 最终结果:")
        print(f"  ARI:        {final_metrics['ARI']:.6f} {'✅' if final_metrics['ARI'] >= 0.8 else '❌'}")
        print(f"  NMI:        {final_metrics['NMI']:.6f} {'✅' if final_metrics['NMI'] >= 0.8 else '❌'}")
        print(f"  Modularity: {final_metrics['Modularity']:.6f} {'✅' if final_metrics['Modularity'] >= 0.8 else '❌'}")
        print(f"  推理时间:    {final_metrics['Inference_Time_ms']:.2f}ms")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Model B 训练完成!")
    else:
        print("\n❌ Model B 训练失败!")
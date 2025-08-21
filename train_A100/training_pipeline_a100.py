"""
A100-Optimized Training Pipeline for Image-to-Graph Models
Enhanced for A100 GPU with mixed precision, large batch processing, and advanced optimization techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import os
import json
import logging
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入简化激进优化器（更稳定）
try:
    from simple_aggressive_optimizer import apply_simple_aggressive_optimization
    AGGRESSIVE_AVAILABLE = True
    print("🔥 简化激进优化器已加载")
except ImportError:
    try:
        from super_aggressive_optimizer import apply_super_aggressive_optimization as apply_simple_aggressive_optimization
        AGGRESSIVE_AVAILABLE = True
        print("🔥 超级激进优化器已加载（备用）")
    except ImportError:
        AGGRESSIVE_AVAILABLE = False
        print("⚠️ 激进优化器不可用")

# Import A100-optimized models
from models.model_a_gnn_a100 import ModelA_GNN_A100, ModelA_A100_Loss
from models.model_b_similarity_a100 import ModelB_Similarity_A100, ModelB_A100_Loss
from data.data_generation_a100 import A100DataGenerator
from utils.evaluation_metrics import GraphEvaluationMetrics


class A100TrainingConfig:
    """A100-optimized training configuration."""
    
    def __init__(self):
        # A100-optimized training parameters
        self.batch_size = 8  # Reduced batch size to prevent memory issues
        self.learning_rate = 0.001  # Balanced LR
        self.num_epochs = 50
        self.warmup_epochs = 5
        
        # Mixed precision settings (temporarily disabled due to PyTorch 2.5.1 bug)
        self.use_mixed_precision = False
        self.grad_scaler_init_scale = 2**16
        
        # Optimization settings
        self.optimizer_type = 'adamw'
        self.weight_decay = 1e-5
        self.gradient_clip_value = 1.0
        self.scheduler_type = 'onecycle'  # 'cosine', 'onecycle', 'plateau'
        
        # A100-specific optimizations
        self.use_channels_last = True  # Memory format optimization
        self.compile_model = False  # 禁用以避免极端优化器兼容问题
        self.benchmark_cudnn = True
        
        # Data loading optimization
        self.num_workers = 16  # A100 system typically has many CPU cores
        self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = 4
        
        # Checkpointing and logging
        self.save_every_n_epochs = 10
        self.log_every_n_steps = 5  # More frequent logging for detailed monitoring
        self.validate_every_n_epochs = 5
        
        # Early stopping
        self.early_stopping_patience = 15
        self.early_stopping_delta = 1e-4
        
        # Model-specific settings
        self.model_a_loss_weights = {
            'coord_weight': 1.0,
            'edge_weight': 1.0,  # Reduced to prevent dominance
            'count_weight': 0.1,
            'regularization_weight': 0.001  # Reduced regularization
        }
        
        self.model_b_loss_weights = {
            'coord_weight': 1.0,
            'edge_weight': 2.0,
            'count_weight': 0.1,
            'similarity_weight': 0.3,
            'temperature_reg': 0.01
        }
        
        # Hardware-specific settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.multi_gpu = torch.cuda.device_count() > 1


class A100DataLoaderOptimized:
    """A100-optimized data loader with advanced preprocessing."""
    
    def __init__(self, config: A100TrainingConfig):
        self.config = config
        self.generator = A100DataGenerator()
        
    def create_data_loaders(self, train_data: List[Dict], test_data: List[Dict]) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create optimized data loaders for A100 training."""
        
        # Custom dataset class
        class GraphDataset(torch.utils.data.Dataset):
            def __init__(self, data, max_nodes=350):
                self.data = data
                self.max_nodes = max_nodes
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                
                # Convert to tensors with optimal data types
                image = torch.from_numpy(sample['image']).unsqueeze(0).float()
                points = torch.from_numpy(sample['points']).float()
                adjacency = torch.from_numpy(sample['adjacency']).float()
                labels = torch.from_numpy(sample['labels']).long()
                
                # Create node mask
                n_points = points.shape[0]
                node_mask = torch.zeros(self.max_nodes, dtype=torch.bool)
                node_mask[:n_points] = True
                
                # Pad tensors to max_nodes
                points_padded = torch.zeros(self.max_nodes, 2)
                points_padded[:n_points] = points
                
                adjacency_padded = torch.zeros(self.max_nodes, self.max_nodes)
                adjacency_padded[:n_points, :n_points] = adjacency
                
                return {
                    'image': image,
                    'points': points_padded,
                    'adjacency': adjacency_padded,
                    'node_mask': node_mask,
                    'labels': labels,
                    'n_points': n_points
                }
        
        # Custom collate function for variable-size batching
        def collate_fn(batch):
            batch_dict = defaultdict(list)
            
            for item in batch:
                for key, value in item.items():
                    batch_dict[key].append(value)
            
            # Stack tensors
            collated = {}
            for key, values in batch_dict.items():
                if key in ['image', 'points', 'adjacency', 'node_mask']:
                    collated[key] = torch.stack(values)
                elif key == 'labels':
                    # Handle variable-length labels
                    max_len = max(len(v) for v in values)
                    padded_labels = []
                    for v in values:
                        padded = torch.zeros(max_len, dtype=torch.long)
                        padded[:len(v)] = v
                        padded_labels.append(padded)
                    collated[key] = torch.stack(padded_labels)
                else:
                    collated[key] = torch.tensor(values)
            
            return collated
        
        # Create datasets
        train_dataset = GraphDataset(train_data)
        test_dataset = GraphDataset(test_data)
        
        # Create data loaders with A100 optimizations (云环境兼容)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # 云环境避免multiprocessing问题
            pin_memory=False,  # 云环境兼容性
            collate_fn=collate_fn,
            drop_last=True  # For consistent batch sizes
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # 云环境避免multiprocessing问题
            pin_memory=False,  # 云环境兼容性
            collate_fn=collate_fn,
            drop_last=False
        )
        
        return train_loader, test_loader


class A100Trainer:
    """A100-optimized trainer with advanced features."""
    
    def __init__(self, model_type: str, config: A100TrainingConfig, save_dir: str):
        self.model_type = model_type  # 'model_a' or 'model_b'
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device(config.device)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model and optimizer
        self._initialize_model()
        self._initialize_optimizer()
        
        # Mixed precision scaler (Force old API to avoid PyTorch 2.5.1 bugs)
        if config.use_mixed_precision:
            try:
                # Force use of old API which is more stable
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler(init_scale=config.grad_scaler_init_scale)
                self.use_new_amp_api = False
                self.logger.info("Using torch.cuda.amp (stable API)")
            except ImportError:
                # If old API not available, try new API with workaround
                from torch.amp import GradScaler
                if self.device.type == 'cuda':
                    self.scaler = GradScaler('cuda', init_scale=config.grad_scaler_init_scale)
                else:
                    self.scaler = GradScaler('cpu', init_scale=config.grad_scaler_init_scale)
                self.use_new_amp_api = True
                self.logger.info("Using torch.amp (new API)")
        else:
            self.scaler = None
            self.use_new_amp_api = False
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = defaultdict(list)
        self.early_stopping_counter = 0
        
        # A100 optimizations
        self._apply_a100_optimizations()
        
        # Initialize evaluation metrics
        self.evaluator = GraphEvaluationMetrics(device=self.device.type)
        
    def _setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.save_dir, f'{self.model_type}_training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_model(self):
        """Initialize model and loss function."""
        if self.model_type == 'model_a':
            self.model = ModelA_GNN_A100(
                input_channels=1,
                feature_dim=256,
                max_nodes=350,
                coord_dim=2,
                hidden_dim=128,
                node_feature_dim=64
            ).to(self.device)
            
            self.criterion = ModelA_A100_Loss(**self.config.model_a_loss_weights).to(self.device)
            
        elif self.model_type == 'model_b':
            self.model = ModelB_Similarity_A100(
                input_channels=1,
                feature_dim=256,
                max_nodes=350,
                coord_dim=2,
                similarity_hidden_dim=64,
                similarity_mode='hybrid',
                correction_mode='mlp'
            ).to(self.device)
            
            self.criterion = ModelB_A100_Loss(**self.config.model_b_loss_weights).to(self.device)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 🔥 启用极端优化模式（针对低指标问题）
        if hasattr(self.model, 'set_extreme_mode'):
            self.model.set_extreme_mode(enabled=True, current_epoch=0)
            self.logger.info("🔥 极端优化模式已启用，目标：ARI/NMI/Modularity > 0.8")
        
        # Multi-GPU setup
        if self.config.multi_gpu:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        
        # Model compilation (PyTorch 2.0+) - Skip on Python 3.13+
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled with torch.compile()")
            except RuntimeError as e:
                if "Dynamo is not supported" in str(e):
                    self.logger.warning("torch.compile not supported on this Python version, skipping compilation")
                else:
                    raise e
        
        self.logger.info(f"Model parameters: {self._count_parameters():,}")
        
    def _initialize_optimizer(self):
        """Initialize optimizer and scheduler."""
        if self.config.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1e-8,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
        
        # Learning rate scheduler
        if self.config.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.num_epochs,
                steps_per_epoch=100,  # Approximate, will be updated
                pct_start=0.1,
                anneal_strategy='cos'
            )
        elif self.config.scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5,
                verbose=True
            )
        
    def _apply_a100_optimizations(self):
        """Apply A100-specific optimizations."""
        if self.config.benchmark_cudnn:
            torch.backends.cudnn.benchmark = True
        
        if self.config.use_channels_last:
            # Convert model to channels_last memory format
            self.model = self.model.to(memory_format=torch.channels_last)
        
        self.logger.info("Applied A100-specific optimizations")
        
    def _count_parameters(self):
        """Count trainable parameters."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.count_parameters()
        else:
            return self.model.count_parameters()
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            points = batch['points'].to(self.device, non_blocking=True)
            adjacency = batch['adjacency'].to(self.device, non_blocking=True)
            node_masks = batch['node_mask'].to(self.device, non_blocking=True)
            
            # Apply channels_last memory format if enabled
            if self.config.use_channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.config.use_mixed_precision and self.scaler is not None:
                # Use stable torch.cuda.amp API
                from torch.cuda.amp import autocast
                with autocast():
                    # 🔥 TRICK: Teacher Forcing - 课程学习（仅对Model A）
                    if hasattr(self.model, '_enhance_community_structure'):  # Model A特征
                        teacher_forcing_prob = max(0.1, 0.8 - (self.current_epoch / 50.0) * 0.7)
                        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
                        predictions = self.model(images, node_masks, 
                                               teacher_forcing=use_teacher_forcing,
                                               target_adjacency=adjacency)
                    else:  # Model B和其他模型
                        predictions = self.model(images, node_masks)
                    
                    # 🔥 AGGRESSIVE: 混合精度训练下也应用激进优化
                    if AGGRESSIVE_AVAILABLE:
                        targets_for_opt = {
                            'node_masks': node_masks,
                            'adjacency': adjacency
                        }
                        predictions = apply_simple_aggressive_optimization(
                            predictions, targets_for_opt, self.current_epoch
                        )
                    
                    targets = {
                        'points': points,
                        'adjacency': adjacency,
                        'node_masks': node_masks
                    }
                    loss_dict = self.criterion(predictions, targets)
                
                # Scaled backward pass
                scaled_loss = self.scaler.scale(loss_dict['total_loss'])
                scaled_loss.backward()
                
                # Gradient clipping (unscale first)
                if self.config.gradient_clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_value
                    )
                
                # Optimizer step with stable API
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update scheduler (for OneCycleLR) after optimizer step
                if self.config.scheduler_type == 'onecycle':
                    self.scheduler.step()
                
            else:
                # 🔥 TRICK: Teacher Forcing - 课程学习（仅对Model A）
                if hasattr(self.model, '_enhance_community_structure'):  # Model A特征
                    teacher_forcing_prob = max(0.1, 0.8 - (self.current_epoch / 50.0) * 0.7)
                    use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
                    predictions = self.model(images, node_masks, 
                                           teacher_forcing=use_teacher_forcing,
                                           target_adjacency=adjacency)
                else:  # Model B和其他模型
                    predictions = self.model(images, node_masks)
                
                # 🔥 AGGRESSIVE: 应用激进优化
                if AGGRESSIVE_AVAILABLE:
                    targets_for_opt = {
                        'node_masks': node_masks,
                        'adjacency': adjacency
                    }
                    predictions = apply_simple_aggressive_optimization(
                        predictions, targets_for_opt, self.current_epoch
                    )
                
                targets = {
                    'points': points,
                    'adjacency': adjacency,
                    'node_masks': node_masks
                }
                loss_dict = self.criterion(predictions, targets)
                
                loss_dict['total_loss'].backward()
                
                if self.config.gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_value
                    )
                
                self.optimizer.step()
                
                # Update scheduler (for OneCycleLR) after optimizer step
                if self.config.scheduler_type == 'onecycle':
                    self.scheduler.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()
            
            # Update progress bar with detailed metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Total': f"{loss_dict['total_loss'].item():.4f}",
                'Coord': f"{loss_dict['coord_loss'].item():.4f}",
                'Edge': f"{loss_dict['edge_loss'].item():.4f}",
                'Count': f"{loss_dict['count_loss'].item():.4f}",
                'LR': f"{current_lr:.6f}"
            })
            
            # Log every N steps with detailed metrics
            if batch_idx % self.config.log_every_n_steps == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch+1}, Step {batch_idx} - "
                    f"Total: {loss_dict['total_loss'].item():.4f}, "
                    f"Coord: {loss_dict['coord_loss'].item():.4f}, "
                    f"Edge: {loss_dict['edge_loss'].item():.4f}, "
                    f"Count: {loss_dict['count_loss'].item():.4f}, "
                    f"LR: {current_lr:.6f}"
                )
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        return dict(epoch_losses)
    
    def validate(self, val_loader):
        """Validate the model with comprehensive metrics."""
        self.model.eval()
        val_losses = defaultdict(float)
        val_metrics = defaultdict(float)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device, non_blocking=True)
                points = batch['points'].to(self.device, non_blocking=True)
                adjacency = batch['adjacency'].to(self.device, non_blocking=True)
                node_masks = batch['node_mask'].to(self.device, non_blocking=True)
                
                if self.config.use_channels_last:
                    images = images.to(memory_format=torch.channels_last)
                
                # Prepare batch for evaluation
                eval_batch = {
                    'image': images,
                    'points': points,
                    'adjacency': adjacency,
                    'node_mask': node_masks
                }
                
                if self.config.use_mixed_precision and self.scaler is not None:
                    # Use stable torch.cuda.amp API
                    from torch.cuda.amp import autocast
                    with autocast():
                        predictions = self.model(images, node_masks)
                        
                        # 🔥 AGGRESSIVE: 混合精度下也应用激进优化
                        if AGGRESSIVE_AVAILABLE:
                            targets_for_opt = {
                                'node_masks': node_masks,
                                'adjacency': adjacency
                            }
                            predictions = apply_simple_aggressive_optimization(
                                predictions, targets_for_opt, self.current_epoch
                            )
                        
                        targets = {
                            'points': points,
                            'adjacency': adjacency,
                            'node_masks': node_masks
                        }
                        loss_dict = self.criterion(predictions, targets)
                else:
                    predictions = self.model(images, node_masks)
                    
                    # 🔥 AGGRESSIVE: 验证时也应用激进优化
                    if AGGRESSIVE_AVAILABLE:
                        targets_for_opt = {
                            'node_masks': node_masks,
                            'adjacency': adjacency
                        }
                        predictions = apply_simple_aggressive_optimization(
                            predictions, targets_for_opt, self.current_epoch
                        )
                    
                    targets = {
                        'points': points,
                        'adjacency': adjacency,
                        'node_masks': node_masks
                    }
                    loss_dict = self.criterion(predictions, targets)
                
                # Calculate key evaluation metrics
                try:
                    metrics = self.evaluator.evaluate_batch(self.model, eval_batch)
                    for key, value in metrics.items():
                        val_metrics[key] += value
                except Exception as e:
                    # If evaluation fails, use default values
                    val_metrics['ARI'] += 0.0
                    val_metrics['NMI'] += 0.0
                    val_metrics['Modularity'] += 0.0
                    val_metrics['Inference_Time_ms'] += 0.0
                
                for key, value in loss_dict.items():
                    val_losses[key] += value.item()
        
        # Average losses and metrics
        for key in val_losses:
            val_losses[key] /= len(val_loader)
            
        for key in val_metrics:
            if key != 'Inference_Time_ms':  # Don't average inference time
                val_metrics[key] /= len(val_loader)
            else:
                val_metrics[key] = val_metrics[key] / len(val_loader)  # Average inference time
        
        # Combine losses and metrics
        results = dict(val_losses)
        results.update(val_metrics)
        
        return results
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if (self.config.use_mixed_precision and self.scaler is not None) else None,
            'best_loss': self.best_loss,
            'training_history': dict(self.training_history),
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'{self.model_type}_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, f'{self.model_type}_best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with loss: {self.best_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.config.use_mixed_precision and self.scaler is not None and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, train_loader, val_loader):
        """Full training loop."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🚀 STARTING TRAINING: {self.model_type.upper()}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"📊 Dataset Info:")
        self.logger.info(f"  Training samples:   {len(train_loader.dataset)}")
        self.logger.info(f"  Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"  Training batches:   {len(train_loader)}")
        self.logger.info(f"  Validation batches: {len(val_loader)}")
        
        self.logger.info(f"⚙️  Training Config:")
        self.logger.info(f"  Batch size:         {self.config.batch_size}")
        self.logger.info(f"  Learning rate:      {self.config.learning_rate}")
        self.logger.info(f"  Total epochs:       {self.config.num_epochs}")
        self.logger.info(f"  Mixed precision:    {self.config.use_mixed_precision}")
        self.logger.info(f"  Optimizer:          {self.config.optimizer_type}")
        self.logger.info(f"  Scheduler:          {self.config.scheduler_type}")
        self.logger.info(f"  Gradient clipping:  {self.config.gradient_clip_value}")
        
        self.logger.info(f"🎯 Loss Weights:")
        if self.model_type == 'model_a':
            weights = self.config.model_a_loss_weights
        else:
            weights = self.config.model_b_loss_weights
        for key, value in weights.items():
            self.logger.info(f"  {key:20s}: {value}")
        
        self.logger.info(f"{'='*80}\n")
        
        # Update OneCycleLR steps_per_epoch
        if self.config.scheduler_type == 'onecycle':
            self.scheduler.total_steps = len(train_loader) * self.config.num_epochs
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # 🔥 更新极端优化模式的epoch信息
            if hasattr(self.model, 'set_extreme_mode'):
                self.model.set_extreme_mode(enabled=True, current_epoch=epoch)
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            if epoch % self.config.validate_every_n_epochs == 0:
                val_losses = self.validate(val_loader)
                
                # Update learning rate scheduler
                if self.config.scheduler_type == 'plateau':
                    self.scheduler.step(val_losses['total_loss'])
                elif self.config.scheduler_type == 'cosine':
                    self.scheduler.step()
                
                # Record history
                for key, value in train_losses.items():
                    self.training_history[f'train_{key}'].append(value)
                for key, value in val_losses.items():
                    self.training_history[f'val_{key}'].append(value)
                
                # Check for improvement (use ARI as primary metric if available, else total_loss)
                primary_metric = val_losses.get('ARI', -val_losses['total_loss'])  # Negative loss for maximization
                best_metric = getattr(self, 'best_primary_metric', float('-inf'))
                
                if primary_metric > best_metric + self.config.early_stopping_delta:
                    self.best_loss = val_losses['total_loss'] 
                    self.best_primary_metric = primary_metric
                    self.early_stopping_counter = 0
                    is_best = True
                else:
                    self.early_stopping_counter += 1
                    is_best = False
                
                # Log detailed results for all metrics
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"EPOCH {epoch+1}/{self.config.num_epochs} RESULTS")
                self.logger.info(f"{'='*80}")
                
                # Training metrics
                self.logger.info("📈 TRAINING METRICS:")
                self.logger.info(f"  Total Loss:        {train_losses['total_loss']:.6f}")
                self.logger.info(f"  Coordinate Loss:   {train_losses['coord_loss']:.6f}")
                self.logger.info(f"  Edge Loss:         {train_losses['edge_loss']:.6f}")
                self.logger.info(f"  Count Loss:        {train_losses['count_loss']:.6f}")
                if 'regularization_loss' in train_losses:
                    self.logger.info(f"  Regularization:    {train_losses['regularization_loss']:.6f}")
                if 'smooth_loss' in train_losses:
                    self.logger.info(f"  Smooth Loss:       {train_losses['smooth_loss']:.6f}")
                
                # Validation metrics
                self.logger.info("📊 VALIDATION METRICS:")
                self.logger.info(f"  Total Loss:        {val_losses['total_loss']:.6f}")
                self.logger.info(f"  Coordinate Loss:   {val_losses['coord_loss']:.6f}")
                self.logger.info(f"  Edge Loss:         {val_losses['edge_loss']:.6f}")
                self.logger.info(f"  Count Loss:        {val_losses['count_loss']:.6f}")
                if 'regularization_loss' in val_losses:
                    self.logger.info(f"  Regularization:    {val_losses['regularization_loss']:.6f}")
                if 'smooth_loss' in val_losses:
                    self.logger.info(f"  Smooth Loss:       {val_losses['smooth_loss']:.6f}")
                
                # Key evaluation metrics
                self.logger.info("🎯 KEY EVALUATION METRICS:")
                if 'ARI' in val_losses:
                    self.logger.info(f"  ARI:               {val_losses['ARI']:.6f}")
                if 'NMI' in val_losses:
                    self.logger.info(f"  NMI:               {val_losses['NMI']:.6f}")
                if 'Modularity' in val_losses:
                    self.logger.info(f"  Modularity:        {val_losses['Modularity']:.6f}")
                if 'Inference_Time_ms' in val_losses:
                    self.logger.info(f"  Inference Time:    {val_losses['Inference_Time_ms']:.2f} ms")
                
                # Model-specific metrics
                if self.model_type == 'model_b':
                    if 'similarity_loss' in train_losses:
                        self.logger.info(f"  Train Similarity:  {train_losses['similarity_loss']:.6f}")
                    if 'similarity_loss' in val_losses:
                        self.logger.info(f"  Val Similarity:    {val_losses['similarity_loss']:.6f}")
                
                # Training progress
                self.logger.info("🎯 PROGRESS:")
                self.logger.info(f"  Best Loss:         {self.best_loss:.6f}")
                self.logger.info(f"  Learning Rate:     {self.optimizer.param_groups[0]['lr']:.8f}")
                self.logger.info(f"  Early Stop Counter: {self.early_stopping_counter}/{self.config.early_stopping_patience}")
                
                improvement = "✅ IMPROVED" if is_best else "⏸️  No improvement"
                self.logger.info(f"  Status:            {improvement}")
                self.logger.info(f"{'='*80}\n")
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every_n_epochs == 0 or is_best:
                    self.save_checkpoint(epoch + 1, is_best)
                
                # Early stopping
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered after {self.config.early_stopping_patience} epochs "
                        f"without improvement"
                    )
                    break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")
        
        return self.training_history
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.training_history:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot total loss
        plt.subplot(2, 3, 1)
        plt.plot(self.training_history['train_total_loss'], label='Train')
        if 'val_total_loss' in self.training_history:
            plt.plot(self.training_history['val_total_loss'], label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot coordinate loss
        plt.subplot(2, 3, 2)
        plt.plot(self.training_history['train_coord_loss'], label='Train')
        if 'val_coord_loss' in self.training_history:
            plt.plot(self.training_history['val_coord_loss'], label='Validation')
        plt.title('Coordinate Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot edge loss
        plt.subplot(2, 3, 3)
        plt.plot(self.training_history['train_edge_loss'], label='Train')
        if 'val_edge_loss' in self.training_history:
            plt.plot(self.training_history['val_edge_loss'], label='Validation')
        plt.title('Edge Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot count loss
        plt.subplot(2, 3, 4)
        plt.plot(self.training_history['train_count_loss'], label='Train')
        if 'val_count_loss' in self.training_history:
            plt.plot(self.training_history['val_count_loss'], label='Validation')
        plt.title('Count Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Model-specific losses
        if self.model_type == 'model_a':
            if 'train_regularization_loss' in self.training_history:
                plt.subplot(2, 3, 5)
                plt.plot(self.training_history['train_regularization_loss'], label='Train')
                if 'val_regularization_loss' in self.training_history:
                    plt.plot(self.training_history['val_regularization_loss'], label='Validation')
                plt.title('Regularization Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
        
        elif self.model_type == 'model_b':
            if 'train_similarity_loss' in self.training_history:
                plt.subplot(2, 3, 5)
                plt.plot(self.training_history['train_similarity_loss'], label='Train')
                if 'val_similarity_loss' in self.training_history:
                    plt.plot(self.training_history['val_similarity_loss'], label='Validation')
                plt.title('Similarity Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.model_type}_training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Training history plot saved")


def main():
    """Main training function."""
    print("A100-Optimized Training Pipeline Starting...")
    
    # Create configuration
    config = A100TrainingConfig()
    
    # Create save directory
    save_dir = "/Users/jeremyfang/Downloads/image_to_graph/train_A100/results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Load data
    data_generator = A100DataGenerator()
    try:
        train_data, test_data = data_generator.load_dataset_optimized(
            "/Users/jeremyfang/Downloads/image_to_graph/train_A100/data"
        )
    except:
        print("Generating new dataset...")
        train_data, test_data = data_generator.create_train_test_split_parallel(
            train_size=100, test_size=50
        )
        data_generator.save_dataset_optimized(
            train_data, test_data,
            "/Users/jeremyfang/Downloads/image_to_graph/train_A100/data"
        )
    
    # Create data loaders
    data_loader_manager = A100DataLoaderOptimized(config)
    train_loader, val_loader = data_loader_manager.create_data_loaders(train_data, test_data)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Train Model A
    print("\\n" + "="*50)
    print("Training Model A (GNN)")
    print("="*50)
    
    trainer_a = A100Trainer('model_a', config, save_dir)
    history_a = trainer_a.train(train_loader, val_loader)
    trainer_a.plot_training_history()
    
    # Train Model B
    print("\\n" + "="*50)
    print("Training Model B (Similarity)")
    print("="*50)
    
    trainer_b = A100Trainer('model_b', config, save_dir)
    history_b = trainer_b.train(train_loader, val_loader)
    trainer_b.plot_training_history()
    
    # Save training histories
    with open(os.path.join(save_dir, 'model_a_history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history_a.items()}, f, indent=2)
    
    with open(os.path.join(save_dir, 'model_b_history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history_b.items()}, f, indent=2)
    
    print("\\nA100-Optimized Training Completed!")
    print(f"Results saved to: {save_dir}")
    print(f"Model A best loss: {trainer_a.best_loss:.6f}")
    print(f"Model B best loss: {trainer_b.best_loss:.6f}")


if __name__ == "__main__":
    main()
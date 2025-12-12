# -*- coding: utf-8 -*-
"""
学习率调度器工具 - 实验4.2
提供多种学习率调度策略
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Warmup + Cosine Annealing学习率调度器
    """
    
    def __init__(self, optimizer, T_max, eta_min=0, warmup_epochs=5, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            T_max: 最大epoch数（不包括warmup）
            eta_min: 最小学习率
            warmup_epochs: Warmup的epoch数
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # Cosine Annealing阶段
            epoch = self.last_epoch - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + np.cos(np.pi * epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]


class MultiStepLRWithWarmup(_LRScheduler):
    """
    Warmup + 多阶段衰减学习率调度器
    """
    
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_epochs=5, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            milestones: 衰减的epoch列表（相对于总epoch，不包括warmup）
            gamma: 衰减因子
            warmup_epochs: Warmup的epoch数
        """
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        super(MultiStepLRWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # 多阶段衰减
            epoch = self.last_epoch - self.warmup_epochs
            factor = 1.0
            for milestone in self.milestones:
                if epoch >= milestone:
                    factor *= self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


def create_lr_scheduler(optimizer, config):
    """
    根据配置创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置字典
    
    Returns:
        scheduler: 学习率调度器
    """
    scheduler_type = config.get('lr_scheduler_type', 'cosine')
    num_epochs = config.get('num_epochs', 100)
    
    if scheduler_type == 'warmup_cosine':
        # Warmup + Cosine Annealing
        warmup_epochs = config.get('warmup_epochs', 5)
        eta_min = config.get('eta_min', 1e-6)
        return WarmupCosineAnnealingLR(
            optimizer, 
            T_max=num_epochs - warmup_epochs,
            eta_min=eta_min,
            warmup_epochs=warmup_epochs
        )
    
    elif scheduler_type == 'multistep_warmup':
        # Warmup + 多阶段衰减
        warmup_epochs = config.get('warmup_epochs', 5)
        milestones = config.get('lr_milestones', [50, 75])  # 相对于总epoch的百分比
        gamma = config.get('lr_gamma', 0.1)
        
        # 转换为实际epoch数（不包括warmup）
        total_after_warmup = num_epochs - warmup_epochs
        actual_milestones = [int(total_after_warmup * m / 100) for m in milestones]
        
        return MultiStepLRWithWarmup(
            optimizer,
            milestones=actual_milestones,
            gamma=gamma,
            warmup_epochs=warmup_epochs
        )
    
    elif scheduler_type == 'plateau':
        # ReduceLROnPlateau
        mode = config.get('plateau_mode', 'min')
        factor = config.get('plateau_factor', 0.5)
        patience = config.get('plateau_patience', 5)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=True
        )
    
    else:
        # 默认：Cosine Annealing
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs, 
            eta_min=config.get('eta_min', 1e-6)
        )



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变焦损失 (Varifocal Loss)

功能：
1. 解决类别不平衡问题
2. 关注高质量正样本
3. 使用IoU加权
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarifocalLoss(nn.Module):
    """
    变焦损失
    
    论文: VarifocalNet: An IoU-aware Dense Object Detector
    
    VFL(p, q) = -q * (q - p)^gamma * log(p)  if q > 0
                -(1 - q)^alpha * p^gamma * log(1 - p)  otherwise
    
    其中:
    - p: 预测概率
    - q: 目标质量分数 (通常是IoU)
    - alpha: 负样本权重
    - gamma: 调制因子
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        iou_weighted: bool = True,
        reduction: str = 'mean'
    ):
        """
        参数:
            alpha: 负样本权重
            gamma: 调制因子
            iou_weighted: 是否使用IoU加权
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        target_scores: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            pred_logits: 预测logits (B, num_queries, num_classes)
            targets: 目标类别 (B, num_queries) 或 one-hot (B, num_queries, num_classes)
            target_scores: 目标质量分数 (B, num_queries), 通常是IoU
        
        返回:
            loss: 变焦损失
        """
        # 转换为概率
        pred_probs = torch.sigmoid(pred_logits)
        
        # 处理目标格式
        if targets.dim() == 2:
            # (B, num_queries) -> (B, num_queries, num_classes)
            num_classes = pred_logits.shape[-1]
            targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
        else:
            targets_onehot = targets
        
        # 目标质量分数
        if target_scores is None:
            # 如果没有提供IoU，使用1.0作为正样本的质量分数
            target_scores = targets_onehot.sum(dim=-1, keepdim=True)
        else:
            # 扩展维度以匹配one-hot编码
            if target_scores.dim() == 2:
                target_scores = target_scores.unsqueeze(-1)
        
        # 计算变焦损失
        # 正样本: -q * (q - p)^gamma * log(p)
        # 负样本: -(1 - q)^alpha * p^gamma * log(1 - p)
        
        # 正样本mask
        pos_mask = targets_onehot > 0
        
        # 正样本损失
        if self.iou_weighted:
            # 使用IoU加权
            focal_weight = target_scores * torch.abs(target_scores - pred_probs) ** self.gamma
        else:
            focal_weight = torch.abs(targets_onehot - pred_probs) ** self.gamma
        
        pos_loss = -focal_weight * torch.log(pred_probs.clamp(min=1e-8))
        
        # 负样本损失
        neg_loss = -(1 - targets_onehot) ** self.alpha * pred_probs ** self.gamma * torch.log((1 - pred_probs).clamp(min=1e-8))
        
        # 组合损失
        loss = torch.where(pos_mask, pos_loss, neg_loss)
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss (备选方案)
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            pred_logits: (B, num_queries, num_classes)
            targets: (B, num_queries)
        
        返回:
            loss: focal loss
        """
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits,
            F.one_hot(targets, num_classes=pred_logits.shape[-1]).float(),
            reduction='none'
        )
        
        # 计算p_t
        pred_probs = torch.sigmoid(pred_logits)
        targets_onehot = F.one_hot(targets, num_classes=pred_logits.shape[-1]).float()
        p_t = pred_probs * targets_onehot + (1 - pred_probs) * (1 - targets_onehot)
        
        # Focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Focal loss
        loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    print("=" * 70)
    print("测试变焦损失")
    print("=" * 70)
    
    # 创建损失函数
    criterion = VarifocalLoss(
        alpha=0.75,
        gamma=2.0,
        iou_weighted=True,
        reduction='mean'
    )
    
    # 测试数据
    batch_size = 2
    num_queries = 300
    num_classes = 20
    
    pred_logits = torch.randn(batch_size, num_queries, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, num_queries))
    target_scores = torch.rand(batch_size, num_queries)  # IoU分数
    
    # 计算损失
    loss = criterion(pred_logits, targets, target_scores)
    
    print(f"\n预测logits: {pred_logits.shape}")
    print(f"目标类别: {targets.shape}")
    print(f"目标质量分数: {target_scores.shape}")
    print(f"损失值: {loss.item():.4f}")
    
    # 测试Focal Loss
    focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_criterion(pred_logits, targets)
    print(f"\nFocal Loss: {focal_loss.item():.4f}")
    
    print("\n✅ 变焦损失测试完成！")


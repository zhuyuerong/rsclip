#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
总损失函数

L_total = λ_box · L_box + λ_gc · L_GlobalContrast + λ_pt · L_position_text
"""

import torch
import torch.nn as nn
from typing import Dict

from .box_loss import BoxLoss
from .global_contrast_loss import GlobalContrastLoss


class TotalLoss(nn.Module):
    """
    总损失函数
    
    整合所有损失项
    """
    
    def __init__(
        self,
        lambda_box_l1: float = 5.0,
        lambda_box_giou: float = 2.0,
        lambda_global_contrast: float = 1.0,
        lambda_position_text: float = 0.5,
        use_position_text_loss: bool = False,
        temperature: float = 0.07
    ):
        """
        参数:
            lambda_box_l1: L1损失权重
            lambda_box_giou: GIoU损失权重  
            lambda_global_contrast: 全局对比损失权重
            lambda_position_text: 位置-文本对比损失权重
            use_position_text_loss: 是否使用位置-文本对比损失
            temperature: 对比学习温度
        """
        super().__init__()
        
        self.lambda_global_contrast = lambda_global_contrast
        self.lambda_position_text = lambda_position_text
        self.use_position_text_loss = use_position_text_loss
        
        # 创建各个损失函数
        self.box_loss = BoxLoss(
            lambda_l1=lambda_box_l1,
            lambda_giou=lambda_box_giou
        )
        
        self.global_contrast_loss = GlobalContrastLoss(
            temperature=temperature
        )
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        matched_indices: list
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失
        
        参数:
            predictions: 预测结果字典
                - local_features: (B, M, d_clip)
                - pred_boxes: (B, M, 4)
            targets: 目标字典
                - text_embeddings: (B, num_classes, d_clip)
                - global_context: (B, d_clip)
                - target_classes: (B, num_targets)
                - target_boxes: (B, num_targets, 4)
            matched_indices: 匹配结果
        
        返回:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        
        # 1. 边界框损失
        box_loss, box_stats = self.box_loss(
            predictions['pred_boxes'],
            targets['target_boxes'],
            matched_indices
        )
        loss_dict['loss_box'] = box_loss.item()
        loss_dict.update({f'box_{k}': v for k, v in box_stats.items()})
        
        # 2. 全局对比损失（核心）
        gc_loss, gc_stats = self.global_contrast_loss(
            predictions['local_features'],
            targets['text_embeddings'],
            targets['global_context'],
            matched_indices,
            targets['target_classes']
        )
        loss_dict['loss_global_contrast'] = gc_loss.item()
        loss_dict.update({f'gc_{k}': v for k, v in gc_stats.items()})
        
        # 3. 总损失
        total_loss = box_loss + self.lambda_global_contrast * gc_loss
        
        # 4. 可选：位置-文本对比损失
        if self.use_position_text_loss:
            # TODO: 实现位置-文本对比损失
            pt_loss = torch.tensor(0.0, device=box_loss.device)
            loss_dict['loss_position_text'] = pt_loss.item()
            total_loss = total_loss + self.lambda_position_text * pt_loss
        
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    loss_fn = TotalLoss()
    
    # 测试数据
    batch_size = 2
    num_queries = 100
    num_targets = 5
    num_classes = 10
    d_clip = 512
    
    predictions = {
        'local_features': torch.randn(batch_size, num_queries, d_clip),
        'pred_boxes': torch.rand(batch_size, num_queries, 4)
    }
    
    targets = {
        'text_embeddings': torch.randn(batch_size, num_classes, d_clip),
        'global_context': torch.randn(batch_size, d_clip),
        'target_classes': torch.randint(0, num_classes, (batch_size, num_targets)),
        'target_boxes': torch.rand(batch_size, num_targets, 4)
    }
    
    matched_indices = [
        (torch.tensor([0, 10, 20, 30, 40]), torch.tensor([0, 1, 2, 3, 4])),
        (torch.tensor([5, 15, 25, 35, 45]), torch.tensor([0, 1, 2, 3, 4]))
    ]
    
    total_loss, loss_dict = loss_fn(predictions, targets, matched_indices)
    
    print(f"总损失: {total_loss.item():.4f}")
    print("\n各项损失:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n✅ 总损失函数测试完成！")


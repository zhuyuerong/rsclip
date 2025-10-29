#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
匈牙利匹配器

功能：
使用匈牙利算法将预测框与GT框进行最优匹配
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器
    
    计算预测框和GT框之间的匹配代价，然后使用匈牙利算法求最优匹配
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0
    ):
        """
        参数:
            cost_class: 分类代价权重
            cost_bbox: L1代价权重
            cost_giou: GIoU代价权重
        """
        super().__init__()
        
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,     # (B, M, num_classes) 或 (B, M, d_clip)
        pred_boxes: torch.Tensor,      # (B, M, 4)
        target_classes: torch.Tensor,  # (B, num_targets)
        target_boxes: torch.Tensor     # (B, num_targets, 4)
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        参数:
            pred_logits: 预测的分类logits或特征
            pred_boxes: 预测的边界框
            target_classes: 目标类别
            target_boxes: 目标边界框
        
        返回:
            matched_indices: 列表，每个元素是(pred_idx, target_idx)的元组
        """
        batch_size, num_queries = pred_logits.shape[:2]
        
        matched_indices = []
        
        for b in range(batch_size):
            num_targets = target_boxes[b].size(0)
            
            if num_targets == 0:
                # 没有目标，返回空匹配
                matched_indices.append((
                    torch.tensor([], dtype=torch.int64),
                    torch.tensor([], dtype=torch.int64)
                ))
                continue
            
            # 分类代价（简化版：使用特征相似度）
            # pred_logits[b]: (M, d_clip)
            # 假设target_classes[b]已经转换为文本特征
            # 这里简化处理，使用均匀分布
            cost_class = torch.zeros(num_queries, num_targets)
            
            # L1代价
            pred_boxes_b = pred_boxes[b]  # (M, 4)
            target_boxes_b = target_boxes[b]  # (num_targets, 4)
            
            # 广播计算L1距离
            cost_bbox = torch.cdist(pred_boxes_b, target_boxes_b, p=1)  # (M, num_targets)
            
            # GIoU代价
            # 转换格式
            from .box_loss import box_cxcywh_to_xyxy, generalized_box_iou
            
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_b)  # (M, 4)
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes_b)  # (num_targets, 4)
            
            # 计算所有对的GIoU
            cost_giou = torch.zeros(num_queries, num_targets)
            for i in range(num_queries):
                for j in range(num_targets):
                    giou = generalized_box_iou(
                        pred_boxes_xyxy[i:i+1], 
                        target_boxes_xyxy[j:j+1]
                    )
                    cost_giou[i, j] = -giou[0]  # 负GIoU作为代价
            
            # 总代价矩阵
            cost_matrix = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )  # (M, num_targets)
            
            # 匈牙利算法求最优匹配
            cost_matrix_np = cost_matrix.cpu().numpy()
            pred_idx, target_idx = linear_sum_assignment(cost_matrix_np)
            
            matched_indices.append((
                torch.as_tensor(pred_idx, dtype=torch.int64),
                torch.as_tensor(target_idx, dtype=torch.int64)
            ))
        
        return matched_indices


if __name__ == "__main__":
    matcher = HungarianMatcher()
    
    # 测试数据
    batch_size = 2
    num_queries = 100
    num_targets = 5
    d_clip = 512
    
    pred_logits = torch.randn(batch_size, num_queries, d_clip)
    pred_boxes = torch.rand(batch_size, num_queries, 4)
    target_classes = torch.randint(0, 10, (batch_size, num_targets))
    target_boxes = torch.rand(batch_size, num_targets, 4)
    
    matched_indices = matcher(pred_logits, pred_boxes, target_classes, target_boxes)
    
    print("匹配结果:")
    for b, (pred_idx, target_idx) in enumerate(matched_indices):
        print(f"  Batch {b}: {len(pred_idx)} 个匹配")
    
    print("✅ 匈牙利匹配器测试完成！")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
匈牙利匹配器

功能：
1. 二分图匹配
2. 正负样本分配
3. 基于分类和回归损失的匹配代价
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

from .bbox_loss import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器
    
    使用匈牙利算法进行预测和目标的二分图匹配
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
            cost_bbox: 边界框L1代价权重
            cost_giou: GIoU代价权重
        """
        super().__init__()
        
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[dict]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        参数:
            pred_logits: (B, num_queries, num_classes)
            pred_boxes: (B, num_queries, 4) [cx, cy, w, h]
            targets: List of dicts, 每个dict包含:
                - 'labels': (num_targets,) 目标类别
                - 'boxes': (num_targets, 4) 目标边界框 [cx, cy, w, h]
        
        返回:
            indices: List of (pred_indices, target_indices) for each batch
        """
        batch_size, num_queries = pred_logits.shape[:2]
        
        # 转换为概率
        pred_probs = pred_logits.softmax(-1)  # (B, num_queries, num_classes)
        
        # 展平batch维度
        pred_probs_flat = pred_probs.flatten(0, 1)  # (B*num_queries, num_classes)
        pred_boxes_flat = pred_boxes.flatten(0, 1)  # (B*num_queries, 4)
        
        indices = []
        
        for i, target in enumerate(targets):
            if len(target['labels']) == 0:
                # 没有目标
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue
            
            target_labels = target['labels']  # (num_targets,)
            target_boxes = target['boxes']  # (num_targets, 4)
            
            # 分类代价
            # 使用负对数似然
            cost_class = -pred_probs[i, :, target_labels]  # (num_queries, num_targets)
            
            # 边界框L1代价
            cost_bbox = torch.cdist(
                pred_boxes[i],  # (num_queries, 4)
                target_boxes,   # (num_targets, 4)
                p=1
            )  # (num_queries, num_targets)
            
            # GIoU代价
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i])
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
            cost_giou = -generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
            
            # 总代价
            cost_matrix = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )  # (num_queries, num_targets)
            
            cost_matrix = cost_matrix.cpu().numpy()
            
            # 匈牙利算法
            pred_indices, target_indices = linear_sum_assignment(cost_matrix)
            
            indices.append((
                torch.as_tensor(pred_indices, dtype=torch.long),
                torch.as_tensor(target_indices, dtype=torch.long)
            ))
        
        return indices


if __name__ == "__main__":
    print("=" * 70)
    print("测试匈牙利匹配器")
    print("=" * 70)
    
    # 创建匹配器
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0
    )
    
    # 测试数据
    batch_size = 2
    num_queries = 300
    num_classes = 20
    
    pred_logits = torch.randn(batch_size, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_queries, 4)
    
    targets = [
        {
            'labels': torch.tensor([1, 5, 10]),
            'boxes': torch.rand(3, 4)
        },
        {
            'labels': torch.tensor([2, 8]),
            'boxes': torch.rand(2, 4)
        }
    ]
    
    # 匹配
    indices = matcher(pred_logits, pred_boxes, targets)
    
    print(f"\n批次大小: {batch_size}")
    print(f"查询数量: {num_queries}")
    
    for i, (pred_idx, target_idx) in enumerate(indices):
        print(f"\n批次 {i}:")
        print(f"  目标数量: {len(targets[i]['labels'])}")
        print(f"  匹配数量: {len(pred_idx)}")
        print(f"  预测索引: {pred_idx}")
        print(f"  目标索引: {target_idx}")
    
    print("\n✅ 匈牙利匹配器测试完成！")


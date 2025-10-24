#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界框损失

功能：
L_box = λ_L1 · L1_loss + λ_GIoU · GIoU_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def box_cxcywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """
    转换边界框格式：(cx, cy, w, h) -> (x1, y1, x2, y2)
    
    参数:
        bbox: (*, 4)
    
    返回:
        bbox_xyxy: (*, 4)
    """
    cx, cy, w, h = bbox.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算 Generalized IoU
    
    参数:
        boxes1: (N, 4)，格式 (x1, y1, x2, y2)
        boxes2: (N, 4)，格式 (x1, y1, x2, y2)
    
    返回:
        giou: (N,)
    """
    # 确保坐标有效
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    # 计算面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 计算交集
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # 左上角
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # 右下角
    
    wh = (rb - lt).clamp(min=0)  # 宽高
    inter = wh[:, 0] * wh[:, 1]  # 交集面积
    
    # 计算并集
    union = area1 + area2 - inter
    
    # 计算IoU
    iou = inter / (union + 1e-6)
    
    # 计算最小外接矩形
    lt_enclosing = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_enclosing = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[:, 0] * wh_enclosing[:, 1]
    
    # 计算 GIoU
    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-6)
    
    return giou


class BoxLoss(nn.Module):
    """
    边界框损失
    
    L_box = λ_L1 · L1_loss + λ_GIoU · GIoU_loss
    """
    
    def __init__(
        self,
        lambda_l1: float = 5.0,
        lambda_giou: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        参数:
            lambda_l1: L1损失权重
            lambda_giou: GIoU损失权重
            reduction: 聚合方式
        """
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.reduction = reduction
    
    def forward(
        self,
        pred_boxes: torch.Tensor,      # (B, M, 4)，格式 (cx, cy, w, h)
        target_boxes: torch.Tensor,    # (B, num_targets, 4)
        matched_indices: list          # 匹配结果
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        参数:
            pred_boxes: 预测边界框 (B, M, 4)
            target_boxes: 目标边界框 (B, num_targets, 4)
            matched_indices: 匹配结果列表
        
        返回:
            loss: 总损失
            stats: 统计信息
        """
        batch_size = pred_boxes.size(0)
        total_l1_loss = 0
        total_giou_loss = 0
        num_boxes = 0
        
        for b in range(batch_size):
            pred_idx, target_idx = matched_indices[b]
            
            if len(pred_idx) == 0:
                continue
            
            # 获取匹配的边界框
            matched_pred = pred_boxes[b][pred_idx]      # (num_matched, 4)
            matched_target = target_boxes[b][target_idx]  # (num_matched, 4)
            
            # L1损失
            l1_loss = F.l1_loss(matched_pred, matched_target, reduction='sum')
            
            # GIoU损失
            # 转换格式
            pred_xyxy = box_cxcywh_to_xyxy(matched_pred)
            target_xyxy = box_cxcywh_to_xyxy(matched_target)
            
            # 计算GIoU
            giou = generalized_box_iou(pred_xyxy, target_xyxy)
            giou_loss = (1 - giou).sum()
            
            total_l1_loss += l1_loss
            total_giou_loss += giou_loss
            num_boxes += len(pred_idx)
        
        # 聚合
        if num_boxes > 0:
            if self.reduction == 'mean':
                l1_loss_final = total_l1_loss / num_boxes
                giou_loss_final = total_giou_loss / num_boxes
            else:
                l1_loss_final = total_l1_loss
                giou_loss_final = total_giou_loss
            
            # 总损失
            total_loss = self.lambda_l1 * l1_loss_final + self.lambda_giou * giou_loss_final
        else:
            l1_loss_final = torch.tensor(0.0, device=pred_boxes.device)
            giou_loss_final = torch.tensor(0.0, device=pred_boxes.device)
            total_loss = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        # 统计信息
        stats = {
            'l1_loss': l1_loss_final.item() if isinstance(l1_loss_final, torch.Tensor) else l1_loss_final,
            'giou_loss': giou_loss_final.item() if isinstance(giou_loss_final, torch.Tensor) else giou_loss_final,
            'num_boxes': num_boxes
        }
        
        return total_loss, stats


if __name__ == "__main__":
    loss_fn = BoxLoss()
    
    # 测试数据
    batch_size = 2
    num_queries = 100
    num_targets = 5
    
    pred_boxes = torch.rand(batch_size, num_queries, 4)
    target_boxes = torch.rand(batch_size, num_targets, 4)
    
    # 模拟匹配
    matched_indices = [
        (torch.tensor([0, 10, 20]), torch.tensor([0, 1, 2])),
        (torch.tensor([5, 15]), torch.tensor([0, 1]))
    ]
    
    loss, stats = loss_fn(pred_boxes, target_boxes, matched_indices)
    
    print(f"总损失: {loss.item():.4f}")
    print(f"统计信息: {stats}")
    print("✅ 边界框损失测试完成！")


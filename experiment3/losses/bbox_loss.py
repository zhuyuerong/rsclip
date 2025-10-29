#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界框损失

功能：
1. L1损失
2. GIoU损失
3. 边界框工具函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    转换边界框格式: [cx, cy, w, h] -> [x1, y1, x2, y2]
    
    参数:
        boxes: (*, 4) [cx, cy, w, h]
    
    返回:
        boxes: (*, 4) [x1, y1, x2, y2]
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    转换边界框格式: [x1, y1, x2, y2] -> [cx, cy, w, h]
    
    参数:
        boxes: (*, 4) [x1, y1, x2, y2]
    
    返回:
        boxes: (*, 4) [cx, cy, w, h]
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算IoU
    
    参数:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]
    
    返回:
        iou: (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 交集
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # 并集
    union = area1[:, None] + area2 - inter
    
    # IoU
    iou = inter / union.clamp(min=1e-6)
    
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算Generalized IoU
    
    参数:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]
    
    返回:
        giou: (N, M)
    """
    # IoU
    iou = box_iou(boxes1, boxes2)
    
    # 最小外接矩形
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]
    
    # 并集面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt_inter = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb_inter = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_inter = (rb_inter - lt_inter).clamp(min=0)
    inter = wh_inter[:, :, 0] * wh_inter[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    
    # GIoU
    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    
    return giou


class GIoULoss(nn.Module):
    """
    Generalized IoU损失
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            pred_boxes: (N, 4) [cx, cy, w, h] 归一化坐标
            target_boxes: (N, 4) [cx, cy, w, h] 归一化坐标
        
        返回:
            loss: GIoU损失
        """
        # 转换为xyxy格式
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # 计算GIoU
        giou = torch.diag(generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy))
        
        # GIoU损失: 1 - GIoU
        loss = 1 - giou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BBoxLoss(nn.Module):
    """
    组合边界框损失
    
    包括L1损失和GIoU损失
    """
    
    def __init__(
        self,
        loss_bbox_weight: float = 5.0,
        loss_giou_weight: float = 2.0
    ):
        """
        参数:
            loss_bbox_weight: L1损失权重
            loss_giou_weight: GIoU损失权重
        """
        super().__init__()
        
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_giou_weight = loss_giou_weight
        self.giou_loss = GIoULoss(reduction='mean')
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> dict:
        """
        前向传播
        
        参数:
            pred_boxes: (N, 4) [cx, cy, w, h]
            target_boxes: (N, 4) [cx, cy, w, h]
        
        返回:
            losses: {'loss_bbox': L1损失, 'loss_giou': GIoU损失}
        """
        # L1损失
        loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction='mean')
        
        # GIoU损失
        loss_giou = self.giou_loss(pred_boxes, target_boxes)
        
        return {
            'loss_bbox': loss_bbox * self.loss_bbox_weight,
            'loss_giou': loss_giou * self.loss_giou_weight
        }


if __name__ == "__main__":
    print("=" * 70)
    print("测试边界框损失")
    print("=" * 70)
    
    # 测试数据
    num_boxes = 100
    pred_boxes = torch.rand(num_boxes, 4)  # [cx, cy, w, h]
    target_boxes = torch.rand(num_boxes, 4)
    
    # 测试GIoU损失
    giou_criterion = GIoULoss(reduction='mean')
    giou_loss = giou_criterion(pred_boxes, target_boxes)
    print(f"GIoU损失: {giou_loss.item():.4f}")
    
    # 测试组合损失
    bbox_criterion = BBoxLoss(loss_bbox_weight=5.0, loss_giou_weight=2.0)
    losses = bbox_criterion(pred_boxes, target_boxes)
    
    print(f"\n组合损失:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # 测试IoU计算
    boxes1 = torch.tensor([[0.2, 0.3, 0.5, 0.6], [0.1, 0.1, 0.3, 0.3]])
    boxes2 = torch.tensor([[0.25, 0.35, 0.55, 0.65], [0.2, 0.2, 0.4, 0.4]])
    
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)
    
    iou = box_iou(boxes1_xyxy, boxes2_xyxy)
    giou = generalized_box_iou(boxes1_xyxy, boxes2_xyxy)
    
    print(f"\nIoU:\n{iou}")
    print(f"\nGIoU:\n{giou}")
    
    print("\n✅ 边界框损失测试完成！")


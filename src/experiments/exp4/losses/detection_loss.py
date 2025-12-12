# -*- coding: utf-8 -*-
"""
Detection Loss Function
L1损失 + GIoU损失 + CAM监督损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.multi_instance_assigner import MultiInstanceAssigner


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes
    
    Args:
        boxes1: [N, 4] (xmin, ymin, xmax, ymax)
        boxes2: [M, 4] (xmin, ymin, xmax, ymax)
    
    Returns:
        giou: [N, M] GIoU matrix
    """
    # Intersection
    inter_xmin = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_ymin = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_xmax = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_ymax = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * \
                 torch.clamp(inter_ymax - inter_ymin, min=0)
    
    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1[:, None] + area2 - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    # Enclosing box
    enclose_xmin = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    enclose_ymin = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    enclose_xmax = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    enclose_ymax = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    
    enclose_area = (enclose_xmax - enclose_xmin) * (enclose_ymax - enclose_ymin)
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    
    return giou


class DetectionLoss(nn.Module):
    """
    检测损失函数
    
    包含:
    - L1损失: 框坐标回归
    - GIoU损失: 框形状优化
    - CAM监督损失: 鼓励框内高响应、框外低响应
    """
    
    def __init__(self, 
                 lambda_l1: float = 1.0,
                 lambda_giou: float = 2.0,
                 lambda_cam: float = 0.5,
                 min_peak_distance: int = 2,
                 min_peak_value: float = 0.3,
                 match_iou_threshold: float = 0.3):
        """
        Args:
            lambda_l1: L1损失权重
            lambda_giou: GIoU损失权重
            lambda_cam: CAM监督损失权重
            min_peak_distance: 多峰检测的最小距离
            min_peak_value: 多峰检测的最小值
            match_iou_threshold: 匹配的IoU阈值
        """
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_cam = lambda_cam
        
        # Multi-instance assigner
        self.assigner = MultiInstanceAssigner(
            min_peak_distance=min_peak_distance,
            min_peak_value=min_peak_value,
            match_iou_threshold=match_iou_threshold
        )
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        计算检测损失
        
        Args:
            outputs: 模型输出
                - cam: [B, C, H, W]
                - pred_boxes: [B, C, H, W, 4]
                - scores: [B, C, H, W]
            
            targets: List[dict] (per image)
                - boxes: [K, 4] 归一化坐标 [xmin, ymin, xmax, ymax]
                - labels: [K] 类别id
        
        Returns:
            loss_dict: {
                'loss_box_l1': ...,
                'loss_box_giou': ...,
                'loss_cam': ...,
                'loss_total': ...
            }
        """
        cam = outputs['cam']
        pred_boxes = outputs['pred_boxes']
        B, C, H, W, _ = pred_boxes.shape
        
        # ===== Step 1: 正样本分配 =====
        loss_l1 = 0.0
        loss_giou = 0.0
        num_pos_samples = 0
        
        for b in range(B):
            gt_boxes = targets[b]['boxes']  # [K, 4]
            gt_labels = targets[b]['labels']  # [K]
            
            if len(gt_boxes) == 0:
                continue
            
            # 使用多峰匹配分配器
            pos_samples = self.assigner.assign(
                cam[b],  # [C, H, W]
                gt_boxes,
                gt_labels
            )
            
            if len(pos_samples) == 0:
                continue
            
            # ===== Step 2: 计算Box回归损失 =====
            for sample in pos_samples:
                i, j = sample['i'], sample['j']
                class_id = sample['class']
                gt_idx = sample['gt_idx']
                
                pred_box = pred_boxes[b, class_id, i, j]  # [4]
                gt_box = gt_boxes[gt_idx]  # [4]
                
                # L1 loss
                loss_l1 += F.l1_loss(pred_box, gt_box)
                
                # GIoU loss
                giou = generalized_box_iou(
                    pred_box.unsqueeze(0),
                    gt_box.unsqueeze(0)
                )[0, 0]
                loss_giou += (1 - giou)
                
                num_pos_samples += 1
        
        if num_pos_samples > 0:
            loss_l1 = loss_l1 / num_pos_samples
            loss_giou = loss_giou / num_pos_samples
        else:
            # 创建需要梯度的零tensor
            loss_l1 = torch.tensor(0.0, device=cam.device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=cam.device, requires_grad=True)
        
        # ===== Step 3: CAM监督损失 =====
        loss_cam = 0.0
        num_cam_samples = 0
        
        for b in range(B):
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            for k, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                label = label.item()
                xmin, ymin, xmax, ymax = box
                
                # 框内mask
                mask_in = torch.zeros(H, W, device=cam.device)
                i_min = max(0, int(ymin * H))
                i_max = min(H - 1, int(ymax * H))
                j_min = max(0, int(xmin * W))
                j_max = min(W - 1, int(xmax * W))
                
                if i_max >= i_min and j_max >= j_min:
                    mask_in[i_min:i_max+1, j_min:j_max+1] = 1
                
                # 框外mask
                mask_out = 1 - mask_in
                
                # CAM响应
                cam_c = cam[b, label]
                
                # Loss: 框内应该高,框外应该低
                mask_in_sum = mask_in.sum()
                mask_out_sum = mask_out.sum()
                
                if mask_in_sum > 0:
                    cam_in = (cam_c * mask_in).sum() / mask_in_sum
                    loss_cam += (1 - cam_in)
                
                if mask_out_sum > 0:
                    cam_out = (cam_c * mask_out).sum() / mask_out_sum
                    loss_cam += cam_out
                
                num_cam_samples += 1
        
        if num_cam_samples > 0:
            loss_cam = loss_cam / num_cam_samples
        else:
            # 创建需要梯度的零tensor
            loss_cam = torch.tensor(0.0, device=cam.device, requires_grad=True)
        
        # ===== Step 4: 总损失 =====
        loss_total = (
            self.lambda_l1 * loss_l1 +
            self.lambda_giou * loss_giou +
            self.lambda_cam * loss_cam
        )
        
        return {
            'loss_box_l1': loss_l1,
            'loss_box_giou': loss_giou,
            'loss_cam': loss_cam,
            'loss_total': loss_total
        }


# -*- coding: utf-8 -*-
"""
GT类别定位损失函数
只计算框回归损失（L1 + GIoU），在GT类别通道上匹配正样本

核心特点：
1. 只使用GT类别通道计算损失
2. 只计算框回归损失（L1 + GIoU）
3. 不使用置信度损失
4. 通过IoU匹配找到最佳位置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from losses.detection_loss import generalized_box_iou


class GTClassLocalizationLoss(nn.Module):
    """
    GT类别定位损失函数
    
    损失项：
    1. 框回归损失（L1 + GIoU）- 唯一损失
    
    关键特点：
    - 只使用GT类别通道
    - 通过IoU匹配找到最佳位置
    - 不使用置信度损失
    """
    
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_giou: float = 2.0,
        pos_radius: float = 1.5,
        pos_iou_threshold: float = 0.3
    ):
        """
        Args:
            lambda_l1: L1损失权重
            lambda_giou: GIoU损失权重
            pos_radius: 正样本半径（在最佳匹配位置周围）
            pos_iou_threshold: 正样本匹配IoU阈值
        """
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.pos_radius = pos_radius
        self.pos_iou_threshold = pos_iou_threshold
    
    def match_gt_to_predictions(
        self, 
        pred_boxes: torch.Tensor,  # [C, H, W, 4]
        gt_boxes: torch.Tensor,  # [N, 4]
        gt_labels: torch.Tensor,  # [N]
        H: int, W: int
    ) -> Dict:
        """
        将GT框匹配到预测框
        
        Args:
            pred_boxes: [C, H, W, 4] 预测框
            gt_boxes: [N, 4] GT框
            gt_labels: [N] GT类别索引
        
        Returns:
            Dict with:
                - matched_positions: List[(gt_idx, class_idx, i, j, iou)]
                - pos_mask: [C, H, W] bool mask
        """
        device = pred_boxes.device
        pos_mask = torch.zeros(pred_boxes.shape[0], H, W, device=device, dtype=torch.bool)
        matched_positions = []
        
        if len(gt_boxes) == 0:
            return {
                'matched_positions': matched_positions,
                'pos_mask': pos_mask
            }
        
        # 对每个GT框，在对应类别通道上找到最佳匹配位置
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            gt_label = gt_label.item()
            
            if gt_label >= pred_boxes.shape[0]:
                continue
            
            # 获取该类别通道的所有预测框
            pred_class_boxes = pred_boxes[gt_label]  # [H, W, 4]
            pred_class_boxes_flat = pred_class_boxes.view(H * W, 4)  # [H*W, 4]
            
            # 计算IoU
            gt_box_expanded = gt_box.unsqueeze(0)  # [1, 4]
            ious = generalized_box_iou(pred_class_boxes_flat, gt_box_expanded)  # [H*W, 1]
            ious = ious[:, 0]  # [H*W]
            ious = ious.view(H, W)  # [H, W]
            
            # 找到最大IoU的位置
            max_iou = ious.max().item()
            max_idx = ious.argmax().item()
            max_i = max_idx // W
            max_j = max_idx % W
            
            # 如果IoU > 阈值，标记为正样本
            if max_iou > self.pos_iou_threshold:
                # 在半径范围内都标记为正样本
                i_min = max(0, int(max_i - self.pos_radius))
                i_max = min(H - 1, int(max_i + self.pos_radius))
                j_min = max(0, int(max_j - self.pos_radius))
                j_max = min(W - 1, int(max_j + self.pos_radius))
                
                pos_mask[gt_label, i_min:i_max+1, j_min:j_max+1] = True
                
                matched_positions.append({
                    'gt_idx': gt_idx,
                    'class_idx': gt_label,
                    'i': max_i,
                    'j': max_j,
                    'iou': max_iou
                })
        
        return {
            'matched_positions': matched_positions,
            'pos_mask': pos_mask
        }
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        计算GT类别定位损失
        
        Args:
            outputs: Dict with 'pred_boxes' [B, C, H, W, 4]
            targets: List[Dict] with 'boxes' and 'labels'
        
        Returns:
            Dict with losses
        """
        pred_boxes = outputs['pred_boxes']  # [B, C, H, W, 4]
        B, C, H, W, _ = pred_boxes.shape
        device = pred_boxes.device
        
        # ===== Step 1: 匹配GT框到预测框并计算损失 =====
        loss_l1 = 0.0
        loss_giou = 0.0
        num_pos_samples = 0
        
        for b in range(B):
            gt_boxes = targets[b]['boxes'].to(device)  # [N, 4]
            gt_labels = targets[b]['labels'].to(device)  # [N]
            
            if len(gt_boxes) == 0:
                continue
            
            # 匹配GT框到预测框
            match_result = self.match_gt_to_predictions(
                pred_boxes[b],  # [C, H, W, 4]
                gt_boxes,
                gt_labels,
                H, W
            )
            
            matched_positions = match_result['matched_positions']
            pos_mask = match_result['pos_mask']  # [C, H, W]
            
            # 计算损失（对每个正样本位置）
            for match in matched_positions:
                gt_idx = match['gt_idx']
                class_idx = match['class_idx']
                max_i = match['i']
                max_j = match['j']
                gt_box = gt_boxes[gt_idx]  # [4]
                
                # 在半径范围内的所有位置都计算损失
                i_min = max(0, int(max_i - self.pos_radius))
                i_max = min(H - 1, int(max_i + self.pos_radius))
                j_min = max(0, int(max_j - self.pos_radius))
                j_max = min(W - 1, int(max_j + self.pos_radius))
                
                for i in range(i_min, i_max + 1):
                    for j in range(j_min, j_max + 1):
                        if pos_mask[class_idx, i, j]:
                            pred_box = pred_boxes[b, class_idx, i, j]  # [4]
                            
                            # L1损失
                            loss_l1 += F.l1_loss(pred_box, gt_box)
                            
                            # GIoU损失
                            giou = generalized_box_iou(
                                pred_box.unsqueeze(0),  # [1, 4]
                                gt_box.unsqueeze(0)  # [1, 4]
                            )[0, 0]
                            loss_giou += (1 - giou)
                            
                            num_pos_samples += 1
        
        # 平均损失
        if num_pos_samples > 0:
            loss_l1 = loss_l1 / num_pos_samples
            loss_giou = loss_giou / num_pos_samples
        else:
            loss_l1 = torch.tensor(0.0, device=device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ===== Step 2: 总损失 =====
        loss_total = (
            self.lambda_l1 * loss_l1 +
            self.lambda_giou * loss_giou
        )
        
        return {
            'loss_box_l1': loss_l1,
            'loss_box_giou': loss_giou,
            'loss_total': loss_total,
            'num_pos_samples': num_pos_samples
        }


# -*- coding: utf-8 -*-
"""
可学习的检测损失函数
使用可学习的峰值检测器替代固定阈值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from models.learnable_multi_instance_assigner import LearnableMultiInstanceAssigner
except ImportError:
    # Fallback if not available
    LearnableMultiInstanceAssigner = None
from losses.detection_loss import generalized_box_iou


class LearnableDetectionLoss(nn.Module):
    """
    可学习的检测损失函数
    
    关键改进:
    1. 使用可学习的峰值检测器（端到端训练）
    2. 结合objectness score
    3. 峰值检测损失（鼓励在GT框内检测到峰值）
    """
    
    def __init__(self, 
                 num_classes: int,
                 lambda_l1: float = 1.0,
                 lambda_giou: float = 2.0,
                 lambda_cam: float = 0.5,
                 lambda_peak: float = 0.5,  # 新增：峰值检测损失权重
                 min_peak_distance: int = 2,
                 init_threshold: float = 0.3,
                 match_iou_threshold: float = 0.3,
                 use_objectness: bool = True):
        """
        Args:
            num_classes: 类别数
            lambda_peak: 峰值检测损失权重
            init_threshold: 初始阈值（可学习）
            use_objectness: 是否使用objectness score
        """
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_cam = lambda_cam
        self.lambda_peak = lambda_peak
        
        # 可学习的多实例分配器
        self.assigner = LearnableMultiInstanceAssigner(
            num_classes=num_classes,
            min_peak_distance=min_peak_distance,
            init_threshold=init_threshold,
            match_iou_threshold=match_iou_threshold,
            use_objectness=use_objectness
        )
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        计算可学习的检测损失
        
        Args:
            outputs: 模型输出
                - cam: [B, C, H, W]
                - pred_boxes: [B, C, H, W, 4]
                - objectness: [B, C, H, W] (可选)
            
            targets: List[dict] (per image)
                - boxes: [K, 4]
                - labels: [K]
        
        Returns:
            loss_dict: {
                'loss_box_l1': ...,
                'loss_box_giou': ...,
                'loss_cam': ...,
                'loss_peak': ...,  # 新增：峰值检测损失
                'loss_total': ...
            }
        """
        cam = outputs['cam']
        pred_boxes = outputs['pred_boxes']
        objectness = outputs.get('objectness', None)
        B, C, H, W, _ = pred_boxes.shape
        
        # ===== Step 1: 可学习的正样本分配 =====
        loss_l1 = 0.0
        loss_giou = 0.0
        num_pos_samples = 0
        
        match_stats = {
            'total_peaks': 0,
            'total_gts': 0,
            'peak_matches': 0,
            'fallback_matches': 0,
            'unmatched_gts': 0,
            'avg_match_iou': []
        }
        
        all_peak_masks = []
        
        for b in range(B):
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            match_stats['total_gts'] += len(gt_boxes)
            
            # 使用可学习的分配器
            assign_result = self.assigner(
                cam[b],  # [C, H, W]
                pred_boxes[b],  # [C, H, W, 4]
                objectness[b] if objectness is not None else None,  # [C, H, W]
                gt_boxes,
                gt_labels
            )
            
            pos_samples = assign_result['pos_samples']
            peak_masks = assign_result['peak_masks']  # [C, H, W]
            all_peak_masks.append(peak_masks)
            
            # 统计峰值数量
            match_stats['total_peaks'] += peak_masks.sum().item()
            
            # 计算Box回归损失
            for sample in pos_samples:
                i, j = sample['i'], sample['j']
                class_id = sample['class']
                gt_idx = sample['gt_idx']
                
                pred_box = pred_boxes[b, class_id, i, j]
                gt_box = gt_boxes[gt_idx]
                
                loss_l1 += F.l1_loss(pred_box, gt_box)
                
                giou = generalized_box_iou(
                    pred_box.unsqueeze(0),
                    gt_box.unsqueeze(0)
                )[0, 0]
                loss_giou += (1 - giou)
                
                if sample['match_type'] == 'peak':
                    match_stats['peak_matches'] += 1
                    match_stats['avg_match_iou'].append(sample['iou'])
                else:
                    match_stats['fallback_matches'] += 1
                
                num_pos_samples += 1
        
        if num_pos_samples > 0:
            loss_l1 = loss_l1 / num_pos_samples
            loss_giou = loss_giou / num_pos_samples
        else:
            loss_l1 = torch.tensor(0.0, device=cam.device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=cam.device, requires_grad=True)
        
        # ===== Step 2: CAM监督损失 =====
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
                cam_c = cam[b, label]
                
                mask_in = torch.zeros(H, W, device=cam.device)
                i_min = max(0, int(ymin * H))
                i_max = min(H - 1, int(ymax * H))
                j_min = max(0, int(xmin * W))
                j_max = min(W - 1, int(xmax * W))
                
                if i_max >= i_min and j_max >= j_min:
                    mask_in[i_min:i_max+1, j_min:j_max+1] = 1
                
                mask_out = 1 - mask_in
                
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
            loss_cam = torch.tensor(0.0, device=cam.device, requires_grad=True)
        
        # ===== Step 3: 峰值检测损失（新增） =====
        # 鼓励在GT框内检测到峰值，在框外不检测峰值
        loss_peak = 0.0
        num_peak_samples = 0
        
        if len(all_peak_masks) > 0:
            for b in range(B):
                gt_boxes = targets[b]['boxes']
                gt_labels = targets[b]['labels']
                
                if len(gt_boxes) == 0:
                    continue
                
                peak_mask = all_peak_masks[b]  # [C, H, W]
                
                for k, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                    label = label.item()
                    xmin, ymin, xmax, ymax = box
                    
                    # GT框内mask
                    mask_in = torch.zeros(H, W, device=cam.device)
                    i_min = max(0, int(ymin * H))
                    i_max = min(H - 1, int(ymax * H))
                    j_min = max(0, int(xmin * W))
                    j_max = min(W - 1, int(xmax * W))
                    
                    if i_max >= i_min and j_max >= j_min:
                        mask_in[i_min:i_max+1, j_min:j_max+1] = 1
                    
                    mask_out = 1 - mask_in
                    
                    # 该类别的峰值mask
                    peak_mask_class = peak_mask[label]  # [H, W]
                    
                    # 损失：框内应该有峰值，框外不应该有峰值
                    mask_in_sum = mask_in.sum()
                    mask_out_sum = mask_out.sum()
                    
                    if mask_in_sum > 0:
                        peak_in = (peak_mask_class * mask_in).sum() / mask_in_sum
                        loss_peak += (1 - peak_in)
                    
                    if mask_out_sum > 0:
                        peak_out = (peak_mask_class * mask_out).sum() / mask_out_sum
                        loss_peak += peak_out
                    
                    num_peak_samples += 1
        
        if num_peak_samples > 0:
            loss_peak = loss_peak / num_peak_samples
        else:
            loss_peak = torch.tensor(0.0, device=cam.device, requires_grad=True)
        
        # 计算匹配统计
        if match_stats['avg_match_iou']:
            match_stats['avg_match_iou'] = sum(match_stats['avg_match_iou']) / len(match_stats['avg_match_iou'])
        else:
            match_stats['avg_match_iou'] = 0.0
        
        # ===== Step 4: 总损失 =====
        loss_total = (
            self.lambda_l1 * loss_l1 +
            self.lambda_giou * loss_giou +
            self.lambda_cam * loss_cam +
            self.lambda_peak * loss_peak  # 新增
        )
        
        return {
            'loss_box_l1': loss_l1,
            'loss_box_giou': loss_giou,
            'loss_cam': loss_cam,
            'loss_peak': loss_peak,  # 新增
            'loss_total': loss_total,
            'match_stats': match_stats
        }


# -*- coding: utf-8 -*-
"""
改进的直接检测损失函数
针对端到端检测的损失函数设计

核心思想：
1. CAM作为输入特征，不是最终目标
2. 损失应该关注检测质量，而不是CAM质量
3. CAM损失应该与检测结果对齐，而不是GT框
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from losses.detection_loss import generalized_box_iou


class ImprovedDirectDetectionLoss(nn.Module):
    """
    改进的直接检测损失函数
    
    损失项：
    1. 框回归损失（L1 + GIoU）- 主要损失
    2. 置信度损失（Focal Loss）- 主要损失
    3. CAM对齐损失（可选）- 辅助损失，让CAM与检测结果对齐
    
    关键改进：
    - 移除传统的CAM监督损失（框内外响应）
    - CAM损失改为：CAM应该在预测框位置有高响应
    - 降低CAM损失权重，或完全移除
    """
    
    def __init__(self,
                 lambda_l1: float = 1.0,
                 lambda_giou: float = 2.0,
                 lambda_conf: float = 1.0,
                 lambda_cam: float = 0.0,  # 默认不使用CAM损失
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 pos_radius: float = 1.5,
                 pos_iou_threshold: float = 0.3,  # 正样本匹配IoU阈值
                 use_cam_alignment: bool = False):  # 是否使用CAM对齐损失
        """
        Args:
            lambda_l1: L1损失权重
            lambda_giou: GIoU损失权重
            lambda_conf: 置信度损失权重
            lambda_cam: CAM损失权重（默认0，不使用）
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
            pos_radius: 正样本半径
            use_cam_alignment: 是否使用CAM对齐损失（让CAM与预测框对齐）
        """
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_conf = lambda_conf
        self.lambda_cam = lambda_cam
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_radius = pos_radius
        self.pos_iou_threshold = pos_iou_threshold
        self.use_cam_alignment = use_cam_alignment
    
    def assign_positives(self, pred_boxes: torch.Tensor,
                        gt_boxes: torch.Tensor,
                        gt_labels: torch.Tensor,
                        H: int, W: int) -> Dict:
        """分配正样本（与原版相同）"""
        B, C, H_out, W_out, _ = pred_boxes.shape
        device = pred_boxes.device
        
        pos_mask = torch.zeros(B, C, H_out, W_out, device=device, dtype=torch.bool)
        matched_gt_indices = torch.full((B, C, H_out, W_out), -1, device=device, dtype=torch.long)
        
        for b in range(B):
            if len(gt_boxes) == 0:
                continue
            
            pred_boxes_flat = pred_boxes[b].view(C, H_out * W_out, 4)
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                gt_label = gt_label.item()
                if gt_label >= C:
                    continue
                
                pred_class = pred_boxes_flat[gt_label]
                gt_box_expanded = gt_box.unsqueeze(0)
                
                ious_matrix = generalized_box_iou(pred_class, gt_box_expanded)
                ious = ious_matrix[:, 0]
                ious = ious.view(H_out, W_out)
                
                max_iou = ious.max()
                max_idx = ious.argmax()
                max_i, max_j = max_idx // W_out, max_idx % W_out
                
                if max_iou > 0.3:
                    i_min = max(0, int(max_i - self.pos_radius))
                    i_max = min(H_out - 1, int(max_i + self.pos_radius))
                    j_min = max(0, int(max_j - self.pos_radius))
                    j_max = min(W_out - 1, int(max_j + self.pos_radius))
                    
                    pos_mask[b, gt_label, i_min:i_max+1, j_min:j_max+1] = True
                    matched_gt_indices[b, gt_label, i_min:i_max+1, j_min:j_max+1] = gt_idx
        
        return {
            'pos_mask': pos_mask,
            'matched_gt_indices': matched_gt_indices
        }
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal Loss"""
        pred = pred.clamp(1e-6, 1 - 1e-6)
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha_t * focal_weight * ce_loss
        return focal_loss
    
    def cam_alignment_loss(self, cam: torch.Tensor, pred_boxes: torch.Tensor,
                          confidences: torch.Tensor, gt_boxes: torch.Tensor,
                          gt_labels: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        CAM对齐损失：让CAM在预测框位置有高响应
        
        思路：
        - 对于高置信度的预测框，CAM应该在该位置有高响应
        - 这样CAM和检测结果就对齐了
        """
        B, C, H_out, W_out = cam.shape
        device = cam.device
        loss_cam = 0.0
        num_samples = 0
        
        for b in range(B):
            if len(gt_boxes) == 0:
                continue
            
            for c in range(C):
                cam_c = cam[b, c]  # [H, W]
                conf_c = confidences[b, c]  # [H, W]
                boxes_c = pred_boxes[b, c]  # [H, W, 4]
                
                # 找到高置信度的位置
                high_conf_mask = conf_c > 0.5  # 高置信度阈值
                
                if high_conf_mask.sum() == 0:
                    continue
                
                # 对于高置信度位置，CAM应该有高响应
                high_conf_positions = high_conf_mask.nonzero(as_tuple=False)  # [N, 2]
                
                for pos in high_conf_positions:
                    i, j = pos[0].item(), pos[1].item()
                    conf_value = conf_c[i, j]
                    cam_value = cam_c[i, j]
                    
                    # 鼓励高置信度位置有高CAM响应
                    # loss = (1 - cam_value) * conf_value
                    loss_cam += (1 - cam_value) * conf_value
                    num_samples += 1
        
        if num_samples > 0:
            loss_cam = loss_cam / num_samples
        else:
            loss_cam = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss_cam
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        计算改进的直接检测损失
        """
        pred_boxes = outputs['pred_boxes']
        confidences = outputs['confidences']
        # 兼容不同的输出格式
        cam = outputs.get('cam', outputs.get('fused_cam', None))
        if cam is None:
            raise KeyError("Outputs must contain 'cam' or 'fused_cam'")
        B, C, H, W, _ = pred_boxes.shape
        
        # ===== Step 1: 分配正样本并计算框回归损失 =====
        loss_l1 = 0.0
        loss_giou = 0.0
        num_pos_samples = 0
        
        pos_mask_all = torch.zeros(B, C, H, W, device=pred_boxes.device, dtype=torch.bool)
        conf_targets = torch.zeros(B, C, H, W, device=pred_boxes.device)
        
        for b in range(B):
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            assign_result = self.assign_positives(
                pred_boxes[b:b+1], gt_boxes, gt_labels, H, W
            )
            
            pos_mask = assign_result['pos_mask'][0]
            matched_gt_indices = assign_result['matched_gt_indices'][0]
            
            pos_mask_all[b] = pos_mask
            
            for c in range(C):
                class_pos_mask = pos_mask[c]
                if class_pos_mask.sum() == 0:
                    continue
                
                class_gt_indices = torch.where(gt_labels == c)[0]
                if len(class_gt_indices) == 0:
                    continue
                
                pos_positions = class_pos_mask.nonzero(as_tuple=False)
                
                for pos in pos_positions:
                    i, j = pos[0].item(), pos[1].item()
                    pred_box = pred_boxes[b, c, i, j]
                    
                    matched_gt_idx = matched_gt_indices[c, i, j].item()
                    if matched_gt_idx >= 0:
                        gt_box = gt_boxes[matched_gt_idx]
                        
                        loss_l1 += F.l1_loss(pred_box, gt_box)
                        
                        giou = generalized_box_iou(
                            pred_box.unsqueeze(0),
                            gt_box.unsqueeze(0)
                        )[0, 0]
                        loss_giou += (1 - giou)
                        
                        num_pos_samples += 1
                        conf_targets[b, c, i, j] = 1.0
        
        if num_pos_samples > 0:
            loss_l1 = loss_l1 / num_pos_samples
            loss_giou = loss_giou / num_pos_samples
        else:
            loss_l1 = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        # ===== Step 2: 置信度损失 =====
        loss_conf = self.focal_loss(confidences, conf_targets)
        loss_conf = loss_conf.mean()
        
        # ===== Step 3: CAM对齐损失（可选） =====
        loss_cam = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        if self.use_cam_alignment and self.lambda_cam > 0:
            loss_cam = self.cam_alignment_loss(
                cam, pred_boxes, confidences,
                targets[0]['boxes'] if len(targets) > 0 else None,
                targets[0]['labels'] if len(targets) > 0 else None,
                H, W
            )
        
        # ===== Step 4: 总损失 =====
        loss_total = (
            self.lambda_l1 * loss_l1 +
            self.lambda_giou * loss_giou +
            self.lambda_conf * loss_conf +
            self.lambda_cam * loss_cam
        )
        
        pos_ratio = pos_mask_all.sum().item() / (B * C * H * W)
        
        return {
            'loss_box_l1': loss_l1,
            'loss_box_giou': loss_giou,
            'loss_conf': loss_conf,
            'loss_cam': loss_cam,
            'loss_total': loss_total,
            'num_pos_samples': num_pos_samples,
            'pos_ratio': pos_ratio
        }


# -*- coding: utf-8 -*-
"""
直接检测损失函数
用于直接检测头，无需峰值检测
类似FCOS的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from losses.detection_loss import generalized_box_iou


class DirectDetectionLoss(nn.Module):
    """
    直接检测损失函数
    
    损失项:
    1. 框回归损失（L1 + GIoU）
    2. 置信度损失（Focal Loss）
    3. CAM监督损失（可选）
    """
    
    def __init__(self,
                 lambda_l1: float = 1.0,
                 lambda_giou: float = 2.0,
                 lambda_conf: float = 1.0,
                 lambda_cam: float = 0.5,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 pos_radius: float = 1.5):  # 正样本半径（相对于GT框）
        """
        Args:
            lambda_l1: L1损失权重
            lambda_giou: GIoU损失权重
            lambda_conf: 置信度损失权重
            lambda_cam: CAM监督损失权重
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
            pos_radius: 正样本半径（网格单位）
        """
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_conf = lambda_conf
        self.lambda_cam = lambda_cam
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_radius = pos_radius
    
    def assign_positives(self, pred_boxes: torch.Tensor,
                        gt_boxes: torch.Tensor,
                        gt_labels: torch.Tensor,
                        H: int, W: int) -> Dict:
        """
        分配正样本（基于IoU和位置）
        
        Args:
            pred_boxes: [B, C, H, W, 4] 预测框
            gt_boxes: [K, 4] GT框
            gt_labels: [K] GT标签
        
        Returns:
            pos_mask: [B, C, H, W] 正样本mask
            matched_gt_indices: [B, C, H, W] 匹配的GT索引（-1表示负样本）
        """
        B, C, H_out, W_out, _ = pred_boxes.shape
        device = pred_boxes.device
        
        pos_mask = torch.zeros(B, C, H_out, W_out, device=device, dtype=torch.bool)
        matched_gt_indices = torch.full((B, C, H_out, W_out), -1, device=device, dtype=torch.long)
        
        # 生成网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H_out, device=device),
            torch.linspace(0, 1, W_out, device=device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        for b in range(B):
            if len(gt_boxes) == 0:
                continue
            
            # 计算每个位置与所有GT框的IoU
            pred_boxes_flat = pred_boxes[b].view(C, H_out * W_out, 4)  # [C, H*W, 4]
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                gt_label = gt_label.item()
                if gt_label >= C:
                    continue
                
                # 计算该类别的所有预测框与当前GT的IoU
                pred_class = pred_boxes_flat[gt_label]  # [H*W, 4]
                gt_box_expanded = gt_box.unsqueeze(0)  # [1, 4]
                
                # generalized_box_iou返回 [N, M] 矩阵
                ious_matrix = generalized_box_iou(pred_class, gt_box_expanded)  # [H*W, 1]
                # 取第一列（每个预测框与GT的IoU）
                ious = ious_matrix[:, 0]  # [H*W]
                
                # 确保size正确
                assert ious.shape[0] == H_out * W_out, f"IoU shape mismatch: {ious.shape} vs {H_out * W_out}"
                ious = ious.view(H_out, W_out)  # [H, W]
                
                # 找到IoU最高的位置
                max_iou = ious.max()
                max_idx = ious.argmax()
                max_i, max_j = max_idx // W_out, max_idx % W_out
                
                # 如果IoU足够高，标记为正样本
                if max_iou > 0.3:  # IoU阈值
                    # 标记该位置及其周围区域为正样本
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
        """
        Focal Loss for confidence prediction
        """
        pred = pred.clamp(1e-6, 1 - 1e-6)
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha_t * focal_weight * ce_loss
        return focal_loss
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        计算直接检测损失
        
        Args:
            outputs: 模型输出
                - pred_boxes: [B, C, H, W, 4]
                - confidences: [B, C, H, W]
                - cam: [B, C, H, W]
            
            targets: List[dict] (per image)
                - boxes: [K, 4]
                - labels: [K]
        
        Returns:
            loss_dict: {
                'loss_box_l1': ...,
                'loss_box_giou': ...,
                'loss_conf': ...,
                'loss_cam': ...,
                'loss_total': ...
            }
        """
        pred_boxes = outputs['pred_boxes']
        confidences = outputs['confidences']
        cam = outputs['cam']
        B, C, H, W, _ = pred_boxes.shape
        
        # ===== Step 1: 分配正样本 =====
        loss_l1 = 0.0
        loss_giou = 0.0
        num_pos_samples = 0
        
        # 准备正样本标签
        pos_mask_all = torch.zeros(B, C, H, W, device=pred_boxes.device, dtype=torch.bool)
        conf_targets = torch.zeros(B, C, H, W, device=pred_boxes.device)
        
        for b in range(B):
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            # 分配正样本
            assign_result = self.assign_positives(
                pred_boxes[b:b+1], gt_boxes, gt_labels, H, W
            )
            
            pos_mask = assign_result['pos_mask'][0]  # [C, H, W]
            matched_gt_indices = assign_result['matched_gt_indices'][0]  # [C, H, W]
            
            pos_mask_all[b] = pos_mask
            
            # 计算框回归损失
            for c in range(C):
                class_pos_mask = pos_mask[c]  # [H, W]
                if class_pos_mask.sum() == 0:
                    continue
                
                # 找到该类别的GT框
                class_gt_indices = torch.where(gt_labels == c)[0]
                if len(class_gt_indices) == 0:
                    continue
                
                # 计算损失
                pos_positions = class_pos_mask.nonzero(as_tuple=False)  # [N, 2]
                
                for pos in pos_positions:
                    i, j = pos[0].item(), pos[1].item()
                    pred_box = pred_boxes[b, c, i, j]  # [4]
                    
                    # 找到匹配的GT框
                    matched_gt_idx = matched_gt_indices[c, i, j].item()
                    if matched_gt_idx >= 0:
                        gt_box = gt_boxes[matched_gt_idx]
                        
                        # L1 loss
                        loss_l1 += F.l1_loss(pred_box, gt_box)
                        
                        # GIoU loss
                        giou = generalized_box_iou(
                            pred_box.unsqueeze(0),
                            gt_box.unsqueeze(0)
                        )[0, 0]
                        loss_giou += (1 - giou)
                        
                        num_pos_samples += 1
                        
                        # 设置置信度目标
                        conf_targets[b, c, i, j] = 1.0
        
        if num_pos_samples > 0:
            loss_l1 = loss_l1 / num_pos_samples
            loss_giou = loss_giou / num_pos_samples
        else:
            loss_l1 = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        # ===== Step 2: 置信度损失（Focal Loss） =====
        loss_conf = self.focal_loss(confidences, conf_targets)
        loss_conf = loss_conf.mean()
        
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
        
        # ===== Step 4: 总损失 =====
        loss_total = (
            self.lambda_l1 * loss_l1 +
            self.lambda_giou * loss_giou +
            self.lambda_conf * loss_conf +
            self.lambda_cam * loss_cam
        )
        
        # 统计信息
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


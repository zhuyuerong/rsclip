#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
损失计算模块

功能：
1. 整合所有损失函数
2. 处理匹配和样本分配
3. 计算总损失
"""

import torch
import torch.nn as nn
from typing import List, Dict
import sys
sys.path.append('..')

from losses.varifocal_loss import VarifocalLoss
from losses.bbox_loss import BBoxLoss, box_cxcywh_to_xyxy, generalized_box_iou
from losses.matcher import HungarianMatcher


class SetCriterion(nn.Module):
    """
    损失计算器
    
    整合：
    1. 变焦损失（分类）
    2. L1损失（边界框）
    3. GIoU损失（边界框）
    """
    
    def __init__(self, config):
        """
        参数:
            config: 配置对象
        """
        super().__init__()
        
        self.config = config
        self.num_classes = config.num_classes
        
        # 匹配器
        self.matcher = HungarianMatcher(
            cost_class=config.loss_cls_weight,
            cost_bbox=config.loss_bbox_weight,
            cost_giou=config.loss_giou_weight
        )
        
        # 损失函数
        self.cls_loss = VarifocalLoss(
            alpha=config.varifocal_alpha,
            gamma=config.varifocal_gamma,
            iou_weighted=config.iou_weighted,
            reduction='mean'
        )
        
        self.bbox_loss = BBoxLoss(
            loss_bbox_weight=config.loss_bbox_weight,
            loss_giou_weight=config.loss_giou_weight
        )
        
        # 权重
        self.loss_cls_weight = config.loss_cls_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            outputs: {
                'pred_logits': (num_layers, B, num_queries, num_classes),
                'pred_boxes': (num_layers, B, num_queries, 4)
            }
            targets: List of {
                'labels': (num_targets,),
                'boxes': (num_targets, 4)
            }
        
        返回:
            losses: {
                'loss_cls': 分类损失,
                'loss_bbox': L1损失,
                'loss_giou': GIoU损失,
                'loss_total': 总损失
            }
        """
        pred_logits = outputs['pred_logits']  # (num_layers, B, num_queries, num_classes)
        pred_boxes = outputs['pred_boxes']    # (num_layers, B, num_queries, 4)
        
        num_layers = pred_logits.shape[0]
        
        # 累积损失
        total_loss_cls = 0
        total_loss_bbox = 0
        total_loss_giou = 0
        
        # 对每一层计算损失
        for layer_idx in range(num_layers):
            layer_logits = pred_logits[layer_idx]  # (B, num_queries, num_classes)
            layer_boxes = pred_boxes[layer_idx]    # (B, num_queries, 4)
            
            # 匹配
            indices = self.matcher(layer_logits, layer_boxes, targets)
            
            # 收集匹配的预测和目标
            pred_logits_matched = []
            pred_boxes_matched = []
            target_labels_matched = []
            target_boxes_matched = []
            target_ious = []
            
            for i, (pred_idx, target_idx) in enumerate(indices):
                if len(pred_idx) == 0:
                    continue
                
                # 预测
                pred_logits_matched.append(layer_logits[i][pred_idx])
                pred_boxes_matched.append(layer_boxes[i][pred_idx])
                
                # 目标
                target_labels_matched.append(targets[i]['labels'][target_idx])
                target_boxes_matched.append(targets[i]['boxes'][target_idx])
                
                # 计算IoU
                pred_boxes_xyxy = box_cxcywh_to_xyxy(layer_boxes[i][pred_idx])
                target_boxes_xyxy = box_cxcywh_to_xyxy(targets[i]['boxes'][target_idx])
                iou = torch.diag(generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy))
                target_ious.append(iou)
            
            if len(pred_logits_matched) == 0:
                # 没有匹配，跳过
                continue
            
            # 连接所有批次
            pred_logits_matched = torch.cat(pred_logits_matched, dim=0)
            pred_boxes_matched = torch.cat(pred_boxes_matched, dim=0)
            target_labels_matched = torch.cat(target_labels_matched, dim=0)
            target_boxes_matched = torch.cat(target_boxes_matched, dim=0)
            target_ious = torch.cat(target_ious, dim=0)
            
            # 分类损失
            # 构建one-hot目标
            target_onehot = torch.zeros(
                pred_logits_matched.shape,
                dtype=pred_logits_matched.dtype,
                device=pred_logits_matched.device
            )
            target_onehot[torch.arange(len(target_labels_matched)), target_labels_matched] = 1.0
            
            loss_cls = self.cls_loss(
                pred_logits_matched,
                target_onehot,
                target_ious
            )
            
            # 边界框损失
            bbox_losses = self.bbox_loss(pred_boxes_matched, target_boxes_matched)
            loss_bbox = bbox_losses['loss_bbox']
            loss_giou = bbox_losses['loss_giou']
            
            # 累积
            total_loss_cls += loss_cls
            total_loss_bbox += loss_bbox
            total_loss_giou += loss_giou
        
        # 平均
        total_loss_cls = total_loss_cls / num_layers
        total_loss_bbox = total_loss_bbox / num_layers
        total_loss_giou = total_loss_giou / num_layers
        
        # 总损失
        total_loss = total_loss_cls + total_loss_bbox + total_loss_giou
        
        return {
            'loss_cls': total_loss_cls * self.loss_cls_weight,
            'loss_bbox': total_loss_bbox,
            'loss_giou': total_loss_giou,
            'loss_total': total_loss
        }


if __name__ == "__main__":
    import sys
    sys.path.append('/home/ubuntu22/Projects/RemoteCLIP-main/experiment3')
    from config.default_config import DefaultConfig
    
    print("=" * 70)
    print("测试损失计算器")
    print("=" * 70)
    
    # 配置
    config = DefaultConfig()
    
    # 创建损失计算器
    criterion = SetCriterion(config)
    
    # 测试数据
    batch_size = 2
    num_queries = 300
    num_classes = 20
    num_layers = 6
    
    outputs = {
        'pred_logits': torch.randn(num_layers, batch_size, num_queries, num_classes),
        'pred_boxes': torch.rand(num_layers, batch_size, num_queries, 4)
    }
    
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
    
    # 计算损失
    losses = criterion(outputs, targets)
    
    print(f"\n损失:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\n✅ 损失计算器测试完成！")


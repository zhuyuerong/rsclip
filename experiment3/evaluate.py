#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本

功能：
1. 计算mAP (mean Average Precision)
2. 计算各类别的AP
3. 生成评估报告
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from collections import defaultdict
import sys

from config.default_config import DefaultConfig
from models.ova_detr import OVADETR
from utils.data_loader import create_data_loader, DIOR_CLASSES
from utils.transforms import get_transforms
from losses.bbox_loss import box_cxcywh_to_xyxy


def compute_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    参数:
        box1: (4,) [x1, y1, x2, y2]
        box2: (4,) [x1, y1, x2, y2]
    
    返回:
        iou: float
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou


def compute_ap(recalls, precisions):
    """
    计算Average Precision (11点插值法)
    
    参数:
        recalls: List of recall values
        precisions: List of precision values
    
    返回:
        ap: Average Precision
    """
    # 11点插值
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def evaluate_detections(
    all_predictions: list,
    all_targets: list,
    num_classes: int,
    iou_threshold: float = 0.5
) -> dict:
    """
    评估检测结果
    
    参数:
        all_predictions: List of predictions for each image
        all_targets: List of targets for each image
        num_classes: 类别数
        iou_threshold: IoU阈值
    
    返回:
        metrics: {
            'mAP': float,
            'AP_per_class': dict,
            'precision': float,
            'recall': float
        }
    """
    # 收集每个类别的预测和目标
    class_predictions = defaultdict(list)  # {class_id: [(confidence, image_id, box)]}
    class_targets = defaultdict(int)       # {class_id: num_instances}
    
    for img_id, (pred, target) in enumerate(zip(all_predictions, all_targets)):
        # 目标
        for box, label in zip(target['boxes'], target['labels']):
            class_targets[label.item()] += 1
        
        # 预测
        if len(pred['boxes']) > 0:
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                class_predictions[label.item()].append({
                    'confidence': score,
                    'image_id': img_id,
                    'box': box
                })
    
    # 计算每个类别的AP
    aps = {}
    
    for class_id in range(num_classes):
        if class_id not in class_targets or class_targets[class_id] == 0:
            # 没有该类别的目标
            continue
        
        # 获取该类别的预测
        preds = class_predictions.get(class_id, [])
        
        if len(preds) == 0:
            aps[class_id] = 0.0
            continue
        
        # 按置信度排序
        preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
        
        # 计算TP和FP
        num_targets = class_targets[class_id]
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        # 记录已匹配的目标
        matched_targets = defaultdict(set)
        
        for pred_idx, pred in enumerate(preds):
            img_id = pred['image_id']
            pred_box = pred['box']
            
            # 获取该图像的目标
            target = all_targets[img_id]
            target_boxes = target['boxes']
            target_labels = target['labels']
            
            # 找到该类别的目标
            class_mask = (target_labels == class_id)
            class_target_boxes = target_boxes[class_mask]
            
            if len(class_target_boxes) == 0:
                fp[pred_idx] = 1
                continue
            
            # 计算与所有目标的IoU
            max_iou = 0
            max_idx = -1
            
            for target_idx, target_box in enumerate(class_target_boxes):
                iou = compute_iou(pred_box, target_box.numpy())
                if iou > max_iou:
                    max_iou = iou
                    max_idx = target_idx
            
            # 判断TP或FP
            if max_iou >= iou_threshold:
                if max_idx not in matched_targets[img_id]:
                    tp[pred_idx] = 1
                    matched_targets[img_id].add(max_idx)
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算Precision和Recall
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / num_targets
        
        # 计算AP
        ap = compute_ap(recalls, precisions)
        aps[class_id] = ap
    
    # 计算mAP
    if len(aps) > 0:
        mAP = np.mean(list(aps.values()))
    else:
        mAP = 0.0
    
    # 转换为类别名称
    ap_per_class = {DIOR_CLASSES[class_id]: ap for class_id, ap in aps.items()}
    
    return {
        'mAP': mAP,
        'AP_per_class': ap_per_class,
        'num_classes_evaluated': len(aps)
    }


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
    text_features,
    score_threshold: float = 0.3
):
    """
    评估模型
    
    返回:
        all_predictions: List of predictions
        all_targets: List of targets
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(data_loader, desc='Evaluation')
    
    for images, targets in pbar:
        # 移动到设备
        images = images.to(device)
        
        # 前向传播
        outputs = model(images, text_features)
        
        # 使用最后一层的输出
        pred_logits = outputs['pred_logits'][-1]  # (B, num_queries, num_classes)
        pred_boxes = outputs['pred_boxes'][-1]    # (B, num_queries, 4)
        
        # 处理每张图像
        batch_size = pred_logits.shape[0]
        for i in range(batch_size):
            # 预测
            logits = pred_logits[i]  # (num_queries, num_classes)
            boxes = pred_boxes[i]    # (num_queries, 4)
            
            # 计算分数和标签
            scores = logits.sigmoid()  # (num_queries, num_classes)
            max_scores, labels = scores.max(dim=-1)  # (num_queries,)
            
            # 过滤低分数
            keep = max_scores > score_threshold
            boxes = boxes[keep]
            scores_keep = max_scores[keep]
            labels_keep = labels[keep]
            
            # 转换边界框：cxcywh (归一化) -> xyxy (归一化)
            if len(boxes) > 0:
                boxes_xyxy = box_cxcywh_to_xyxy(boxes)
                
                # 转换为像素坐标
                orig_h, orig_w = targets[i]['orig_size']
                boxes_xyxy[:, [0, 2]] *= orig_w
                boxes_xyxy[:, [1, 3]] *= orig_h
                
                predictions = {
                    'boxes': boxes_xyxy.cpu(),
                    'scores': scores_keep.cpu(),
                    'labels': labels_keep.cpu()
                }
            else:
                predictions = {
                    'boxes': torch.zeros((0, 4)),
                    'scores': torch.zeros((0,)),
                    'labels': torch.zeros((0,), dtype=torch.long)
                }
            
            all_predictions.append(predictions)
            
            # 目标
            target_boxes = targets[i]['boxes']  # (N, 4) cxcywh归一化
            target_labels = targets[i]['labels']
            
            # 转换为xyxy像素坐标
            if len(target_boxes) > 0:
                target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
                orig_h, orig_w = targets[i]['orig_size']
                target_boxes_xyxy[:, [0, 2]] *= orig_w
                target_boxes_xyxy[:, [1, 3]] *= orig_h
            else:
                target_boxes_xyxy = torch.zeros((0, 4))
            
            all_targets.append({
                'boxes': target_boxes_xyxy.cpu(),
                'labels': target_labels.cpu()
            })
    
    return all_predictions, all_targets


def main(args):
    """主函数"""
    
    print("=" * 70)
    print("OVA-DETR评估")
    print("=" * 70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 加载检查点
    print(f"\n加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', DefaultConfig())
    
    # 创建模型
    print("\n创建模型...")
    model = OVADETR(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型加载成功 (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # 提取文本特征
    print("\n提取文本特征...")
    with torch.no_grad():
        text_features = model.backbone.forward_text(DIOR_CLASSES)
        text_features = text_features.to(device)
    
    print(f"✅ 文本特征: {text_features.shape}")
    
    # 加载数据
    print("\n加载数据...")
    val_transforms = get_transforms(mode='val', image_size=config.image_size)
    val_loader = create_data_loader(
        root_dir=args.data_dir,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=val_transforms
    )
    
    print(f"✅ 验证集: {len(val_loader.dataset)}张图片")
    
    # 评估
    print("\n" + "=" * 70)
    print("开始评估")
    print("=" * 70)
    
    all_predictions, all_targets = evaluate_model(
        model, val_loader, device, text_features, args.score_threshold
    )
    
    # 计算指标
    print("\n计算指标...")
    metrics = evaluate_detections(
        all_predictions, all_targets,
        num_classes=len(DIOR_CLASSES),
        iou_threshold=args.iou_threshold
    )
    
    # 打印结果
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"\nmAP@{args.iou_threshold}: {metrics['mAP']:.4f}")
    print(f"评估类别数: {metrics['num_classes_evaluated']}/{len(DIOR_CLASSES)}")
    
    print("\n各类别AP:")
    for class_name, ap in sorted(metrics['AP_per_class'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name:30s}: {ap:.4f}")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'mAP': float(metrics['mAP']),
            'iou_threshold': args.iou_threshold,
            'score_threshold': args.score_threshold,
            'num_classes_evaluated': metrics['num_classes_evaluated'],
            'AP_per_class': {k: float(v) for k, v in metrics['AP_per_class'].items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ 结果保存到: {output_path}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OVA-DETR评估')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data_dir', type=str,
                       default='/home/ubuntu22/Projects/RemoteCLIP-main/datasets/DIOR',
                       help='数据集目录')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='输出结果路径')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                       help='分数阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU阈值')
    
    args = parser.parse_args()
    
    main(args)


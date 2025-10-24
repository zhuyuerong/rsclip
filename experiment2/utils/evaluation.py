#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 评估模块

功能：
1. mAP计算
2. AP per class
3. Precision/Recall
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict


def compute_iou(box1, box2):
    """
    计算IoU
    
    参数:
        box1, box2: [x1, y1, x2, y2] - XYXY格式
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def compute_ap(recalls, precisions):
    """计算AP (11点插值)"""
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_detections(all_predictions, all_targets, num_classes, iou_threshold=0.5):
    """评估检测结果"""
    
    # 收集预测和目标
    class_predictions = defaultdict(list)
    class_targets = defaultdict(int)
    
    for img_id, (pred, target) in enumerate(zip(all_predictions, all_targets)):
        # 目标
        for label in target['labels']:
            class_targets[int(label)] += 1
        
        # 预测
        if len(pred['boxes']) > 0:
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                class_predictions[int(label)].append({
                    'confidence': float(score),
                    'image_id': img_id,
                    'box': box
                })
    
    # 计算AP
    aps = {}
    for class_id in range(num_classes):
        if class_id not in class_targets or class_targets[class_id] == 0:
            continue
        
        preds = class_predictions.get(class_id, [])
        if len(preds) == 0:
            aps[class_id] = 0.0
            continue
        
        preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
        
        num_targets = class_targets[class_id]
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        matched_targets = defaultdict(set)
        
        for pred_idx, pred in enumerate(preds):
            img_id = pred['image_id']
            pred_box = pred['box']
            
            target = all_targets[img_id]
            target_boxes = target['boxes']
            target_labels = target['labels']
            
            class_mask = (target_labels == class_id)
            class_target_boxes = target_boxes[class_mask]
            
            if len(class_target_boxes) == 0:
                fp[pred_idx] = 1
                continue
            
            max_iou = 0
            max_idx = -1
            
            for target_idx, target_box in enumerate(class_target_boxes):
                iou = compute_iou(pred_box, target_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = target_idx
            
            if max_iou >= iou_threshold:
                if max_idx not in matched_targets[img_id]:
                    tp[pred_idx] = 1
                    matched_targets[img_id].add(max_idx)
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / num_targets
        
        ap = compute_ap(recalls, precisions)
        aps[class_id] = ap
    
    mAP = np.mean(list(aps.values())) if len(aps) > 0 else 0.0
    
    return {
        'mAP': mAP,
        'AP_per_class': aps,
        'num_classes_evaluated': len(aps)
    }


def compute_map(all_predictions, all_targets, num_classes=20, iou_threshold=0.5):
    """计算mAP"""
    return evaluate_detections(all_predictions, all_targets, num_classes, iou_threshold)


if __name__ == "__main__":
    print("=" * 70)
    print("测试Experiment2数据加载器")
    print("=" * 70)
    
    dataset = DIORDataset(
        root_dir='../datasets/mini_dataset',
        split='train'
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    image, target = dataset[0]
    print(f"\n样本0:")
    print(f"  图像: {image.shape}")
    print(f"  边界框: {target['boxes'].shape}")
    print(f"  标签: {target['labels']}")
    
    print("\n✅ 测试完成！")


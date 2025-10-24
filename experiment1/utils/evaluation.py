#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment1 统一评估模块

功能：
1. IoU计算
2. mAP计算（11点插值法）
3. Precision/Recall计算
4. AP per class
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的IoU
    
    参数:
        box1: (4,) [x1, y1, x2, y2] - xyxy格式，像素坐标
        box2: (4,) [x1, y1, x2, y2] - xyxy格式，像素坐标
    
    返回:
        iou: float [0, 1]
    """
    # 交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 并集
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算IoU矩阵
    
    参数:
        boxes1: (N, 4) [x1, y1, x2, y2] - xyxy格式
        boxes2: (M, 4) [x1, y1, x2, y2] - xyxy格式
    
    返回:
        iou_matrix: (N, M) IoU矩阵
    """
    N = len(boxes1)
    M = len(boxes2)
    
    iou_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])
    
    return iou_matrix


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    计算Average Precision (11点插值法)
    
    参数:
        recalls: recall数组
        precisions: precision数组
    
    返回:
        ap: Average Precision
    """
    # 11点插值法 (PASCAL VOC)
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def match_predictions_to_targets(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    target_boxes: List[np.ndarray],
    target_labels: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Tuple[List, List]:
    """
    将预测匹配到目标
    
    参数:
        pred_boxes: List of (N_i, 4) 预测框
        pred_scores: List of (N_i,) 预测分数
        pred_labels: List of (N_i,) 预测标签
        target_boxes: List of (M_i, 4) 目标框
        target_labels: List of (M_i,) 目标标签
        iou_threshold: IoU阈值
    
    返回:
        all_matches: List of matches for each image
        all_scores: List of scores for each image
    """
    all_matches = []
    all_scores = []
    
    for pred_box, pred_score, pred_label, target_box, target_label in zip(
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels
    ):
        matches = []
        scores = []
        
        if len(pred_box) == 0:
            continue
        
        # 对每个预测
        for i, (box, score, label) in enumerate(zip(pred_box, pred_score, pred_label)):
            # 找到同类别的目标
            same_class_mask = (target_label == label)
            same_class_targets = target_box[same_class_mask]
            
            if len(same_class_targets) == 0:
                matches.append({'match': False, 'label': label})
                scores.append(score)
                continue
            
            # 计算IoU
            ious = np.array([compute_iou(box, target) for target in same_class_targets])
            max_iou = np.max(ious)
            max_idx = np.argmax(ious)
            
            # 判断匹配
            if max_iou >= iou_threshold:
                matches.append({
                    'match': True,
                    'label': label,
                    'iou': max_iou,
                    'target_idx': max_idx
                })
            else:
                matches.append({'match': False, 'label': label})
            
            scores.append(score)
        
        all_matches.append(matches)
        all_scores.append(scores)
    
    return all_matches, all_scores


def compute_precision_recall(
    all_predictions: List[Dict],
    all_targets: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict:
    """
    计算Precision和Recall
    
    参数:
        all_predictions: List of {'boxes': (N,4), 'scores': (N,), 'labels': (N,)}
        all_targets: List of {'boxes': (M,4), 'labels': (M,)}
        num_classes: 类别数量
        iou_threshold: IoU阈值
    
    返回:
        metrics: {'precision': float, 'recall': float, 'ap_per_class': dict}
    """
    # 收集每个类别的预测和目标
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
    
    # 计算每个类别的AP
    aps = {}
    
    for class_id in range(num_classes):
        if class_id not in class_targets or class_targets[class_id] == 0:
            continue
        
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
        
        matched_targets = defaultdict(set)
        
        for pred_idx, pred in enumerate(preds):
            img_id = pred['image_id']
            pred_box = pred['box']
            
            # 获取该图像的目标
            target = all_targets[img_id]
            target_boxes = target['boxes']
            target_labels = target['labels']
            
            # 找到同类别的目标
            class_mask = (target_labels == class_id)
            class_target_boxes = target_boxes[class_mask]
            
            if len(class_target_boxes) == 0:
                fp[pred_idx] = 1
                continue
            
            # 计算IoU
            max_iou = 0
            max_idx = -1
            
            for target_idx, target_box in enumerate(class_target_boxes):
                iou = compute_iou(pred_box, target_box)
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
    
    # 计算整体Precision和Recall
    total_tp = sum(1 for pred in class_predictions.values() for p in pred)
    total_targets = sum(class_targets.values())
    
    overall_precision = total_tp / (total_tp + 1) if total_tp > 0 else 0
    overall_recall = total_tp / total_targets if total_targets > 0 else 0
    
    return {
        'mAP': mAP,
        'AP_per_class': aps,
        'precision': overall_precision,
        'recall': overall_recall,
        'num_classes_evaluated': len(aps)
    }


def evaluate_detections(
    all_predictions: List[Dict],
    all_targets: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict:
    """
    评估检测结果
    
    参数:
        all_predictions: 所有预测
        all_targets: 所有目标
        num_classes: 类别数
        iou_threshold: IoU阈值
    
    返回:
        metrics: 评估指标
    """
    return compute_precision_recall(
        all_predictions,
        all_targets,
        num_classes,
        iou_threshold
    )


if __name__ == "__main__":
    print("=" * 70)
    print("测试评估模块")
    print("=" * 70)
    
    # 测试IoU
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([20, 20, 60, 60])
    iou = compute_iou(box1, box2)
    print(f"\nIoU测试: {iou:.4f}")
    
    # 测试IoU矩阵
    boxes1 = np.array([[10, 10, 50, 50], [100, 100, 150, 150]])
    boxes2 = np.array([[20, 20, 60, 60], [110, 110, 160, 160]])
    iou_matrix = compute_iou_matrix(boxes1, boxes2)
    print(f"\nIoU矩阵:\n{iou_matrix}")
    
    # 测试AP计算
    recalls = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    precisions = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    ap = compute_ap(recalls, precisions)
    print(f"\nAP: {ap:.4f}")
    
    print("\n✅ 评估模块测试完成！")


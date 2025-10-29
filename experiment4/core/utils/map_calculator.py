# -*- coding: utf-8 -*-
"""
mAP计算模块
实现目标检测的平均精度(AP)和平均AP(mAP)计算
"""

import numpy as np
from typing import List, Dict, Tuple


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)
    
    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]
    
    Returns:
        iou: float, IoU值 (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # 交集面积
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # 各自的面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # 并集面积
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def calculate_ap(pred_bboxes: List, pred_scores: List, gt_bboxes: List, iou_threshold: float = 0.5) -> float:
    """
    计算单个类别的平均精度(AP)
    
    使用11点插值法计算AP（PASCAL VOC标准）
    
    Args:
        pred_bboxes: List[[x1,y1,x2,y2]] 预测框列表
        pred_scores: List[float] 预测分数列表（相似度）
        gt_bboxes: List[[x1,y1,x2,y2]] 真实框列表
        iou_threshold: IoU阈值，默认0.5
    
    Returns:
        ap: float, 平均精度
    """
    if len(pred_bboxes) == 0:
        return 0.0
    
    if len(gt_bboxes) == 0:
        return 0.0
    
    # 按分数降序排序
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_bboxes_sorted = [pred_bboxes[i] for i in sorted_indices]
    pred_scores_sorted = [pred_scores[i] for i in sorted_indices]
    
    # 初始化TP和FP
    num_preds = len(pred_bboxes_sorted)
    tp = np.zeros(num_preds)
    fp = np.zeros(num_preds)
    
    # 记录已匹配的GT框
    matched_gt = set()
    
    # 对每个预测框，找到最佳匹配的GT框
    for i, pred_box in enumerate(pred_bboxes_sorted):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_bboxes):
            if j in matched_gt:
                continue  # 已匹配的GT框跳过
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # 判断是TP还是FP
        if best_iou >= iou_threshold:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1
    
    # 计算累积的TP和FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # 计算Precision和Recall
    num_gt = len(gt_bboxes)
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # 11点插值法计算AP
    ap = 0
    for t in np.linspace(0, 1, 11):
        # 找到recall >= t的所有precision
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return float(ap)


def calculate_map(all_predictions: Dict[str, List[Dict]], 
                  all_ground_truths: Dict[str, List], 
                  iou_threshold: float = 0.5) -> Tuple[float, Dict[str, float]]:
    """
    计算所有类别的mAP (mean Average Precision)
    
    Args:
        all_predictions: Dict[class_name -> List[{'bbox': [x1,y1,x2,y2], 'score': float}]]
        all_ground_truths: Dict[class_name -> List[[x1,y1,x2,y2]]]
        iou_threshold: IoU阈值，默认0.5
    
    Returns:
        mAP: float, 所有类别的平均AP
        per_class_ap: Dict[class_name -> AP], 每个类别的AP
    """
    aps = []
    per_class_ap = {}
    
    # 获取所有类别（GT中有的）
    all_classes = sorted(all_ground_truths.keys())
    
    for class_name in all_classes:
        # 获取该类别的预测和GT
        pred_data = all_predictions.get(class_name, [])
        gt_bboxes = all_ground_truths[class_name]
        
        # 如果没有GT，跳过
        if len(gt_bboxes) == 0:
            continue
        
        # 提取预测框和分数
        if len(pred_data) == 0:
            ap = 0.0
        else:
            pred_bboxes = [p['bbox'] for p in pred_data]
            pred_scores = [p['score'] for p in pred_data]
            ap = calculate_ap(pred_bboxes, pred_scores, gt_bboxes, iou_threshold)
        
        aps.append(ap)
        per_class_ap[class_name] = ap
    
    # 计算mAP
    if len(aps) == 0:
        mAP = 0.0
    else:
        mAP = float(np.mean(aps))
    
    return mAP, per_class_ap


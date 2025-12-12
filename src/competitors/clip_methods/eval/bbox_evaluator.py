# -*- coding: utf-8 -*-
"""
CLIP热图边界框评估工具（共享模块）

从热图生成边界框并计算IoU，支持多阈值评估
供所有clip_heatmap方法使用（surgeryclip_rs_det, diffclip_rs等）

来源: src/legacy_experiments/experiment6/exp/exp1/utils/bbox_evaluator.py
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple


def mask_to_bbox(mask: np.ndarray) -> List[float]:
    """
    从mask生成边界框
    
    Args:
        mask: [H, W] 二值mask（布尔数组或0/1数组）
    
    Returns:
        box: [x1, y1, x2, y2] 边界框坐标
    """
    if isinstance(mask, np.ndarray):
        mask = mask.astype(bool)
    else:
        mask = np.array(mask, dtype=bool)
    
    if mask.sum() == 0:
        return [0, 0, 0, 0]
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


def heatmap_to_bboxes(heatmap: np.ndarray, 
                      threshold: float = 0.7,
                      min_area: int = 20,
                      max_area: int = 40000) -> List[Dict]:
    """
    从热图生成边界框
    
    Args:
        heatmap: [H, W] 热图，范围[0, 1]
        threshold: 激活阈值（建议0.7-0.9）
        min_area: 最小区域面积（像素）
        max_area: 最大区域面积（像素），过滤掉太大的框
    
    Returns:
        bboxes: 边界框列表，每个包含 {'box': [x1, y1, x2, y2], 'score': float, 'area': int}
    """
    # 二值化
    binary = (heatmap > threshold).astype(np.uint8) * 255
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 添加尺寸过滤（最小和最大）
        if area < min_area or area > max_area:
            continue
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 添加宽高比过滤（可选，过滤极端长条形）
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        if aspect_ratio > 10:
            continue
        
        # 计算该区域的平均激活值作为score
        mask = np.zeros_like(heatmap, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, -1)
        score = float(heatmap[mask > 0].mean())
        
        bboxes.append({
            'box': [x, y, x + w, y + h],
            'score': score,
            'area': int(area),
            'width': w,
            'height': h
        })
    
    # 按score排序
    bboxes.sort(key=lambda x: x['score'], reverse=True)
    
    return bboxes


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        iou: IoU值
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0.0
    
    return iou


def evaluate_bboxes_with_gt(pred_bboxes: List[Dict], gt_bboxes: List[Dict], 
                             iou_threshold: float = 0.5) -> Dict:
    """
    评估预测框与GT框的匹配情况
    
    Args:
        pred_bboxes: 预测框列表
        gt_bboxes: GT框列表，每个包含 {'box': [x1, y1, x2, y2]}
        iou_threshold: IoU阈值
    
    Returns:
        metrics: 评估指标
    """
    if len(pred_bboxes) == 0:
        return {
            'num_pred': 0,
            'num_gt': len(gt_bboxes),
            'num_matched': 0,
            'precision': 0.0,
            'recall': 0.0,
            'avg_iou': 0.0,
            'matched_pairs': []
        }
    
    if len(gt_bboxes) == 0:
        return {
            'num_pred': len(pred_bboxes),
            'num_gt': 0,
            'num_matched': 0,
            'precision': 0.0,
            'recall': 0.0,
            'avg_iou': 0.0,
            'matched_pairs': []
        }
    
    # 计算IoU矩阵
    iou_matrix = np.zeros((len(pred_bboxes), len(gt_bboxes)))
    for i, pred in enumerate(pred_bboxes):
        for j, gt in enumerate(gt_bboxes):
            iou_matrix[i, j] = compute_iou(pred['box'], gt['box'])
    
    # 贪婪匹配
    matched_pairs = []
    matched_gt = set()
    
    for i in range(len(pred_bboxes)):
        best_j = -1
        best_iou = iou_threshold
        
        for j in range(len(gt_bboxes)):
            if j in matched_gt:
                continue
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        
        if best_j >= 0:
            matched_pairs.append({
                'pred_idx': i,
                'gt_idx': best_j,
                'iou': float(best_iou)
            })
            matched_gt.add(best_j)
    
    num_matched = len(matched_pairs)
    precision = num_matched / len(pred_bboxes) if len(pred_bboxes) > 0 else 0.0
    recall = num_matched / len(gt_bboxes) if len(gt_bboxes) > 0 else 0.0
    avg_iou = np.mean([p['iou'] for p in matched_pairs]) if matched_pairs else 0.0
    
    return {
        'num_pred': len(pred_bboxes),
        'num_gt': len(gt_bboxes),
        'num_matched': num_matched,
        'precision': precision,
        'recall': recall,
        'avg_iou': avg_iou,
        'matched_pairs': matched_pairs
    }


def multi_threshold_evaluation(heatmap: np.ndarray, gt_bboxes: List[Dict], 
                                 thresholds: List[float] = [0.5, 0.7, 0.9],
                                 min_area: int = 20,
                                 max_area: int = 40000) -> Dict:
    """
    使用多个阈值评估热图
    
    Args:
        heatmap: [H, W] 热图
        gt_bboxes: GT框列表
        thresholds: 要测试的阈值列表
        min_area: 最小区域面积
        max_area: 最大区域面积
    
    Returns:
        results: 每个阈值的评估结果
    """
    results = {}
    
    for thresh in thresholds:
        # 生成bbox
        pred_bboxes = heatmap_to_bboxes(
            heatmap, 
            threshold=thresh, 
            min_area=min_area,
            max_area=max_area
        )
        
        # 评估
        metrics = evaluate_bboxes_with_gt(pred_bboxes, gt_bboxes, iou_threshold=0.5)
        
        results[f'thresh_{thresh:.1f}'] = {
            'threshold': thresh,
            'pred_bboxes': pred_bboxes,
            'metrics': metrics
        }
    
    return results


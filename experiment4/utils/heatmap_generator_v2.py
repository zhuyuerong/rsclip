# -*- coding: utf-8 -*-
"""
热图生成器 V2 - 修复阈值逻辑
在原始相似度值上计算百分位，不归一化
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def generate_similarity_heatmap(image_features, text_features):
    """
    生成相似度热图（patch-text余弦相似度）
    
    Args:
        image_features: [B, N+1, D] 包含CLS token的特征
        text_features: [K, D] 文本特征
    
    Returns:
        similarity_map: [B, H, W, K] 空间化的相似度图
    """
    # 提取patch特征（去掉CLS token）
    patch_features = image_features[:, 1:, :]  # [B, N, D]
    
    # L2归一化
    patch_norm = F.normalize(patch_features, dim=-1, p=2)
    text_norm = F.normalize(text_features, dim=-1, p=2)
    
    # 计算余弦相似度
    similarity = torch.einsum('bnd,kd->bnk', patch_norm, text_norm)
    
    # Reshape到空间维度
    B, N, K = similarity.shape
    H = W = int(N ** 0.5)  # 例如 49 -> 7
    similarity_map = similarity.reshape(B, H, W, K)
    
    return similarity_map


def generate_bboxes_from_heatmap_v2(heatmap, image_size=224, threshold_percentile=75, min_box_size=10):
    """
    从热图生成边界框 - V2版本（修复阈值逻辑）
    
    关键修复：在原始相似度值上计算百分位，不进行归一化
    
    Args:
        heatmap: [H, W] numpy array, 原始相似度值（例如0.12-0.24）
        image_size: 目标图像尺寸
        threshold_percentile: 百分位阈值（直接在原始值上计算）
        min_box_size: 最小框尺寸
    
    Returns:
        bboxes: List of [x_min, y_min, x_max, y_max] in pixel coordinates
    """
    # 确保是numpy数组
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # 上采样到图像尺寸
    if heatmap.shape[0] != image_size:
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (image_size, image_size), 
                                     interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap.astype(np.float32)
    
    # ===== 关键修复：在原始值上计算百分位 =====
    # 不归一化！直接使用原始相似度值
    threshold = np.percentile(heatmap_resized, threshold_percentile)
    
    # 二值化
    mask = (heatmap_resized >= threshold).astype(np.uint8) * 255
    
    # 形态学操作（平滑）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 提取边界框
    bboxes = []
    for i in range(1, num_labels):  # 跳过背景（label=0）
        x, y, w, h, area = stats[i]
        
        # 过滤小框
        if w < min_box_size or h < min_box_size:
            continue
        
        # 转换为[x_min, y_min, x_max, y_max]格式
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)
    
    return bboxes


def generate_bboxes_topk(heatmap, image_size=224, top_k_ratio=0.1, min_box_size=10):
    """
    从热图生成边界框 - Top-K方法
    
    保留相似度最高的k%像素
    
    Args:
        heatmap: [H, W] numpy array
        image_size: 目标图像尺寸
        top_k_ratio: 保留的像素比例（例如0.1 = 前10%）
        min_box_size: 最小框尺寸
    
    Returns:
        bboxes: List of [x_min, y_min, x_max, y_max]
    """
    # 确保是numpy数组
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # 上采样
    if heatmap.shape[0] != image_size:
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (image_size, image_size))
    else:
        heatmap_resized = heatmap.astype(np.float32)
    
    # 计算top-k阈值
    num_pixels = heatmap_resized.size
    k = max(1, int(num_pixels * top_k_ratio))
    
    # 使用partition找第k大的值（更高效）
    flat_heatmap = heatmap_resized.flatten()
    threshold = np.partition(flat_heatmap, -k)[-k]
    
    # 二值化
    mask = (heatmap_resized >= threshold).astype(np.uint8) * 255
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 提取边界框
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if w < min_box_size or h < min_box_size:
            continue
        
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)
    
    return bboxes


def compute_bbox_score(heatmap, bbox, image_size=224):
    """
    计算边界框内的平均相似度得分
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # 上采样
    if heatmap.shape[0] != image_size:
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (image_size, image_size))
    else:
        heatmap_resized = heatmap.astype(np.float32)
    
    # 提取框内区域
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(int(x_min), image_size - 1))
    y_min = max(0, min(int(y_min), image_size - 1))
    x_max = max(0, min(int(x_max), image_size))
    y_max = max(0, min(int(y_max), image_size))
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    # 计算平均值
    region = heatmap_resized[y_min:y_max, x_min:x_max]
    score = float(region.mean())
    
    return score


# -*- coding: utf-8 -*-
"""
热图生成模块
基于CLIP Surgery论文的方法生成图像-文本相似度热图
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def generate_similarity_heatmap(image_features, text_features, attn_type='vv'):
    """
    生成图像-文本相似度热图
    
    基于CLIP Surgery论文的方法：
    1. 提取patch特征（去掉CLS token）
    2. 计算patch特征与文本特征的余弦相似度
    3. Reshape到空间维度形成热图
    
    Args:
        image_features: [B, N+1, 512] 最后一层输出（包含CLS）
        text_features: [K, 512] 类别文本特征
        attn_type: 'vv', 'qk', 或 'mixed' (暂不使用，仅保留接口兼容性)
    
    Returns:
        similarity_map: [B, H, W, K] 每个位置对每个类别的相似度
    """
    # 1. 提取patch特征（去掉CLS token）
    patch_features = image_features[:, 1:, :]  # [B, N, 512]
    
    # 2. L2归一化
    patch_features = F.normalize(patch_features, dim=-1, p=2)
    text_features = F.normalize(text_features, dim=-1, p=2)
    
    # 3. 计算余弦相似度
    # patch_features: [B, N, 512]
    # text_features: [K, 512]
    # similarity: [B, N, K]
    similarity = torch.einsum('bnd,kd->bnk', patch_features, text_features)
    
    # 4. Reshape到空间维度
    B, N, K = similarity.shape
    H = W = int(N ** 0.5)  # 7x7 for ViT-B/32
    assert H * W == N, f"Patch数量{N}不是完全平方数"
    
    similarity_map = similarity.reshape(B, H, W, K)
    
    return similarity_map


def generate_bboxes_from_heatmap(heatmap, threshold_percentile=90, image_size=224, min_box_size=5):
    """
    从热图生成检测框（阈值分割方法）
    
    使用连通域分析提取激活区域的边界框
    
    Args:
        heatmap: [H, W] 单个类别的热图 (numpy array)
        threshold_percentile: 激活阈值百分位（90 = top 10%）
        image_size: 原始图像尺寸
        min_box_size: 最小框尺寸（像素）
    
    Returns:
        bboxes: List[[x_min, y_min, x_max, y_max]] 检测框列表
    """
    # 确保是numpy array
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # 确保是float32类型
    if heatmap.dtype != np.float32:
        heatmap = heatmap.astype(np.float32)
    
    # 1. 上采样到原图尺寸
    heatmap_resized = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    
    # 2. 归一化到0-1
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # 3. 阈值分割（top 10%）
    threshold = np.percentile(heatmap_norm, threshold_percentile)
    mask = (heatmap_norm >= threshold).astype(np.uint8) * 255
    
    # 4. 形态学操作（去除噪声）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 5. 连通域分析
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. 提取边界框
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤太小的框
        if w > min_box_size and h > min_box_size:
            bboxes.append([x, y, x + w, y + h])
    
    # 如果没有检测到框，返回整个激活区域的边界
    if len(bboxes) == 0:
        # 找到所有非零位置
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            if x_max - x_min > min_box_size and y_max - y_min > min_box_size:
                bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    
    return bboxes


def compute_bbox_score(heatmap, bbox, image_size=224):
    """
    计算检测框的分数（框内平均相似度）
    
    Args:
        heatmap: [H, W] 热图 (numpy array或已上采样到image_size)
        bbox: [x_min, y_min, x_max, y_max] 像素坐标
        image_size: 原始图像尺寸
    
    Returns:
        score: float, 框内的平均相似度
    """
    # 确保是numpy array
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # 确保是float32类型
    if heatmap.dtype != np.float32:
        heatmap = heatmap.astype(np.float32)
    
    # 如果热图尺寸不是image_size，则上采样
    if heatmap.shape[0] != image_size:
        heatmap = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    
    # 提取bbox区域
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(x_min, image_size - 1))
    y_min = max(0, min(y_min, image_size - 1))
    x_max = max(0, min(x_max, image_size))
    y_max = max(0, min(y_max, image_size))
    
    # 计算平均值
    bbox_region = heatmap[y_min:y_max, x_min:x_max]
    
    if bbox_region.size == 0:
        return 0.0
    
    score = float(bbox_region.mean())
    return score


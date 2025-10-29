#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteCLIP 区域采样策略模块
提供三种采样策略：
1. 多阈值分层采样 (Multi-Threshold Layered Sampling)
2. 多尺度金字塔采样 (Multi-Scale Pyramid Sampling)
3. 多阈值显著性采样 (Multi-Threshold Saliency Sampling) - 轻量级
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """
    计算两个bbox的IoU
    
    参数:
        box1, box2: (x1, y1, x2, y2)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def non_max_suppression_regions(regions: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    NMS去除高度重叠的区域
    
    参数:
        regions: 区域列表
        iou_threshold: IoU阈值
    """
    if len(regions) == 0:
        return []
    
    regions = sorted(regions, key=lambda x: x.get('score', 0), reverse=True)
    
    keep = []
    while len(regions) > 0:
        best = regions.pop(0)
        keep.append(best)
        
        regions = [
            r for r in regions 
            if compute_iou(best['bbox'], r['bbox']) < iou_threshold
        ]
    
    return keep


def fallback_grid_sampling(image: np.ndarray, used_area: np.ndarray, n_needed: int) -> List[Dict]:
    """
    兜底的网格采样，填补空白区域
    
    参数:
        image: 输入图像
        used_area: 已使用区域的mask
        n_needed: 需要的区域数量
    """
    h, w = image.shape[:2]
    grid_size = 256
    stride = grid_size // 2
    
    fallback_regions = []
    for y in range(0, h - grid_size, stride):
        for x in range(0, w - grid_size, stride):
            # 检查这个区域是否已被覆盖
            patch_area = used_area[y:y+grid_size, x:x+grid_size]
            if patch_area.mean() < 0.3:  # 少于30%被覆盖
                fallback_regions.append({
                    'bbox': (x, y, x+grid_size, y+grid_size),
                    'area': grid_size * grid_size,
                    'saliency': 0,
                    'priority': 'fallback',
                    'weight': 0.2,
                    'score': 0
                })
                
                if len(fallback_regions) >= n_needed:
                    break
        if len(fallback_regions) >= n_needed:
            break
    
    return fallback_regions


def compute_coverage_and_supplement(
    image: np.ndarray, 
    regions: List[Dict], 
    min_coverage: float = 0.5
) -> List[Dict]:
    """
    计算覆盖率，如果不足则补充区域
    
    参数:
        image: 输入图像
        regions: 已有区域列表
        min_coverage: 最小覆盖率
    """
    # 计算覆盖率
    coverage_map = np.zeros(image.shape[:2], dtype=bool)
    for r in regions:
        x1, y1, x2, y2 = r['bbox']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        coverage_map[y1:y2, x1:x2] = True
    
    coverage_ratio = coverage_map.sum() / coverage_map.size
    print(f"   图像覆盖率: {coverage_ratio*100:.1f}%")
    
    if coverage_ratio < min_coverage:
        print(f"   ⚠️  覆盖率不足，添加补充区域...")
        n_supplement = max(10, int(len(regions) * 0.3))
        supplement_regions = fallback_grid_sampling(image, coverage_map, n_supplement)
        regions.extend(supplement_regions)
        print(f"   ✓ 补充了 {len(supplement_regions)} 个区域")
    
    return regions


# ==================== 策略1：多阈值分层采样 ====================

def multi_threshold_layered_sampling(
    image: np.ndarray, 
    min_regions: int = 50, 
    max_regions: int = 200
) -> List[Dict]:
    """
    策略1: 多阈值分层采样
    
    核心思想：
    - 低阈值保证覆盖 → 高召回
    - 分优先级 → 可以按重要性处理
    
    参数:
        image: 输入图像 (RGB或BGR格式)
        min_regions: 最少区域数
        max_regions: 最多区域数
    
    返回:
        区域列表，每个区域包含 bbox, area, saliency, priority, weight, score
    """
    print("\n🔍 策略1: 多阈值分层采样")
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 1. 计算显著性
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(gray)
    saliency_map = (saliency_map * 255).astype(np.uint8)
    
    # 增强对比度（提升弱显著区域的可见性）
    saliency_map = cv2.equalizeHist(saliency_map)
    
    # 2. 多个阈值分层
    thresholds = [
        ('critical', 200, 1.0),   # 极高优先级
        ('high', 150, 0.8),        # 高优先级
        ('medium', 100, 0.6),      # 中优先级
        ('low', 50, 0.4),          # 低优先级（保证召回）
    ]
    
    all_regions = []
    used_area = np.zeros(image.shape[:2], dtype=bool)  # 避免重复
    
    for priority, threshold, weight in thresholds:
        # 二值化
        _, binary = cv2.threshold(saliency_map, threshold, 255, cv2.THRESH_BINARY)
        
        # 排除已处理区域
        binary = binary & (~used_area.astype(np.uint8) * 255)
        
        # 形态学处理
        kernel_size = 11 if priority == 'low' else 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # 收集这一层的区域
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # 动态面积阈值（低优先级允许更小的区域）
            min_area = 300 if priority == 'critical' else 200 if priority == 'high' else 100
            
            if area > min_area:
                mask = (labels == i)
                avg_saliency = saliency_map[mask].mean()
                
                all_regions.append({
                    'bbox': (x, y, x+w, y+h),
                    'area': area,
                    'saliency': avg_saliency,
                    'priority': priority,
                    'weight': weight,
                    'score': avg_saliency * weight * np.log(area + 1)
                })
                
                # 标记为已使用（加padding避免紧邻）
                padding = 10
                y1 = max(0, y - padding)
                y2 = min(image.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(image.shape[1], x + w + padding)
                used_area[y1:y2, x1:x2] = True
    
    # 3. 排序但不截断（保留所有）
    all_regions.sort(key=lambda x: x['score'], reverse=True)
    
    # 4. 如果区域太少，补充网格采样
    if len(all_regions) < min_regions:
        print(f"   ⚠️  只找到{len(all_regions)}个区域，补充网格采样...")
        all_regions.extend(
            fallback_grid_sampling(image, used_area, min_regions - len(all_regions))
        )
    
    # 5. 如果太多，可以截断（但保留低优先级的）
    if len(all_regions) > max_regions:
        # 保证每个优先级都有代表
        kept_regions = []
        for priority in ['critical', 'high', 'medium', 'low']:
            priority_regions = [r for r in all_regions if r.get('priority') == priority]
            kept_regions.extend(priority_regions[:max_regions//4])
        all_regions = kept_regions
    
    print(f"   ✓ 采样得到 {len(all_regions)} 个区域")
    print(f"     - Critical: {sum(1 for r in all_regions if r.get('priority')=='critical')}")
    print(f"     - High: {sum(1 for r in all_regions if r.get('priority')=='high')}")
    print(f"     - Medium: {sum(1 for r in all_regions if r.get('priority')=='medium')}")
    print(f"     - Low: {sum(1 for r in all_regions if r.get('priority')=='low')}")
    
    return all_regions


# ==================== 策略2：多尺度金字塔采样 ====================

def multi_scale_pyramid_sampling(
    image: np.ndarray, 
    scales: List[float] = [1.0, 0.5, 0.25],
    max_regions: int = 200
) -> List[Dict]:
    """
    策略2: 多尺度金字塔采样
    
    核心思想：
    - 原尺度：找大目标
    - 0.5倍：找中等目标  
    - 0.25倍：找小目标聚集区
    
    参数:
        image: 输入图像
        scales: 尺度列表
        max_regions: 最大区域数
    
    返回:
        区域列表
    """
    print("\n🔍 策略2: 多尺度金字塔采样")
    
    all_regions = []
    
    for scale in scales:
        # 缩放图像
        h, w = image.shape[:2]
        scaled_h, scaled_w = int(h * scale), int(w * scale)
        
        if len(image.shape) == 3:
            scaled_img = cv2.resize(image, (scaled_w, scaled_h))
            gray = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.resize(image, (scaled_w, scaled_h))
        
        # 在当前尺度下采样
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, saliency_map = saliency.computeSaliency(gray)
        saliency_map = (saliency_map * 255).astype(np.uint8)
        
        # 自适应阈值（小尺度用更低阈值）
        threshold = 100 if scale == 1.0 else 70 if scale == 0.5 else 40
        _, binary = cv2.threshold(saliency_map, threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学
        kernel_size = max(5, int(9 * scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        # 提取区域并映射回原尺度
        for i in range(1, num_labels):
            x, y, w_box, h_box, area = stats[i]
            
            # 尺度相关的面积阈值
            min_area = 200 * (scale ** 2)
            if area > min_area:
                # 映射回原图坐标
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w_box / scale)
                h_orig = int(h_box / scale)
                
                mask = (labels == i)
                avg_saliency = saliency_map[mask].mean()
                
                all_regions.append({
                    'bbox': (x_orig, y_orig, x_orig + w_orig, y_orig + h_orig),
                    'area': w_orig * h_orig,
                    'saliency': avg_saliency,
                    'scale': scale,
                    'score': avg_saliency * np.log(area + 1)
                })
    
    # 去重（NMS）
    all_regions = non_max_suppression_regions(all_regions, iou_threshold=0.5)
    
    # 排序
    all_regions.sort(key=lambda x: x['score'], reverse=True)
    
    # 限制数量
    if len(all_regions) > max_regions:
        all_regions = all_regions[:max_regions]
    
    print(f"   ✓ 多尺度采样得到 {len(all_regions)} 个区域")
    for scale in scales:
        count = sum(1 for r in all_regions if r.get('scale') == scale)
        print(f"     - Scale {scale}: {count} 个区域")
    
    # 覆盖率检查
    all_regions = compute_coverage_and_supplement(image, all_regions)
    
    return all_regions


# ==================== 策略3：多阈值显著性采样（轻量级） ====================

def multi_threshold_saliency_sampling(
    image: np.ndarray,
    thresholds: List[float] = [0.1, 0.3, 0.5, 0.7],
    max_regions: int = 200,
    min_coverage: float = 0.5
) -> List[Dict]:
    """
    策略3: 多阈值显著性采样（轻量级）
    
    核心思想：
    - 只计算一次显著性图
    - 使用多个阈值模拟多尺度
    - 通过阈值捕获不同显著性水平的区域
    
    具体步骤：
    1. 计算原图的显著性图
    2. 使用一组阈值对显著性图二值化，得到多个二值图
    3. 从每个二值图中提取连通域，并记录其阈值（即显著性水平）
    4. 合并所有连通域，并计算每个区域的平均显著性
    5. 使用NMS去重，然后按平均显著性排序
    6. 覆盖度检查，补充未覆盖区域
    
    参数:
        image: 输入图像 (RGB或BGR格式)
        thresholds: 显著性阈值列表 (0-1之间)
        max_regions: 最大区域数
        min_coverage: 最小覆盖率
    
    返回:
        区域列表
    """
    print("\n🔍 策略3: 多阈值显著性采样（轻量级）")
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 1. 计算显著性图（只计算一次）
    print("   计算显著性图...")
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(gray)
    
    if not success:
        print("   ❌ 显著性计算失败")
        return []
    
    # 归一化到 0-1
    saliency_map = saliency_map.astype(np.float32)
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # 2. 使用多个阈值提取区域
    all_regions = []
    
    for threshold in thresholds:
        # 二值化
        binary = (saliency_map > threshold).astype(np.uint8) * 255
        
        # 形态学处理（根据阈值调整kernel大小）
        kernel_size = 5 if threshold > 0.5 else 9
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 3. 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # 提取区域
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # 动态面积阈值（高阈值允许更小区域）
            min_area = 500 if threshold < 0.3 else 300 if threshold < 0.5 else 150
            
            if area > min_area:
                # 4. 计算该区域的平均显著性（使用原显著性图）
                mask = (labels == i)
                avg_saliency = saliency_map[mask].mean()
                
                all_regions.append({
                    'bbox': (x, y, x+w, y+h),
                    'area': area,
                    'saliency': avg_saliency,
                    'threshold': threshold,
                    'score': avg_saliency * np.log(area + 1)
                })
    
    print(f"   ✓ 提取到 {len(all_regions)} 个初始区域")
    
    # 5. NMS去重
    print("   执行NMS去重...")
    all_regions = non_max_suppression_regions(all_regions, iou_threshold=0.5)
    
    # 按平均显著性排序
    all_regions.sort(key=lambda x: x['score'], reverse=True)
    
    # 限制数量
    if len(all_regions) > max_regions:
        all_regions = all_regions[:max_regions]
    
    print(f"   ✓ NMS后保留 {len(all_regions)} 个区域")
    for threshold in thresholds:
        count = sum(1 for r in all_regions if r.get('threshold') == threshold)
        print(f"     - Threshold {threshold}: {count} 个区域")
    
    # 6. 覆盖度检查和补充
    all_regions = compute_coverage_and_supplement(image, all_regions, min_coverage)
    
    return all_regions


# ==================== 统一接口 ====================

def sample_regions(
    image: np.ndarray,
    strategy: str = "multi_threshold_saliency",
    **kwargs
) -> List[Dict]:
    """
    统一的区域采样接口
    
    参数:
        image: 输入图像
        strategy: 采样策略
            - "layered": 多阈值分层采样
            - "pyramid": 多尺度金字塔采样
            - "multi_threshold_saliency": 多阈值显著性采样（默认，轻量级）
        **kwargs: 传递给具体策略的参数
    
    返回:
        区域列表
    """
    if strategy == "layered":
        return multi_threshold_layered_sampling(image, **kwargs)
    elif strategy == "pyramid":
        return multi_scale_pyramid_sampling(image, **kwargs)
    elif strategy == "multi_threshold_saliency":
        return multi_threshold_saliency_sampling(image, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # 测试代码
    print("测试采样策略...")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # 测试三种策略
    for strategy in ["layered", "pyramid", "multi_threshold_saliency"]:
        print(f"\n{'='*70}")
        print(f"测试策略: {strategy}")
        print(f"{'='*70}")
        
        regions = sample_regions(test_image, strategy=strategy, max_regions=50)
        print(f"最终得到 {len(regions)} 个区域\n")


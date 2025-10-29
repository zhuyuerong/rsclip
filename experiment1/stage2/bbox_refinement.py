#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界框微调模块
结合采样显著性和对比学习得分进行框优化
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Dict, Optional


class BBoxRefinement:
    """边界框微调器"""
    
    def __init__(self, model, preprocess, device='cuda'):
        """
        初始化
        
        参数:
            model: RemoteCLIP模型
            preprocess: 图像预处理函数
            device: 计算设备
        """
        self.model = model
        self.preprocess = preprocess
        self.device = device
    
    def compute_contrast_score(self, crop_tensor, positive_prototype, negative_prototype):
        """
        计算对比得分
        
        参数:
            crop_tensor: 裁剪图像tensor
            positive_prototype: 正样本原型特征
            negative_prototype: 负样本原型特征
        
        返回:
            对比得分
        """
        with torch.no_grad():
            feat = self.model.encode_image(crop_tensor.to(self.device))
            feat /= feat.norm(dim=-1, keepdim=True)
            
            # 与正原型的相似度
            pos_sim = (feat @ positive_prototype.T).item()
            
            # 与负原型的相似度（负数）
            if negative_prototype is not None:
                neg_sim = (feat @ negative_prototype.T).item()
                # 对比分数 = 正相似度 - 负相似度
                contrast_score = pos_sim - neg_sim
            else:
                contrast_score = pos_sim
        
        return contrast_score
    
    def refine_bbox_with_saliency_and_contrast(
        self,
        image: np.ndarray,
        initial_bbox: Tuple[int, int, int, int],
        saliency_map: np.ndarray,
        positive_prototype: torch.Tensor,
        negative_prototype: Optional[torch.Tensor] = None,
        search_radius: int = 20,
        size_delta: int = 30,
        n_candidates: int = 50
    ) -> Dict:
        """
        结合显著性和对比学习得分进行边界框微调
        
        策略：
        1. 基于显著性图生成候选框（位置微调）
        2. 基于对比得分评估候选框（大小微调）
        3. 选择综合得分最高的框
        
        参数:
            image: 原始图像 (RGB)
            initial_bbox: 初始边界框 (x1, y1, x2, y2)
            saliency_map: 显著性图（第一步采样生成的）
            positive_prototype: 正样本原型特征
            negative_prototype: 负样本原型特征
            search_radius: 位置搜索半径（像素）
            size_delta: 尺寸调整范围（像素）
            n_candidates: 候选框数量
        
        返回:
            优化后的结果字典
        """
        x1, y1, x2, y2 = initial_bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        candidates = []
        
        # 策略1: 基于显著性的位置微调
        # 在初始框周围寻找显著性峰值
        search_region = saliency_map[
            max(0, cy - search_radius):min(saliency_map.shape[0], cy + search_radius),
            max(0, cx - search_radius):min(saliency_map.shape[1], cx + search_radius)
        ]
        
        if search_region.size > 0:
            # 找显著性最高的位置
            local_max_y, local_max_x = np.unravel_index(
                search_region.argmax(), search_region.shape
            )
            new_cx = max(0, cx - search_radius) + local_max_x
            new_cy = max(0, cy - search_radius) + local_max_y
        else:
            new_cx, new_cy = cx, cy
        
        # 策略2: 生成多个候选框（位置+尺寸变化）
        for i in range(n_candidates):
            # 2.1 位置偏移（在显著性引导的中心周围采样）
            if i == 0:
                # 第一个候选：原始框
                cand_cx, cand_cy = cx, cy
                cand_w, cand_h = w, h
            elif i == 1:
                # 第二个候选：显著性引导的中心
                cand_cx, cand_cy = new_cx, new_cy
                cand_w, cand_h = w, h
            else:
                # 其他候选：随机扰动
                offset_x = np.random.randint(-search_radius, search_radius + 1)
                offset_y = np.random.randint(-search_radius, search_radius + 1)
                cand_cx = cx + offset_x
                cand_cy = cy + offset_y
                
                # 2.2 尺寸变化
                delta_w = np.random.randint(-size_delta, size_delta + 1)
                delta_h = np.random.randint(-size_delta, size_delta + 1)
                cand_w = max(20, w + delta_w)
                cand_h = max(20, h + delta_h)
            
            # 计算新的边界框
            new_x1 = max(0, cand_cx - cand_w // 2)
            new_y1 = max(0, cand_cy - cand_h // 2)
            new_x2 = min(image.shape[1], cand_cx + cand_w // 2)
            new_y2 = min(image.shape[0], cand_cy + cand_h // 2)
            
            # 确保框有效
            if new_x2 - new_x1 < 10 or new_y2 - new_y1 < 10:
                continue
            
            # 裁剪并计算得分
            crop = image[new_y1:new_y2, new_x1:new_x2]
            if crop.size == 0:
                continue
            
            # 计算显著性得分（框内显著性平均值）
            bbox_saliency = saliency_map[new_y1:new_y2, new_x1:new_x2].mean()
            
            # 计算对比得分
            crop_pil = Image.fromarray(crop)
            crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
            contrast_score = self.compute_contrast_score(
                crop_tensor, positive_prototype, negative_prototype
            )
            
            # 综合得分 = α * 对比得分 + β * 显著性得分
            alpha, beta = 0.7, 0.3  # 对比得分权重更高
            composite_score = alpha * contrast_score + beta * (bbox_saliency / 255.0)
            
            candidates.append({
                'bbox': (new_x1, new_y1, new_x2, new_y2),
                'saliency_score': float(bbox_saliency / 255.0),
                'contrast_score': float(contrast_score),
                'composite_score': float(composite_score),
                'center': (cand_cx, cand_cy),
                'size': (cand_w, cand_h)
            })
        
        # 选择综合得分最高的框
        if len(candidates) == 0:
            return {
                'bbox': initial_bbox,
                'saliency_score': 0.0,
                'contrast_score': 0.0,
                'composite_score': 0.0,
                'refined': False
            }
        
        candidates.sort(key=lambda x: x['composite_score'], reverse=True)
        best_candidate = candidates[0]
        best_candidate['refined'] = True
        best_candidate['n_candidates_tested'] = len(candidates)
        
        return best_candidate
    
    def refine_bbox_boundary(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        saliency_map: np.ndarray,
        positive_prototype: torch.Tensor,
        negative_prototype: Optional[torch.Tensor] = None,
        edge_step: int = 5,
        max_iterations: int = 10
    ) -> Dict:
        """
        基于边界优化的框微调
        
        策略：
        1. 从四个边界分别向内/向外移动
        2. 每次移动后计算综合得分
        3. 保留得分提升的移动
        4. 迭代直到收敛或达到最大次数
        
        参数:
            image: 原始图像
            bbox: 初始边界框
            saliency_map: 显著性图
            positive_prototype: 正样本原型
            negative_prototype: 负样本原型
            edge_step: 边界移动步长
            max_iterations: 最大迭代次数
        
        返回:
            优化后的结果
        """
        x1, y1, x2, y2 = bbox
        best_bbox = bbox
        
        # 计算初始得分
        initial_crop = image[y1:y2, x1:x2]
        if initial_crop.size == 0:
            return {'bbox': bbox, 'refined': False}
        
        initial_crop_pil = Image.fromarray(initial_crop)
        initial_tensor = self.preprocess(initial_crop_pil).unsqueeze(0)
        best_score = self.compute_contrast_score(
            initial_tensor, positive_prototype, negative_prototype
        )
        
        # 迭代优化
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 尝试移动四条边界
            moves = [
                ('left', -edge_step, 0, 0, 0),   # 左边向左
                ('left', edge_step, 0, 0, 0),    # 左边向右
                ('top', 0, -edge_step, 0, 0),    # 上边向上
                ('top', 0, edge_step, 0, 0),     # 上边向下
                ('right', 0, 0, edge_step, 0),   # 右边向右
                ('right', 0, 0, -edge_step, 0),  # 右边向左
                ('bottom', 0, 0, 0, edge_step),  # 下边向下
                ('bottom', 0, 0, 0, -edge_step), # 下边向上
            ]
            
            for edge_name, dx1, dy1, dx2, dy2 in moves:
                new_x1 = max(0, x1 + dx1)
                new_y1 = max(0, y1 + dy1)
                new_x2 = min(image.shape[1], x2 + dx2)
                new_y2 = min(image.shape[0], y2 + dy2)
                
                # 确保框有效
                if new_x2 - new_x1 < 20 or new_y2 - new_y1 < 20:
                    continue
                
                # 裁剪并计算得分
                new_crop = image[new_y1:new_y2, new_x1:new_x2]
                if new_crop.size == 0:
                    continue
                
                # 显著性得分
                saliency_score = saliency_map[new_y1:new_y2, new_x1:new_x2].mean() / 255.0
                
                # 对比得分
                crop_pil = Image.fromarray(new_crop)
                crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
                contrast_score = self.compute_contrast_score(
                    crop_tensor, positive_prototype, negative_prototype
                )
                
                # 综合得分
                composite = 0.7 * contrast_score + 0.3 * saliency_score
                
                if composite > best_score:
                    best_score = composite
                    best_bbox = (new_x1, new_y1, new_x2, new_y2)
                    x1, y1, x2, y2 = best_bbox
                    improved = True
        
        return {
            'bbox': best_bbox,
            'initial_bbox': bbox,
            'contrast_score': float(best_score),
            'iterations': iteration,
            'refined': best_bbox != bbox
        }
    
    def refine_bbox_multi_scale(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        saliency_map: np.ndarray,
        positive_prototype: torch.Tensor,
        negative_prototype: Optional[torch.Tensor] = None,
        scale_factors: list = [0.8, 0.9, 1.0, 1.1, 1.2]
    ) -> Dict:
        """
        多尺度框微调
        
        策略：
        1. 保持中心不变
        2. 尝试不同的尺度因子
        3. 结合显著性和对比得分选择最佳尺度
        
        参数:
            image: 原始图像
            bbox: 初始边界框
            saliency_map: 显著性图
            positive_prototype: 正样本原型
            negative_prototype: 负样本原型
            scale_factors: 尺度因子列表
        
        返回:
            优化后的结果
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = x2 - x1, y2 - y1
        
        best_score = -float('inf')
        best_result = None
        
        for scale in scale_factors:
            # 计算新的尺寸
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 保持中心，调整边界
            new_x1 = max(0, cx - new_w // 2)
            new_y1 = max(0, cy - new_h // 2)
            new_x2 = min(image.shape[1], cx + new_w // 2)
            new_y2 = min(image.shape[0], cy + new_h // 2)
            
            # 确保框有效
            if new_x2 - new_x1 < 10 or new_y2 - new_y1 < 10:
                continue
            
            # 裁剪
            crop = image[new_y1:new_y2, new_x1:new_x2]
            if crop.size == 0:
                continue
            
            # 显著性得分（框内显著性均值）
            bbox_saliency = saliency_map[new_y1:new_y2, new_x1:new_x2].mean() / 255.0
            
            # 对比得分
            crop_pil = Image.fromarray(crop)
            crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
            contrast_score = self.compute_contrast_score(
                crop_tensor, positive_prototype, negative_prototype
            )
            
            # 综合得分
            composite = 0.7 * contrast_score + 0.3 * bbox_saliency
            
            if composite > best_score:
                best_score = composite
                best_result = {
                    'bbox': (new_x1, new_y1, new_x2, new_y2),
                    'scale': scale,
                    'saliency_score': float(bbox_saliency),
                    'contrast_score': float(contrast_score),
                    'composite_score': float(composite),
                    'size': (new_w, new_h)
                }
        
        if best_result is None:
            best_result = {
                'bbox': bbox,
                'scale': 1.0,
                'saliency_score': 0.0,
                'contrast_score': 0.0,
                'composite_score': 0.0,
                'refined': False
            }
        else:
            best_result['refined'] = best_result['bbox'] != bbox
            best_result['initial_bbox'] = bbox
        
        return best_result
    
    def refine_bbox_hybrid(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        saliency_map: np.ndarray,
        positive_prototype: torch.Tensor,
        negative_prototype: Optional[torch.Tensor] = None,
        method: str = 'both'
    ) -> Dict:
        """
        混合微调策略
        
        结合位置搜索和尺度优化
        
        参数:
            image: 原始图像
            bbox: 初始边界框
            saliency_map: 显著性图
            positive_prototype: 正样本原型
            negative_prototype: 负样本原型
            method: 'position', 'scale', 'both', 'boundary'
        
        返回:
            优化结果
        """
        if method == 'position':
            # 只优化位置（基于显著性峰值）
            return self.refine_bbox_with_saliency_and_contrast(
                image, bbox, saliency_map, positive_prototype, negative_prototype,
                search_radius=20, size_delta=0, n_candidates=20
            )
        
        elif method == 'scale':
            # 只优化尺寸
            return self.refine_bbox_multi_scale(
                image, bbox, saliency_map, positive_prototype, negative_prototype
            )
        
        elif method == 'boundary':
            # 边界逐步优化
            return self.refine_bbox_boundary(
                image, bbox, saliency_map, positive_prototype, negative_prototype
            )
        
        else:  # 'both'
            # 先优化位置，再优化尺度
            # 第一步：位置优化
            result1 = self.refine_bbox_with_saliency_and_contrast(
                image, bbox, saliency_map, positive_prototype, negative_prototype,
                search_radius=15, size_delta=0, n_candidates=15
            )
            
            # 第二步：基于优化后的位置进行尺度优化
            result2 = self.refine_bbox_multi_scale(
                image, result1['bbox'], saliency_map, 
                positive_prototype, negative_prototype
            )
            
            result2['method'] = 'hybrid'
            result2['position_refined'] = result1.get('refined', False)
            result2['scale_refined'] = result2.get('refined', False)
            result2['both_refined'] = result2['position_refined'] or result2['scale_refined']
            
            return result2


def compute_saliency_map(image: np.ndarray) -> np.ndarray:
    """
    计算显著性图（与采样策略保持一致）
    
    参数:
        image: 输入图像 (RGB)
    
    返回:
        显著性图 (0-255)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(gray)
    
    if not success:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    saliency_map = (saliency_map * 255).astype(np.uint8)
    saliency_map = cv2.equalizeHist(saliency_map)
    
    return saliency_map


if __name__ == "__main__":
    # 测试代码
    print("边界框微调模块测试")
    print("="*70)
    
    # 创建测试图像和显著性图
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    saliency_map = compute_saliency_map(test_image)
    
    # 创建测试bbox
    test_bbox = (100, 100, 200, 200)
    
    print(f"初始bbox: {test_bbox}")
    print(f"显著性图形状: {saliency_map.shape}")
    print("✅ 模块导入成功")


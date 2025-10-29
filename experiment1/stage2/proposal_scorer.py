#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2: 候选框打分模块
对分类后的候选框进行置信度打分和排序
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
import os
import sys

# 添加父目录到路径以导入原有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ProposalScorer:
    """候选框打分器"""
    
    def __init__(self, scoring_method: str = 'composite'):
        """
        初始化候选框打分器
        
        参数:
            scoring_method: 打分方法 ('confidence', 'saliency', 'composite')
        """
        self.scoring_method = scoring_method
        self.supported_methods = ['confidence', 'saliency', 'composite']
        
        if scoring_method not in self.supported_methods:
            raise ValueError(f"不支持的打分方法: {scoring_method}")
    
    def compute_saliency_score(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        计算显著性分数
        
        参数:
            image: 输入图像
            bbox: 边界框 (x1, y1, x2, y2)
        
        返回:
            显著性分数
        """
        x1, y1, x2, y2 = bbox
        
        # 裁剪区域
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return 0.0
        
        # 转换为灰度图
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        
        # 计算显著性
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(gray)
        
        if not success:
            return 0.0
        
        # 计算平均显著性
        saliency_score = saliency_map.mean()
        
        return float(saliency_score)
    
    def compute_contrast_score(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        计算对比度分数
        
        参数:
            image: 输入图像
            bbox: 边界框 (x1, y1, x2, y2)
        
        返回:
            对比度分数
        """
        x1, y1, x2, y2 = bbox
        
        # 裁剪区域
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return 0.0
        
        # 转换为灰度图
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        
        # 计算对比度（标准差）
        contrast_score = gray.std() / 255.0
        
        return float(contrast_score)
    
    def compute_size_score(self, bbox: Tuple[int, int, int, int], 
                          image_shape: Tuple[int, int],
                          optimal_size_ratio: float = 0.1) -> float:
        """
        计算尺寸分数
        
        参数:
            bbox: 边界框 (x1, y1, x2, y2)
            image_shape: 图像尺寸 (height, width)
            optimal_size_ratio: 最优尺寸比例
        
        返回:
            尺寸分数
        """
        x1, y1, x2, y2 = bbox
        h, w = image_shape
        
        # 计算边界框面积
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = h * w
        
        # 计算面积比例
        size_ratio = bbox_area / image_area
        
        # 计算与最优比例的差异（越小越好）
        size_diff = abs(size_ratio - optimal_size_ratio)
        
        # 转换为分数（0-1，越接近最优比例分数越高）
        size_score = max(0, 1 - size_diff / optimal_size_ratio)
        
        return float(size_score)
    
    def compute_aspect_ratio_score(self, bbox: Tuple[int, int, int, int],
                                  optimal_aspect_ratio: float = 1.0) -> float:
        """
        计算宽高比分数
        
        参数:
            bbox: 边界框 (x1, y1, x2, y2)
            optimal_aspect_ratio: 最优宽高比
        
        返回:
            宽高比分数
        """
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        
        if height == 0:
            return 0.0
        
        aspect_ratio = width / height
        
        # 计算与最优宽高比的差异
        aspect_diff = abs(aspect_ratio - optimal_aspect_ratio)
        
        # 转换为分数
        aspect_score = max(0, 1 - aspect_diff / optimal_aspect_ratio)
        
        return float(aspect_score)
    
    def compute_composite_score(self, proposal: Dict, image: np.ndarray,
                               weights: Dict[str, float] = None) -> float:
        """
        计算综合分数
        
        参数:
            proposal: 候选框信息
            image: 输入图像
            weights: 权重字典
        
        返回:
            综合分数
        """
        if weights is None:
            weights = {
                'confidence': 0.4,
                'saliency': 0.3,
                'contrast': 0.1,
                'size': 0.1,
                'aspect_ratio': 0.1
            }
        
        bbox = proposal['bbox']
        
        # 置信度分数
        confidence_score = proposal.get('prediction_confidence', 0)
        
        # 显著性分数
        saliency_score = self.compute_saliency_score(image, bbox)
        
        # 对比度分数
        contrast_score = self.compute_contrast_score(image, bbox)
        
        # 尺寸分数
        size_score = self.compute_size_score(bbox, image.shape[:2])
        
        # 宽高比分数
        aspect_ratio_score = self.compute_aspect_ratio_score(bbox)
        
        # 加权综合分数
        composite_score = (
            weights['confidence'] * confidence_score +
            weights['saliency'] * saliency_score +
            weights['contrast'] * contrast_score +
            weights['size'] * size_score +
            weights['aspect_ratio'] * aspect_ratio_score
        )
        
        return float(composite_score)
    
    def score_proposals(self, proposals: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        对候选框进行打分
        
        参数:
            proposals: 候选框列表
            image: 输入图像
        
        返回:
            打分后的候选框列表
        """
        print(f"\n🔧 对 {len(proposals)} 个候选框进行打分...")
        
        scored_proposals = []
        
        for proposal in proposals:
            # 根据打分方法计算分数
            if self.scoring_method == 'confidence':
                score = proposal.get('prediction_confidence', 0)
            elif self.scoring_method == 'saliency':
                score = self.compute_saliency_score(image, proposal['bbox'])
            elif self.scoring_method == 'composite':
                score = self.compute_composite_score(proposal, image)
            else:
                score = 0
            
            # 更新候选框信息
            proposal['score'] = score
            proposal['scoring_method'] = self.scoring_method
            
            # 添加详细分数信息
            if self.scoring_method == 'composite':
                proposal['detailed_scores'] = {
                    'confidence': proposal.get('prediction_confidence', 0),
                    'saliency': self.compute_saliency_score(image, proposal['bbox']),
                    'contrast': self.compute_contrast_score(image, proposal['bbox']),
                    'size': self.compute_size_score(proposal['bbox'], image.shape[:2]),
                    'aspect_ratio': self.compute_aspect_ratio_score(proposal['bbox'])
                }
            
            scored_proposals.append(proposal)
        
        print(f"✅ 打分完成")
        
        return scored_proposals
    
    def rank_proposals_by_score(self, proposals: List[Dict]) -> List[Dict]:
        """
        按分数排序候选框
        
        参数:
            proposals: 候选框列表
        
        返回:
            排序后的候选框列表
        """
        print(f"\n🔧 按分数排序候选框...")
        
        sorted_proposals = sorted(
            proposals, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )
        
        print(f"✅ 排序完成")
        
        return sorted_proposals
    
    def filter_proposals_by_score(self, proposals: List[Dict], 
                                min_score: float = 0.1) -> List[Dict]:
        """
        按分数过滤候选框
        
        参数:
            proposals: 候选框列表
            min_score: 最小分数阈值
        
        返回:
            过滤后的候选框列表
        """
        print(f"\n🔧 按分数过滤候选框 (阈值: {min_score})...")
        
        filtered_proposals = []
        
        for proposal in proposals:
            score = proposal.get('score', 0)
            
            if score >= min_score:
                filtered_proposals.append(proposal)
        
        print(f"✅ 分数过滤: {len(proposals)} -> {len(filtered_proposals)} 个候选框")
        
        return filtered_proposals
    
    def score_proposals_pipeline(self, proposals: List[Dict], image: np.ndarray,
                               min_score: float = 0.1,
                               top_k: int = 10) -> List[Dict]:
        """
        完整的候选框打分流水线
        
        参数:
            proposals: 候选框列表
            image: 输入图像
            min_score: 最小分数阈值
            top_k: 返回前K个结果
        
        返回:
            打分后的候选框列表
        """
        print(f"\n🚀 开始候选框打分流水线...")
        
        # 1. 打分
        scored_proposals = self.score_proposals(proposals, image)
        
        # 2. 按分数过滤
        filtered_proposals = self.filter_proposals_by_score(
            scored_proposals, 
            min_score
        )
        
        # 3. 按分数排序
        ranked_proposals = self.rank_proposals_by_score(filtered_proposals)
        
        # 4. 返回Top-K
        final_proposals = ranked_proposals[:top_k]
        
        print(f"✅ 候选框打分流水线完成，返回 {len(final_proposals)} 个候选框")
        
        return final_proposals
    
    def get_scoring_statistics(self, proposals: List[Dict]) -> Dict:
        """
        获取打分统计信息
        
        参数:
            proposals: 候选框列表
        
        返回:
            统计信息字典
        """
        if not proposals:
            return {}
        
        scores = [p.get('score', 0) for p in proposals]
        
        stats = {
            'total_proposals': len(proposals),
            'score_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'scoring_method': self.scoring_method
        }
        
        # 如果有详细分数信息，添加统计
        if proposals and 'detailed_scores' in proposals[0]:
            detailed_stats = {}
            for key in proposals[0]['detailed_scores'].keys():
                values = [p['detailed_scores'][key] for p in proposals]
                detailed_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            stats['detailed_score_stats'] = detailed_stats
        
        return stats


def main():
    """测试候选框打分器"""
    print("=" * 70)
    print("测试候选框打分器")
    print("=" * 70)
    
    # 创建模拟候选框
    mock_proposals = [
        {
            'proposal_id': 0,
            'bbox': (100, 100, 200, 200),
            'predicted_class': 'airplane',
            'prediction_confidence': 0.8
        },
        {
            'proposal_id': 1,
            'bbox': (300, 200, 400, 300),
            'predicted_class': 'building',
            'prediction_confidence': 0.6
        },
        {
            'proposal_id': 2,
            'bbox': (150, 250, 250, 350),
            'predicted_class': 'vehicle',
            'prediction_confidence': 0.9
        }
    ]
    
    # 创建模拟图像
    mock_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # 测试不同打分方法
    methods = ['confidence', 'saliency', 'composite']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"测试打分方法: {method}")
        print(f"{'='*50}")
        
        # 创建打分器
        scorer = ProposalScorer(method)
        
        # 运行完整流水线
        scored_proposals = scorer.score_proposals_pipeline(
            mock_proposals,
            mock_image,
            min_score=0.1,
            top_k=5
        )
        
        # 获取统计信息
        stats = scorer.get_scoring_statistics(scored_proposals)
        print(f"\n📊 打分统计:")
        print(f"  总候选框数: {stats.get('total_proposals', 0)}")
        print(f"  分数范围: {stats.get('score_stats', {}).get('min', 0):.3f} - {stats.get('score_stats', {}).get('max', 0):.3f}")
        print(f"  平均分数: {stats.get('score_stats', {}).get('mean', 0):.3f}")
        
        # 显示每个候选框的分数
        print(f"\n📋 打分结果:")
        for i, proposal in enumerate(scored_proposals):
            print(f"  候选框 {i+1}:")
            print(f"    位置: {proposal['bbox']}")
            print(f"    类别: {proposal.get('predicted_class', 'unknown')}")
            print(f"    分数: {proposal.get('score', 0):.3f}")
            
            # 显示详细分数（如果是综合打分）
            if 'detailed_scores' in proposal:
                print(f"    详细分数:")
                for key, value in proposal['detailed_scores'].items():
                    print(f"      {key}: {value:.3f}")
    
    print("\n✅ 候选框打分器测试完成!")


if __name__ == "__main__":
    main()

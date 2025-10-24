#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1: 区域采样模块
基于原有sampling.py，专门用于实验中的区域采样
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import sys

# 添加父目录到路径以导入原有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sampling import (
    sample_regions, 
    multi_threshold_saliency_sampling,
    multi_threshold_layered_sampling,
    multi_scale_pyramid_sampling,
    non_max_suppression_regions
)


class ExperimentRegionSampler:
    """实验用区域采样器"""
    
    def __init__(self, strategy: str = 'multi_threshold_saliency'):
        """
        初始化区域采样器
        
        参数:
            strategy: 采样策略
        """
        self.strategy = strategy
        self.supported_strategies = [
            'multi_threshold_saliency',
            'layered', 
            'pyramid'
        ]
        
        if strategy not in self.supported_strategies:
            raise ValueError(f"不支持的采样策略: {strategy}")
    
    def sample_regions(self, image: np.ndarray, max_regions: int = 50) -> List[Dict]:
        """
        采样感兴趣区域
        
        参数:
            image: 输入图像
            max_regions: 最大区域数
        
        返回:
            区域列表
        """
        print(f"\n🔍 使用策略 '{self.strategy}' 进行区域采样...")
        
        regions = sample_regions(
            image, 
            strategy=self.strategy, 
            max_regions=max_regions
        )
        
        print(f"✅ 采样得到 {len(regions)} 个区域")
        
        # 添加策略信息到每个区域
        for region in regions:
            region['sampling_strategy'] = self.strategy
        
        return regions
    
    def sample_with_parameters(self, image: np.ndarray, **kwargs) -> List[Dict]:
        """
        使用自定义参数进行采样
        
        参数:
            image: 输入图像
            **kwargs: 采样参数
        
        返回:
            区域列表
        """
        print(f"\n🔍 使用自定义参数进行区域采样...")
        
        if self.strategy == 'multi_threshold_saliency':
            regions = multi_threshold_saliency_sampling(image, **kwargs)
        elif self.strategy == 'layered':
            regions = multi_threshold_layered_sampling(image, **kwargs)
        elif self.strategy == 'pyramid':
            regions = multi_scale_pyramid_sampling(image, **kwargs)
        else:
            regions = sample_regions(image, strategy=self.strategy, **kwargs)
        
        print(f"✅ 采样得到 {len(regions)} 个区域")
        
        # 添加策略信息到每个区域
        for region in regions:
            region['sampling_strategy'] = self.strategy
        
        return regions
    
    def filter_regions_by_priority(self, regions: List[Dict], 
                                 min_priority_score: float = 0.5) -> List[Dict]:
        """
        按优先级过滤区域
        
        参数:
            regions: 区域列表
            min_priority_score: 最小优先级分数
        
        返回:
            过滤后的区域列表
        """
        filtered_regions = []
        
        for region in regions:
            # 计算优先级分数
            score = region.get('score', 0)
            saliency = region.get('saliency', 0)
            area = region.get('area', 0)
            
            # 综合优先级分数
            priority_score = (score + saliency) / 2.0
            
            if priority_score >= min_priority_score:
                region['priority_score'] = priority_score
                filtered_regions.append(region)
        
        print(f"✅ 按优先级过滤: {len(regions)} -> {len(filtered_regions)} 个区域")
        
        return filtered_regions
    
    def apply_nms(self, regions: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        应用非最大抑制
        
        参数:
            regions: 区域列表
            iou_threshold: IoU阈值
        
        返回:
            NMS后的区域列表
        """
        print(f"\n🔄 应用NMS (IoU阈值: {iou_threshold})...")
        
        nms_regions = non_max_suppression_regions(regions, iou_threshold)
        
        print(f"✅ NMS后保留: {len(regions)} -> {len(nms_regions)} 个区域")
        
        return nms_regions
    
    def get_sampling_statistics(self, regions: List[Dict]) -> Dict:
        """
        获取采样统计信息
        
        参数:
            regions: 区域列表
        
        返回:
            统计信息字典
        """
        if not regions:
            return {}
        
        scores = [r.get('score', 0) for r in regions]
        saliencies = [r.get('saliency', 0) for r in regions]
        areas = [r.get('area', 0) for r in regions]
        
        stats = {
            'total_regions': len(regions),
            'score_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'saliency_stats': {
                'mean': np.mean(saliencies),
                'std': np.std(saliencies),
                'min': np.min(saliencies),
                'max': np.max(saliencies)
            },
            'area_stats': {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas)
            }
        }
        
        return stats


def main():
    """测试区域采样器"""
    print("=" * 70)
    print("测试实验区域采样器")
    print("=" * 70)
    
    # 测试图像
    test_image_path = "assets/airport.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    # 加载测试图像
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 测试不同策略
    strategies = ['multi_threshold_saliency', 'layered', 'pyramid']
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"测试策略: {strategy}")
        print(f"{'='*50}")
        
        # 创建采样器
        sampler = ExperimentRegionSampler(strategy)
        
        # 采样区域
        regions = sampler.sample_regions(image, max_regions=30)
        
        # 获取统计信息
        stats = sampler.get_sampling_statistics(regions)
        print(f"\n📊 采样统计:")
        print(f"  总区域数: {stats.get('total_regions', 0)}")
        print(f"  分数范围: {stats.get('score_stats', {}).get('min', 0):.3f} - {stats.get('score_stats', {}).get('max', 0):.3f}")
        print(f"  显著性范围: {stats.get('saliency_stats', {}).get('min', 0):.3f} - {stats.get('saliency_stats', {}).get('max', 0):.3f}")
        
        # 应用NMS
        nms_regions = sampler.apply_nms(regions, iou_threshold=0.5)
        
        # 按优先级过滤
        filtered_regions = sampler.filter_regions_by_priority(regions, min_priority_score=0.3)
        
        print(f"  NMS后: {len(nms_regions)} 个区域")
        print(f"  优先级过滤后: {len(filtered_regions)} 个区域")
    
    print("\n✅ 区域采样器测试完成!")


if __name__ == "__main__":
    main()

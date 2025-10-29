#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2: 边界框微调模块
基于原有bbox_refinement.py，专门用于实验中的边界框优化
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import os
import sys

# 添加父目录到路径以导入原有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bbox_refinement import BBoxRefinement, compute_saliency_map


class ExperimentBBoxRefiner:
    """实验用边界框微调器"""
    
    def __init__(self, model, preprocess, device: str = 'cuda'):
        """
        初始化边界框微调器
        
        参数:
            model: RemoteCLIP模型
            preprocess: 图像预处理函数
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.bbox_refiner = BBoxRefinement(model, preprocess, self.device)
    
    def refine_proposals(self, image: np.ndarray, proposals: List[Dict],
                        saliency_map: np.ndarray = None,
                        refinement_method: str = 'both') -> List[Dict]:
        """
        微调候选框
        
        参数:
            image: 输入图像
            proposals: 候选框列表
            saliency_map: 显著性图
            refinement_method: 微调方法
        
        返回:
            微调后的候选框列表
        """
        print(f"\n🔧 微调 {len(proposals)} 个候选框 (方法: {refinement_method})...")
        
        if saliency_map is None:
            print("  计算显著性图...")
            saliency_map = compute_saliency_map(image)
        
        refined_proposals = []
        
        for i, proposal in enumerate(proposals):
            bbox = proposal['bbox']
            
            # 创建正负样本原型（简化版）
            positive_prototype = torch.randn(1, 512).to(self.device)  # 模拟正样本特征
            negative_prototype = torch.randn(1, 512).to(self.device)  # 模拟负样本特征
            
            try:
                # 执行边界框微调
                refine_result = self.bbox_refiner.refine_bbox_hybrid(
                    image=image,
                    bbox=bbox,
                    saliency_map=saliency_map,
                    positive_prototype=positive_prototype,
                    negative_prototype=negative_prototype,
                    method=refinement_method
                )
                
                # 更新候选框信息
                proposal['refined_bbox'] = refine_result['bbox']
                proposal['refinement_applied'] = refine_result.get('refined', False)
                proposal['refinement_score'] = refine_result.get('composite_score', 0.0)
                proposal['refinement_method'] = refinement_method
                
                if 'scale' in refine_result:
                    proposal['refinement_scale'] = refine_result['scale']
                
                if 'iterations' in refine_result:
                    proposal['refinement_iterations'] = refine_result['iterations']
                
                refined_proposals.append(proposal)
                
            except Exception as e:
                print(f"  警告: 候选框 {i} 微调失败: {e}")
                # 保留原始候选框
                proposal['refined_bbox'] = bbox
                proposal['refinement_applied'] = False
                proposal['refinement_score'] = proposal.get('score', 0.0)
                proposal['refinement_method'] = refinement_method
                refined_proposals.append(proposal)
        
        print(f"✅ 边界框微调完成")
        
        return refined_proposals
    
    def refine_proposals_with_multiple_methods(self, image: np.ndarray, proposals: List[Dict],
                                             methods: List[str] = None) -> Dict[str, List[Dict]]:
        """
        使用多种方法微调候选框
        
        参数:
            image: 输入图像
            proposals: 候选框列表
            methods: 微调方法列表
        
        返回:
            每种方法的微调结果字典
        """
        if methods is None:
            methods = ['position', 'scale', 'both', 'boundary']
        
        print(f"\n🔧 使用多种方法微调候选框: {methods}")
        
        saliency_map = compute_saliency_map(image)
        all_results = {}
        
        for method in methods:
            print(f"\n  使用方法: {method}")
            
            # 为每种方法创建独立的候选框副本
            method_proposals = []
            for proposal in proposals:
                method_proposal = proposal.copy()
                method_proposals.append(method_proposal)
            
            # 执行微调
            refined_proposals = self.refine_proposals(
                image, method_proposals, saliency_map, method
            )
            
            all_results[method] = refined_proposals
        
        return all_results
    
    def evaluate_refinement_quality(self, proposals: List[Dict]) -> Dict:
        """
        评估微调质量
        
        参数:
            proposals: 微调后的候选框列表
        
        返回:
            质量评估结果
        """
        if not proposals:
            return {}
        
        refined_count = sum(1 for p in proposals if p.get('refinement_applied', False))
        total_count = len(proposals)
        
        refinement_scores = [p.get('refinement_score', 0) for p in proposals]
        
        quality_metrics = {
            'total_proposals': total_count,
            'refined_proposals': refined_count,
            'refinement_rate': refined_count / total_count if total_count > 0 else 0,
            'average_refinement_score': np.mean(refinement_scores) if refinement_scores else 0,
            'max_refinement_score': np.max(refinement_scores) if refinement_scores else 0,
            'min_refinement_score': np.min(refinement_scores) if refinement_scores else 0
        }
        
        return quality_metrics
    
    def compare_refinement_methods(self, refinement_results: Dict[str, List[Dict]]) -> Dict:
        """
        比较不同微调方法的效果
        
        参数:
            refinement_results: 不同方法的微调结果
        
        返回:
            方法比较结果
        """
        print(f"\n📊 比较微调方法效果...")
        
        comparison_results = {}
        
        for method, proposals in refinement_results.items():
            quality_metrics = self.evaluate_refinement_quality(proposals)
            comparison_results[method] = quality_metrics
        
        # 找出最佳方法
        best_method = max(comparison_results.keys(), 
                         key=lambda x: comparison_results[x]['average_refinement_score'])
        
        comparison_results['best_method'] = best_method
        
        print(f"✅ 最佳微调方法: {best_method}")
        
        return comparison_results
    
    def get_refinement_statistics(self, proposals: List[Dict]) -> Dict:
        """
        获取微调统计信息
        
        参数:
            proposals: 候选框列表
        
        返回:
            统计信息字典
        """
        if not proposals:
            return {}
        
        # 统计微调方法分布
        method_counts = {}
        for proposal in proposals:
            method = proposal.get('refinement_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # 统计微调应用情况
        applied_count = sum(1 for p in proposals if p.get('refinement_applied', False))
        
        # 统计分数分布
        scores = [p.get('refinement_score', 0) for p in proposals]
        
        stats = {
            'total_proposals': len(proposals),
            'refinement_applied_count': applied_count,
            'refinement_rate': applied_count / len(proposals) if proposals else 0,
            'method_distribution': method_counts,
            'score_stats': {
                'mean': np.mean(scores) if scores else 0,
                'std': np.std(scores) if scores else 0,
                'min': np.min(scores) if scores else 0,
                'max': np.max(scores) if scores else 0
            }
        }
        
        return stats


def main():
    """测试边界框微调器"""
    print("=" * 70)
    print("测试边界框微调器")
    print("=" * 70)
    
    # 测试图像
    test_image_path = "assets/airport.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    # 加载测试图像
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建模拟候选框
    mock_proposals = [
        {
            'proposal_id': 0,
            'bbox': (100, 100, 200, 200),
            'score': 0.8,
            'predicted_class': 'airplane'
        },
        {
            'proposal_id': 1,
            'bbox': (300, 200, 400, 300),
            'score': 0.6,
            'predicted_class': 'building'
        },
        {
            'proposal_id': 2,
            'bbox': (150, 250, 250, 350),
            'score': 0.9,
            'predicted_class': 'vehicle'
        }
    ]
    
    # 创建模拟模型和预处理函数
    class MockModel:
        def encode_image(self, x):
            return torch.randn(x.shape[0], 512)
    
    class MockPreprocess:
        def __call__(self, x):
            return torch.randn(1, 3, 224, 224)
    
    mock_model = MockModel()
    mock_preprocess = MockPreprocess()
    
    # 创建边界框微调器
    refiner = ExperimentBBoxRefiner(mock_model, mock_preprocess)
    
    # 测试单方法微调
    print(f"\n{'='*50}")
    print("测试单方法微调")
    print(f"{'='*50}")
    
    refined_proposals = refiner.refine_proposals(
        image, mock_proposals, refinement_method='both'
    )
    
    # 获取统计信息
    stats = refiner.get_refinement_statistics(refined_proposals)
    print(f"\n📊 微调统计:")
    print(f"  总候选框数: {stats['total_proposals']}")
    print(f"  微调应用数: {stats['refinement_applied_count']}")
    print(f"  微调率: {stats['refinement_rate']:.2%}")
    print(f"  平均分数: {stats['score_stats']['mean']:.3f}")
    
    # 测试多方法微调
    print(f"\n{'='*50}")
    print("测试多方法微调")
    print(f"{'='*50}")
    
    methods = ['position', 'scale', 'both']
    multi_results = refiner.refine_proposals_with_multiple_methods(
        image, mock_proposals, methods
    )
    
    # 比较方法效果
    comparison = refiner.compare_refinement_methods(multi_results)
    print(f"\n📊 方法比较:")
    for method, metrics in comparison.items():
        if method != 'best_method':
            print(f"  {method}: 微调率={metrics['refinement_rate']:.2%}, 平均分数={metrics['average_refinement_score']:.3f}")
    
    print(f"\n🏆 最佳方法: {comparison['best_method']}")
    
    print("\n✅ 边界框微调器测试完成!")


if __name__ == "__main__":
    main()

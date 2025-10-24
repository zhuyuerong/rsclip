#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2: 目标检测模块
基于原有target_detection.py，专门用于实验中的目标检测
"""

import torch
import open_clip
from PIL import Image
import numpy as np
import cv2
import argparse
from typing import List, Dict
import os
import sys

# 添加父目录到路径以导入原有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sampling import sample_regions
from wordnet_vocabulary import get_expansion_words, get_synonyms, WORDNET_REMOTE_SENSING_CLASSES


class ExperimentTargetDetector:
    """实验用目标检测器"""
    
    def __init__(self, model_name: str = 'RN50', device: str = 'cuda'):
        """
        初始化目标检测器
        
        参数:
            model_name: RemoteCLIP模型名称
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
        # 初始化模型
        self._load_model()
    
    def _load_model(self):
        """加载RemoteCLIP模型"""
        print(f"🔄 加载RemoteCLIP模型: {self.model_name}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        
        checkpoint_path = f"checkpoints/RemoteCLIP-{self.model_name}.pt"
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device).eval()
        
        print(f"✅ 模型已加载到 {self.device}")
    
    def detect_target_with_contrastive_learning(self, image: np.ndarray, target_class: str,
                                              strategy: str = 'multi_threshold_saliency',
                                              max_regions: int = 50) -> List[Dict]:
        """
        使用对比学习检测目标类别
        
        参数:
            image: 输入图像
            target_class: 目标类别
            strategy: 采样策略
            max_regions: 最大区域数
        
        返回:
            检测结果列表
        """
        print(f"\n🎯 目标检测: {target_class}")
        
        # 1. 定义类别体系
        target_synonyms = get_synonyms(target_class)
        if target_synonyms:
            positive_classes = [target_class] + target_synonyms[:4]
        else:
            positive_classes = [target_class]
        
        negative_classes = [c for c in WORDNET_REMOTE_SENSING_CLASSES 
                          if c not in positive_classes]
        
        all_classes = positive_classes + negative_classes
        n_positive = len(positive_classes)
        
        print(f"📋 类别设置: 正样本{n_positive}个, 负样本{len(negative_classes)}个")
        
        # 2. 预编码文本特征
        text = self.tokenizer(all_classes)
        with torch.no_grad():
            text_features = self.model.encode_text(text.to(self.device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        positive_text_features = text_features[:n_positive]
        negative_text_features = text_features[n_positive:]
        
        # 3. 区域采样
        regions = sample_regions(image, strategy=strategy, max_regions=max_regions)
        print(f"✅ 提取到 {len(regions)} 个候选区域")
        
        # 4. 对比学习检测
        results = []
        
        for idx, region in enumerate(regions):
            x1, y1, x2, y2 = region['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            # 裁剪
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            
            crop_pil = Image.fromarray(crop)
            crop_tensor = self.preprocess(crop_pil).unsqueeze(0)
            
            # 编码
            with torch.no_grad():
                image_features = self.model.encode_image(crop_tensor.to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # 计算相似度
                similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
                
                # 正负样本得分
                positive_scores = similarities[:n_positive]
                positive_avg = positive_scores.mean()
                positive_max = positive_scores.max()
                best_positive_idx = positive_scores.argmax()
                
                negative_scores = similarities[n_positive:]
                negative_avg = negative_scores.mean()
                negative_max = negative_scores.max()
                
                # 对比得分
                base_contrast = positive_avg - negative_avg
                background_penalty = negative_max * 0.5
                contrast_score = base_contrast - background_penalty
                
                # 判断条件
                is_target = (positive_avg > negative_avg) and (positive_max > negative_max)
            
            if is_target:
                results.append({
                    'bbox': (x1, y1, x2, y2),
                    'region_idx': idx,
                    'target_class': positive_classes[best_positive_idx],
                    'target_score': float(positive_max),
                    'positive_avg': float(positive_avg),
                    'negative_avg': float(negative_avg),
                    'contrast_score': float(contrast_score),
                    'saliency': region.get('saliency', 0),
                    'priority': region.get('priority', 'N/A')
                })
        
        # 按对比得分排序
        results.sort(key=lambda x: x['contrast_score'], reverse=True)
        
        print(f"✅ 找到 {len(results)} 个{target_class}候选框")
        
        return results
    
    def detect_multiple_targets(self, image: np.ndarray, target_classes: List[str],
                              strategy: str = 'multi_threshold_saliency',
                              max_regions: int = 50) -> Dict[str, List[Dict]]:
        """
        检测多个目标类别
        
        参数:
            image: 输入图像
            target_classes: 目标类别列表
            strategy: 采样策略
            max_regions: 最大区域数
        
        返回:
            每个类别的检测结果字典
        """
        print(f"\n🎯 多目标检测: {target_classes}")
        
        all_results = {}
        
        for target_class in target_classes:
            results = self.detect_target_with_contrastive_learning(
                image, target_class, strategy, max_regions
            )
            all_results[target_class] = results
        
        return all_results
    
    def get_detection_statistics(self, results: List[Dict]) -> Dict:
        """
        获取检测统计信息
        
        参数:
            results: 检测结果列表
        
        返回:
            统计信息字典
        """
        if not results:
            return {}
        
        contrast_scores = [r['contrast_score'] for r in results]
        target_scores = [r['target_score'] for r in results]
        
        stats = {
            'total_detections': len(results),
            'contrast_score_stats': {
                'mean': np.mean(contrast_scores),
                'std': np.std(contrast_scores),
                'min': np.min(contrast_scores),
                'max': np.max(contrast_scores)
            },
            'target_score_stats': {
                'mean': np.mean(target_scores),
                'std': np.std(target_scores),
                'min': np.min(target_scores),
                'max': np.max(target_scores)
            }
        }
        
        return stats


def main():
    """测试目标检测器"""
    print("=" * 70)
    print("测试实验目标检测器")
    print("=" * 70)
    
    # 测试图像
    test_image_path = "assets/airport.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    # 加载测试图像
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建目标检测器
    detector = ExperimentTargetDetector()
    
    # 测试单目标检测
    target_class = "airplane"
    results = detector.detect_target_with_contrastive_learning(
        image, target_class, max_regions=30
    )
    
    # 获取统计信息
    stats = detector.get_detection_statistics(results)
    print(f"\n📊 检测统计:")
    print(f"  总检测数: {stats.get('total_detections', 0)}")
    print(f"  对比分数范围: {stats.get('contrast_score_stats', {}).get('min', 0):.3f} - {stats.get('contrast_score_stats', {}).get('max', 0):.3f}")
    
    # 显示检测结果
    print(f"\n📋 检测结果:")
    for i, result in enumerate(results[:5]):  # 显示前5个
        print(f"  检测 {i+1}:")
        print(f"    位置: {result['bbox']}")
        print(f"    类别: {result['target_class']}")
        print(f"    对比分数: {result['contrast_score']:.3f}")
    
    # 测试多目标检测
    print(f"\n{'='*50}")
    print("测试多目标检测")
    print(f"{'='*50}")
    
    target_classes = ["airplane", "building", "runway"]
    all_results = detector.detect_multiple_targets(
        image, target_classes, max_regions=30
    )
    
    for target_class, class_results in all_results.items():
        print(f"\n{target_class}: {len(class_results)} 个检测结果")
    
    print("\n✅ 目标检测器测试完成!")


if __name__ == "__main__":
    main()

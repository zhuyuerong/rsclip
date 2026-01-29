# -*- coding: utf-8 -*-
"""
CAL操作模块：特征空间和相似度空间
"""
import torch
from typing import List
from .cal_config import CALConfig


class CALFeatureSpace:
    """Q3-特征空间CAL操作"""
    
    def __init__(self, config: CALConfig):
        self.config = config
    
    def apply(self, 
              image_features: torch.Tensor,
              text_features: torch.Tensor,
              negative_features: torch.Tensor) -> torch.Tensor:
        """
        在特征空间应用CAL减法
        
        Args:
            image_features: [B, N, D] 图像特征
            text_features: [num_texts, D] 文本特征
            negative_features: [num_neg, D] 负样本特征
        
        Returns:
            similarity: [B, N, num_texts] 相似度
        """
        if self.config.verbose:
            print(f"  [CAL-FeatureSpace] 特征空间操作")
            print(f"    image_features: {image_features.shape}")
            print(f"    text_features: {text_features.shape}")
            print(f"    negative_features: {negative_features.shape}")
        
        # 计算正样本相似度
        similarity_pos = image_features @ text_features.t()  # [B, N, num_texts]
        
        # 计算负样本相似度
        negative_features_mean = negative_features.mean(dim=0, keepdim=True)  # [1, D]
        similarity_neg = image_features @ negative_features_mean.t()  # [B, N, 1]
        
        # Q2: 加权减法
        if self.config.use_weighted_subtraction:
            similarity = similarity_pos - self.config.alpha * similarity_neg
            if self.config.verbose:
                print(f"    加权减法: alpha={self.config.alpha}")
        else:
            similarity = similarity_pos - similarity_neg
        
        if self.config.verbose:
            print(f"    similarity_pos: min={similarity_pos.min():.6f}, max={similarity_pos.max():.6f}")
            print(f"    similarity_neg: min={similarity_neg.min():.6f}, max={similarity_neg.max():.6f}")
            print(f"    similarity_final: min={similarity.min():.6f}, max={similarity.max():.6f}")
        
        return similarity


class CALSimilaritySpace:
    """Q3-相似度空间CAL操作"""
    
    def __init__(self, config: CALConfig):
        self.config = config
    
    def apply(self,
              similarity_maps: torch.Tensor,
              image_features_patches: torch.Tensor,
              negative_features: torch.Tensor) -> torch.Tensor:
        """
        在相似度空间应用CAL减法
        
        Args:
            similarity_maps: [B, N, num_texts] 已有的相似度图
            image_features_patches: [B, N, D] patch特征
            negative_features: [num_neg, D] 负样本特征
        
        Returns:
            similarity_maps_cal: [B, N, num_texts] CAL后的相似度
        """
        if self.config.verbose:
            print(f"  [CAL-SimilaritySpace] 相似度空间操作")
            print(f"    similarity_maps: {similarity_maps.shape}")
            print(f"    image_features_patches: {image_features_patches.shape}")
            print(f"    negative_features: {negative_features.shape}")
        
        # 计算负样本相似度
        similarity_neg = image_features_patches @ negative_features.t()  # [B, N, num_neg]
        similarity_neg_mean = similarity_neg.mean(dim=-1, keepdim=True)  # [B, N, 1]
        
        # Q2: 加权减法
        if self.config.use_weighted_subtraction:
            similarity_maps_cal = similarity_maps - self.config.alpha * similarity_neg_mean
            if self.config.verbose:
                print(f"    加权减法: alpha={self.config.alpha}")
        else:
            similarity_maps_cal = similarity_maps - similarity_neg_mean
        
        if self.config.verbose:
            print(f"    similarity_maps原始: min={similarity_maps.min():.6f}, max={similarity_maps.max():.6f}")
            print(f"    similarity_neg_mean: min={similarity_neg_mean.min():.6f}, max={similarity_neg_mean.max():.6f}")
            print(f"    similarity_maps_cal: min={similarity_maps_cal.min():.6f}, max={similarity_maps_cal.max():.6f}")
        
        return similarity_maps_cal


class ExperimentTracker:
    """实验追踪器"""
    
    def __init__(self, output_dir: str = "outputs/cal_experiments"):
        import os
        self.output_dir = output_dir
        self.experiments = []
        os.makedirs(output_dir, exist_ok=True)
    
    def log_experiment(self, config: CALConfig, results: dict):
        """记录实验"""
        import json
        import os
        from datetime import datetime
        
        experiment_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': config.get_experiment_id(),
            'config': config.to_dict(),
            'results': results
        }
        
        self.experiments.append(experiment_data)
        
        # 保存到文件
        log_file = os.path.join(self.output_dir, f"{config.get_experiment_id()}.json")
        with open(log_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        if config.verbose:
            print(f"✅ 实验记录已保存: {log_file}")













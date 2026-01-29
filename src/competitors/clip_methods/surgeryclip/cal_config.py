# -*- coding: utf-8 -*-
"""
CAL (Counterfactual Attention Learning) 配置模块
可插拔设计，通过配置控制CAL功能
"""
from dataclasses import dataclass
from typing import List, Optional, Literal
import random


@dataclass
class CALConfig:
    """CAL实验配置"""
    
    # ============ 总开关 ============
    enable_cal: bool = False  # 是否启用CAL（False时完全不影响原有逻辑）
    
    # ============ Q1: 负样本策略 ============
    negative_mode: Literal['fixed', 'dynamic', 'random', 'combined'] = 'fixed'
    # - 'fixed': 固定负样本 ["background", "irrelevant"]
    # - 'dynamic': 动态负样本（其他类别）
    # - 'random': 随机负样本（随机文本）
    # - 'combined': 组合负样本（固定+动态）
    
    fixed_negatives: List[str] = None  # 固定负样本列表
    num_dynamic_negatives: int = 3  # 动态负样本数量
    num_random_negatives: int = 3  # 随机负样本数量
    random_seed: int = 42  # 随机种子
    
    # ============ Q2: 加权减法 ============
    use_weighted_subtraction: bool = True  # 是否使用加权减法
    alpha: float = 1.0  # 减法权重: similarity_pos - alpha * similarity_neg
    # alpha=1.0: 直接减法
    # alpha=0.5: 减半
    # alpha=2.0: 加倍
    
    # ============ Q3: 操作位置 ============
    cal_space: Literal['feature', 'similarity', 'both'] = 'similarity'
    # - 'feature': 特征空间操作（在clip_feature_surgery中）
    # - 'similarity': 相似度空间操作（在generate_heatmap中）
    # - 'both': 双重操作（特征空间 + 相似度空间）
    
    # ============ 实验追踪 ============
    experiment_name: str = "cal_baseline"  # 实验名称
    verbose: bool = True  # 是否打印调试信息
    
    def __post_init__(self):
        """初始化默认值"""
        if self.fixed_negatives is None:
            self.fixed_negatives = [
                "background",
                "irrelevant objects",
                "other things"
            ]
    
    def get_experiment_id(self) -> str:
        """生成实验唯一ID"""
        return f"{self.experiment_name}_neg{self.negative_mode}_alpha{self.alpha}_space{self.cal_space}"
    
    def to_dict(self) -> dict:
        """转换为字典（用于保存配置）"""
        return {
            'enable_cal': self.enable_cal,
            'negative_mode': self.negative_mode,
            'fixed_negatives': self.fixed_negatives,
            'num_dynamic_negatives': self.num_dynamic_negatives,
            'num_random_negatives': self.num_random_negatives,
            'alpha': self.alpha,
            'cal_space': self.cal_space,
            'experiment_name': self.experiment_name
        }


class NegativeSampleGenerator:
    """负样本生成器 - Q1的4种策略"""
    
    def __init__(self, config: CALConfig, all_classes: List[str]):
        self.config = config
        self.all_classes = all_classes
        random.seed(config.random_seed)
    
    def generate(self, target_classes: List[str]) -> List[str]:
        """
        根据配置生成负样本
        
        Args:
            target_classes: 目标类别列表
        
        Returns:
            负样本文本列表
        """
        if self.config.negative_mode == 'fixed':
            return self._generate_fixed()
        
        elif self.config.negative_mode == 'dynamic':
            return self._generate_dynamic(target_classes)
        
        elif self.config.negative_mode == 'random':
            return self._generate_random()
        
        elif self.config.negative_mode == 'combined':
            return self._generate_combined(target_classes)
        
        else:
            raise ValueError(f"未知的negative_mode: {self.config.negative_mode}")
    
    def _generate_fixed(self) -> List[str]:
        """Q1-策略1: 固定负样本"""
        return self.config.fixed_negatives
    
    def _generate_dynamic(self, target_classes: List[str]) -> List[str]:
        """Q1-策略2: 动态负样本（其他类别）"""
        other_classes = [c for c in self.all_classes if c not in target_classes]
        
        # 随机选择N个其他类别
        num_samples = min(self.config.num_dynamic_negatives, len(other_classes))
        if num_samples == 0:
            return self.config.fixed_negatives  # 回退到固定负样本
        
        selected = random.sample(other_classes, num_samples)
        return selected
    
    def _generate_random(self) -> List[str]:
        """Q1-策略3: 随机负样本"""
        random_templates = [
            "random object {}",
            "unrelated thing {}",
            "noise pattern {}",
            "arbitrary texture {}",
            "meaningless shape {}"
        ]
        
        negatives = []
        for i in range(self.config.num_random_negatives):
            template = random.choice(random_templates)
            negatives.append(template.format(i))
        
        return negatives
    
    def _generate_combined(self, target_classes: List[str]) -> List[str]:
        """Q1-策略4: 组合负样本（固定+动态）"""
        fixed = self._generate_fixed()
        dynamic = self._generate_dynamic(target_classes)
        
        return fixed + dynamic













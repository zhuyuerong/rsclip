#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型架构配置
定义各个模块的详细配置
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """模型架构配置"""
    
    # ==================== Encoder 配置 ====================
    @dataclass
    class EncoderConfig:
        """编码器配置"""
        backbone: str = "RN50"          # CLIP 主干网络
        pretrained: bool = True         # 是否使用预训练权重
        freeze: bool = True             # 是否冻结参数
        
        # 多尺度特征提取
        extract_layers: List[int] = None  # 提取的层索引
        feature_dims: List[int] = None    # 各层特征维度
        
        def __post_init__(self):
            if self.extract_layers is None:
                # 默认提取 layer2, layer3, layer4, layer5
                self.extract_layers = [2, 3, 4, 5]
            if self.feature_dims is None:
                # RN50 的特征维度
                self.feature_dims = [512, 1024, 2048, 2048]
    
    # ==================== Decoder 配置 ====================
    @dataclass
    class DecoderConfig:
        """解码器配置"""
        num_layers: int = 6             # 解码器层数 L
        d_model: int = 256              # 模型维度
        num_heads: int = 8              # 注意力头数
        d_ffn: int = 1024               # FFN 维度
        dropout: float = 0.1            # Dropout
        activation: str = "relu"        # 激活函数
        
        # 可变形注意力
        num_feature_levels: int = 4     # 特征层数
        num_sampling_points: int = 4    # 采样点数
        
        # 文本调制
        use_text_condition: bool = True # 是否使用文本调制
        text_condition_method: str = "add"  # "add" 或 "concat"
        
        # 上下文门控（核心）
        use_context_gating: bool = True # 是否使用上下文门控
        context_gating_type: str = "film"  # "film" 或 "concat_mlp"
        context_gating_hidden_dim: int = 512  # 门控隐藏层维度
    
    # ==================== Prediction Head 配置 ====================
    @dataclass
    class PredictionConfig:
        """预测头配置"""
        d_model: int = 256              # 输入维度
        d_clip: int = 512               # CLIP 空间维度
        
        # 分类头
        use_classification_head: bool = True
        classification_hidden_dim: int = 512
        
        # 回归头
        use_regression_head: bool = True
        regression_hidden_dim: int = 256
        num_bbox_params: int = 4        # 边界框参数数量（x, y, w, h）
        
        # 局部隐式细化
        use_implicit_refiner: bool = False
        refiner_hidden_dim: int = 128
    
    # ==================== Loss 配置 ====================
    @dataclass
    class LossConfig:
        """损失函数配置"""
        # 边界框损失
        use_box_loss: bool = True
        box_loss_type: str = "l1_giou"  # "l1", "giou", "l1_giou"
        lambda_l1: float = 5.0
        lambda_giou: float = 2.0
        
        # 全局对比损失（核心）
        use_global_contrast_loss: bool = True
        lambda_global_contrast: float = 1.0
        temperature: float = 0.07       # τ
        
        # 位置-文本对比损失（可选）
        use_position_text_loss: bool = False
        lambda_position_text: float = 0.5
        
        # 匹配器
        matcher_type: str = "hungarian"  # 匈牙利匹配
        matcher_cost_class: float = 1.0
        matcher_cost_bbox: float = 5.0
        matcher_cost_giou: float = 2.0
    
    # 创建子配置实例
    encoder: EncoderConfig = None
    decoder: DecoderConfig = None
    prediction: PredictionConfig = None
    loss: LossConfig = None
    
    def __post_init__(self):
        """初始化子配置"""
        if self.encoder is None:
            self.encoder = self.EncoderConfig()
        if self.decoder is None:
            self.decoder = self.DecoderConfig()
        if self.prediction is None:
            self.prediction = self.PredictionConfig()
        if self.loss is None:
            self.loss = self.LossConfig()
    
    def __str__(self):
        """打印配置"""
        lines = ["=" * 70, "Model Configuration", "=" * 70]
        
        lines.append("\nEncoder Configuration:")
        for key, value in self.encoder.__dict__.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("\nDecoder Configuration:")
        for key, value in self.decoder.__dict__.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("\nPrediction Configuration:")
        for key, value in self.prediction.__dict__.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("\nLoss Configuration:")
        for key, value in self.loss.__dict__.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


# 创建默认模型配置
default_model_config = ModelConfig()


if __name__ == "__main__":
    # 测试配置
    config = ModelConfig()
    print(config)


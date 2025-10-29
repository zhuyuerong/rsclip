#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
默认配置文件
包含所有超参数、路径和训练配置
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DefaultConfig:
    """默认配置类"""
    
    # ==================== 模型架构 ====================
    # 查询配置
    num_queries: int = 100              # M，查询数量
    num_decoder_layers: int = 6         # L，解码器层数
    d_model: int = 256                  # 模型维度
    d_clip: int = 512                   # CLIP 空间维度
    d_ffn: int = 1024                   # FFN 维度
    num_heads: int = 8                  # 注意力头数
    dropout: float = 0.1                # Dropout 比例
    
    # 可变形注意力配置
    num_feature_levels: int = 4         # 特征金字塔层数
    num_sampling_points: int = 4        # 采样点数
    
    # 上下文门控配置
    context_gating_type: str = "film"   # "film" 或 "concat_mlp"
    
    # ==================== 损失函数 ====================
    # 损失权重
    lambda_box_l1: float = 5.0          # L1 边界框损失权重
    lambda_box_giou: float = 2.0        # GIoU 损失权重
    lambda_global_contrast: float = 1.0 # 全局对比损失权重（核心）
    lambda_position_text: float = 0.5   # 位置-文本对比损失权重（可选）
    
    # 对比学习配置
    temperature: float = 0.07           # τ，对比学习温度
    
    # 匹配器配置
    matcher_cost_class: float = 1.0     # 分类代价权重
    matcher_cost_bbox: float = 5.0      # 边界框代价权重
    matcher_cost_giou: float = 2.0      # GIoU 代价权重
    
    # ==================== 训练配置 ====================
    # 优化器
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 0.1
    
    # 学习率调度
    lr_scheduler: str = "step"          # "step" 或 "cosine"
    lr_drop_epochs: List[int] = field(default_factory=lambda: [40])
    lr_drop_rate: float = 0.1
    
    # 训练参数
    num_epochs: int = 50
    warmup_epochs: int = 5
    batch_size: int = 16
    num_workers: int = 4
    
    # 混合精度训练
    use_amp: bool = True                # 自动混合精度
    
    # ==================== 数据配置 ====================
    # 数据路径
    data_root: str = "datasets/"
    train_annotations: str = "annotations/train.json"
    val_annotations: str = "annotations/val.json"
    test_annotations: str = "annotations/test.json"
    
    # 图像配置
    image_size: Tuple[int, int] = (800, 800)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # 数据增强
    use_augmentation: bool = True
    random_flip: float = 0.5
    random_crop: bool = True
    color_jitter: bool = True
    
    # ==================== CLIP 配置 ====================
    clip_model_name: str = "RN50"       # "RN50", "ViT-B-32", "ViT-L-14"
    clip_checkpoint: str = "../checkpoints/RemoteCLIP-RN50.pt"
    freeze_clip_backbone: bool = True   # 是否冻结 CLIP 主干
    
    # ==================== 推理配置 ====================
    # 后处理
    score_threshold: float = 0.5        # 分数阈值
    nms_threshold: float = 0.7          # NMS 阈值
    max_detections: int = 100           # 最大检测数
    
    # ==================== 日志和保存 ====================
    # 输出路径
    output_dir: str = "experiment2/outputs"
    checkpoint_dir: str = "experiment2/outputs/checkpoints"
    log_dir: str = "experiment2/outputs/logs"
    vis_dir: str = "experiment2/outputs/visualizations"
    
    # 日志
    log_interval: int = 10              # 打印日志间隔（steps）
    save_interval: int = 5              # 保存 checkpoint 间隔（epochs）
    eval_interval: int = 5              # 验证间隔（epochs）
    
    # TensorBoard
    use_tensorboard: bool = True
    
    # ==================== 分布式训练 ====================
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    gpu: int = 0
    dist_backend: str = "nccl"
    dist_url: str = "env://"
    
    # ==================== 其他 ====================
    seed: int = 42                      # 随机种子
    device: str = "cuda"                # "cuda" 或 "cpu"
    
    def __post_init__(self):
        """初始化后处理"""
        import os
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def to_dict(self):
        """转换为字典"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def __str__(self):
        """打印配置"""
        lines = ["=" * 70, "Configuration", "=" * 70]
        
        sections = {
            "Model Architecture": [
                "num_queries", "num_decoder_layers", "d_model", "d_clip",
                "context_gating_type"
            ],
            "Loss Functions": [
                "lambda_box_l1", "lambda_box_giou", "lambda_global_contrast",
                "lambda_position_text", "temperature"
            ],
            "Training": [
                "num_epochs", "batch_size", "learning_rate", "warmup_epochs"
            ],
            "Data": [
                "image_size", "use_augmentation"
            ],
            "CLIP": [
                "clip_model_name", "freeze_clip_backbone"
            ],
            "Inference": [
                "score_threshold", "nms_threshold", "max_detections"
            ]
        }
        
        for section, keys in sections.items():
            lines.append(f"\n{section}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    lines.append(f"  {key}: {value}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


# 创建默认配置实例
default_config = DefaultConfig()


if __name__ == "__main__":
    # 测试配置
    config = DefaultConfig()
    print(config)
    print(f"\n配置字典:\n{config.to_dict()}")


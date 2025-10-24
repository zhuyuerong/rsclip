#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment3 默认配置
OVA-DETR with RemoteCLIP
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DefaultConfig:
    """默认配置 - OVA-DETR with RemoteCLIP"""
    
    # ==================== 模型架构 ====================
    # 基础配置
    num_queries: int = 300              # 查询数量
    num_decoder_layers: int = 6         # 解码器层数
    d_model: int = 256                  # 模型维度
    num_heads: int = 8                  # 注意力头数
    dim_feedforward: int = 2048         # FFN维度
    dropout: float = 0.1
    
    # RemoteCLIP配置
    remoteclip_model: str = 'RN50'      # RemoteCLIP模型
    remoteclip_checkpoint: str = 'checkpoints/RemoteCLIP-RN50.pt'
    freeze_remoteclip: bool = True      # 冻结RemoteCLIP
    
    # 文本特征配置
    txt_dim: int = 1024                 # 文本特征维度
    vision_aug_text: bool = True        # 视觉增强文本
    multi_level_fusion: bool = True     # 多层级融合
    
    # ==================== 损失函数 ====================
    # 分类损失（变焦损失）
    loss_cls_weight: float = 1.0
    varifocal_alpha: float = 0.75
    varifocal_gamma: float = 2.0
    iou_weighted: bool = True
    
    # 回归损失
    loss_bbox_weight: float = 5.0
    loss_giou_weight: float = 2.0
    
    # ==================== 训练配置 ====================
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # 文本采样
    num_neg_samples: int = 48           # 负样本文本数量
    max_num_samples: int = 48           # 最大文本样本数
    
    # ==================== 数据配置 ====================
    image_size: tuple = (800, 800)
    num_classes: int = 20               # DIOR数据集类别数
    
    # ==================== 推理配置 ====================
    score_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 100
    
    def __str__(self):
        return f"""
Experiment3 配置 (OVA-DETR with RemoteCLIP)
{'='*50}
模型:
  - RemoteCLIP: {self.remoteclip_model}
  - 查询数: {self.num_queries}
  - 文本维度: {self.txt_dim}

损失:
  - 分类权重: {self.loss_cls_weight}
  - 回归权重: {self.loss_bbox_weight}

训练:
  - Batch size: {self.batch_size}
  - Learning rate: {self.learning_rate}
  - Epochs: {self.num_epochs}
{'='*50}
"""


# 默认配置实例
default_config = DefaultConfig()


if __name__ == "__main__":
    config = DefaultConfig()
    print(config)


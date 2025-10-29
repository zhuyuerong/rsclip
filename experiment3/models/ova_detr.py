#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVA-DETR模型

整合所有组件：
1. RemoteCLIP骨干网络
2. FPN特征金字塔
3. 混合编码器
4. 文本-视觉融合
5. Transformer解码器
6. 检测头
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import sys
sys.path.append('..')

from backbone.remoteclip_backbone import RemoteCLIPBackbone
from encoder.fpn import FPN
from encoder.hybrid_encoder import HybridEncoder
from encoder.text_vision_fusion import TextVisionFusion
from decoder.transformer_decoder import TransformerDecoder
from decoder.query_generator import QueryGenerator
from head.classification_head import ContrastiveClassificationHead
from head.regression_head import BBoxRegressionHead, MultiscaleBBoxRegressionHead


class OVADETR(nn.Module):
    """
    OVA-DETR: Open-Vocabulary Object Detection with RemoteCLIP
    
    架构:
    1. RemoteCLIP Backbone (冻结)
    2. FPN
    3. Hybrid Encoder
    4. Text-Vision Fusion
    5. Transformer Decoder
    6. Detection Heads
    """
    
    def __init__(self, config):
        """
        参数:
            config: 配置对象
        """
        super().__init__()
        
        self.config = config
        
        # ==================== 骨干网络 ====================
        print("=" * 70)
        print("初始化OVA-DETR模型")
        print("=" * 70)
        
        self.backbone = RemoteCLIPBackbone(
            model_name=config.remoteclip_model,
            pretrained_path=config.remoteclip_checkpoint,
            freeze_backbone=config.freeze_remoteclip
        )
        
        # 根据模型类型设置特征通道
        if 'RN50' in config.remoteclip_model:
            # ResNet-50
            backbone_channels = [512, 1024, 2048]
        elif 'ViT-B' in config.remoteclip_model:
            # ViT-B
            backbone_channels = [768, 768, 768]
        elif 'ViT-L' in config.remoteclip_model:
            # ViT-L
            backbone_channels = [1024, 1024, 1024]
        else:
            backbone_channels = [512, 1024, 2048]
        
        # ==================== FPN ====================
        self.fpn = FPN(
            in_channels=backbone_channels,
            out_channels=config.d_model,
            num_outs=4
        )
        
        # ==================== 混合编码器 ====================
        self.encoder = HybridEncoder(
            d_model=config.d_model,
            num_encoder_layers=6,
            num_heads=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_feature_levels=4
        )
        
        # ==================== 文本-视觉融合 ====================
        self.text_vision_fusion = TextVisionFusion(
            text_dim=config.txt_dim,
            vision_dim=config.d_model,
            output_dim=config.d_model,
            num_levels=4,
            enable_vat=config.vision_aug_text
        )
        
        # ==================== 查询生成器 ====================
        self.query_generator = QueryGenerator(
            num_queries=config.num_queries,
            d_model=config.d_model,
            separate_pos_content=True
        )
        
        # ==================== Transformer解码器 ====================
        self.decoder = TransformerDecoder(
            num_layers=config.num_decoder_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            text_guided=True,
            return_intermediate=True
        )
        
        # ==================== 检测头 ====================
        # 分类头（对比学习）
        self.cls_head = ContrastiveClassificationHead(
            d_model=config.d_model,
            temperature=0.07,
            normalize=True
        )
        
        # 回归头（多尺度）
        self.bbox_head = MultiscaleBBoxRegressionHead(
            num_decoder_layers=config.num_decoder_layers,
            d_model=config.d_model,
            num_layers=3,
            hidden_dim=config.d_model
        )
        
        print(f"✅ OVA-DETR初始化完成")
        print(f"   - 骨干网络: {config.remoteclip_model}")
        print(f"   - 查询数量: {config.num_queries}")
        print(f"   - 解码器层数: {config.num_decoder_layers}")
        print(f"   - 模型维度: {config.d_model}")
        print("=" * 70)
    
    def forward(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            images: (B, 3, H, W)
            text_features: (B, num_classes, txt_dim) 或 (num_classes, txt_dim)
            targets: 训练时的目标标注
        
        返回:
            outputs: {
                'pred_logits': (num_layers, B, num_queries, num_classes),
                'pred_boxes': (num_layers, B, num_queries, 4),
                'text_features': (B, num_classes, d_model)
            }
        """
        batch_size = images.shape[0]
        
        # 处理文本特征维度
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # ==================== 图像特征提取 ====================
        # RemoteCLIP backbone
        img_features = self.backbone.forward_image(images)  # List[(B, C, H, W)]
        
        # FPN
        fpn_features = self.fpn(img_features)  # List[(B, d_model, H, W)]
        
        # 混合编码器
        encoded_features = self.encoder(fpn_features)  # List[(B, d_model, H, W)]
        
        # ==================== 文本-视觉融合 ====================
        enhanced_text, enhanced_vision = self.text_vision_fusion(
            text_features, encoded_features
        )  # (B, num_classes, d_model), List[(B, d_model, H, W)]
        
        # ==================== 准备解码器输入 ====================
        # 展平视觉特征
        vision_flatten = []
        for feat in enhanced_vision:
            B, C, H, W = feat.shape
            vision_flatten.append(feat.flatten(2).transpose(1, 2))
        
        # 连接所有层级
        memory = torch.cat(vision_flatten, dim=1)  # (B, N_total, d_model)
        
        # 生成查询
        query_content, query_pos = self.query_generator(batch_size)
        
        # ==================== Transformer解码 ====================
        decoder_output = self.decoder(
            tgt=query_content,
            memory=memory,
            tgt_pos=query_pos,
            memory_pos=None,
            text_features=enhanced_text
        )  # (num_layers, B, num_queries, d_model)
        
        # ==================== 预测头 ====================
        # 分类
        pred_logits = self.cls_head(decoder_output, enhanced_text)
        # (num_layers, B, num_queries, num_classes)
        
        # 回归
        pred_boxes = self.bbox_head(decoder_output)
        # (num_layers, B, num_queries, 4)
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'text_features': enhanced_text
        }
        
        return outputs
    
    def inference(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        max_detections: int = 100
    ) -> List[Dict[str, torch.Tensor]]:
        """
        推理模式
        
        参数:
            images: (B, 3, H, W)
            text_features: (num_classes, txt_dim)
            score_threshold: 分数阈值
            nms_threshold: NMS阈值
            max_detections: 最大检测数
        
        返回:
            results: List of {
                'boxes': (N, 4),
                'scores': (N,),
                'labels': (N,)
            }
        """
        # 前向传播
        outputs = self.forward(images, text_features)
        
        # 使用最后一层的输出
        pred_logits = outputs['pred_logits'][-1]  # (B, num_queries, num_classes)
        pred_boxes = outputs['pred_boxes'][-1]    # (B, num_queries, 4)
        
        # 后处理
        results = []
        for i in range(pred_logits.shape[0]):
            # 计算分数
            scores = pred_logits[i].sigmoid()  # (num_queries, num_classes)
            max_scores, labels = scores.max(dim=-1)  # (num_queries,)
            
            # 过滤低分数
            keep = max_scores > score_threshold
            boxes = pred_boxes[i][keep]
            scores_keep = max_scores[keep]
            labels_keep = labels[keep]
            
            # NMS (简化版)
            if len(boxes) > 0:
                # 按分数排序
                sorted_indices = scores_keep.argsort(descending=True)
                boxes = boxes[sorted_indices]
                scores_keep = scores_keep[sorted_indices]
                labels_keep = labels_keep[sorted_indices]
                
                # 限制数量
                if len(boxes) > max_detections:
                    boxes = boxes[:max_detections]
                    scores_keep = scores_keep[:max_detections]
                    labels_keep = labels_keep[:max_detections]
            
            results.append({
                'boxes': boxes,
                'scores': scores_keep,
                'labels': labels_keep
            })
        
        return results


if __name__ == "__main__":
    import sys
    sys.path.append('/home/ubuntu22/Projects/RemoteCLIP-main/experiment3')
    from config.default_config import DefaultConfig
    
    print("=" * 70)
    print("测试OVA-DETR模型")
    print("=" * 70)
    
    # 配置
    config = DefaultConfig()
    
    # 创建模型
    model = OVADETR(config)
    model = model.cuda().eval()
    
    # 测试数据
    batch_size = 2
    num_classes = 20
    
    images = torch.randn(batch_size, 3, 800, 800).cuda()
    text_features = torch.randn(num_classes, 1024).cuda()
    
    # 前向传播
    with torch.no_grad():
        outputs = model(images, text_features)
    
    print(f"\n输入:")
    print(f"  图像: {images.shape}")
    print(f"  文本特征: {text_features.shape}")
    
    print(f"\n输出:")
    print(f"  分类logits: {outputs['pred_logits'].shape}")
    print(f"  边界框: {outputs['pred_boxes'].shape}")
    
    # 测试推理
    results = model.inference(images, text_features)
    print(f"\n推理结果:")
    for i, result in enumerate(results):
        print(f"  图像{i}: 检测到{len(result['boxes'])}个目标")
    
    print("\n✅ OVA-DETR模型测试完成！")


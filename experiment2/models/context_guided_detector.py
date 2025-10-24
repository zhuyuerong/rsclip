#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文引导检测器（主模型）- 基于RemoteCLIP

完整前向传播：
1. Stage1: F_s, I_g = RemoteCLIPEncoder(I); t_c = RemoteCLIPTextEncoder("airplane")
2. Stage2: queries = Decoder(F_s, I_g, t_c)  # 上下文门控+迭代优化
3. Stage3: f_m, b_m = PredictionHeads(queries)
4. Inference: scores = <f_m, t_c>; outputs = NMS(b_m, scores)
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1_encoder import CLIPImageEncoder, CLIPTextEncoder, GlobalContextExtractor
from stage2_decoder import QueryInitializer, TextConditioner, ContextGating
from stage3_prediction import ClassificationHead, RegressionHead
from stage4_supervision import HungarianMatcher, TotalLoss


class ContextGuidedDetector(nn.Module):
    """
    上下文引导检测器 - 基于RemoteCLIP
    
    核心创新：
    1. 使用全局上下文 I_g 作为自动负样本
    2. 通过上下文门控引导局部查询
    3. 基于RemoteCLIP的遥感专用特征
    """
    
    def __init__(
        self,
        model_name: str = 'RN50',
        pretrained_path: str = 'checkpoints/RemoteCLIP-RN50.pt',
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        d_model: int = 256,
        d_clip: int = 512,
        context_gating_type: str = 'film',
        freeze_clip: bool = True
    ):
        """
        参数:
            model_name: RemoteCLIP模型名称
            pretrained_path: RemoteCLIP预训练权重路径
            num_queries: 查询数量 M
            num_decoder_layers: 解码器层数 L
            d_model: 模型维度
            d_clip: CLIP空间维度
            context_gating_type: 上下文门控类型
            freeze_clip: 是否冻结RemoteCLIP
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.d_clip = d_clip
        
        # Stage1: RemoteCLIP编码器
        self.image_encoder = CLIPImageEncoder(
            model_name=model_name,
            pretrained_path=pretrained_path,
            freeze=freeze_clip
        )
        
        self.text_encoder = CLIPTextEncoder(
            model_name=model_name,
            pretrained_path=pretrained_path
        )
        
        self.global_context_extractor = GlobalContextExtractor(d_clip=d_clip)
        
        # Stage2: 解码器组件
        self.query_initializer = QueryInitializer(
            num_queries=num_queries,
            d_model=d_model
        )
        
        self.text_conditioner = TextConditioner(
            d_model=d_model,
            d_text=d_clip
        )
        
        self.context_gating = ContextGating(
            d_model=d_model,
            d_context=d_clip,
            gating_type=context_gating_type
        )
        
        # 简化的解码器：直接使用组件而不实现完整的可变形注意力
        # （完整实现需要Deformable Attention等复杂模块）
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Stage3: 预测头
        self.classification_head = ClassificationHead(
            d_model=d_model,
            d_clip=d_clip
        )
        
        self.regression_head = RegressionHead(
            d_model=d_model
        )
        
        # Stage4: 匹配和损失（仅在训练时使用）
        self.matcher = HungarianMatcher()
        self.criterion = TotalLoss()
    
    def forward(
        self,
        images: torch.Tensor,
        text_queries: list,
        targets: dict = None
    ) -> dict:
        """
        完整前向传播
        
        参数:
            images: 输入图像 (B, 3, H, W)
            text_queries: 文本查询列表
            targets: 目标字典（训练时提供）
        
        返回:
            outputs: 输出字典
                - local_features: (B, M, d_clip)
                - pred_boxes: (B, M, 4)
                - scores: (B, M)（如果是推理模式）
        """
        batch_size = images.size(0)
        
        # ==================== Stage1: 特征提取 ====================
        # 图像编码
        multi_scale_features, global_embedding = self.image_encoder(images)
        
        # 提取全局上下文 I_g
        global_context = self.global_context_extractor(global_embedding)
        
        # 文本编码 t_c
        text_features = self.text_encoder(text_queries)  # (num_classes, d_clip)
        
        # 扩展到batch维度
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_classes, d_clip)
        
        # ==================== Stage2: 解码器 ====================
        # 初始化查询
        query_embed, query_pos = self.query_initializer(batch_size)
        
        # 扩展到batch维度
        queries = query_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, d_model)
        
        # 取第一个文本作为目标文本（简化）
        target_text = text_features[:, 0, :]  # (B, d_clip)
        
        # 迭代解码
        for layer in self.decoder_layers:
            # a. 文本调制
            queries = self.text_conditioner(queries, target_text)
            
            # b. 上下文门控（核心）
            queries = self.context_gating(queries, global_context)
            
            # c. 自注意力（简化的解码器层）
            # 注意：完整实现应该使用Deformable Attention
            queries = layer(queries, queries)
        
        # ==================== Stage3: 预测 ====================
        # 分类头：映射到CLIP空间
        local_features = self.classification_head(queries)  # (B, M, d_clip)
        
        # 回归头：预测边界框
        pred_boxes = self.regression_head(queries)  # (B, M, 4)
        
        # 输出
        outputs = {
            'local_features': local_features,
            'pred_boxes': pred_boxes,
            'global_context': global_context,
            'text_features': text_features
        }
        
        # ==================== Stage4: 训练模式 ====================
        if targets is not None and self.training:
            # 匹配
            matched_indices = self.matcher(
                local_features,
                pred_boxes,
                targets['target_classes'],
                targets['target_boxes']
            )
            
            # 计算损失
            loss_targets = {
                'text_embeddings': text_features,
                'global_context': global_context,
                'target_classes': targets['target_classes'],
                'target_boxes': targets['target_boxes']
            }
            
            total_loss, loss_dict = self.criterion(outputs, loss_targets, matched_indices)
            
            outputs['loss'] = total_loss
            outputs['loss_dict'] = loss_dict
            outputs['matched_indices'] = matched_indices
        
        # ==================== 推理模式 ====================
        else:
            # 计算分数 s_m = <f_m, t_c>
            scores = torch.bmm(local_features, target_text.unsqueeze(-1)).squeeze(-1)  # (B, M)
            outputs['scores'] = scores
        
        return outputs


if __name__ == "__main__":
    print("=" * 70)
    print("测试上下文引导检测器（基于RemoteCLIP）")
    print("=" * 70)
    
    # 创建模型（使用RemoteCLIP权重）
    model = ContextGuidedDetector(
        model_name='RN50',
        pretrained_path='checkpoints/RemoteCLIP-RN50.pt',
        num_queries=100,
        d_model=256,
        d_clip=512
    )
    model = model.cuda().eval()
    
    # 测试数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    text_queries = ["airplane", "building"]
    
    # 前向传播（推理模式）
    with torch.no_grad():
        outputs = model(images, text_queries)
    
    print("\n推理输出:")
    print(f"  局部特征: {outputs['local_features'].shape}")
    print(f"  预测框: {outputs['pred_boxes'].shape}")
    print(f"  分数: {outputs['scores'].shape}")
    print(f"  全局上下文: {outputs['global_context'].shape}")
    
    # 测试训练模式
    model.train()
    targets = {
        'target_classes': torch.randint(0, 2, (batch_size, 5)).cuda(),
        'target_boxes': torch.rand(batch_size, 5, 4).cuda()
    }
    
    outputs_train = model(images, text_queries, targets)
    
    print("\n训练输出:")
    print(f"  总损失: {outputs_train['loss'].item():.4f}")
    print(f"  损失详情: {list(outputs_train['loss_dict'].keys())}")
    
    print("\n✅ 上下文引导检测器测试完成！")


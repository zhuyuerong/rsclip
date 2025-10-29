#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteCLIP 图像编码器

功能：
提取图像的全局特征用于上下文引导
"""

import torch
import torch.nn as nn
import open_clip


class CLIPImageEncoder(nn.Module):
    """RemoteCLIP 图像编码器"""
    
    def __init__(
        self,
        model_name: str = 'RN50',
        pretrained_path: str = 'checkpoints/RemoteCLIP-RN50.pt',
        freeze: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # 加载RemoteCLIP模型
        print(f"🔄 加载RemoteCLIP图像编码器: {model_name}")
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        
        if pretrained_path:
            print(f"📦 加载RemoteCLIP权重: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location='cpu')
            self.model.load_state_dict(ckpt)
            print(f"✅ RemoteCLIP权重加载成功")
        
        # 冻结参数
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    def forward(self, images: torch.Tensor):
        """
        前向传播
        
        参数:
            images: (B, 3, H, W) 图像张量
        
        返回:
            multi_scale_features: 多尺度特征（简化版，返回全局特征）
            global_embedding: (B, d_clip) 全局图像特征
        """
        # 如果是训练模式，允许梯度
        if self.training:
            global_embedding = self.model.encode_image(images)
            global_embedding = global_embedding / global_embedding.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                global_embedding = self.model.encode_image(images)
                global_embedding = global_embedding / global_embedding.norm(dim=-1, keepdim=True)
        
        # 返回全局特征（多尺度特征简化为全局特征）
        multi_scale_features = global_embedding  # 简化实现
        
        return multi_scale_features, global_embedding


if __name__ == "__main__":
    encoder = CLIPImageEncoder(
        model_name='RN50',
        pretrained_path='../../checkpoints/RemoteCLIP-RN50.pt'
    )
    
    images = torch.randn(2, 3, 800, 800)
    features = encoder(images)
    
    print(f"图像特征形状: {features.shape}")
    print("✅ RemoteCLIP图像编码器测试完成！")


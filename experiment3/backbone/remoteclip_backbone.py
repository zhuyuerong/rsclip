#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteCLIP骨干网络

基于OV-ADETR的设计，使用RemoteCLIP替代CLIP

功能：
1. 图像特征提取（RemoteCLIP图像编码器）
2. 文本特征提取（RemoteCLIP文本编码器）
3. 图像-文本对齐
"""

import torch
import torch.nn as nn
import open_clip
from typing import List, Tuple


class RemoteCLIPBackbone(nn.Module):
    """
    RemoteCLIP骨干网络
    
    结合RemoteCLIP的图像和文本编码器
    """
    
    def __init__(
        self,
        model_name: str = 'RN50',
        pretrained_path: str = 'checkpoints/RemoteCLIP-RN50.pt',
        freeze_backbone: bool = True,
        output_layers: List[int] = None
    ):
        """
        参数:
            model_name: RemoteCLIP模型名称
            pretrained_path: RemoteCLIP权重路径
            freeze_backbone: 是否冻结骨干网络
            output_layers: 输出的特征层
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # 加载RemoteCLIP模型
        print(f"🔄 加载RemoteCLIP骨干网络: {model_name}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        
        # 加载预训练权重
        if pretrained_path:
            print(f"📦 加载RemoteCLIP权重: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location='cpu')
            self.model.load_state_dict(ckpt)
            print(f"✅ RemoteCLIP骨干网络加载成功")
        
        # 分词器
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # 冻结参数
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # 输出层配置
        if output_layers is None:
            if 'ViT' in model_name:
                self.output_layers = [6, 9, 12]  # ViT的多层输出
            else:
                self.output_layers = [2, 3, 4]   # ResNet的多层输出
        else:
            self.output_layers = output_layers
        
        # 注册hook提取中间特征
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册hook以提取中间层特征"""
        def get_activation(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        if 'ViT' in self.model_name:
            # ViT架构
            if hasattr(self.model.visual, 'transformer'):
                for idx in self.output_layers:
                    if idx < len(self.model.visual.transformer.resblocks):
                        self.model.visual.transformer.resblocks[idx].register_forward_hook(
                            get_activation(f'layer_{idx}')
                        )
        else:
            # ResNet架构
            if hasattr(self.model.visual, 'layer2'):
                self.model.visual.layer2.register_forward_hook(get_activation('layer_2'))
            if hasattr(self.model.visual, 'layer3'):
                self.model.visual.layer3.register_forward_hook(get_activation('layer_3'))
            if hasattr(self.model.visual, 'layer4'):
                self.model.visual.layer4.register_forward_hook(get_activation('layer_4'))
    
    def forward_image(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        图像特征提取
        
        参数:
            images: (B, 3, H, W)
        
        返回:
            multi_level_feats: 多层级特征列表
        """
        self.features = {}
        
        # 前向传播
        with torch.set_grad_enabled(not self.freeze_backbone):
            _ = self.model.encode_image(images)
        
        # 提取多层级特征
        multi_level_feats = []
        for layer_name in sorted(self.features.keys()):
            feat = self.features[layer_name]
            
            # 处理不同架构
            if 'ViT' in self.model_name and len(feat.shape) == 3:
                # ViT: (B, N+1, d) -> (B, N, d) -> (B, d, H, W)
                feat = feat[:, 1:, :]  # 去掉CLS token
                B, N, d = feat.shape
                H = W = int(N ** 0.5)
                feat = feat.reshape(B, H, W, d).permute(0, 3, 1, 2)
            
            multi_level_feats.append(feat)
        
        return multi_level_feats
    
    def forward_text(self, texts: List[str]) -> torch.Tensor:
        """
        文本特征提取
        
        参数:
            texts: 文本列表
        
        返回:
            text_features: (B, num_texts, d)
        """
        # 分词
        text_tokens = self.tokenizer(texts).to(next(self.model.parameters()).device)
        
        # 编码
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def forward(
        self,
        images: torch.Tensor,
        texts: List[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        完整前向传播
        
        参数:
            images: 图像张量
            texts: 文本列表
        
        返回:
            img_feats: 多层级图像特征
            txt_feats: 文本特征
        """
        img_feats = self.forward_image(images)
        txt_feats = self.forward_text(texts)
        
        return img_feats, txt_feats


if __name__ == "__main__":
    print("=" * 70)
    print("测试RemoteCLIP骨干网络")
    print("=" * 70)
    
    # 创建骨干网络
    backbone = RemoteCLIPBackbone(
        model_name='RN50',
        pretrained_path='checkpoints/RemoteCLIP-RN50.pt'
    )
    backbone = backbone.cuda().eval()
    
    # 测试数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 800, 800).cuda()
    texts = ["airplane", "ship", "harbor", "bridge"]
    
    # 前向传播
    with torch.no_grad():
        img_feats, txt_feats = backbone(images, texts)
    
    print(f"\n图像特征:")
    for i, feat in enumerate(img_feats):
        print(f"  层{i}: {feat.shape}")
    
    print(f"\n文本特征: {txt_feats.shape}")
    
    print("\n✅ RemoteCLIP骨干网络测试完成！")


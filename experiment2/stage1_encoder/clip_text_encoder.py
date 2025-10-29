#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteCLIP 文本编码器

功能：
输入仅目标类别（如 "airplane", "ship"）
输出归一化的正文本嵌入 t_c (目标)
"""

import torch
import torch.nn as nn
import open_clip


class CLIPTextEncoder(nn.Module):
    """RemoteCLIP 文本编码器"""
    
    def __init__(
        self,
        model_name: str = 'RN50',
        pretrained_path: str = 'checkpoints/RemoteCLIP-RN50.pt'
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # 加载RemoteCLIP模型
        print(f"🔄 加载RemoteCLIP文本编码器: {model_name}")
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        if pretrained_path:
            print(f"📦 加载RemoteCLIP权重: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location='cpu')
            self.model.load_state_dict(ckpt)
            print(f"✅ RemoteCLIP权重加载成功")
        
        self.model.eval()
    
    def forward(self, text_queries: list) -> torch.Tensor:
        """
        前向传播
        
        参数:
            text_queries: 文本查询列表，如 ["airplane", "ship"]
        
        返回:
            text_features: 归一化的文本嵌入 (N, d_clip)
        """
        # 分词
        text = self.tokenizer(text_queries).to(next(self.model.parameters()).device)
        
        # 编码
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features


if __name__ == "__main__":
    encoder = CLIPTextEncoder(
        model_name='RN50',
        pretrained_path='checkpoints/RemoteCLIP-RN50.pt'
    )
    encoder = encoder.cuda()
    
    texts = ["airplane", "ship", "building"]
    features = encoder(texts)
    
    print(f"文本特征形状: {features.shape}")
    print("✅ RemoteCLIP文本编码器测试完成！")


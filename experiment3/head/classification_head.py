#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比学习分类头

功能：
1. 基于对比学习的分类
2. 使用文本嵌入作为分类权重
3. 支持开放词汇检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveClassificationHead(nn.Module):
    """
    对比学习分类头
    
    通过计算查询特征与文本特征的相似度来进行分类
    """
    
    def __init__(
        self,
        d_model: int = 256,
        temperature: float = 0.07,
        normalize: bool = True
    ):
        """
        参数:
            d_model: 模型维度
            temperature: 温度参数
            normalize: 是否归一化特征
        """
        super().__init__()
        
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        
        # 查询特征投影
        self.query_proj = nn.Linear(d_model, d_model)
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
    
    def forward(
        self,
        query_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query_features: 查询特征 (B, num_queries, d_model) 或 (num_layers, B, num_queries, d_model)
            text_features: 文本特征 (B, num_classes, d_model)
        
        返回:
            logits: 分类logits (B, num_queries, num_classes) 或 (num_layers, B, num_queries, num_classes)
        """
        # 处理多层输出
        is_multi_layer = query_features.dim() == 4
        if is_multi_layer:
            num_layers, B, num_queries, d = query_features.shape
            query_features = query_features.reshape(-1, num_queries, d)
            # 扩展文本特征
            text_features = text_features.unsqueeze(0).repeat(num_layers, 1, 1, 1)
            text_features = text_features.reshape(-1, text_features.shape[2], text_features.shape[3])
        
        # 投影查询特征
        query_feat = self.query_proj(query_features)  # (B, num_queries, d_model)
        
        # 归一化
        if self.normalize:
            query_feat = F.normalize(query_feat, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度
        # (B, num_queries, d_model) @ (B, d_model, num_classes) -> (B, num_queries, num_classes)
        logits = torch.bmm(query_feat, text_features.transpose(1, 2))
        
        # 缩放
        logit_scale = self.logit_scale.exp()
        logits = logits * logit_scale
        
        # 恢复多层形状
        if is_multi_layer:
            logits = logits.reshape(num_layers, B, num_queries, -1)
        
        return logits


class MLPClassificationHead(nn.Module):
    """
    MLP分类头（备选方案）
    
    使用多层感知机进行分类
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 80,
        num_layers: int = 3,
        hidden_dim: int = 256
    ):
        """
        参数:
            d_model: 模型维度
            num_classes: 类别数
            num_layers: MLP层数
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # 构建MLP
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(d_model, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query_features: (B, num_queries, d_model) 或 (num_layers, B, num_queries, d_model)
        
        返回:
            logits: (B, num_queries, num_classes) 或 (num_layers, B, num_queries, num_classes)
        """
        return self.mlp(query_features)


if __name__ == "__main__":
    print("=" * 70)
    print("测试对比学习分类头")
    print("=" * 70)
    
    # 创建分类头
    cls_head = ContrastiveClassificationHead(
        d_model=256,
        temperature=0.07,
        normalize=True
    )
    
    # 测试数据
    batch_size = 2
    num_queries = 300
    num_classes = 20
    num_layers = 6
    
    # 单层输出
    query_features = torch.randn(batch_size, num_queries, 256)
    text_features = torch.randn(batch_size, num_classes, 256)
    
    logits = cls_head(query_features, text_features)
    print(f"\n单层输出:")
    print(f"  查询特征: {query_features.shape}")
    print(f"  文本特征: {text_features.shape}")
    print(f"  分类logits: {logits.shape}")
    
    # 多层输出
    query_features_multi = torch.randn(num_layers, batch_size, num_queries, 256)
    text_features = torch.randn(batch_size, num_classes, 256)
    
    logits_multi = cls_head(query_features_multi, text_features)
    print(f"\n多层输出:")
    print(f"  查询特征: {query_features_multi.shape}")
    print(f"  文本特征: {text_features.shape}")
    print(f"  分类logits: {logits_multi.shape}")
    
    print("\n✅ 对比学习分类头测试完成！")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界框回归头

功能：
1. 预测边界框坐标
2. 支持多层输出
3. 输出归一化的边界框坐标 [cx, cy, w, h]
"""

import torch
import torch.nn as nn


class BBoxRegressionHead(nn.Module):
    """
    边界框回归头
    
    使用MLP预测边界框坐标
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 3,
        hidden_dim: int = 256
    ):
        """
        参数:
            d_model: 模型维度
            num_layers: MLP层数
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.d_model = d_model
        
        # 构建MLP
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(d_model, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        
        # 输出层 (4个坐标: cx, cy, w, h)
        layers.append(nn.Linear(hidden_dim, 4))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 最后一层使用特殊初始化
        nn.init.constant_(self.mlp[-1].bias, 0)
    
    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query_features: (B, num_queries, d_model) 或 (num_layers, B, num_queries, d_model)
        
        返回:
            bbox_pred: (B, num_queries, 4) 或 (num_layers, B, num_queries, 4)
                      [cx, cy, w, h] 归一化到 [0, 1]
        """
        # MLP预测
        bbox_pred = self.mlp(query_features)
        
        # Sigmoid归一化到[0, 1]
        bbox_pred = torch.sigmoid(bbox_pred)
        
        return bbox_pred


class MultiscaleBBoxRegressionHead(nn.Module):
    """
    多尺度边界框回归头
    
    为不同的解码器层使用不同的回归头
    """
    
    def __init__(
        self,
        num_decoder_layers: int = 6,
        d_model: int = 256,
        num_layers: int = 3,
        hidden_dim: int = 256
    ):
        """
        参数:
            num_decoder_layers: 解码器层数
            d_model: 模型维度
            num_layers: MLP层数
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 为每个解码器层创建独立的回归头
        self.bbox_heads = nn.ModuleList([
            BBoxRegressionHead(
                d_model=d_model,
                num_layers=num_layers,
                hidden_dim=hidden_dim
            )
            for _ in range(num_decoder_layers)
        ])
    
    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query_features: (num_decoder_layers, B, num_queries, d_model)
        
        返回:
            bbox_preds: (num_decoder_layers, B, num_queries, 4)
        """
        assert query_features.dim() == 4, "需要多层输出"
        
        num_layers = query_features.shape[0]
        bbox_preds = []
        
        for i in range(num_layers):
            bbox_pred = self.bbox_heads[i](query_features[i])
            bbox_preds.append(bbox_pred)
        
        return torch.stack(bbox_preds, dim=0)


if __name__ == "__main__":
    print("=" * 70)
    print("测试边界框回归头")
    print("=" * 70)
    
    # 创建回归头
    bbox_head = BBoxRegressionHead(
        d_model=256,
        num_layers=3,
        hidden_dim=256
    )
    
    # 测试数据
    batch_size = 2
    num_queries = 300
    num_layers = 6
    
    # 单层输出
    query_features = torch.randn(batch_size, num_queries, 256)
    bbox_pred = bbox_head(query_features)
    
    print(f"\n单层输出:")
    print(f"  查询特征: {query_features.shape}")
    print(f"  边界框预测: {bbox_pred.shape}")
    print(f"  边界框范围: [{bbox_pred.min():.3f}, {bbox_pred.max():.3f}]")
    
    # 多层输出
    query_features_multi = torch.randn(num_layers, batch_size, num_queries, 256)
    bbox_pred_multi = bbox_head(query_features_multi)
    
    print(f"\n多层输出:")
    print(f"  查询特征: {query_features_multi.shape}")
    print(f"  边界框预测: {bbox_pred_multi.shape}")
    
    # 测试多尺度回归头
    multiscale_head = MultiscaleBBoxRegressionHead(
        num_decoder_layers=6,
        d_model=256
    )
    
    bbox_pred_multiscale = multiscale_head(query_features_multi)
    print(f"\n多尺度回归头:")
    print(f"  输入: {query_features_multi.shape}")
    print(f"  输出: {bbox_pred_multiscale.shape}")
    
    print("\n✅ 边界框回归头测试完成！")


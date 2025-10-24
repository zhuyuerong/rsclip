#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局-局部对比损失（核心创新）⭐

这是整个框架的核心创新点。通过"局部实例"与"全局场景"的自对比来自动学习背景抑制，
无需外部负样本。

数学原理:
----------
对于匹配到 GT 的查询 m：
- 正对: (f_m, t_c)  # 局部特征 vs 目标文本
- 负对: (f_m, I_g)  # 局部特征 vs 全局上下文

InfoNCE 损失：
L_GlobalContrast = -log[
    exp(<f_m, t_c> / τ) / 
    (exp(<f_m, t_c> / τ) + exp(<f_m, I_g> / τ))
]

直觉理解:
----------
假设我们在检测飞机：
- f_m: 某个候选框的局部特征（可能是飞机）
- t_c: "airplane" 的文本嵌入
- I_g: 整张图的全局嵌入（可能是 "天空+跑道+建筑" 的混合表示）

损失函数迫使:
1. f_m 接近 t_c（"飞机"）
2. f_m 远离 I_g（"天空"主导的全局场景）

这样模型会学会：
- 如果某个区域是飞机，它应该与 "airplane" 文本相似
- 如果某个区域是背景（天空/跑道），它会与全局上下文相似，从而被自动抑制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GlobalContrastLoss(nn.Module):
    """
    全局-局部对比损失
    
    参数:
        temperature (float): 对比学习温度参数 τ，控制分布的尖锐程度
                            τ 越小，分布越尖锐，难样本权重越大
        normalize (bool): 是否对特征进行 L2 归一化
        reduction (str): 损失聚合方式，'mean' 或 'sum'
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.reduction = reduction
        
        # 用于数值稳定性
        self.eps = 1e-8
    
    def forward(
        self,
        local_features: torch.Tensor,      # f_m: (N, M, d_clip)
        text_embeddings: torch.Tensor,     # t_c: (N, num_classes, d_clip)
        global_context: torch.Tensor,      # I_g: (N, d_clip)
        matched_indices: list,             # 匹配的查询索引
        target_classes: torch.Tensor       # 目标类别 (N, num_targets)
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        参数:
            local_features: 局部特征 f_m，形状 (batch_size, num_queries, d_clip)
            text_embeddings: 文本嵌入 t_c，形状 (batch_size, num_classes, d_clip)
            global_context: 全局上下文 I_g，形状 (batch_size, d_clip)
            matched_indices: 匈牙利匹配结果，每个元素是 (query_idx, target_idx) 的元组
            target_classes: 目标类别索引，形状 (batch_size, num_targets)
        
        返回:
            loss: 标量损失值
            stats: 统计信息字典，包含：
                - positive_sim: 正对相似度 <f_m, t_c>
                - negative_sim: 负对相似度 <f_m, I_g>
                - margin: 相似度差距 (positive - negative)
        """
        # L2 归一化
        if self.normalize:
            local_features = F.normalize(local_features, p=2, dim=-1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            global_context = F.normalize(global_context, p=2, dim=-1)
        
        batch_size = local_features.size(0)
        total_loss = 0
        num_matched = 0
        
        # 用于统计
        all_positive_sims = []
        all_negative_sims = []
        
        # 遍历 batch
        for b in range(batch_size):
            # 获取该 batch 的匹配结果
            query_idx, target_idx = matched_indices[b]
            
            if len(query_idx) == 0:
                # 该图像没有匹配的查询
                continue
            
            # 获取匹配的局部特征
            matched_features = local_features[b][query_idx]  # (num_matched, d_clip)
            
            # 获取对应的目标类别
            matched_target_classes = target_classes[b][target_idx]  # (num_matched,)
            
            # 获取对应的文本嵌入（正样本）
            matched_text = text_embeddings[b][matched_target_classes]  # (num_matched, d_clip)
            
            # 获取全局上下文（负样本）
            global_ctx = global_context[b].unsqueeze(0)  # (1, d_clip)
            global_ctx = global_ctx.expand(matched_features.size(0), -1)  # (num_matched, d_clip)
            
            # 计算相似度
            # 正对相似度: <f_m, t_c>
            positive_sim = torch.sum(matched_features * matched_text, dim=-1)  # (num_matched,)
            
            # 负对相似度: <f_m, I_g>
            negative_sim = torch.sum(matched_features * global_ctx, dim=-1)  # (num_matched,)
            
            # InfoNCE 损失
            # L = -log[exp(pos/τ) / (exp(pos/τ) + exp(neg/τ))]
            #   = -log[1 / (1 + exp((neg-pos)/τ))]
            #   = log(1 + exp((neg-pos)/τ))
            #   = log_softmax([pos, neg], dim=-1)[0] 的负数
            
            # 为了数值稳定，使用 logsumexp
            logits = torch.stack([positive_sim, negative_sim], dim=-1) / self.temperature  # (num_matched, 2)
            
            # InfoNCE: -log P(positive | {positive, negative})
            loss = -F.log_softmax(logits, dim=-1)[:, 0]  # (num_matched,)
            
            # 聚合
            if self.reduction == 'mean':
                total_loss += loss.mean()
            elif self.reduction == 'sum':
                total_loss += loss.sum()
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")
            
            num_matched += len(query_idx)
            
            # 统计
            all_positive_sims.append(positive_sim.detach())
            all_negative_sims.append(negative_sim.detach())
        
        # 计算平均损失
        if num_matched > 0:
            if self.reduction == 'mean':
                final_loss = total_loss / batch_size
            else:
                final_loss = total_loss
        else:
            # 没有匹配的查询，返回零损失
            final_loss = torch.tensor(0.0, device=local_features.device, requires_grad=True)
        
        # 统计信息
        if all_positive_sims:
            all_positive_sims = torch.cat(all_positive_sims)
            all_negative_sims = torch.cat(all_negative_sims)
            
            stats = {
                'positive_sim': all_positive_sims.mean().item(),
                'negative_sim': all_negative_sims.mean().item(),
                'margin': (all_positive_sims - all_negative_sims).mean().item(),
                'num_matched': num_matched
            }
        else:
            stats = {
                'positive_sim': 0.0,
                'negative_sim': 0.0,
                'margin': 0.0,
                'num_matched': 0
            }
        
        return final_loss, stats
    
    def compute_similarity_matrix(
        self,
        local_features: torch.Tensor,      # (N, M, d)
        text_embeddings: torch.Tensor,     # (N, C, d)
        global_context: torch.Tensor       # (N, d)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算相似度矩阵（用于可视化分析）
        
        返回:
            text_sim: 局部特征与文本的相似度 (N, M, C)
            global_sim: 局部特征与全局上下文的相似度 (N, M)
        """
        # 归一化
        if self.normalize:
            local_features = F.normalize(local_features, p=2, dim=-1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
            global_context = F.normalize(global_context, p=2, dim=-1)
        
        # 计算局部-文本相似度
        # (N, M, d) @ (N, d, C) -> (N, M, C)
        text_sim = torch.bmm(local_features, text_embeddings.transpose(1, 2))
        
        # 计算局部-全局相似度
        # (N, M, d) @ (N, d, 1) -> (N, M, 1) -> (N, M)
        global_sim = torch.bmm(
            local_features, 
            global_context.unsqueeze(-1)
        ).squeeze(-1)
        
        return text_sim, global_sim


def test_global_contrast_loss():
    """测试全局对比损失"""
    print("=" * 70)
    print("测试全局对比损失")
    print("=" * 70)
    
    # 创建损失函数
    loss_fn = GlobalContrastLoss(temperature=0.07)
    
    # 模拟数据
    batch_size = 2
    num_queries = 100
    d_clip = 512
    num_classes = 10
    
    # 局部特征
    local_features = torch.randn(batch_size, num_queries, d_clip)
    
    # 文本嵌入
    text_embeddings = torch.randn(batch_size, num_classes, d_clip)
    
    # 全局上下文
    global_context = torch.randn(batch_size, d_clip)
    
    # 匹配结果（每个图像有5个匹配）
    matched_indices = [
        (torch.tensor([0, 10, 20, 30, 40]), torch.tensor([0, 1, 2, 0, 1])),
        (torch.tensor([5, 15, 25, 35, 45]), torch.tensor([1, 2, 3, 1, 2]))
    ]
    
    # 目标类别
    target_classes = torch.randint(0, num_classes, (batch_size, 5))
    
    # 计算损失
    loss, stats = loss_fn(
        local_features,
        text_embeddings,
        global_context,
        matched_indices,
        target_classes
    )
    
    print(f"\n损失值: {loss.item():.4f}")
    print(f"统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 测试相似度矩阵
    text_sim, global_sim = loss_fn.compute_similarity_matrix(
        local_features, text_embeddings, global_context
    )
    
    print(f"\n相似度矩阵形状:")
    print(f"  text_sim: {text_sim.shape}")
    print(f"  global_sim: {global_sim.shape}")
    
    print("\n✅ 全局对比损失测试完成！")


if __name__ == "__main__":
    test_global_contrast_loss()


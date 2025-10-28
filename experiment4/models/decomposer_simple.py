# -*- coding: utf-8 -*-
"""
简化版分解器 - 直接逐元素乘法
完全符合Surgery论文的思想：简单有效，无额外参数

与CrossAttention版本对比：
- 无MultiheadAttention的Q/K/V投影
- 参数量减少~90%
- 更适合小数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedTextGuidedDecomposer(nn.Module):
    """
    简化版文本引导分解器
    
    数学构思：
    1. 构建交互空间：H[j,k,d] = F̃[j,d] × T_k[d] (直接乘法)
    2. Surgery去冗余：H_clean = H - mean(H, dim=k)
    3. 分解到M个原子模式
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_components = config.n_components  # 20
        self.embed_dim = config.embed_dim  # 512
        
        # 只需要分解MLP（无attention参数）
        self.decompose_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.n_components)
        )
        
        # 原子模式模板
        self.component_templates = nn.Parameter(
            torch.randn(self.n_components, self.embed_dim)
        )
        nn.init.orthogonal_(self.component_templates)
    
    def forward(self, F_clean, text_features):
        """
        执行简化的文本引导分解
        
        Args:
            F_clean: [B, N, D] 去噪后的图像特征
            text_features: [K, D] K个文本特征
        
        Returns:
            Q: [B, N, D, M] 稀疏分解结果
        """
        B, N, D = F_clean.shape
        K = len(text_features)
        
        # 确保dtype一致
        if text_features.dtype != F_clean.dtype:
            text_features = text_features.to(F_clean.dtype)
        
        # ===== 步骤1: 构建交互空间H（直接逐元素乘法）=====
        # F_clean: [B, N, D] → [B, N, 1, D]
        # text_features: [K, D] → [1, 1, K, D]
        F_expanded = F_clean.unsqueeze(2)  # [B, N, 1, D]
        T_expanded = text_features.unsqueeze(0).unsqueeze(0)  # [1, 1, K, D]
        
        # 直接乘法构建交互空间
        H = F_expanded * T_expanded  # [B, N, K, D]
        
        # ===== 步骤2: Surgery去冗余（跨K个类别）=====
        R_H = H.mean(dim=2, keepdim=True)  # [B, N, 1, D]
        H_clean = H - R_H  # [B, N, K, D]
        
        # ===== 步骤3: 分解到M个原子模式 =====
        # 对每个patch的K个交互特征分别分解
        Q_list = []
        
        for j in range(N):
            # 当前patch的交互特征
            H_j = H_clean[:, j, :, :]  # [B, K, D]
            
            # 对每个文本的交互特征，映射到M维
            components_j = self.decompose_mlp(H_j)  # [B, K, M]
            
            # 聚合所有文本的贡献（平均）
            components_aggregated = components_j.mean(dim=1)  # [B, M]
            
            Q_list.append(components_aggregated)
        
        Q_patch_level = torch.stack(Q_list, dim=1)  # [B, N, M]
        
        # ===== 步骤4: 扩展到特征维度 =====
        # 使用可学习的原子模式模板
        templates_expanded = self.component_templates.unsqueeze(0).unsqueeze(0)  # [1, 1, M, D]
        F_expanded = F_clean.unsqueeze(-1)  # [B, N, D, 1]
        Q_patch_expanded = Q_patch_level.unsqueeze(2)  # [B, N, 1, M]
        
        # 组合
        Q = F_expanded * templates_expanded.transpose(2, 3) * Q_patch_expanded  # [B, N, D, M]
        
        # ===== 步骤5: 稀疏化 =====
        Q_sparse = self.apply_sparsity(Q)
        
        return Q_sparse
    
    def apply_sparsity(self, Q):
        """稀疏化：只保留前k%的元素"""
        B, N, D, M = Q.shape
        
        k = int(D * self.config.sparsity_ratio)
        if k < 1:
            k = 1
        
        # Reshape
        Q_reshaped = Q.permute(0, 1, 3, 2).reshape(B * N * M, D)  # [B*N*M, D]
        
        # Top-k选择
        topk_vals, topk_idx = torch.topk(Q_reshaped.abs(), k, dim=1)
        
        # 稀疏mask
        sparse_mask = torch.zeros_like(Q_reshaped)
        sparse_mask.scatter_(1, topk_idx, 1.0)
        
        # 应用mask
        Q_sparse_reshaped = Q_reshaped * sparse_mask
        
        # Reshape回原形状
        Q_sparse = Q_sparse_reshaped.reshape(B, N, M, D).permute(0, 1, 3, 2)
        
        return Q_sparse


class SimplifiedImageOnlyDecomposer(nn.Module):
    """
    简化版图像分解器
    
    纯图像分解（无文本引导）
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_components = config.n_components
        self.embed_dim = config.embed_dim
        
        # Self-attention（保留，用于捕捉patch间关系）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 分解MLP
        self.decompose_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.n_components)
        )
        
        # 原子模式模板
        self.component_templates = nn.Parameter(
            torch.randn(self.n_components, self.embed_dim)
        )
        nn.init.orthogonal_(self.component_templates)
    
    def forward(self, F_clean):
        """
        执行纯图像分解
        
        Args:
            F_clean: [B, N, D]
        
        Returns:
            Q: [B, N, D, M]
        """
        B, N, D = F_clean.shape
        
        # Self-attention（patch间交互）
        attn_output, _ = self.self_attn(
            F_clean, F_clean, F_clean
        )  # [B, N, D]
        
        # 残差连接
        F_enhanced = F_clean + attn_output
        
        # 分解
        components = self.decompose_mlp(F_enhanced)  # [B, N, M]
        
        # 扩展到特征维度
        templates_expanded = self.component_templates.unsqueeze(0).unsqueeze(0)  # [1, 1, M, D]
        F_expanded = F_clean.unsqueeze(-1)  # [B, N, D, 1]
        components_expanded = components.unsqueeze(2)  # [B, N, 1, M]
        
        Q = F_expanded * templates_expanded.transpose(2, 3) * components_expanded  # [B, N, D, M]
        
        # 稀疏化
        Q_sparse = self.apply_sparsity(Q)
        
        return Q_sparse
    
    def apply_sparsity(self, Q):
        """稀疏化"""
        B, N, D, M = Q.shape
        
        k = int(D * self.config.sparsity_ratio)
        if k < 1:
            k = 1
        
        Q_reshaped = Q.permute(0, 1, 3, 2).reshape(B * N * M, D)
        topk_vals, topk_idx = torch.topk(Q_reshaped.abs(), k, dim=1)
        
        sparse_mask = torch.zeros_like(Q_reshaped)
        sparse_mask.scatter_(1, topk_idx, 1.0)
        
        Q_sparse_reshaped = Q_reshaped * sparse_mask
        Q_sparse = Q_sparse_reshaped.reshape(B, N, M, D).permute(0, 1, 3, 2)
        
        return Q_sparse


def test_simplified_decomposer():
    """测试简化版分解器"""
    from experiment4.config import get_config
    
    config = get_config()
    
    # 创建模型
    text_decomposer = SimplifiedTextGuidedDecomposer(config)
    img_decomposer = SimplifiedImageOnlyDecomposer(config)
    
    # 测试输入
    B, N, D = 2, 49, 512
    K = 14
    
    F_clean = torch.randn(B, N, D)
    text_features = torch.randn(K, D)
    
    # 前向传播
    Q_text = text_decomposer(F_clean, text_features)
    Q_img = img_decomposer(F_clean)
    
    print("简化版分解器测试:")
    print(f"  F_clean: {F_clean.shape}")
    print(f"  text_features: {text_features.shape}")
    print(f"  Q_text: {Q_text.shape}")
    print(f"  Q_img: {Q_img.shape}")
    
    # 参数统计
    text_params = sum(p.numel() for p in text_decomposer.parameters() if p.requires_grad)
    img_params = sum(p.numel() for p in img_decomposer.parameters() if p.requires_grad)
    
    print(f"\n参数量:")
    print(f"  Text分解器: {text_params:,}")
    print(f"  Image分解器: {img_params:,}")
    print(f"  总计: {(text_params + img_params):,}")
    
    print("\n✓ 测试通过！")


if __name__ == "__main__":
    test_simplified_decomposer()


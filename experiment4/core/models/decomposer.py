# -*- coding: utf-8 -*-
"""
文本引导的稀疏分解器
使用Cross Attention实现文本-图像交互，并分解到原子模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedDecomposer(nn.Module):
    """
    学习的稀疏分解器：F_clean + Text → Q[B, 196, 512, 20]
    
    核心思想：
    1. 文本-图像交互（Cross Attention）
    2. 分解到20个原子模式
    3. 稀疏化（90%为0）
    4. 正交约束
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_components = config.n_components
        self.embed_dim = config.embed_dim
        
        # 文本-图像交互模块（Cross Attention）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        # 分解头：将交互特征分解到20个原子模式
        self.decompose_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.n_components)
        )
        
        # 原子模式的"模板"（可学习）
        self.component_templates = nn.Parameter(
            torch.randn(self.n_components, self.embed_dim)
        )
        nn.init.orthogonal_(self.component_templates)
        
        # 稀疏化参数（可学习的阈值）
        self.sparsity_threshold = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, F_clean, text_features):
        """
        执行文本引导的稀疏分解
        
        Args:
            F_clean: [B, 196, 512] 去噪后的图像特征
            text_features: [K, 512] K个文本特征（WordNet词或dummy）
        
        Returns:
            Q: [B, 196, 512, 20] 稀疏分解结果
            attn_weights: [B, K, 196] 文本对patch的注意力
        """
        B, N, D = F_clean.shape
        K = len(text_features)
        
        # 确保text_features的dtype与F_clean一致（half精度兼容）
        if text_features.dtype != F_clean.dtype:
            text_features = text_features.to(F_clean.dtype)
        
        # ===== 步骤1: 文本引导的交互 =====
        # 文本作为query，patch作为key/value
        text_expanded = text_features.unsqueeze(0).expand(B, -1, -1)  # [B, K, 512]
        
        # Cross Attention
        attn_output, attn_weights = self.cross_attn(
            query=self.norm1(text_expanded),
            key=self.norm1(F_clean),
            value=F_clean
        )  # attn_output: [B, K, 512], attn_weights: [B, K, 196]
        
        attn_output = self.norm2(attn_output)
        
        # ===== 步骤2: 逐patch计算分解 =====
        Q_list = []
        
        for j in range(N):
            # 当前patch特征
            patch_j = F_clean[:, j, :]  # [B, 512]
            
            # 与每个文本的交互（逐元素乘积 + 加权）
            # [B, 1, 512] * [B, K, 512] → [B, K, 512]
            patch_text_interact = patch_j.unsqueeze(1) * attn_output
            
            # 加权（用attention权重）
            weights = attn_weights[:, :, j].unsqueeze(-1)  # [B, K, 1]
            patch_text_weighted = patch_text_interact * weights
            
            # Surgery去冗余（跨K个文本）
            redundant_j = patch_text_weighted.mean(dim=1, keepdim=True)  # [B, 1, 512]
            patch_text_clean = patch_text_weighted - redundant_j  # [B, K, 512]
            
            # 分解到20个原子模式
            # 对每个文本的交互特征，映射到20维
            components_j = self.decompose_mlp(patch_text_clean)  # [B, K, 20]
            
            # 聚合所有文本的贡献（加权和）
            components_aggregated = (components_j * weights).sum(dim=1)  # [B, 20]
            
            Q_list.append(components_aggregated)
        
        Q_patch_level = torch.stack(Q_list, dim=1)  # [B, 196, 20]
        
        # ===== 步骤3: 扩展到特征维度 =====
        # 使用可学习的原子模式模板
        # Q[b,j,d,m] = Q_patch_level[b,j,m] * component_templates[m,d] * F_clean[b,j,d]
        
        # 广播乘法
        templates_expanded = self.component_templates.unsqueeze(0).unsqueeze(0)  # [1, 1, 20, 512]
        F_expanded = F_clean.unsqueeze(-1)  # [B, 196, 512, 1]
        Q_patch_expanded = Q_patch_level.unsqueeze(2)  # [B, 196, 1, 20]
        
        # 组合
        Q = F_expanded * templates_expanded.transpose(2, 3) * Q_patch_expanded  # [B, 196, 512, 20]
        
        # ===== 步骤4: 稀疏化 =====
        Q_sparse = self.apply_sparsity(Q)
        
        return Q_sparse, attn_weights
    
    def apply_sparsity(self, Q):
        """
        稀疏化：只保留前k%的元素
        
        Args:
            Q: [B, 196, 512, 20]
        
        Returns:
            Q_sparse: [B, 196, 512, 20]
        """
        B, N, D, M = Q.shape
        
        # 对每个(batch, patch, component)，找到512维中的top-k
        k = int(D * self.config.sparsity_ratio)
        if k < 1:
            k = 1  # 至少保留1个元素
        
        # Reshape以便处理
        Q_reshaped = Q.permute(0, 1, 3, 2).reshape(B * N * M, D)  # [B*196*20, 512]
        
        # Top-k选择（基于绝对值）
        topk_vals, topk_idx = torch.topk(Q_reshaped.abs(), k, dim=1)
        
        # 创建mask
        mask = torch.zeros_like(Q_reshaped)
        mask.scatter_(1, topk_idx, 1.0)
        
        # 应用mask
        Q_sparse = Q_reshaped * mask
        Q_sparse = Q_sparse.reshape(B, N, M, D).permute(0, 1, 3, 2)  # [B, 196, 512, 20]
        
        # 归一化每个原子模式
        Q_norm = F.normalize(Q_sparse, dim=2, eps=1e-6)  # 在512维上归一化
        
        return Q_norm
    
    def get_component_templates(self):
        """
        获取原子模式模板（用于可视化）
        
        Returns:
            templates: [20, 512]
        """
        return self.component_templates.detach()


class ImageOnlyDecomposer(nn.Module):
    """
    纯图像分解器（不依赖文本）
    
    用于Zero-shot泛化到unseen类
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_components = config.n_components
        self.embed_dim = config.embed_dim
        
        # Self-attention模块
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
            F_clean: [B, 196, 512]
        
        Returns:
            Q: [B, 196, 512, 20]
        """
        B, N, D = F_clean.shape
        
        # Self-attention增强特征
        F_enhanced, _ = self.self_attn(
            query=F_clean,
            key=F_clean,
            value=F_clean
        )  # [B, 196, 512]
        
        # 残差连接
        F_enhanced = F_enhanced + F_clean
        
        # 分解到20个原子模式
        Q_patch_level = self.decompose_mlp(F_enhanced)  # [B, 196, 20]
        
        # 扩展到特征维度
        templates_expanded = self.component_templates.unsqueeze(0).unsqueeze(0)  # [1, 1, 20, 512]
        F_expanded = F_clean.unsqueeze(-1)  # [B, 196, 512, 1]
        Q_patch_expanded = Q_patch_level.unsqueeze(2)  # [B, 196, 1, 20]
        
        Q = F_expanded * templates_expanded.transpose(2, 3) * Q_patch_expanded  # [B, 196, 512, 20]
        
        # 稀疏化
        Q_sparse = self._apply_sparsity(Q)
        
        return Q_sparse
    
    def _apply_sparsity(self, Q):
        """稀疏化"""
        B, N, D, M = Q.shape
        k = int(D * self.config.sparsity_ratio)
        if k < 1:
            k = 1
        
        Q_reshaped = Q.permute(0, 1, 3, 2).reshape(B * N * M, D)
        topk_vals, topk_idx = torch.topk(Q_reshaped.abs(), k, dim=1)
        
        mask = torch.zeros_like(Q_reshaped)
        mask.scatter_(1, topk_idx, 1.0)
        
        Q_sparse = Q_reshaped * mask
        Q_sparse = Q_sparse.reshape(B, N, M, D).permute(0, 1, 3, 2)
        Q_norm = F.normalize(Q_sparse, dim=2, eps=1e-6)
        
        return Q_norm


def test_decomposer():
    """测试分解器"""
    print("测试文本引导分解器...")
    
    # 模拟配置
    class DummyConfig:
        n_components = 20
        embed_dim = 512
        cross_attn_heads = 8
        cross_attn_dropout = 0.1
        sparsity_ratio = 0.1
    
    config = DummyConfig()
    
    # 创建分解器
    decomposer = TextGuidedDecomposer(config)
    
    # 模拟输入
    F_clean = torch.randn(4, 196, 512)
    text_features = torch.randn(20, 512)
    
    # 分解
    Q, attn_weights = decomposer(F_clean, text_features)
    
    print(f"输入形状: {F_clean.shape}")
    print(f"输出形状: {Q.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"稀疏度: {(Q.abs() > 1e-6).float().mean():.4f}")
    
    print("\n测试纯图像分解器...")
    img_decomposer = ImageOnlyDecomposer(config)
    Q_img = img_decomposer(F_clean)
    print(f"图像分解输出形状: {Q_img.shape}")
    print(f"稀疏度: {(Q_img.abs() > 1e-6).float().mean():.4f}")
    
    print("测试通过！")


if __name__ == "__main__":
    test_decomposer()


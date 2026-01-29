"""
Query Injection模块 (Q-Use)

三种使用策略:
1. Use-1 初始化替换/混入 (MVP必做)
2. Use-2 对齐loss (稳定增益)
3. Use-3 Prior loss (复杂但上限高)

关键: 保证与Deformable DETR的query_embed接口完全兼容
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class QueryMixer(nn.Module):
    """
    Query混合器: 将pseudo queries与learnable queries混合
    
    支持多种混合模式:
    - replace: 100%替换
    - concat: [pseudo, learnable] 拼接
    - ratio: 按比例混合
    - attention: 用attention融合
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_learnable_queries: int = 300,
        num_pseudo_queries: int = 100,
        mix_mode: str = 'concat',  # 'replace', 'concat', 'ratio', 'attention'
        pseudo_ratio: float = 0.5,  # 用于ratio模式
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_learnable_queries = num_learnable_queries
        self.num_pseudo_queries = num_pseudo_queries
        self.mix_mode = mix_mode
        self.pseudo_ratio = pseudo_ratio
        
        # Learnable queries (原Deformable DETR的query_embed)
        self.learnable_query_embed = nn.Embedding(num_learnable_queries, hidden_dim * 2)
        
        # 用于attention融合模式
        if mix_mode == 'attention':
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 用于ratio模式的gate
        if mix_mode == 'ratio':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.learnable_query_embed.weight)
    
    def forward(
        self,
        pseudo_queries: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        混合pseudo queries和learnable queries
        
        Args:
            pseudo_queries: dict from HeatmapQueryGenerator
                - query_embed: [B, K, 2*d]
                - reference_points: [B, K, 2]
            batch_size: 批次大小
            
        Returns:
            query_embed: [B, Q, 2*d] 用于decoder
            reference_points: [B, Q, 2] 用于decoder (如果pseudo_queries提供)
        """
        device = self.learnable_query_embed.weight.device
        
        # 获取learnable queries
        learnable = self.learnable_query_embed.weight  # [N_learn, 2*d]
        learnable = learnable.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_learn, 2*d]
        
        if pseudo_queries is None or self.mix_mode == 'replace_learnable':
            # 纯learnable模式 (baseline)
            return learnable, None
        
        pseudo_embed = pseudo_queries['query_embed']  # [B, K, 2*d]
        pseudo_ref = pseudo_queries.get('reference_points', None)  # [B, K, 2]
        
        if self.mix_mode == 'replace':
            # 完全用pseudo替换
            return pseudo_embed, pseudo_ref
        
        elif self.mix_mode == 'concat':
            # 拼接: [pseudo, learnable的一部分]
            # 总数保持为 num_pseudo + (num_learnable - num_pseudo)
            num_keep = self.num_learnable_queries - self.num_pseudo_queries
            if num_keep > 0:
                mixed = torch.cat([pseudo_embed, learnable[:, :num_keep]], dim=1)
            else:
                mixed = pseudo_embed[:, :self.num_learnable_queries]
            
            # reference_points只对pseudo部分有效
            if pseudo_ref is not None:
                ref_pad = torch.zeros(batch_size, num_keep, 2, device=device)
                mixed_ref = torch.cat([pseudo_ref, ref_pad], dim=1) if num_keep > 0 else pseudo_ref
            else:
                mixed_ref = None
            
            return mixed, mixed_ref
        
        elif self.mix_mode == 'ratio':
            # 按比例软混合 (需要两者数量相同)
            K = min(pseudo_embed.shape[1], learnable.shape[1])
            pseudo_embed = pseudo_embed[:, :K]
            learnable = learnable[:, :K]
            
            # 学习一个per-query的gate
            concat_feat = torch.cat([pseudo_embed, learnable], dim=-1)  # [B, K, 4*d]
            gate = self.gate(concat_feat)  # [B, K, 1]
            
            mixed = gate * pseudo_embed + (1 - gate) * learnable
            
            return mixed, pseudo_ref[:, :K] if pseudo_ref is not None else None
        
        elif self.mix_mode == 'attention':
            # 用attention融合
            K = pseudo_embed.shape[1]
            
            # pseudo作为query, learnable作为key/value
            fused, _ = self.fusion_attn(pseudo_embed, learnable, learnable)
            fused = self.fusion_norm(pseudo_embed + fused)
            
            return fused, pseudo_ref
        
        else:
            raise ValueError(f"Unknown mix_mode: {self.mix_mode}")


class QueryAlignmentLoss(nn.Module):
    """
    Use-2: Query对齐Loss
    
    让decoder输出的query与pseudo query在特征空间接近
    支持: L2, Cosine, InfoNCE
    """
    
    def __init__(
        self,
        loss_type: str = 'l2',  # 'l2', 'cosine', 'infonce'
        temperature: float = 0.07,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.loss_weight = loss_weight
    
    def forward(
        self,
        decoder_queries: torch.Tensor,
        pseudo_queries: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算对齐loss
        
        Args:
            decoder_queries: [B, Q, d] decoder第一层输出的query features
            pseudo_queries: [B, K, d] pseudo query content (stopgrad)
            mask: [B, K] 有效的pseudo query mask
            
        Returns:
            loss: scalar
        """
        # 只对有pseudo的位置计算loss
        K = pseudo_queries.shape[1]
        decoder_queries = decoder_queries[:, :K]  # [B, K, d]
        
        # Stopgrad pseudo queries
        pseudo_queries = pseudo_queries.detach()
        
        if self.loss_type == 'l2':
            loss = F.mse_loss(decoder_queries, pseudo_queries, reduction='none')
            if mask is not None:
                loss = (loss.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
            else:
                loss = loss.mean()
                
        elif self.loss_type == 'cosine':
            # 1 - cosine_similarity
            decoder_norm = F.normalize(decoder_queries, p=2, dim=-1)
            pseudo_norm = F.normalize(pseudo_queries, p=2, dim=-1)
            similarity = (decoder_norm * pseudo_norm).sum(dim=-1)  # [B, K]
            loss = 1 - similarity
            if mask is not None:
                loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            else:
                loss = loss.mean()
                
        elif self.loss_type == 'infonce':
            # InfoNCE对比学习loss
            B, K, d = decoder_queries.shape
            
            decoder_norm = F.normalize(decoder_queries, p=2, dim=-1)  # [B, K, d]
            pseudo_norm = F.normalize(pseudo_queries, p=2, dim=-1)    # [B, K, d]
            
            # 计算相似度矩阵
            sim_matrix = torch.bmm(decoder_norm, pseudo_norm.transpose(1, 2))  # [B, K, K]
            sim_matrix = sim_matrix / self.temperature
            
            # 对角线是正样本
            labels = torch.arange(K, device=decoder_queries.device).unsqueeze(0).expand(B, -1)
            loss = F.cross_entropy(sim_matrix.reshape(B*K, K), labels.reshape(B*K))
            
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss * self.loss_weight


class AttentionPriorLoss(nn.Module):
    """
    Use-3: Attention Prior Loss
    
    约束decoder的cross-attention map与heatmap匹配
    或约束预测框中心落在高响应区域
    
    注意: 这个loss比较复杂,容易踩坑,建议最后做
    """
    
    def __init__(
        self,
        loss_type: str = 'center',  # 'center', 'attn_map'
        loss_weight: float = 1.0,
        sigma: float = 0.1,  # 用于soft constraint
    ):
        super().__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.sigma = sigma
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        heatmap: torch.Tensor,
        cross_attn_weights: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算prior loss
        
        Args:
            pred_boxes: [B, Q, 4] 预测框 (cx, cy, w, h) 归一化
            heatmap: [B, H, W] 热图
            cross_attn_weights: [B, Q, H*W] decoder cross-attention权重 (可选)
            reference_points: [B, Q, 2] decoder参考点 (可选)
            
        Returns:
            loss: scalar
        """
        if self.loss_type == 'center':
            # 预测框中心应该落在高响应区域
            return self._center_prior_loss(pred_boxes, heatmap)
        
        elif self.loss_type == 'attn_map':
            # Cross-attention map应该与heatmap匹配
            if cross_attn_weights is None:
                return torch.tensor(0.0, device=pred_boxes.device)
            return self._attn_map_loss(cross_attn_weights, heatmap)
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def _center_prior_loss(self, pred_boxes: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        """预测框中心应该在高响应区域"""
        B, Q, _ = pred_boxes.shape
        _, H, W = heatmap.shape
        
        # 获取预测框中心
        centers = pred_boxes[:, :, :2]  # [B, Q, 2] (cx, cy)
        
        # 在热图上采样中心位置的值
        grid = centers.clone()
        grid[..., 0] = 2 * grid[..., 0] - 1  # x: [0,1] -> [-1,1]
        grid[..., 1] = 2 * grid[..., 1] - 1  # y: [0,1] -> [-1,1]
        grid = grid.view(B, Q, 1, 2)
        
        heatmap_at_centers = F.grid_sample(
            heatmap.unsqueeze(1), grid, 
            mode='bilinear', padding_mode='border', align_corners=True
        ).view(B, Q)  # [B, Q]
        
        # Loss: 鼓励中心落在高响应区域 (负log likelihood)
        # heatmap值越高越好
        eps = 1e-6
        loss = -torch.log(heatmap_at_centers + eps).mean()
        
        return loss * self.loss_weight
    
    def _attn_map_loss(self, cross_attn: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        """Cross-attention map与heatmap匹配"""
        B, Q, HW = cross_attn.shape
        _, H, W = heatmap.shape
        
        # Resize heatmap to match attention map spatial size
        # 假设attention是在某个尺度上
        heatmap_flat = F.adaptive_avg_pool2d(
            heatmap.unsqueeze(1), 
            output_size=(int(HW**0.5), int(HW**0.5))
        ).flatten(1)  # [B, HW']
        
        # 只用pseudo queries对应的attention
        K = min(Q, 100)  # 假设前K个是pseudo queries
        cross_attn = cross_attn[:, :K]  # [B, K, HW]
        
        # 归一化
        cross_attn = F.softmax(cross_attn, dim=-1)
        heatmap_flat = F.softmax(heatmap_flat, dim=-1).unsqueeze(1).expand(-1, K, -1)
        
        # KL散度
        loss = F.kl_div(
            cross_attn.log(), heatmap_flat,
            reduction='batchmean'
        )
        
        return loss * self.loss_weight


class PseudoQueryCriterion(nn.Module):
    """
    整合所有Q-Use的loss
    """
    
    def __init__(
        self,
        use_alignment_loss: bool = True,
        use_prior_loss: bool = False,
        alignment_config: Optional[Dict] = None,
        prior_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.use_alignment = use_alignment_loss
        self.use_prior = use_prior_loss
        
        if use_alignment_loss:
            alignment_config = alignment_config or {}
            self.alignment_loss = QueryAlignmentLoss(**alignment_config)
        
        if use_prior_loss:
            prior_config = prior_config or {}
            self.prior_loss = AttentionPriorLoss(**prior_config)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        pseudo_queries: Dict[str, torch.Tensor],
        heatmap: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有pseudo query相关的loss
        
        Args:
            outputs: model输出
                - decoder_queries: [B, Q, d] (从decoder第一层提取)
                - pred_boxes: [B, Q, 4]
                - cross_attn_weights: [B, Q, HW] (可选)
            pseudo_queries: HeatmapQueryGenerator的输出
            heatmap: [B, H, W] 原始热图
            
        Returns:
            losses dict
        """
        losses = {}
        
        if self.use_alignment and 'decoder_queries' in outputs:
            losses['loss_alignment'] = self.alignment_loss(
                outputs['decoder_queries'],
                pseudo_queries['query_content']
            )
        
        if self.use_prior and heatmap is not None:
            losses['loss_prior'] = self.prior_loss(
                outputs['pred_boxes'],
                heatmap,
                outputs.get('cross_attn_weights'),
                pseudo_queries.get('reference_points')
            )
        
        return losses


# ============ 测试代码 ============
if __name__ == '__main__':
    print("Testing QueryMixer...")
    
    B, K, d = 2, 100, 256
    
    # 模拟pseudo queries
    pseudo_queries = {
        'query_embed': torch.randn(B, K, d * 2),
        'query_content': torch.randn(B, K, d),
        'query_pos': torch.randn(B, K, d),
        'reference_points': torch.rand(B, K, 2),
    }
    
    # 测试不同mix模式
    for mode in ['replace', 'concat', 'ratio', 'attention']:
        print(f"\nTesting mode: {mode}")
        mixer = QueryMixer(
            hidden_dim=d,
            num_learnable_queries=300,
            num_pseudo_queries=K,
            mix_mode=mode
        )
        
        mixed_embed, mixed_ref = mixer(pseudo_queries, batch_size=B)
        print(f"  mixed_embed shape: {mixed_embed.shape}")
        if mixed_ref is not None:
            print(f"  mixed_ref shape: {mixed_ref.shape}")
    
    print("\n\nTesting QueryAlignmentLoss...")
    
    decoder_queries = torch.randn(B, K, d)
    pseudo_content = torch.randn(B, K, d)
    
    for loss_type in ['l2', 'cosine', 'infonce']:
        loss_fn = QueryAlignmentLoss(loss_type=loss_type)
        loss = loss_fn(decoder_queries, pseudo_content)
        print(f"  {loss_type} loss: {loss.item():.4f}")
    
    print("\n\nTesting AttentionPriorLoss...")
    
    pred_boxes = torch.rand(B, K, 4)  # cx, cy, w, h
    heatmap = torch.rand(B, 32, 32)
    
    prior_loss = AttentionPriorLoss(loss_type='center')
    loss = prior_loss(pred_boxes, heatmap)
    print(f"  center prior loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")

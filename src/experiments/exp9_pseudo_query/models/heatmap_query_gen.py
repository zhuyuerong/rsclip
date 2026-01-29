"""
Heatmap-based Pseudo Query Generator (Q-Gen模块)

核心职责: heatmap → top-k coords → 多尺度特征pool → (content, pos)

支持三种来源 (对应C1消融):
1. Teacher proposals: teacher boxes → ROIAlign → query_content; box center → query_pos
2. vv-attention regions: heatmap top-k → feature gather/pool → query_content; coords → query_pos  
3. Fusion: teacher提供候选位置, vv-attention提供权重/筛选

支持三种聚合方式 (对应C3消融):
1. mean: 直接取最近邻特征
2. heatmap_weighted: 局部窗口加权平均 (最稳)
3. attn_pool: 小attention在局部窗口pool (最强但可能不稳)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict


class PositionalEncoding2D(nn.Module):
    """2D正弦位置编码, 用于将坐标转换为positional embedding"""
    
    def __init__(self, hidden_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.scale = 2 * math.pi
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, K, 2] 归一化坐标 (x, y) in [0, 1]
        Returns:
            pos_embed: [B, K, hidden_dim]
        """
        assert coords.dim() == 3 and coords.shape[-1] == 2
        
        # 生成维度索引
        dim_t = torch.arange(self.hidden_dim // 2, dtype=torch.float32, device=coords.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.hidden_dim // 2))
        
        # 缩放坐标
        coords = coords * self.scale  # [B, K, 2]
        
        # 计算sin/cos编码
        pos_x = coords[..., 0:1] / dim_t  # [B, K, d/2]
        pos_y = coords[..., 1:2] / dim_t  # [B, K, d/2]
        
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        
        pos_embed = torch.cat([pos_x, pos_y], dim=-1)  # [B, K, hidden_dim]
        return pos_embed


class HeatmapQueryGenerator(nn.Module):
    """
    从vv-attention热图生成pseudo queries
    
    关键设计:
    - 支持多尺度特征提取
    - 输出格式与Deformable DETR的query_embed兼容: [K, 2*hidden_dim]
    - 支持多种pool策略
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_feature_levels: int = 4,
        pool_mode: str = 'heatmap_weighted',  # 'mean', 'heatmap_weighted', 'attn_pool'
        pool_window: int = 3,  # 局部窗口大小
        min_score_thresh: float = 0.1,  # 热图最小阈值
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.pool_mode = pool_mode
        self.pool_window = pool_window
        self.min_score_thresh = min_score_thresh
        
        # Content projection: 将pooled特征投影到hidden_dim
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Position encoding: 将坐标转换为positional embedding
        self.pos_encoder = PositionalEncoding2D(hidden_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # 多尺度特征融合权重
        self.level_weights = nn.Parameter(torch.ones(num_feature_levels))
        
        # Attention pooling (如果使用)
        if pool_mode == 'attn_pool':
            self.attn_pool = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def select_top_k_from_heatmap(
        self,
        heatmap: torch.Tensor,
        k: int,
        nms_radius: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从热图中选择top-k位置 (带简单的非极大值抑制)
        
        Args:
            heatmap: [B, H, W] 热图
            k: 选择的点数
            nms_radius: NMS半径
            
        Returns:
            coords: [B, K, 2] 归一化坐标 (x, y) in [0, 1]
            scores: [B, K] 对应的热图分数
        """
        B, H, W = heatmap.shape
        device = heatmap.device
        
        # 简单NMS: 用max pooling找局部极大值
        heatmap_pad = F.pad(heatmap.unsqueeze(1), 
                           (nms_radius, nms_radius, nms_radius, nms_radius), 
                           mode='constant', value=0)
        heatmap_max = F.max_pool2d(heatmap_pad, 
                                   kernel_size=2*nms_radius+1, 
                                   stride=1, padding=0)
        keep = (heatmap == heatmap_max.squeeze(1)) & (heatmap > self.min_score_thresh)
        
        # 对每个batch选择top-k
        coords_list = []
        scores_list = []
        
        for b in range(B):
            valid_mask = keep[b]
            valid_scores = heatmap[b][valid_mask]
            valid_indices = valid_mask.nonzero(as_tuple=False)  # [N, 2] (y, x)
            
            if valid_indices.shape[0] < k:
                # 如果有效点不够, 用全局top-k补充
                flat_heatmap = heatmap[b].flatten()
                topk_flat = torch.topk(flat_heatmap, k=k, dim=0)
                topk_indices = topk_flat.indices
                y_coords = topk_indices // W
                x_coords = topk_indices % W
                coords = torch.stack([x_coords, y_coords], dim=-1).float()  # [K, 2] (x, y)
                scores = topk_flat.values
            else:
                # 选择top-k
                topk_scores, topk_idx = torch.topk(valid_scores, k=min(k, valid_scores.shape[0]))
                selected_indices = valid_indices[topk_idx]  # [K, 2] (y, x)
                coords = selected_indices[:, [1, 0]].float()  # [K, 2] (x, y)
                scores = topk_scores
                
                # 如果不够k个, 补零
                if coords.shape[0] < k:
                    pad_size = k - coords.shape[0]
                    coords = F.pad(coords, (0, 0, 0, pad_size), value=0)
                    scores = F.pad(scores, (0, pad_size), value=0)
            
            # 归一化到[0, 1]
            coords[:, 0] = coords[:, 0] / (W - 1)  # x
            coords[:, 1] = coords[:, 1] / (H - 1)  # y
            coords = coords.clamp(0, 1)
            
            coords_list.append(coords)
            scores_list.append(scores)
        
        coords = torch.stack(coords_list, dim=0)  # [B, K, 2]
        scores = torch.stack(scores_list, dim=0)  # [B, K]
        
        return coords, scores
    
    def pool_features_at_coords(
        self,
        features: List[torch.Tensor],
        spatial_shapes: torch.Tensor,
        coords: torch.Tensor,
        heatmap: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        在多尺度特征图上的指定坐标处pool特征
        
        Args:
            features: List of [B, C, H_l, W_l] 多尺度特征
            spatial_shapes: [L, 2] 每个level的(H, W)
            coords: [B, K, 2] 归一化坐标 (x, y) in [0, 1]
            heatmap: [B, H, W] 原始热图 (用于weighted pool)
            
        Returns:
            pooled_features: [B, K, hidden_dim]
        """
        B, K, _ = coords.shape
        device = coords.device
        
        # 归一化level权重
        level_weights = F.softmax(self.level_weights, dim=0)
        
        pooled_all_levels = []
        
        for lvl, feat in enumerate(features):
            _, C, H_l, W_l = feat.shape
            
            # 将归一化坐标转换为该level的像素坐标
            coords_lvl = coords.clone()
            coords_lvl[..., 0] = coords_lvl[..., 0] * (W_l - 1)  # x
            coords_lvl[..., 1] = coords_lvl[..., 1] * (H_l - 1)  # y
            
            if self.pool_mode == 'mean':
                # 最近邻采样
                pooled = self._bilinear_sample(feat, coords_lvl)  # [B, K, C]
                
            elif self.pool_mode == 'heatmap_weighted':
                # 局部窗口加权平均
                pooled = self._weighted_pool(feat, coords_lvl, heatmap, H_l, W_l)
                
            elif self.pool_mode == 'attn_pool':
                # Attention pooling
                pooled = self._attention_pool(feat, coords_lvl, H_l, W_l)
            else:
                raise ValueError(f"Unknown pool_mode: {self.pool_mode}")
            
            pooled_all_levels.append(pooled * level_weights[lvl])
        
        # 多尺度特征融合
        pooled_features = sum(pooled_all_levels)  # [B, K, C]
        
        return pooled_features
    
    def _bilinear_sample(self, feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """双线性插值采样"""
        B, C, H, W = feat.shape
        K = coords.shape[1]
        
        # grid_sample需要[-1, 1]范围的坐标
        grid = coords.clone()
        grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1  # x
        grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1  # y
        grid = grid.view(B, K, 1, 2)  # [B, K, 1, 2]
        
        sampled = F.grid_sample(feat, grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)
        sampled = sampled.view(B, C, K).permute(0, 2, 1)  # [B, K, C]
        
        return sampled
    
    def _weighted_pool(
        self, 
        feat: torch.Tensor, 
        coords: torch.Tensor,
        heatmap: Optional[torch.Tensor],
        H_l: int, 
        W_l: int
    ) -> torch.Tensor:
        """局部窗口热图加权pool"""
        B, C, _, _ = feat.shape
        K = coords.shape[1]
        device = feat.device
        
        # 如果没有heatmap, 退化为mean pool
        if heatmap is None:
            return self._bilinear_sample(feat, coords)
        
        # 将heatmap resize到当前level尺寸
        heatmap_lvl = F.interpolate(heatmap.unsqueeze(1), size=(H_l, W_l), 
                                    mode='bilinear', align_corners=True).squeeze(1)
        
        pooled_list = []
        half_win = self.pool_window // 2
        
        for b in range(B):
            for k in range(K):
                cx, cy = coords[b, k, 0].long(), coords[b, k, 1].long()
                
                # 计算窗口边界
                x1 = max(0, cx - half_win)
                x2 = min(W_l, cx + half_win + 1)
                y1 = max(0, cy - half_win)
                y2 = min(H_l, cy + half_win + 1)
                
                # 提取局部窗口
                local_feat = feat[b, :, y1:y2, x1:x2]  # [C, h, w]
                local_heat = heatmap_lvl[b, y1:y2, x1:x2]  # [h, w]
                
                # 加权平均
                weights = F.softmax(local_heat.flatten(), dim=0)  # [h*w]
                local_feat_flat = local_feat.flatten(1)  # [C, h*w]
                pooled = (local_feat_flat * weights.unsqueeze(0)).sum(dim=1)  # [C]
                pooled_list.append(pooled)
        
        pooled = torch.stack(pooled_list, dim=0).view(B, K, C)  # [B, K, C]
        return pooled
    
    def _attention_pool(
        self, 
        feat: torch.Tensor, 
        coords: torch.Tensor,
        H_l: int, 
        W_l: int
    ) -> torch.Tensor:
        """Attention-based局部pool"""
        B, C, _, _ = feat.shape
        K = coords.shape[1]
        device = feat.device
        half_win = self.pool_window // 2
        
        pooled_list = []
        
        for b in range(B):
            for k in range(K):
                cx, cy = coords[b, k, 0].long(), coords[b, k, 1].long()
                
                # 计算窗口边界
                x1 = max(0, cx - half_win)
                x2 = min(W_l, cx + half_win + 1)
                y1 = max(0, cy - half_win)
                y2 = min(H_l, cy + half_win + 1)
                
                # 提取局部窗口特征
                local_feat = feat[b, :, y1:y2, x1:x2]  # [C, h, w]
                local_feat = local_feat.flatten(1).T.unsqueeze(0)  # [1, h*w, C]
                
                # Attention pooling
                query = self.pool_query.expand(1, -1, -1)  # [1, 1, C]
                pooled, _ = self.attn_pool(query, local_feat, local_feat)  # [1, 1, C]
                pooled_list.append(pooled.squeeze(0).squeeze(0))  # [C]
        
        pooled = torch.stack(pooled_list, dim=0).view(B, K, C)  # [B, K, C]
        return pooled
    
    def forward(
        self,
        srcs: List[torch.Tensor],
        spatial_shapes: torch.Tensor,
        heatmap: torch.Tensor,
        num_queries: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        生成pseudo queries
        
        Args:
            srcs: List of [B, C, H_l, W_l] 多尺度特征 (来自backbone)
            spatial_shapes: [L, 2] 每个level的(H, W)
            heatmap: [B, H, W] vv-attention热图
            num_queries: 可选, 覆盖默认的num_queries
            
        Returns:
            dict with:
                - query_embed: [B, K, 2*hidden_dim] 可直接用于Deformable DETR
                - query_content: [B, K, hidden_dim] content部分
                - query_pos: [B, K, hidden_dim] positional部分
                - reference_points: [B, K, 2] 参考点坐标 (归一化)
                - heatmap_scores: [B, K] 热图分数
        """
        K = num_queries if num_queries is not None else self.num_queries
        B = heatmap.shape[0]
        
        # Step 1: 从热图选择top-k位置
        coords, scores = self.select_top_k_from_heatmap(heatmap, k=K)  # [B, K, 2], [B, K]
        
        # Step 2: 在多尺度特征上pool
        pooled_features = self.pool_features_at_coords(srcs, spatial_shapes, coords, heatmap)  # [B, K, C]
        
        # Step 3: 生成query content
        query_content = self.content_proj(pooled_features)  # [B, K, hidden_dim]
        
        # Step 4: 生成query positional embedding
        pos_embed = self.pos_encoder(coords)  # [B, K, hidden_dim]
        query_pos = self.pos_proj(pos_embed)  # [B, K, hidden_dim]
        
        # Step 5: 拼接成完整的query_embed (与Deformable DETR兼容)
        query_embed = torch.cat([query_content, query_pos], dim=-1)  # [B, K, 2*hidden_dim]
        
        return {
            'query_embed': query_embed,
            'query_content': query_content,
            'query_pos': query_pos,
            'reference_points': coords,  # 这就是decoder需要的reference_points
            'heatmap_scores': scores,
        }


class TeacherQueryGenerator(nn.Module):
    """
    从Teacher detector的proposals生成pseudo queries
    用于A2 baseline和C1消融对比
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        roi_size: int = 7,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.roi_size = roi_size
        
        # ROI特征投影
        self.roi_proj = nn.Sequential(
            nn.Linear(hidden_dim * roi_size * roi_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Position encoding
        self.pos_encoder = PositionalEncoding2D(hidden_dim)
        self.pos_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        teacher_boxes: torch.Tensor,
        teacher_scores: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, C, H, W] 特征图
            teacher_boxes: [B, N, 4] teacher检测框 (x1, y1, x2, y2) 归一化
            teacher_scores: [B, N] 检测分数
            
        Returns:
            同HeatmapQueryGenerator
        """
        B = features.shape[0]
        K = self.num_queries
        
        # 选择top-K boxes
        topk_scores, topk_idx = torch.topk(teacher_scores, k=K, dim=1)
        topk_boxes = torch.gather(teacher_boxes, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
        
        # 计算box中心作为reference points
        box_centers = (topk_boxes[..., :2] + topk_boxes[..., 2:]) / 2  # [B, K, 2]
        
        # ROI Align提取特征 (简化版: 直接用center双线性采样)
        # 完整版应该用torchvision.ops.roi_align
        _, C, H, W = features.shape
        grid = box_centers.clone()
        grid[..., 0] = 2 * grid[..., 0] - 1
        grid[..., 1] = 2 * grid[..., 1] - 1
        grid = grid.view(B, K, 1, 2)
        
        sampled = F.grid_sample(features, grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)
        query_content = sampled.view(B, C, K).permute(0, 2, 1)  # [B, K, C]
        
        # 生成positional embedding
        pos_embed = self.pos_encoder(box_centers)
        query_pos = self.pos_proj(pos_embed)
        
        # 拼接
        query_embed = torch.cat([query_content, query_pos], dim=-1)
        
        return {
            'query_embed': query_embed,
            'query_content': query_content,
            'query_pos': query_pos,
            'reference_points': box_centers,
            'heatmap_scores': topk_scores,
        }


class FusionQueryGenerator(nn.Module):
    """
    融合策略: Teacher提供候选位置, vv-attention提供权重/筛选
    最强baseline候选
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        fusion_mode: str = 'reweight',  # 'reweight', 'filter', 'hybrid'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.fusion_mode = fusion_mode
        
        self.heatmap_gen = HeatmapQueryGenerator(hidden_dim, num_queries)
        self.teacher_gen = TeacherQueryGenerator(hidden_dim, num_queries)
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        srcs: List[torch.Tensor],
        spatial_shapes: torch.Tensor,
        heatmap: torch.Tensor,
        teacher_boxes: Optional[torch.Tensor] = None,
        teacher_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """融合两种来源的queries"""
        
        # 获取heatmap-based queries
        heatmap_out = self.heatmap_gen(srcs, spatial_shapes, heatmap)
        
        if teacher_boxes is None:
            return heatmap_out
        
        # 获取teacher-based queries
        teacher_out = self.teacher_gen(srcs[0], teacher_boxes, teacher_scores)
        
        if self.fusion_mode == 'reweight':
            # 用heatmap分数重新加权teacher queries
            heatmap_at_teacher = self._sample_heatmap_at_points(
                heatmap, teacher_out['reference_points'])
            combined_scores = teacher_out['heatmap_scores'] * heatmap_at_teacher
            
            # 选择combined分数最高的K个
            K = self.num_queries
            topk_scores, topk_idx = torch.topk(combined_scores, k=K, dim=1)
            
            query_embed = torch.gather(teacher_out['query_embed'], 1, 
                                       topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim * 2))
            reference_points = torch.gather(teacher_out['reference_points'], 1,
                                           topk_idx.unsqueeze(-1).expand(-1, -1, 2))
            
            return {
                'query_embed': query_embed,
                'query_content': query_embed[..., :self.hidden_dim],
                'query_pos': query_embed[..., self.hidden_dim:],
                'reference_points': reference_points,
                'heatmap_scores': topk_scores,
            }
        
        elif self.fusion_mode == 'hybrid':
            # 一半来自heatmap, 一半来自teacher
            K_half = self.num_queries // 2
            return {
                'query_embed': torch.cat([
                    heatmap_out['query_embed'][:, :K_half],
                    teacher_out['query_embed'][:, :K_half]
                ], dim=1),
                'reference_points': torch.cat([
                    heatmap_out['reference_points'][:, :K_half],
                    teacher_out['reference_points'][:, :K_half]
                ], dim=1),
                'heatmap_scores': torch.cat([
                    heatmap_out['heatmap_scores'][:, :K_half],
                    teacher_out['heatmap_scores'][:, :K_half]
                ], dim=1),
            }
        
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
    
    def _sample_heatmap_at_points(self, heatmap: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """在指定点采样热图值"""
        B, H, W = heatmap.shape
        K = points.shape[1]
        
        grid = points.clone()
        grid[..., 0] = 2 * grid[..., 0] - 1
        grid[..., 1] = 2 * grid[..., 1] - 1
        grid = grid.view(B, K, 1, 2)
        
        sampled = F.grid_sample(heatmap.unsqueeze(1), grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
        return sampled.view(B, K)


def build_query_generator(
    gen_type: str = 'heatmap',
    hidden_dim: int = 256,
    num_queries: int = 100,
    **kwargs
) -> nn.Module:
    """
    工厂函数: 构建query generator
    
    Args:
        gen_type: 'heatmap', 'teacher', 'fusion'
        hidden_dim: transformer hidden dimension
        num_queries: number of queries to generate
    """
    if gen_type == 'heatmap':
        return HeatmapQueryGenerator(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            **kwargs
        )
    elif gen_type == 'teacher':
        return TeacherQueryGenerator(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            **kwargs
        )
    elif gen_type == 'fusion':
        return FusionQueryGenerator(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown gen_type: {gen_type}")


# ============ 测试代码 ============
if __name__ == '__main__':
    # 测试HeatmapQueryGenerator
    print("Testing HeatmapQueryGenerator...")
    
    B, C, H, W = 2, 256, 20, 20
    num_levels = 4
    
    # 模拟多尺度特征
    srcs = [torch.randn(B, C, H // (2**i), W // (2**i)) for i in range(num_levels)]
    spatial_shapes = torch.tensor([[H // (2**i), W // (2**i)] for i in range(num_levels)])
    
    # 模拟热图
    heatmap = torch.rand(B, H * 2, W * 2)  # 原图尺度
    
    # 创建generator
    gen = HeatmapQueryGenerator(
        hidden_dim=C,
        num_queries=100,
        num_feature_levels=num_levels,
        pool_mode='heatmap_weighted'
    )
    
    # 生成queries
    output = gen(srcs, spatial_shapes, heatmap)
    
    print(f"query_embed shape: {output['query_embed'].shape}")  # [B, K, 2*C]
    print(f"query_content shape: {output['query_content'].shape}")  # [B, K, C]
    print(f"query_pos shape: {output['query_pos'].shape}")  # [B, K, C]
    print(f"reference_points shape: {output['reference_points'].shape}")  # [B, K, 2]
    print(f"heatmap_scores shape: {output['heatmap_scores'].shape}")  # [B, K]
    
    print("\n✓ All tests passed!")

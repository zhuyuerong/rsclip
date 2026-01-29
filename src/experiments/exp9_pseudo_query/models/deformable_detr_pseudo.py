"""
Deformable DETR with Pseudo Query Injection

基于官方Deformable DETR修改, 支持:
1. 从heatmap生成pseudo queries
2. 多种query混合策略
3. 额外的alignment/prior loss

关键修改点:
- forward()中的query_embed可以来自pseudo或learnable
- 添加heatmap输入接口
- 添加pseudo query相关的loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import copy
import math

# 导入自定义模块
from .heatmap_query_gen import HeatmapQueryGenerator, build_query_generator
from .query_injection import QueryMixer, PseudoQueryCriterion


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    """简单的多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDETRPseudo(nn.Module):
    """
    Deformable DETR with Pseudo Query Support
    
    相比原版的关键修改:
    1. 添加HeatmapQueryGenerator模块
    2. 添加QueryMixer模块
    3. forward()接受heatmap输入
    4. 支持返回decoder intermediate queries (用于alignment loss)
    """
    
    def __init__(
        self,
        backbone,
        transformer,
        num_classes: int,
        num_queries: int = 300,
        num_feature_levels: int = 4,
        aux_loss: bool = True,
        with_box_refine: bool = False,
        two_stage: bool = False,
        # Pseudo query相关参数
        use_pseudo_query: bool = True,
        pseudo_query_config: Optional[Dict] = None,
        query_mix_mode: str = 'concat',
        num_pseudo_queries: int = 100,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        
        # 检测头
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        
        # 原始learnable query (two_stage时不需要)
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        
        # Input projection
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        
        # ========== Pseudo Query相关 ==========
        self.use_pseudo_query = use_pseudo_query
        self.num_pseudo_queries = num_pseudo_queries
        
        if use_pseudo_query:
            # Query生成器
            pseudo_query_config = pseudo_query_config or {}
            self.query_generator = HeatmapQueryGenerator(
                hidden_dim=hidden_dim,
                num_queries=num_pseudo_queries,
                num_feature_levels=num_feature_levels,
                **pseudo_query_config
            )
            
            # Query混合器
            self.query_mixer = QueryMixer(
                hidden_dim=hidden_dim,
                num_learnable_queries=num_queries,
                num_pseudo_queries=num_pseudo_queries,
                mix_mode=query_mix_mode,
            )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.class_embed.out_features) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        
        # Box refinement设置
        num_pred = (self.transformer.decoder.num_layers + 1) if self.two_stage else self.transformer.decoder.num_layers
        
        if self.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        
        if self.two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
    
    def forward(
        self,
        samples,
        heatmap: Optional[torch.Tensor] = None,
        return_intermediate_queries: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            samples: NestedTensor (images + masks)
            heatmap: [B, H, W] vv-attention热图 (可选)
            return_intermediate_queries: 是否返回decoder中间层的queries (用于alignment loss)
            
        Returns:
            outputs dict:
                - pred_logits: [B, Q, num_classes]
                - pred_boxes: [B, Q, 4]
                - aux_outputs: list of intermediate outputs
                - pseudo_queries: dict (如果使用pseudo query)
                - decoder_queries: [B, Q, d] (如果return_intermediate_queries)
        """
        # ========== Backbone特征提取 ==========
        if not hasattr(samples, 'decompose'):
            from util.misc import NestedTensor, nested_tensor_from_tensor_list
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        
        # 额外的feature levels
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        # ========== Query准备 ==========
        bs = srcs[0].shape[0]
        pseudo_queries_output = None
        
        if self.use_pseudo_query and heatmap is not None:
            # 从热图生成pseudo queries
            spatial_shapes = torch.tensor(
                [[src.shape[2], src.shape[3]] for src in srcs],
                device=srcs[0].device
            )
            
            pseudo_queries_output = self.query_generator(
                srcs, spatial_shapes, heatmap, 
                num_queries=self.num_pseudo_queries
            )
            
            # 混合pseudo和learnable queries
            query_embeds, pseudo_reference = self.query_mixer(
                pseudo_queries_output, batch_size=bs
            )
        else:
            # 纯learnable queries
            query_embeds = self.query_embed.weight if not self.two_stage else None
            pseudo_reference = None
        
        # ========== Transformer ==========
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embeds)
        
        # ========== 预测头 ==========
        outputs_classes = []
        outputs_coords = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = self._inverse_sigmoid(reference)
            
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        
        # ========== 输出组装 ==========
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {
                'pred_logits': enc_outputs_class,
                'pred_boxes': enc_outputs_coord
            }
        
        # Pseudo query相关输出
        if pseudo_queries_output is not None:
            out['pseudo_queries'] = pseudo_queries_output
        
        # 返回decoder中间查询 (用于alignment loss)
        if return_intermediate_queries:
            out['decoder_queries'] = hs[0]  # 第一层decoder输出
        
        return out
    
    @staticmethod
    def _inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DeformableDETRPseudoCriterion(nn.Module):
    """
    完整的loss计算模块
    
    包含:
    1. 原始Deformable DETR的detection loss (classification + box)
    2. Pseudo query alignment loss
    3. Attention prior loss
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher,
        weight_dict: Dict[str, float],
        losses: List[str],
        focal_alpha: float = 0.25,
        # Pseudo query loss配置
        use_alignment_loss: bool = True,
        use_prior_loss: bool = False,
        alignment_weight: float = 1.0,
        prior_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        
        # Pseudo query criterion
        self.pseudo_criterion = PseudoQueryCriterion(
            use_alignment_loss=use_alignment_loss,
            use_prior_loss=use_prior_loss,
            alignment_config={'loss_weight': alignment_weight},
            prior_config={'loss_weight': prior_weight},
        )
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        heatmap: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有loss
        
        Args:
            outputs: model输出
            targets: ground truth
            heatmap: 原始热图 (用于prior loss)
            
        Returns:
            losses dict
        """
        # 原始detection loss (这里简化,实际需要完整实现)
        losses = {}
        
        # TODO: 添加完整的detection loss计算
        # losses.update(self._compute_detection_loss(outputs, targets))
        
        # Pseudo query loss
        if 'pseudo_queries' in outputs:
            pseudo_losses = self.pseudo_criterion(
                outputs,
                outputs['pseudo_queries'],
                heatmap
            )
            losses.update(pseudo_losses)
        
        return losses


def build_deformable_detr_pseudo(args):
    """
    构建DeformableDETRPseudo模型
    
    需要的args:
    - hidden_dim: transformer hidden dimension
    - num_queries: total number of queries
    - num_feature_levels: number of feature levels
    - aux_loss: whether to use auxiliary loss
    - with_box_refine: whether to use iterative box refinement
    - two_stage: whether to use two-stage
    - use_pseudo_query: whether to use pseudo query
    - num_pseudo_queries: number of pseudo queries
    - query_mix_mode: 'concat', 'replace', 'ratio', 'attention'
    - pseudo_pool_mode: 'mean', 'heatmap_weighted', 'attn_pool'
    """
    # 这里需要导入backbone和transformer builder
    # from models.backbone import build_backbone
    # from models.deformable_transformer import build_deforamble_transformer
    
    # backbone = build_backbone(args)
    # transformer = build_deforamble_transformer(args)
    
    # 暂时返回配置信息
    config = {
        'hidden_dim': getattr(args, 'hidden_dim', 256),
        'num_queries': getattr(args, 'num_queries', 300),
        'num_feature_levels': getattr(args, 'num_feature_levels', 4),
        'use_pseudo_query': getattr(args, 'use_pseudo_query', True),
        'num_pseudo_queries': getattr(args, 'num_pseudo_queries', 100),
        'query_mix_mode': getattr(args, 'query_mix_mode', 'concat'),
        'pseudo_query_config': {
            'pool_mode': getattr(args, 'pseudo_pool_mode', 'heatmap_weighted'),
            'pool_window': getattr(args, 'pseudo_pool_window', 3),
        }
    }
    
    return config


# ============ 测试代码 ============
if __name__ == '__main__':
    print("DeformableDETRPseudo module loaded successfully!")
    print("\n关键组件:")
    print("  - HeatmapQueryGenerator: heatmap → pseudo queries")
    print("  - QueryMixer: 混合pseudo和learnable queries")
    print("  - PseudoQueryCriterion: alignment + prior loss")
    print("\n使用方法:")
    print("  1. 构建model = DeformableDETRPseudo(...)")
    print("  2. 准备heatmap (来自vv-attention)")
    print("  3. outputs = model(images, heatmap=heatmap)")
    print("  4. losses = criterion(outputs, targets, heatmap)")

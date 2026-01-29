#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pseudo Query 训练脚本 (A2/A3实验)

支持:
- A2: Teacher proposals → pseudo query (管线自检)
- A3: Heatmap → pseudo query (核心方法)

与A0的差异:
- 注入pseudo queries到Deformable DETR decoder
- 支持多种Q-Gen和Q-Use策略
"""

import argparse
import datetime
import json
import os
import sys
import time
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'external' / 'Deformable-DETR'))

# Deformable DETR imports
from models import build_model
from models.deformable_detr import SetCriterion
from models.matcher import build_matcher
import util.misc as utils

# 使用Deformable DETR的NestedTensor
from util.misc import NestedTensor, nested_tensor_from_tensor_list

# 本地模块
from src.experiments.exp9_pseudo_query.datasets import (
    build_dior_dataset,
    build_dior_with_heatmap,
    DIOR_CLASSES,
)
from src.experiments.exp9_pseudo_query.models.heatmap_query_gen import (
    HeatmapQueryGenerator,
    TeacherQueryGenerator,
)
from src.experiments.exp9_pseudo_query.models.query_injection import (
    QueryMixer,
    QueryAlignmentLoss,
    AttentionPriorLoss,
)

# 简单日志
HAS_TENSORBOARD = False


def get_args_parser():
    parser = argparse.ArgumentParser('Pseudo Query Training', add_help=False)
    
    # 实验类型
    parser.add_argument('--exp_type', default='A3', choices=['A2', 'A3', 'B1', 'B2'],
                        help='A2=teacher, A3=heatmap, B1=random, B2=shuffled')
    
    # 训练参数
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    
    # 模型参数
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    
    # Deformable DETR变体
    parser.add_argument('--with_box_refine', action='store_true')
    parser.add_argument('--two_stage', action='store_true')
    
    # Pseudo Query参数 (核心)
    parser.add_argument('--num_pseudo_queries', default=100, type=int,
                        help='pseudo query数量 (K)')
    parser.add_argument('--num_learnable_queries', default=200, type=int,
                        help='learnable query数量 (总queries = pseudo + learnable)')
    parser.add_argument('--pool_mode', default='heatmap_weighted', 
                        choices=['mean', 'heatmap_weighted', 'attn_pool'],
                        help='特征池化方式')
    parser.add_argument('--mix_mode', default='concat',
                        choices=['replace', 'concat', 'ratio', 'attention'],
                        help='query混合方式')
    
    # Loss参数
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Pseudo Query额外loss
    parser.add_argument('--use_align_loss', action='store_true',
                        help='使用query alignment loss')
    parser.add_argument('--align_loss_type', default='l2', choices=['l2', 'cosine', 'infonce'])
    parser.add_argument('--align_loss_coef', default=1.0, type=float)
    parser.add_argument('--use_prior_loss', action='store_true',
                        help='使用attention prior loss')
    parser.add_argument('--prior_loss_type', default='center', choices=['center', 'attn_map'])
    parser.add_argument('--prior_loss_coef', default=0.5, type=float)
    
    # 数据集参数
    parser.add_argument('--dior_path', default='datasets/DIOR', type=str)
    parser.add_argument('--num_classes', default=20, type=int)
    parser.add_argument('--image_size', default=800, type=int)
    parser.add_argument('--dataset_file', default='dior', type=str)
    
    # 热图相关
    parser.add_argument('--heatmap_cache_dir', default='outputs/heatmap_cache/dior_trainval', type=str)
    parser.add_argument('--checkpoint_path', default='checkpoints/RemoteCLIP-ViT-B-32.pt', type=str)
    parser.add_argument('--generate_heatmap_on_fly', action='store_true',
                        help='在线生成热图 (慢但省空间)')
    
    # 运行参数
    parser.add_argument('--output_dir', default='outputs/exp9_pseudo_query', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_epochs', default=[1, 5, 10, 20, 30, 40, 50], type=int, nargs='+')
    parser.add_argument('--save_epochs', default=[10, 20, 30, 40, 50], type=int, nargs='+')
    
    # Deformable DETR兼容参数
    parser.add_argument('--masks', default=False, action='store_true')
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    
    return parser


class DeformableDETRWithPseudoQuery(nn.Module):
    """
    带Pseudo Query的Deformable DETR
    
    核心改动:
    - 将原始query_embed替换为QueryMixer输出
    - 支持热图引导的pseudo query生成
    """
    
    def __init__(self, detr_model, query_gen, query_mixer, args):
        super().__init__()
        self.detr = detr_model
        self.query_gen = query_gen
        self.query_mixer = query_mixer
        self.args = args
        self.exp_type = args.exp_type
        
    def forward(self, samples: NestedTensor, heatmaps=None, targets=None):
        """
        Args:
            samples: NestedTensor (images)
            heatmaps: [B, H, W] 热图 (A3模式)
            targets: 目标标注 (A2模式可能用到GT boxes)
        """
        # 获取backbone特征
        features, pos = self.detr.backbone(samples)
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.detr.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        
        if self.detr.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = samples.mask
                mask = nn.functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        # 生成pseudo queries
        batch_size = srcs[0].shape[0]
        pseudo_queries = None
        
        if self.exp_type == 'A3' and heatmaps is not None and self.query_gen is not None:
            # A3: 从热图生成pseudo queries
            spatial_shapes = torch.tensor([[src.shape[2], src.shape[3]] for src in srcs], 
                                         device=srcs[0].device)
            pseudo_queries = self.query_gen(srcs, spatial_shapes, heatmaps)
            
        elif self.exp_type == 'A2' and targets is not None and self.query_gen is not None:
            # A2: 从teacher proposals (GT boxes作为teacher)
            # 简化版：使用GT boxes作为teacher
            spatial_shapes = torch.tensor([[src.shape[2], src.shape[3]] for src in srcs],
                                         device=srcs[0].device)
            # 从targets提取boxes (需要归一化到[0,1])
            teacher_boxes = [t['boxes'] for t in targets]  # list of [N, 4]
            pseudo_queries = self.query_gen(srcs, spatial_shapes, teacher_boxes)
            
        elif self.exp_type == 'B1':
            # B1: 随机queries
            pseudo_queries = self._generate_random_queries(batch_size, srcs[0].device)
            
        elif self.exp_type == 'B2' and heatmaps is not None and self.query_gen is not None:
            # B2: 打乱热图-图像对应
            shuffled_heatmaps = heatmaps[torch.randperm(heatmaps.shape[0])]
            spatial_shapes = torch.tensor([[src.shape[2], src.shape[3]] for src in srcs],
                                         device=srcs[0].device)
            pseudo_queries = self.query_gen(srcs, spatial_shapes, shuffled_heatmaps)
        
        # 混合queries
        query_embed, ref_points = self.query_mixer(pseudo_queries, batch_size)
        
        # Deformable Transformer forward
        # query_embed: [B, Q, 2*d] -> 需要变换格式
        query_embed = query_embed.transpose(0, 1)  # [Q, B, 2*d] (Deformable DETR期望的格式)
        
        # 准备其他输入
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes_list = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_list.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # [B, HW, C]
            mask = mask.flatten(1)  # [B, HW]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [B, HW, C]
            lvl_pos_embed = pos_embed + self.detr.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self._get_valid_ratio(m) for m in masks], 1)
        
        # Transformer forward
        memory = self.detr.transformer.encoder(
            src_flatten, spatial_shapes, level_start_index, valid_ratios,
            lvl_pos_embed_flatten, mask_flatten
        )
        
        # Split query_embed
        hidden_dim = self.detr.transformer.d_model
        query_embed_2d = query_embed.transpose(0, 1)  # [B, Q, 2*d]
        tgt = query_embed_2d[..., :hidden_dim]  # [B, Q, d]
        query_pos = query_embed_2d[..., hidden_dim:]  # [B, Q, d]
        
        # Reference points
        if ref_points is not None:
            reference_points = ref_points  # [B, Q, 2]
        else:
            reference_points = self.detr.transformer.reference_points(query_pos).sigmoid()
        init_reference = reference_points
        
        # Decoder forward
        hs, inter_references = self.detr.transformer.decoder(
            tgt.transpose(0, 1), reference_points, memory, spatial_shapes,
            level_start_index, valid_ratios, query_pos.transpose(0, 1), mask_flatten
        )
        
        # Output heads
        inter_references_out = inter_references
        
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = self.detr.transformer.inverse_sigmoid(reference)
            outputs_class = self.detr.class_embed[lvl](hs[lvl])
            tmp = self.detr.bbox_embed[lvl](hs[lvl])
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
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)
        
        # 额外返回pseudo_queries用于loss
        out['pseudo_queries'] = pseudo_queries
        out['decoder_queries'] = hs[-1]  # 最后一层decoder输出
        
        return out
    
    def _generate_random_queries(self, batch_size, device):
        """生成随机queries (B1实验)"""
        K = self.args.num_pseudo_queries
        d = self.args.hidden_dim
        
        query_content = torch.randn(batch_size, K, d, device=device)
        query_pos = torch.randn(batch_size, K, d, device=device)
        query_embed = torch.cat([query_content, query_pos], dim=-1)
        reference_points = torch.rand(batch_size, K, 2, device=device)
        
        return {
            'query_embed': query_embed,
            'reference_points': reference_points,
            'query_content': query_content,
            'query_pos': query_pos,
        }
    
    def _get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


def collate_fn(batch):
    """Collate函数"""
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def collate_fn_with_heatmap(batch):
    """带热图的Collate函数"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    images = nested_tensor_from_tensor_list(images)
    
    # 处理热图
    heatmaps = None
    if 'heatmap' in targets[0]:
        heatmaps = torch.stack([t['heatmap'] for t in targets])
    
    return images, targets, heatmaps


def build_criterion(args, device):
    """构建损失函数"""
    matcher = build_matcher(args)
    
    weight_dict = {
        'loss_ce': args.cls_loss_coef, 
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef
    }
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(
        args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha
    )
    criterion.to(device)
    
    return criterion, weight_dict


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm, args):
    """训练一个epoch"""
    model.train()
    criterion.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 50
    
    # 根据实验类型选择数据格式
    needs_heatmap = args.exp_type in ['A3', 'B2']
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        if needs_heatmap and len(batch_data) == 3:
            samples, targets, heatmaps = batch_data
        else:
            samples, targets = batch_data[:2]
            heatmaps = None
        
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        if heatmaps is not None:
            heatmaps = heatmaps.to(device)
        
        # 前向传播
        outputs = model(samples, heatmaps=heatmaps, targets=targets)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # 检查loss
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        
        optimizer.step()
        
        # 记录
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        
        for k, v in loss_dict.items():
            if k in weight_dict:
                metric_logger.update(**{k: v.item()})
    
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch, args):
    """评估模型"""
    model.eval()
    criterion.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    needs_heatmap = args.exp_type in ['A3', 'B2']
    
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    for batch_data in metric_logger.log_every(data_loader, 50, header):
        if needs_heatmap and len(batch_data) == 3:
            samples, targets, heatmaps = batch_data
        else:
            samples, targets = batch_data[:2]
            heatmaps = None
        
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        if heatmaps is not None:
            heatmaps = heatmaps.to(device)
        
        outputs = model(samples, heatmaps=heatmaps, targets=targets)
        
        # 计算loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        for k, v in loss_dict.items():
            if k in weight_dict:
                metric_logger.update(**{k: v.item()})
        
        # 简单的recall计算
        pred_boxes = outputs['pred_boxes']  # [B, Q, 4]
        pred_logits = outputs['pred_logits']  # [B, Q, C]
        
        for b in range(pred_boxes.shape[0]):
            gt_boxes = targets[b]['boxes']
            if len(gt_boxes) == 0:
                continue
            
            # 获取top-100预测
            scores = pred_logits[b].sigmoid().max(dim=-1)[0]  # [Q]
            topk = min(100, scores.shape[0])
            _, topk_idx = scores.topk(topk)
            
            pred_b = pred_boxes[b, topk_idx]  # [100, 4]
            
            # 计算IoU
            iou = box_iou(pred_b, gt_boxes)  # [100, M]
            matched = (iou.max(dim=0)[0] >= 0.5).sum().item()
            
            total_tp += matched
            total_gt += len(gt_boxes)
    
    recall = total_tp / (total_gt + 1e-6)
    
    print("Averaged stats:", metric_logger)
    print(f"  Recall@100 (IoU>0.5): {recall:.4f}")
    
    return {'recall_100': recall}


def box_iou(boxes1, boxes2):
    """计算IoU (cxcywh格式)"""
    # 转换为xyxy
    def cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack([x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2], dim=-1)
    
    boxes1 = cxcywh_to_xyxy(boxes1)
    boxes2 = cxcywh_to_xyxy(boxes2)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def main(args):
    print("=" * 70)
    print(f"Pseudo Query Training: {args.exp_type}")
    print("=" * 70)
    
    device = torch.device(args.device)
    
    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n📁 Output: {output_dir}")
    print(f"🎲 Seed: {seed}")
    print(f"🖥️  Device: {device}")
    print(f"📊 Exp Type: {args.exp_type}")
    
    # ========================================
    # 构建基础模型
    # ========================================
    print("\n📦 Building model...")
    base_model, _, postprocessors = build_model(args)
    base_model.to(device)
    
    # 构建Query Generator
    query_gen = None
    if args.exp_type in ['A3', 'B2']:
        query_gen = HeatmapQueryGenerator(
            hidden_dim=args.hidden_dim,
            num_queries=args.num_pseudo_queries,
            num_feature_levels=args.num_feature_levels,
            pool_mode=args.pool_mode,
        )
        query_gen.to(device)
        print(f"   Query Gen: HeatmapQueryGenerator (K={args.num_pseudo_queries})")
    elif args.exp_type == 'A2':
        query_gen = TeacherQueryGenerator(
            hidden_dim=args.hidden_dim,
            num_queries=args.num_pseudo_queries,
            num_feature_levels=args.num_feature_levels,
        )
        query_gen.to(device)
        print(f"   Query Gen: TeacherQueryGenerator (K={args.num_pseudo_queries})")
    
    # 构建Query Mixer
    query_mixer = QueryMixer(
        hidden_dim=args.hidden_dim,
        num_learnable_queries=args.num_learnable_queries,
        num_pseudo_queries=args.num_pseudo_queries,
        mix_mode=args.mix_mode,
    )
    query_mixer.to(device)
    print(f"   Query Mixer: {args.mix_mode} (pseudo={args.num_pseudo_queries}, learnable={args.num_learnable_queries})")
    
    # 组合模型
    model = DeformableDETRWithPseudoQuery(base_model, query_gen, query_mixer, args)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_parameters:,}")
    
    # ========================================
    # 构建数据集
    # ========================================
    print("\n📊 Building datasets...")
    dior_path = project_root / args.dior_path
    checkpoint_path = project_root / args.checkpoint_path
    heatmap_cache_dir = project_root / args.heatmap_cache_dir
    
    needs_heatmap = args.exp_type in ['A3', 'B2']
    
    if needs_heatmap:
        dataset_train = build_dior_with_heatmap(
            root=str(dior_path),
            image_set='train',
            image_size=args.image_size,
            heatmap_cache_dir=str(heatmap_cache_dir),
            checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
            device=str(device),
            generate_on_fly=args.generate_heatmap_on_fly,
        )
        dataset_val = build_dior_with_heatmap(
            root=str(dior_path),
            image_set='val',
            image_size=args.image_size,
            heatmap_cache_dir=str(heatmap_cache_dir).replace('trainval', 'test'),
            checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
            device=str(device),
            generate_on_fly=args.generate_heatmap_on_fly,
        )
        collate = collate_fn_with_heatmap
    else:
        dataset_train = build_dior_dataset(
            root=str(dior_path),
            image_set='train',
            image_size=args.image_size,
        )
        dataset_val = build_dior_dataset(
            root=str(dior_path),
            image_set='val',
            image_size=args.image_size,
        )
        collate = collate_fn
    
    print(f"   Train: {len(dataset_train)} images")
    print(f"   Val: {len(dataset_val)} images")
    
    # DataLoader
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    
    # ========================================
    # 构建优化器
    # ========================================
    print("\n⚙️  Building optimizer...")
    
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    criterion, weight_dict = build_criterion(args, device)
    
    # ========================================
    # 训练循环
    # ========================================
    print("\n🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    
    start_time = time.time()
    best_recall = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args
        )
        lr_scheduler.step()
        
        # 评估
        if (epoch + 1) in args.eval_epochs or epoch == args.epochs - 1:
            print(f"\n📊 Evaluating epoch {epoch + 1}...")
            val_stats = evaluate(model, criterion, data_loader_val, device, epoch + 1, args)
            
            if val_stats['recall_100'] > best_recall:
                best_recall = val_stats['recall_100']
                checkpoint = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'recall_100': best_recall,
                }
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"   ✅ New best Recall@100: {best_recall:.4f}")
        
        # 保存checkpoint
        if (epoch + 1) in args.save_epochs:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            torch.save(checkpoint, output_dir / f'checkpoint_{epoch + 1:04d}.pth')
        
        # 保存日志
        log_stats = {
            'epoch': epoch + 1,
            **{f'train_{k}': v for k, v in train_stats.items()},
        }
        with open(output_dir / 'log.txt', 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ Training completed! Time: {datetime.timedelta(seconds=int(total_time))}")
    print(f"   Best Recall@100: {best_recall:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pseudo Query Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

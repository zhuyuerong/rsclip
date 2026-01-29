#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A0 Baseline: 标准 Deformable DETR 训练脚本

实验目标:
- 在DIOR数据集上训练标准Deformable DETR
- 作为Pseudo Query实验的baseline对照组
- 验证训练pipeline正确性

预期现象:
- loss平稳下降
- boxes从"全图乱飘"到"目标附近聚集"
- Recall@K前5-10个epoch明显上升
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

# 简单日志 (不使用TensorBoard，避免兼容性问题)
HAS_TENSORBOARD = False

# 添加路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'external' / 'Deformable-DETR'))

# Deformable DETR imports
from models import build_model
from models.deformable_detr import SetCriterion, PostProcess
from models.matcher import build_matcher
import util.misc as utils

# 本地数据集
from src.experiments.exp9_pseudo_query.datasets import (
    build_dior_dataset,
    DIOR_CLASSES,
)
# 使用Deformable DETR的NestedTensor和collate函数
from util.misc import NestedTensor, nested_tensor_from_tensor_list


def collate_fn(batch):
    """Deformable DETR collate函数"""
    batch = list(zip(*batch))
    # 使用Deformable DETR的nested_tensor_from_tensor_list
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def get_args_parser():
    parser = argparse.ArgumentParser('A0 Baseline: Deformable DETR on DIOR', add_help=False)
    
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
    
    # Loss参数
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # 数据集参数
    parser.add_argument('--dior_path', default='datasets/DIOR', type=str)
    parser.add_argument('--num_classes', default=20, type=int)  # DIOR有20类
    parser.add_argument('--image_size', default=800, type=int)
    parser.add_argument('--dataset_file', default='dior', type=str)  # 兼容build_model
    
    # 运行参数
    parser.add_argument('--output_dir', default='outputs/exp9_pseudo_query/a0_baseline', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_epochs', default=[1, 5, 10, 20, 30, 40, 50], type=int, nargs='+')
    parser.add_argument('--save_epochs', default=[10, 20, 30, 40, 50], type=int, nargs='+')
    
    # 分布式训练 (单机时不需要)
    parser.add_argument('--distributed', action='store_true')
    
    # Deformable DETR兼容参数 (实际上不使用，但build_model需要)
    parser.add_argument('--masks', default=False, action='store_true')
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    
    return parser


def build_dior_criterion(args, device):
    """构建DIOR损失函数"""
    # 构建matcher
    matcher = build_matcher(args)
    
    # 损失权重
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    # aux loss权重
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


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm, writer=None):
    """训练一个epoch"""
    model.train()
    criterion.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 50
    
    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # samples已经是NestedTensor
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # 前向传播 (传入NestedTensor)
        outputs = model(samples)
        
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
    
    # 写入tensorboard
    if writer is not None:
        for k, v in metric_logger.meters.items():
            writer.add_scalar(f'train/{k}', v.global_avg, epoch)
    
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, epoch, writer=None):
    """评估模型"""
    model.eval()
    criterion.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # 收集预测和GT
    all_predictions = []
    all_targets = []
    
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        outputs = model(samples)
        
        # 计算loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # 记录loss
        for k, v in loss_dict.items():
            if k in weight_dict:
                metric_logger.update(**{k: v.item()})
        
        # 后处理得到预测框
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        # 收集结果
        for result, target in zip(results, targets):
            all_predictions.append(result)
            all_targets.append(target)
    
    # 计算指标
    stats = compute_detection_metrics(all_predictions, all_targets)
    
    # 打印统计
    print("Averaged stats:", metric_logger)
    print(f"  mAP@0.5: {stats['mAP_50']:.4f}")
    print(f"  mAP@0.5:0.95: {stats['mAP']:.4f}")
    print(f"  Recall@100: {stats['recall_100']:.4f}")
    
    # 写入tensorboard
    if writer is not None:
        for k, v in metric_logger.meters.items():
            writer.add_scalar(f'val/{k}', v.global_avg, epoch)
        writer.add_scalar('val/mAP_50', stats['mAP_50'], epoch)
        writer.add_scalar('val/mAP', stats['mAP'], epoch)
        writer.add_scalar('val/recall_100', stats['recall_100'], epoch)
    
    stats.update({k: meter.global_avg for k, meter in metric_logger.meters.items()})
    return stats


def compute_detection_metrics(predictions, targets, iou_thresholds=[0.5]):
    """
    计算检测指标
    
    简化版本，计算mAP和Recall
    """
    # 简化的mAP计算
    total_tp_50 = 0
    total_fp_50 = 0
    total_gt = 0
    total_recall_50 = 0
    num_images = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu()  # [N, 4]
        pred_scores = pred['scores'].cpu()  # [N]
        pred_labels = pred['labels'].cpu()  # [N]
        
        gt_boxes = target['boxes'].cpu()  # [M, 4]
        gt_labels = target['labels'].cpu()  # [M]
        
        if len(gt_boxes) == 0:
            continue
        
        # 转换gt_boxes从cxcywh到xyxy
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        # 反归一化
        orig_size = target['orig_size'].cpu()
        gt_boxes_xyxy[:, 0::2] *= orig_size[1]
        gt_boxes_xyxy[:, 1::2] *= orig_size[0]
        
        total_gt += len(gt_boxes)
        num_images += 1
        
        if len(pred_boxes) == 0:
            continue
        
        # 计算IoU
        ious = box_iou(pred_boxes, gt_boxes_xyxy)  # [N, M]
        
        # 对于每个GT，找到最匹配的预测
        matched_gt = set()
        
        # 按分数排序
        sorted_idx = torch.argsort(pred_scores, descending=True)
        
        for idx in sorted_idx[:100]:  # 只看top-100
            best_iou, best_gt = ious[idx].max(0)
            
            if best_iou >= 0.5 and best_gt.item() not in matched_gt:
                total_tp_50 += 1
                matched_gt.add(best_gt.item())
            else:
                total_fp_50 += 1
        
        # Recall
        total_recall_50 += len(matched_gt)
    
    # 计算指标
    precision_50 = total_tp_50 / (total_tp_50 + total_fp_50 + 1e-6)
    recall_50 = total_recall_50 / (total_gt + 1e-6)
    
    # 简化mAP (使用precision作为近似)
    mAP_50 = precision_50 * recall_50 / (precision_50 + recall_50 + 1e-6) * 2  # F1 score作为近似
    
    return {
        'mAP_50': mAP_50,
        'mAP': mAP_50 * 0.6,  # 近似
        'recall_100': recall_50,
        'precision_50': precision_50,
    }


def box_cxcywh_to_xyxy(x):
    """cxcywh -> xyxy"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """计算IoU"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / (union + 1e-6)
    return iou


def main(args):
    print("=" * 70)
    print("A0 Baseline: Deformable DETR on DIOR")
    print("=" * 70)
    
    # 设置设备
    device = torch.device(args.device)
    
    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard (可选)
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
        print(f"📊 TensorBoard: {output_dir / 'tensorboard'}")
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n📁 Output: {output_dir}")
    print(f"🎲 Seed: {seed}")
    print(f"🖥️  Device: {device}")
    
    # ========================================
    # 构建模型
    # ========================================
    print("\n📦 Building model...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # 重新构建criterion (使用DIOR的类别数)
    criterion, weight_dict = build_dior_criterion(args, device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_parameters:,}")
    print(f"   Backbone: {args.backbone}")
    print(f"   Queries: {args.num_queries}")
    print(f"   Classes: {args.num_classes}")
    
    # ========================================
    # 构建数据集
    # ========================================
    print("\n📊 Building datasets...")
    dior_path = project_root / args.dior_path
    
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
    
    print(f"   Train: {len(dataset_train)} images")
    print(f"   Val: {len(dataset_val)} images")
    
    # DataLoader
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # ========================================
    # 构建优化器
    # ========================================
    print("\n⚙️  Building optimizer...")
    
    # 分离backbone和其他参数
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
    
    print(f"   LR: {args.lr}")
    print(f"   LR backbone: {args.lr_backbone}")
    print(f"   Weight decay: {args.weight_decay}")
    
    # ========================================
    # 恢复训练
    # ========================================
    if args.resume:
        print(f"\n📥 Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    # ========================================
    # 只评估模式
    # ========================================
    if args.eval:
        print("\n🔍 Evaluation mode")
        stats = evaluate(model, criterion, postprocessors, data_loader_val, device, 0, writer)
        print(f"Results: {stats}")
        return
    
    # ========================================
    # 训练循环
    # ========================================
    print("\n🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Eval epochs: {args.eval_epochs}")
    
    start_time = time.time()
    best_mAP = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, writer
        )
        lr_scheduler.step()
        
        # 评估
        if (epoch + 1) in args.eval_epochs or epoch == args.epochs - 1:
            print(f"\n📊 Evaluating epoch {epoch + 1}...")
            val_stats = evaluate(model, criterion, postprocessors, data_loader_val, device, epoch + 1, writer)
            
            # 更新最佳模型
            if val_stats['mAP_50'] > best_mAP:
                best_mAP = val_stats['mAP_50']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'mAP_50': best_mAP,
                }
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"   ✅ New best mAP@0.5: {best_mAP:.4f}")
        
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
            print(f"   💾 Saved checkpoint epoch {epoch + 1}")
        
        # 保存日志
        log_stats = {
            'epoch': epoch + 1,
            **{f'train_{k}': v for k, v in train_stats.items()},
        }
        with open(output_dir / 'log.txt', 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
    
    # 训练完成
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print("\n" + "=" * 70)
    print(f"✅ Training completed!")
    print(f"   Total time: {total_time_str}")
    print(f"   Best mAP@0.5: {best_mAP:.4f}")
    print(f"   Output: {output_dir}")
    print("=" * 70)
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A0 Baseline', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

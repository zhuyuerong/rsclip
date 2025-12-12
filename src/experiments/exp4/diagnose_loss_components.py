#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验1.1: 损失组件诊断
诊断GIoU和CAM损失不下降的根本原因
"""

import torch
import torch.nn as nn
import sys
import json
from pathlib import Path
from tqdm import tqdm
import yaml
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_surgery_cam_detector import create_simple_surgery_cam_detector
from datasets.dior_detection import get_detection_dataloader
from losses.detection_loss import DetectionLoss, generalized_box_iou


def diagnose_loss_components(model, dataloader, criterion, device, num_batches=50):
    """
    诊断损失组件的详细统计
    
    Returns:
        dict: 诊断结果
    """
    model.eval()
    
    stats = {
        'num_batches': 0,
        'num_pos_samples': [],
        'num_zero_pos_batches': 0,
        'iou_values': [],
        'cam_in_values': [],
        'cam_out_values': [],
        'cam_contrast': [],
        'peak_detection_stats': defaultdict(list),
        'match_quality': {
            'peak_matches': 0,
            'fallback_matches': 0,
            'unmatched_gts': 0
        }
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="诊断中")):
            if batch_idx >= num_batches:
                break
            
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            boxes = batch['boxes']
            labels = batch['labels']
            
            # 准备targets
            targets = []
            for b in range(len(boxes)):
                targets.append({
                    'boxes': boxes[b].to(device),
                    'labels': labels[b].to(device)
                })
            
            # 前向传播
            outputs = model(images, text_queries)
            cam = outputs['cam']
            pred_boxes = outputs['pred_boxes']
            B, C, H, W, _ = pred_boxes.shape
            
            # 统计每个batch
            for b in range(B):
                gt_boxes = targets[b]['boxes']
                gt_labels = targets[b]['labels']
                
                if len(gt_boxes) == 0:
                    continue
                
                # 1. 正样本分配统计
                pos_samples = criterion.assigner.assign(
                    cam[b], gt_boxes, gt_labels
                )
                
                num_pos = len(pos_samples)
                stats['num_pos_samples'].append(num_pos)
                
                if num_pos == 0:
                    stats['num_zero_pos_batches'] += 1
                
                # 2. IoU统计（匹配的预测框与GT框）
                for sample in pos_samples:
                    i, j = sample['i'], sample['j']
                    class_id = sample['class']
                    gt_idx = sample['gt_idx']
                    
                    pred_box = pred_boxes[b, class_id, i, j]
                    gt_box = gt_boxes[gt_idx]
                    
                    # 计算IoU
                    giou = generalized_box_iou(
                        pred_box.unsqueeze(0),
                        gt_box.unsqueeze(0)
                    )[0, 0].item()
                    stats['iou_values'].append(giou)
                
                # 3. CAM响应统计
                for k, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
                    label = label.item()
                    xmin, ymin, xmax, ymax = box
                    cam_c = cam[b, label]
                    
                    # 框内mask
                    mask_in = torch.zeros(H, W, device=cam.device)
                    i_min = max(0, int(ymin * H))
                    i_max = min(H - 1, int(ymax * H))
                    j_min = max(0, int(xmin * W))
                    j_max = min(W - 1, int(xmax * W))
                    
                    if i_max >= i_min and j_max >= j_min:
                        mask_in[i_min:i_max+1, j_min:j_max+1] = 1
                    
                    mask_out = 1 - mask_in
                    
                    mask_in_sum = mask_in.sum()
                    mask_out_sum = mask_out.sum()
                    
                    if mask_in_sum > 0 and mask_out_sum > 0:
                        cam_in = (cam_c * mask_in).sum() / mask_in_sum
                        cam_out = (cam_c * mask_out).sum() / mask_out_sum
                        
                        cam_in_val = cam_in.item()
                        cam_out_val = cam_out.item()
                        
                        stats['cam_in_values'].append(cam_in_val)
                        stats['cam_out_values'].append(cam_out_val)
                        
                        # 对比度
                        if cam_out_val > 1e-6:
                            contrast = cam_in_val / (cam_out_val + 1e-6)
                            stats['cam_contrast'].append(contrast)
                
                # 4. 峰值检测统计（按类别）
                unique_classes = torch.unique(gt_labels)
                for class_id in unique_classes:
                    class_id = class_id.item()
                    cam_class = cam[b, class_id]
                    
                    # 检测峰值
                    peaks = criterion.assigner.peak_detector.detect_peaks(cam_class)
                    
                    stats['peak_detection_stats'][f'class_{class_id}'].append({
                        'num_peaks': len(peaks),
                        'max_cam_value': cam_class.max().item(),
                        'mean_cam_value': cam_class.mean().item()
                    })
                
                # 5. 匹配质量统计
                for sample in pos_samples:
                    match_type = sample.get('match_type', 'unknown')
                    if match_type == 'peak':
                        stats['match_quality']['peak_matches'] += 1
                    elif match_type == 'fallback':
                        stats['match_quality']['fallback_matches'] += 1
                
                # 统计未匹配的GT
                matched_gt_indices = set(s['gt_idx'] for s in pos_samples)
                stats['match_quality']['unmatched_gts'] += len(gt_boxes) - len(matched_gt_indices)
            
            stats['num_batches'] += 1
    
    # 计算汇总统计
    summary = {
        'num_batches_analyzed': stats['num_batches'],
        'total_samples': len(stats['num_pos_samples']),
        'positive_samples': {
            'mean': np.mean(stats['num_pos_samples']) if stats['num_pos_samples'] else 0,
            'median': np.median(stats['num_pos_samples']) if stats['num_pos_samples'] else 0,
            'min': np.min(stats['num_pos_samples']) if stats['num_pos_samples'] else 0,
            'max': np.max(stats['num_pos_samples']) if stats['num_pos_samples'] else 0,
            'zero_pos_batch_ratio': stats['num_zero_pos_batches'] / stats['num_batches'] if stats['num_batches'] > 0 else 0
        },
        'iou_statistics': {
            'mean': np.mean(stats['iou_values']) if stats['iou_values'] else 0,
            'median': np.median(stats['iou_values']) if stats['iou_values'] else 0,
            'std': np.std(stats['iou_values']) if stats['iou_values'] else 0,
            'min': np.min(stats['iou_values']) if stats['iou_values'] else 0,
            'max': np.max(stats['iou_values']) if stats['iou_values'] else 0,
            'num_samples': len(stats['iou_values'])
        },
        'cam_statistics': {
            'cam_in': {
                'mean': np.mean(stats['cam_in_values']) if stats['cam_in_values'] else 0,
                'median': np.median(stats['cam_in_values']) if stats['cam_in_values'] else 0,
                'std': np.std(stats['cam_in_values']) if stats['cam_in_values'] else 0,
                'min': np.min(stats['cam_in_values']) if stats['cam_in_values'] else 0,
                'max': np.max(stats['cam_in_values']) if stats['cam_in_values'] else 0
            },
            'cam_out': {
                'mean': np.mean(stats['cam_out_values']) if stats['cam_out_values'] else 0,
                'median': np.median(stats['cam_out_values']) if stats['cam_out_values'] else 0,
                'std': np.std(stats['cam_out_values']) if stats['cam_out_values'] else 0,
                'min': np.min(stats['cam_out_values']) if stats['cam_out_values'] else 0,
                'max': np.max(stats['cam_out_values']) if stats['cam_out_values'] else 0
            },
            'contrast': {
                'mean': np.mean(stats['cam_contrast']) if stats['cam_contrast'] else 0,
                'median': np.median(stats['cam_contrast']) if stats['cam_contrast'] else 0,
                'std': np.std(stats['cam_contrast']) if stats['cam_contrast'] else 0
            }
        },
        'match_quality': stats['match_quality'],
        'peak_detection_summary': {}
    }
    
    # 汇总峰值检测统计
    for class_id, peak_stats in stats['peak_detection_stats'].items():
        if peak_stats:
            summary['peak_detection_summary'][class_id] = {
                'mean_num_peaks': np.mean([s['num_peaks'] for s in peak_stats]),
                'mean_max_cam': np.mean([s['max_cam_value'] for s in peak_stats]),
                'mean_mean_cam': np.mean([s['mean_cam_value'] for s in peak_stats])
            }
    
    return summary


def print_diagnosis_report(summary):
    """打印诊断报告"""
    print("=" * 80)
    print("损失组件诊断报告")
    print("=" * 80)
    
    print(f"\n【1. 正样本分配统计】")
    print(f"  分析的batch数: {summary['num_batches_analyzed']}")
    print(f"  平均正样本数/batch: {summary['positive_samples']['mean']:.2f}")
    print(f"  中位数正样本数: {summary['positive_samples']['median']:.2f}")
    print(f"  零正样本batch比例: {summary['positive_samples']['zero_pos_batch_ratio']:.2%}")
    
    if summary['positive_samples']['zero_pos_batch_ratio'] > 0.5:
        print(f"  ⚠️  警告: 超过50%的batch没有正样本！这是GIoU损失不下降的主要原因。")
        print(f"      → 建议: 降低min_peak_value阈值或改进峰值检测算法")
    
    print(f"\n【2. IoU统计（匹配质量）】")
    if summary['iou_statistics']['num_samples'] > 0:
        print(f"  平均IoU: {summary['iou_statistics']['mean']:.4f}")
        print(f"  中位数IoU: {summary['iou_statistics']['median']:.4f}")
        print(f"  IoU标准差: {summary['iou_statistics']['std']:.4f}")
        print(f"  IoU范围: [{summary['iou_statistics']['min']:.4f}, {summary['iou_statistics']['max']:.4f}]")
        
        if summary['iou_statistics']['mean'] < 0.1:
            print(f"  ⚠️  警告: 平均IoU < 0.1，匹配质量很差！")
            print(f"      → 建议: 改进匹配算法，使用预测框IoU而不是峰值位置")
        elif summary['iou_statistics']['mean'] < 0.3:
            print(f"  ⚠️  警告: 平均IoU < 0.3，匹配质量需要改进")
    else:
        print(f"  ❌ 没有匹配的样本，无法计算IoU")
    
    print(f"\n【3. CAM响应统计】")
    print(f"  框内CAM响应:")
    print(f"    平均值: {summary['cam_statistics']['cam_in']['mean']:.4f}")
    print(f"    中位数: {summary['cam_statistics']['cam_in']['median']:.4f}")
    print(f"    范围: [{summary['cam_statistics']['cam_in']['min']:.4f}, {summary['cam_statistics']['cam_in']['max']:.4f}]")
    
    print(f"  框外CAM响应:")
    print(f"    平均值: {summary['cam_statistics']['cam_out']['mean']:.4f}")
    print(f"    中位数: {summary['cam_statistics']['cam_out']['median']:.4f}")
    
    print(f"  对比度 (框内/框外):")
    print(f"    平均值: {summary['cam_statistics']['contrast']['mean']:.2f}")
    print(f"    中位数: {summary['cam_statistics']['contrast']['median']:.2f}")
    
    if summary['cam_statistics']['cam_in']['mean'] < 0.1:
        print(f"  ⚠️  警告: 框内CAM响应 < 0.1，CAM质量很差！")
        print(f"      → 建议: 增加CAM损失权重或改进CAM生成器")
    
    if summary['cam_statistics']['contrast']['mean'] < 2.0:
        print(f"  ⚠️  警告: CAM对比度 < 2.0，框内外区分度不够！")
        print(f"      → 建议: 改进CAM损失函数，使用更敏感的损失")
    
    print(f"\n【4. 匹配质量统计】")
    print(f"  峰值匹配: {summary['match_quality']['peak_matches']}")
    print(f"  Fallback匹配: {summary['match_quality']['fallback_matches']}")
    print(f"  未匹配GT: {summary['match_quality']['unmatched_gts']}")
    
    total_matches = summary['match_quality']['peak_matches'] + summary['match_quality']['fallback_matches']
    if total_matches > 0:
        fallback_ratio = summary['match_quality']['fallback_matches'] / total_matches
        if fallback_ratio > 0.5:
            print(f"  ⚠️  警告: Fallback匹配比例 > 50%，峰值检测效果不佳")
    
    print(f"\n【5. 峰值检测统计（按类别）】")
    for class_id, stats in summary['peak_detection_summary'].items():
        print(f"  {class_id}:")
        print(f"    平均峰值数: {stats['mean_num_peaks']:.2f}")
        print(f"    平均最大CAM: {stats['mean_max_cam']:.4f}")
        print(f"    平均CAM值: {stats['mean_mean_cam']:.4f}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='诊断损失组件')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_simple_model.pth',
                       help='模型checkpoint路径（可选）')
    parser.add_argument('--num-batches', type=int, default=50,
                       help='分析的batch数量')
    parser.add_argument('--output', type=str, default='outputs/diagnosis/loss_components_report.json',
                       help='输出报告路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config.get('device', 'cuda')
    print(f"使用设备: {device}")
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print("创建模型...")
    model = create_simple_surgery_cam_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        upsample_cam=config.get('upsample_cam', False),
        device=device,
        unfreeze_cam_last_layer=True
    )
    
    # 加载checkpoint（如果提供）
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_state = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        trainable_state = {}
        for key, value in model_state.items():
            if key in model_dict and ('box_head' in key or 'cam_generator.learnable_proj' in key):
                trainable_state[key] = value
        model_dict.update(trainable_state)
        model.load_state_dict(model_dict, strict=False)
        print(f"✅ 已加载checkpoint: {args.checkpoint}")
    
    # 加载数据
    print("加载数据...")
    train_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='trainval',
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment=False,  # 诊断时不使用数据增强
        train_only_seen=config.get('train_only_seen', True)
    )
    
    # 损失函数
    criterion = DetectionLoss(
        lambda_l1=config.get('lambda_l1', 1.0),
        lambda_giou=config.get('lambda_giou', 2.0),
        lambda_cam=config.get('lambda_cam', 0.5),
        min_peak_distance=config.get('min_peak_distance', 2),
        min_peak_value=config.get('min_peak_value', 0.3),
        match_iou_threshold=config.get('match_iou_threshold', 0.3)
    ).to(device)
    
    # 运行诊断
    print(f"\n开始诊断（分析 {args.num_batches} 个batch）...")
    summary = diagnose_loss_components(
        model, train_loader, criterion, device, num_batches=args.num_batches
    )
    
    # 打印报告
    print_diagnosis_report(summary)
    
    # 保存报告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换numpy类型为Python原生类型以便JSON序列化
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    summary_serializable = convert_to_serializable(summary)
    
    with open(output_path, 'w') as f:
        json.dump(summary_serializable, f, indent=2)
    
    print(f"\n✅ 诊断报告已保存到: {output_path}")


if __name__ == '__main__':
    main()


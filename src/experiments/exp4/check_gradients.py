#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验1.2: 梯度流诊断
检查梯度是否正常传播，识别梯度消失或爆炸问题
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
from losses.detection_loss import DetectionLoss


def check_gradients(model, dataloader, criterion, device, num_batches=10):
    """
    检查梯度流
    
    Returns:
        dict: 梯度诊断结果
    """
    model.train()  # 设置为训练模式以计算梯度
    
    stats = {
        'box_head_gradients': defaultdict(list),
        'cam_generator_gradients': defaultdict(list),
        'gradient_norms': {
            'box_head': [],
            'cam_generator': []
        },
        'gradient_ratios': {
            'box_head': [],
            'cam_generator': []
        }
    }
    
    # 获取一个batch
    batch = next(iter(dataloader))
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
    
    # 计算损失
    loss_dict = criterion(outputs, targets)
    loss = loss_dict['loss_total']
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 检查梯度
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grad_norm = grad.norm().item()
            param_norm = param.norm().item()
            
            # 计算梯度/参数比例
            if param_norm > 1e-8:
                grad_ratio = grad_norm / param_norm
            else:
                grad_ratio = float('inf')
            
            # 分类统计
            if 'box_head' in name:
                stats['box_head_gradients'][name] = {
                    'grad_norm': grad_norm,
                    'param_norm': param_norm,
                    'grad_ratio': grad_ratio,
                    'grad_mean': grad.mean().item(),
                    'grad_std': grad.std().item(),
                    'grad_min': grad.min().item(),
                    'grad_max': grad.max().item()
                }
                stats['gradient_norms']['box_head'].append(grad_norm)
                stats['gradient_ratios']['box_head'].append(grad_ratio)
            
            elif 'cam_generator' in name:
                stats['cam_generator_gradients'][name] = {
                    'grad_norm': grad_norm,
                    'param_norm': param_norm,
                    'grad_ratio': grad_ratio,
                    'grad_mean': grad.mean().item(),
                    'grad_std': grad.std().item(),
                    'grad_min': grad.min().item(),
                    'grad_max': grad.max().item()
                }
                stats['gradient_norms']['cam_generator'].append(grad_norm)
                stats['gradient_ratios']['cam_generator'].append(grad_ratio)
    
    # 计算汇总统计
    summary = {
        'box_head': {
            'num_params': len(stats['box_head_gradients']),
            'gradient_norms': {
                'mean': np.mean(stats['gradient_norms']['box_head']) if stats['gradient_norms']['box_head'] else 0,
                'median': np.median(stats['gradient_norms']['box_head']) if stats['gradient_norms']['box_head'] else 0,
                'min': np.min(stats['gradient_norms']['box_head']) if stats['gradient_norms']['box_head'] else 0,
                'max': np.max(stats['gradient_norms']['box_head']) if stats['gradient_norms']['box_head'] else 0,
                'std': np.std(stats['gradient_norms']['box_head']) if stats['gradient_norms']['box_head'] else 0
            },
            'gradient_ratios': {
                'mean': np.mean(stats['gradient_ratios']['box_head']) if stats['gradient_ratios']['box_head'] else 0,
                'median': np.median(stats['gradient_ratios']['box_head']) if stats['gradient_ratios']['box_head'] else 0,
                'min': np.min(stats['gradient_ratios']['box_head']) if stats['gradient_ratios']['box_head'] else 0,
                'max': np.max(stats['gradient_ratios']['box_head']) if stats['gradient_ratios']['box_head'] else 0
            },
            'detailed': stats['box_head_gradients']
        },
        'cam_generator': {
            'num_params': len(stats['cam_generator_gradients']),
            'gradient_norms': {
                'mean': np.mean(stats['gradient_norms']['cam_generator']) if stats['gradient_norms']['cam_generator'] else 0,
                'median': np.median(stats['gradient_norms']['cam_generator']) if stats['gradient_norms']['cam_generator'] else 0,
                'min': np.min(stats['gradient_norms']['cam_generator']) if stats['gradient_norms']['cam_generator'] else 0,
                'max': np.max(stats['gradient_norms']['cam_generator']) if stats['gradient_norms']['cam_generator'] else 0,
                'std': np.std(stats['gradient_norms']['cam_generator']) if stats['gradient_norms']['cam_generator'] else 0
            },
            'gradient_ratios': {
                'mean': np.mean(stats['gradient_ratios']['cam_generator']) if stats['gradient_ratios']['cam_generator'] else 0,
                'median': np.median(stats['gradient_ratios']['cam_generator']) if stats['gradient_ratios']['cam_generator'] else 0,
                'min': np.min(stats['gradient_ratios']['cam_generator']) if stats['gradient_ratios']['cam_generator'] else 0,
                'max': np.max(stats['gradient_ratios']['cam_generator']) if stats['gradient_ratios']['cam_generator'] else 0
            },
            'detailed': stats['cam_generator_gradients']
        },
        'loss_components': {
            'loss_total': loss_dict['loss_total'].item(),
            'loss_l1': loss_dict['loss_box_l1'].item(),
            'loss_giou': loss_dict['loss_box_giou'].item(),
            'loss_cam': loss_dict['loss_cam'].item()
        }
    }
    
    return summary


def print_gradient_report(summary):
    """打印梯度诊断报告"""
    print("=" * 80)
    print("梯度流诊断报告")
    print("=" * 80)
    
    print(f"\n【1. BoxHead梯度统计】")
    print(f"  可训练参数数量: {summary['box_head']['num_params']}")
    
    if summary['box_head']['num_params'] > 0:
        grad_norms = summary['box_head']['gradient_norms']
        print(f"  梯度范数:")
        print(f"    平均值: {grad_norms['mean']:.6e}")
        print(f"    中位数: {grad_norms['median']:.6e}")
        print(f"    范围: [{grad_norms['min']:.6e}, {grad_norms['max']:.6e}]")
        
        if grad_norms['mean'] < 1e-6:
            print(f"  ⚠️  警告: BoxHead梯度范数 < 1e-6，可能存在梯度消失！")
            print(f"      → 建议: 增加学习率或检查损失函数")
        elif grad_norms['mean'] > 1.0:
            print(f"  ⚠️  警告: BoxHead梯度范数 > 1.0，可能存在梯度爆炸！")
            print(f"      → 建议: 使用梯度裁剪或降低学习率")
        else:
            print(f"  ✅ BoxHead梯度范数正常")
        
        grad_ratios = summary['box_head']['gradient_ratios']
        print(f"  梯度/参数比例:")
        print(f"    平均值: {grad_ratios['mean']:.6e}")
        print(f"    中位数: {grad_ratios['median']:.6e}")
    else:
        print(f"  ❌ 没有找到BoxHead的可训练参数")
    
    print(f"\n【2. CAM生成器梯度统计】")
    print(f"  可训练参数数量: {summary['cam_generator']['num_params']}")
    
    if summary['cam_generator']['num_params'] > 0:
        grad_norms = summary['cam_generator']['gradient_norms']
        print(f"  梯度范数:")
        print(f"    平均值: {grad_norms['mean']:.6e}")
        print(f"    中位数: {grad_norms['median']:.6e}")
        print(f"    范围: [{grad_norms['min']:.6e}, {grad_norms['max']:.6e}]")
        
        if grad_norms['mean'] < 1e-7:
            print(f"  ⚠️  警告: CAM生成器梯度范数 < 1e-7，可能存在梯度消失！")
            print(f"      → 建议: 增加CAM生成器学习率或检查损失函数")
        elif grad_norms['mean'] > 1.0:
            print(f"  ⚠️  警告: CAM生成器梯度范数 > 1.0，可能存在梯度爆炸！")
            print(f"      → 建议: 使用梯度裁剪或降低学习率")
        else:
            print(f"  ✅ CAM生成器梯度范数正常")
        
        grad_ratios = summary['cam_generator']['gradient_ratios']
        print(f"  梯度/参数比例:")
        print(f"    平均值: {grad_ratios['mean']:.6e}")
        print(f"    中位数: {grad_ratios['median']:.6e}")
    else:
        print(f"  ❌ 没有找到CAM生成器的可训练参数")
    
    print(f"\n【3. 损失组件】")
    loss_comp = summary['loss_components']
    print(f"  总损失: {loss_comp['loss_total']:.4f}")
    print(f"  L1损失: {loss_comp['loss_l1']:.4f}")
    print(f"  GIoU损失: {loss_comp['loss_giou']:.4f}")
    print(f"  CAM损失: {loss_comp['loss_cam']:.4f}")
    
    print(f"\n【4. 详细梯度信息】")
    print(f"  BoxHead参数梯度:")
    for name, grad_info in list(summary['box_head']['detailed'].items())[:5]:
        print(f"    {name}:")
        print(f"      梯度范数: {grad_info['grad_norm']:.6e}")
        print(f"      梯度均值: {grad_info['grad_mean']:.6e}")
        print(f"      梯度范围: [{grad_info['grad_min']:.6e}, {grad_info['grad_max']:.6e}]")
    
    if summary['cam_generator']['num_params'] > 0:
        print(f"  CAM生成器参数梯度:")
        for name, grad_info in summary['cam_generator']['detailed'].items():
            print(f"    {name}:")
            print(f"      梯度范数: {grad_info['grad_norm']:.6e}")
            print(f"      梯度均值: {grad_info['grad_mean']:.6e}")
            print(f"      梯度范围: [{grad_info['grad_min']:.6e}, {grad_info['grad_max']:.6e}]")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='检查梯度流')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_simple_model.pth',
                       help='模型checkpoint路径（可选）')
    parser.add_argument('--output', type=str, default='outputs/diagnosis/gradient_report.json',
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
        augment=False,
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
    
    # 运行梯度检查
    print("\n开始检查梯度...")
    summary = check_gradients(model, train_loader, criterion, device)
    
    # 打印报告
    print_gradient_report(summary)
    
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
    
    print(f"\n✅ 梯度报告已保存到: {output_path}")


if __name__ == '__main__':
    main()



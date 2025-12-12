#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进检测器训练脚本
原图 + 多层特征 + 多层CAM
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import math

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent.parent))

from models.improved_direct_detection_detector import create_improved_direct_detection_detector
from datasets.dior_detection import get_detection_dataloader
from losses.improved_direct_detection_loss import ImprovedDirectDetectionLoss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine annealing with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    loss_l1_sum = 0.0
    loss_giou_sum = 0.0
    loss_conf_sum = 0.0
    num_batches = 0
    
    pos_ratios = []
    cam_contrasts = []  # CAM对比度
    layer_weights_list = []  # 层权重
    
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config['num_epochs']}]", disable=True)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        text_queries = batch['text_queries']
        boxes = batch['boxes']
        labels = batch['labels']
        
        targets = []
        for b in range(len(boxes)):
            targets.append({
                'boxes': boxes[b].to(device),
                'labels': labels[b].to(device)
            })
        
        outputs = model(images, text_queries)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss_total']
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        loss_l1_sum += loss_dict['loss_box_l1'].item()
        loss_giou_sum += loss_dict['loss_box_giou'].item()
        loss_conf_sum += loss_dict['loss_conf'].item()
        pos_ratios.append(loss_dict['pos_ratio'])
        num_batches += 1
        
        # 监控CAM质量（每10个batch）
        if batch_idx % 10 == 0:
            fused_cam = outputs['fused_cam']
            # 计算CAM对比度：框内平均响应 / 框外平均响应
            if len(targets) > 0 and len(targets[0]['boxes']) > 0:
                b = 0
                gt_boxes = targets[b]['boxes']
                gt_labels = targets[b]['labels']
                if len(gt_boxes) > 0:
                    # 使用第一个GT框和对应类别
                    gt_box = gt_boxes[0]
                    gt_label = gt_labels[0].item()
                    cam_c = fused_cam[b, gt_label]  # [H, W]
                    
                    # 将GT框坐标转换为CAM空间索引
                    H, W = cam_c.shape
                    x1, y1, x2, y2 = gt_box
                    x1_idx = int(x1 * W)
                    y1_idx = int(y1 * H)
                    x2_idx = int(x2 * W)
                    y2_idx = int(y2 * H)
                    x1_idx = max(0, min(x1_idx, W-1))
                    y1_idx = max(0, min(y1_idx, H-1))
                    x2_idx = max(0, min(x2_idx, W-1))
                    y2_idx = max(0, min(y2_idx, H-1))
                    
                    # 创建框内外mask
                    mask = torch.zeros(H, W, dtype=torch.bool, device=cam_c.device)
                    mask[y1_idx:y2_idx+1, x1_idx:x2_idx+1] = True
                    
                    # 计算框内外平均响应
                    inside_response = cam_c[mask].mean().item()
                    outside_response = cam_c[~mask].mean().item()
                    
                    # 对比度 = 框内 / 框外
                    contrast = inside_response / (outside_response + 1e-6)
                    cam_contrasts.append(contrast)
        
        # 监控层权重（总是监控）
        layer_weights = outputs.get('layer_weights', None)
        if layer_weights is not None:
            layer_weights_list.append(layer_weights)
    
    avg_loss = total_loss / num_batches
    avg_l1 = loss_l1_sum / num_batches
    avg_giou = loss_giou_sum / num_batches
    avg_conf = loss_conf_sum / num_batches
    avg_pos_ratio = sum(pos_ratios) / len(pos_ratios) if pos_ratios else 0.0
    avg_cam_contrast = sum(cam_contrasts) / len(cam_contrasts) if cam_contrasts else 0.0
    avg_layer_weights = None
    if layer_weights_list:
        import numpy as np
        avg_layer_weights = np.mean(layer_weights_list, axis=0)
    
    return avg_loss, avg_l1, avg_giou, avg_conf, avg_pos_ratio, avg_cam_contrast, avg_layer_weights


def main():
    parser = argparse.ArgumentParser(description='训练改进检测器')
    parser.add_argument('--config', type=str, default='configs/improved_detector_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config.get('device', 'cuda')
    print(f"使用设备: {device}")
    print(f"✅ 使用改进检测器：原图 + 多层特征 + 多层CAM")
    print(f"   架构: 多层提取 + CAM融合 + 原图编码 + 多输入检测头")
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print(f"\n创建改进检测器...")
    model = create_improved_direct_detection_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        device=device,
        unfreeze_cam_last_layer=config.get('unfreeze_cam_last_layer', True)
    )
    model.to(device)
    
    # 加载数据
    train_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='trainval',
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment=True,
        train_only_seen=config.get('train_only_seen', True)
    )
    
    # 使用改进的直接检测损失函数
    criterion = ImprovedDirectDetectionLoss(
        lambda_l1=config.get('lambda_l1', 1.0),
        lambda_giou=config.get('lambda_giou', 2.0),
        lambda_conf=config.get('lambda_conf', 1.0),
        lambda_cam=config.get('lambda_cam', 0.0),
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
        pos_radius=config.get('pos_radius', 1.5),
        pos_iou_threshold=config.get('pos_iou_threshold', 0.3),  # P0: 降低到0.15
        use_cam_alignment=False
    ).to(device)
    
    print(f"\n损失函数配置:")
    print(f"  L1权重: {config.get('lambda_l1', 1.0)}")
    print(f"  GIoU权重: {config.get('lambda_giou', 2.0)}")
    print(f"  置信度权重: {config.get('lambda_conf', 1.0)}")
    print(f"  CAM权重: {config.get('lambda_cam', 0.0)} (已移除)")
    
    # 优化器（分组学习率）
    image_encoder_params = []
    detection_head_params = []
    cam_generator_params = []
    cam_fusion_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'image_encoder' in name:
                image_encoder_params.append(param)
            elif 'cam_fusion' in name:
                cam_fusion_params.append(param)
            elif 'cam_generator' in name:
                cam_generator_params.append(param)
            else:
                detection_head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': image_encoder_params, 'lr': config.get('image_encoder_lr', 1e-4)},
        {'params': detection_head_params, 'lr': config.get('learning_rate', 1e-4)},
        {'params': cam_generator_params, 'lr': config.get('cam_generator_lr', 5e-5)},
        {'params': cam_fusion_params, 'lr': config.get('cam_fusion_lr', 1e-4)}
    ], weight_decay=config.get('weight_decay', 0.01))
    
    print(f"\n优化器参数:")
    print(f"  原图编码器学习率: {config.get('image_encoder_lr', 1e-4)}")
    print(f"  检测头学习率: {config.get('learning_rate', 1e-4)}")
    print(f"  CAM生成器学习率: {config.get('cam_generator_lr', 5e-5)}")
    print(f"  CAM融合学习率: {config.get('cam_fusion_lr', 1e-4)}")
    
    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # 处理SurgeryCLIP动态attention层结构变化
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            # 跳过attention层的不匹配键（这些会在forward时自动创建）
            if 'attn.in_proj_weight' in key or 'attn.in_proj_bias' in key:
                # 检查是否有对应的qkv版本
                qkv_key = key.replace('in_proj_weight', 'qkv.weight').replace('in_proj_bias', 'qkv.bias')
                if qkv_key not in state_dict:
                    continue  # 跳过，让模型使用默认初始化
            elif 'attn.qkv.weight' in key or 'attn.qkv.bias' in key:
                # 保留qkv版本
                if key in model_state_dict:
                    filtered_state_dict[key] = value
            elif key in model_state_dict:
                # 其他键直接匹配
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"⚠️  跳过形状不匹配的键: {key} (模型: {model_state_dict[key].shape}, checkpoint: {value.shape})")
        
        # 加载过滤后的state_dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        criterion.load_state_dict(checkpoint.get('criterion_state_dict', {}))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"✅ 从epoch {start_epoch}恢复训练（已处理动态attention层结构）")
    
    # 学习率调度器（Warmup + Cosine）
    # 如果恢复训练，需要调整总步数
    if args.resume:
        # 恢复训练时，warmup已经完成，使用剩余epochs计算
        remaining_epochs = config.get('num_epochs', 50) - start_epoch
        num_warmup_steps = 0  # 恢复训练时不再warmup
        num_training_steps = remaining_epochs * len(train_loader)
        
        # 重要：恢复训练时，重新创建scheduler，不恢复旧状态
        # 这样可以确保学习率从当前配置开始，而不是从旧状态继续
        print(f"⚠️  恢复训练：重新创建scheduler（不恢复旧状态）")
        print(f"   当前学习率: {config.get('learning_rate', 1e-4)}")
        
        # 更新optimizer的学习率到新配置
        for param_group in optimizer.param_groups:
            if 'image_encoder' in str(param_group.get('name', '')):
                param_group['lr'] = config.get('image_encoder_lr', 5e-5)
            elif 'cam_generator' in str(param_group.get('name', '')):
                param_group['lr'] = config.get('cam_generator_lr', 2.5e-5)
            elif 'cam_fusion' in str(param_group.get('name', '')):
                param_group['lr'] = config.get('cam_fusion_lr', 5e-5)
            else:
                param_group['lr'] = config.get('learning_rate', 5e-5)
    else:
        num_warmup_steps = config.get('warmup_epochs', 5) * len(train_loader)
        num_training_steps = config.get('num_epochs', 50) * len(train_loader)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    # 不再恢复scheduler状态，让学习率从新配置开始
    
    print(f"\n学习率调度:")
    print(f"  Warmup epochs: {config.get('warmup_epochs', 5)}")
    print(f"  Scheduler: {config.get('scheduler', 'cosine')}")
    print(f"  剩余训练步数: {num_training_steps}")
    
    # 创建checkpoint目录
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/improved_detector'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练日志
    log_file = checkpoint_dir / f'training_improved_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 训练循环
    print(f"\n开始训练（改进检测器）...")
    print(f"总epoch数: {config.get('num_epochs', 50)}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"日志文件: {log_file}")
    
    for epoch in range(start_epoch, config.get('num_epochs', 50)):
        avg_loss, avg_l1, avg_giou, avg_conf, avg_pos_ratio, avg_cam_contrast, avg_layer_weights = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        scheduler.step()
        
        # 记录日志
        log_msg = (f"Epoch [{epoch+1}/{config.get('num_epochs', 50)}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"L1: {avg_l1:.4f} | "
                  f"GIoU: {avg_giou:.4f} | "
                  f"Conf: {avg_conf:.4f} | "
                  f"PosRatio: {avg_pos_ratio:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_cam_contrast > 0:
            log_msg += f" | CAM_Contrast: {avg_cam_contrast:.2f}"
        
        if avg_layer_weights is not None:
            log_msg += f" | LayerWeights: [{avg_layer_weights[0]:.3f}, {avg_layer_weights[1]:.3f}, {avg_layer_weights[2]:.3f}]"
        
        print(log_msg)
        
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        # 保存checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss
        }
        
        torch.save(checkpoint, checkpoint_dir / 'latest_improved_detector_model.pth')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint['best_loss'] = best_loss
            torch.save(checkpoint, checkpoint_dir / 'best_improved_detector_model.pth')
            print(f"✅ 保存最佳模型 (loss: {best_loss:.4f})")
    
    print(f"\n训练完成！")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型保存在: {checkpoint_dir}")


if __name__ == '__main__':
    main()


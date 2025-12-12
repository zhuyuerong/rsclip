#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验4.2: 学习率调度优化训练
使用改进的学习率调度策略
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

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent.parent))

from models.enhanced_simple_surgery_cam_detector import create_enhanced_simple_surgery_cam_detector
from datasets.dior_detection import get_detection_dataloader
from losses.combined_improved_loss import CombinedImprovedDetectionLoss
from utils.lr_scheduler import create_lr_scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    loss_l1_sum = 0.0
    loss_giou_sum = 0.0
    loss_cam_sum = 0.0
    loss_cam_peak_sum = 0.0
    num_batches = 0
    
    match_stats_agg = {
        'total_peaks': 0,
        'total_gts': 0,
        'peak_matches': 0,
        'fallback_matches': 0,
        'unmatched_gts': 0,
        'avg_match_iou': []
    }
    
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
        loss_cam_sum += loss_dict['loss_cam'].item()
        loss_cam_peak_sum += loss_dict.get('loss_cam_peak', 0.0)
        num_batches += 1
        
        if 'match_stats' in loss_dict:
            stats = loss_dict['match_stats']
            match_stats_agg['total_peaks'] += stats['total_peaks']
            match_stats_agg['total_gts'] += stats['total_gts']
            match_stats_agg['peak_matches'] += stats['peak_matches']
            match_stats_agg['fallback_matches'] += stats['fallback_matches']
            match_stats_agg['unmatched_gts'] += stats['unmatched_gts']
            if isinstance(stats['avg_match_iou'], list):
                match_stats_agg['avg_match_iou'].extend(stats['avg_match_iou'])
            elif stats['avg_match_iou'] > 0:
                match_stats_agg['avg_match_iou'].append(stats['avg_match_iou'])
    
    avg_loss = total_loss / num_batches
    avg_l1 = loss_l1_sum / num_batches
    avg_giou = loss_giou_sum / num_batches
    avg_cam = loss_cam_sum / num_batches
    avg_cam_peak = loss_cam_peak_sum / num_batches
    
    avg_match_iou = 0.0
    if match_stats_agg['avg_match_iou']:
        avg_match_iou = sum(match_stats_agg['avg_match_iou']) / len(match_stats_agg['avg_match_iou'])
    
    return avg_loss, avg_l1, avg_giou, avg_cam, avg_cam_peak, match_stats_agg, avg_match_iou


def main():
    parser = argparse.ArgumentParser(description='训练学习率调度优化的SurgeryCAM模型')
    parser.add_argument('--config', type=str, default='configs/exp4.2_lr_schedule.yaml',
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
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print(f"创建增强的SurgeryCAM模型（学习率调度优化）...")
    model = create_enhanced_simple_surgery_cam_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        upsample_cam=config.get('upsample_cam', False),
        device=device,
        unfreeze_cam_last_layer=True
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
    
    # 使用组合改进的损失函数
    criterion = CombinedImprovedDetectionLoss(
        lambda_l1=config.get('lambda_l1', 1.0),
        lambda_giou=config.get('lambda_giou', 1.0),
        lambda_cam=config.get('lambda_cam', 2.0),
        min_peak_distance=config.get('min_peak_distance', 2),
        min_peak_value=config.get('min_peak_value', 0.1),
        match_iou_threshold=config.get('match_iou_threshold', 0.3),
        use_improved_peak_detector=True
    ).to(device)
    
    # 优化器
    cam_params = []
    box_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'cam_generator' in name:
                cam_params.append(param)
            elif 'box_head' in name:
                box_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': cam_params, 'lr': config.get('cam_generator_lr', 5e-5)},
        {'params': box_params, 'lr': config.get('learning_rate', 1e-4)}
    ], weight_decay=config.get('weight_decay', 0.01))
    
    print(f"优化器参数:")
    print(f"  CAM生成器学习率: {config.get('cam_generator_lr', 5e-5)}")
    print(f"  BoxHead学习率: {config.get('learning_rate', 1e-4)}")
    
    # 使用改进的学习率调度器
    scheduler = create_lr_scheduler(optimizer, config)
    scheduler_type = config.get('lr_scheduler_type', 'cosine')
    print(f"学习率调度器: {scheduler_type}")
    
    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"从epoch {start_epoch}恢复训练")
    
    # 创建checkpoint目录
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/exp4.2'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练日志
    log_file = checkpoint_dir / f'training_exp4.2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 训练循环
    print(f"\n开始训练（实验4.2：学习率调度优化）...")
    print(f"总epoch数: {config.get('num_epochs', 100)}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"日志文件: {log_file}")
    
    for epoch in range(start_epoch, config.get('num_epochs', 100)):
        avg_loss, avg_l1, avg_giou, avg_cam, avg_cam_peak, match_stats, avg_match_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # 更新学习率
        if scheduler_type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        
        # 获取当前学习率
        if hasattr(scheduler, 'get_last_lr'):
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # 记录日志
        log_msg = (f"Epoch [{epoch+1}/{config.get('num_epochs', 100)}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"L1: {avg_l1:.4f} | "
                  f"GIoU: {avg_giou:.4f} | "
                  f"CAM: {avg_cam:.4f} | "
                  f"CAM_Peak: {avg_cam_peak:.4f} | "
                  f"MatchIoU: {avg_match_iou:.4f} | "
                  f"LR: {current_lr:.6f}")
        print(log_msg)
        
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        # 保存checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'match_stats': match_stats
        }
        
        if hasattr(scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / 'latest_exp4.2_model.pth')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint['best_loss'] = best_loss
            torch.save(checkpoint, checkpoint_dir / 'best_exp4.2_model.pth')
            print(f"✅ 保存最佳模型 (loss: {best_loss:.4f})")
    
    print(f"\n训练完成！")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型保存在: {checkpoint_dir}")


if __name__ == '__main__':
    main()



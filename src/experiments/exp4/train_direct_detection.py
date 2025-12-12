#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接检测训练脚本
CAM + 图像特征 → 直接预测框，无需阈值检测
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

from models.direct_detection_detector import create_direct_detection_detector
from datasets.dior_detection import get_detection_dataloader
from losses.direct_detection_loss import DirectDetectionLoss
from losses.improved_direct_detection_loss import ImprovedDirectDetectionLoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    loss_l1_sum = 0.0
    loss_giou_sum = 0.0
    loss_conf_sum = 0.0
    loss_cam_sum = 0.0
    num_batches = 0
    
    pos_ratios = []
    
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
        loss_cam_sum += loss_dict['loss_cam'].item()
        pos_ratios.append(loss_dict['pos_ratio'])
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_l1 = loss_l1_sum / num_batches
    avg_giou = loss_giou_sum / num_batches
    avg_conf = loss_conf_sum / num_batches
    avg_cam = loss_cam_sum / num_batches
    avg_pos_ratio = sum(pos_ratios) / len(pos_ratios) if pos_ratios else 0.0
    
    return avg_loss, avg_l1, avg_giou, avg_conf, avg_cam, avg_pos_ratio


def main():
    parser = argparse.ArgumentParser(description='训练直接检测模型')
    parser.add_argument('--config', type=str, default='configs/direct_detection_config.yaml',
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
    print(f"✅ 使用直接检测方法：CAM + 图像特征 → 直接预测框")
    print(f"   无需阈值检测，端到端训练")
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print(f"\n创建直接检测模型...")
    model = create_direct_detection_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        upsample_cam=config.get('upsample_cam', False),
        device=device,
        unfreeze_cam_last_layer=True,
        use_image_features=config.get('use_image_features', True)
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
    
    # 选择损失函数（根据配置）
    use_improved_loss = config.get('use_improved_loss', False)
    
    if use_improved_loss:
        print("✅ 使用改进的损失函数（移除CAM损失，专注于检测质量）")
        criterion = ImprovedDirectDetectionLoss(
            lambda_l1=config.get('lambda_l1', 1.0),
            lambda_giou=config.get('lambda_giou', 2.0),
            lambda_conf=config.get('lambda_conf', 1.0),
            lambda_cam=config.get('lambda_cam', 0.0),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            pos_radius=config.get('pos_radius', 1.5),
            use_cam_alignment=config.get('use_cam_alignment', False)
        ).to(device)
    else:
        print("使用标准直接检测损失函数")
        criterion = DirectDetectionLoss(
            lambda_l1=config.get('lambda_l1', 1.0),
            lambda_giou=config.get('lambda_giou', 2.0),
            lambda_conf=config.get('lambda_conf', 1.0),
            lambda_cam=config.get('lambda_cam', 0.5),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            pos_radius=config.get('pos_radius', 1.5)
        ).to(device)
    
    # 优化器
    cam_params = []
    detection_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'cam_generator' in name:
                cam_params.append(param)
            else:
                detection_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': cam_params, 'lr': config.get('cam_generator_lr', 5e-5)},
        {'params': detection_params, 'lr': config.get('learning_rate', 1e-4)}
    ], weight_decay=config.get('weight_decay', 0.01))
    
    print(f"\n优化器参数:")
    print(f"  CAM生成器学习率: {config.get('cam_generator_lr', 5e-5)}")
    print(f"  检测头学习率: {config.get('learning_rate', 1e-4)}")
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('num_epochs', 50), eta_min=1e-6
    )
    
    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        criterion.load_state_dict(checkpoint.get('criterion_state_dict', {}))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"从epoch {start_epoch}恢复训练")
    
    # 创建checkpoint目录
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/direct_detection'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练日志
    log_file = checkpoint_dir / f'training_direct_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 训练循环
    print(f"\n开始训练（直接检测方法）...")
    print(f"总epoch数: {config.get('num_epochs', 50)}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"日志文件: {log_file}")
    
    for epoch in range(start_epoch, config.get('num_epochs', 50)):
        avg_loss, avg_l1, avg_giou, avg_conf, avg_cam, avg_pos_ratio = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        scheduler.step()
        
        # 记录日志
        log_msg = (f"Epoch [{epoch+1}/{config.get('num_epochs', 50)}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"L1: {avg_l1:.4f} | "
                  f"GIoU: {avg_giou:.4f} | "
                  f"Conf: {avg_conf:.4f} | "
                  f"CAM: {avg_cam:.4f} | "
                  f"PosRatio: {avg_pos_ratio:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
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
        
        torch.save(checkpoint, checkpoint_dir / 'latest_direct_detection_model.pth')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint['best_loss'] = best_loss
            torch.save(checkpoint, checkpoint_dir / 'best_direct_detection_model.pth')
            print(f"✅ 保存最佳模型 (loss: {best_loss:.4f})")
    
    print(f"\n训练完成！")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型保存在: {checkpoint_dir}")


if __name__ == '__main__':
    main()


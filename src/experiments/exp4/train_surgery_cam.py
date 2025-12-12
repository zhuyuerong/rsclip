#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SurgeryCAM Training Script
训练轻量BoxHead，冻结SurgeryCLIP+AAF+CAMGenerator
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

# Add project root for imports
project_root = current_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.surgery_cam_detector import create_surgery_cam_detector
from datasets.dior_detection import get_detection_dataloader
from losses.detection_loss import DetectionLoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """训练一个epoch"""
    model.train()
    
    # 注意: SurgeryCLIP+AAF+CAM冻结,只有BoxHead在train mode
    # 但SurgeryAAF已经在eval mode，所以不需要额外设置
    
    total_loss = 0.0
    loss_l1_sum = 0.0
    loss_giou_sum = 0.0
    loss_cam_sum = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config['num_epochs']}]")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)  # [B, 3, H, W]
        text_queries = batch['text_queries']  # List[str] 所有训练类
        boxes = batch['boxes']  # List of [K, 4]
        labels = batch['labels']  # List of [K]
        
        # 准备targets
        targets = []
        for b in range(len(boxes)):
            targets.append({
                'boxes': boxes[b].to(device),
                'labels': labels[b].to(device)
            })
        
        # 前向
        outputs = model(images, text_queries)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss_total']
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (重要!)
        torch.nn.utils.clip_grad_norm_(model.box_head.parameters(), 1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        loss_l1_sum += loss_dict['loss_box_l1'].item()
        loss_giou_sum += loss_dict['loss_box_giou'].item()
        loss_cam_sum += loss_dict['loss_cam'].item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l1': f'{loss_dict["loss_box_l1"].item():.4f}',
            'giou': f'{loss_dict["loss_box_giou"].item():.4f}',
            'cam': f'{loss_dict["loss_cam"].item():.4f}'
        })
        
        # Logging
        if batch_idx % config.get('log_interval', 50) == 0:
            print(f"\nEpoch [{epoch+1}/{config['num_epochs']}] "
                  f"Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} | "
                  f"L1: {loss_dict['loss_box_l1'].item():.4f} | "
                  f"GIoU: {loss_dict['loss_box_giou'].item():.4f} | "
                  f"CAM: {loss_dict['loss_cam'].item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_l1 = loss_l1_sum / num_batches if num_batches > 0 else 0.0
    avg_giou = loss_giou_sum / num_batches if num_batches > 0 else 0.0
    avg_cam = loss_cam_sum / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'loss_l1': avg_l1,
        'loss_giou': avg_giou,
        'loss_cam': avg_cam
    }


def main():
    parser = argparse.ArgumentParser(description='Train SurgeryCAM Detector')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Creating default config...")
        create_default_config(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config.get('device', 'cuda')
    
    # ===== 1. Load pre-trained SurgeryCLIP =====
    print("=" * 80)
    print("Loading SurgeryCAM Detector...")
    print("=" * 80)
    
    checkpoint_path = config['surgery_clip_checkpoint']
    if not os.path.isabs(checkpoint_path):
        project_root = Path(__file__).parent.parent.parent.parent
        checkpoint_path = project_root / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)
    
    model = create_surgery_cam_detector(
        surgery_clip_checkpoint=checkpoint_path,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        upsample_cam=config.get('upsample_cam', False),
        device=device
    )
    
    # ===== 2. Data loading =====
    print("\nLoading data...")
    # 训练时只使用seen类别
    train_only_seen = config.get('train_only_seen', True)
    print(f"训练模式: {'只在seen类别上训练' if train_only_seen else '所有类别训练'}")
    
    train_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='trainval',
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment=config.get('augment', True),
        train_only_seen=train_only_seen  # 训练时只使用seen类别
    )
    
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment=False
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # ===== 3. Loss and Optimizer =====
    criterion = DetectionLoss(
        lambda_l1=config.get('lambda_l1', 1.0),
        lambda_giou=config.get('lambda_giou', 2.0),
        lambda_cam=config.get('lambda_cam', 0.5),
        min_peak_distance=config.get('min_peak_distance', 2),
        min_peak_value=config.get('min_peak_value', 0.3),
        match_iou_threshold=config.get('match_iou_threshold', 0.3)
    )
    
    optimizer = torch.optim.AdamW(
        model.box_head.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('num_epochs', 50)
    )
    
    # ===== 4. Resume from checkpoint =====
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.box_head.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # ===== 5. Training loop =====
    print("\n" + "=" * 80)
    print("Start training...")
    print("=" * 80)
    
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config.get('num_epochs', 50)):
        # Training phase
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{config.get('num_epochs', 50)}]")
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"L1: {train_metrics['loss_l1']:.4f} | "
              f"GIoU: {train_metrics['loss_giou']:.4f} | "
              f"CAM: {train_metrics['loss_cam']:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*80}\n")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.box_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_metrics['loss'],
            'config': config
        }
        
        # Save best model
        if train_metrics['loss'] < best_val_loss:
            best_val_loss = train_metrics['loss']
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print("✅ Best model saved!")
        
        # Periodic checkpoint
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"✅ Checkpoint saved: epoch_{epoch+1}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


def create_default_config(config_path: Path):
    """Create default config file"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        'surgery_clip_checkpoint': 'checkpoints/RemoteCLIP-ViT-B-32.pt',
        'num_classes': 20,
        'cam_resolution': 7,
        'upsample_cam': False,
        'device': 'cuda',
        'batch_size': 8,
        'num_workers': 4,
        'image_size': 224,
        'augment': True,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'lambda_l1': 1.0,
        'lambda_giou': 2.0,
        'lambda_cam': 0.5,
        'min_peak_distance': 2,
        'min_peak_value': 0.3,
        'match_iou_threshold': 0.3,
        'log_interval': 50,
        'save_interval': 10,
        'checkpoint_dir': 'checkpoints',
        'dataset_root': None
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)


if __name__ == '__main__':
    main()


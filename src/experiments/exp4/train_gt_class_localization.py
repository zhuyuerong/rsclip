#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GT类别定位训练脚本
只训练检测头，使用GT类别文本，验证定位能力
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

from models.gt_class_localization_detector import create_gt_class_localization_detector
from datasets.dior_detection import get_detection_dataloader
from losses.gt_class_localization_loss import GTClassLocalizationLoss

# DIOR类别列表
DIOR_CLASSES = [
    "airplane", "airport", "baseball field", "basketball court",
    "bridge", "chimney", "dam", "expressway service area",
    "expressway toll station", "golf course", "ground track field",
    "harbor", "overpass", "ship", "stadium", "storage tank",
    "tennis court", "train station", "vehicle", "wind mill"
]


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
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config['num_epochs']}]")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        boxes = batch['boxes']
        labels = batch['labels']
        
        # 准备targets
        targets = []
        for b in range(len(boxes)):
            targets.append({
                'boxes': boxes[b].to(device),
                'labels': labels[b].to(device)
            })
        
        # 前向传播（使用所有类别文本，损失函数会只使用GT类别通道）
        outputs = model(images)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss_total']
        
        # 反向传播（只更新检测头参数）
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        loss_l1_sum += loss_dict['loss_box_l1'].item()
        loss_giou_sum += loss_dict['loss_box_giou'].item()
        num_batches += 1
        
        # 更新进度条
        if batch_idx % config.get('log_interval', 50) == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'l1': f"{loss_dict['loss_box_l1'].item():.4f}",
                'giou': f"{loss_dict['loss_box_giou'].item():.4f}",
                'pos': loss_dict['num_pos_samples']
            })
    
    avg_loss = total_loss / num_batches
    avg_l1 = loss_l1_sum / num_batches
    avg_giou = loss_giou_sum / num_batches
    
    return avg_loss, avg_l1, avg_giou


def main():
    parser = argparse.ArgumentParser(description='训练GT类别定位检测器')
    parser.add_argument('--config', type=str, default='configs/gt_class_localization_config.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device_str = config.get('device', 'cuda')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"使用设备: {device}")
    print(f"✅ GT类别定位实验：只训练检测头，使用GT类别文本")
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print(f"\n创建GT类别定位检测器...")
    model = create_gt_class_localization_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        device=device
    )
    model.to(device)
    
    # 验证只有检测头参数可训练
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n✅ 可训练参数数量: {len(trainable_params)}")
    print(f"   只训练检测头和原图编码器")
    
    # 加载数据
    train_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='trainval',
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment=config.get('augment', True),
        train_only_seen=config.get('train_only_seen', True)
    )
    
    # 创建损失函数
    criterion = GTClassLocalizationLoss(
        lambda_l1=config.get('lambda_l1', 1.0),
        lambda_giou=config.get('lambda_giou', 2.0),
        pos_radius=config.get('pos_radius', 1.5),
        pos_iou_threshold=config.get('pos_iou_threshold', 0.3)
    ).to(device)
    
    print(f"\n损失函数配置:")
    print(f"  L1权重: {config.get('lambda_l1', 1.0)}")
    print(f"  GIoU权重: {config.get('lambda_giou', 2.0)}")
    print(f"  只计算框回归损失（无置信度损失）")
    
    # 优化器（只优化检测头参数）
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get('detection_head_lr', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    print(f"\n优化器参数:")
    print(f"  检测头学习率: {config.get('detection_head_lr', 1e-4)}")
    print(f"  权重衰减: {config.get('weight_decay', 0.01)}")
    
    # 学习率调度器
    num_warmup_steps = config.get('warmup_epochs', 5) * len(train_loader)
    num_training_steps = config.get('num_epochs', 50) * len(train_loader)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    print(f"\n学习率调度:")
    print(f"  Warmup epochs: {config.get('warmup_epochs', 5)}")
    print(f"  总训练步数: {num_training_steps}")
    
    # 创建checkpoint目录
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/gt_class_localization'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练日志
    log_file = checkpoint_dir / f'training_gt_class_localization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 训练循环
    print(f"\n开始训练（GT类别定位）...")
    print(f"总epoch数: {config.get('num_epochs', 50)}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"日志文件: {log_file}")
    
    best_loss = float('inf')
    
    for epoch in range(config.get('num_epochs', 50)):
        avg_loss, avg_l1, avg_giou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        scheduler.step()
        
        # 记录日志
        log_msg = (f"Epoch [{epoch+1}/{config.get('num_epochs', 50)}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"L1: {avg_l1:.4f} | "
                  f"GIoU: {avg_giou:.4f} | "
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
        
        torch.save(checkpoint, checkpoint_dir / 'latest_gt_class_localization_model.pth')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint['best_loss'] = best_loss
            torch.save(checkpoint, checkpoint_dir / 'best_gt_class_localization_model.pth')
            print(f"✅ 保存最佳模型 (loss: {best_loss:.4f})")
    
    print(f"\n训练完成！")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型保存在: {checkpoint_dir}")


if __name__ == '__main__':
    main()



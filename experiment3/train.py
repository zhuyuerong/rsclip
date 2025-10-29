#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVA-DETR训练脚本

功能：
1. 训练OVA-DETR模型
2. 支持断点续训
3. 保存检查点
4. 可视化训练过程
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import sys

from config.default_config import DefaultConfig
from models.ova_detr import OVADETR
from models.criterion import SetCriterion
from utils.data_loader import create_data_loader, DIOR_CLASSES
from utils.transforms import get_transforms


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader,
    optimizer,
    device,
    epoch: int,
    text_features: torch.Tensor,
    writer: SummaryWriter = None
):
    """
    训练一个epoch
    
    参数:
        model: 模型
        criterion: 损失函数
        data_loader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        text_features: 文本特征
        writer: TensorBoard writer
    
    返回:
        avg_loss: 平均损失
    """
    model.train()
    criterion.train()
    
    total_loss = 0
    total_loss_cls = 0
    total_loss_bbox = 0
    total_loss_giou = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # 移动到设备
        images = images.to(device)
        for target in targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
        
        # 前向传播
        outputs = model(images, text_features)
        
        # 计算损失
        losses = criterion(outputs, targets)
        loss = losses['loss_total']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        optimizer.step()
        
        # 累积损失
        total_loss += loss.item()
        total_loss_cls += losses['loss_cls'].item()
        total_loss_bbox += losses['loss_bbox'].item()
        total_loss_giou += losses['loss_giou'].item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{losses['loss_cls'].item():.4f}",
            'bbox': f"{losses['loss_bbox'].item():.4f}",
            'giou': f"{losses['loss_giou'].item():.4f}"
        })
        
        # TensorBoard记录
        if writer is not None:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Train/loss_total', loss.item(), global_step)
            writer.add_scalar('Train/loss_cls', losses['loss_cls'].item(), global_step)
            writer.add_scalar('Train/loss_bbox', losses['loss_bbox'].item(), global_step)
            writer.add_scalar('Train/loss_giou', losses['loss_giou'].item(), global_step)
    
    # 计算平均损失
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_loss_cls = total_loss_cls / num_batches
    avg_loss_bbox = total_loss_bbox / num_batches
    avg_loss_giou = total_loss_giou / num_batches
    
    print(f"\nEpoch {epoch} 平均损失:")
    print(f"  总损失: {avg_loss:.4f}")
    print(f"  分类损失: {avg_loss_cls:.4f}")
    print(f"  边界框损失: {avg_loss_bbox:.4f}")
    print(f"  GIoU损失: {avg_loss_giou:.4f}")
    
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader,
    device,
    text_features: torch.Tensor
):
    """
    验证
    
    返回:
        avg_loss: 平均损失
    """
    model.eval()
    criterion.eval()
    
    total_loss = 0
    total_loss_cls = 0
    total_loss_bbox = 0
    total_loss_giou = 0
    
    pbar = tqdm(data_loader, desc='Validation')
    
    for images, targets in pbar:
        # 移动到设备
        images = images.to(device)
        for target in targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)
        
        # 前向传播
        outputs = model(images, text_features)
        
        # 计算损失
        losses = criterion(outputs, targets)
        
        # 累积损失
        total_loss += losses['loss_total'].item()
        total_loss_cls += losses['loss_cls'].item()
        total_loss_bbox += losses['loss_bbox'].item()
        total_loss_giou += losses['loss_giou'].item()
    
    # 计算平均损失
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_loss_cls = total_loss_cls / num_batches
    avg_loss_bbox = total_loss_bbox / num_batches
    avg_loss_giou = total_loss_giou / num_batches
    
    print(f"\n验证损失:")
    print(f"  总损失: {avg_loss:.4f}")
    print(f"  分类损失: {avg_loss_cls:.4f}")
    print(f"  边界框损失: {avg_loss_bbox:.4f}")
    print(f"  GIoU损失: {avg_loss_giou:.4f}")
    
    return avg_loss


def main(args):
    """主函数"""
    
    print("=" * 70)
    print("OVA-DETR训练")
    print("=" * 70)
    
    # 配置
    config = DefaultConfig()
    
    # 覆盖配置
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs')
    
    # ==================== 数据加载 ====================
    print("\n" + "=" * 70)
    print("加载数据")
    print("=" * 70)
    
    train_transforms = get_transforms(mode='train', image_size=config.image_size)
    val_transforms = get_transforms(mode='val', image_size=config.image_size)
    
    train_loader = create_data_loader(
        root_dir=args.data_dir,
        split='train',
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        transforms=train_transforms  # 已经传递了
    )
    
    val_loader = create_data_loader(
        root_dir=args.data_dir,
        split='val',
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        transforms=val_transforms  # 已经传递了
    )
    
    print(f"\n训练集: {len(train_loader.dataset)}张图片, {len(train_loader)}个批次")
    print(f"验证集: {len(val_loader.dataset)}张图片, {len(val_loader)}个批次")
    
    # ==================== 模型创建 ====================
    print("\n" + "=" * 70)
    print("创建模型")
    print("=" * 70)
    
    model = OVADETR(config).to(device)
    criterion = SetCriterion(config).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # ==================== 文本特征 ====================
    print("\n提取文本特征...")
    with torch.no_grad():
        text_features = model.backbone.forward_text(DIOR_CLASSES)
        text_features = text_features.to(device)
    
    print(f"文本特征: {text_features.shape}")
    
    # ==================== 优化器 ====================
    # 分组参数
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'backbone' not in n and p.requires_grad],
            'lr': config.learning_rate
        }
    ]
    
    # 如果不冻结backbone，添加较小的学习率
    if not config.freeze_remoteclip:
        param_groups.append({
            'params': [p for n, p in model.named_parameters() 
                      if 'backbone' in n and p.requires_grad],
            'lr': config.learning_rate * 0.1
        })
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.1
    )
    
    # ==================== 训练 ====================
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*70}")
        
        # 训练
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch, text_features, writer
        )
        
        # 验证
        val_loss = validate(model, criterion, val_loader, device, text_features)
        
        # 学习率调度
        lr_scheduler.step()
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/learning_rate', current_lr, epoch)
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        
        # 保存最新
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # 保存最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            print(f"\n✅ 保存最佳模型 (val_loss: {val_loss:.4f})")
        
        # 定期保存
        if epoch % 10 == 0:
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch}.pth')
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {checkpoint_dir}")
    print("=" * 70)
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OVA-DETR训练')
    
    parser.add_argument('--data_dir', type=str,
                       default='/home/ubuntu22/Projects/RemoteCLIP-main/datasets/DIOR',
                       help='数据集目录')
    parser.add_argument('--output_dir', type=str,
                       default='./outputs',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    
    args = parser.parse_args()
    
    main(args)


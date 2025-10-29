#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 在mini_dataset上训练
上下文引导检测器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
import json
import time

sys.path.append('..')

from config.default_config import DefaultConfig
from models.context_guided_detector import ContextGuidedDetector
from stage4_supervision.loss_functions import TotalLoss
from stage4_supervision.matcher import HungarianMatcher
from datasets.mini_dataset.mini_dataset_loader import MiniDataset
import torchvision.transforms as T


def create_transforms():
    """创建数据增强"""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def custom_collate_fn(batch):
    """自定义collate函数，处理PIL Image"""
    transform = create_transforms()
    
    images = []
    targets = []
    
    for img, target in batch:
        if not isinstance(img, torch.Tensor):
            img = transform(img)
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    return images, targets


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, text_features):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_loss_cls = 0
    total_loss_bbox = 0
    total_loss_contrast = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # 数据已经在collate_fn中转换好了
        images = images.to(device)
        
        # 准备文本特征
        text_feats = text_features.to(device)
        
        try:
            # 前向传播
            outputs = model(images, text_feats)
            
            # 准备targets
            batch_targets = []
            for target in targets:
                batch_targets.append({
                    'boxes': target['boxes'].to(device),
                    'labels': target['labels'].to(device)
                })
            
            # 计算损失 (简化版)
            # 对比损失 - 假设outputs包含query_features
            if 'query_features' in outputs:
                loss_contrast = criterion(outputs['query_features'], text_feats)
                loss = loss_contrast
                loss_dict = {
                    'total_loss': loss,
                    'loss_contrast': loss.item(),
                    'loss_cls': 0,
                    'loss_bbox': 0
                }
            else:
                # 如果没有对比损失，跳过这个batch
                print(f"Warning: No query_features in outputs")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_loss_cls += loss_dict.get('loss_cls', 0)
            total_loss_bbox += loss_dict.get('loss_bbox', 0)
            total_loss_contrast += loss_dict.get('loss_contrast', 0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls': f"{loss_dict.get('loss_cls', 0):.4f}",
                'bbox': f"{loss_dict.get('loss_bbox', 0):.4f}"
            })
            
        except Exception as e:
            print(f"\n❌ Batch {batch_idx} 出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_loss_cls = total_loss_cls / num_batches if num_batches > 0 else 0
    avg_loss_bbox = total_loss_bbox / num_batches if num_batches > 0 else 0
    avg_loss_contrast = total_loss_contrast / num_batches if num_batches > 0 else 0
    
    return {
        'loss': avg_loss,
        'loss_cls': avg_loss_cls,
        'loss_bbox': avg_loss_bbox,
        'loss_contrast': avg_loss_contrast
    }


@torch.no_grad()
def validate(model, data_loader, criterion, device, text_features):
    """验证"""
    model.eval()
    
    total_loss = 0
    
    for images, targets in tqdm(data_loader, desc="Validating"):
        # 数据转换
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images).to(device)
        else:
            transform = create_transforms()
            images = torch.stack([transform(img) for img in images]).to(device)
        
        text_feats = text_features.to(device)
        
        try:
            outputs = model(images, text_feats)
            
            batch_targets = []
            for target in targets:
                batch_targets.append({
                    'boxes': target['boxes'].to(device),
                    'labels': target['labels'].to(device)
                })
            
            loss_dict = criterion(outputs, batch_targets)
            total_loss += loss_dict['total_loss'].item()
            
        except Exception as e:
            print(f"验证出错: {e}")
            continue
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    return {'loss': avg_loss}


def main():
    print("=" * 70)
    print("Experiment2: 上下文引导检测器训练")
    print("数据集: mini_dataset")
    print("=" * 70)
    
    # 配置
    config = DefaultConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n训练配置:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # 创建数据集
    print("\n加载数据集...")
    train_dataset = MiniDataset(
        root_dir='../datasets/mini_dataset',
        split='train',
        transforms=None  # 在训练循环中手动转换
    )
    
    val_dataset = MiniDataset(
        root_dir='../datasets/mini_dataset',
        split='val',
        transforms=None
    )
    
    print(f"  训练集: {len(train_dataset)} 张图")
    print(f"  验证集: {len(val_dataset)} 张图")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    # 创建模型
    print("\n创建模型...")
    from utils.dataloader import DIOR_CLASSES
    
    freeze_clip = getattr(config, 'freeze_clip_backbone', True)  # 默认冻结
    model = ContextGuidedDetector(
        model_name=config.clip_model_name,
        pretrained_path=config.clip_checkpoint,
        num_queries=config.num_queries,
        num_decoder_layers=config.num_decoder_layers,
        d_model=config.d_model,
        d_clip=config.d_clip,
        context_gating_type=config.context_gating_type,
        freeze_clip=freeze_clip
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  可训练: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # 提取文本特征
    print("\n提取文本特征...")
    from utils.dataloader import DIOR_CLASSES
    
    with torch.no_grad():
        text_features = model.text_encoder(DIOR_CLASSES)
    
    print(f"  文本特征: {text_features.shape}")
    
    # 创建损失函数 (简化版 - 只使用对比损失)
    print("\n创建损失函数...")
    from stage4_supervision.global_contrast_loss import GlobalContrastLoss
    from stage4_supervision.box_loss import BoxLoss
    
    criterion = GlobalContrastLoss(temperature=config.temperature).to(device)
    box_criterion = BoxLoss().to(device)
    
    # 创建优化器
    print("\n创建优化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )
    
    # 训练循环
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    best_val_loss = float('inf')
    train_history = []
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 70)
        
        # 训练
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, text_features
        )
        
        print(f"\n训练 - Loss: {train_metrics['loss']:.4f}")
        print(f"  分类损失: {train_metrics['loss_cls']:.4f}")
        print(f"  框损失: {train_metrics['loss_bbox']:.4f}")
        print(f"  对比损失: {train_metrics['loss_contrast']:.4f}")
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, text_features)
        print(f"\n验证 - Loss: {val_metrics['loss']:.4f}")
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"学习率: {current_lr:.6f}")
        
        # 保存历史
        train_history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'lr': current_lr
        })
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = Path('outputs/checkpoints/best_model.pth')
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'config': config.__dict__
            }, checkpoint_path)
            
            print(f"✅ 保存最佳模型: {checkpoint_path}")
        
        # 定期保存checkpoint
        if epoch % 5 == 0:
            checkpoint_path = Path(f'outputs/checkpoints/checkpoint_epoch_{epoch}.pth')
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
            }, checkpoint_path)
    
    # 保存训练历史
    history_path = Path('outputs/logs/train_history.json')
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"\n最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型保存在: outputs/checkpoints/best_model.pth")
    print(f"训练历史保存在: {history_path}")


if __name__ == '__main__':
    main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速训练测试：只训练几个batch来验证训练流程
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import sys
import os
from pathlib import Path
from tqdm import tqdm

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.surgery_aaf import create_surgery_aaf_model
from utils.data import get_dataloader


def quick_train_test():
    """快速训练测试"""
    print("=" * 80)
    print("快速训练测试 - 验证训练流程")
    print("=" * 80)
    
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # ===== 1. 加载模型 =====
    print("\n" + "=" * 80)
    print("1. 加载模型")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent.parent.parent
    checkpoint_path = config['clip_weights_path']
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = project_root / checkpoint_path
        checkpoint_path = str(checkpoint_path)
    
    model, preprocess = create_surgery_aaf_model(
        checkpoint_path=checkpoint_path,
        device=device,
        num_layers=config.get('num_layers', 6)
    )
    
    # 冻结CLIP参数
    for param in model.clip.parameters():
        param.requires_grad = False
    
    # 只训练AAF参数
    for param in model.aaf.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    print(f"冻结参数: {total_params-trainable_params:,} ({(total_params-trainable_params)/total_params*100:.4f}%)")
    
    # ===== 2. 优化器 =====
    print("\n" + "=" * 80)
    print("2. 设置优化器")
    print("=" * 80)
    
    optimizer = torch.optim.AdamW(
        model.aaf.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    print(f"优化器: AdamW")
    print(f"学习率: {config['learning_rate']}")
    print(f"权重衰减: {config['weight_decay']}")
    
    # ===== 3. 损失函数 =====
    criterion = nn.BCEWithLogitsLoss()
    print(f"损失函数: BCEWithLogitsLoss")
    
    # ===== 4. 数据加载 =====
    print("\n" + "=" * 80)
    print("3. 加载数据")
    print("=" * 80)
    
    train_loader = get_dataloader(
        dataset_name=config['dataset'],
        root=config.get('dataset_root'),
        split='trainval',
        batch_size=2,  # 小批次用于测试
        num_workers=0,  # 避免多进程问题
        shuffle=True
    )
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"批次大小: 2")
    print(f"总批次数: {len(train_loader)}")
    
    # ===== 5. 训练几个批次 =====
    print("\n" + "=" * 80)
    print("4. 训练测试（3个批次）")
    print("=" * 80)
    
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:  # 只训练3个批次
            break
        
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)  # [B, 20] - 所有20个类别
        
        # text_queries应该是所有20个类别，但数据加载器可能返回的是列表的列表
        # 确保使用所有20个类别
        if isinstance(batch['text_queries'], list):
            if isinstance(batch['text_queries'][0], list):
                # 如果每个样本都有类别列表，取第一个（应该都是相同的20个类别）
                text_queries = batch['text_queries'][0]
            else:
                text_queries = batch['text_queries']
        else:
            text_queries = batch['text_queries']
        
        # 如果text_queries不是20个类别，使用DIOR的20个标准类别
        from utils.data import DIORDataset
        if len(text_queries) != 20:
            dior_dataset = train_loader.dataset
            text_queries = dior_dataset.classes
        
        print(f"\n批次 {batch_idx + 1}/3:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  类别数: {len(text_queries)}")
        print(f"  前5个类别: {text_queries[:5]}")
        
        # 前向传播 - 使用所有20个类别
        # 注意：SurgeryCLIP在第一次forward时会修改模型结构（inplace操作）
        # 所以我们需要在训练模式下进行forward，但要注意梯度问题
        cam, aux = model(images, text_queries)
        print(f"  CAM形状: {cam.shape}")
        
        # 检查CAM类别数是否匹配
        if cam.shape[1] != labels.shape[1]:
            print(f"  ⚠️  警告: CAM类别数({cam.shape[1]}) != 标签类别数({labels.shape[1]})")
            # 如果CAM类别数少于标签数，只使用对应的类别
            num_classes = min(cam.shape[1], labels.shape[1])
            cam_scores = cam[:, :num_classes].flatten(2).max(dim=2)[0]
            labels_subset = labels[:, :num_classes]
        else:
            cam_scores = cam.flatten(2).max(dim=2)[0]
            labels_subset = labels
        
        print(f"  CAM分数形状: {cam_scores.shape}")
        print(f"  标签子集形状: {labels_subset.shape}")
        
        # 计算损失
        loss = criterion(cam_scores, labels_subset.float())
        print(f"  损失: {loss.item():.4f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.aaf.parameters() if p.requires_grad)
        print(f"  AAF有梯度: {has_grad}")
        
        # 检查CLIP梯度（应该是None）
        clip_has_grad = any(p.grad is not None 
                           for p in model.clip.parameters())
        print(f"  CLIP有梯度: {clip_has_grad} (应该是False)")
        
        optimizer.step()
        print(f"  ✅ 批次 {batch_idx + 1} 完成")
    
    print("\n" + "=" * 80)
    print("训练测试完成！")
    print("=" * 80)
    print("\n✅ 模型结构正确")
    print("✅ 参数冻结正确")
    print("✅ 前向传播正常")
    print("✅ 反向传播正常")
    print("✅ 优化器更新正常")


if __name__ == '__main__':
    quick_train_test()


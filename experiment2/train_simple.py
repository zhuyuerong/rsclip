#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 简化训练脚本
只训练编码器，快速验证流程
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
import json

sys.path.append('..')

from config.default_config import DefaultConfig
from stage1_encoder.clip_text_encoder import CLIPTextEncoder
from stage1_encoder.clip_image_encoder import CLIPImageEncoder
from stage4_supervision.global_contrast_loss import GlobalContrastLoss
from datasets.mini_dataset.mini_dataset_loader import MiniDataset
import torchvision.transforms as T


def collate_fn(batch):
    """转换图像为tensor"""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images, targets = [], []
    for img, target in batch:
        images.append(transform(img))
        targets.append(target)
    
    return torch.stack(images), targets


def train():
    print("=" * 70)
    print("Experiment2 简化训练 (只训练编码器)")
    print("=" * 70)
    
    device = torch.device('cuda')
    config = DefaultConfig()
    
    # 数据
    print("\n加载数据...")
    train_dataset = MiniDataset('../datasets/mini_dataset', 'train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # 模型
    print("\n创建模型...")
    text_encoder = CLIPTextEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt').cuda()
    image_encoder = CLIPImageEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt', freeze=False).cuda()
    
    # 确保可训练
    for param in image_encoder.parameters():
        param.requires_grad = True
    image_encoder.train()
    
    # 损失
    criterion = GlobalContrastLoss(temperature=0.07).cuda()
    
    # 优化器
    optimizer = torch.optim.AdamW(image_encoder.parameters(), lr=1e-4)
    
    # 提取文本特征
    from utils.dataloader import DIOR_CLASSES
    with torch.no_grad():
        text_features = text_encoder(DIOR_CLASSES).cuda()  # [20, 1024]
    
    print(f"文本特征: {text_features.shape}")
    print(f"\n开始训练 (10 epochs)...")
    
    for epoch in range(1, 11):
        total_loss = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.cuda()
            
            # 前向
            _, image_features = image_encoder(images)  # [B, 1024]
            
            # 计算loss (简单的对比学习)
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            logits = image_features @ text_features_norm.T  # [B, 20]
            
            # 简单的对比损失
            labels = torch.tensor([target['labels'][0] for target in targets if len(target['labels']) > 0]).cuda()
            
            if len(labels) > 0:
                loss = nn.CrossEntropyLoss()(logits[:len(labels)] / 0.07, labels)
            else:
                continue
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
        
        # 保存
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'image_encoder': image_encoder.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'loss': avg_loss
            }, f'outputs/checkpoints/simple_epoch_{epoch}.pth')
            print(f"✅ 保存 checkpoint")
    
    print("\n✅ 训练完成！")


if __name__ == '__main__':
    train()


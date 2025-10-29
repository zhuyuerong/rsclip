#!/usr/bin/env python3
"""简化的测试脚本，检查维度问题"""

import torch
import sys
sys.path.append('..')

from config.default_config import DefaultConfig
from models.ova_detr import OVADETR
from models.criterion import SetCriterion
from utils.data_loader import create_data_loader, DIOR_CLASSES

config = DefaultConfig()
device = torch.device('cuda')

print("加载数据...")
train_loader = create_data_loader(
    root_dir='../datasets/DIOR',
    split='train',
    batch_size=1,  # 单个样本测试
    num_workers=0
)

print("创建模型...")
model = OVADETR(config).to(device)
criterion = SetCriterion(config).to(device)

print("提取文本特征...")
with torch.no_grad():
    text_features = model.backbone.forward_text(DIOR_CLASSES).to(device)
print(f"text_features: {text_features.shape}")

# 测试一个batch
print("\n测试forward...")
images, targets = next(iter(train_loader))
images = images.to(device)
for target in targets:
    target['boxes'] = target['boxes'].to(device)
    target['labels'] = target['labels'].to(device)
    
print(f"images: {images.shape}")
print(f"targets[0]['boxes']: {targets[0]['boxes'].shape}")
print(f"targets[0]['labels']: {targets[0]['labels'].shape}")

try:
    outputs = model(images, text_features)
    print("\n模型输出:")
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    
    print("\n计算损失...")
    losses = criterion(outputs, targets)
    print("✅ 损失计算成功！")
    print(f"  loss_cls: {losses['loss_cls'].item():.4f}")
    print(f"  loss_bbox: {losses['loss_bbox'].item():.4f}")
    print(f"  loss_giou: {losses['loss_giou'].item():.4f}")
    print(f"  loss_total: {losses['loss_total'].item():.4f}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()


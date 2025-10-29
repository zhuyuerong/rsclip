#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 正确版本 - 完整DIOR数据集训练
文本驱动的位置预测 + 全局-局部对比学习
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
from stage1_encoder.clip_text_encoder import CLIPTextEncoder
from stage1_encoder.clip_image_encoder import CLIPImageEncoder
from utils.dataloader import DIORDataset, DIOR_CLASSES
import torchvision.transforms as T


class TextToPositionProjector(nn.Module):
    """文本到位置的投影网络"""
    def __init__(self, text_dim=1024, hidden_dim=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )
    
    def forward(self, text_features):
        return self.projector(text_features)


class AdaptiveGlobalLocalContrastLoss(nn.Module):
    """自适应全局-局部对比损失"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, local_features, text_features, global_features, labels):
        # 归一化
        local_features = local_features / (local_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        global_features = global_features / (global_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 局部-文本相似度
        local_text_sim = (local_features @ text_features.T) / self.temperature
        
        # 局部-全局相似度（背景）
        local_global_sim = (local_features * global_features).sum(dim=-1, keepdim=True) / self.temperature
        
        # 对比学习
        logits = torch.cat([local_text_sim, local_global_sim], dim=-1)
        loss = nn.CrossEntropyLoss()(logits[:, :-1], labels)
        
        return loss


class SimpleDeformableQueryExtractor(nn.Module):
    """Query提取器"""
    def __init__(self, d_model=1024):
        super().__init__()
        self.position_embed = nn.Linear(4, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, global_features, boxes):
        pos_embed = self.position_embed(boxes)
        combined = torch.cat([global_features, pos_embed], dim=-1)
        return self.fusion(combined)


class BoxRegressor(nn.Module):
    """框回归器"""
    def __init__(self, d_model=1024):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )
    
    def forward(self, local_features):
        return self.regressor(local_features)


def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """GIoU"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)
    
    lti = torch.min(boxes1[:, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    
    whi = (rbi - lti).clamp(min=0)
    areai = whi[:, 0] * whi[:, 1]
    
    giou = iou - (areai - union) / (areai + 1e-6)
    
    return giou


def train():
    print("=" * 70)
    print("Experiment2 正确版本 - 完整DIOR数据集训练")
    print("文本驱动位置预测 + 全局-局部对比学习")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    # 训练配置
    batch_size = 8
    num_epochs = 100
    num_workers = 4
    
    print(f"\n配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Workers: {num_workers}")
    
    # 数据
    print("\n加载数据...")
    train_dataset = DIORDataset(
        root_dir='../datasets/DIOR',
        split='train',
        image_size=(224, 224),
        augment=True  # 训练时使用数据增强
    )
    val_dataset = DIORDataset(
        root_dir='../datasets/DIOR',
        split='val',
        image_size=(224, 224),
        augment=False  # 验证时不使用数据增强
    )
    
    # 自定义collate函数
    def collate_fn(batch):
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        return torch.stack(images), targets
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"  训练集: {len(train_dataset)} 张图")
    print(f"  验证集: {len(val_dataset)} 张图")
    
    # 创建模型
    print("\n创建模型...")
    text_encoder = CLIPTextEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt').cuda()
    image_encoder = CLIPImageEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt', freeze=False).cuda()
    text_to_position = TextToPositionProjector(text_dim=1024).cuda()
    query_extractor = SimpleDeformableQueryExtractor(d_model=1024).cuda()
    box_regressor = BoxRegressor(d_model=1024).cuda()
    
    for param in image_encoder.parameters():
        param.requires_grad = True
    image_encoder.train()
    
    print(f"  ✅ Text Encoder (RemoteCLIP, 冻结)")
    print(f"  ✅ Image Encoder (RemoteCLIP, 可训练)")
    print(f"  ✅ Text-to-Position Projector ⭐")
    print(f"  ✅ Query Extractor")
    print(f"  ✅ Box Regressor")
    
    # 损失
    contrast_criterion = AdaptiveGlobalLocalContrastLoss(temperature=0.07).cuda()
    
    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': image_encoder.parameters(), 'lr': 5e-6},
        {'params': text_to_position.parameters(), 'lr': 2e-4},  # 更高学习率
        {'params': query_extractor.parameters(), 'lr': 1e-4},
        {'params': box_regressor.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # 提取文本特征
    with torch.no_grad():
        text_features = text_encoder(DIOR_CLASSES).cuda()
    
    print(f"  文本特征: {text_features.shape}")
    
    print(f"\n开始训练 ({num_epochs} epochs)...")
    print("  1. 文本tc → 位置投影 → 初始框")
    print("  2. 初始框 + Ig → Query Extractor → fm")
    print("  3. fm vs tc vs Ig → 对比损失")
    print("  4. fm → Box Regressor → 精修框")
    
    train_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        image_encoder.train()
        text_to_position.train()
        query_extractor.train()
        box_regressor.train()
        
        total_loss = 0
        total_contrast = 0
        total_bbox = 0
        total_giou = 0
        total_position = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for images, targets in pbar:
            images = images.cuda()
            
            # 提取全局特征
            _, global_features = image_encoder(images)
            
            batch_loss = 0
            batch_contrast = 0
            batch_bbox = 0
            batch_giou = 0
            batch_position = 0
            valid_samples = 0
            
            for i, target in enumerate(targets):
                if len(target['labels']) == 0:
                    continue
                
                gt_boxes = target['boxes'].cuda()
                gt_labels = target['labels'].cuda()
                
                # 文本预测初始位置
                text_feats_for_labels = text_features[gt_labels]
                predicted_init_boxes = text_to_position(text_feats_for_labels)
                
                # 位置预测损失
                position_loss = nn.L1Loss()(predicted_init_boxes, gt_boxes)
                
                # 提取局部特征
                global_feat_i = global_features[i:i+1].expand(len(gt_labels), -1)
                local_features = query_extractor(global_feat_i, predicted_init_boxes)
                
                # 对比损失
                contrast_loss = contrast_criterion(
                    local_features,
                    text_features,
                    global_features[i:i+1].expand(len(gt_labels), -1),
                    gt_labels
                )
                
                # 框回归
                refined_boxes = box_regressor(local_features)
                bbox_l1_loss = nn.L1Loss()(refined_boxes, gt_boxes)
                
                pred_boxes_xyxy = box_cxcywh_to_xyxy(refined_boxes)
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
                giou = generalized_box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                giou_loss = (1 - giou).mean()
                
                # 总损失
                loss_i = (
                    contrast_loss + 
                    2.0 * position_loss +
                    5.0 * bbox_l1_loss +
                    2.0 * giou_loss
                )
                
                batch_loss += loss_i
                batch_contrast += contrast_loss.item()
                batch_bbox += bbox_l1_loss.item()
                batch_giou += giou_loss.item()
                batch_position += position_loss.item()
                valid_samples += 1
            
            if valid_samples == 0:
                continue
            
            loss = batch_loss / valid_samples
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(image_encoder.parameters()) + 
                list(text_to_position.parameters()) +
                list(query_extractor.parameters()) + 
                list(box_regressor.parameters()), 
                max_norm=0.1
            )
            optimizer.step()
            
            total_loss += loss.item()
            total_contrast += batch_contrast / valid_samples
            total_bbox += batch_bbox / valid_samples
            total_giou += batch_giou / valid_samples
            total_position += batch_position / valid_samples
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'pos': f"{batch_position/valid_samples:.4f}",
                'contrast': f"{batch_contrast/valid_samples:.3f}"
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_contrast = total_contrast / num_batches if num_batches > 0 else 0
        avg_bbox = total_bbox / num_batches if num_batches > 0 else 0
        avg_giou = total_giou / num_batches if num_batches > 0 else 0
        avg_position = total_position / num_batches if num_batches > 0 else 0
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (num_epochs - epoch)
        
        print(f"\nEpoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        print(f"  位置: {avg_position:.4f} | 对比: {avg_contrast:.4f} | 精修: {avg_bbox:.5f} | GIoU: {avg_giou:.4f}")
        print(f"  用时: {elapsed/60:.1f}min | 预计剩余: {eta/60:.1f}min")
        
        train_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'position_loss': avg_position,
            'contrast_loss': avg_contrast,
            'bbox_loss': avg_bbox,
            'giou_loss': avg_giou,
            'lr': current_lr
        })
        
        # 保存checkpoint
        if epoch % 10 == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                is_best = True
            else:
                is_best = False
            
            checkpoint = {
                'epoch': epoch,
                'image_encoder': image_encoder.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'text_to_position': text_to_position.state_dict(),
                'query_extractor': query_extractor.state_dict(),
                'box_regressor': box_regressor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'history': train_history
            }
            
            Path('outputs/checkpoints').mkdir(parents=True, exist_ok=True)
            
            if is_best:
                torch.save(checkpoint, 'outputs/checkpoints/DIOR_best_model.pth')
                print(f"  🌟 保存最佳模型 (loss: {avg_loss:.4f})")
            
            if epoch % 10 == 0:
                torch.save(checkpoint, f'outputs/checkpoints/DIOR_epoch_{epoch}.pth')
                print(f"  ✅ 保存checkpoint epoch_{epoch}")
        
        # 保存训练历史
        if epoch % 5 == 0:
            Path('outputs/logs').mkdir(parents=True, exist_ok=True)
            with open('outputs/logs/DIOR_train_history.json', 'w') as f:
                json.dump(train_history, f, indent=2)
    
    # 保存最终模型
    torch.save(checkpoint, 'outputs/checkpoints/DIOR_final.pth')
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"\n总用时: {total_time/60:.2f} 分钟")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"最终损失: {avg_loss:.4f}")
    print(f"\n保存的模型:")
    print(f"  - DIOR_best_model.pth (最佳)")
    print(f"  - DIOR_final.pth (最终)")
    print(f"  - DIOR_epoch_*.pth (每10个epoch)")


if __name__ == '__main__':
    train()


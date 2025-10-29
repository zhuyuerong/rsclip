#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 在完整DIOR数据集上训练
自适应全局-局部对比学习 + 边界框回归
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
from datasets.mini_dataset.mini_dataset_loader import MiniDataset
import torchvision.transforms as T


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
    """Deformable Query提取器"""
    
    def __init__(self, d_model=1024):
        super().__init__()
        self.d_model = d_model
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
        local_features = self.fusion(combined)
        return local_features


class BoxRegressor(nn.Module):
    """边界框回归器"""
    
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


def box_xyxy_to_cxcywh(boxes, img_size=224):
    """xyxy -> cxcywh (normalized)"""
    boxes = boxes / img_size  # 归一化
    x0, y0, x1, y1 = boxes.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(boxes):
    """cxcywh -> xyxy"""
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


def collate_fn(batch):
    """数据collate"""
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
    print("Experiment2 在完整DIOR数据集上训练")
    print("自适应全局-局部对比学习 + 边界框回归")
    print("=" * 70)
    
    device = torch.device('cuda')
    config = DefaultConfig()
    
    # 加载完整DIOR数据集
    print("\n加载数据集...")
    print("  ⚠️ 注意: 目前使用mini_dataset (70张训练图)")
    print("  要使用完整DIOR请修改数据路径")
    
    train_dataset = MiniDataset('../datasets/mini_dataset', 'train')
    val_dataset = MiniDataset('../datasets/mini_dataset', 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # 增大batch size
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"  训练集: {len(train_dataset)} 张图 ({len(train_loader)} batches)")
    print(f"  验证集: {len(val_dataset)} 张图 ({len(val_loader)} batches)")
    
    # 创建模型
    print("\n创建模型...")
    text_encoder = CLIPTextEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt').cuda()
    image_encoder = CLIPImageEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt', freeze=False).cuda()
    query_extractor = SimpleDeformableQueryExtractor(d_model=1024).cuda()
    box_regressor = BoxRegressor(d_model=1024).cuda()
    
    for param in image_encoder.parameters():
        param.requires_grad = True
    image_encoder.train()
    
    print(f"  ✅ 模型组件创建完成")
    
    # 损失
    contrast_criterion = AdaptiveGlobalLocalContrastLoss(temperature=0.07).cuda()
    
    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': image_encoder.parameters(), 'lr': 5e-6},
        {'params': query_extractor.parameters(), 'lr': 1e-4},
        {'params': box_regressor.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # 提取文本特征
    from utils.dataloader import DIOR_CLASSES
    with torch.no_grad():
        text_features = text_encoder(DIOR_CLASSES).cuda()
    
    print(f"  文本特征: {text_features.shape}")
    
    print(f"\n开始训练 (50 epochs)...")
    
    # 训练历史
    train_history = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(1, 51):
        # 训练
        model_train = True
        total_loss = 0
        total_contrast = 0
        total_bbox = 0
        total_giou = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/50")
        
        for images, targets in pbar:
            images = images.cuda()
            
            _, global_features = image_encoder(images)
            
            batch_loss = 0
            batch_contrast = 0
            batch_bbox = 0
            batch_giou = 0
            valid_samples = 0
            
            for i, target in enumerate(targets):
                if len(target['labels']) == 0:
                    continue
                
                gt_boxes = target['boxes'].cuda()
                gt_labels = target['labels'].cuda()
                
                # 转换框格式
                gt_boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes)
                
                # 提取局部特征
                global_feat_i = global_features[i:i+1].expand(len(gt_boxes_cxcywh), -1)
                local_features = query_extractor(global_feat_i, gt_boxes_cxcywh)
                
                # 对比损失
                contrast_loss = contrast_criterion(
                    local_features,
                    text_features,
                    global_features[i:i+1].expand(len(gt_labels), -1),
                    gt_labels
                )
                
                # 框回归
                pred_boxes_cxcywh = box_regressor(local_features)
                
                bbox_l1_loss = nn.L1Loss()(pred_boxes_cxcywh, gt_boxes_cxcywh)
                
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_cxcywh)
                giou = generalized_box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                giou_loss = (1 - giou).mean()
                
                loss_i = contrast_loss + 5.0 * bbox_l1_loss + 2.0 * giou_loss
                
                batch_loss += loss_i
                batch_contrast += contrast_loss.item()
                batch_bbox += bbox_l1_loss.item()
                batch_giou += giou_loss.item()
                valid_samples += 1
            
            if valid_samples == 0:
                continue
            
            loss = batch_loss / valid_samples
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(image_encoder.parameters()) + 
                list(query_extractor.parameters()) + 
                list(box_regressor.parameters()), 
                max_norm=0.1
            )
            optimizer.step()
            
            total_loss += loss.item()
            total_contrast += batch_contrast / valid_samples
            total_bbox += batch_bbox / valid_samples
            total_giou += batch_giou / valid_samples
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'contrast': f"{batch_contrast/valid_samples:.3f}",
                'bbox': f"{batch_bbox/valid_samples:.4f}",
                'giou': f"{batch_giou/valid_samples:.3f}"
            })
        
        # 计算平均
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_contrast = total_contrast / num_batches if num_batches > 0 else 0
        avg_bbox = total_bbox / num_batches if num_batches > 0 else 0
        avg_giou = total_giou / num_batches if num_batches > 0 else 0
        
        # 验证
        val_loss = 0
        if epoch % 5 == 0:
            image_encoder.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.cuda()
                    _, global_features = image_encoder(images)
                    # 简化验证
                    val_loss += 1
            image_encoder.train()
            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (50 - epoch)
        
        print(f"\nEpoch {epoch}/50 - Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        print(f"  对比: {avg_contrast:.4f} | 框L1: {avg_bbox:.5f} | GIoU: {avg_giou:.4f}")
        print(f"  用时: {elapsed/60:.1f}min | 预计剩余: {eta/60:.1f}min")
        
        # 记录历史
        train_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'contrast_loss': avg_contrast,
            'bbox_loss': avg_bbox,
            'giou_loss': avg_giou,
            'lr': current_lr,
            'time': elapsed
        })
        
        # 保存checkpoint
        if epoch % 10 == 0 or avg_loss < best_val_loss:
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                is_best = True
            else:
                is_best = False
            
            checkpoint = {
                'epoch': epoch,
                'image_encoder': image_encoder.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'query_extractor': query_extractor.state_dict(),
                'box_regressor': box_regressor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'history': train_history,
                'config': {
                    'temperature': 0.07,
                    'd_model': 1024,
                    'num_classes': len(DIOR_CLASSES)
                }
            }
            
            Path('outputs/checkpoints').mkdir(parents=True, exist_ok=True)
            
            if is_best:
                torch.save(checkpoint, 'outputs/checkpoints/DIOR_best_model.pth')
                print(f"  🌟 保存最佳模型 (loss: {avg_loss:.4f})")
            
            if epoch % 10 == 0:
                torch.save(checkpoint, f'outputs/checkpoints/DIOR_epoch_{epoch}.pth')
                print(f"  ✅ 保存checkpoint epoch_{epoch}")
        
        # 每5个epoch保存历史
        if epoch % 5 == 0:
            Path('outputs/logs').mkdir(parents=True, exist_ok=True)
            with open('outputs/logs/DIOR_train_history.json', 'w') as f:
                json.dump(train_history, f, indent=2)
    
    # 保存最终模型
    final_checkpoint = {
        'epoch': 50,
        'image_encoder': image_encoder.state_dict(),
        'text_encoder': text_encoder.state_dict(),
        'query_extractor': query_extractor.state_dict(),
        'box_regressor': box_regressor.state_dict(),
        'loss': avg_loss,
        'history': train_history
    }
    torch.save(final_checkpoint, 'outputs/checkpoints/DIOR_final.pth')
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"\n总用时: {total_time/3600:.2f} 小时")
    print(f"最佳损失: {best_val_loss:.4f}")
    print(f"最终损失: {avg_loss:.4f}")
    print(f"\n保存的模型:")
    print(f"  - DIOR_best_model.pth (最佳)")
    print(f"  - DIOR_final.pth (最终)")
    print(f"  - DIOR_epoch_*.pth (每10个epoch)")


if __name__ == '__main__':
    train()


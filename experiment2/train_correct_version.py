#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 正确版本
文本驱动的位置预测 + 全局-局部对比学习

正确流程：
1. 文本tc → 位置投影网络 → 初始框位置
2. 初始框位置 + 全局特征Ig → Query Extractor → 局部特征fm
3. fm vs tc vs Ig → 对比损失
4. fm → Box Regressor → 精修后的框
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


class TextToPositionProjector(nn.Module):
    """
    文本到位置的投影网络
    将文本特征tc投影为初始框位置
    """
    
    def __init__(self, text_dim=1024, hidden_dim=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # 输出 cxcywh
            nn.Sigmoid()  # 归一化到[0,1]
        )
    
    def forward(self, text_features):
        """
        Args:
            text_features: [C, D] 文本特征
        
        Returns:
            init_boxes: [C, 4] 初始框位置 (cxcywh归一化)
        """
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
        
        # 对比学习：fm接近tc，远离Ig
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
    """框回归器（精修）"""
    
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
    boxes = boxes / img_size
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
    print("Experiment2 正确版本训练")
    print("文本驱动位置预测 + 全局-局部对比学习")
    print("=" * 70)
    
    device = torch.device('cuda')
    config = DefaultConfig()
    
    # 数据
    print("\n加载数据...")
    train_dataset = MiniDataset('../datasets/mini_dataset', 'train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    print(f"  训练集: {len(train_dataset)} 张图")
    
    # 创建模型
    print("\n创建模型...")
    text_encoder = CLIPTextEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt').cuda()
    image_encoder = CLIPImageEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt', freeze=False).cuda()
    
    # 新增：文本到位置的投影网络！
    text_to_position = TextToPositionProjector(text_dim=1024).cuda()
    
    query_extractor = SimpleDeformableQueryExtractor(d_model=1024).cuda()
    box_regressor = BoxRegressor(d_model=1024).cuda()
    
    for param in image_encoder.parameters():
        param.requires_grad = True
    image_encoder.train()
    
    print(f"  ✅ Text Encoder (RemoteCLIP, 冻结)")
    print(f"  ✅ Image Encoder (RemoteCLIP, 可训练)")
    print(f"  ✅ Text-to-Position Projector (新增!) ⭐")
    print(f"  ✅ Query Extractor (Deformable)")
    print(f"  ✅ Box Regressor (精修)")
    
    # 损失
    contrast_criterion = AdaptiveGlobalLocalContrastLoss(temperature=0.07).cuda()
    
    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': image_encoder.parameters(), 'lr': 5e-6},
        {'params': text_to_position.parameters(), 'lr': 1e-4},  # 文本→位置
        {'params': query_extractor.parameters(), 'lr': 1e-4},
        {'params': box_regressor.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # 提取文本特征
    from utils.dataloader import DIOR_CLASSES
    with torch.no_grad():
        text_features = text_encoder(DIOR_CLASSES).cuda()  # [20, 1024]
    
    print(f"  文本特征: {text_features.shape}")
    
    print(f"\n开始训练 (50 epochs)...")
    print("  流程:")
    print("  1. 文本tc → 位置投影 → 初始框")
    print("  2. 初始框 + Ig → Query Extractor → fm")
    print("  3. fm vs tc vs Ig → 对比损失")
    print("  4. fm → Box Regressor → 精修框")
    
    train_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, 51):
        total_loss = 0
        total_contrast = 0
        total_bbox = 0
        total_giou = 0
        total_position = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/50")
        
        for images, targets in pbar:
            images = images.cuda()
            
            # 提取全局特征 Ig
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
                
                gt_boxes = target['boxes'].cuda()  # [N, 4] cxcywh归一化
                gt_labels = target['labels'].cuda()  # [N]
                
                # 关键：用文本特征预测初始框位置！
                text_feats_for_labels = text_features[gt_labels]  # [N, 1024]
                predicted_init_boxes = text_to_position(text_feats_for_labels)  # [N, 4]
                
                # 计算位置预测损失（让文本能预测粗略位置）
                position_loss = nn.L1Loss()(predicted_init_boxes, gt_boxes)
                
                # 使用预测的位置初始化query（不是GT！）
                global_feat_i = global_features[i:i+1].expand(len(gt_labels), -1)
                local_features = query_extractor(global_feat_i, predicted_init_boxes)  # 用预测位置！
                
                # 对比损失
                contrast_loss = contrast_criterion(
                    local_features,
                    text_features,
                    global_features[i:i+1].expand(len(gt_labels), -1),
                    gt_labels
                )
                
                # 框回归（精修）
                refined_boxes = box_regressor(local_features)  # [N, 4]
                
                bbox_l1_loss = nn.L1Loss()(refined_boxes, gt_boxes)
                
                pred_boxes_xyxy = box_cxcywh_to_xyxy(refined_boxes)
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
                giou = generalized_box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                giou_loss = (1 - giou).mean()
                
                # 总损失
                loss_i = (
                    contrast_loss + 
                    2.0 * position_loss +  # 位置预测损失
                    5.0 * bbox_l1_loss +   # 框精修损失
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
                'contrast': f"{batch_contrast/valid_samples:.3f}",
                'refine': f"{batch_bbox/valid_samples:.4f}"
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_contrast = total_contrast / num_batches if num_batches > 0 else 0
        avg_bbox = total_bbox / num_batches if num_batches > 0 else 0
        avg_giou = total_giou / num_batches if num_batches > 0 else 0
        avg_position = total_position / num_batches if num_batches > 0 else 0
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        elapsed = time.time() - start_time
        eta = elapsed / epoch * (50 - epoch)
        
        print(f"\nEpoch {epoch}/50 - Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
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
                'text_to_position': text_to_position.state_dict(),  # 保存文本→位置网络
                'query_extractor': query_extractor.state_dict(),
                'box_regressor': box_regressor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'history': train_history
            }
            
            Path('outputs/checkpoints').mkdir(parents=True, exist_ok=True)
            
            if is_best:
                torch.save(checkpoint, 'outputs/checkpoints/correct_best_model.pth')
                print(f"  🌟 保存最佳模型 (loss: {avg_loss:.4f})")
            
            if epoch % 10 == 0:
                torch.save(checkpoint, f'outputs/checkpoints/correct_epoch_{epoch}.pth')
                print(f"  ✅ 保存checkpoint epoch_{epoch}")
        
        if epoch % 5 == 0:
            Path('outputs/logs').mkdir(parents=True, exist_ok=True)
            with open('outputs/logs/correct_train_history.json', 'w') as f:
                json.dump(train_history, f, indent=2)
    
    # 保存最终模型
    torch.save(checkpoint, 'outputs/checkpoints/correct_final.pth')
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"\n总用时: {total_time/60:.2f} 分钟")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"最终损失: {avg_loss:.4f}")
    print(f"\n关键创新:")
    print(f"  ✅ 文本tc → 位置投影 → 初始框")
    print(f"  ✅ 训练和推理一致（都用文本预测位置）")
    print(f"  ✅ 全局-局部对比学习")


if __name__ == '__main__':
    train()


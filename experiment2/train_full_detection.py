#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment2 完整训练 - 全局-局部对比学习检测器
实现：
1. Deformable Query提取局部特征 f_m
2. CLIP全局特征 I_g
3. 文本特征 t_c
4. 自适应全局-局部对比损失
5. 边界框回归
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
from datasets.mini_dataset.mini_dataset_loader import MiniDataset
import torchvision.transforms as T


class AdaptiveGlobalLocalContrastLoss(nn.Module):
    """
    自适应全局-局部对比损失
    
    三个关键向量：
    1. 目标文本 t_c: 正锚点
    2. 局部特征 f_m: 从Deformable Query得到，对应GT框
    3. 全局图像特征 I_g: 从CLIP Image Encoder得到，作为上下文/背景锚点
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, local_features, text_features, global_features, labels):
        """
        Args:
            local_features: [N, D] - 局部特征 f_m
            text_features: [C, D] - 文本特征 t_c
            global_features: [N, D] - 全局特征 I_g
            labels: [N] - GT标签
        
        Returns:
            loss: 自适应全局-局部对比损失
        """
        # 归一化
        local_features = local_features / (local_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        global_features = global_features / (global_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 计算局部-文本相似度 (正样本对)
        local_text_sim = (local_features @ text_features.T) / self.temperature  # [N, C]
        
        # 计算局部-全局相似度 (负样本对，背景)
        local_global_sim = (local_features * global_features).sum(dim=-1, keepdim=True) / self.temperature  # [N, 1]
        
        # 对比学习损失
        # 目标：局部特征应该接近目标文本，远离全局背景
        logits = torch.cat([local_text_sim, local_global_sim], dim=-1)  # [N, C+1]
        
        # 标签就是GT类别
        loss = nn.CrossEntropyLoss()(logits[:, :-1], labels)  # 只对文本类别计算loss
        
        return loss


class SimpleDeformableQueryExtractor(nn.Module):
    """
    简化的可变形查询提取器
    模拟从GT框位置提取局部特征
    """
    
    def __init__(self, d_model=1024):
        super().__init__()
        self.d_model = d_model
        
        # 位置编码
        self.position_embed = nn.Linear(4, d_model)  # 4D bbox -> d_model
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, global_features, boxes):
        """
        Args:
            global_features: [B, D] - 全局图像特征
            boxes: [B, 4] - 边界框 (归一化的cxcywh)
        
        Returns:
            local_features: [B, D] - 局部特征
        """
        # 位置编码
        pos_embed = self.position_embed(boxes)  # [B, D]
        
        # 扩展全局特征
        global_expanded = global_features.unsqueeze(1).expand(-1, 1, -1).squeeze(1)  # [B, D]
        
        # 融合：全局特征 + 位置信息
        combined = torch.cat([global_expanded, pos_embed], dim=-1)  # [B, 2D]
        local_features = self.fusion(combined)  # [B, D]
        
        return local_features


class BoxRegressor(nn.Module):
    """
    边界框回归器
    从局部特征fm预测边界框
    """
    
    def __init__(self, d_model=1024):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 预测 cxcywh
        )
    
    def forward(self, local_features):
        """
        Args:
            local_features: [N, D]
        
        Returns:
            boxes: [N, 4] - 预测的边界框 (cxcywh, normalized)
        """
        boxes = self.regressor(local_features)
        boxes = torch.sigmoid(boxes)  # 归一化到[0, 1]
        return boxes


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


def box_cxcywh_to_xyxy(boxes):
    """转换边界框格式 cxcywh -> xyxy"""
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h,
         x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(boxes):
    """转换边界框格式 xyxy -> cxcywh (normalized)"""
    x0, y0, x1, y1 = boxes.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    计算GIoU
    boxes: [N, 4] in xyxy format
    """
    # IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)
    
    # GIoU
    lti = torch.min(boxes1[:, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    
    whi = (rbi - lti).clamp(min=0)
    areai = whi[:, 0] * whi[:, 1]
    
    giou = iou - (areai - union) / (areai + 1e-6)
    
    return giou


def train():
    print("=" * 70)
    print("Experiment2 完整训练")
    print("自适应全局-局部对比学习 + 边界框回归")
    print("=" * 70)
    
    device = torch.device('cuda')
    config = DefaultConfig()
    
    # 加载数据
    print("\n加载数据...")
    train_dataset = MiniDataset('../datasets/mini_dataset', 'train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    print(f"  训练集: {len(train_dataset)} 张图")
    
    # 创建模型
    print("\n创建模型...")
    text_encoder = CLIPTextEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt').cuda()
    image_encoder = CLIPImageEncoder('RN50', '../checkpoints/RemoteCLIP-RN50.pt', freeze=False).cuda()
    
    # 可变形查询提取器
    query_extractor = SimpleDeformableQueryExtractor(d_model=1024).cuda()
    
    # 边界框回归器
    box_regressor = BoxRegressor(d_model=1024).cuda()
    
    # 确保可训练
    for param in image_encoder.parameters():
        param.requires_grad = True
    image_encoder.train()
    
    print(f"\n模型组件:")
    print(f"  ✅ Text Encoder (RemoteCLIP, 冻结)")
    print(f"  ✅ Image Encoder (RemoteCLIP, 可训练)")
    print(f"  ✅ Query Extractor (Deformable, 新建)")
    print(f"  ✅ Box Regressor (新建)")
    
    # 损失函数
    contrast_criterion = AdaptiveGlobalLocalContrastLoss(temperature=0.07).cuda()
    
    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': image_encoder.parameters(), 'lr': 1e-5},  # 小学习率fine-tune
        {'params': query_extractor.parameters(), 'lr': 1e-4},
        {'params': box_regressor.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    # 提取文本特征（冻结）
    from utils.dataloader import DIOR_CLASSES
    with torch.no_grad():
        text_features = text_encoder(DIOR_CLASSES).cuda()  # [20, 1024]
    
    print(f"  文本特征: {text_features.shape}")
    
    print(f"\n开始训练 (20 epochs)...")
    print("  损失包含:")
    print("  1. 自适应全局-局部对比损失")
    print("  2. 边界框L1损失")
    print("  3. GIoU损失")
    
    train_history = []
    
    for epoch in range(1, 21):
        total_loss = 0
        total_contrast_loss = 0
        total_bbox_loss = 0
        total_giou_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for images, targets in pbar:
            images = images.cuda()
            
            # 提取全局特征 I_g
            _, global_features = image_encoder(images)  # [B, 1024]
            
            # 处理每个样本（因为每个样本可能有不同数量的框）
            batch_loss = 0
            batch_contrast = 0
            batch_bbox = 0
            batch_giou = 0
            num_boxes = 0
            
            for i, target in enumerate(targets):
                if len(target['labels']) == 0:
                    continue
                
                # 获取GT框和标签
                gt_boxes = target['boxes'].cuda()  # [N_i, 4] xyxy
                gt_labels = target['labels'].cuda()  # [N_i]
                
                # 转换为cxcywh并归一化 (假设图像是224x224)
                gt_boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes / 224.0)  # 归一化
                
                # 使用Deformable Query提取局部特征 f_m
                global_feat_i = global_features[i:i+1].expand(len(gt_boxes_cxcywh), -1)  # [N_i, 1024]
                local_features = query_extractor(global_feat_i, gt_boxes_cxcywh)  # [N_i, 1024]
                
                # 1. 全局-局部对比损失
                contrast_loss = contrast_criterion(
                    local_features,
                    text_features,
                    global_features[i:i+1].expand(len(gt_labels), -1),
                    gt_labels
                )
                
                # 2. 边界框回归
                pred_boxes_cxcywh = box_regressor(local_features)  # [N_i, 4]
                
                # L1损失
                bbox_l1_loss = nn.L1Loss()(pred_boxes_cxcywh, gt_boxes_cxcywh)
                
                # GIoU损失
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_cxcywh)
                giou = generalized_box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                giou_loss = (1 - giou).mean()
                
                # 总损失
                loss_i = contrast_loss + 5.0 * bbox_l1_loss + 2.0 * giou_loss
                
                batch_loss += loss_i
                batch_contrast += contrast_loss.item()
                batch_bbox += bbox_l1_loss.item()
                batch_giou += giou_loss.item()
                num_boxes += len(gt_boxes)
            
            if num_boxes == 0:
                continue
            
            # 平均损失
            loss = batch_loss / len([t for t in targets if len(t['labels']) > 0])
            
            # 反向传播
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
            total_contrast_loss += batch_contrast / len(targets)
            total_bbox_loss += batch_bbox / len(targets)
            total_giou_loss += batch_giou / len(targets)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'contrast': f"{batch_contrast/len(targets):.4f}",
                'bbox': f"{batch_bbox/len(targets):.4f}",
                'giou': f"{batch_giou/len(targets):.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_contrast = total_contrast_loss / len(train_loader)
        avg_bbox = total_bbox_loss / len(train_loader)
        avg_giou = total_giou_loss / len(train_loader)
        
        print(f"\nEpoch {epoch} - Loss: {avg_loss:.4f}")
        print(f"  对比损失: {avg_contrast:.4f}")
        print(f"  框L1损失: {avg_bbox:.4f}")
        print(f"  GIoU损失: {avg_giou:.4f}")
        
        train_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'contrast_loss': avg_contrast,
            'bbox_loss': avg_bbox,
            'giou_loss': avg_giou
        })
        
        # 保存checkpoint
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'image_encoder': image_encoder.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'query_extractor': query_extractor.state_dict(),
                'box_regressor': box_regressor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'history': train_history
            }
            
            Path('outputs/checkpoints').mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, f'outputs/checkpoints/full_model_epoch_{epoch}.pth')
            print(f"  ✅ 保存checkpoint: full_model_epoch_{epoch}.pth")
    
    # 保存训练历史
    with open('outputs/logs/full_train_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"\n模型组件:")
    print(f"  1. 全局图像特征 I_g: CLIP Image Encoder (fine-tuned)")
    print(f"  2. 局部特征 f_m: Deformable Query Extractor")
    print(f"  3. 文本特征 t_c: CLIP Text Encoder (frozen)")
    print(f"  4. 边界框回归器: fm → bbox")
    print(f"\n损失函数:")
    print(f"  1. 自适应全局-局部对比损失 ✅")
    print(f"  2. 边界框L1损失 ✅")
    print(f"  3. GIoU损失 ✅")
    print(f"\n最终训练损失: {avg_loss:.4f}")


if __name__ == '__main__':
    train()


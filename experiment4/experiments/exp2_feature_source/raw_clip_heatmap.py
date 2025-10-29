# -*- coding: utf-8 -*-
"""
实验2.1：使用原始CLIP特征生成热图（不经过Surgery去冗余）
这是最关键的修复实验
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import json
import clip
import numpy as np

# 添加路径
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

from experiment4.config import Config
from experiment4.data.dataset import get_dataloaders
from experiment4.utils.heatmap_generator import generate_bboxes_from_heatmap, compute_bbox_score
from experiment4.utils.map_calculator import calculate_map
from experiment4.utils.visualization import visualize_heatmap_and_boxes, save_visualization


def generate_heatmap_from_raw_clip(clip_model, images, text_features, device):
    """
    使用原始CLIP特征生成热图（无Surgery去冗余）
    """
    with torch.no_grad():
        # 确保输入类型匹配
        if images.dtype != clip_model.visual.conv1.weight.dtype:
            images = images.to(clip_model.visual.conv1.weight.dtype)
        
        # 提取完整特征（包含CLS）
        x = clip_model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = torch.cat([
            clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)
        
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
        x = clip_model.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)
        x = clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        x = clip_model.visual.ln_post(x)
        
        # 投影到512维
        if hasattr(clip_model.visual, 'proj') and clip_model.visual.proj is not None:
            B, N, D = x.shape
            x_reshaped = x.reshape(B * N, D)
            x_proj = x_reshaped @ clip_model.visual.proj
            features = x_proj.reshape(B, N, -1)
        else:
            features = x
        
        # 提取patch特征（去掉CLS）
        patch_features = features[:, 1:, :]  # [B, N, 512]
        
        # L2归一化
        patch_norm = F.normalize(patch_features, dim=-1, p=2)
        text_norm = F.normalize(text_features, dim=-1, p=2)
        
        # 计算相似度
        similarity = torch.einsum('bnd,kd->bnk', patch_norm, text_norm)
        
        # Reshape到空间维度
        B, N, K = similarity.shape
        H = W = int(N ** 0.5)
        similarity_map = similarity.reshape(B, H, W, K)
    
    return similarity_map


def evaluate_raw_clip_heatmap(clip_model, val_loader, device, max_visualizations=10):
    """
    使用原始CLIP特征评估
    """
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    visualizations = []
    
    print(f"\n使用原始CLIP特征生成热图（无Surgery去冗余）...")
    
    for batch in tqdm(val_loader, desc="评估"):
        images = batch['image'].to(device)
        class_names = batch['class_name']
        gt_bboxes = batch['bbox']
        has_bbox = batch['has_bbox']
        
        # 获取文本特征
        unique_classes = list(set(class_names))
        text_tokens = clip.tokenize(unique_classes).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 生成热图
        similarity_map = generate_heatmap_from_raw_clip(clip_model, images, text_features, device)
        
        # 对每个样本生成检测框
        for i in range(len(images)):
            if not has_bbox[i]:
                continue
            
            class_name = class_names[i]
            class_idx = unique_classes.index(class_name)
            
            # 热图
            heatmap = similarity_map[i, :, :, class_idx].cpu().numpy()
            
            # 生成检测框（使用75%阈值）
            pred_bboxes = generate_bboxes_from_heatmap(heatmap, threshold_percentile=75)
            
            # 计算分数
            for bbox in pred_bboxes:
                score = compute_bbox_score(heatmap, bbox)
                all_predictions[class_name].append({
                    'bbox': bbox,
                    'score': score
                })
            
            # 保存GT
            gt_bbox_norm = gt_bboxes[i].cpu().numpy()
            gt_bbox_pixel = [
                int(gt_bbox_norm[0] * 224),
                int(gt_bbox_norm[1] * 224),
                int(gt_bbox_norm[2] * 224),
                int(gt_bbox_norm[3] * 224)
            ]
            all_ground_truths[class_name].append(gt_bbox_pixel)
            
            # 可视化
            if len(visualizations) < max_visualizations:
                vis_fig = visualize_heatmap_and_boxes(
                    images[i], heatmap, pred_bboxes, gt_bbox_pixel, class_name
                )
                visualizations.append(vis_fig)
    
    # 计算mAP
    print(f"\n计算mAP...")
    mAP_005, per_class_ap_005 = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.05)
    mAP_050, per_class_ap_050 = calculate_map(all_predictions, all_ground_truths, iou_threshold=0.50)
    
    print(f"\n评估完成:")
    print(f"  样本数: {sum(len(bboxes) for bboxes in all_ground_truths.values())}")
    print(f"  类别数: {len(all_ground_truths)}")
    print(f"  mAP@0.05: {mAP_005:.4f}")
    print(f"  mAP@0.50: {mAP_050:.4f}")
    
    return {
        'mAP_005': mAP_005,
        'mAP_050': mAP_050,
        'per_class_ap_005': per_class_ap_005,
        'per_class_ap_050': per_class_ap_050,
        'visualizations': visualizations
    }


def main():
    """主函数"""
    print("="*70)
    print("实验2.1：原始CLIP特征热图（无Surgery去冗余）")
    print("="*70)
    
    config = Config()
    device = config.device
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    
    # 加载原始CLIP模型
    print("\n加载原始CLIP模型...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # 加载RemoteCLIP权重
    remoteclip_path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
    if remoteclip_path.exists():
        print(f"加载RemoteCLIP权重: {remoteclip_path}")
        checkpoint = torch.load(remoteclip_path, map_location=device)
        if 'state_dict' in checkpoint:
            clip_model.load_state_dict(checkpoint['state_dict'])
        else:
            clip_model.load_state_dict(checkpoint)
    
    clip_model.eval()
    
    # 评估
    results = evaluate_raw_clip_heatmap(clip_model, val_loader, device)
    
    # 保存结果
    output_dir = Path("experiment4/experiments/exp2_feature_source/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数值
    results_to_save = {
        'mAP_005': results['mAP_005'],
        'mAP_050': results['mAP_050'],
        'per_class_ap_005': results['per_class_ap_005'],
        'per_class_ap_050': results['per_class_ap_050']
    }
    
    with open(output_dir / "raw_clip_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    # 保存可视化
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    for idx, fig in enumerate(results['visualizations']):
        save_visualization(fig, vis_dir / f"sample_{idx:03d}.png")
    
    print(f"\n✅ 结果已保存: {output_dir}")
    print(f"   数值: raw_clip_results.json")
    print(f"   可视化: visualizations/ ({len(results['visualizations'])}张)")
    
    # 对比分析
    print(f"\n" + "="*70)
    print(f"与Surgery版本对比:")
    print(f"="*70)
    print(f"\n原始CLIP (无Surgery):")
    print(f"  mAP@0.05: {results['mAP_005']:.4f}")
    print(f"  mAP@0.50: {results['mAP_050']:.4f}")
    
    print(f"\nSurgery版本 (参考):")
    print(f"  mAP@0.05: 0.2780")
    print(f"  mAP@0.50: 0.0000")
    
    improvement_005 = results['mAP_005'] - 0.2780
    improvement_050 = results['mAP_050'] - 0.0000
    
    print(f"\n提升:")
    print(f"  mAP@0.05: {improvement_005:+.4f} ({improvement_005/0.2780*100:+.1f}%)")
    print(f"  mAP@0.50: {improvement_050:+.4f}")
    
    if results['mAP_005'] > 0.28:
        print(f"\n✅ 原始CLIP特征显著优于Surgery！")
        print(f"   → 确认Surgery去冗余是问题根源")
    else:
        print(f"\n⚠️ 原始CLIP未显著改善")
        print(f"   → 问题可能不只是Surgery")


if __name__ == "__main__":
    main()


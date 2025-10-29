# -*- coding: utf-8 -*-
"""
实验1.3：测试反转热图
如果反转后目标变红色，说明Surgery确实抑制了目标特征
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加路径
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.core.data.dataset import get_dataloaders
from experiment4.core.utils.map_calculator import calculate_map
from collections import defaultdict


def generate_similarity_heatmap_inverted(image_features, text_features):
    """
    生成反转的相似度热图
    
    使用 -similarity 或 1 - similarity
    """
    # 提取patch特征
    patch_features = image_features[:, 1:, :]  # [B, N, 512]
    
    # L2归一化
    patch_norm = F.normalize(patch_features, dim=-1, p=2)
    text_norm = F.normalize(text_features, dim=-1, p=2)
    
    # 计算相似度
    similarity = torch.einsum('bnd,kd->bnk', patch_norm, text_norm)
    
    # 反转！
    similarity_inverted = -similarity  # 使用负值反转
    
    # Reshape
    B, N, K = similarity_inverted.shape
    H = W = int(N ** 0.5)
    similarity_map = similarity_inverted.reshape(B, H, W, K)
    
    return similarity_map


def visualize_comparison(image, heatmap_normal, heatmap_inverted, gt_bbox, class_name):
    """
    对比可视化：正常热图 vs 反转热图
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 处理图像
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().permute(1, 2, 0).numpy()
    else:
        img_np = image
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    # GT框（像素坐标）
    gt_bbox_pixel = [
        int(gt_bbox[0] * 224),
        int(gt_bbox[1] * 224),
        int(gt_bbox[2] * 224),
        int(gt_bbox[3] * 224)
    ]
    
    # 1. 原图 + GT
    axes[0].imshow(img_np)
    rect = patches.Rectangle(
        (gt_bbox_pixel[0], gt_bbox_pixel[1]),
        gt_bbox_pixel[2] - gt_bbox_pixel[0],
        gt_bbox_pixel[3] - gt_bbox_pixel[1],
        linewidth=2, edgecolor='green', facecolor='none'
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f'Original + GT\n{class_name}')
    axes[0].axis('off')
    
    # 2. 正常热图
    axes[1].imshow(img_np)
    heat_normal = cv2.resize(heatmap_normal.astype(np.float32), (224, 224))
    heat_normal_norm = (heat_normal - heat_normal.min()) / (heat_normal.max() - heat_normal.min() + 1e-8)
    axes[1].imshow(heat_normal_norm, alpha=0.5, cmap='jet')
    axes[1].set_title('正常热图 (Surgery)\n目标=蓝色?')
    axes[1].axis('off')
    
    # 3. 反转热图
    axes[2].imshow(img_np)
    heat_inv = cv2.resize(heatmap_inverted.astype(np.float32), (224, 224))
    heat_inv_norm = (heat_inv - heat_inv.min()) / (heat_inv.max() - heat_inv.min() + 1e-8)
    axes[2].imshow(heat_inv_norm, alpha=0.5, cmap='jet')
    axes[2].set_title('反转热图 (-similarity)\n目标=红色?')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """主函数"""
    print("="*70)
    print("实验1.3：反转热图测试")
    print("="*70)
    
    config = Config()
    device = config.device
    
    # 加载数据和模型
    print("\n加载数据...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    
    print("\n加载CLIP Surgery模型...")
    model = CLIPSurgeryWrapper(config)
    
    # 测试前3个样本
    print("\n生成对比可视化...")
    
    output_dir = Path("experiment4/experiments/exp1_threshold_fix/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_count = 0
    max_samples = 3
    
    for batch in val_loader:
        images = batch['image']
        class_names = batch['class_name']
        bboxes = batch['bbox']
        has_bbox = batch['has_bbox']
        
        for i in range(len(images)):
            if not has_bbox[i]:
                continue
            
            # 提取特征
            image_tensor = images[i].unsqueeze(0).to(device)
            image_features = model.get_all_features(image_tensor)
            text_features = model.encode_text([class_names[i]])
            
            # 生成两种热图
            from experiment4.core.utils.heatmap_generator import generate_similarity_heatmap
            heatmap_normal = generate_similarity_heatmap(image_features, text_features)[0, :, :, 0].cpu().numpy()
            heatmap_inverted = generate_similarity_heatmap_inverted(image_features, text_features)[0, :, :, 0].cpu().numpy()
            
            # 可视化
            fig = visualize_comparison(
                images[i],
                heatmap_normal,
                heatmap_inverted,
                bboxes[i],
                class_names[i]
            )
            
            # 保存
            save_path = output_dir / f"comparison_{sample_count:03d}_{class_names[i]}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ 已保存: {save_path.name}")
            
            sample_count += 1
            if sample_count >= max_samples:
                break
        
        if sample_count >= max_samples:
            break
    
    print(f"\n" + "="*70)
    print(f"结论:")
    print(f"="*70)
    print(f"请查看生成的对比图：{output_dir}")
    print(f"\n如果反转热图显示目标=红色、背景=蓝色，则证明：")
    print(f"  ❌ Surgery去冗余确实抑制了目标特征")
    print(f"  ✅ 应该使用反转的热图（-similarity）")
    print(f"  或者不使用Surgery去冗余")


if __name__ == "__main__":
    main()


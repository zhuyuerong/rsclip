#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断CAM类别映射问题
检查text_queries顺序、text_features顺序、CAM顺序是否一致
"""

import torch
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_surgery_cam_detector import create_simple_surgery_cam_detector
from datasets.dior_detection import get_detection_dataloader
import yaml

# DIOR类别列表
DIOR_CLASSES = [
    "airplane", "airport", "baseball field", "basketball court",
    "bridge", "chimney", "dam", "expressway service area",
    "expressway toll station", "golf course", "ground track field",
    "harbor", "overpass", "ship", "stadium", "storage tank",
    "tennis court", "train station", "vehicle", "wind mill"
]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载配置
    with open('configs/surgery_cam_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print("加载模型...")
    model = create_simple_surgery_cam_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=20,
        cam_resolution=7,
        device=device,
        unfreeze_cam_last_layer=True
    )
    
    # 加载checkpoint
    checkpoint_path = 'checkpoints/best_simple_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    trainable_state = {}
    for key, value in model_state.items():
        if key in model_dict and ('box_head' in key or 'cam_generator.learnable_proj' in key):
            trainable_state[key] = value
    model_dict.update(trainable_state)
    model.load_state_dict(model_dict, strict=False)
    model.eval()
    
    # 加载一张测试图像
    loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=1,
        num_workers=0,
        image_size=224,
        shuffle=False
    )
    
    batch = next(iter(loader))
    images = batch['images'].to(device)
    gt_boxes = batch['boxes'][0]
    gt_labels = batch['labels'][0]
    
    print(f"\n图像GT标签: {gt_labels.tolist()}")
    print(f"GT类别名称: {[DIOR_CLASSES[l.item()] for l in gt_labels]}")
    
    # 使用所有20个类别
    text_queries = DIOR_CLASSES.copy()
    print(f"\n使用text_queries顺序:")
    for i, q in enumerate(text_queries):
        print(f"  [{i}] {q}")
    
    # 生成CAM
    print("\n生成CAM...")
    with torch.no_grad():
        outputs = model(images, text_queries)
        cam = outputs['cam'][0]  # [C, H, W]
    
    print(f"CAM shape: {cam.shape}")
    
    # 检查每个GT类别对应的CAM响应
    print("\n" + "=" * 80)
    print("检查GT类别对应的CAM响应")
    print("=" * 80)
    
    for i, label in enumerate(gt_labels):
        label_idx = label.item()
        class_name = DIOR_CLASSES[label_idx]
        box = gt_boxes[i]
        x1, y1, x2, y2 = box
        
        # 将归一化坐标转换为CAM空间坐标
        H_cam, W_cam = cam.shape[1], cam.shape[2]
        x1_cam = int(x1 * W_cam)
        y1_cam = int(y1 * H_cam)
        x2_cam = int(x2 * W_cam)
        y2_cam = int(y2 * H_cam)
        
        x1_cam = max(0, min(x1_cam, W_cam-1))
        y1_cam = max(0, min(y1_cam, H_cam-1))
        x2_cam = max(0, min(x2_cam, W_cam-1))
        y2_cam = max(0, min(y2_cam, H_cam-1))
        
        # 计算正确类别的CAM响应
        cam_correct = cam[label_idx]
        cam_inside = cam_correct[y1_cam:y2_cam+1, x1_cam:x2_cam+1]
        inside_mean = cam_inside.mean().item() if cam_inside.numel() > 0 else 0
        
        # 计算所有类别的CAM响应（在GT框内）
        all_class_responses = []
        for c in range(cam.shape[0]):
            cam_class = cam[c]
            cam_inside_class = cam_class[y1_cam:y2_cam+1, x1_cam:x2_cam+1]
            inside_mean_class = cam_inside_class.mean().item() if cam_inside_class.numel() > 0 else 0
            all_class_responses.append((c, DIOR_CLASSES[c], inside_mean_class))
        
        # 按响应排序
        all_class_responses.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nGT类别 [{label_idx}] {class_name}:")
        print(f"  正确CAM (cam[{label_idx}]): {inside_mean:.4f}")
        print(f"  所有类别CAM响应（在GT框内）:")
        for rank, (c, name, resp) in enumerate(all_class_responses[:5]):
            marker = "✅" if c == label_idx else "  "
            print(f"    {marker} Rank {rank+1}: cam[{c}] '{name}' = {resp:.4f}")
        
        if all_class_responses[0][0] != label_idx:
            print(f"  ⚠️  警告: 最高响应类别是 '{all_class_responses[0][1]}' (cam[{all_class_responses[0][0]}])，不是正确的 '{class_name}' (cam[{label_idx}])")
            print(f"     这可能表示:")
            print(f"     1. CAM质量差（需要更好的训练）")
            print(f"     2. 类别映射错误（需要检查text_queries顺序）")
            print(f"     3. 类别相似度高（如airplane和airport）")
    
    # 可视化CAM热图对比
    print("\n" + "=" * 80)
    print("生成CAM可视化对比")
    print("=" * 80)
    
    output_dir = Path('outputs/cam_mapping_diagnosis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个GT类别生成可视化
    for i, label in enumerate(gt_labels[:3]):  # 只可视化前3个
        label_idx = label.item()
        class_name = DIOR_CLASSES[label_idx]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'GT类别: [{label_idx}] {class_name}', fontsize=16, fontweight='bold')
        
        # 显示正确类别的CAM
        cam_correct = cam[label_idx].cpu().numpy()
        axes[0, 0].imshow(cam_correct, cmap='jet')
        axes[0, 0].set_title(f'正确CAM: cam[{label_idx}] "{class_name}"')
        axes[0, 0].axis('off')
        
        # 显示响应最高的3个其他类别
        all_responses = [(c, DIOR_CLASSES[c], cam[c].cpu().numpy().mean()) 
                        for c in range(cam.shape[0]) if c != label_idx]
        all_responses.sort(key=lambda x: x[2], reverse=True)
        
        for j, (c, name, _) in enumerate(all_responses[:3]):
            cam_other = cam[c].cpu().numpy()
            row = 0 if j < 2 else 1
            col = j + 1 if j < 2 else j - 1
            axes[row, col].imshow(cam_other, cmap='jet')
            axes[row, col].set_title(f'其他CAM: cam[{c}] "{name}"')
            axes[row, col].axis('off')
        
        # 第二行：显示所有类别的平均响应
        all_means = [cam[c].cpu().numpy().mean() for c in range(cam.shape[0])]
        axes[1, 2].bar(range(len(DIOR_CLASSES)), all_means)
        axes[1, 2].axvline(label_idx, color='r', linestyle='--', label='GT类别')
        axes[1, 2].set_title('所有类别CAM平均响应')
        axes[1, 2].set_xlabel('类别索引')
        axes[1, 2].set_ylabel('平均CAM值')
        axes[1, 2].legend()
        axes[1, 2].set_xticks(range(len(DIOR_CLASSES)))
        axes[1, 2].set_xticklabels([DIOR_CLASSES[i][:5] for i in range(len(DIOR_CLASSES))], 
                                    rotation=45, ha='right')
        
        plt.tight_layout()
        output_path = output_dir / f'gt_class_{label_idx}_{class_name}_cam_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 已保存: {output_path}")

if __name__ == '__main__':
    main()



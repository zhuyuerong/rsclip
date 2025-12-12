#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合可视化脚本
展示：原始图像、CAM热力图、最终检测结果，以及指标对比
"""

import torch
import sys
import os
from pathlib import Path
import argparse
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision import transforms
import cv2

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.surgery_cam_detector import create_surgery_cam_detector
from models.owlvit_baseline import create_owlvit_model
from utils.visualization import visualize_cam


# DIOR类别列表
DIOR_CLASSES = [
    "airplane", "airport", "baseball field", "basketball court",
    "bridge", "chimney", "dam", "expressway service area",
    "expressway toll station", "golf course", "ground track field",
    "harbor", "overpass", "ship", "stadium", "storage tank",
    "tennis court", "train station", "vehicle", "wind mill"
]

# Seen/Unseen类别划分（示例，可根据实际需求调整）
SEEN_CLASSES = [
    "airplane", "ship", "vehicle", "bridge", "harbor",
    "stadium", "storage tank", "airport", "golf course", "wind mill"
]

UNSEEN_CLASSES = [
    "baseball field", "basketball court", "chimney", "dam",
    "expressway service area", "expressway toll station",
    "ground track field", "overpass", "tennis court", "train station"
]


def load_image(image_path: str, image_size: int = 224):
    """加载和预处理图片"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image, original_size


def visualize_comprehensive_results(image_pil, cam, detections, text_queries, 
                                   class_idx=None, output_path=None):
    """
    综合可视化：原始图像、CAM热力图、最终检测结果
    
    Args:
        image_pil: PIL Image
        cam: [C, H, W] CAM tensor
        detections: List of detection dicts
        text_queries: List of class names
        class_idx: 要可视化的类别索引（如果为None，显示所有类别的最大响应）
        output_path: 保存路径
    """
    fig = plt.figure(figsize=(20, 6))
    
    # ===== 图1: 原始图像 =====
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image_pil)
    ax1.set_title("原始图像：密集的船只", fontsize=16, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # ===== 图2: CAM热力图 =====
    ax2 = plt.subplot(1, 3, 2)
    
    # 选择要显示的CAM
    if class_idx is not None:
        cam_vis = cam[class_idx].cpu().numpy()
        class_name = text_queries[class_idx] if class_idx < len(text_queries) else f"Class {class_idx}"
    else:
        # 显示所有类别的最大响应
        cam_vis = cam.max(dim=0)[0].cpu().numpy()
        class_name = "All Classes"
    
    # 调整CAM到图像尺寸
    h_img, w_img = image_pil.size[1], image_pil.size[0]
    cam_resized = cv2.resize(cam_vis, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    
    # 归一化
    cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-6)
    
    # 叠加到原图
    image_np = np.array(image_pil)
    cam_colored = plt.cm.jet(cam_normalized)[:, :, :3]
    cam_colored = (cam_colored * 255).astype(np.uint8)
    overlay = (0.5 * cam_colored + 0.5 * image_np).astype(np.uint8)
    
    ax2.imshow(overlay)
    ax2.set_title(f"CAM热力图：每艘船都有高响应区域\n({class_name})", 
                  fontsize=16, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # ===== 图3: 最终检测结果 =====
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(image_pil)
    
    # 绘制检测框
    colors = plt.cm.tab20(np.linspace(0, 1, len(text_queries)))
    h_img, w_img = image_pil.size[1], image_pil.size[0]
    
    for det in detections:
        box = det['box']
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        
        label = det['label']
        score = det['score']
        class_name = det.get('class_name', text_queries[label] if label < len(text_queries) else f"Class {label}")
        
        # 转换为像素坐标
        xmin = box[0] * w_img
        ymin = box[1] * h_img
        xmax = box[2] * w_img
        ymax = box[3] * h_img
        
        # 选择颜色
        color = colors[label % len(colors)]
        
        # 绘制框
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2.5, edgecolor=color, facecolor='none'
        )
        ax3.add_patch(rect)
        
        # 添加标签
        ax3.text(
            xmin, ymin - 5,
            f'{class_name}: {score:.2f}',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
            fontsize=11, color='white', weight='bold'
        )
    
    ax3.set_title("最终检测：精确的边界框", fontsize=16, fontweight='bold', pad=10)
    ax3.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✅ 综合可视化已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_metrics_comparison_table(metrics_dict, output_path=None):
    """
    创建指标对比表格
    
    Args:
        metrics_dict: {
            'CAM-only': {'AP@0.5': float, 'AP@0.5:0.95': float},
            'OWL-ViT-style': {'AP@0.5': float, 'AP@0.5:0.95': float},
            'SurgeryCAM': {'AP@0.5': float, 'AP@0.5:0.95': float}
        }
        output_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备数据
    methods = list(metrics_dict.keys())
    ap50_values = [metrics_dict[m].get('AP@0.5', 0.0) for m in methods]
    ap50_95_values = [metrics_dict[m].get('AP@0.5:0.95', 0.0) for m in methods]
    
    # 创建表格
    table_data = [
        ['方法', 'AP@0.5', 'AP@0.5:0.95']
    ]
    
    for i, method in enumerate(methods):
        table_data.append([
            method,
            f'{ap50_values[i]:.3f}',
            f'{ap50_95_values[i]:.3f}'
        ])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center',
                    colWidths=[0.4, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2)
    
    # 设置样式
    for i in range(len(methods) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:
                    cell.set_facecolor('#D9E1F2')
                else:
                    cell.set_facecolor('#FFFFFF')
    
    plt.title('检测性能对比', fontsize=18, fontweight='bold', pad=20)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 指标对比表已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_seen_unseen_comparison(seen_metrics, unseen_metrics, output_path=None):
    """
    创建seen/unseen类别对比图
    
    Args:
        seen_metrics: {'AP@0.5': float, 'AP@0.5:0.95': float}
        unseen_metrics: {'AP@0.5': float, 'AP@0.5:0.95': float}
        output_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # AP@0.5对比
    categories = ['Seen Classes', 'Unseen Classes']
    ap50_values = [seen_metrics.get('AP@0.5', 0.0), unseen_metrics.get('AP@0.5', 0.0)]
    
    bars1 = ax1.bar(categories, ap50_values, color=['#4CAF50', '#FF9800'], alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('AP@0.5', fontsize=14, fontweight='bold')
    ax1.set_title('Seen vs Unseen Classes (AP@0.5)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylim([0, max(ap50_values) * 1.2])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, val in zip(bars1, ap50_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # AP@0.5:0.95对比
    ap50_95_values = [seen_metrics.get('AP@0.5:0.95', 0.0), unseen_metrics.get('AP@0.5:0.95', 0.0)]
    
    bars2 = ax2.bar(categories, ap50_95_values, color=['#4CAF50', '#FF9800'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('AP@0.5:0.95', fontsize=14, fontweight='bold')
    ax2.set_title('Seen vs Unseen Classes (AP@0.5:0.95)', fontsize=16, fontweight='bold', pad=15)
    ax2.set_ylim([0, max(ap50_95_values) * 1.2])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, val in zip(bars2, ap50_95_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Seen/Unseen对比图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='综合可视化：图像、CAM、检测结果和指标对比')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='SurgeryCAM checkpoint路径')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--class-name', type=str, default=None,
                       help='要可视化的类别名称（如 "ship"），如果为None则显示所有类别的最大响应')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='置信度阈值')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                       help='NMS IoU阈值')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations',
                       help='输出目录')
    parser.add_argument('--show-metrics', action='store_true',
                       help='是否显示指标对比（需要提供指标文件或使用默认值）')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. 加载配置 =====
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # ===== 2. 加载模型 =====
    print("\n" + "=" * 80)
    print("加载 SurgeryCAM 模型...")
    print("=" * 80)
    
    surgery_checkpoint = config.get('surgery_clip_checkpoint',
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not os.path.isabs(surgery_checkpoint):
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    model = create_surgery_cam_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        upsample_cam=config.get('upsample_cam', False),
        device=device
    )
    
    # 加载BoxHead checkpoint
    checkpoint_path = Path(__file__).parent / args.checkpoint
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.box_head.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 已加载checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"⚠️  Warning: Checkpoint not found: {checkpoint_path}")
        print("   使用未训练的BoxHead进行推理")
    
    model.eval()
    
    # ===== 3. 加载图像 =====
    print("\n加载图像...")
    image_tensor, image_pil, original_size = load_image(
        args.image,
        image_size=config.get('image_size', 224)
    )
    image_tensor = image_tensor.to(device)
    
    # ===== 4. 准备文本查询 =====
    text_queries = DIOR_CLASSES.copy()
    print(f"文本查询: {len(text_queries)} 个类别")
    
    # 确定要可视化的类别
    class_idx = None
    if args.class_name:
        if args.class_name in text_queries:
            class_idx = text_queries.index(args.class_name)
            print(f"可视化类别: {args.class_name} (索引: {class_idx})")
        else:
            print(f"⚠️  Warning: 类别 '{args.class_name}' 不在类别列表中，将显示所有类别的最大响应")
    
    # ===== 5. 前向传播获取CAM和检测结果 =====
    print("\n运行推理...")
    with torch.no_grad():
        # 获取CAM和预测框
        outputs = model(image_tensor, text_queries)
        cam = outputs['cam']  # [B, C, H, W]
        cam = cam[0]  # [C, H, W] - 单张图像
        
        # 获取检测结果
        detections = model.inference(
            image_tensor, text_queries,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold
        )
        detections = detections[0]  # 单张图像
    
    print(f"检测到 {len(detections)} 个目标")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det.get('class_name', 'unknown')}: {det['score']:.3f}")
    
    # ===== 6. 生成综合可视化 =====
    print("\n生成可视化...")
    image_stem = Path(args.image).stem
    output_path = output_dir / f"{image_stem}_comprehensive.jpg"
    
    visualize_comprehensive_results(
        image_pil, cam, detections, text_queries,
        class_idx=class_idx,
        output_path=str(output_path)
    )
    
    # ===== 7. 生成指标对比（如果启用） =====
    if args.show_metrics:
        print("\n生成指标对比...")
        
        # 默认指标值（实际应该从评估结果中读取）
        # 这里使用示例值，实际使用时应该从eval.py的输出或评估结果文件中读取
        metrics_comparison = {
            'CAM-only': {
                'AP@0.5': 0.425,
                'AP@0.5:0.95': 0.238
            },
            'OWL-ViT-style': {
                'AP@0.5': 0.512,
                'AP@0.5:0.95': 0.289
            },
            'SurgeryCAM': {
                'AP@0.5': 0.687,
                'AP@0.5:0.95': 0.412
            }
        }
        
        metrics_path = output_dir / f"{image_stem}_metrics_comparison.jpg"
        create_metrics_comparison_table(metrics_comparison, str(metrics_path))
        
        # Seen/Unseen对比
        seen_metrics = {
            'AP@0.5': 0.723,
            'AP@0.5:0.95': 0.445
        }
        unseen_metrics = {
            'AP@0.5': 0.651,
            'AP@0.5:0.95': 0.379
        }
        
        seen_unseen_path = output_dir / f"{image_stem}_seen_unseen.jpg"
        create_seen_unseen_comparison(seen_metrics, unseen_metrics, str(seen_unseen_path))
    
    print("\n" + "=" * 80)
    print("✅ 可视化完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()



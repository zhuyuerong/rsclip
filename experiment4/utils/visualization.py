# -*- coding: utf-8 -*-
"""
可视化工具
用于可视化热图和检测框
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import torch


def visualize_heatmap_and_boxes(image, heatmap, pred_bboxes, gt_bbox, class_name, image_size=224):
    """
    可视化热图和检测框
    
    生成三列图像：
    1. 原始图像 + GT框
    2. 热图叠加
    3. 预测框 + GT框对比
    
    Args:
        image: torch.Tensor [3, H, W] 或 numpy.ndarray [H, W, 3]
        heatmap: numpy.ndarray [H, W] 热图
        pred_bboxes: List[[x_min, y_min, x_max, y_max]] 预测框
        gt_bbox: [x_min, y_min, x_max, y_max] 真实框
        class_name: str 类别名称
        image_size: int 图像尺寸
    
    Returns:
        fig: matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 处理图像格式
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().permute(1, 2, 0).numpy()
    else:
        img_np = image
    
    # 归一化到0-1
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    # 确保图像尺寸正确
    if img_np.shape[0] != image_size or img_np.shape[1] != image_size:
        img_np = cv2.resize(img_np, (image_size, image_size))
    
    # ===== 1. 原始图像 + GT框 =====
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original Image\nClass: {class_name}', fontsize=10)
    axes[0].axis('off')
    
    # 绘制GT框（绿色）
    if gt_bbox is not None and len(gt_bbox) == 4:
        x_min, y_min, x_max, y_max = gt_bbox
        width = x_max - x_min
        height = y_max - y_min
        rect_gt = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='green', facecolor='none',
            label='Ground Truth'
        )
        axes[0].add_patch(rect_gt)
        axes[0].legend(loc='upper right', fontsize=8)
    
    # ===== 2. 热图叠加 =====
    axes[1].imshow(img_np)
    
    # 确保热图是float32类型
    if heatmap.dtype != np.float32:
        heatmap = heatmap.astype(np.float32)
    
    # 上采样热图到图像尺寸
    if heatmap.shape[0] != image_size:
        heatmap_resized = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap
    
    # 归一化热图
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # 叠加热图（jet colormap，透明度0.5）
    axes[1].imshow(heatmap_norm, alpha=0.5, cmap='jet')
    axes[1].set_title('Similarity Heatmap Overlay', fontsize=10)
    axes[1].axis('off')
    
    # 添加colorbar
    cbar = plt.colorbar(axes[1].imshow(heatmap_norm, alpha=0, cmap='jet'), ax=axes[1], fraction=0.046)
    cbar.set_label('Similarity', fontsize=8)
    
    # ===== 3. 检测框对比 =====
    axes[2].imshow(img_np)
    axes[2].set_title('Detection Results', fontsize=10)
    axes[2].axis('off')
    
    # 绘制GT框（绿色，实线）
    if gt_bbox is not None and len(gt_bbox) == 4:
        x_min, y_min, x_max, y_max = gt_bbox
        width = x_max - x_min
        height = y_max - y_min
        rect_gt = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='green', facecolor='none',
            linestyle='-', label='GT'
        )
        axes[2].add_patch(rect_gt)
    
    # 绘制预测框（红色，虚线）
    if pred_bboxes is not None and len(pred_bboxes) > 0:
        for i, bbox in enumerate(pred_bboxes):
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            
            label = 'Pred' if i == 0 else None  # 只为第一个框添加图例
            rect_pred = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=1.5, edgecolor='red', facecolor='none',
                linestyle='--', label=label
            )
            axes[2].add_patch(rect_pred)
    
    axes[2].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig


def save_visualization(fig, save_path):
    """
    保存可视化图像
    
    Args:
        fig: matplotlib.figure.Figure
        save_path: str 保存路径
    """
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


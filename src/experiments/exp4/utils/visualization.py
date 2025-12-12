# -*- coding: utf-8 -*-
"""
Visualization Utilities
可视化工具：用于可视化检测结果、CAM热图等
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Dict, Optional, Tuple
import cv2


def visualize_detections(image: Image.Image,
                        detections: List[Dict],
                        text_queries: List[str],
                        output_path: Optional[str] = None,
                        show_labels: bool = True,
                        show_scores: bool = True) -> None:
    """
    可视化检测结果
    
    Args:
        image: PIL Image
        detections: List of detection dicts with keys:
            - 'box': [xmin, ymin, xmax, ymax] normalized
            - 'label': int
            - 'score': float
            - 'class_name': str
        text_queries: List of class names
        output_path: Path to save visualization
        show_labels: Whether to show class labels
        show_scores: Whether to show confidence scores
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    ax.axis('off')
    
    # Color palette for different classes
    colors = plt.cm.tab20(np.linspace(0, 1, len(text_queries)))
    
    # 绘制检测框
    for det in detections:
        box = det['box']
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        label = det['label']
        score = det['score']
        class_name = det.get('class_name', text_queries[label])
        
        # 转换为像素坐标
        h, w = image.size[1], image.size[0]
        xmin = box[0] * w
        ymin = box[1] * h
        xmax = box[2] * w
        ymax = box[3] * h
        
        # 选择颜色
        color = colors[label % len(colors)]
        
        # 绘制框
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        if show_labels or show_scores:
            label_text = class_name if show_labels else ""
            if show_scores:
                if label_text:
                    label_text += f": {score:.2f}"
                else:
                    label_text = f"{score:.2f}"
            
            if label_text:
                ax.text(
                    xmin, ymin - 5,
                    label_text,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                    fontsize=10, color='white', weight='bold'
                )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_cam(cam: torch.Tensor,
                  image: Image.Image,
                  class_names: List[str],
                  class_idx: Optional[int] = None,
                  output_path: Optional[str] = None,
                  alpha: float = 0.5) -> None:
    """
    可视化CAM热图
    
    Args:
        cam: [C, H, W] or [H, W] CAM tensor
        image: PIL Image
        class_names: List of class names
        class_idx: Specific class to visualize (if None, shows all)
        output_path: Path to save visualization
        alpha: Transparency for CAM overlay
    """
    if cam.dim() == 3:
        # [C, H, W] - select specific class or average
        if class_idx is not None:
            cam_vis = cam[class_idx].cpu().numpy()
            title = f"CAM: {class_names[class_idx]}"
        else:
            cam_vis = cam.max(dim=0)[0].cpu().numpy()
            title = "CAM: Max across classes"
    else:
        # [H, W]
        cam_vis = cam.cpu().numpy()
        title = "CAM"
    
    # Resize CAM to image size
    cam_vis = cv2.resize(cam_vis, image.size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1]
    cam_vis = (cam_vis - cam_vis.min()) / (cam_vis.max() - cam_vis.min() + 1e-6)
    
    # Apply colormap
    cam_colored = plt.cm.jet(cam_vis)[:, :, :3]  # Remove alpha channel
    cam_colored = (cam_colored * 255).astype(np.uint8)
    
    # Overlay on image
    image_np = np.array(image)
    if image_np.shape[2] == 3:
        overlay = (alpha * cam_colored + (1 - alpha) * image_np).astype(np.uint8)
    else:
        overlay = image_np
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(cam_colored)
    axes[1].set_title("CAM Heatmap")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"CAM visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_peaks(cam: torch.Tensor,
                    peaks: List[Tuple[int, int, float]],
                    image: Image.Image,
                    output_path: Optional[str] = None) -> None:
    """
    可视化CAM上的检测到的峰
    
    Args:
        cam: [H, W] CAM tensor
        peaks: List of (i, j, score) tuples
        image: PIL Image
        output_path: Path to save visualization
    """
    cam_np = cam.cpu().numpy()
    
    # Resize CAM to image size for visualization
    cam_resized = cv2.resize(cam_np, image.size, interpolation=cv2.INTER_LINEAR)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image with peaks
    axes[0].imshow(image)
    axes[0].set_title("Detected Peaks")
    axes[0].axis('off')
    
    # Map peaks to image coordinates
    H, W = cam.shape
    h_img, w_img = image.size[1], image.size[0]
    
    for i, j, score in peaks:
        # Convert CAM coordinates to image coordinates
        x = (j + 0.5) / W * w_img
        y = (i + 0.5) / H * h_img
        
        # Draw peak
        circle = plt.Circle((x, y), radius=10, color='red', fill=True, alpha=0.7)
        axes[0].add_patch(circle)
        axes[0].text(x, y - 15, f'{score:.2f}', 
                    ha='center', va='bottom', color='red', weight='bold')
    
    # CAM heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title("CAM with Peaks")
    axes[1].axis('off')
    
    # Draw peaks on CAM
    for i, j, score in peaks:
        axes[1].plot(j, i, 'ro', markersize=10, markeredgecolor='white', 
                    markeredgewidth=2)
        axes[1].text(j, i - 0.5, f'{score:.2f}', 
                    ha='center', va='bottom', color='white', weight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Peaks visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_matching(peaks: List[Tuple[int, int, float]],
                      gt_boxes: torch.Tensor,
                      matches: List[Tuple[int, int]],
                      image: Image.Image,
                      cam_height: int,
                      cam_width: int,
                      output_path: Optional[str] = None) -> None:
    """
    可视化峰和GT的匹配结果
    
    Args:
        peaks: List of (i, j, score) tuples
        gt_boxes: [K, 4] GT boxes (normalized)
        matches: List of (peak_idx, gt_idx) tuples
        image: PIL Image
        cam_height, cam_width: CAM resolution
        output_path: Path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    ax.axis('off')
    ax.set_title("Peak-GT Matching")
    
    h_img, w_img = image.size[1], image.size[0]
    
    # Draw GT boxes
    for k, gt_box in enumerate(gt_boxes):
        if isinstance(gt_box, torch.Tensor):
            gt_box = gt_box.cpu().numpy()
        xmin, ymin, xmax, ymax = gt_box
        
        xmin_px = xmin * w_img
        ymin_px = ymin * h_img
        xmax_px = xmax * w_img
        ymax_px = ymax * h_img
        
        rect = patches.Rectangle(
            (xmin_px, ymin_px), xmax_px - xmin_px, ymax_px - ymin_px,
            linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(xmin_px, ymin_px - 5, f'GT{k}', 
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
               fontsize=8, color='white')
    
    # Draw peaks and matches
    for peak_idx, (i, j, score) in enumerate(peaks):
        x = (j + 0.5) / cam_width * w_img
        y = (i + 0.5) / cam_height * h_img
        
        # Check if this peak is matched
        matched = False
        for p_idx, g_idx in matches:
            if p_idx == peak_idx:
                matched = True
                # Draw line to matched GT
                gt_box = gt_boxes[g_idx]
                if isinstance(gt_box, torch.Tensor):
                    gt_box = gt_box.cpu().numpy()
                cx_gt = (gt_box[0] + gt_box[2]) / 2 * w_img
                cy_gt = (gt_box[1] + gt_box[3]) / 2 * h_img
                
                ax.plot([x, cx_gt], [y, cy_gt], 'r-', linewidth=1, alpha=0.5)
                break
        
        # Draw peak
        color = 'red' if matched else 'orange'
        circle = plt.Circle((x, y), radius=8, color=color, fill=True, alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y - 12, f'P{peak_idx}\n{score:.2f}', 
               ha='center', va='bottom', color=color, weight='bold', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Matching visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_detection_results(detections: List[Dict],
                          image_id: str,
                          output_dir: str,
                          format: str = 'json') -> None:
    """
    保存检测结果到文件
    
    Args:
        detections: List of detection dicts
        image_id: Image ID
        output_dir: Output directory
        format: 'json' or 'txt'
    """
    import json
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        output_path = output_dir / f"{image_id}_detections.json"
        results = {
            'image_id': image_id,
            'num_detections': len(detections),
            'detections': [
                {
                    'class_name': d.get('class_name', ''),
                    'label': d['label'],
                    'score': float(d['score']),
                    'box': d['box'].tolist() if isinstance(d['box'], torch.Tensor) else d['box']
                }
                for d in detections
            ]
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    else:  # txt
        output_path = output_dir / f"{image_id}_detections.txt"
        with open(output_path, 'w') as f:
            f.write(f"Image ID: {image_id}\n")
            f.write(f"Number of detections: {len(detections)}\n")
            f.write("-" * 80 + "\n")
            for i, d in enumerate(detections):
                f.write(f"\nDetection {i+1}:\n")
                f.write(f"  Class: {d.get('class_name', '')}\n")
                f.write(f"  Label: {d['label']}\n")
                f.write(f"  Score: {d['score']:.4f}\n")
                box = d['box']
                if isinstance(box, torch.Tensor):
                    box = box.tolist()
                f.write(f"  Box: {box}\n")
    
    print(f"Detection results saved to: {output_path}")



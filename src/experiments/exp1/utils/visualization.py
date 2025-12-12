# -*- coding: utf-8 -*-
"""
Visualization utilities for CAM display
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F


def visualize_cam(image, cam, text_queries, save_path, alpha=0.5):
    """
    Visualize CAM overlayed on image
    
    Args:
        image: [3, H, W] or [H, W, 3] - image tensor or numpy array
        cam: [C, H, W] or [H, W] - CAM tensor or numpy array
        text_queries: List[str] - class names
        save_path: Path to save visualization
        alpha: Transparency for CAM overlay
    """
    # Convert to numpy if tensor
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            # [3, H, W] -> [H, W, 3]
            image_np = image.cpu().permute(1, 2, 0).numpy()
        else:
            image_np = image.cpu().numpy()
    else:
        image_np = np.array(image)
    
    if isinstance(cam, torch.Tensor):
        cam_np = cam.cpu().numpy()
    else:
        cam_np = np.array(cam)
    
    # Normalize image to [0, 1]
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    image_np = np.clip(image_np, 0, 1)
    
    # Handle CAM shape
    if cam_np.ndim == 2:
        # Single CAM: [H, W]
        cam_np = cam_np[np.newaxis, :, :]  # [1, H, W]
    
    if cam_np.ndim == 3:
        # Multi-class CAM: [C, H, W]
        num_classes = cam_np.shape[0]
    else:
        raise ValueError(f"Unexpected CAM shape: {cam_np.shape}")
    
    # Resize CAM to match image if needed
    if cam_np.shape[1:] != image_np.shape[:2]:
        cam_resized = []
        for c in range(num_classes):
            cam_c = F.interpolate(
                torch.from_numpy(cam_np[c:c+1, np.newaxis, :, :]),
                size=(image_np.shape[0], image_np.shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            cam_resized.append(cam_c)
        cam_np = np.array(cam_resized)
    
    # Create subplots
    num_classes = min(num_classes, len(text_queries))
    fig, axes = plt.subplots(1, num_classes + 1, figsize=(4 * (num_classes + 1), 4))
    
    if num_classes == 0:
        axes = [axes]
    
    # Display original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display CAM for each class
    for i in range(num_classes):
        # Normalize CAM to [0, 1]
        heatmap = cam_np[i]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Overlay on image
        axes[i + 1].imshow(image_np)
        im = axes[i + 1].imshow(heatmap, alpha=alpha, cmap='jet')
        axes[i + 1].set_title(f'{text_queries[i] if i < len(text_queries) else f"Class {i}"}')
        axes[i + 1].axis('off')
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_cam_single(image, cam, class_name, save_path, alpha=0.5):
    """
    Visualize single-class CAM
    
    Args:
        image: [3, H, W] or [H, W, 3] - image tensor or numpy array
        cam: [H, W] - single CAM
        class_name: str - class name
        save_path: Path to save
        alpha: Transparency
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image_np = image.cpu().permute(1, 2, 0).numpy()
        else:
            image_np = image.cpu().numpy()
    else:
        image_np = np.array(image)
    
    if isinstance(cam, torch.Tensor):
        cam_np = cam.cpu().numpy()
    else:
        cam_np = np.array(cam)
    
    # Normalize
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    image_np = np.clip(image_np, 0, 1)
    
    # Normalize CAM
    if cam_np.ndim == 2:
        heatmap = cam_np
    elif cam_np.ndim == 3 and cam_np.shape[0] == 1:
        heatmap = cam_np[0]
    else:
        raise ValueError(f"Unexpected CAM shape: {cam_np.shape}")
    
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize if needed
    if heatmap.shape != image_np.shape[:2]:
        heatmap = F.interpolate(
            torch.from_numpy(heatmap[np.newaxis, np.newaxis, :, :]),
            size=(image_np.shape[0], image_np.shape[1]),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(image_np)
    im = axes[1].imshow(heatmap, alpha=alpha, cmap='jet')
    axes[1].set_title(f'CAM: {class_name}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()






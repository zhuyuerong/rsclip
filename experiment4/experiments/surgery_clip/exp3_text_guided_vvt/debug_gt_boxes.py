# -*- coding: utf-8 -*-
"""
GT Bounding Box Debug Visualizer

Display GT boxes with detailed coordinate information
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add paths
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))
surgery_clip_dir = Path(__file__).parent.parent
sys.path.append(str(surgery_clip_dir))

from utils.seen_unseen_split import SeenUnseenDataset


def visualize_gt_boxes_debug(dataset, num_samples=5):
    """
    Visualize GT boxes with detailed information
    
    Shows:
    - Original image with GT boxes
    - Image ID and class name
    - Original size and scaled coordinates
    - All bbox coordinates
    """
    output_dir = Path(__file__).parent / 'gt_box_debug'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        
        # Get image info
        image_id = sample.get('image_id', f'sample_{idx}')
        class_name = sample['class_name']
        original_size = sample.get('original_size', (224, 224))  # (H, W)
        bboxes = sample['bboxes']
        
        # Load original image (before transform)
        image_path = sample.get('image_path', '')
        if image_path and Path(image_path).exists():
            img_pil = Image.open(image_path).convert('RGB')
            img_np = np.array(img_pil)
        else:
            # Use transformed image and denormalize
            img_tensor = sample['image']
            img = img_tensor.permute(1, 2, 0).numpy()
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img = img * std + mean
            img_np = np.clip(img, 0, 1)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # ===== Left: Original image with GT boxes (224x224) =====
        axes[0].imshow(img_np)
        axes[0].set_title(f'Image ID: {image_id}\nClass: {class_name}\n224x224 (Scaled)', fontsize=10)
        
        # Calculate scale factors
        orig_h, orig_w = original_size
        scale_x = 224.0 / orig_w
        scale_y = 224.0 / orig_h
        
        # Draw all bboxes
        for i, bbox in enumerate(bboxes):
            # Scale bbox
            xmin = bbox['xmin'] * scale_x
            ymin = bbox['ymin'] * scale_y
            xmax = bbox['xmax'] * scale_x
            ymax = bbox['ymax'] * scale_y
            w = xmax - xmin
            h = ymax - ymin
            
            # Draw rectangle
            rect = patches.Rectangle((xmin, ymin), w, h, 
                                    linewidth=2, edgecolor='lime', facecolor='none')
            axes[0].add_patch(rect)
            
            # Add bbox number label
            axes[0].text(xmin, ymin-5, f'Box{i}', color='lime', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        axes[0].axis('off')
        
        # ===== Right: Coordinate information text =====
        axes[1].axis('off')
        
        # Build info text
        info_lines = [
            f"IMAGE INFORMATION",
            f"=" * 50,
            f"Image ID: {image_id}",
            f"Class: {class_name}",
            f"Original Size: {orig_w}x{orig_h} pixels",
            f"Scaled Size: 224x224 pixels",
            f"Scale Factor: x={scale_x:.4f}, y={scale_y:.4f}",
            f"",
            f"BOUNDING BOXES ({len(bboxes)} total)",
            f"=" * 50,
        ]
        
        for i, bbox in enumerate(bboxes):
            xmin_orig = bbox['xmin']
            ymin_orig = bbox['ymin']
            xmax_orig = bbox['xmax']
            ymax_orig = bbox['ymax']
            
            xmin_scaled = xmin_orig * scale_x
            ymin_scaled = ymin_orig * scale_y
            xmax_scaled = xmax_orig * scale_x
            ymax_scaled = ymax_orig * scale_y
            
            info_lines.extend([
                f"",
                f"Box {i}: {bbox.get('class', class_name)}",
                f"  Original coords (800x800):",
                f"    xmin={xmin_orig}, ymin={ymin_orig}",
                f"    xmax={xmax_orig}, ymax={ymax_orig}",
                f"    size={xmax_orig-xmin_orig}x{ymax_orig-ymin_orig}",
                f"  Scaled coords (224x224):",
                f"    xmin={xmin_scaled:.1f}, ymin={ymin_scaled:.1f}",
                f"    xmax={xmax_scaled:.1f}, ymax={ymax_scaled:.1f}",
                f"    size={xmax_scaled-xmin_scaled:.1f}x{ymax_scaled-ymin_scaled:.1f}",
            ])
        
        info_text = '\n'.join(info_lines)
        axes[1].text(0.05, 0.95, info_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'debug_sample{idx}_{image_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sample {idx} ({image_id}) debug visualization saved")
    
    print(f"\nAll debug visualizations saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Debug GT bounding boxes')
    parser.add_argument('--dataset', type=str, default='datasets/mini_dataset')
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()
    
    print("=" * 60)
    print("GT Bounding Box Debug Visualizer")
    print("=" * 60)
    
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    dataset = SeenUnseenDataset(args.dataset, split='all', mode='val', unseen_classes=unseen_classes)
    
    print(f"Dataset: {args.dataset}")
    print(f"Total samples: {len(dataset)}")
    print(f"Visualizing: {args.num_samples} samples\n")
    
    visualize_gt_boxes_debug(dataset, num_samples=args.num_samples)
    
    print("\n" + "=" * 60)
    print("Check the output images to verify GT box positions!")
    print("=" * 60)



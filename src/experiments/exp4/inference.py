#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference Script
单张图片推理，支持可视化检测结果
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
from torchvision import transforms

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.owlvit_baseline import create_owlvit_model
from models.surgery_cam_detector import create_surgery_cam_detector


def load_image(image_path: str, image_size: int = 224):
    """加载和预处理图片"""
    image = Image.open(image_path).convert('RGB')
    
    # 保存原始尺寸
    original_size = image.size
    
    # 预处理
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


def visualize_detections(image, detections, text_queries, output_path=None):
    """可视化检测结果"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    ax.axis('off')
    
    # 绘制检测框
    for det in detections:
        box = det['box'].numpy()  # [xmin, ymin, xmax, ymax] normalized
        label = det['label']
        score = det['score']
        class_name = det['class_name']
        
        # 转换为像素坐标
        h, w = image.size[1], image.size[0]
        xmin = box[0] * w
        ymin = box[1] * h
        xmax = box[2] * w
        ymax = box[3] * h
        
        # 绘制框
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        ax.text(
            xmin, ymin - 5,
            f'{class_name}: {score:.2f}',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
            fontsize=10, color='white', weight='bold'
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Inference on Single Image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, choices=['surgery_cam', 'owlvit'],
                       default='surgery_cam', help='Model type')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (for surgery_cam)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--text-queries', type=str, nargs='+', default=None,
                       help='Text queries (class names). If None, uses all DIOR classes')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                       help='NMS IoU threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # ===== 1. Load model =====
    print("=" * 80)
    print(f"Loading {args.model} model...")
    print("=" * 80)
    
    if args.model == 'surgery_cam':
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for surgery_cam model")
        
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
        
        # Load BoxHead checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.box_head.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:  # owlvit
        model = create_owlvit_model(
            model_name=config.get('model_name', 'google/owlvit-base-patch32'),
            device=device
        )
    
    # ===== 2. Load image =====
    print("\nLoading image...")
    image_tensor, image_pil, original_size = load_image(
        args.image, 
        image_size=config.get('image_size', 224)
    )
    image_tensor = image_tensor.to(device)
    
    # ===== 3. Prepare text queries =====
    if args.text_queries is None:
        # Use all DIOR classes
        text_queries = [
            "airplane", "airport", "baseball field", "basketball court",
            "bridge", "chimney", "dam", "expressway service area",
            "expressway toll station", "golf course", "ground track field",
            "harbor", "overpass", "ship", "stadium", "storage tank",
            "tennis court", "train station", "vehicle", "wind mill"
        ]
    else:
        text_queries = args.text_queries
    
    print(f"Text queries: {text_queries}")
    
    # ===== 4. Inference =====
    print("\nRunning inference...")
    with torch.no_grad():
        if args.model == 'surgery_cam':
            detections = model.inference(
                image_tensor, text_queries,
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold
            )
            detections = detections[0]  # Single image
        else:  # owlvit
            outputs = model.inference(
                image_tensor, text_queries,
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold
            )
            # Convert to detections format
            detections = []
            for i in range(len(outputs['pred_boxes'][0])):
                detections.append({
                    'box': outputs['pred_boxes'][0][i].cpu(),
                    'label': outputs['pred_labels'][0][i].item(),
                    'score': outputs['pred_scores'][0][i].item(),
                    'class_name': text_queries[outputs['pred_labels'][0][i].item()]
                })
    
    # ===== 5. Print results =====
    print("\n" + "=" * 80)
    print("Detection Results:")
    print("=" * 80)
    print(f"Found {len(detections)} detections:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']:25s} "
              f"score: {det['score']:.3f} "
              f"box: [{det['box'][0]:.3f}, {det['box'][1]:.3f}, "
              f"{det['box'][2]:.3f}, {det['box'][3]:.3f}]")
    
    # ===== 6. Visualize =====
    if args.output or True:  # Always visualize
        output_path = args.output or f"{Path(args.image).stem}_detections.jpg"
        visualize_detections(image_pil, detections, text_queries, output_path)


if __name__ == '__main__':
    main()



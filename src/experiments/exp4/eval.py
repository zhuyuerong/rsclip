#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation Script
支持OWL-ViT和SurgeryCAM两种模型的评估
"""

import torch
import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.dior_detection import get_detection_dataloader
from utils.metrics import DetectionMetrics
from models.owlvit_baseline import create_owlvit_model
from models.surgery_cam_detector import create_surgery_cam_detector


def evaluate_model(model, dataloader, device, model_type='surgery_cam', 
                  conf_threshold=0.3, nms_threshold=0.5):
    """
    评估模型
    
    Args:
        model: 模型实例
        dataloader: 数据加载器
        device: 设备
        model_type: 'surgery_cam' or 'owlvit'
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
    
    Returns:
        metrics字典
    """
    model.eval()
    metrics = DetectionMetrics(num_classes=20)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            
            # 获取预测
            if model_type == 'surgery_cam':
                detections = model.inference(
                    images, text_queries,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
                
                # 转换为metrics格式
                pred_boxes = [d['boxes'].to(device) for d in detections]
                pred_labels = [d['labels'].to(device) for d in detections]
                pred_scores = [d['scores'].to(device) for d in detections]
            else:  # owlvit
                outputs = model.inference(
                    images, text_queries,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
                pred_boxes = outputs['pred_boxes']
                pred_labels = outputs['pred_labels']
                pred_scores = outputs['pred_scores']
            
            # 更新metrics
            metrics.update(
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                gt_boxes=batch['boxes'],
                gt_labels=batch['labels'],
                image_ids=batch['image_ids']
            )
    
    # 计算mAP
    results = metrics.compute_map()
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Detection Models')
    parser.add_argument('--model', type=str, choices=['surgery_cam', 'owlvit'],
                       default='surgery_cam', help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                       help='NMS IoU threshold')
    
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
        # Load SurgeryCAM model
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
        # OWL-ViT doesn't need checkpoint loading for zero-shot
    
    # ===== 2. Load data =====
    print("\nLoading data...")
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment=False
    )
    
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # ===== 3. Evaluate =====
    print("\n" + "=" * 80)
    print("Evaluating...")
    print("=" * 80)
    
    results = evaluate_model(
        model, val_loader, device,
        model_type=args.model,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold
    )
    
    # ===== 4. Print results =====
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
    
    # Per-class APs
    print("\nPer-class AP@0.5:")
    classes = val_loader.dataset.classes
    for i, (cls_name, ap) in enumerate(zip(classes, results['per_class_AP@0.5'])):
        print(f"  {cls_name:25s}: {ap:.4f}")
    
    # Save results
    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f'{args.model}_eval_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"{args.model.upper()} Evaluation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"mAP@0.5: {results['mAP@0.5']:.4f}\n")
        f.write(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}\n")
        f.write(f"\nConfidence threshold: {args.conf_threshold}\n")
        f.write(f"NMS threshold: {args.nms_threshold}\n")
        f.write("\nPer-class AP@0.5:\n")
        for i, (cls_name, ap) in enumerate(zip(classes, results['per_class_AP@0.5'])):
            f.write(f"  {cls_name:25s}: {ap:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()



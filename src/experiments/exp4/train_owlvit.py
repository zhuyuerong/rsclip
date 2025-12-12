#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OWL-ViT Training Script
用于验证OWL-ViT在DIOR数据集上的表现
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.owlvit_baseline import create_owlvit_model
from datasets.dior_detection import get_detection_dataloader
from utils.metrics import DetectionMetrics


def evaluate(model, dataloader, device, conf_threshold=0.3):
    """Evaluate model on validation set"""
    model.eval()
    metrics = DetectionMetrics(num_classes=20)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            
            # Get predictions
            outputs = model.inference(
                images, text_queries,
                conf_threshold=conf_threshold,
                nms_threshold=0.5
            )
            
            # Update metrics
            metrics.update(
                pred_boxes=outputs['pred_boxes'],
                pred_labels=outputs['pred_labels'],
                pred_scores=outputs['pred_scores'],
                gt_boxes=batch['boxes'],
                gt_labels=batch['labels'],
                image_ids=batch['image_ids']
            )
    
    # Compute mAP
    results = metrics.compute_map()
    return results


def main():
    parser = argparse.ArgumentParser(description='Train/Evaluate OWL-ViT on DIOR')
    parser.add_argument('--config', type=str, default='configs/owlvit_config.yaml',
                       help='Path to config file')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate, do not train')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (for evaluation)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Creating default config...")
        create_default_config(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config.get('device', 'cuda')
    
    # Create model
    print("=" * 80)
    print("Loading OWL-ViT model...")
    print("=" * 80)
    
    model = create_owlvit_model(
        model_name=config.get('model_name', 'google/owlvit-base-patch32'),
        device=device
    )
    
    # Create data loaders
    print("\nLoading data...")
    train_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='trainval',
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment=False
    )
    
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
        image_size=config.get('image_size', 224),
        augment=False
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Evaluation only
    if args.eval_only:
        print("\n" + "=" * 80)
        print("Evaluating OWL-ViT...")
        print("=" * 80)
        
        results = evaluate(
            model, val_loader, device,
            conf_threshold=config.get('conf_threshold', 0.3)
        )
        
        print("\n" + "=" * 80)
        print("Evaluation Results:")
        print("=" * 80)
        print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
        print(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
        
        # Per-class APs
        print("\nPer-class AP@0.5:")
        classes = train_loader.dataset.classes
        for i, (cls_name, ap) in enumerate(zip(classes, results['per_class_AP@0.5'])):
            print(f"  {cls_name:25s}: {ap:.4f}")
        
        return
    
    # Note: OWL-ViT is typically used in zero-shot mode
    # Fine-tuning requires additional implementation
    print("\n" + "=" * 80)
    print("OWL-ViT Zero-Shot Evaluation")
    print("=" * 80)
    print("Note: OWL-ViT is typically used in zero-shot mode.")
    print("For fine-tuning, additional implementation is required.")
    
    # Evaluate
    results = evaluate(
        model, val_loader, device,
        conf_threshold=config.get('conf_threshold', 0.3)
    )
    
    print("\n" + "=" * 80)
    print("Zero-Shot Evaluation Results:")
    print("=" * 80)
    print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
    
    # Save results
    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'owlvit_results.txt'
    with open(results_file, 'w') as f:
        f.write("OWL-ViT Zero-Shot Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"mAP@0.5: {results['mAP@0.5']:.4f}\n")
        f.write(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}\n")
        f.write("\nPer-class AP@0.5:\n")
        classes = train_loader.dataset.classes
        for i, (cls_name, ap) in enumerate(zip(classes, results['per_class_AP@0.5'])):
            f.write(f"  {cls_name:25s}: {ap:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")


def create_default_config(config_path: Path):
    """Create default config file"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        'model_name': 'google/owlvit-base-patch32',
        'device': 'cuda',
        'batch_size': 4,
        'num_workers': 4,
        'image_size': 224,
        'conf_threshold': 0.3,
        'dataset_root': None,
        'output_dir': 'outputs'
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)


if __name__ == '__main__':
    main()



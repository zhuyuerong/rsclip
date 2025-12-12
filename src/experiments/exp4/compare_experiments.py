#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比实验脚本
对比不同分辨率、不同正样本分配策略、不同模型
"""

import torch
import sys
import os
from pathlib import Path
import argparse
import yaml
import json
from datetime import datetime
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.dior_detection import get_detection_dataloader
from utils.metrics import DetectionMetrics
from models.owlvit_baseline import create_owlvit_model
from models.surgery_cam_detector import create_surgery_cam_detector


def evaluate_model(model, dataloader, device, model_type='surgery_cam',
                  conf_threshold=0.3, nms_threshold=0.5):
    """评估模型并返回metrics"""
    model.eval()
    metrics = DetectionMetrics(num_classes=20)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            
            if model_type == 'surgery_cam':
                detections = model.inference(
                    images, text_queries,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
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
            
            metrics.update(
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                gt_boxes=batch['boxes'],
                gt_labels=batch['labels'],
                image_ids=batch['image_ids']
            )
    
    results = metrics.compute_map()
    return results


def compare_resolutions(config_base, checkpoints, device):
    """对比不同CAM分辨率"""
    print("\n" + "=" * 80)
    print("Experiment 1: CAM Resolution Comparison")
    print("=" * 80)
    
    results = {}
    
    for resolution_name, checkpoint_path in checkpoints.items():
        print(f"\nEvaluating {resolution_name}...")
        
        # Load model
        model = create_surgery_cam_detector(
            surgery_clip_checkpoint=config_base['surgery_clip_checkpoint'],
            num_classes=config_base.get('num_classes', 20),
            cam_resolution=config_base.get('cam_resolution', 7),
            upsample_cam=(resolution_name == 'upsampled'),
            device=device
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.box_head.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        val_loader = get_detection_dataloader(
            root=config_base.get('dataset_root'),
            split='test',
            batch_size=4,
            num_workers=2,
            image_size=224,
            augment=False
        )
        
        metrics = evaluate_model(
            model, val_loader, device,
            model_type='surgery_cam',
            conf_threshold=0.3,
            nms_threshold=0.5
        )
        
        results[resolution_name] = {
            'mAP@0.5': metrics['mAP@0.5'],
            'mAP@0.5:0.95': metrics['mAP@0.5:0.95']
        }
        
        print(f"  mAP@0.5: {metrics['mAP@0.5']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    
    return results


def compare_models(config_base, checkpoints, device):
    """对比不同模型（OWL-ViT vs SurgeryCAM）"""
    print("\n" + "=" * 80)
    print("Experiment 2: Model Comparison (OWL-ViT vs SurgeryCAM)")
    print("=" * 80)
    
    results = {}
    
    # Load data
    val_loader = get_detection_dataloader(
        root=config_base.get('dataset_root'),
        split='test',
        batch_size=4,
        num_workers=2,
        image_size=224,
        augment=False
    )
    
    # Evaluate OWL-ViT
    print("\nEvaluating OWL-ViT...")
    owlvit_model = create_owlvit_model(
        model_name=config_base.get('owlvit_model_name', 'google/owlvit-base-patch32'),
        device=device
    )
    
    owlvit_metrics = evaluate_model(
        owlvit_model, val_loader, device,
        model_type='owlvit',
        conf_threshold=0.3,
        nms_threshold=0.5
    )
    
    results['OWL-ViT'] = {
        'mAP@0.5': owlvit_metrics['mAP@0.5'],
        'mAP@0.5:0.95': owlvit_metrics['mAP@0.5:0.95']
    }
    print(f"  mAP@0.5: {owlvit_metrics['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {owlvit_metrics['mAP@0.5:0.95']:.4f}")
    
    # Evaluate SurgeryCAM
    if 'surgery_cam' in checkpoints:
        print("\nEvaluating SurgeryCAM...")
        surgery_model = create_surgery_cam_detector(
            surgery_clip_checkpoint=config_base['surgery_clip_checkpoint'],
            num_classes=config_base.get('num_classes', 20),
            cam_resolution=config_base.get('cam_resolution', 7),
            upsample_cam=config_base.get('upsample_cam', False),
            device=device
        )
        
        checkpoint = torch.load(checkpoints['surgery_cam'], map_location=device)
        surgery_model.box_head.load_state_dict(checkpoint['model_state_dict'])
        
        surgery_metrics = evaluate_model(
            surgery_model, val_loader, device,
            model_type='surgery_cam',
            conf_threshold=0.3,
            nms_threshold=0.5
        )
        
        results['SurgeryCAM'] = {
            'mAP@0.5': surgery_metrics['mAP@0.5'],
            'mAP@0.5:0.95': surgery_metrics['mAP@0.5:0.95']
        }
        print(f"  mAP@0.5: {surgery_metrics['mAP@0.5']:.4f}")
        print(f"  mAP@0.5:0.95: {surgery_metrics['mAP@0.5:0.95']:.4f}")
    
    return results


def compare_strategies(config_base, checkpoints, device):
    """对比不同正样本分配策略"""
    print("\n" + "=" * 80)
    print("Experiment 3: Assignment Strategy Comparison")
    print("=" * 80)
    print("Note: This requires models trained with different assignment strategies")
    print("For now, we compare single-peak vs multi-peak (if available)")
    
    # This would require models trained with different strategies
    # For now, just return placeholder
    results = {
        'single_peak': {'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0},
        'multi_peak': {'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0}
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare Experiments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['resolution', 'models', 'strategies', 'all'],
                       default=['all'], help='Experiments to run')
    parser.add_argument('--checkpoints', type=str, nargs='+',
                       help='Checkpoint paths (format: name:path)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Parse checkpoints
    checkpoints = {}
    if args.checkpoints:
        for cp in args.checkpoints:
            name, path = cp.split(':')
            checkpoints[name] = path
    
    all_results = {}
    
    # Run experiments
    if 'all' in args.experiments or 'resolution' in args.experiments:
        if 'original' in checkpoints and 'upsampled' in checkpoints:
            res_results = compare_resolutions(config, checkpoints, device)
            all_results['resolution_comparison'] = res_results
        else:
            print("Warning: Need 'original' and 'upsampled' checkpoints for resolution comparison")
    
    if 'all' in args.experiments or 'models' in args.experiments:
        model_results = compare_models(config, checkpoints, device)
        all_results['model_comparison'] = model_results
    
    if 'all' in args.experiments or 'strategies' in args.experiments:
        strategy_results = compare_strategies(config, checkpoints, device)
        all_results['strategy_comparison'] = strategy_results
    
    # Save results
    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'comparison_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    
    for exp_name, exp_results in all_results.items():
        print(f"\n{exp_name}:")
        for model_name, metrics in exp_results.items():
            print(f"  {model_name}:")
            print(f"    mAP@0.5: {metrics['mAP@0.5']:.4f}")
            print(f"    mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()



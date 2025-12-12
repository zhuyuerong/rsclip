#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for SurgeryCLIP + AAF + p2p experiment
"""

import torch
import numpy as np
import yaml
import sys
import os
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.surgery_aaf import create_surgery_aaf_model
from utils.data import get_dataloader
from utils.visualization import visualize_cam, visualize_cam_single
from utils.metrics import compute_metrics, compute_ap


def evaluate(config):
    """
    Evaluate SurgeryAAF model
    """
    device = config['device']
    
    # ===== 1. Load model =====
    print("=" * 80)
    print("Loading model...")
    print("=" * 80)
    
    checkpoint_path = config['clip_weights_path']
    if not os.path.isabs(checkpoint_path):
        project_root = Path(__file__).parent.parent.parent.parent
        checkpoint_path = project_root / checkpoint_path
        checkpoint_path = str(checkpoint_path)
    
    model, preprocess = create_surgery_aaf_model(
        checkpoint_path=checkpoint_path,
        device=device,
        num_layers=config.get('num_layers', 6)
    )
    
    # Load trained AAF weights
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_file = checkpoint_dir / 'best_model.pth'
    
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.aaf.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        print("⚠️  No checkpoint found, using randomly initialized AAF")
    
    model.eval()
    
    # ===== 2. Load test data =====
    print("\nLoading test data...")
    test_loader = get_dataloader(
        dataset_name=config['dataset'],
        root=config.get('dataset_root'),
        split='test',
        batch_size=1,  # Evaluate one by one
        num_workers=config['num_workers'],
        shuffle=False
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # ===== 3. Evaluation =====
    print("\n" + "=" * 80)
    print("Evaluating...")
    print("=" * 80)
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    all_metrics = []
    all_aps = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)  # [1, C]
            text_queries = batch['text_queries'][0]  # List of class names
            image_id = batch['image_id'][0]
            
            # Generate CAM
            cam, aux = model(images, text_queries)
            # cam: [1, C, N, N]
            
            # Upsample CAM to original image size
            original_size = images.shape[2:]  # [H, W]
            cam_upsampled = F.interpolate(
                cam,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            # cam_upsampled: [1, C, H, W]
            
            # Compute metrics
            metrics = compute_metrics(
                cam=cam_upsampled[0].cpu(),  # [C, H, W]
                labels=labels[0].cpu(),  # [C]
                text_queries=text_queries,
                threshold=config.get('eval_threshold', 0.5)
            )
            all_metrics.append(metrics)
            
            # Compute AP
            ap_per_class, mean_ap = compute_ap(
                cam=cam_upsampled[0].cpu(),
                labels=labels[0].cpu(),
                text_queries=text_queries
            )
            all_aps.append(mean_ap)
            
            # Visualization (first N samples)
            if batch_idx < config.get('visualize_num', 100):
                # Denormalize image for visualization
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                image_vis = images[0].cpu() * std + mean
                image_vis = torch.clamp(image_vis, 0, 1)
                
                # Visualize all classes
                save_path = vis_dir / f'{image_id}_cam.png'
                visualize_cam(
                    image=image_vis,
                    cam=cam_upsampled[0].cpu(),
                    text_queries=text_queries,
                    save_path=str(save_path)
                )
                
                # Visualize individual classes that are present
                present_classes = labels[0].cpu().numpy() > 0
                for i, (class_name, is_present) in enumerate(zip(text_queries, present_classes)):
                    if is_present and i < cam_upsampled.shape[1]:
                        save_path_single = vis_dir / f'{image_id}_{class_name}_cam.png'
                        visualize_cam_single(
                            image=image_vis,
                            cam=cam_upsampled[0, i].cpu(),
                            class_name=class_name,
                            save_path=str(save_path_single)
                        )
    
    # ===== 4. Aggregate results =====
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    
    # Average metrics
    avg_metrics = {}
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        print(f"{key.capitalize()}: {avg_metrics[key]:.4f}")
    
    mean_ap = np.mean(all_aps)
    print(f"Mean AP: {mean_ap:.4f}")
    
    # Per-class statistics
    print("\nPer-class statistics:")
    class_stats = {}
    for class_name in text_queries:
        class_stats[class_name] = {
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    for metrics in all_metrics:
        for class_name, class_metrics in metrics['per_class'].items():
            if class_name in class_stats:
                # Extract per-class metrics if available
                # (simplified - actual per-class metrics would need more computation)
                pass
    
    # Save results
    results = {
        'avg_metrics': avg_metrics,
        'mean_ap': float(mean_ap),
        'all_metrics': all_metrics,
        'all_aps': all_aps
    }
    
    import json
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save numpy arrays
    np.save(output_dir / 'all_metrics.npy', all_metrics)
    np.save(output_dir / 'all_aps.npy', all_aps)
    
    print(f"\n✅ Results saved to {output_dir}")
    print(f"   - Metrics: {results_path}")
    print(f"   - Visualizations: {vis_dir}")
    
    return results


if __name__ == '__main__':
    # Load configuration
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start evaluation
    evaluate(config)






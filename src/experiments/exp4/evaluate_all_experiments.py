#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验5.1: 完整评估
评估所有改进方案的性能
"""

import torch
import sys
import json
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.simple_surgery_cam_detector import create_simple_surgery_cam_detector
from models.enhanced_simple_surgery_cam_detector import create_enhanced_simple_surgery_cam_detector
from datasets.dior_detection import get_detection_dataloader
from utils.class_split import ALL_CLASSES, get_seen_class_indices, get_unseen_class_indices
from utils.detection_metrics import DetectionMetrics


def evaluate_model(model, dataloader, device, conf_threshold=0.1, nms_threshold=0.5):
    """
    评估模型性能
    
    Returns:
        dict: 评估结果
    """
    model.eval()
    
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []
    all_gt_boxes = []
    all_gt_labels = []
    
    text_queries = ALL_CLASSES.copy()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            images = batch['images'].to(device)
            boxes = batch['boxes']
            labels = batch['labels']
            
            # 推理
            detections_list = model.inference(
                images, text_queries,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )
            
            # 处理每个图像的检测结果
            for b in range(len(detections_list)):
                detections = detections_list[b]
                
                if len(detections) > 0:
                    pred_boxes = torch.stack([d['box'] for d in detections])
                    pred_labels = torch.tensor([d['class'] for d in detections])
                    pred_scores = torch.tensor([d['score'] for d in detections])
                else:
                    pred_boxes = torch.zeros((0, 4))
                    pred_labels = torch.zeros((0,), dtype=torch.long)
                    pred_scores = torch.zeros((0,))
                
                all_pred_boxes.append(pred_boxes)
                all_pred_labels.append(pred_labels)
                all_pred_scores.append(pred_scores)
                
                # GT
                all_gt_boxes.append(boxes[b])
                all_gt_labels.append(labels[b])
    
    # 计算指标
    metrics = DetectionMetrics()
    map_results = metrics.compute_map(
        all_pred_boxes, all_pred_labels, all_pred_scores,
        all_gt_boxes, all_gt_labels
    )
    
    # 计算seen/unseen mAP
    seen_indices = get_seen_class_indices()
    unseen_indices = get_unseen_class_indices()
    
    per_class_ap = map_results.get('per_class_AP@0.5', {})
    
    seen_ap = [per_class_ap.get(i, 0.0) for i in seen_indices]
    unseen_ap = [per_class_ap.get(i, 0.0) for i in unseen_indices]
    
    seen_map = np.mean(seen_ap) if seen_ap else 0.0
    unseen_map = np.mean(unseen_ap) if unseen_ap else 0.0
    
    results = {
        'mAP@0.5': map_results.get('mAP@0.5', 0.0),
        'mAP@0.5:0.95': map_results.get('mAP@0.5:0.95', 0.0),
        'seen_mAP@0.5': seen_map,
        'unseen_mAP@0.5': unseen_map,
        'per_class_AP@0.5': per_class_ap
    }
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='评估所有实验方案')
    parser.add_argument('--config', type=str, default='configs/surgery_cam_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'enhanced'],
                       help='模型类型')
    parser.add_argument('--output', type=str, default='outputs/final_evaluation/comprehensive_results.json',
                       help='输出结果路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config.get('device', 'cuda')
    print(f"使用设备: {device}")
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    print(f"创建模型（类型: {args.model_type}）...")
    if args.model_type == 'enhanced':
        model = create_enhanced_simple_surgery_cam_detector(
            surgery_clip_checkpoint=surgery_checkpoint,
            num_classes=config.get('num_classes', 20),
            cam_resolution=config.get('cam_resolution', 7),
            upsample_cam=config.get('upsample_cam', False),
            device=device,
            unfreeze_cam_last_layer=True
        )
    else:
        model = create_simple_surgery_cam_detector(
            surgery_clip_checkpoint=surgery_checkpoint,
            num_classes=config.get('num_classes', 20),
            cam_resolution=config.get('cam_resolution', 7),
            upsample_cam=config.get('upsample_cam', False),
            device=device,
            unfreeze_cam_last_layer=True
        )
    
    # 加载checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # 加载可训练参数
    trainable_state = {}
    for key, value in model_state.items():
        if key in model_dict and ('box_head' in key or 'cam_generator' in key):
            trainable_state[key] = value
    
    model_dict.update(trainable_state)
    model.load_state_dict(model_dict, strict=False)
    print(f"✅ 已加载checkpoint: {checkpoint_path}")
    
    # 加载验证集
    print("加载验证集...")
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=1,
        num_workers=0,
        image_size=config.get('image_size', 224),
        augment=False,
        train_only_seen=False
    )
    
    # 评估
    print("\n开始评估...")
    results = evaluate_model(
        model, val_loader, device,
        conf_threshold=config.get('conf_threshold', 0.1),
        nms_threshold=config.get('nms_threshold', 0.5)
    )
    
    # 打印结果
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
    print(f"Seen mAP@0.5: {results['seen_mAP@0.5']:.4f}")
    print(f"Unseen mAP@0.5: {results['unseen_mAP@0.5']:.4f}")
    print("\n每类别AP@0.5:")
    for class_idx, class_name in enumerate(ALL_CLASSES):
        ap = results['per_class_AP@0.5'].get(class_idx, 0.0)
        print(f"  {class_name}: {ap:.4f}")
    print("=" * 80)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加元数据
    results['checkpoint'] = str(checkpoint_path)
    results['model_type'] = args.model_type
    results['config'] = str(config_path)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 评估结果已保存到: {output_path}")


if __name__ == '__main__':
    main()



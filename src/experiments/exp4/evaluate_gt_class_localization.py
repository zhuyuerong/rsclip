#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GT类别定位评估脚本
在GT类别下评估定位性能
"""

import torch
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.gt_class_localization_detector import create_gt_class_localization_detector
from datasets.dior_detection import get_detection_dataloader
from losses.detection_loss import generalized_box_iou

# DIOR类别列表
DIOR_CLASSES = [
    "airplane", "airport", "baseball field", "basketball court",
    "bridge", "chimney", "dam", "expressway service area",
    "expressway toll station", "golf course", "ground track field",
    "harbor", "overpass", "ship", "stadium", "storage tank",
    "tennis court", "train station", "vehicle", "wind mill"
]


def evaluate_localization_accuracy(model, dataloader, device, num_samples=None):
    """
    评估定位准确率
    
    对于每个GT框：
    1. 使用GT类别对应的预测框
    2. 找到IoU最大的位置
    3. 计算IoU
    4. 统计定位准确率
    """
    model.eval()
    
    all_ious = []
    per_class_ious = defaultdict(list)
    iou_thresholds = [0.3, 0.5, 0.7]
    per_class_acc = {threshold: defaultdict(int) for threshold in iou_thresholds}
    per_class_total = defaultdict(int)
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="评估定位准确率")
        if num_samples:
            iterator = list(iterator)[:num_samples]
        
        for batch_idx, batch in enumerate(iterator):
            images = batch['images'].to(device)
            boxes_list = batch['boxes']
            labels_list = batch['labels']
            
            # 前向传播
            outputs = model(images)
            pred_boxes = outputs['pred_boxes']  # [B, C, H, W, 4]
            
            B = pred_boxes.shape[0]
            H, W = pred_boxes.shape[2], pred_boxes.shape[3]
            
            for b in range(B):
                gt_boxes = boxes_list[b].to(device)  # [N, 4]
                gt_labels = labels_list[b].to(device)  # [N]
                
                if len(gt_boxes) == 0:
                    continue
                
                # 对每个GT框，在对应类别通道上找到最佳匹配
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    gt_label = gt_label.item()
                    
                    if gt_label >= pred_boxes.shape[1]:
                        continue
                    
                    # 获取该类别通道的所有预测框
                    pred_class_boxes = pred_boxes[b, gt_label]  # [H, W, 4]
                    pred_class_boxes_flat = pred_class_boxes.view(H * W, 4)  # [H*W, 4]
                    
                    # 计算IoU
                    gt_box_expanded = gt_box.unsqueeze(0)  # [1, 4]
                    ious = generalized_box_iou(pred_class_boxes_flat, gt_box_expanded)  # [H*W, 1]
                    ious = ious[:, 0]  # [H*W]
                    
                    # 找到最大IoU
                    max_iou = ious.max().item()
                    max_idx = ious.argmax().item()
                    
                    # 记录
                    all_ious.append(max_iou)
                    per_class_ious[gt_label].append(max_iou)
                    per_class_total[gt_label] += 1
                    
                    # 统计不同阈值下的准确率
                    for threshold in iou_thresholds:
                        if max_iou >= threshold:
                            per_class_acc[threshold][gt_label] += 1
    
    # 计算统计信息
    if len(all_ious) == 0:
        print("⚠️  没有匹配的GT框")
        return {}
    
    results = {
        'mean_iou': np.mean(all_ious),
        'median_iou': np.median(all_ious),
        'std_iou': np.std(all_ious),
        'per_class_mean_iou': {},
        'per_class_accuracy': {threshold: {} for threshold in iou_thresholds},
        'overall_accuracy': {threshold: 0.0 for threshold in iou_thresholds}
    }
    
    # 每个类别的平均IoU
    for cls_idx, ious in per_class_ious.items():
        results['per_class_mean_iou'][cls_idx] = np.mean(ious)
    
    # 每个类别的定位准确率
    for threshold in iou_thresholds:
        correct = 0
        total = 0
        for cls_idx in per_class_total.keys():
            acc = per_class_acc[threshold][cls_idx] / per_class_total[cls_idx] if per_class_total[cls_idx] > 0 else 0.0
            results['per_class_accuracy'][threshold][cls_idx] = acc
            correct += per_class_acc[threshold][cls_idx]
            total += per_class_total[cls_idx]
        results['overall_accuracy'][threshold] = correct / total if total > 0 else 0.0
    
    return results


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*80)
    print("GT类别定位评估结果")
    print("="*80)
    
    print(f"\n总体定位性能:")
    print(f"  平均IoU: {results['mean_iou']:.4f}")
    print(f"  中位数IoU: {results['median_iou']:.4f}")
    print(f"  IoU标准差: {results['std_iou']:.4f}")
    
    print(f"\n定位准确率（不同IoU阈值）:")
    for threshold in [0.3, 0.5, 0.7]:
        acc = results['overall_accuracy'][threshold]
        print(f"  IoU > {threshold}: {acc*100:.2f}%")
    
    # 每个类别的性能（只显示seen类别）
    seen_class_indices = {0, 1, 4, 9, 11, 13, 14, 15, 18, 19}
    
    print(f"\n每个类别的定位性能（Seen类别）:")
    print(f"{'类别':<20} {'平均IoU':<12} {'IoU>0.5':<12} {'IoU>0.7':<12}")
    print("-" * 60)
    
    for cls_idx in sorted(seen_class_indices):
        if cls_idx in results['per_class_mean_iou']:
            cls_name = DIOR_CLASSES[cls_idx]
            mean_iou = results['per_class_mean_iou'][cls_idx]
            acc_05 = results['per_class_accuracy'][0.5].get(cls_idx, 0.0)
            acc_07 = results['per_class_accuracy'][0.7].get(cls_idx, 0.0)
            print(f"{cls_name:<20} {mean_iou:<12.4f} {acc_05*100:<12.2f}% {acc_07*100:<12.2f}%")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='评估GT类别定位性能')
    parser.add_argument('--config', type=str, default='configs/gt_class_localization_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型checkpoint路径（默认使用latest）')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='评估的样本数量（None表示全部）')
    parser.add_argument('--split', type=str, default='trainval', choices=['trainval', 'test'],
                       help='数据集划分')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device_str = config.get('device', 'cuda')
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
    
    model = create_gt_class_localization_detector(
        surgery_clip_checkpoint=str(surgery_checkpoint),
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        device=device
    )
    model.to(device)
    
    # 加载checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_dir = Path(__file__).parent / config.get('checkpoint_dir', 'checkpoints/gt_class_localization')
        checkpoint_path = checkpoint_dir / 'latest_gt_class_localization_model.pth'
    
    if checkpoint_path.exists():
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ 已加载checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"⚠️  Checkpoint不存在: {checkpoint_path}")
        print("   使用随机初始化的检测头进行评估")
    
    # 加载数据
    print("\n加载数据...")
    dataloader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split=args.split,
        batch_size=4,
        num_workers=2,
        image_size=config.get('image_size', 224),
        augment=False,
        train_only_seen=config.get('train_only_seen', True)
    )
    print(f"✅ 数据集加载成功，共 {len(dataloader.dataset)} 张图像")
    
    # 评估
    print("\n开始评估...")
    results = evaluate_localization_accuracy(
        model, dataloader, device, num_samples=args.num_samples
    )
    
    # 打印结果
    if results:
        print_results(results)
        
        # 保存结果
        import json
        output_dir = Path(__file__).parent / config.get('output_dir', 'outputs/gt_class_localization')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_summary = {
            'mean_iou': float(results['mean_iou']),
            'median_iou': float(results['median_iou']),
            'std_iou': float(results['std_iou']),
            'overall_accuracy': {str(k): float(v) for k, v in results['overall_accuracy'].items()},
            'per_class_mean_iou': {str(k): float(v) for k, v in results['per_class_mean_iou'].items()},
            'per_class_accuracy': {
                str(threshold): {str(k): float(v) for k, v in acc_dict.items()}
                for threshold, acc_dict in results['per_class_accuracy'].items()
            }
        }
        
        with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 结果已保存到: {output_dir / 'evaluation_results.json'}")
    else:
        print("❌ 评估失败")


if __name__ == '__main__':
    main()



# -*- coding: utf-8 -*-
"""
分析分类准确率
验证怀疑2：分类是否正确？

关键问题：
1. 没有显式的分类步骤，直接假设每个类别通道就是该类别的检测
2. 置信度 ≠ 分类概率，只表示"有物体"，不表示"是哪个类别"
3. 如果CAM质量差，分类就错了

验证方法：
- 在训练集上统计预测类别 vs GT类别的混淆矩阵
- 看分类准确率是多少
- 如果分类准确率<50% → 根本问题在这里！
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from models.improved_direct_detection_detector import create_improved_direct_detection_detector
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

# Seen类别索引
SEEN_CLASS_INDICES = {0, 1, 4, 9, 11, 13, 14, 15, 18, 19}


def match_predictions_to_gt(pred_boxes, pred_confidences, gt_boxes, gt_labels, 
                            iou_threshold=0.5, conf_threshold=0.1):
    """
    将预测框匹配到GT框
    
    Args:
        pred_boxes: [C, H, W, 4] 每个类别每个位置的预测框
        pred_confidences: [C, H, W] 每个类别每个位置的置信度
        gt_boxes: [N, 4] GT框
        gt_labels: [N] GT类别
        iou_threshold: IoU阈值
        conf_threshold: 置信度阈值
    
    Returns:
        matches: List of (pred_class, gt_class, iou) tuples
    """
    matches = []
    
    if len(gt_boxes) == 0:
        return matches
    
    C, H, W = pred_confidences.shape
    
    # 收集所有预测框和置信度
    all_pred_boxes = []
    all_pred_classes = []
    all_pred_confs = []
    
    for c in range(C):
        conf_class = pred_confidences[c]  # [H, W]
        boxes_class = pred_boxes[c]  # [H, W, 4]
        
        # 找到高置信度的位置
        high_conf_mask = conf_class > conf_threshold
        
        if high_conf_mask.sum() == 0:
            continue
        
        # 获取高置信度位置的框
        high_conf_positions = high_conf_mask.nonzero(as_tuple=False)  # [N, 2]
        
        for pos in high_conf_positions:
            i, j = pos[0].item(), pos[1].item()
            conf_value = conf_class[i, j].item()
            box = boxes_class[i, j]  # [4]
            
            all_pred_boxes.append(box)
            all_pred_classes.append(c)
            all_pred_confs.append(conf_value)
    
    if len(all_pred_boxes) == 0:
        return matches
    
    # 转换为tensor
    all_pred_boxes = torch.stack(all_pred_boxes)  # [M, 4]
    all_pred_classes = torch.tensor(all_pred_classes, device=all_pred_boxes.device)
    all_pred_confs = torch.tensor(all_pred_confs, device=all_pred_boxes.device)
    
    # 计算IoU矩阵
    iou_matrix = generalized_box_iou(all_pred_boxes, gt_boxes)  # [M, N]
    
    # 贪心匹配：每个GT框匹配IoU最高的预测框
    matched_pred = set()
    matched_gt = set()
    
    # 按IoU降序排序
    iou_flat = iou_matrix.flatten()
    indices_flat = torch.arange(iou_matrix.numel(), device=iou_matrix.device)
    sorted_indices = indices_flat[iou_flat.argsort(descending=True)]
    
    for idx_flat in sorted_indices:
        pred_idx = idx_flat // iou_matrix.shape[1]
        gt_idx = idx_flat % iou_matrix.shape[1]
        
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        
        iou_value = iou_matrix[pred_idx, gt_idx].item()
        
        if iou_value >= iou_threshold:
            pred_class = all_pred_classes[pred_idx].item()
            gt_class = gt_labels[gt_idx].item()
            
            matches.append((pred_class, gt_class, iou_value))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
    
    return matches


def analyze_classification_accuracy(model, dataloader, device, 
                                   conf_threshold=0.1, iou_threshold=0.5,
                                   num_samples=None):
    """
    分析分类准确率
    
    Returns:
        confusion_matrix: [C, C] 混淆矩阵
        accuracy: 分类准确率
        per_class_accuracy: 每个类别的准确率
    """
    model.eval()
    
    # 统计
    confusion_matrix = np.zeros((20, 20), dtype=np.int32)
    total_matches = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    all_matches = []
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="分析分类准确率")
        if num_samples:
            iterator = list(iterator)[:num_samples]
        
        for batch_idx, batch in enumerate(iterator):
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            gt_boxes_list = batch['boxes']
            gt_labels_list = batch['labels']
            
            # 前向传播
            outputs = model(images, text_queries)
            pred_boxes = outputs['pred_boxes']  # [B, C, H, W, 4]
            confidences = outputs['confidences']  # [B, C, H, W]
            
            B = pred_boxes.shape[0]
            
            for b in range(B):
                gt_boxes = gt_boxes_list[b].to(device)  # [N, 4]
                gt_labels = gt_labels_list[b].to(device)  # [N]
                
                if len(gt_boxes) == 0:
                    continue
                
                # 匹配预测和GT
                matches = match_predictions_to_gt(
                    pred_boxes[b],  # [C, H, W, 4]
                    confidences[b],  # [C, H, W]
                    gt_boxes,
                    gt_labels,
                    iou_threshold=iou_threshold,
                    conf_threshold=conf_threshold
                )
                
                # 更新混淆矩阵
                for pred_class, gt_class, iou in matches:
                    confusion_matrix[gt_class, pred_class] += 1
                    total_matches += 1
                    per_class_total[gt_class] += 1
                    
                    if pred_class == gt_class:
                        per_class_correct[gt_class] += 1
                    
                    all_matches.append({
                        'pred_class': pred_class,
                        'gt_class': gt_class,
                        'iou': iou,
                        'pred_name': DIOR_CLASSES[pred_class],
                        'gt_name': DIOR_CLASSES[gt_class]
                    })
    
    # 计算准确率
    if total_matches > 0:
        accuracy = sum(per_class_correct.values()) / total_matches
    else:
        accuracy = 0.0
    
    # 每个类别的准确率
    per_class_acc = {}
    for cls_idx in range(20):
        if cls_idx in per_class_total:
            per_class_acc[cls_idx] = per_class_correct[cls_idx] / per_class_total[cls_idx]
        else:
            per_class_acc[cls_idx] = 0.0
    
    return {
        'confusion_matrix': confusion_matrix,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'total_matches': total_matches,
        'all_matches': all_matches
    }


def visualize_confusion_matrix(confusion_matrix, output_path, title="分类混淆矩阵"):
    """可视化混淆矩阵"""
    # 只显示seen类别
    seen_indices = sorted(list(SEEN_CLASS_INDICES))
    cm_seen = confusion_matrix[np.ix_(seen_indices, seen_indices)]
    
    # 归一化（按行归一化，显示召回率）
    cm_normalized = cm_seen.astype(float)
    row_sums = cm_normalized.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 避免除零
    cm_normalized = cm_normalized / row_sums[:, np.newaxis]
    
    # 类别名称
    class_names = [DIOR_CLASSES[i] for i in seen_indices]
    
    # 绘制
    plt.figure(figsize=(12, 10))
    
    # 子图1：原始混淆矩阵
    plt.subplot(2, 1, 1)
    sns.heatmap(cm_seen, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '匹配数量'})
    plt.title(f'{title} - 原始数量')
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 子图2：归一化混淆矩阵（召回率）
    plt.subplot(2, 1, 2)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '召回率'})
    plt.title(f'{title} - 归一化（召回率）')
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析分类准确率')
    parser.add_argument('--config', type=str, default='configs/improved_detector_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型checkpoint路径（默认使用latest）')
    parser.add_argument('--conf_threshold', type=float, default=0.1,
                       help='置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU阈值')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='分析的样本数量（None表示全部）')
    parser.add_argument('--output_dir', type=str, default='outputs/classification_analysis',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device_str = config.get('device', 'cuda')
    # 检查CUDA是否可用
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("\n加载模型...")
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
    
    model = create_improved_direct_detection_detector(
        surgery_clip_checkpoint=str(surgery_checkpoint),
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        device=device,
        unfreeze_cam_last_layer=config.get('unfreeze_cam_last_layer', True)
    )
    model.to(device)
    
    # 加载checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_dir = Path(__file__).parent / config.get('checkpoint_dir', 'checkpoints/improved_detector')
        checkpoint_path = checkpoint_dir / 'latest_improved_detector_model.pth'
    
    if checkpoint_path.exists():
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ 已加载checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"⚠️  Checkpoint不存在: {checkpoint_path}")
        print("   使用随机初始化的模型进行分析")
    
    # 加载数据
    print("\n加载数据...")
    train_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='trainval',
        batch_size=4,  # 使用较小的batch size
        num_workers=2,
        image_size=config.get('image_size', 224),
        augment=False,  # 不使用数据增强
        train_only_seen=config.get('train_only_seen', True)
    )
    print(f"✅ 数据集加载成功，共 {len(train_loader.dataset)} 张图像")
    
    # 分析分类准确率
    print("\n" + "="*80)
    print("开始分析分类准确率...")
    print("="*80)
    
    results = analyze_classification_accuracy(
        model, train_loader, device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        num_samples=args.num_samples
    )
    
    # 打印结果
    print("\n" + "="*80)
    print("分类准确率分析结果")
    print("="*80)
    print(f"\n总体分类准确率: {results['accuracy']*100:.2f}%")
    print(f"总匹配数: {results['total_matches']}")
    
    if results['accuracy'] < 0.5:
        print("\n⚠️  ⚠️  ⚠️  警告：分类准确率 < 50%！")
        print("   这可能是根本问题所在！")
        print("   如果分类都错了，框再准也是0 mAP！")
    
    # 每个类别的准确率
    print("\n每个类别的分类准确率（Seen类别）:")
    for cls_idx in sorted(SEEN_CLASS_INDICES):
        cls_name = DIOR_CLASSES[cls_idx]
        acc = results['per_class_accuracy'][cls_idx]
        print(f"  {cls_idx:2d}: {cls_name:30s} {acc*100:6.2f}%")
    
    # 混淆矩阵
    print("\n混淆矩阵（Seen类别，行=真实类别，列=预测类别）:")
    seen_indices = sorted(list(SEEN_CLASS_INDICES))
    cm_seen = results['confusion_matrix'][np.ix_(seen_indices, seen_indices)]
    
    print("\n" + " " * 35 + "预测类别")
    print(" " * 12 + " ".join([f"{i:4d}" for i in seen_indices]))
    for i, cls_idx in enumerate(seen_indices):
        cls_name = DIOR_CLASSES[cls_idx]
        row = cm_seen[i]
        print(f"{cls_name:12s} " + " ".join([f"{int(x):4d}" for x in row]))
    
    # 可视化混淆矩阵
    output_path = output_dir / 'confusion_matrix.png'
    visualize_confusion_matrix(results['confusion_matrix'], output_path)
    
    # 保存详细结果
    import json
    results_summary = {
        'accuracy': float(results['accuracy']),
        'total_matches': int(results['total_matches']),
        'per_class_accuracy': {str(k): float(v) for k, v in results['per_class_accuracy'].items()},
        'conf_threshold': args.conf_threshold,
        'iou_threshold': args.iou_threshold
    }
    
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()


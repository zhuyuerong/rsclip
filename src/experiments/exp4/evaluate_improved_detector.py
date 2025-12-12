#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估改进检测器
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

from models.improved_direct_detection_detector import create_improved_direct_detection_detector
from datasets.dior_detection import get_detection_dataloader
from utils.metrics import DetectionMetrics
from utils.class_split import SEEN_CLASSES, UNSEEN_CLASSES, ALL_CLASSES

def evaluate_model(model, dataloader, device, conf_threshold=0.3, nms_threshold=0.5):
    """评估模型"""
    model.eval()
    metrics = DetectionMetrics(num_classes=20)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            
            # 推理
            detections = model.inference(
                images, text_queries,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )
            
            # 转换为metrics格式
            pred_boxes = []
            pred_labels = []
            pred_scores = []
            
            for img_detections in detections:
                if len(img_detections) > 0:
                    boxes_list = []
                    labels_list = []
                    scores_list = []
                    
                    for det in img_detections:
                        box = det['box']
                        if isinstance(box, (list, tuple)):
                            box = torch.tensor(box, device=device)
                        elif not isinstance(box, torch.Tensor):
                            box = torch.tensor([box['xmin'], box['ymin'], box['xmax'], box['ymax']], device=device)
                        
                        boxes_list.append(box.unsqueeze(0))
                        labels_list.append(det['class'])
                        scores_list.append(det.get('confidence', det.get('score', 0.5)))
                    
                    pred_boxes.append(torch.cat(boxes_list, dim=0).to(device))
                    pred_labels.append(torch.tensor(labels_list, dtype=torch.long, device=device))
                    pred_scores.append(torch.tensor(scores_list, device=device))
                else:
                    pred_boxes.append(torch.zeros((0, 4), device=device))
                    pred_labels.append(torch.zeros((0,), dtype=torch.long, device=device))
                    pred_scores.append(torch.zeros((0,), device=device))
            
            # 更新metrics
            metrics.update(
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                gt_boxes=batch['boxes'],
                gt_labels=batch['labels'],
                image_ids=batch.get('image_ids', None)
            )
    
    # 计算mAP
    map_results = metrics.compute_map()
    
    return map_results

def visualize_samples(model, dataloader, device, num_samples=10, output_dir='outputs/visualizations'):
    """可视化样本"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torchvision import transforms
    
    model.eval()
    sample_count = 0
    
    # 反归一化
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    denorm = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            
            # 获取CAM和检测结果
            outputs = model(images, text_queries)
            detections = model.inference(
                images, text_queries,
                conf_threshold=0.3,
                nms_threshold=0.5
            )
            
            for b in range(len(images)):
                if sample_count >= num_samples:
                    break
                
                img_tensor = images[b].cpu()
                img_denorm = denorm(img_tensor)
                img_denorm = torch.clamp(img_denorm, 0, 1)
                img_np = img_denorm.permute(1, 2, 0).numpy()
                
                img_detections = detections[b]
                fused_cam = outputs['fused_cam'][b].cpu()
                
                # 可视化
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                
                # 原图+检测框
                axes[0].imshow(img_np)
                axes[0].axis('off')
                axes[0].set_title('Detections')
                
                # 绘制GT框（绿色）
                gt_boxes_np = gt_boxes[b].cpu().numpy()
                gt_labels_np = gt_labels[b].cpu().numpy()
                for gt_box, gt_label in zip(gt_boxes_np, gt_labels_np):
                    x1, y1, x2, y2 = gt_box
                    w, h = img_np.shape[1], img_np.shape[0]
                    rect = patches.Rectangle(
                        (x1*w, y1*h), (x2-x1)*w, (y2-y1)*h,
                        linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
                    )
                    axes[0].add_patch(rect)
                    class_name = text_queries[gt_label] if gt_label < len(text_queries) else f"class_{gt_label}"
                    axes[0].text(x1*w, y1*h-5, f'GT: {class_name}', 
                                bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                                fontsize=8, color='white')
                
                # 绘制预测框（红色）
                for det in img_detections:
                    box = det['box']
                    # 处理不同格式的box
                    if isinstance(box, torch.Tensor):
                        # Tensor格式: [xmin, ymin, xmax, ymax]
                        box_np = box.cpu().numpy() if box.is_cuda else box.numpy()
                        if box_np.ndim == 1:
                            x1, y1, x2, y2 = float(box_np[0]), float(box_np[1]), float(box_np[2]), float(box_np[3])
                        else:
                            x1, y1, x2, y2 = float(box_np[0, 0]), float(box_np[0, 1]), float(box_np[0, 2]), float(box_np[0, 3])
                    elif isinstance(box, (list, tuple)):
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    elif isinstance(box, dict):
                        x1, y1, x2, y2 = float(box['xmin']), float(box['ymin']), float(box['xmax']), float(box['ymax'])
                    else:
                        # numpy array
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    
                    w, h = img_np.shape[1], img_np.shape[0]
                    rect = patches.Rectangle(
                        (x1*w, y1*h), (x2-x1)*w, (y2-y1)*h,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    axes[0].add_patch(rect)
                    class_name = det.get('class_name', text_queries[det['class']] if det['class'] < len(text_queries) else f"class_{det['class']}")
                    conf = det.get('confidence', det.get('score', 0.0))
                    axes[0].text(x1*w, y1*h-5, f'{class_name}: {conf:.2f}',
                                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                                fontsize=8, color='white')
                
                # CAM热图
                cam_np = fused_cam.max(dim=0)[0].numpy()  # 取所有类别的最大响应
                im = axes[1].imshow(cam_np, cmap='jet', alpha=0.5)
                axes[1].imshow(img_np, alpha=0.5)
                axes[1].axis('off')
                axes[1].set_title('CAM Heatmap')
                plt.colorbar(im, ax=axes[1])
                
                output_path = output_dir / f'sample_{sample_count:03d}.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                sample_count += 1

def main():
    parser = argparse.ArgumentParser(description='评估改进检测器')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--config', type=str, default='configs/improved_detector_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='评估数据集split')
    parser.add_argument('--conf_threshold', type=float, default=0.1,
                       help='置信度阈值（默认0.1，因为模型置信度较低）')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                       help='NMS阈值')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化样本')
    parser.add_argument('--num_vis_samples', type=int, default=20,
                       help='可视化样本数')
    
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
    
    print("\n创建模型...")
    model = create_improved_direct_detection_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        device=device,
        unfreeze_cam_last_layer=config.get('unfreeze_cam_last_layer', True)
    )
    model.to(device)
    
    # 加载checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent / checkpoint_path
    
    print(f"\n加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 处理SurgeryCLIP动态attention层结构变化
    # 模型在第一次forward时会修改attention层（in_proj -> qkv）
    # 需要过滤掉不匹配的键
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        # 跳过attention层的不匹配键（这些会在forward时自动创建）
        if 'attn.in_proj_weight' in key or 'attn.in_proj_bias' in key:
            # 检查是否有对应的qkv版本
            qkv_key = key.replace('in_proj_weight', 'qkv.weight').replace('in_proj_bias', 'qkv.bias')
            if qkv_key not in state_dict:
                continue  # 跳过，让模型使用默认初始化
        elif 'attn.qkv.weight' in key or 'attn.qkv.bias' in key:
            # 保留qkv版本
            if key in model_state_dict:
                filtered_state_dict[key] = value
        elif key in model_state_dict:
            # 其他键直接匹配
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"⚠️  跳过形状不匹配的键: {key} (模型: {model_state_dict[key].shape}, checkpoint: {value.shape})")
    
    # 加载过滤后的state_dict
    model.load_state_dict(filtered_state_dict, strict=False)
    print("✅ 模型加载成功（已处理动态attention层结构）")
    
    # 加载数据
    print(f"\n加载{args.split}数据集...")
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split=args.split,
        batch_size=8,
        num_workers=4,
        image_size=config.get('image_size', 224),
        augment=False,
        train_only_seen=False  # 评估所有类别
    )
    
    # 获取类别名称
    all_classes = ALL_CLASSES
    
    # 评估
    print("\n开始评估...")
    results = evaluate_model(
        model, val_loader, device,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold
    )
    
    # 打印结果
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    
    # 提取结果
    map_50 = results.get('map_50', results.get('mAP_50', results.get('mAP@0.5', 0.0)))
    map_50_95 = results.get('map_50_95', results.get('mAP_50_95', results.get('mAP@0.5:0.95', 0.0)))
    
    # 处理每类AP
    ap_per_class = {}
    if 'ap_per_class' in results:
        ap_per_class = results['ap_per_class']
    elif 'per_class_AP@0.5' in results:
        # 如果是列表格式，转换为字典
        per_class_aps = results['per_class_AP@0.5']
        if isinstance(per_class_aps, list):
            for i, ap in enumerate(per_class_aps):
                if i < len(all_classes):
                    ap_per_class[all_classes[i]] = float(ap)
        elif isinstance(per_class_aps, dict):
            ap_per_class = per_class_aps
    
    print(f"mAP@0.5: {map_50:.4f}")
    print(f"mAP@0.5:0.95: {map_50_95:.4f}")
    
    # 计算seen和unseen类别的mAP
    seen_aps = []
    unseen_aps = []
    for class_name, ap in ap_per_class.items():
        if class_name in SEEN_CLASSES:
            seen_aps.append(ap)
        elif class_name in UNSEEN_CLASSES:
            unseen_aps.append(ap)
    
    seen_map = np.mean(seen_aps) if seen_aps else 0.0
    unseen_map = np.mean(unseen_aps) if unseen_aps else 0.0
    
    print(f"\nSeen类别 mAP@0.5: {seen_map:.4f}")
    print(f"Unseen类别 mAP@0.5: {unseen_map:.4f}")
    
    # 打印每类AP
    print("\n每类AP@0.5:")
    for class_name in all_classes:
        ap = ap_per_class.get(class_name, 0.0)
        class_type = "Seen" if class_name in SEEN_CLASSES else "Unseen"
        print(f"  {class_name:15s} ({class_type:6s}): {ap:.4f}")
    
    # 可视化
    if args.visualize:
        print(f"\n可视化{args.num_vis_samples}个样本...")
        visualize_samples(
            model, val_loader, device,
            num_samples=args.num_vis_samples,
            output_dir=f'outputs/improved_detector_visualizations'
        )
        print(f"可视化结果已保存到: outputs/improved_detector_visualizations/")
    
    # 保存结果
    output_file = Path(__file__).parent / 'outputs' / 'improved_detector_evaluation.txt'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("改进检测器评估结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"mAP@0.5: {map_50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map_50_95:.4f}\n")
        f.write(f"\nSeen类别 mAP@0.5: {seen_map:.4f}\n")
        f.write(f"Unseen类别 mAP@0.5: {unseen_map:.4f}\n")
        f.write("\n每类AP@0.5:\n")
        for class_name in all_classes:
            ap = ap_per_class.get(class_name, 0.0)
            class_type = "Seen" if class_name in SEEN_CLASSES else "Unseen"
            f.write(f"  {class_name:15s} ({class_type:6s}): {ap:.4f}\n")
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == '__main__':
    main()


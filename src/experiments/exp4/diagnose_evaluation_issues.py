#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断评估问题：训练损失下降但mAP=0
检查：坐标系统、置信度、IoU、类别映射
"""

import torch
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.improved_direct_detection_detector import create_improved_direct_detection_detector
from datasets.dior_detection import get_detection_dataloader
from utils.class_split import ALL_CLASSES

def compute_iou(box1, box2):
    """计算IoU"""
    # box: [x1, y1, x2, y2]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def diagnose_model(model, dataloader, device, conf_threshold=0.1, nms_threshold=0.5, num_samples=10):
    """诊断模型输出"""
    model.eval()
    
    # 统计信息
    stats = {
        'num_detections_before_conf': [],
        'num_detections_after_conf': [],
        'num_detections_after_nms': [],
        'confidences': [],
        'pred_box_coords': {'x1': [], 'y1': [], 'x2': [], 'y2': []},
        'gt_box_coords': {'x1': [], 'y1': [], 'x2': [], 'y2': []},
        'ious': [],
        'class_predictions': [],
        'gt_classes': []
    }
    
    sample_detections = []  # 保存前几个样本的详细信息
    
    print("\n" + "="*80)
    print("🔍 开始诊断...")
    print("="*80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="诊断中")):
            if batch_idx >= num_samples:
                break
                
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            gt_boxes = batch['boxes']  # List of tensors
            gt_labels = batch['labels']  # List of tensors
            
            B = images.shape[0]
            
            # ===== 检查1: 模型原始输出 =====
            outputs = model(images, text_queries)
            boxes_raw = outputs['boxes']  # [B, C, H, W, 4]
            confidences_raw = outputs['confidences']  # [B, C, H, W]
            
            # 统计置信度分布
            conf_values = confidences_raw.cpu().numpy().flatten()
            stats['confidences'].extend(conf_values.tolist())
            
            # ===== 检查2: 置信度过滤前 =====
            num_before_conf = (confidences_raw > conf_threshold).sum().item()
            stats['num_detections_before_conf'].append(num_before_conf)
            
            # ===== 检查3: 使用inference方法 =====
            detections = model.inference(
                images, text_queries,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )
            
            # 统计每个图像的检测数量
            for b in range(B):
                img_detections = detections[b]
                stats['num_detections_after_nms'].append(len(img_detections))
                
                # 收集预测框坐标
                for det in img_detections:
                    box = det['box']
                    if isinstance(box, torch.Tensor):
                        box = box.cpu().numpy()
                    elif isinstance(box, (list, tuple)):
                        box = np.array(box)
                    else:
                        box = np.array([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
                    
                    x1, y1, x2, y2 = box
                    stats['pred_box_coords']['x1'].append(x1)
                    stats['pred_box_coords']['y1'].append(y1)
                    stats['pred_box_coords']['x2'].append(x2)
                    stats['pred_box_coords']['y2'].append(y2)
                    stats['class_predictions'].append(det['class'])
                
                # 收集GT框坐标
                gt_boxes_b = gt_boxes[b]
                gt_labels_b = gt_labels[b]
                
                if isinstance(gt_boxes_b, torch.Tensor):
                    gt_boxes_b = gt_boxes_b.cpu().numpy()
                else:
                    gt_boxes_b = np.array(gt_boxes_b)
                
                for gt_box, gt_label in zip(gt_boxes_b, gt_labels_b):
                    if len(gt_box) >= 4:
                        x1, y1, x2, y2 = gt_box[:4]
                        stats['gt_box_coords']['x1'].append(x1)
                        stats['gt_box_coords']['y1'].append(y1)
                        stats['gt_box_coords']['x2'].append(x2)
                        stats['gt_box_coords']['y2'].append(y2)
                        stats['gt_classes'].append(gt_label.item() if isinstance(gt_label, torch.Tensor) else gt_label)
                
                # 计算IoU
                if len(img_detections) > 0 and len(gt_boxes_b) > 0:
                    for det in img_detections:
                        box_pred = det['box']
                        if isinstance(box_pred, torch.Tensor):
                            box_pred = box_pred.cpu().numpy()
                        elif isinstance(box_pred, (list, tuple)):
                            box_pred = np.array(box_pred)
                        else:
                            box_pred = np.array([box_pred['xmin'], box_pred['ymin'], box_pred['xmax'], box_pred['ymax']])
                        
                        max_iou = 0.0
                        for gt_box in gt_boxes_b:
                            if len(gt_box) >= 4:
                                iou = compute_iou(box_pred, gt_box[:4])
                                max_iou = max(max_iou, iou)
                        stats['ious'].append(max_iou)
                
                # 保存样本信息
                if batch_idx < 5:
                    sample_detections.append({
                        'image_idx': batch_idx * B + b,
                        'detections': img_detections,
                        'gt_boxes': gt_boxes_b,
                        'gt_labels': gt_labels_b,
                        'image': images[b].cpu()
                    })
    
    # ===== 打印诊断结果 =====
    print("\n" + "="*80)
    print("📊 诊断结果")
    print("="*80)
    
    # 检查1: 预测框数量
    print("\n【检查1】预测框数量统计")
    print(f"  置信度过滤前 (>{conf_threshold}): {np.mean(stats['num_detections_before_conf']):.1f} 个/图")
    print(f"  NMS后: {np.mean(stats['num_detections_after_nms']):.1f} 个/图")
    print(f"  总检测数: {sum(stats['num_detections_after_nms'])} 个")
    
    if np.mean(stats['num_detections_after_nms']) == 0:
        print("  ⚠️  警告: NMS后没有检测框！")
    
    # 检查2: 置信度分布
    print("\n【检查2】置信度分布")
    if len(stats['confidences']) > 0:
        conf_arr = np.array(stats['confidences'])
        print(f"  最大值: {conf_arr.max():.4f}")
        print(f"  平均值: {conf_arr.mean():.4f}")
        print(f"  中位数: {np.median(conf_arr):.4f}")
        print(f"  标准差: {conf_arr.std():.4f}")
        print(f"  >0.1的数量: {(conf_arr > 0.1).sum()} ({100*(conf_arr > 0.1).mean():.2f}%)")
        print(f"  >0.01的数量: {(conf_arr > 0.01).sum()} ({100*(conf_arr > 0.01).mean():.2f}%)")
        print(f"  >0.001的数量: {(conf_arr > 0.001).sum()} ({100*(conf_arr > 0.001).mean():.2f}%)")
        
        if conf_arr.max() < conf_threshold:
            print(f"  ⚠️  警告: 最大置信度({conf_arr.max():.4f}) < 阈值({conf_threshold})！")
    else:
        print("  ⚠️  警告: 没有置信度数据！")
    
    # 检查3: 坐标范围
    print("\n【检查3】坐标范围检查")
    if len(stats['pred_box_coords']['x1']) > 0:
        pred_x1 = np.array(stats['pred_box_coords']['x1'])
        pred_y1 = np.array(stats['pred_box_coords']['y1'])
        pred_x2 = np.array(stats['pred_box_coords']['x2'])
        pred_y2 = np.array(stats['pred_box_coords']['y2'])
        
        print("  预测框坐标:")
        print(f"    x1: min={pred_x1.min():.4f}, max={pred_x1.max():.4f}, mean={pred_x1.mean():.4f}")
        print(f"    y1: min={pred_y1.min():.4f}, max={pred_y1.max():.4f}, mean={pred_y1.mean():.4f}")
        print(f"    x2: min={pred_x2.min():.4f}, max={pred_x2.max():.4f}, mean={pred_x2.mean():.4f}")
        print(f"    y2: min={pred_y2.min():.4f}, max={pred_y2.max():.4f}, mean={pred_y2.mean():.4f}")
        
        # 检查是否归一化
        if pred_x2.max() <= 1.0 and pred_y2.max() <= 1.0:
            print("  ✅ 预测框似乎是归一化坐标 [0,1]")
        elif pred_x2.max() > 100:
            print("  ✅ 预测框似乎是像素坐标")
        else:
            print("  ⚠️  预测框坐标范围异常")
    else:
        print("  ⚠️  警告: 没有预测框数据！")
    
    if len(stats['gt_box_coords']['x1']) > 0:
        gt_x1 = np.array(stats['gt_box_coords']['x1'])
        gt_y1 = np.array(stats['gt_box_coords']['y1'])
        gt_x2 = np.array(stats['gt_box_coords']['x2'])
        gt_y2 = np.array(stats['gt_box_coords']['y2'])
        
        print("  GT框坐标:")
        print(f"    x1: min={gt_x1.min():.4f}, max={gt_x1.max():.4f}, mean={gt_x1.mean():.4f}")
        print(f"    y1: min={gt_y1.min():.4f}, max={gt_y1.max():.4f}, mean={gt_y1.mean():.4f}")
        print(f"    x2: min={gt_x2.min():.4f}, max={gt_x2.max():.4f}, mean={gt_x2.mean():.4f}")
        print(f"    y2: min={gt_y2.min():.4f}, max={gt_y2.max():.4f}, mean={gt_y2.mean():.4f}")
        
        # 检查是否归一化
        if gt_x2.max() <= 1.0 and gt_y2.max() <= 1.0:
            print("  ✅ GT框似乎是归一化坐标 [0,1]")
        elif gt_x2.max() > 100:
            print("  ✅ GT框似乎是像素坐标")
        else:
            print("  ⚠️  GT框坐标范围异常")
        
        # 检查坐标系统是否匹配
        if len(stats['pred_box_coords']['x1']) > 0:
            pred_max = max(pred_x2.max(), pred_y2.max())
            gt_max = max(gt_x2.max(), gt_y2.max())
            if abs(pred_max - gt_max) > 10:
                print(f"  ⚠️  警告: 坐标系统可能不匹配！")
                print(f"    预测框最大坐标: {pred_max:.2f}")
                print(f"    GT框最大坐标: {gt_max:.2f}")
    else:
        print("  ⚠️  警告: 没有GT框数据！")
    
    # 检查4: IoU分布
    print("\n【检查4】IoU分布")
    if len(stats['ious']) > 0:
        ious_arr = np.array(stats['ious'])
        print(f"  最大IoU: {ious_arr.max():.4f}")
        print(f"  平均IoU: {ious_arr.mean():.4f}")
        print(f"  中位数IoU: {np.median(ious_arr):.4f}")
        print(f"  IoU>0.5的数量: {(ious_arr > 0.5).sum()} ({100*(ious_arr > 0.5).mean():.2f}%)")
        print(f"  IoU>0.3的数量: {(ious_arr > 0.3).sum()} ({100*(ious_arr > 0.3).mean():.2f}%)")
        print(f"  IoU>0.1的数量: {(ious_arr > 0.1).sum()} ({100*(ious_arr > 0.1).mean():.2f}%)")
        
        if ious_arr.max() < 0.5:
            print(f"  ⚠️  警告: 最大IoU({ious_arr.max():.4f}) < 0.5，无法达到mAP@0.5！")
    else:
        print("  ⚠️  警告: 没有IoU数据（可能没有检测框或GT框）！")
    
    # 检查5: 类别分布
    print("\n【检查5】类别预测分布")
    if len(stats['class_predictions']) > 0:
        class_pred_arr = np.array(stats['class_predictions'])
        unique, counts = np.unique(class_pred_arr, return_counts=True)
        print(f"  预测的类别索引: {unique.tolist()}")
        print(f"  各类别数量: {counts.tolist()}")
        print(f"  总预测数: {len(class_pred_arr)}")
    else:
        print("  ⚠️  警告: 没有类别预测数据！")
    
    if len(stats['gt_classes']) > 0:
        gt_class_arr = np.array(stats['gt_classes'])
        unique, counts = np.unique(gt_class_arr, return_counts=True)
        print(f"  GT类别索引: {unique.tolist()}")
        print(f"  各类别数量: {counts.tolist()}")
        print(f"  总GT数: {len(gt_class_arr)}")
    
    # ===== 可视化样本 =====
    print("\n【检查6】可视化样本（前5个）")
    visualize_samples(sample_detections, output_dir='outputs/diagnosis_visualizations')
    
    print("\n" + "="*80)
    print("✅ 诊断完成！")
    print("="*80)
    
    return stats

def visualize_samples(sample_detections, output_dir='outputs/diagnosis_visualizations'):
    """可视化样本"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    denormalize = transforms.Normalize(
        mean=-mean / std,
        std=1 / std
    )
    
    for sample in sample_detections:
        img_idx = sample['image_idx']
        image = sample['image']
        detections = sample['detections']
        gt_boxes = sample['gt_boxes']
        gt_labels = sample['gt_labels']
        
        # 反归一化图像
        image_denorm = denormalize(image)
        image_denorm = torch.clamp(image_denorm, 0, 1)
        image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image_np)
        ax.set_title(f'Sample {img_idx}\nPred: {len(detections)} boxes, GT: {len(gt_boxes)} boxes', fontsize=12)
        
        # 绘制GT框（红色）
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            if len(gt_box) >= 4:
                x1, y1, x2, y2 = gt_box[:4]
                w = x2 - x1
                h = y2 - y1
                rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                label_name = ALL_CLASSES[gt_label] if gt_label < len(ALL_CLASSES) else f"class_{gt_label}"
                ax.text(x1, y1-5, f'GT: {label_name}', color='red', fontsize=8, weight='bold')
        
        # 绘制预测框（蓝色，前10个）
        for i, det in enumerate(detections[:10]):
            box = det['box']
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            elif isinstance(box, (list, tuple)):
                box = np.array(box)
            else:
                box = np.array([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
            
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            conf = det.get('confidence', 0.0)
            class_idx = det.get('class', -1)
            class_name = det.get('class_name', f'class_{class_idx}')
            
            rect = plt.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x1, y1+h+5, f'Pred: {class_name} ({conf:.3f})', color='blue', fontsize=8)
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_{img_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  保存: {output_dir / f'sample_{img_idx}.png'}")
    
    print(f"  ✅ 可视化完成，保存在: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--config', type=str, default='configs/improved_detector_config.yaml', help='配置文件')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--conf_threshold', type=float, default=0.1, help='置信度阈值')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='NMS阈值')
    parser.add_argument('--num_samples', type=int, default=10, help='诊断样本数')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get('device', 'cuda'))
    
    # 创建模型
    print("创建模型...")
    # 处理checkpoint路径
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    model = create_improved_direct_detection_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        device=device,
        unfreeze_cam_last_layer=config.get('unfreeze_cam_last_layer', True)
    )
    
    # 加载checkpoint
    print(f"加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 处理动态attention层
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    for key, value in state_dict.items():
        if 'attn.in_proj_weight' in key or 'attn.in_proj_bias' in key:
            qkv_key = key.replace('in_proj_weight', 'qkv.weight').replace('in_proj_bias', 'qkv.bias')
            if qkv_key not in state_dict:
                continue
        elif 'attn.qkv.weight' in key or 'attn.qkv.bias' in key:
            if key in model_state_dict:
                filtered_state_dict[key] = value
        elif key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    print("✅ 模型加载成功")
    
    # 加载数据
    print(f"\n加载{args.split}数据集...")
    from datasets.dior_detection import get_detection_dataloader
    
    dataloader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split=args.split if args.split != 'val' else 'trainval',  # val images are in trainval folder
        batch_size=4,
        num_workers=2,
        image_size=config.get('image_size', 224),
        augment=False,
        train_only_seen=False  # 评估所有类别
    )
    
    # 诊断
    stats = diagnose_model(
        model, dataloader, device,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        num_samples=args.num_samples
    )
    
    print("\n诊断完成！请查看上面的统计信息和可视化结果。")


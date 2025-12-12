#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练收敛诊断脚本
Step 1: 检查学习率是否在衰减
Step 2: 可视化诊断（对比Epoch 150和178的预测）
Step 3: 统计IoU分布
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent.parent))

from models.improved_direct_detection_detector import create_improved_direct_detection_detector
from datasets.dior_detection import get_detection_dataloader
from torchvision.ops import generalized_box_iou


def step1_check_learning_rate(log_file):
    """Step 1: 检查学习率是否在衰减"""
    print("=" * 60)
    print("Step 1: 检查学习率是否在衰减")
    print("=" * 60)
    
    lrs = []
    epochs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch [' in line and 'LR:' in line:
                # 提取epoch和LR
                parts = line.split('|')
                epoch_part = parts[0].strip()
                lr_part = None
                for p in parts:
                    if 'LR:' in p:
                        lr_part = p.strip()
                        break
                
                if epoch_part and lr_part:
                    epoch = int(epoch_part.split('[')[1].split('/')[0])
                    lr_str = lr_part.split('LR:')[1].strip()
                    lr = float(lr_str)
                    epochs.append(epoch)
                    lrs.append(lr)
    
    if len(lrs) == 0:
        print("❌ 未找到学习率信息")
        return False
    
    print(f"\n找到 {len(lrs)} 个epoch的学习率记录")
    print(f"\n前10个epoch的学习率:")
    for i in range(min(10, len(lrs))):
        print(f"  Epoch {epochs[i]}: LR = {lrs[i]:.6f}")
    
    print(f"\n后10个epoch的学习率:")
    for i in range(max(0, len(lrs)-10), len(lrs)):
        print(f"  Epoch {epochs[i]}: LR = {lrs[i]:.6f}")
    
    # 检查是否在衰减
    unique_lrs = set(lrs)
    print(f"\n唯一学习率值: {sorted(unique_lrs)}")
    
    if len(unique_lrs) == 1:
        print("\n❌ 学习率调度器失效！所有epoch的学习率都是相同的")
        print(f"   当前学习率: {lrs[0]:.6f}")
        return False
    else:
        # 检查趋势
        first_half_avg = np.mean(lrs[:len(lrs)//2])
        second_half_avg = np.mean(lrs[len(lrs)//2:])
        
        if second_half_avg < first_half_avg:
            print(f"\n✅ 学习率在衰减")
            print(f"   前半段平均: {first_half_avg:.6f}")
            print(f"   后半段平均: {second_half_avg:.6f}")
            print(f"   衰减幅度: {(1 - second_half_avg/first_half_avg)*100:.2f}%")
            return True
        else:
            print(f"\n⚠️  学习率没有明显衰减")
            print(f"   前半段平均: {first_half_avg:.6f}")
            print(f"   后半段平均: {second_half_avg:.6f}")
            return False


def step2_visualize_comparison(config, epoch150_checkpoint, epoch178_checkpoint, num_samples=5):
    """Step 2: 可视化诊断（对比Epoch 150和178的预测）"""
    print("\n" + "=" * 60)
    print("Step 2: 可视化诊断（对比Epoch 150和178的预测）")
    print("=" * 60)
    
    device = config.get('device', 'cuda')
    
    # 加载数据
    val_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='val',
        batch_size=1,
        num_workers=2,
        image_size=config.get('image_size', 224),
        augment=False,
        train_only_seen=False
    )
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
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
    model.to(device)
    
    # 加载Epoch 150的模型
    print(f"\n加载Epoch 150的模型: {epoch150_checkpoint}")
    checkpoint150 = torch.load(epoch150_checkpoint, map_location=device)
    state_dict150 = checkpoint150['model_state_dict']
    
    # 处理动态attention层
    model_state_dict = model.state_dict()
    filtered_state_dict150 = {}
    for key, value in state_dict150.items():
        if 'attn.in_proj_weight' in key or 'attn.in_proj_bias' in key:
            qkv_key = key.replace('in_proj_weight', 'qkv.weight').replace('in_proj_bias', 'qkv.bias')
            if qkv_key not in state_dict150:
                continue
        elif 'attn.qkv.weight' in key or 'attn.qkv.bias' in key:
            if key in model_state_dict:
                filtered_state_dict150[key] = value
        elif key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict150[key] = value
    
    model.load_state_dict(filtered_state_dict150, strict=False)
    model.eval()
    
    # 随机选择样本
    dataset = val_loader.dataset
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    print(f"\n随机选择 {len(sample_indices)} 个样本进行可视化")
    
    outputs_dir = Path(__file__).parent / 'outputs' / 'convergence_diagnosis'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, sample_idx in enumerate(sample_indices):
        sample = dataset[sample_idx]
        image = sample['image'].unsqueeze(0).to(device)
        text_queries = sample['text_queries']
        gt_boxes = sample['boxes']
        gt_labels = sample['labels']
        
        with torch.no_grad():
            outputs = model(image, text_queries)
            pred_boxes = outputs['pred_boxes']  # [1, C, H, W, 4]
            confidences = outputs['confidences']  # [1, C, H, W]
        
        # 提取预测框（简化版，只取最高置信度的位置）
        B, C, H, W = confidences.shape
        best_detections = []
        
        for c in range(C):
            conf_c = confidences[0, c]  # [H, W]
            max_conf = conf_c.max()
            if max_conf > 0.1:  # 置信度阈值
                max_idx = conf_c.argmax()
                i, j = max_idx // W, max_idx % W
                pred_box = pred_boxes[0, c, i, j].cpu().numpy()  # [4]
                best_detections.append({
                    'box': pred_box,
                    'conf': max_conf.item(),
                    'class': c
                })
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：GT框
        ax1 = axes[0]
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        ax1.imshow(img_np)
        ax1.set_title(f'Sample {sample_idx} - GT Boxes (Epoch 150)', fontsize=12)
        
        # 绘制GT框
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = gt_box.numpy()
            x1 *= img_np.shape[1]
            y1 *= img_np.shape[0]
            x2 *= img_np.shape[1]
            y2 *= img_np.shape[0]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x1, y1-5, f'GT: {text_queries[gt_label.item()]}', 
                    color='red', fontsize=10, weight='bold')
        
        # 绘制预测框（Epoch 150）
        for det in best_detections[:5]:  # 只显示前5个
            x1, y1, x2, y2 = det['box']
            x1 *= img_np.shape[1]
            y1 *= img_np.shape[0]
            x2 *= img_np.shape[1]
            y2 *= img_np.shape[0]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor='blue', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x1, y1-5, f'Pred: {det["conf"]:.2f}', 
                    color='blue', fontsize=10)
        
        ax1.axis('off')
        
        # 右图：Epoch 178的预测（需要重新加载模型）
        ax2 = axes[1]
        ax2.imshow(img_np)
        ax2.set_title(f'Sample {sample_idx} - Predictions (Epoch 178)', fontsize=12)
        
        # 绘制GT框
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = gt_box.numpy()
            x1 *= img_np.shape[1]
            y1 *= img_np.shape[0]
            x2 *= img_np.shape[1]
            y2 *= img_np.shape[0]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor='red', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, f'GT: {text_queries[gt_label.item()]}',
                    color='red', fontsize=10, weight='bold')
        
        # 加载Epoch 178的模型并预测
        print(f"  加载Epoch 178的模型进行预测...")
        checkpoint178 = torch.load(epoch178_checkpoint, map_location=device)
        state_dict178 = checkpoint178['model_state_dict']
        
        filtered_state_dict178 = {}
        for key, value in state_dict178.items():
            if 'attn.in_proj_weight' in key or 'attn.in_proj_bias' in key:
                qkv_key = key.replace('in_proj_weight', 'qkv.weight').replace('in_proj_bias', 'qkv.bias')
                if qkv_key not in state_dict178:
                    continue
            elif 'attn.qkv.weight' in key or 'attn.qkv.bias' in key:
                if key in model_state_dict:
                    filtered_state_dict178[key] = value
            elif key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict178[key] = value
        
        model.load_state_dict(filtered_state_dict178, strict=False)
        model.eval()
        
        with torch.no_grad():
            outputs178 = model(image, text_queries)
            pred_boxes178 = outputs178['pred_boxes']
            confidences178 = outputs178['confidences']
        
        # 提取预测框（Epoch 178）
        best_detections178 = []
        for c in range(C):
            conf_c = confidences178[0, c]
            max_conf = conf_c.max()
            if max_conf > 0.1:
                max_idx = conf_c.argmax()
                i, j = max_idx // W, max_idx % W
                pred_box = pred_boxes178[0, c, i, j].cpu().numpy()
                best_detections178.append({
                    'box': pred_box,
                    'conf': max_conf.item(),
                    'class': c
                })
        
        # 绘制预测框（Epoch 178）
        for det in best_detections178[:5]:
            x1, y1, x2, y2 = det['box']
            x1 *= img_np.shape[1]
            y1 *= img_np.shape[0]
            x2 *= img_np.shape[1]
            y2 *= img_np.shape[0]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor='green', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, f'Pred: {det["conf"]:.2f}',
                    color='green', fontsize=10)
        
        ax2.axis('off')
        
        plt.tight_layout()
        save_path = outputs_dir / f'sample_{sample_idx}_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存对比图: {save_path}")
    
    print(f"\n✅ 可视化完成，共生成 {len(sample_indices)} 张对比图")
    print(f"   保存位置: {outputs_dir}")


def step3_stat_iou_distribution(config, checkpoint_path, epoch_name):
    """Step 3: 统计IoU分布"""
    print("\n" + "=" * 60)
    print(f"Step 3: 统计IoU分布 ({epoch_name})")
    print("=" * 60)
    
    device = config.get('device', 'cuda')
    
    # 加载数据
    train_loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='trainval',
        batch_size=8,
        num_workers=2,
        image_size=config.get('image_size', 224),
        augment=False,
        train_only_seen=config.get('train_only_seen', True)
    )
    
    # 创建模型
    surgery_checkpoint = config.get('surgery_clip_checkpoint',
                                   'checkpoints/RemoteCLIP-ViT-B-32.pt')
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
    model.to(device)
    
    # 加载checkpoint
    print(f"\n加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 处理动态attention层
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
    model.eval()
    
    # 统计IoU
    all_ious = []
    ious_above_05 = []
    num_pos_samples = 0
    
    print(f"\n在训练集上统计IoU分布（使用正样本分配逻辑）...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing")):
            if batch_idx >= 50:  # 只处理前50个batch
                break
            
            images = batch['images'].to(device)
            text_queries = batch['text_queries']
            boxes = batch['boxes']
            labels = batch['labels']
            
            outputs = model(images, text_queries)
            pred_boxes = outputs['pred_boxes']  # [B, C, H, W, 4]
            confidences = outputs['confidences']  # [B, C, H, W]
            
            B, C, H, W, _ = pred_boxes.shape
            
            for b in range(B):
                gt_boxes_b = boxes[b].to(device)
                gt_labels_b = labels[b].to(device)
                
                if len(gt_boxes_b) == 0:
                    continue
                
                # 使用与损失函数相同的正样本分配逻辑
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes_b, gt_labels_b)):
                    gt_label = gt_label.item()
                    if gt_label >= C:
                        continue
                    
                    pred_class = pred_boxes[b, gt_label].view(H * W, 4)  # [H*W, 4]
                    gt_box_expanded = gt_box.unsqueeze(0)  # [1, 4]
                    
                    ious_matrix = generalized_box_iou(pred_class, gt_box_expanded)  # [H*W, 1]
                    ious = ious_matrix[:, 0]  # [H*W]
                    
                    # 找到最大IoU
                    max_iou = ious.max().item()
                    max_idx = ious.argmax()
                    max_i, max_j = max_idx // W, max_idx % W
                    
                    # 如果IoU > 阈值，认为是正样本
                    pos_iou_threshold = config.get('pos_iou_threshold', 0.15)
                    if max_iou > pos_iou_threshold:
                        num_pos_samples += 1
                        all_ious.append(max_iou)
                        
                        if max_iou > 0.5:
                            ious_above_05.append(max_iou)
    
    if len(all_ious) == 0:
        print("❌ 没有找到正样本")
        return None
    
    avg_iou = np.mean(all_ious)
    iou_above_05_ratio = len(ious_above_05) / len(all_ious) * 100
    
    print(f"\n统计结果 ({epoch_name}):")
    print(f"  正样本数量: {num_pos_samples}")
    print(f"  平均IoU: {avg_iou:.4f}")
    print(f"  IoU > 0.5的比例: {iou_above_05_ratio:.2f}% ({len(ious_above_05)}/{len(all_ious)})")
    print(f"  IoU分布:")
    print(f"    最小值: {min(all_ious):.4f}")
    print(f"    最大值: {max(all_ious):.4f}")
    print(f"    中位数: {np.median(all_ious):.4f}")
    print(f"    25%分位数: {np.percentile(all_ious, 25):.4f}")
    print(f"    75%分位数: {np.percentile(all_ious, 75):.4f}")
    
    return {
        'avg_iou': avg_iou,
        'iou_above_05_ratio': iou_above_05_ratio,
        'num_pos_samples': num_pos_samples,
        'all_ious': all_ious
    }


def main():
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'improved_detector_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_dir = Path(__file__).parent / config.get('checkpoint_dir', 'checkpoints/improved_detector')
    
    # 查找checkpoints
    # Epoch 150的checkpoint（从之前的训练）
    epoch150_checkpoint = checkpoint_dir / 'training_improved_detector_20251210_140315.log'
    # 需要找到实际的checkpoint文件，但可能不存在，所以我们需要从日志中找到epoch 150
    
    # 最新的checkpoint（应该是Epoch 178或更高）
    latest_checkpoint = checkpoint_dir / 'latest_improved_detector_model.pth'
    
    # 日志文件
    log_file = checkpoint_dir / 'training_improved_detector_20251211_111605.log'
    if not log_file.exists():
        log_file = checkpoint_dir / 'training_fixed_20251211_111554.log'
    
    print("=" * 60)
    print("训练收敛诊断")
    print("=" * 60)
    
    # Step 1: 检查学习率
    if log_file.exists():
        lr_ok = step1_check_learning_rate(log_file)
        if not lr_ok:
            print("\n⚠️  警告：学习率调度器可能失效，建议检查scheduler配置")
    else:
        print(f"❌ 未找到日志文件: {log_file}")
    
    # Step 2: 可视化诊断
    # 注意：需要找到Epoch 150和178的checkpoint
    # 由于checkpoint可能只保存latest和best，我们需要从日志中推断
    print("\n⚠️  Step 2需要Epoch 150和178的具体checkpoint文件")
    print("   由于checkpoint可能只保存latest，建议手动指定checkpoint路径")
    
    # Step 3: 统计IoU分布
    if latest_checkpoint.exists():
        print("\n使用latest checkpoint进行IoU统计...")
        stats = step3_stat_iou_distribution(config, latest_checkpoint, "Latest (Epoch ~202)")
    else:
        print(f"\n❌ 未找到checkpoint文件: {latest_checkpoint}")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == '__main__':
    main()


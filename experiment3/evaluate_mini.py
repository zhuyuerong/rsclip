#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment3 在 mini_dataset 上的简化评估

由于没有训练好的模型，这里测试模型架构和推理流程
"""

import torch
import time
import sys
from pathlib import Path
import json

sys.path.append('..')

from config.default_config import DefaultConfig
from models.ova_detr import OVADETR
from utils.data_loader import DIOR_CLASSES, DiorDataset, collate_fn
from utils.transforms import get_transforms
from torch.utils.data import DataLoader


def evaluate_architecture():
    """评估模型架构"""
    
    print("=" * 70)
    print("Experiment3: OVA-DETR 架构评估")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 配置
    config = DefaultConfig()
    config.batch_size = 2
    
    print("\n📋 模型配置:")
    print(f"  RemoteCLIP: {config.remoteclip_model}")
    print(f"  查询数量: {config.num_queries}")
    print(f"  解码器层数: {config.num_decoder_layers}")
    print(f"  模型维度: {config.d_model}")
    print(f"  冻结backbone: {config.freeze_remoteclip}")
    
    # 创建模型
    print("\n创建模型...")
    model = OVADETR(config).to(device)
    model.eval()
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n📊 模型参数:")
    print(f"  总参数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  冻结: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
    
    # 提取文本特征
    print("\n提取文本特征...")
    with torch.no_grad():
        text_features = model.backbone.forward_text(DIOR_CLASSES).to(device)
    print(f"  文本特征: {text_features.shape}")
    print(f"  类别数: {len(DIOR_CLASSES)}")
    
    # 加载数据
    print("\n加载 mini_dataset...")
    transforms = get_transforms(mode='val', image_size=config.image_size)
    
    dataset = DiorDataset(
        root_dir='datasets/mini_dataset',
        split='train',
        transforms=transforms
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"  数据集大小: {len(dataset)}")
    print(f"  批次数: {len(data_loader)}")
    
    # 测试推理
    print("\n🔬 测试推理流程...")
    
    total_time = 0
    num_images = 0
    num_detections = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            batch_start = time.time()
            
            images = images.to(device)
            
            # 前向传播
            outputs = model(images, text_features)
            
            # 统计
            pred_logits = outputs['pred_logits'][-1]
            scores = pred_logits.sigmoid().max(dim=-1)[0]
            
            for i in range(scores.shape[0]):
                detections = (scores[i] > 0.3).sum().item()
                num_detections += detections
                num_images += 1
            
            batch_time = time.time() - batch_start
            total_time += batch_time
            
            if batch_idx == 0:
                print(f"\n  批次 0:")
                print(f"    图像shape: {images.shape}")
                print(f"    pred_logits: {outputs['pred_logits'].shape}")
                print(f"    pred_boxes: {outputs['pred_boxes'].shape}")
                print(f"    批次用时: {batch_time:.3f}秒")
            
            if batch_idx >= 5:  # 只测试前几个批次
                break
    
    inference_time = time.time() - start_time
    
    # GPU内存
    if torch.cuda.is_available():
        gpu_memory = {
            'allocated_MB': torch.cuda.memory_allocated() / 1024**2,
            'reserved_MB': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated_MB': torch.cuda.max_memory_allocated() / 1024**2
        }
    else:
        gpu_memory = {}
    
    # 结果
    print("\n" + "=" * 70)
    print("推理性能测试")
    print("=" * 70)
    print(f"  测试图片数: {num_images}")
    print(f"  总用时: {inference_time:.2f}秒")
    print(f"  FPS: {num_images/inference_time:.2f}")
    print(f"  平均检测数/图: {num_detections/num_images:.1f}")
    
    if gpu_memory:
        print(f"\n💾 GPU内存:")
        print(f"  已分配: {gpu_memory['allocated_MB']:.1f} MB")
        print(f"  已保留: {gpu_memory['reserved_MB']:.1f} MB")
        print(f"  峰值: {gpu_memory['max_allocated_MB']:.1f} MB")
    
    # 保存结果
    results = {
        'experiment': 'Experiment3',
        'model': 'OVA-DETR with RemoteCLIP',
        'architecture': {
            'backbone': config.remoteclip_model,
            'num_queries': config.num_queries,
            'num_decoder_layers': config.num_decoder_layers,
            'd_model': config.d_model,
            'freeze_backbone': config.freeze_remoteclip
        },
        'parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'total_M': total_params / 1e6,
            'trainable_M': trainable_params / 1e6
        },
        'performance': {
            'num_images_tested': num_images,
            'inference_time': inference_time,
            'fps': num_images / inference_time,
            'avg_detections_per_image': num_detections / num_images
        },
        'gpu_memory': gpu_memory,
        'note': '未训练模型，仅测试架构和推理速度'
    }
    
    output_file = Path('experiment3/results_mini_dataset.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果保存到: {output_file}")
    
    return results


if __name__ == '__main__':
    results = evaluate_architecture()
    
    print("\n" + "=" * 70)
    print("Experiment3 架构评估完成！")
    print("=" * 70)



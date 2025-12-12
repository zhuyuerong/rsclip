#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
显示模型结构
"""

import torch
import sys
import os
from pathlib import Path
import yaml

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.surgery_aaf import create_surgery_aaf_model


def show_model_structure():
    """显示模型结构"""
    print("=" * 80)
    print("SurgeryCLIP + AAF + p2p 模型结构")
    print("=" * 80)
    
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 查找检查点
    project_root = Path(__file__).parent.parent.parent.parent
    checkpoint_path = config['clip_weights_path']
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = project_root / checkpoint_path
        if not checkpoint_path.exists():
            print(f"❌ 检查点不存在: {checkpoint_path}")
            return
        checkpoint_path = str(checkpoint_path)
    
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    print(f"检查点: {checkpoint_path}\n")
    
    # 创建模型
    model, preprocess = create_surgery_aaf_model(
        checkpoint_path=checkpoint_path,
        device=device,
        num_layers=config.get('num_layers', 6)
    )
    
    # 冻结CLIP参数（模拟训练时的设置）
    for param in model.clip.parameters():
        param.requires_grad = False
    
    # 只训练AAF参数
    for param in model.aaf.parameters():
        param.requires_grad = True
    
    print("\n" + "=" * 80)
    print("1. 整体模型结构")
    print("=" * 80)
    print(model)
    
    print("\n" + "=" * 80)
    print("2. 模型组件")
    print("=" * 80)
    print(f"CLIP模型类型: {type(model.clip)}")
    print(f"AAF层类型: {type(model.aaf)}")
    print(f"CAM生成器类型: {type(model.cam_generator)}")
    
    print("\n" + "=" * 80)
    print("3. 参数统计")
    print("=" * 80)
    
    # 总参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    # CLIP参数
    clip_params = sum(p.numel() for p in model.clip.parameters())
    print(f"\nCLIP参数: {clip_params:,} (全部冻结)")
    
    # AAF参数
    aaf_params = sum(p.numel() for p in model.aaf.parameters())
    print(f"AAF参数: {aaf_params:,} (全部可训练)")
    
    # CAM生成器参数
    cam_params = sum(p.numel() for p in model.cam_generator.parameters())
    print(f"CAM生成器参数: {cam_params:,} (无参数，仅计算)")
    
    print("\n" + "=" * 80)
    print("4. AAF层详细结构")
    print("=" * 80)
    print(model.aaf)
    
    print("\n" + "=" * 80)
    print("5. AAF层参数详情")
    print("=" * 80)
    for name, param in model.aaf.named_parameters():
        print(f"{name}:")
        print(f"  形状: {param.shape}")
        print(f"  可训练: {param.requires_grad}")
        print(f"  参数数量: {param.numel()}")
        print()
    
    print("=" * 80)
    print("6. 测试前向传播")
    print("=" * 80)
    
    # 创建测试输入
    batch_size = 1
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    text_queries = ["airplane", "ship", "car"]
    
    print(f"输入图像形状: {images.shape}")
    print(f"文本查询: {text_queries}")
    
    model.eval()
    with torch.no_grad():
        cam, aux = model(images, text_queries)
    
    print(f"\n输出CAM形状: {cam.shape}")
    print(f"辅助输出:")
    for key, value in aux.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                print(f"    First item shape: {value[0].shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    print("\n" + "=" * 80)
    print("模型结构显示完成！")
    print("=" * 80)


if __name__ == '__main__':
    show_model_structure()


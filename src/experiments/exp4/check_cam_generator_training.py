#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查CAM生成器是否真的在训练
"""

import torch
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.improved_direct_detection_detector import create_improved_direct_detection_detector

def check_cam_generator_training():
    """检查CAM生成器的训练状态"""
    
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'improved_detector_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理checkpoint路径
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    # 创建模型
    print("创建模型...")
    model = create_improved_direct_detection_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        device='cuda',
        unfreeze_cam_last_layer=config.get('unfreeze_cam_last_layer', True)
    )
    
    print("\n" + "="*80)
    print("检查CAM生成器训练状态")
    print("="*80)
    
    # 检查所有包含"cam"的参数
    cam_params = []
    cam_trainable = []
    cam_frozen = []
    
    for name, param in model.named_parameters():
        if 'cam' in name.lower() or 'generator' in name.lower():
            cam_params.append(name)
            if param.requires_grad:
                cam_trainable.append(name)
            else:
                cam_frozen.append(name)
    
    print(f"\n找到 {len(cam_params)} 个CAM相关参数")
    print(f"  可训练: {len(cam_trainable)}")
    print(f"  冻结: {len(cam_frozen)}")
    
    if len(cam_trainable) > 0:
        print("\n✅ 可训练的CAM参数:")
        for name in cam_trainable[:10]:  # 只显示前10个
            param = dict(model.named_parameters())[name]
            print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        if len(cam_trainable) > 10:
            print(f"  ... 还有 {len(cam_trainable) - 10} 个")
    else:
        print("\n❌ 没有可训练的CAM参数！")
        print("  这是问题！CAM生成器完全冻结了！")
    
    if len(cam_frozen) > 0:
        print("\n⚠️  冻结的CAM参数（前5个）:")
        for name in cam_frozen[:5]:
            param = dict(model.named_parameters())[name]
            print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    # 检查SimpleSurgeryCAM中的CAM生成器
    print("\n" + "="*80)
    print("检查SimpleSurgeryCAM中的CAM生成器")
    print("="*80)
    
    if hasattr(model, 'simple_surgery_cam'):
        if hasattr(model.simple_surgery_cam, 'cam_generator'):
            cam_gen = model.simple_surgery_cam.cam_generator
            print(f"\nCAM生成器类型: {type(cam_gen)}")
            
            cam_gen_params = []
            cam_gen_trainable = []
            for name, param in cam_gen.named_parameters():
                cam_gen_params.append(name)
                if param.requires_grad:
                    cam_gen_trainable.append(name)
            
            print(f"CAM生成器参数总数: {len(cam_gen_params)}")
            print(f"可训练参数: {len(cam_gen_trainable)}")
            
            if len(cam_gen_trainable) > 0:
                print("\n✅ CAM生成器有可训练参数:")
                for name in cam_gen_trainable:
                    param = dict(cam_gen.named_parameters())[name]
                    print(f"  {name}: shape={param.shape}")
            else:
                print("\n❌ CAM生成器完全冻结！")
                print("  需要解冻CAM生成器的最后一层！")
        else:
            print("\n⚠️  找不到cam_generator属性")
    else:
        print("\n⚠️  找不到simple_surgery_cam属性")
    
    # 检查优化器中的CAM参数
    print("\n" + "="*80)
    print("检查优化器配置")
    print("="*80)
    
    # 模拟优化器参数分组
    cam_generator_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'cam_generator' in name:
                cam_generator_params.append(param)
    
    print(f"\n优化器中的CAM生成器参数: {len(cam_generator_params)}")
    
    if len(cam_generator_params) == 0:
        print("❌ 优化器中没有CAM生成器参数！")
        print("  这意味着CAM生成器不会更新！")
    else:
        total_params = sum(p.numel() for p in cam_generator_params)
        print(f"✅ CAM生成器参数总数: {total_params:,}")
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    if len(cam_trainable) == 0 and len(cam_generator_params) == 0:
        print("❌ 严重问题: CAM生成器完全冻结，没有在训练！")
        print("   这是mAP=0的根本原因之一！")
        print("\n修复建议:")
        print("  1. 检查_unfreeze_cam_last_layer()方法是否正确实现")
        print("  2. 确保unfreeze_cam_last_layer=True")
        print("  3. 可能需要解冻更多层，而不仅仅是最后一层")
    elif len(cam_trainable) > 0:
        print("✅ CAM生成器有可训练参数")
        print(f"   可训练参数数: {len(cam_trainable)}")
        if len(cam_trainable) < 10:
            print("⚠️  可训练参数很少，可能需要解冻更多层")
    else:
        print("⚠️  状态不明确，需要进一步检查")

if __name__ == '__main__':
    check_cam_generator_training()



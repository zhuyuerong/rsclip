#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有CAL实验的简化脚本（不依赖命令行参数）
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 直接导入surgeryclip模块，避免触发其他模块
import importlib.util

# 导入surgeryclip的model_wrapper
surgeryclip_dir = project_root / 'src' / 'competitors' / 'clip_methods' / 'surgeryclip'
sys.path.insert(0, str(surgeryclip_dir))

# 先导入依赖模块
base_interface_path = project_root / 'src' / 'competitors' / 'clip_methods' / 'base_interface.py'
spec = importlib.util.spec_from_file_location("base_interface", base_interface_path)
base_interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_interface)

# 导入surgeryclip相关模块
for module_name in ['clip_model', 'clip_surgery_model', 'build_model', 'clip']:
    module_path = surgeryclip_dir / f'{module_name}.py'
    if module_path.exists():
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[f'src.competitors.clip_methods.surgeryclip.{module_name}'] = module

# 导入model_wrapper
model_wrapper_path = surgeryclip_dir / 'model_wrapper.py'
spec = importlib.util.spec_from_file_location("model_wrapper", model_wrapper_path)
model_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_wrapper)
SurgeryCLIPWrapper = model_wrapper.SurgeryCLIPWrapper

# 导入实验配置
config_path = Path(__file__).parent / 'experiment_configs.py'
spec = importlib.util.spec_from_file_location("experiment_configs", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
ALL_CAL_CONFIGS = config_module.ALL_CAL_CONFIGS
from PIL import Image
import time
from datetime import datetime
import os


def find_test_images():
    """查找测试图像"""
    possible_paths = [
        'datasets/mini-DIOR/test/images',
        'datasets/mini-DIOR/images',
        'datasets/mini-DIOR',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            images = list(Path(path).glob('*.jpg')) + list(Path(path).glob('*.png'))
            if images:
                return images[:3]  # 返回前3张
    
    return []


def main():
    print("=" * 80)
    print("🚀 CAL实验批量运行")
    print("=" * 80)
    
    # 查找测试图像
    print("\n🔍 查找测试图像...")
    test_images = find_test_images()
    
    if not test_images:
        print("❌ 未找到测试图像")
        print("   请确保数据集路径正确，或手动指定图像路径")
        return
    
    print(f"✅ 找到 {len(test_images)} 张测试图像:")
    for img in test_images:
        print(f"   - {img}")
    
    # 使用默认类别（可以根据实际情况修改）
    class_names = ['vehicle', 'airplane', 'ship'][:len(test_images)]
    
    # 检查checkpoint
    checkpoint_path = 'checkpoints/ViT-B-32.pt'
    if not os.path.exists(checkpoint_path):
        print(f"\n⚠️  模型权重不存在: {checkpoint_path}")
        print("   请确保权重文件存在")
        return
    
    print(f"\n✅ 使用模型权重: {checkpoint_path}")
    
    # 检查设备
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ 使用设备: {device}")
    
    # 导入运行函数（内联实现，避免循环导入）
    from .run_all_experiments import run_all_experiments
    
    # 运行所有实验
    print(f"\n{'='*80}")
    print("🚀 开始运行所有实验")
    print(f"{'='*80}\n")
    
    results = run_all_experiments(
        image_paths=[str(img) for img in test_images],
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        device=device,
        output_dir='outputs/exp3_cal',
        skip_existing=True
    )
    
    print("\n" + "=" * 80)
    print("✅ 所有实验完成！")
    print("=" * 80)
    print(f"\n📊 结果总结:")
    print(f"   ✅ 完成: {results['completed']}")
    print(f"   ⏭️  跳过: {results['skipped']}")
    print(f"   ❌ 失败: {results['failed']}")
    print(f"\n📁 结果保存在: outputs/exp3_cal/")


if __name__ == '__main__':
    main()


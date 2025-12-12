#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接运行CAL实验（避免导入问题）
"""
import sys
import os
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 直接导入surgeryclip相关模块，避免触发其他模块
import importlib.util

# 导入surgeryclip模块
surgeryclip_path = project_root / 'src' / 'competitors' / 'clip_methods' / 'surgeryclip'
sys.path.insert(0, str(surgeryclip_path))

# 导入model_wrapper（需要先导入依赖）
spec = importlib.util.spec_from_file_location(
    "model_wrapper", 
    surgeryclip_path / 'model_wrapper.py'
)
model_wrapper_module = importlib.util.module_from_spec(spec)

# 临时设置sys.modules以避免相对导入问题
import sys as sys_module
sys_module.modules['src.competitors.clip_methods.surgeryclip'] = type(sys_module)('surgeryclip')
sys_module.modules['src.competitors.clip_methods.surgeryclip.clip_model'] = importlib.import_module('clip_model', package=str(surgeryclip_path))
sys_module.modules['src.competitors.clip_methods.surgeryclip.clip_surgery_model'] = importlib.import_module('clip_surgery_model', package=str(surgeryclip_path))
sys_module.modules['src.competitors.clip_methods.surgeryclip.build_model'] = importlib.import_module('build_model', package=str(surgeryclip_path))
sys_module.modules['src.competitors.clip_methods.surgeryclip.clip'] = importlib.import_module('clip', package=str(surgeryclip_path))
sys_module.modules['src.competitors.clip_methods.base_interface'] = importlib.import_module('base_interface', package=str(project_root / 'src' / 'competitors' / 'clip_methods'))

spec.loader.exec_module(model_wrapper_module)
SurgeryCLIPWrapper = model_wrapper_module.SurgeryCLIPWrapper

# 导入实验配置
exp_config_path = Path(__file__).parent / 'experiment_configs.py'
spec = importlib.util.spec_from_file_location("experiment_configs", exp_config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
ALL_CAL_CONFIGS = config_module.ALL_CAL_CONFIGS

from PIL import Image
import time
from datetime import datetime
import torch


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
                return images[:3]
    
    return []


def run_single_experiment(config_name, cal_config, image_path, class_name, checkpoint_path, device, output_dir):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"🧪 {config_name}: {Path(image_path).name} ({class_name})")
    print(f"{'='*60}")
    
    try:
        # 创建模型
        model = SurgeryCLIPWrapper(
            model_name='surgeryclip',
            checkpoint_path=checkpoint_path,
            device=device,
            use_surgery_single='empty',
            use_surgery_multi=True,
            cal_config=cal_config
        )
        model.load_model()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 生成热图
        heatmap = model.generate_heatmap(image, [class_name])
        
        # 保存结果
        output_path = Path(output_dir) / config_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        import matplotlib.pyplot as plt
        image_name = Path(image_path).stem
        save_path = output_path / f"{image_name}_{class_name}_cal.png"
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image)
        axes[0].set_title(f'Original: {class_name}')
        axes[0].axis('off')
        
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f'CAL: {config_name}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 保存: {save_path}")
        return True, str(save_path)
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def main():
    print("=" * 80)
    print("🚀 CAL实验批量运行")
    print("=" * 80)
    
    # 查找测试图像
    print("\n🔍 查找测试图像...")
    test_images = find_test_images()
    
    if not test_images:
        print("❌ 未找到测试图像")
        return
    
    print(f"✅ 找到 {len(test_images)} 张图像")
    class_names = ['vehicle', 'airplane', 'ship'][:len(test_images)]
    
    # 检查checkpoint
    checkpoint_path = 'checkpoints/ViT-B-32.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型权重不存在: {checkpoint_path}")
        return
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ 设备: {device}")
    
    output_dir = 'outputs/exp3_cal'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 运行所有实验
    print(f"\n{'='*80}")
    print(f"🚀 开始运行 {len(ALL_CAL_CONFIGS)} 个实验配置")
    print(f"{'='*80}\n")
    
    results = {'completed': 0, 'failed': 0}
    start_time = time.time()
    
    for config_idx, (config_name, cal_config) in enumerate(ALL_CAL_CONFIGS.items(), 1):
        print(f"\n📦 [{config_idx}/{len(ALL_CAL_CONFIGS)}] {config_name}")
        
        for img_idx, (image_path, class_name) in enumerate(zip(test_images, class_names), 1):
            success, result = run_single_experiment(
                config_name, cal_config, str(image_path), class_name,
                checkpoint_path, device, output_dir
            )
            
            if success:
                results['completed'] += 1
            else:
                results['failed'] += 1
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("📊 总结")
    print(f"{'='*80}")
    print(f"✅ 完成: {results['completed']}")
    print(f"❌ 失败: {results['failed']}")
    print(f"⏰ 耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()







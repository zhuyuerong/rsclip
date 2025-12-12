#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接运行CAL实验 - 避免导入问题
"""
import sys
import os
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# 使用Python模块方式导入
print("📦 导入模块...")
try:
    # 直接使用模块导入方式
    import importlib
    import importlib.util
    
    # 导入实验配置
    config_path = Path(__file__).parent / 'experiment_configs.py'
    spec = importlib.util.spec_from_file_location("exp_configs", config_path)
    exp_configs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_configs)
    ALL_CAL_CONFIGS = exp_configs.ALL_CAL_CONFIGS
    print(f"✅ 加载了 {len(ALL_CAL_CONFIGS)} 个实验配置")
    
    # 直接加载model_wrapper，绕过__init__.py（避免触发diffclip）
    import types
    
    # 创建必要的模块占位符
    for mod_name in ['src', 'src.competitors', 'src.competitors.clip_methods', 
                     'src.competitors.clip_methods.surgeryclip']:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    
    # 先导入依赖
    base_interface_path = project_root / 'src' / 'competitors' / 'clip_methods' / 'base_interface.py'
    spec = importlib.util.spec_from_file_location('base_interface', base_interface_path)
    base_interface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_interface)
    sys.modules['src.competitors.clip_methods.base_interface'] = base_interface
    
    # 导入surgeryclip的依赖模块（按依赖顺序）
    surgeryclip_dir = project_root / 'src' / 'competitors' / 'clip_methods' / 'surgeryclip'
    surgeryclip_pkg = sys.modules['src.competitors.clip_methods.surgeryclip']
    
    # 按依赖顺序导入（需要先导入clip_model和clip_surgery_model，然后是build_model和clip）
    module_files = ['clip_model.py', 'clip_surgery_model.py', 'build_model.py', 'clip.py']
    for module_file in module_files:
        module_path = surgeryclip_dir / module_file
        if module_path.exists():
            module_name = module_file.replace('.py', '')
            full_module_name = f'src.competitors.clip_methods.surgeryclip.{module_name}'
            
            # 设置__package__属性以支持相对导入
            spec = importlib.util.spec_from_file_location(full_module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            module.__package__ = 'src.competitors.clip_methods.surgeryclip'
            module.__name__ = full_module_name
            module.__file__ = str(module_path)
            
            # 执行模块
            spec.loader.exec_module(module)
            
            # 注册到sys.modules和包中
            sys.modules[full_module_name] = module
            setattr(surgeryclip_pkg, module_name, module)
    
    # 导入model_wrapper
    model_wrapper_path = surgeryclip_dir / 'model_wrapper.py'
    spec = importlib.util.spec_from_file_location('model_wrapper', model_wrapper_path)
    model_wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_wrapper)
    SurgeryCLIPWrapper = model_wrapper.SurgeryCLIPWrapper
    print("✅ 成功导入SurgeryCLIPWrapper")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    print("\n💡 建议:")
    print("   1. 确保在项目根目录运行")
    print("   2. 设置PYTHONPATH: export PYTHONPATH=$(pwd):$PYTHONPATH")
    print("   3. 或使用: python -m src.experiments.exp3_cal.run_now")
    import traceback
    traceback.print_exc()
    sys.exit(1)

from PIL import Image
import torch
import time
from datetime import datetime
import matplotlib.pyplot as plt


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


def main():
    print("=" * 80)
    print("🚀 CAL实验批量运行")
    print("=" * 80)
    
    # 查找测试图像
    print("\n🔍 查找测试图像...")
    test_images = find_test_images()
    
    if not test_images:
        print("❌ 未找到测试图像")
        print("   请确保数据集路径正确")
        return
    
    print(f"✅ 找到 {len(test_images)} 张测试图像:")
    for img in test_images:
        print(f"   - {img}")
    
    class_names = ['vehicle', 'airplane', 'ship'][:len(test_images)]
    
    # 检查checkpoint
    checkpoint_path = 'checkpoints/ViT-B-32.pt'
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ 模型权重不存在: {checkpoint_path}")
        return
    
    print(f"\n✅ 使用模型权重: {checkpoint_path}")
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ 使用设备: {device}")
    
    output_dir = Path('outputs/exp3_cal')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行所有实验
    print(f"\n{'='*80}")
    print(f"🚀 开始运行 {len(ALL_CAL_CONFIGS)} 个实验配置")
    print(f"{'='*80}\n")
    
    results = {'completed': 0, 'failed': 0, 'skipped': 0}
    start_time = time.time()
    
    for config_idx, (config_name, cal_config) in enumerate(ALL_CAL_CONFIGS.items(), 1):
        print(f"\n{'='*60}")
        print(f"📦 [{config_idx}/{len(ALL_CAL_CONFIGS)}] {config_name}")
        print(f"{'='*60}")
        print(f"   实验ID: {cal_config.get_experiment_id()}")
        print(f"   负样本模式: {cal_config.negative_mode}")
        print(f"   加权系数: alpha={cal_config.alpha}")
        print(f"   操作位置: {cal_config.cal_space}")
        
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
            print("   ✅ 模型加载成功")
            
            # 处理每张图像
            for img_idx, (image_path, class_name) in enumerate(zip(test_images, class_names), 1):
                print(f"\n   🖼️  图像 {img_idx}/{len(test_images)}: {image_path.name} ({class_name})")
                
                # 检查是否已存在
                image_name = image_path.stem
                save_path = output_dir / config_name / f"{image_name}_{class_name}_cal.png"
                
                if save_path.exists():
                    print(f"      ⏭️  跳过（已存在）")
                    results['skipped'] += 1
                    continue
                
                try:
                    # 加载图像
                    image = Image.open(image_path).convert('RGB')
                    
                    # 生成热图
                    heatmap = model.generate_heatmap(image, [class_name])
                    
                    # 保存结果
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
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
                    
                    print(f"      ✅ 保存: {save_path}")
                    print(f"         统计: min={heatmap.min():.4f}, max={heatmap.max():.4f}, mean={heatmap.mean():.4f}")
                    
                    results['completed'] += 1
                    
                except Exception as e:
                    print(f"      ❌ 失败: {e}")
                    results['failed'] += 1
            
        except Exception as e:
            print(f"   ❌ 模型创建失败: {e}")
            results['failed'] += len(test_images)
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print("📊 实验总结")
    print(f"{'='*80}")
    print(f"   ✅ 完成: {results['completed']}")
    print(f"   ⏭️  跳过: {results['skipped']}")
    print(f"   ❌ 失败: {results['failed']}")
    print(f"   ⏰ 总耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
    print(f"   ⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"\n📁 结果保存在: {output_dir}/")


if __name__ == '__main__':
    main()


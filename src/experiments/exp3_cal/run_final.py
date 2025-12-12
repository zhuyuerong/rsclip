#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终运行脚本 - 使用标准导入方式
"""
import sys
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print("📦 导入模块...")

# 导入实验配置
import importlib.util
config_path = Path(__file__).parent / 'experiment_configs.py'
spec = importlib.util.spec_from_file_location("exp_configs", config_path)
exp_configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_configs)
ALL_CAL_CONFIGS = exp_configs.ALL_CAL_CONFIGS
print(f"✅ 加载了 {len(ALL_CAL_CONFIGS)} 个实验配置")

# 使用标准导入（已修复__init__.py，不会触发diffclip）
try:
    from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper
    print("✅ 成功导入SurgeryCLIPWrapper")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    print("\n💡 如果遇到timm错误，请安装: pip install timm")
    import traceback
    traceback.print_exc()
    sys.exit(1)

from PIL import Image
import torch
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt错误
import matplotlib.pyplot as plt


def normalize_class_name(class_name):
    """标准化类别名称（处理DIOR数据集中的命名不一致）"""
    # DIOR数据集中的类别名称映射
    class_mapping = {
        'baseballfield': 'baseball field',
        'basketballcourt': 'basketball court',
        'groundtrackfield': 'ground track field',
        'ExpresswayServiceArea': 'Expressway Service Area',
        'Expresswaytollstation': 'Expressway toll station',
        'storagetank': 'storage tank',
        'tenniscourt': 'tennis court',
        'trainstation': 'train station',
        'windmill': 'wind mill',
    }
    return class_mapping.get(class_name.lower(), class_name)


def get_class_from_annotation(image_path):
    """从标注文件获取图片的主要类别"""
    image_id = image_path.stem
    
    # 尝试多个标注路径
    annotation_paths = [
        Path('datasets/DIOR/annotations/horizontal') / f"{image_id}.xml",
        Path('datasets/DIOR/annotations') / f"{image_id}.xml",
        Path('datasets/mini-DIOR/annotations') / f"{image_id}.xml",
    ]
    
    for anno_path in annotation_paths:
        if anno_path.exists():
            try:
                tree = ET.parse(anno_path)
                root = tree.getroot()
                classes = [obj.find('name').text for obj in root.findall('object')]
                if classes:
                    # 返回出现最多的类别，并标准化名称
                    from collections import Counter
                    most_common = Counter(classes).most_common(1)[0][0]
                    return normalize_class_name(most_common)
            except Exception as e:
                print(f"      ⚠️  解析标注文件失败 {anno_path}: {e}")
    
    return None


def find_test_images():
    """查找测试图像并获取对应的类别"""
    possible_paths = [
        'datasets/mini-DIOR/test/images',
        'datasets/mini-DIOR/images',
        'datasets/mini-DIOR',
        'datasets/DIOR/images/test',
        'datasets/DIOR/images/trainval',
    ]
    
    images = []
    for path in possible_paths:
        if os.path.exists(path):
            found = list(Path(path).glob('*.jpg')) + list(Path(path).glob('*.png'))
            if found:
                images = found[:3]
                break
    
    if not images:
        return [], []
    
    # 获取每张图片的类别
    class_names = []
    for img_path in images:
        class_name = get_class_from_annotation(img_path)
        if class_name:
            class_names.append(class_name)
        else:
            # 如果无法从标注获取，使用默认值
            print(f"      ⚠️  无法获取 {img_path.name} 的类别，请手动指定")
            class_names.append(None)
    
    return images, class_names


def main():
    print("=" * 80)
    print("🚀 CAL实验批量运行")
    print("=" * 80)
    
    # 查找测试图像并获取类别
    print("\n🔍 查找测试图像...")
    test_images, class_names = find_test_images()
    
    if not test_images:
        print("❌ 未找到测试图像")
        print("   请确保数据集路径正确")
        return
    
    print(f"✅ 找到 {len(test_images)} 张测试图像:")
    for img, cls in zip(test_images, class_names):
        if cls:
            print(f"   - {img.name} → {cls}")
        else:
            print(f"   - {img.name} → ⚠️  类别未知")
    
    # 检查是否有未获取到类别的图片
    if any(c is None for c in class_names):
        print("\n❌ 部分图片无法获取类别，请检查标注文件或手动指定")
        print("   可以修改代码中的 class_names 列表来手动指定类别")
        return
    
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
                
                # 检查是否已存在（可选：添加 --force 参数强制重新运行）
                image_name = image_path.stem
                # 处理类别名称中的空格，替换为下划线（避免文件名问题）
                safe_class_name = class_name.replace(' ', '_')
                save_path = output_dir / config_name / f"{image_name}_{safe_class_name}_cal.png"
                
                # 如果需要强制重新运行，取消下面的注释
                # if save_path.exists():
                #     print(f"      ⏭️  跳过（已存在）")
                #     results['skipped'] += 1
                #     continue
                
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
    
    # 保存实验记录
    import json
    experiment_records = []
    for config_name, cal_config in ALL_CAL_CONFIGS.items():
        config_dir = output_dir / config_name
        if config_dir.exists():
            png_files = list(config_dir.glob("*.png"))
            for png_file in png_files:
                experiment_records.append({
                    'config_name': config_name,
                    'experiment_id': cal_config.get_experiment_id(),
                    'image_file': png_file.name,
                    'output_path': str(png_file),
                    'config': cal_config.to_dict()
                })
    
    # 保存总结JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_configs': len(ALL_CAL_CONFIGS),
        'total_images': len(test_images),
        'results': {
            'completed': results['completed'],
            'skipped': results['skipped'],
            'failed': results['failed'],
            'total': results['completed'] + results['skipped'] + results['failed']
        },
        'elapsed_time_seconds': elapsed,
        'experiments': experiment_records
    }
    
    summary_file = output_dir / 'experiments_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
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
    print(f"📄 实验记录: {summary_file}")
    print(f"📊 共记录 {len(experiment_records)} 个实验结果")


if __name__ == '__main__':
    main()


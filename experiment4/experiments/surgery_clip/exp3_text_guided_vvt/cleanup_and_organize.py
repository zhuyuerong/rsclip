# -*- coding: utf-8 -*-
"""
目录清理和整理脚本

整理exp3_text_guided_vvt目录，保留核心文件，清理冗余文件
"""

import shutil
from pathlib import Path

def cleanup_directory():
    """清理和整理目录"""
    
    current_dir = Path(__file__).parent
    
    print("=" * 60)
    print("exp3_text_guided_vvt 目录清理和整理")
    print("=" * 60)
    
    # 创建备份目录
    backup_dir = current_dir / 'backup_original_files'
    backup_dir.mkdir(exist_ok=True)
    
    # 要备份的原始文件
    original_files = [
        'text_guided_vvt.py',
        'multi_class_heatmap.py', 
        'debug_gt_boxes.py',
        'comprehensive_comparison.py'
    ]
    
    print("1. 备份原始文件...")
    for file_name in original_files:
        src = current_dir / file_name
        if src.exists():
            dst = backup_dir / file_name
            shutil.copy2(src, dst)
            print(f"  ✓ 备份: {file_name}")
    
    # 创建新的目录结构
    print("\n2. 创建新的目录结构...")
    
    # 创建子目录
    subdirs = [
        'results/multi_class_4mode',
        'results/debug_gt_boxes', 
        'results/comprehensive_comparison',
        'results/legacy_outputs',
        'docs'
    ]
    
    for subdir in subdirs:
        (current_dir / subdir).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ 创建目录: {subdir}")
    
    # 移动现有结果文件
    print("\n3. 整理现有结果文件...")
    
    # 移动multi_class_results
    multi_class_src = current_dir / 'multi_class_results'
    multi_class_dst = current_dir / 'results/multi_class_4mode'
    if multi_class_src.exists():
        if multi_class_dst.exists():
            shutil.rmtree(multi_class_dst)
        shutil.move(str(multi_class_src), str(multi_class_dst))
        print(f"  ✓ 移动: multi_class_results → results/multi_class_4mode")
    
    # 移动comprehensive_comparison_results
    comp_src = current_dir / 'comprehensive_comparison_results'
    comp_dst = current_dir / 'results/comprehensive_comparison'
    if comp_src.exists():
        if comp_dst.exists():
            shutil.rmtree(comp_dst)
        shutil.move(str(comp_src), str(comp_dst))
        print(f"  ✓ 移动: comprehensive_comparison_results → results/comprehensive_comparison")
    
    # 移动其他结果文件到legacy
    legacy_files = [
        'gt_responses.json',
        '*.png'  # 其他PNG文件
    ]
    
    for pattern in legacy_files:
        for file_path in current_dir.glob(pattern):
            if file_path.is_file():
                dst = current_dir / 'results/legacy_outputs' / file_path.name
                shutil.move(str(file_path), str(dst))
                print(f"  ✓ 移动: {file_path.name} → results/legacy_outputs/")
    
    # 移动文档文件
    print("\n4. 整理文档文件...")
    doc_files = [
        'README.txt',
        'MULTI_CLASS_README.txt', 
        'FINAL_SUMMARY.txt',
        'COMPLETE_SUMMARY.txt',
        'IMAGE_LIST.txt',
        '4_mode_comparison_report.md'
    ]
    
    for doc_file in doc_files:
        src = current_dir / doc_file
        if src.exists():
            dst = current_dir / 'docs' / doc_file
            shutil.move(str(src), str(dst))
            print(f"  ✓ 移动: {doc_file} → docs/")
    
    # 创建新的README
    print("\n5. 创建新的README...")
    create_new_readme(current_dir)
    
    print(f"\n{'='*60}")
    print("目录整理完成！")
    print(f"{'='*60}")
    print(f"新的目录结构:")
    print(f"├── unified_heatmap_generator.py  # 统一脚本")
    print(f"├── cleanup_and_organize.py      # 本脚本")
    print(f"├── backup_original_files/       # 原始文件备份")
    print(f"├── results/")
    print(f"│   ├── multi_class_4mode/       # 4模式对比结果")
    print(f"│   ├── debug_gt_boxes/          # GT调试结果")
    print(f"│   ├── comprehensive_comparison/ # 全面对比结果")
    print(f"│   └── legacy_outputs/          # 历史输出")
    print(f"└── docs/                        # 文档")
    print(f"{'='*60}")

def create_new_readme(directory):
    """创建新的README文件"""
    
    readme_content = """# exp3_text_guided_vvt - 统一热图生成器

## 概述

本目录包含文本引导的VV^T热图生成实验，整合了4个核心功能模块。

## 核心文件

### 主要脚本
- `unified_heatmap_generator.py` - **统一热图生成器**（推荐使用）
- `cleanup_and_organize.py` - 目录整理脚本

### 原始文件（已备份）
- `backup_original_files/text_guided_vvt.py` - 文本引导VV^T热图
- `backup_original_files/multi_class_heatmap.py` - 多类别4模式对比
- `backup_original_files/debug_gt_boxes.py` - GT边界框调试
- `backup_original_files/comprehensive_comparison.py` - 全面对比实验

## 功能特性

### 4种模式对比
1. **With Surgery** - 标准RemoteCLIP + Feature Surgery去冗余
2. **Without Surgery** - 标准RemoteCLIP + 余弦相似度
3. **With VV** - CLIPSurgery (VV机制) + 余弦相似度
4. **Complete Surgery** - CLIPSurgery (VV机制) + Feature Surgery

### 支持的分析
- **12层热图**: L1-L12 多层特征分析
- **多类别图像**: 每个类别独立查询和热图生成
- **GT边界框**: 精确坐标缩放和可视化
- **调试工具**: GT框位置验证和坐标信息

## 使用方法

### 运行所有功能
```bash
cd experiment4/experiments/surgery_clip/exp3_text_guided_vvt
PYTHONPATH=../../.. ovadetr_env/bin/python3.9 unified_heatmap_generator.py --mode all --max-samples 5
```

### 运行特定功能
```bash
# 多类别4模式对比
PYTHONPATH=../../.. ovadetr_env/bin/python3.9 unified_heatmap_generator.py --mode multi_class --max-samples 3

# GT边界框调试
PYTHONPATH=../../.. ovadetr_env/bin/python3.9 unified_heatmap_generator.py --mode debug_gt --debug-samples 5

# 全面对比实验
PYTHONPATH=../../.. ovadetr_env/bin/python3.9 unified_heatmap_generator.py --mode comprehensive --max-samples 3
```

### 参数说明
- `--dataset`: 数据集路径 (默认: datasets/mini_dataset)
- `--mode`: 运行模式 (multi_class/debug_gt/comprehensive/all)
- `--max-samples`: 最大样本数 (默认: 5)
- `--layers`: 分析层 (默认: 1-12)
- `--debug-samples`: GT调试样本数 (默认: 3)

## 输出结果

### 目录结构
```
results/
├── multi_class_4mode/          # 4模式对比结果
│   ├── DIOR_03135_vehicle.png
│   ├── DIOR_03135_Expressway-toll-station.png
│   └── ...
├── debug_gt_boxes/             # GT调试结果
│   ├── debug_sample0_DIOR_03135.png
│   └── ...
├── comprehensive_comparison/    # 全面对比结果
│   ├── comprehensive_comparison_sample0.png
│   └── ...
└── legacy_outputs/             # 历史输出文件
```

### 图像格式
- **4模式对比**: 4行×13列 (4模式 × 1原图+12层)
- **GT调试**: 2列布局 (原图+坐标信息)
- **全面对比**: 4行×13列 (4模式 × 1原图+12层)

## 技术细节

### VV机制
- 替换最后6层注意力为V-V attention
- 双路径设计：保留原始QK路径和VV路径
- CLS token使用原始路径，patches使用VV路径

### Feature Surgery
- 基于CLIP Surgery论文的去冗余方法
- 计算类别权重，剔除多类别共享特征
- 保留类别特异性特征

### GT边界框处理
- 自动缩放：原始尺寸 → 224×224
- 精确坐标：支持任意原始尺寸
- 类别过滤：只显示查询类别的框

## 文档

详细文档请查看 `docs/` 目录：
- `4_mode_comparison_report.md` - 4模式对比修复报告
- `FINAL_SUMMARY.txt` - 最终总结
- `MULTI_CLASS_README.txt` - 多类别处理说明

## 更新日志

- **2024-10-30**: 整合4个核心文件为统一脚本
- **2024-10-30**: 修复4种模式对比实现
- **2024-10-30**: 完善GT边界框处理
- **2024-10-30**: 优化目录结构和文档
"""
    
    readme_path = directory / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"  ✓ 创建: README.md")

if __name__ == '__main__':
    cleanup_directory()

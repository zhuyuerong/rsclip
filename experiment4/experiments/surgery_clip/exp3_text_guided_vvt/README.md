# exp3_text_guided_vvt - 统一热图生成器

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

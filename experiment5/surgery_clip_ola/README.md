# Experiment 5: Surgery CLIP - OLA去接缝版本

## 概述

基于Experiment 3的统一热图生成器，添加**OLA (Overlap-Add) 加权拼接**功能，消除滑窗热图的接缝条纹，提升小目标可见性。

## 核心改进

### 问题诊断
- **接缝条纹**: 热图出现明显的横竖高低峰值条纹
- **根本原因**: 滑窗拼接时边缘极端值被叠加，缺乏平滑过渡
- **影响**: 小目标/多目标的响应被条纹干扰，难以观察

### 解决方案
**OLA (Overlap-Add) 加权拼接**:
1. **平滑权重窗口**: 使用余弦窗口（Hann），中心权重高、边缘权重低
2. **加权累积**: `accumulator += tile * weight` 而非硬覆盖
3. **加权平均**: `final_heatmap = accumulator / (weight_sum + eps)`
4. **统一归一化**: 禁用每窗min-max，改为全图统一分位归一化
5. **诊断输出**: 提供权重和图（acc_w）用于验证覆盖均匀性

## 技术实现

### OLA核心函数
```python
create_blending_weight(h, w, blend_type='cosine'):
    # 生成平滑权重窗口（中心≈1，边缘→0）
    
stitch_ola(tiles, coords, out_h, out_w, ...):
    # 重叠-加权-平均拼接 + 全图统一归一化
    
visualize_ola_diagnosis(image, heatmap, acc_w, ...):
    # 诊断OLA拼接质量（覆盖均匀性）
```

### 集成点
- `UnifiedHeatmapGenerator.__init__(enable_ola=...)`: 添加OLA配置
- `generate_multi_mode_heatmaps(...)`: 支持OLA/非OLA两种路径
- `process_multi_class_images(...)`: 自动保存诊断图

## 使用方法

### 基础模式（无OLA，与exp3相同）
```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

PYTHONPATH=. ovadetr_env/bin/python3.9 experiment5/surgery_clip_ola/unified_heatmap_generator.py \
  --mode multi_class \
  --max-samples 10 \
  --layers 1 2 3 4 5 6 7 8 9 10 11 12
```

### OLA模式（去接缝条纹）
```bash
PYTHONPATH=. ovadetr_env/bin/python3.9 experiment5/surgery_clip_ola/unified_heatmap_generator.py \
  --mode multi_class \
  --max-samples 10 \
  --layers 1 2 3 4 5 6 7 8 9 10 11 12 \
  --use-ola \
  --tile-size 224 \
  --tile-stride 112 \
  --percentile \
  --pmin 5.0 \
  --pmax 95.0
```

### 快速测试（3层）
```bash
PYTHONPATH=. ovadetr_env/bin/python3.9 experiment5/surgery_clip_ola/unified_heatmap_generator.py \
  --mode multi_class \
  --max-samples 3 \
  --layers 6 9 12 \
  --use-ola \
  --percentile
```

## 参数说明

### 通用参数
- `--dataset`: 数据集路径（默认：datasets/mini_dataset）
- `--mode`: 运行模式（multi_class/debug_gt/all）
- `--max-samples`: 最大样本数（默认：10）
- `--layers`: 要分析的层（默认：1-12）

### OLA参数
- `--use-ola`: 启用OLA加权拼接（默认：False）
- `--tile-size`: 滑窗大小（默认：224）
- `--tile-stride`: 滑窗步长（默认：112，推荐tile-size//2）
- `--percentile`: 使用分位归一化（抗异常值）
- `--pmin`: 分位下限百分比（默认：5.0）
- `--pmax`: 分位上限百分比（默认：95.0）

## 输出结果

### 目录结构
```
experiment5/surgery_clip_ola/
├── unified_heatmap_generator.py    # 主脚本
├── README.md                       # 本文档
└── results/
    ├── multi_class_4mode/          # 4模式对比热图
    │   ├── DIOR_03135_Expressway-toll-station.png
    │   ├── DIOR_03135_vehicle.png
    │   ├── DIOR_05386_overpass.png
    │   └── DIOR_05386_vehicle.png
    └── ola_diagnosis/              # OLA诊断图（仅--use-ola时）
        ├── DIOR_03135_Expressway-toll-station_diagnosis.png
        ├── DIOR_03135_vehicle_diagnosis.png
        ├── DIOR_05386_overpass_diagnosis.png
        └── DIOR_05386_vehicle_diagnosis.png
```

### 图像格式
- **4模式对比**: 4行×(1+N)列（4模式 × 1原图+N层）
- **诊断图**: 1行×4列（原图、叠加、纯热图、权重和）

### 诊断图说明
诊断图包含4列：
1. **Original Image**: 原图
2. **Heatmap (OLA)**: 叠加热图（alpha=0.7）
3. **Pure Heatmap**: 纯热图（无RGB）
4. **Weight Sum**: 权重和图（均匀=无接缝）

**覆盖统计**:
- `std < 0.1 * mean`: ✓ 覆盖均匀，无可见接缝
- `std > 0.1 * mean`: ⚠ 覆盖不均，建议减小stride

## 技术细节

### 4种模式对比
1. **With Surgery**: 标准RemoteCLIP + Feature Surgery去冗余
2. **Without Surgery**: 标准RemoteCLIP + 余弦相似度
3. **With VV**: CLIPSurgery (VV机制) + 余弦相似度
4. **Complete Surgery**: CLIPSurgery (VV机制) + Feature Surgery

### OLA原理
```python
# ❌ 原始做法（直接叠加/取max）
final_heatmap[y:y+h, x:x+w] = tile_heatmap  # 硬覆盖，接缝明显

# ✅ OLA做法（加权平均）
weight = cosine_window(h, w)  # 中心权重高，边缘低
accumulator[y:y+h, x:x+w] += tile_heatmap * weight
weight_sum[y:y+h, x:x+w] += weight
final_heatmap = accumulator / (weight_sum + 1e-8)  # 平滑过渡
```

### 归一化策略
- **分位归一化**: 使用5%-95%分位，抗极端值
- **全图归一化**: 禁用每窗独立归一化，避免接缝跳变
- **裁切处理**: 值域限制在[0,1]

## 性能对比

### 无OLA (基准)
- 速度: 快（单次前向传播）
- 质量: 可能有接缝条纹
- 小目标: 7×7分辨率限制

### 有OLA
- 速度: 略慢（4个滑窗 = 2×2）
- 质量: 平滑无接缝
- 小目标: 分位归一化提升可见性

## 验证结果

### OLA测试（2个样本，3层）
- ✓ OLA模式正常运行
- ✓ 生成4个热图 + 4个诊断图
- ✓ 覆盖统计显示：std=0.2793（覆盖稍不均，可调stride）

### 建议优化
- **覆盖更均匀**: 减小stride至tile_size//3 (例如224->75)
- **更高分辨率**: 将输入升至336/448（需pos_embed插值）
- **多层融合**: 融合L11-L12增强定位

## 更新日志

- **2024-10-30**: 创建experiment5目录
- **2024-10-30**: 从exp3迁移代码
- **2024-10-30**: 集成OLA加权拼接功能
- **2024-10-30**: 添加诊断可视化
- **2024-10-30**: 测试验证OLA效果

## 相关文件

- `experiment4/core/models/clip_surgery.py`: 模型包装器（已添加compute_similarity）
- `experiment4/experiments/surgery_clip/exp3_text_guided_vvt/`: 原始版本（无OLA）
- `experiment5/surgery_clip_ola/`: OLA去接缝版本（本目录）

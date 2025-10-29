# VV机制与多层特征分析实验

## 实验目标

对比3种模式在seen/unseen数据集上的性能，并分析不同层特征的表现：

1. **标准RemoteCLIP**：直接使用RemoteCLIP特征
2. **Surgery去冗余**：使用Feature Surgery去冗余
3. **Surgery+VV机制**：使用Feature Surgery + VV自注意力机制

## 目录结构

```
04_vv_multi_layer_analysis/
├── config_experiments.py       # 3种模式配置
├── compare_three_modes.py      # 主对比脚本
├── layer_analysis.py           # 多层特征分析（待实现）
├── patch_similarity_matrix.py  # patch相似度矩阵（待实现）
├── text_guided_vvt.py          # 文本引导VV^T热图（待实现）
├── utils/
│   ├── seen_unseen_split.py    # seen/unseen数据集划分
│   ├── heatmap_utils.py        # 热图生成工具（待实现）
│   └── visualization.py        # 可视化工具（待实现）
└── outputs/
    ├── mode_comparison/        # 3种模式对比结果
    ├── layer_analysis/         # 多层分析结果
    └── vvt_heatmaps/           # VV^T热图
```

## 快速开始

### 1. 测试seen/unseen数据集划分

```bash
cd experiment4/experiments/04_vv_multi_layer_analysis/utils
python seen_unseen_split.py
```

### 2. 快速验证（mini_dataset）

```bash
cd experiment4/experiments/04_vv_multi_layer_analysis
python compare_three_modes.py --quick-test
```

### 3. 完整评估（DIOR数据集）

```bash
python compare_three_modes.py --dataset /path/to/DIOR --full-eval
```

## 配置说明

### Unseen类别

默认5个unseen类别：
- airplane
- bridge
- storagetank
- vehicle
- windmill

### 3种模式配置

| 模式 | use_surgery | use_vv_mechanism | 描述 |
|------|------------|------------------|------|
| mode1_baseline | False | False | 标准RemoteCLIP |
| mode2_surgery | True | False | Surgery去冗余 |
| mode3_surgery_vv | True | True | Surgery+VV机制 |

## 核心功能

### Feature Surgery

基于CLIP Surgery论文的去冗余实现：

```python
from experiment4.core.models.clip_surgery import clip_feature_surgery

# 计算去冗余后的相似度
similarity = clip_feature_surgery(image_features, text_features, t=2)
```

### 多层特征提取

```python
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper

model = CLIPSurgeryWrapper(config)
layer_features = model.get_layer_features(images, layer_indices=[1, 6, 9, 12])
```

## 预期输出

### 热图对比

- `outputs/mode_comparison/mode_comparison_seen.png`: Seen数据集热图网格
- `outputs/mode_comparison/mode_comparison_unseen.png`: Unseen数据集热图网格

### mAP对比

- `outputs/mode_comparison/map_comparison.json`: mAP对比表

## 待完成任务

- [x] 重构clip_surgery.py支持3种模式
- [x] 更新config.py添加模式控制配置
- [x] 创建seen/unseen数据集划分工具
- [x] 实现3种模式对比脚本（基础版）
- [ ] 实现完整mAP计算
- [ ] 实现多层特征分析（layer_analysis.py）
- [ ] 实现patch相似度矩阵分析
- [ ] 实现文本引导VV^T热图
- [ ] 在完整DIOR数据集上运行评估
- [ ] 生成最终报告

## 参考

- 计划文档: `/vv-------.plan.md`
- CLIP Surgery论文: https://arxiv.org/abs/2304.05653


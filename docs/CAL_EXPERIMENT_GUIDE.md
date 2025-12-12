# CAL实验指南

## 📋 概述

本实验框架实现了**CAL (Counterfactual Attention Learning) + SurgeryCLIP**的改进方法，采用**可插拔设计**，可以随时切换回原始实验。

---

## 🎯 实验设计

### Q1: 负样本策略（4个实验）

| 实验ID | 策略 | 说明 |
|--------|------|------|
| `q1_exp1_fixed` | 固定负样本 | 使用固定的负样本文本：`["background", "irrelevant objects"]` |
| `q1_exp2_dynamic` | 动态负样本 | 从DIOR数据集中随机选择3个其他类别作为负样本 |
| `q1_exp3_random` | 随机负样本 | 使用随机生成的文本作为负样本 |
| `q1_exp4_combined` | 组合负样本 | 固定负样本 + 动态负样本的组合 |

### Q2: 加权减法（4个实验）

| 实验ID | alpha值 | 说明 |
|--------|---------|------|
| `q2_exp1_alpha05` | 0.5 | 减半权重：`similarity_pos - 0.5 * similarity_neg` |
| `q2_exp2_alpha10` | 1.0 | 直接减法（baseline）：`similarity_pos - 1.0 * similarity_neg` |
| `q2_exp3_alpha15` | 1.5 | 1.5倍权重：`similarity_pos - 1.5 * similarity_neg` |
| `q2_exp4_alpha20` | 2.0 | 2倍权重：`similarity_pos - 2.0 * similarity_neg` |

### Q3: 操作位置（3个实验）

| 实验ID | 操作位置 | 说明 |
|--------|---------|------|
| `q3_exp1_feature` | 特征空间 | 在`clip_feature_surgery`函数中进行CAL操作 |
| `q3_exp2_similarity` | 相似度空间 | 在`generate_heatmap`函数中进行CAL操作 |
| `q3_exp3_both` | 双重操作 | 特征空间 + 相似度空间双重CAL操作 |

---

## 🚀 使用方法

### 1. 运行CAL实验

```python
from configs.cal_experiments import ALL_CAL_CONFIGS
from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper

# 选择实验配置
cal_config = ALL_CAL_CONFIGS['q1_exp1_fixed']

# 创建模型（启用CAL）
model = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    device='cuda',
    use_surgery_single='empty',
    use_surgery_multi=True,
    cal_config=cal_config  # 🔥 传入CAL配置
)

model.load_model()

# 生成热图（会自动应用CAL）
heatmap = model.generate_heatmap(image, ['vehicle'])
```

### 2. 切回原始实验（不使用CAL）

```python
# 方法1: 不传入cal_config（推荐）
model = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    device='cuda',
    use_surgery_single='empty',
    use_surgery_multi=True
    # 不传入cal_config，完全使用原始逻辑
)

# 方法2: 传入enable_cal=False的配置
from src.competitors.clip_methods.surgeryclip.cal_config import CALConfig

cal_config_disabled = CALConfig(enable_cal=False)
model = SurgeryCLIPWrapper(
    ...,
    cal_config=cal_config_disabled
)
```

### 3. 自定义CAL配置

```python
from src.competitors.clip_methods.surgeryclip.cal_config import CALConfig

# 自定义配置
my_cal_config = CALConfig(
    enable_cal=True,
    negative_mode='combined',  # 组合负样本
    fixed_negatives=["background"],
    num_dynamic_negatives=3,
    alpha=1.2,  # 自定义权重
    cal_space='similarity',  # 相似度空间
    experiment_name='my_custom_exp',
    verbose=True
)

model = SurgeryCLIPWrapper(..., cal_config=my_cal_config)
```

---

## 📁 文件结构

```
src/competitors/clip_methods/surgeryclip/
├── cal_config.py          # CAL配置类和负样本生成器
├── cal_modules.py         # CAL操作模块（特征空间+相似度空间）
├── clip.py                # CLIP工具函数（从外部CLIP_Surgery导入）
├── model_wrapper.py       # 主模型包装器（已集成CAL）
└── ...

configs/
└── cal_experiments.py     # 所有实验配置定义

docs/
└── CAL_EXPERIMENT_GUIDE.md  # 本说明文档
```

---

## 🔧 核心设计

### 可插拔机制

1. **条件导入**: 如果CAL模块不存在，代码会自动回退到原始逻辑
2. **配置开关**: 通过`cal_config.enable_cal`控制是否启用CAL
3. **向后兼容**: 不传入`cal_config`时，完全使用原始SurgeryCLIP逻辑

### 代码位置

- **CAL相似度空间操作**: `model_wrapper.py` 第364行之后
- **CAL特征空间操作**: 需要在`clip_feature_surgery`中实现（当前版本暂未实现）

---

## 📊 实验追踪

每个实验会自动记录：

1. **配置信息**: 保存在`outputs/cal_experiments/{experiment_id}.json`
2. **实验结果**: 包含热图统计信息（min, max, mean, std）

查看实验结果：
```python
import json
with open('outputs/cal_experiments/q1_exp1_fixed_negfixed_alpha1.0_spacesimilarity.json') as f:
    data = json.load(f)
    print(data)
```

---

## ⚠️ 注意事项

1. **实验模式**: 这是一个实验性功能，默认不启用
2. **性能影响**: CAL会增加计算开销（需要编码负样本）
3. **结果对比**: 建议同时运行原始实验和CAL实验进行对比

---

## 🔄 快速切换

### 切换到CAL实验
```python
from configs.cal_experiments import ALL_CAL_CONFIGS
model = SurgeryCLIPWrapper(..., cal_config=ALL_CAL_CONFIGS['q1_exp1_fixed'])
```

### 切回原始实验
```python
model = SurgeryCLIPWrapper(...)  # 不传入cal_config即可
```

---

## 📝 实验记录

运行实验后，结果保存在：
- `outputs/cal_experiments/{experiment_id}.json` - 实验配置和结果
- `outputs/cal_experiments/{experiment_id}/` - 热图输出目录（如果使用运行脚本）

---

## 🐛 故障排除

### 问题1: 导入错误
```
ImportError: cannot import name 'CALConfig'
```
**解决**: 确保`cal_config.py`文件存在于`surgeryclip`目录下

### 问题2: CAL未生效
**检查**:
1. `cal_config.enable_cal`是否为`True`
2. `cal_config.cal_space`是否匹配（'similarity'或'both'）
3. 查看控制台输出是否有"✅ CAL已启用"消息

### 问题3: 想完全禁用CAL
**解决**: 不传入`cal_config`参数，或传入`CALConfig(enable_cal=False)`

---

## 📚 相关文件

- `src/competitors/clip_methods/surgeryclip/cal_config.py` - 配置定义
- `src/competitors/clip_methods/surgeryclip/cal_modules.py` - CAL操作实现
- `configs/cal_experiments.py` - 实验配置
- `src/competitors/clip_methods/surgeryclip/model_wrapper.py` - 主实现

---

## ✅ 验证清单

运行实验前，确认：
- [ ] CAL模块文件已创建（cal_config.py, cal_modules.py）
- [ ] clip.py文件存在（用于导入外部函数）
- [ ] model_wrapper.py已更新（包含CAL逻辑）
- [ ] 实验配置文件存在（cal_experiments.py）
- [ ] 可以正常导入CALConfig

---

**最后更新**: 2024年
**维护者**: CAL实验框架







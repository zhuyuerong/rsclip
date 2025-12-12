# CAL实验实现总结

## 📋 实现概述

本次实现了一个**可插拔的CAL (Counterfactual Attention Learning) 实验框架**，可以随时切换回原始实验，不影响原有功能。

---

## 📁 新增文件

### 1. 核心模块文件

| 文件路径 | 说明 |
|---------|------|
| `src/competitors/clip_methods/surgeryclip/cal_config.py` | CAL配置类和负样本生成器（Q1的4种策略） |
| `src/competitors/clip_methods/surgeryclip/cal_modules.py` | CAL操作模块（特征空间+相似度空间，Q3） |
| `src/competitors/clip_methods/surgeryclip/clip.py` | CLIP工具函数导入（从外部CLIP_Surgery） |

### 2. 配置文件

| 文件路径 | 说明 |
|---------|------|
| `configs/cal_experiments.py` | 所有实验配置定义（Q1/Q2/Q3共11个实验） |

### 3. 文档文件

| 文件路径 | 说明 |
|---------|------|
| `docs/CAL_EXPERIMENT_GUIDE.md` | 完整实验指南 |
| `docs/CAL_QUICK_REFERENCE.md` | 快速参考 |
| `docs/CAL_IMPLEMENTATION_SUMMARY.md` | 本文件（实现总结） |

### 4. 测试脚本

| 文件路径 | 说明 |
|---------|------|
| `scripts/test_cal_experiment.py` | CAL功能快速测试脚本 |

---

## 🔧 修改的文件

### `src/competitors/clip_methods/surgeryclip/model_wrapper.py`

**修改内容**:
1. 添加CAL模块的条件导入（如果不存在不影响原有功能）
2. 在`__init__`中添加`cal_config`参数（可选）
3. 在`generate_heatmap`方法的第364行后添加CAL相似度空间操作

**关键代码位置**:
- 第24-32行: CAL模块条件导入
- 第31-39行: `__init__`方法签名（添加`cal_config`参数）
- 第64-78行: CAL模块初始化逻辑
- 第364-390行: CAL相似度空间操作（在排除class token之后）

**向后兼容性**: ✅ 完全兼容
- 不传入`cal_config`时，完全使用原始逻辑
- CAL模块不存在时，自动回退到原始逻辑

---

## 🎯 实验设计

### Q1: 负样本策略（4个实验）

1. **固定负样本** (`q1_exp1_fixed`)
   - 使用固定的负样本文本：`["background", "irrelevant objects"]`

2. **动态负样本** (`q1_exp2_dynamic`)
   - 从DIOR数据集中随机选择3个其他类别作为负样本

3. **随机负样本** (`q1_exp3_random`)
   - 使用随机生成的文本作为负样本

4. **组合负样本** (`q1_exp4_combined`)
   - 固定负样本 + 动态负样本的组合

### Q2: 加权减法（4个实验）

- `q2_exp1_alpha05`: alpha=0.5
- `q2_exp2_alpha10`: alpha=1.0 (baseline)
- `q2_exp3_alpha15`: alpha=1.5
- `q2_exp4_alpha20`: alpha=2.0

### Q3: 操作位置（3个实验）

- `q3_exp1_feature`: 特征空间操作
- `q3_exp2_similarity`: 相似度空间操作（已实现）
- `q3_exp3_both`: 双重操作

---

## 🔄 如何切换回原始实验

### 方法1: 不传入cal_config（推荐）

```python
# 原始实验（不使用CAL）
model = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    device='cuda',
    use_surgery_single='empty',
    use_surgery_multi=True
    # 不传入cal_config，完全使用原始逻辑
)
```

### 方法2: 禁用CAL配置

```python
from src.competitors.clip_methods.surgeryclip.cal_config import CALConfig

# 创建禁用CAL的配置
cal_config_disabled = CALConfig(enable_cal=False)

model = SurgeryCLIPWrapper(
    ...,
    cal_config=cal_config_disabled
)
```

### 方法3: 临时删除CAL代码（不推荐）

如果需要完全移除CAL代码，可以：
1. 删除`cal_config.py`、`cal_modules.py`文件
2. 恢复`model_wrapper.py`到原始版本（删除CAL相关代码）

**注意**: 由于使用了条件导入，即使删除CAL文件，原有功能仍然可以正常工作。

---

## ✅ 验证清单

运行实验前，确认：

- [x] CAL模块文件已创建
  - [x] `cal_config.py` - 配置和负样本生成器
  - [x] `cal_modules.py` - CAL操作模块
  - [x] `clip.py` - CLIP函数导入

- [x] 配置文件已创建
  - [x] `configs/cal_experiments.py` - 所有实验配置

- [x] 文档已创建
  - [x] `docs/CAL_EXPERIMENT_GUIDE.md` - 完整指南
  - [x] `docs/CAL_QUICK_REFERENCE.md` - 快速参考

- [x] 主文件已修改
  - [x] `model_wrapper.py` - 集成CAL功能（可插拔）

- [x] 向后兼容性
  - [x] 不传入`cal_config`时使用原始逻辑
  - [x] CAL模块不存在时自动回退

---

## 🧪 测试方法

### 1. 快速测试

```bash
python scripts/test_cal_experiment.py
```

### 2. 运行单个实验

```python
from configs.cal_experiments import ALL_CAL_CONFIGS
from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper

cal_config = ALL_CAL_CONFIGS['q1_exp1_fixed']
model = SurgeryCLIPWrapper(..., cal_config=cal_config)
model.load_model()
heatmap = model.generate_heatmap(image, ['vehicle'])
```

### 3. 验证原始功能

```python
# 不传入cal_config，验证原始功能正常
model = SurgeryCLIPWrapper(...)  # 不传入cal_config
model.load_model()
heatmap = model.generate_heatmap(image, ['vehicle'])
```

---

## 📊 实验追踪

每个实验会自动记录配置和结果到：
- `outputs/cal_experiments/{experiment_id}.json`

实验ID格式：`{experiment_name}_neg{negative_mode}_alpha{alpha}_space{cal_space}`

---

## 🔍 代码检查点

### CAL是否启用

查看控制台输出：
```
✅ CAL已启用: q1_exp1_fixed_negfixed_alpha1.0_spacesimilarity
```

### CAL操作位置

在`model_wrapper.py`的`generate_heatmap`方法中：
- 第364行: 排除class token
- 第367-390行: CAL相似度空间操作（如果启用）

### 原始逻辑位置

原始SurgeryCLIP逻辑保持不变：
- 第232-358行: 原有的Surgery逻辑
- 第364行: 排除class token（原有）
- 第370行: 生成热图（原有）

---

## 🎓 设计原则

1. **可插拔**: 通过配置控制，不影响核心代码
2. **向后兼容**: 不传入配置时完全使用原始逻辑
3. **可追溯**: 每个实验自动记录配置和结果
4. **易扩展**: 新增实验只需添加配置

---

## 📝 注意事项

1. **实验性质**: 这是一个实验性功能，默认不启用
2. **性能影响**: CAL会增加计算开销（需要编码负样本）
3. **结果对比**: 建议同时运行原始实验和CAL实验进行对比
4. **特征空间操作**: Q3的特征空间操作（`cal_space='feature'`）当前版本暂未完全实现，需要进一步开发

---

## 🔗 相关文档

- **完整指南**: `docs/CAL_EXPERIMENT_GUIDE.md`
- **快速参考**: `docs/CAL_QUICK_REFERENCE.md`
- **实验配置**: `configs/cal_experiments.py`

---

## ✅ 完成状态

- [x] CAL配置模块（cal_config.py）
- [x] CAL操作模块（cal_modules.py）
- [x] CLIP函数导入（clip.py）
- [x] 主模型集成（model_wrapper.py）
- [x] 实验配置（cal_experiments.py）
- [x] 完整文档（CAL_EXPERIMENT_GUIDE.md）
- [x] 快速参考（CAL_QUICK_REFERENCE.md）
- [x] 测试脚本（test_cal_experiment.py）
- [x] 向后兼容性验证

---

**实现日期**: 2024年
**状态**: ✅ 完成，可投入使用







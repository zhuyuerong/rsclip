# CAL实验快速参考

## 🚀 快速开始

### 1. 运行CAL实验

```python
from configs.cal_experiments import ALL_CAL_CONFIGS
from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper

# 选择实验
cal_config = ALL_CAL_CONFIGS['q1_exp1_fixed']

# 创建模型（启用CAL）
model = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    device='cuda',
    cal_config=cal_config  # 🔥 关键：传入CAL配置
)

model.load_model()
heatmap = model.generate_heatmap(image, ['vehicle'])
```

### 2. 切回原始实验

```python
# 方法1: 不传入cal_config（推荐）
model = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    device='cuda'
    # 不传入cal_config即可
)

# 方法2: 禁用CAL
from src.competitors.clip_methods.surgeryclip.cal_config import CALConfig
model = SurgeryCLIPWrapper(
    ...,
    cal_config=CALConfig(enable_cal=False)
)
```

---

## 📋 所有实验配置

### Q1: 负样本策略
- `q1_exp1_fixed` - 固定负样本
- `q1_exp2_dynamic` - 动态负样本
- `q1_exp3_random` - 随机负样本
- `q1_exp4_combined` - 组合负样本

### Q2: 加权减法
- `q2_exp1_alpha05` - alpha=0.5
- `q2_exp2_alpha10` - alpha=1.0 (baseline)
- `q2_exp3_alpha15` - alpha=1.5
- `q2_exp4_alpha20` - alpha=2.0

### Q3: 操作位置
- `q3_exp1_feature` - 特征空间
- `q3_exp2_similarity` - 相似度空间
- `q3_exp3_both` - 双重操作

### 组合实验
- `best_combination` - 最佳组合

---

## 🔧 自定义配置

```python
from src.competitors.clip_methods.surgeryclip.cal_config import CALConfig

my_config = CALConfig(
    enable_cal=True,
    negative_mode='combined',  # 'fixed' | 'dynamic' | 'random' | 'combined'
    fixed_negatives=["background"],
    num_dynamic_negatives=3,
    alpha=1.2,  # 加权系数
    cal_space='similarity',  # 'feature' | 'similarity' | 'both'
    experiment_name='my_exp',
    verbose=True
)
```

---

## 📁 文件位置

- **配置**: `configs/cal_experiments.py`
- **实现**: `src/competitors/clip_methods/surgeryclip/`
- **文档**: `docs/CAL_EXPERIMENT_GUIDE.md`

---

## ✅ 验证清单

- [ ] CAL模块文件存在
- [ ] 可以导入`CALConfig`
- [ ] 模型可以正常创建
- [ ] CAL功能可以启用/禁用

---

## 🐛 常见问题

**Q: 如何确认CAL已启用？**
A: 查看控制台输出，应该看到"✅ CAL已启用: {experiment_id}"

**Q: 如何完全禁用CAL？**
A: 不传入`cal_config`参数，或传入`CALConfig(enable_cal=False)`

**Q: CAL未生效怎么办？**
A: 检查`cal_config.enable_cal`是否为`True`，以及`cal_config.cal_space`是否正确

---

**详细文档**: 查看 `docs/CAL_EXPERIMENT_GUIDE.md`







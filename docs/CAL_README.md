# CAL实验框架 - 快速开始

## 🎯 这是什么？

这是一个**可插拔的CAL (Counterfactual Attention Learning) 实验框架**，用于改进SurgeryCLIP的热图生成效果。

**核心特点**:
- ✅ **可插拔**: 通过配置控制，不影响原有代码
- ✅ **可追溯**: 自动记录实验配置和结果
- ✅ **易切换**: 随时切回原始实验

---

## 🚀 5分钟快速开始

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
    cal_config=cal_config  # 🔥 关键
)

model.load_model()
heatmap = model.generate_heatmap(image, ['vehicle'])
```

### 2. 切回原始实验

```python
# 不传入cal_config即可
model = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    device='cuda'
    # 不传入cal_config，使用原始逻辑
)
```

---

## 📋 实验列表

### Q1: 负样本策略（4个）
- `q1_exp1_fixed` - 固定负样本
- `q1_exp2_dynamic` - 动态负样本
- `q1_exp3_random` - 随机负样本
- `q1_exp4_combined` - 组合负样本

### Q2: 加权减法（4个）
- `q2_exp1_alpha05` - alpha=0.5
- `q2_exp2_alpha10` - alpha=1.0
- `q2_exp3_alpha15` - alpha=1.5
- `q2_exp4_alpha20` - alpha=2.0

### Q3: 操作位置（3个）
- `q3_exp1_feature` - 特征空间
- `q3_exp2_similarity` - 相似度空间
- `q3_exp3_both` - 双重操作

**总共11个实验配置**

---

## 📁 文件结构

```
src/competitors/clip_methods/surgeryclip/
├── cal_config.py          # CAL配置和负样本生成器
├── cal_modules.py         # CAL操作模块
├── clip.py                # CLIP函数导入
└── model_wrapper.py       # 主模型（已集成CAL）

configs/
└── cal_experiments.py     # 所有实验配置

docs/
├── CAL_README.md          # 本文件
├── CAL_EXPERIMENT_GUIDE.md # 完整指南
├── CAL_QUICK_REFERENCE.md  # 快速参考
└── CAL_IMPLEMENTATION_SUMMARY.md # 实现总结
```

---

## 🔧 自定义配置

```python
from src.competitors.clip_methods.surgeryclip.cal_config import CALConfig

my_config = CALConfig(
    enable_cal=True,
    negative_mode='combined',  # 'fixed' | 'dynamic' | 'random' | 'combined'
    alpha=1.2,                  # 加权系数
    cal_space='similarity',     # 'feature' | 'similarity' | 'both'
    experiment_name='my_exp'
)
```

---

## ✅ 验证

运行测试脚本：
```bash
python scripts/test_cal_experiment.py
```

---

## 📚 详细文档

- **完整指南**: [CAL_EXPERIMENT_GUIDE.md](CAL_EXPERIMENT_GUIDE.md)
- **快速参考**: [CAL_QUICK_REFERENCE.md](CAL_QUICK_REFERENCE.md)
- **实现总结**: [CAL_IMPLEMENTATION_SUMMARY.md](CAL_IMPLEMENTATION_SUMMARY.md)

---

## ⚠️ 重要提示

1. **这是实验性功能**，默认不启用
2. **可以随时切回原始实验**，不影响原有功能
3. **建议对比实验**，同时运行原始和CAL版本

---

**最后更新**: 2024年







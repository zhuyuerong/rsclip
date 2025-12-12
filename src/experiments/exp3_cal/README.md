# CAL实验 - exp3_cal

## 📋 概述

本文件夹包含**CAL (Counterfactual Attention Learning) 实验**的所有相关文件。

---

## 📁 文件结构

```
exp3_cal/
├── cal_config.py          # CAL配置类和负样本生成器
├── cal_modules.py         # CAL操作模块（特征空间+相似度空间）
├── experiment_configs.py  # 所有实验配置定义（11个实验）
├── run_experiment.py      # 实验运行脚本
└── README.md              # 本文件
```

---

## 🚀 快速开始

### 1. 运行单个实验

```bash
python src/experiments/exp3_cal/run_experiment.py \
    --config q1_exp1_fixed \
    --image datasets/mini-DIOR/test/images/00679.jpg \
    --class vehicle \
    --checkpoint checkpoints/ViT-B-32.pt \
    --device cuda
```

### 2. 在Python中使用

```python
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper
from src.experiments.exp3_cal.experiment_configs import ALL_CAL_CONFIGS

# 选择实验配置
cal_config = ALL_CAL_CONFIGS['q1_exp1_fixed']

# 创建模型
model = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    device='cuda',
    cal_config=cal_config
)

model.load_model()

# 生成热图
from PIL import Image
image = Image.open('path/to/image.jpg').convert('RGB')
heatmap = model.generate_heatmap(image, ['vehicle'])
```

---

## 📋 实验列表

### Q1: 负样本策略（4个实验）
- `q1_exp1_fixed` - 固定负样本
- `q1_exp2_dynamic` - 动态负样本
- `q1_exp3_random` - 随机负样本
- `q1_exp4_combined` - 组合负样本

### Q2: 加权减法（4个实验）
- `q2_exp1_alpha05` - alpha=0.5
- `q2_exp2_alpha10` - alpha=1.0 (baseline)
- `q2_exp3_alpha15` - alpha=1.5
- `q2_exp4_alpha20` - alpha=2.0

### Q3: 操作位置（3个实验）
- `q3_exp1_feature` - 特征空间
- `q3_exp2_similarity` - 相似度空间
- `q3_exp3_both` - 双重操作

**总计**: 11个实验配置

---

## 🔧 自定义配置

```python
from src.experiments.exp3_cal.cal_config import CALConfig

# 创建自定义配置
my_config = CALConfig(
    enable_cal=True,
    negative_mode='combined',
    fixed_negatives=["background"],
    num_dynamic_negatives=3,
    alpha=1.2,
    cal_space='similarity',
    experiment_name='my_custom_exp'
)
```

---

## 📊 输出结果

实验结果保存在：
- `outputs/exp3_cal/{config_name}/{image_name}_{class_name}_cal.png`

实验记录保存在：
- `outputs/cal_experiments/{experiment_id}.json`

---

## 🔄 切回原始实验

不使用CAL时，只需不传入`cal_config`参数：

```python
model = SurgeryCLIPWrapper(
    model_name='surgeryclip',
    checkpoint_path='checkpoints/ViT-B-32.pt',
    device='cuda'
    # 不传入cal_config，使用原始逻辑
)
```

---

## 📚 相关文档

- **完整指南**: `docs/CAL_EXPERIMENT_GUIDE.md`
- **快速参考**: `docs/CAL_QUICK_REFERENCE.md`
- **实现总结**: `docs/CAL_IMPLEMENTATION_SUMMARY.md`

---

## ⚠️ 注意事项

1. 确保模型权重文件存在：`checkpoints/ViT-B-32.pt`
2. 确保测试图像路径正确
3. 确保有足够的GPU内存（或使用`--device cpu`）

---

**实验文件夹**: `src/experiments/exp3_cal/`
**创建日期**: 2024年







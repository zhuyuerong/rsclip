# Exp9 依赖说明文档

## 📋 概述

实验9 (Pseudo Query for Object Detection) 依赖于**标准Deformable DETR**代码库。本文档说明所有必需的依赖文件及其来源。

---

## 🏗️ 依赖架构

```
exp9_pseudo_query/
├── models/
│   ├── deformable_detr_pseudo.py   # 实验9自定义: Pseudo Query包装器
│   ├── heatmap_query_gen.py        # 实验9自定义: 热图→query生成
│   └── query_injection.py          # 实验9自定义: Query混合策略
│
└── [依赖] external/Deformable-DETR/
    ├── models/
    │   ├── __init__.py             # ✅ 必需: build_model()
    │   ├── deformable_detr.py      # ✅ 必需: DeformableDETR, SetCriterion, PostProcess
    │   ├── deformable_transformer.py # ✅ 必需: DeformableTransformer
    │   ├── matcher.py              # ✅ 必需: HungarianMatcher
    │   ├── backbone.py             # ✅ 必需: ResNet backbone
    │   ├── position_encoding.py    # ✅ 必需: PositionEmbeddingSine
    │   ├── segmentation.py         # ✅ 必需: Loss functions (focal loss, dice loss)
    │   └── ops/                    # ✅ 必需: Multi-scale Deformable Attention CUDA算子
    │       ├── functions/
    │       │   └── ms_deform_attn_func.py
    │       ├── modules/
    │       │   └── ms_deform_attn.py
    │       ├── src/
    │       │   ├── cuda/           # CUDA实现
    │       │   └── cpu/            # CPU实现
    │       ├── setup.py            # 编译脚本
    │       └── make.sh             # 编译脚本
    └── util/
        ├── misc.py                 # ✅ 必需: NestedTensor, 工具函数
        └── box_ops.py              # ✅ 必需: Box操作 (IoU, center等)
```

---

## 🔑 关键依赖说明

### 1. **标准Deformable DETR模型** (`external/Deformable-DETR/models/`)

实验9的A0 baseline和A2/A3实验都使用**官方Deformable DETR实现**:

| 文件 | 用途 | 被哪些脚本导入 |
|------|------|---------------|
| `__init__.py` | 提供`build_model()`函数 | `train_a0_baseline.py`, `train_pseudo_query.py` |
| `deformable_detr.py` | 核心模型: `DeformableDETR`, `SetCriterion`, `PostProcess` | 所有训练脚本 |
| `deformable_transformer.py` | Deformable Transformer实现 | 通过`build_model()`间接调用 |
| `matcher.py` | Hungarian Matcher (用于loss计算) | 所有训练脚本 |
| `backbone.py` | ResNet backbone | 通过`build_model()`间接调用 |
| `position_encoding.py` | 位置编码 | 通过`build_model()`间接调用 |
| `segmentation.py` | Loss functions (focal loss, dice loss) | `SetCriterion`内部使用 |

**代码导入示例** (来自`train_a0_baseline.py`):
```python
sys.path.insert(0, str(project_root / 'external' / 'Deformable-DETR'))

from models import build_model
from models.deformable_detr import SetCriterion, PostProcess
from models.matcher import build_matcher
import util.misc as utils
from util.misc import NestedTensor, nested_tensor_from_tensor_list
```

### 2. **Multi-scale Deformable Attention CUDA算子** (`models/ops/`)

这是Deformable DETR的**核心创新**，必须编译才能运行:

```bash
cd external/Deformable-DETR/models/ops
bash make.sh
# 或
python setup.py build install
```

**验证编译成功**:
```bash
cd models/ops
python test.py
```

**为什么必需**:
- Deformable DETR使用可变形注意力机制，需要自定义CUDA算子
- 没有编译算子则无法运行训练
- 算子提供高效的多尺度可变形采样

### 3. **工具函数** (`external/Deformable-DETR/util/`)

| 文件 | 用途 |
|------|------|
| `misc.py` | `NestedTensor`, `nested_tensor_from_tensor_list`, 分布式训练工具 |
| `box_ops.py` | Box IoU, box center, box格式转换 |

---

## 📦 实验9自定义模块

实验9在标准Deformable DETR基础上**新增**了以下模块:

### 1. `models/deformable_detr_pseudo.py` (445行)

**功能**: Deformable DETR + Pseudo Query包装器

```python
class DeformableDETRPseudo(nn.Module):
    """
    在标准Deformable DETR基础上添加:
    1. Pseudo query生成接口
    2. Query混合机制
    3. Pseudo query相关loss
    """
```

**关键改动**:
- 接受热图或teacher boxes作为输入
- 生成pseudo queries
- 与learnable queries混合
- 支持额外的alignment/prior loss

### 2. `models/heatmap_query_gen.py` (661行)

**功能**: Q-Gen模块 - 从热图或teacher boxes生成pseudo queries

**核心类**:
- `HeatmapQueryGenerator`: 热图 → pseudo queries
- `TeacherQueryGenerator`: Teacher boxes → pseudo queries
- `PositionalEncoding2D`: 2D位置编码

**支持的Query生成模式**:
- `mean`: 简单平均池化
- `heatmap_weighted`: 热图加权池化 (推荐)
- `attn_pool`: 注意力池化

### 3. `models/query_injection.py` (456行)

**功能**: Q-Use模块 - Query混合策略 + 额外loss

**核心类**:
- `QueryMixer`: Pseudo query与learnable query混合
  - 支持: `replace`, `concat`, `ratio`, `attention`
- `QueryAlignmentLoss`: Query对齐loss (L2/cosine/NCE)
- `AttentionPriorLoss`: Attention先验loss

---

## 🔄 实验类型与依赖关系

### A0: Baseline (无Pseudo Query)
```
train_a0_baseline.py
└── external/Deformable-DETR/
    ├── models/deformable_detr.py  # 标准DETR
    ├── models/matcher.py
    └── util/misc.py
```

**说明**: 纯标准Deformable DETR，不使用任何实验9自定义模块。

### A2: Teacher Proposals → Pseudo Query
```
train_pseudo_query.py (exp_type='A2')
├── external/Deformable-DETR/
│   ├── models/deformable_detr.py  # 标准DETR
│   └── models/matcher.py
└── exp9/models/
    ├── heatmap_query_gen.py       # TeacherQueryGenerator
    └── query_injection.py         # QueryMixer
```

**说明**: 使用GT boxes作为teacher，生成pseudo queries。

### A3: Heatmap → Pseudo Query (核心方法)
```
train_pseudo_query.py (exp_type='A3')
├── external/Deformable-DETR/
│   ├── models/deformable_detr.py  # 标准DETR
│   └── models/matcher.py
├── exp9/models/
│   ├── heatmap_query_gen.py       # HeatmapQueryGenerator
│   └── query_injection.py         # QueryMixer
└── exp9/datasets/
    └── dior_with_heatmap.py       # 热图数据集
```

**说明**: 从vv-attention热图生成pseudo queries。

### B1/B2: 证伪实验
```
train_pseudo_query.py (exp_type='B1' or 'B2')
├── external/Deformable-DETR/
│   └── models/deformable_detr.py
└── exp9/models/
    ├── heatmap_query_gen.py       # Random/Shuffled query生成
    └── query_injection.py
```

---

## 📥 如何获取依赖

### 方法1: 克隆官方Deformable DETR (推荐)

```bash
cd /path/to/RemoteCLIP-main/external/

# 克隆官方仓库
git clone https://github.com/fundamentalvision/Deformable-DETR.git

# 编译CUDA算子
cd Deformable-DETR/models/ops
bash make.sh

# 验证
python test.py
```

### 方法2: 使用项目提供的副本

如果`external/Deformable-DETR/`已经存在，直接编译算子:

```bash
cd external/Deformable-DETR/models/ops
bash make.sh
python test.py
```

---

## ✅ 依赖检查清单

在运行实验9之前，请确认以下文件存在:

### 必需的Deformable DETR文件
- [ ] `external/Deformable-DETR/models/__init__.py`
- [ ] `external/Deformable-DETR/models/deformable_detr.py`
- [ ] `external/Deformable-DETR/models/deformable_transformer.py`
- [ ] `external/Deformable-DETR/models/matcher.py`
- [ ] `external/Deformable-DETR/models/backbone.py`
- [ ] `external/Deformable-DETR/models/position_encoding.py`
- [ ] `external/Deformable-DETR/models/segmentation.py`
- [ ] `external/Deformable-DETR/models/ops/` (整个目录)
- [ ] `external/Deformable-DETR/util/misc.py`
- [ ] `external/Deformable-DETR/util/box_ops.py`

### 必需的实验9自定义文件
- [ ] `models/deformable_detr_pseudo.py`
- [ ] `models/heatmap_query_gen.py`
- [ ] `models/query_injection.py`
- [ ] `datasets/dior_deformable.py`
- [ ] `datasets/dior_with_heatmap.py`

### 编译检查
- [ ] CUDA算子编译成功 (`python models/ops/test.py`)

### 自动验证脚本

```bash
# 运行环境验证脚本
bash scripts/verify_environment.sh

# 检查包含:
# 1. Deformable DETR文件完整性
# 2. CUDA算子编译状态
# 3. Python依赖安装
# 4. 数据集路径
```

---

## 🔧 常见问题

### Q1: 为什么不直接把Deformable DETR代码放到exp9里?

**A**: 
1. **版权原因**: Deformable DETR是独立项目，有自己的许可证
2. **代码复用**: 其他实验也可能用到Deformable DETR
3. **更新维护**: 方便跟踪官方更新
4. **清晰分离**: 明确区分"标准实现"和"实验9改动"

### Q2: 如果没有`external/Deformable-DETR/`怎么办?

**A**: 按照"方法1"克隆官方仓库:
```bash
cd external/
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR/models/ops
bash make.sh
```

### Q3: CUDA算子编译失败怎么办?

**A**: 检查:
1. CUDA版本是否匹配 (需要CUDA 11.3+)
2. PyTorch版本是否匹配 (需要PyTorch 1.10+)
3. 编译日志中的具体错误信息
4. 参考`external/Deformable-DETR/README.md`

### Q4: 可以只复制需要的文件吗?

**A**: 可以，最小依赖集合为:
```
external/Deformable-DETR/
├── models/
│   ├── __init__.py
│   ├── deformable_detr.py
│   ├── deformable_transformer.py
│   ├── matcher.py
│   ├── backbone.py
│   ├── position_encoding.py
│   ├── segmentation.py
│   └── ops/ (完整目录)
└── util/
    ├── misc.py
    └── box_ops.py
```

但推荐完整克隆，以避免遗漏间接依赖。

---

## 📚 参考资料

### Deformable DETR官方资源
- **论文**: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)
- **官方仓库**: https://github.com/fundamentalvision/Deformable-DETR
- **许可证**: Apache License 2.0

### 实验9相关文档
- `README.md`: 实验9总体说明
- `EXPERIMENT_CHECKLIST.md`: 实验清单
- `FILES_INVENTORY.md`: 文件清单
- `SETUP_SUMMARY.md`: 环境配置

---

## 📝 版本信息

- **Deformable DETR版本**: v1.0 (2020年官方发布)
- **实验9版本**: v1.0
- **文档更新日期**: 2026-01-29
- **维护者**: Exp9 Team

---

## ⚠️ 重要提示

1. **必须编译CUDA算子**: Deformable DETR无法在纯CPU模式下运行
2. **路径设置**: 训练脚本中已自动添加`external/Deformable-DETR`到`sys.path`
3. **版本兼容**: 确保PyTorch 1.10+ 和 CUDA 11.3+
4. **GitHub上传**: 如果要上传到GitHub，建议使用git submodule管理`external/Deformable-DETR`

---

**总结**: 实验9依赖标准Deformable DETR代码库，请确保`external/Deformable-DETR/`目录完整且CUDA算子编译成功。✅

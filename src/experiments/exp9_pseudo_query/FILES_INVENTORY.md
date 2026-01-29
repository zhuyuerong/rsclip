# Exp9 Pseudo Query 文件清单

**生成时间**: 2026-01-29  
**总文件数**: 29个关键文件

---

## 📂 目录结构

```
src/experiments/exp9_pseudo_query/
├── configs/                    # 实验配置 (2个)
│   ├── experiment_config.py
│   └── experiment_config_v2.py
├── datasets/                   # 数据集 (3个)
│   ├── __init__.py
│   ├── dior_deformable.py
│   └── dior_with_heatmap.py
├── models/                     # 模型模块 (4个)
│   ├── __init__.py
│   ├── deformable_detr_pseudo.py
│   ├── heatmap_query_gen.py
│   └── query_injection.py
├── scripts/                    # 脚本 (9个)
│   ├── compare_experiments.py
│   ├── run_a0.sh
│   ├── run_a2_teacher.sh
│   ├── run_a3_heatmap.sh
│   ├── run_b1_random.sh
│   ├── run_b2_shuffled.sh
│   ├── setup_env.sh
│   ├── train_a0_baseline.py
│   ├── train_pseudo_query.py
│   └── verify_environment.sh
├── utils/                      # 工具 (2个)
│   ├── check_heatmap_format.py
│   └── run_manager.py
├── EXPERIMENT_CHECKLIST.md     # 实验完整清单
├── FILES_INVENTORY.md          # 本文件
├── NEXT_STEPS.md               # 4周实验计划
├── QUICK_REFERENCE.md          # 快速参考
├── README.md                   # 项目文档
├── requirements.txt            # 依赖清单
├── SETUP_SUMMARY.md            # 环境配置总结
└── test_modules.py             # 单元测试
```

---

## 📋 文件详细清单

### 1. 配置文件 (2个)

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `configs/experiment_config.py` | 298 | 旧版配置 (参考) | ✅ |
| `configs/experiment_config_v2.py` | 627 | 新版配置 (完整消融) | ✅ |

**用途**: 定义所有实验的配置参数，支持A0-D4全部实验

---

### 2. 数据集模块 (3个)

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `datasets/__init__.py` | ~50 | 数据集导出 | ✅ |
| `datasets/dior_deformable.py` | ~300 | DIOR基础数据集 | ✅ |
| `datasets/dior_with_heatmap.py` | ~400 | DIOR+热图数据集 | ✅ |

**功能**:
- 加载DIOR VOC格式标注
- 转换为Deformable DETR输入格式
- 集成热图生成/加载
- 支持数据增强和transforms

---

### 3. 模型模块 (4个)

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `models/__init__.py` | ~20 | 模型导出 | ✅ |
| `models/heatmap_query_gen.py` | ~661 | Q-Gen模块 | ✅ |
| `models/query_injection.py` | ~456 | Q-Use模块 | ✅ |
| `models/deformable_detr_pseudo.py` | ~200 | 包装模型 | ✅ |

**核心功能**:

#### `heatmap_query_gen.py` (Q-Gen)
- `PositionalEncoding2D`: 2D位置编码
- `HeatmapQueryGenerator`: 热图→pseudo query
- `TeacherQueryGenerator`: Teacher boxes→pseudo query
- 支持3种pool模式: mean/heatmap_weighted/attn_pool

#### `query_injection.py` (Q-Use)
- `QueryMixer`: Pseudo query与learnable query混合
- `QueryAlignmentLoss`: Query对齐loss
- `AttentionPriorLoss`: Attention先验loss
- 支持4种混合模式: replace/concat/ratio/attention

---

### 4. 训练脚本 (9个)

#### Shell脚本 (6个)

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `scripts/setup_env.sh` | ~80 | 环境变量设置 | ✅ |
| `scripts/run_a0.sh` | ~60 | A0 Baseline | ✅ |
| `scripts/run_a2_teacher.sh` | ~70 | A2 Teacher | ✅ |
| `scripts/run_a3_heatmap.sh` | ~80 | A3 Heatmap (核心) | ✅ |
| `scripts/run_b1_random.sh` | ~50 | B1 Random | ✅ |
| `scripts/run_b2_shuffled.sh` | ~60 | B2 Shuffled | ✅ |

#### Python脚本 (3个)

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `scripts/train_a0_baseline.py` | ~700 | A0训练脚本 | ✅ 运行中 |
| `scripts/train_pseudo_query.py` | ~1000 | A2/A3/B1/B2统一训练 | ✅ |
| `scripts/compare_experiments.py` | ~350 | 实验对比分析 | ✅ |
| `scripts/verify_environment.sh` | ~250 | 环境验证 | ✅ |

---

### 5. 工具模块 (2个)

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `utils/run_manager.py` | ~400 | 训练管理器 | ✅ |
| `utils/check_heatmap_format.py` | ~150 | 热图格式验证 | ✅ |

**功能**:
- `run_manager.py`: 可审计训练协议，记录所有超参和环境
- `check_heatmap_format.py`: 验证热图格式与HeatmapQueryGenerator兼容性

---

### 6. 测试文件 (1个)

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `test_modules.py` | ~300 | 单元测试 | ✅ |

**测试内容**:
- HeatmapQueryGenerator
- QueryMixer
- QueryAlignmentLoss
- AttentionPriorLoss

---

### 7. 文档文件 (6个)

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `README.md` | ~346 | 项目文档 | ✅ |
| `NEXT_STEPS.md` | ~250 | 4周实验计划 | ✅ |
| `EXPERIMENT_CHECKLIST.md` | ~600 | 实验完整清单 | ✅ |
| `SETUP_SUMMARY.md` | ~400 | 环境配置总结 | ✅ |
| `QUICK_REFERENCE.md` | ~150 | 快速参考 | ✅ |
| `FILES_INVENTORY.md` | - | 本文件 | ✅ |

---

### 8. 依赖文件 (1个)

| 文件 | 行数 | 说明 | 状态 |
|------|------|------|------|
| `requirements.txt` | ~40 | Python依赖清单 | ✅ |

**主要依赖**:
- torch==1.10.1+cu113
- torchvision==0.11.2+cu113
- numpy==1.24.4
- opencv-python==4.8.0.74
- pycocotools
- lxml

---

## 📊 代码统计

### 按类型统计

| 类型 | 数量 | 总行数 (估算) |
|------|------|---------------|
| Python代码 | 14 | ~5,500 |
| Shell脚本 | 6 | ~500 |
| Markdown文档 | 6 | ~2,000 |
| 配置文件 | 1 | ~40 |
| **总计** | **27** | **~8,040** |

### 按功能统计

| 功能模块 | 文件数 | 说明 |
|----------|--------|------|
| 核心模型 | 4 | Q-Gen + Q-Use |
| 数据集 | 3 | DIOR + 热图 |
| 训练脚本 | 9 | A0/A2/A3/B1/B2 + 工具 |
| 配置 | 2 | 实验配置 |
| 工具 | 2 | 管理器 + 验证 |
| 测试 | 1 | 单元测试 |
| 文档 | 6 | 完整文档 |

---

## 🔗 依赖关系

### 核心依赖链

```
train_pseudo_query.py
├── models/heatmap_query_gen.py
│   └── torch.nn
├── models/query_injection.py
│   └── torch.nn
├── datasets/dior_with_heatmap.py
│   ├── datasets/dior_deformable.py
│   └── src/competitors/clip_methods/surgeryclip/
└── external/Deformable-DETR/
    ├── models/deformable_detr.py
    └── models/ops/ (CUDA)
```

### 外部依赖

```
external/
├── Deformable-DETR/          # Deformable DETR代码库
│   ├── models/
│   │   ├── deformable_detr.py
│   │   ├── matcher.py
│   │   └── ops/              # CUDA算子 ✅ 已编译
│   ├── util/
│   └── datasets/
└── Pseudo-Q/                 # Pseudo-Q参考代码
```

---

## ✅ 完整性检查

### 必需文件 (全部存在)

- [x] 模型模块: 4/4 ✅
- [x] 数据集模块: 3/3 ✅
- [x] 训练脚本: 9/9 ✅
- [x] 配置文件: 2/2 ✅
- [x] 工具脚本: 2/2 ✅
- [x] 文档文件: 6/6 ✅
- [x] 依赖清单: 1/1 ✅

### 外部依赖 (全部就绪)

- [x] Deformable DETR ✅
- [x] CUDA算子编译 ✅
- [x] DIOR数据集 ✅
- [x] RemoteCLIP权重 ✅

---

## 🎯 使用指南

### 新手入门
1. 阅读 `QUICK_REFERENCE.md` - 快速上手
2. 运行 `scripts/verify_environment.sh` - 验证环境
3. 查看 `EXPERIMENT_CHECKLIST.md` - 了解实验流程

### 运行实验
1. 使用 `scripts/run_*.sh` - 运行各个实验
2. 参考 `SETUP_SUMMARY.md` - 配置说明
3. 查看 `README.md` - 详细文档

### 调试问题
1. 检查 `EXPERIMENT_CHECKLIST.md` - 故障排查
2. 运行 `scripts/verify_environment.sh` - 环境检查
3. 查看训练日志 - `outputs/exp9_pseudo_query/*.log`

### 修改代码
1. 模型修改: `models/`
2. 数据集修改: `datasets/`
3. 配置修改: `configs/experiment_config_v2.py`

---

## 📝 维护日志

| 日期 | 更新内容 | 文件数变化 |
|------|----------|------------|
| 2026-01-27 | 初始创建 | +15 |
| 2026-01-28 | 添加数据集和热图支持 | +3 |
| 2026-01-29 | 完成A0/A2/A3/B1/B2脚本 | +9 |
| 2026-01-29 | 添加文档和工具 | +6 |

---

## 🔄 版本信息

- **项目版本**: Exp9 v1.0
- **代码版本**: 基于Deformable DETR v1.0
- **最后更新**: 2026-01-29
- **维护者**: Exp9 Team

---

**总结**: 所有必需文件已准备完毕，环境配置完成，A0实验运行中。✅

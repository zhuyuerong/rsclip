# Exp9 Pseudo Query 实验环境配置总结

**最后更新**: 2026-01-29

---

## ✅ 已完成的准备工作

### 1. 代码模块 (14个Python文件)

#### 核心模型
- ✅ `models/heatmap_query_gen.py` - Q-Gen模块 (热图→query)
- ✅ `models/query_injection.py` - Q-Use模块 (query混合+loss)
- ✅ `models/deformable_detr_pseudo.py` - 包装模型

#### 数据集
- ✅ `datasets/dior_deformable.py` - DIOR基础数据集
- ✅ `datasets/dior_with_heatmap.py` - DIOR+热图数据集
- ✅ `datasets/__init__.py` - 数据集导出

#### 配置
- ✅ `configs/experiment_config_v2.py` - 完整实验配置 (支持A0-D4所有消融)
- ✅ `configs/experiment_config.py` - 旧版配置 (参考)

#### 工具
- ✅ `utils/run_manager.py` - 训练管理器
- ✅ `utils/check_heatmap_format.py` - 热图格式验证
- ✅ `test_modules.py` - 单元测试

### 2. 训练脚本 (6个Shell脚本 + 2个Python脚本)

#### Shell脚本
- ✅ `scripts/setup_env.sh` - 环境变量设置
- ✅ `scripts/run_a0.sh` - A0 Baseline
- ✅ `scripts/run_a2_teacher.sh` - A2 Teacher proposals
- ✅ `scripts/run_a3_heatmap.sh` - A3 Heatmap pseudo (核心)
- ✅ `scripts/run_b1_random.sh` - B1 Random query (证伪)
- ✅ `scripts/run_b2_shuffled.sh` - B2 Shuffled heatmap (证伪)

#### Python脚本
- ✅ `scripts/train_a0_baseline.py` - A0训练脚本
- ✅ `scripts/train_pseudo_query.py` - A2/A3/B1/B2统一训练脚本

### 3. 工具脚本
- ✅ `scripts/verify_environment.sh` - 环境验证脚本
- ✅ `scripts/compare_experiments.py` - 实验对比分析

### 4. 文档
- ✅ `README.md` - 项目文档
- ✅ `NEXT_STEPS.md` - 4周实验计划
- ✅ `EXPERIMENT_CHECKLIST.md` - 实验完整清单
- ✅ `requirements.txt` - 依赖清单
- ✅ `SETUP_SUMMARY.md` - 本文件

### 5. 外部依赖
- ✅ Deformable DETR代码库 (`external/Deformable-DETR/`)
- ✅ Deformable Attention CUDA算子编译完成
- ✅ Pseudo-Q参考代码 (`external/Pseudo-Q/`)

### 6. 数据准备
- ✅ DIOR数据集 (`datasets/DIOR/`)
  - JPEGImages: ~17,591张图像
  - Annotations: ~17,591个XML标注
  - ImageSets: train/val/test划分
- ✅ 热图格式确认 (SurgeryCLIP + CAL scene_neg)
- ⏳ 热图缓存 (可选，支持在线生成)

### 7. 环境配置
- ✅ Conda环境: `samrs`
- ✅ PyTorch 1.10.1 + CUDA 11.3
- ✅ 所有依赖包安装
- ✅ 环境变量配置脚本

---

## 📊 当前实验状态

| 实验 | 状态 | 进度 | 备注 |
|------|------|------|------|
| A0 | 🟢 运行中 | Epoch 1/50 | PID: 84120 |
| A2 | ⏳ 待运行 | - | 等待A0完成 |
| A3 | ⏳ 待运行 | - | 核心方法 |
| B1 | ⏳ 待运行 | - | 证伪实验 |
| B2 | ⏳ 待运行 | - | 证伪实验 |

---

## 🚀 快速开始

### 环境验证
```bash
conda activate samrs
bash src/experiments/exp9_pseudo_query/scripts/verify_environment.sh
```

### 运行实验
```bash
# A0: Baseline (已运行)
bash src/experiments/exp9_pseudo_query/scripts/run_a0.sh

# A2: Teacher proposals
bash src/experiments/exp9_pseudo_query/scripts/run_a2_teacher.sh

# A3: Heatmap pseudo (核心)
bash src/experiments/exp9_pseudo_query/scripts/run_a3_heatmap.sh

# B1/B2: 证伪实验
bash src/experiments/exp9_pseudo_query/scripts/run_b1_random.sh
bash src/experiments/exp9_pseudo_query/scripts/run_b2_shuffled.sh
```

### 监控训练
```bash
# 查看日志
tail -f outputs/exp9_pseudo_query/a0_training.log

# 查看GPU
watch -n 1 nvidia-smi

# 检查进程
ps aux | grep train
```

### 对比分析
```bash
# 等所有实验完成后
python src/experiments/exp9_pseudo_query/scripts/compare_experiments.py \
    --exp_dirs outputs/exp9_pseudo_query/a0_* \
               outputs/exp9_pseudo_query/a2_* \
               outputs/exp9_pseudo_query/a3_*
```

---

## 📦 依赖清单

### 核心包
```
torch==1.10.1+cu113
torchvision==0.11.2+cu113
numpy==1.24.4
opencv-python==4.8.0.74
Pillow==9.5.0
matplotlib==3.7.5
scipy==1.10.1
tqdm==4.67.1
pycocotools
lxml
```

### 安装方法
```bash
conda activate samrs
pip install -r src/experiments/exp9_pseudo_query/requirements.txt
```

---

## 📁 目录结构

```
src/experiments/exp9_pseudo_query/
├── configs/              # 实验配置
├── datasets/             # 数据集加载器
├── models/               # 模型模块
├── scripts/              # 训练和工具脚本
├── utils/                # 工具函数
├── *.md                  # 文档
└── requirements.txt      # 依赖清单

external/
├── Deformable-DETR/      # Deformable DETR代码库
└── Pseudo-Q/             # Pseudo-Q参考代码

datasets/
└── DIOR/                 # DIOR数据集

outputs/
├── exp9_pseudo_query/    # 实验输出
└── heatmap_cache/        # 热图缓存

checkpoints/
└── RemoteCLIP-ViT-B-32.pt  # RemoteCLIP权重
```

---

## 🔑 关键配置参数

### A0 Baseline
```python
num_queries = 300          # 全部learnable
epochs = 50
batch_size = 2
lr = 2e-4
```

### A2 Teacher
```python
num_pseudo_queries = 100   # 从GT boxes
num_learnable_queries = 200
mix_mode = 'concat'
pool_mode = 'heatmap_weighted'
```

### A3 Heatmap (核心)
```python
num_pseudo_queries = 100   # 从热图top-k
num_learnable_queries = 200
mix_mode = 'concat'
pool_mode = 'heatmap_weighted'
heatmap_source = 'SurgeryCLIP + CAL(scene_neg)'
```

### B1 Random
```python
num_pseudo_queries = 100   # 随机生成
num_learnable_queries = 200
mix_mode = 'concat'
```

### B2 Shuffled
```python
num_pseudo_queries = 100   # 打乱热图
num_learnable_queries = 200
mix_mode = 'concat'
pool_mode = 'heatmap_weighted'
```

---

## 📊 评估指标

### 主要指标
- **mAP@0.5**: 主要检测性能
- **Recall@100**: 前100个预测的召回率
- **AP_small**: 小目标性能

### 对比维度
- **Early epoch** (1, 5, 10): 收敛速度
- **Mid epoch** (20, 30): 稳定性
- **Final epoch** (50): 最终性能

### 成功判据
| 实验 | 判据 |
|------|------|
| A2 | Epoch 10 Recall@100 > A0 + 3~10% |
| A3 | Recall@100 ≥ A2，不全指标劣于A2 |
| B1 | 明显差于A2/A3 |
| B2 | 明显差于A3 |

---

## 🐛 故障排查

### 常见问题
1. **ImportError: libc10.so**
   ```bash
   export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
   ```

2. **CUDA out of memory**
   ```bash
   # 减小batch_size
   --batch_size 1
   ```

3. **Deformable Attention编译失败**
   ```bash
   cd external/Deformable-DETR/models/ops
   bash make.sh
   ```

4. **热图加载失败**
   ```bash
   # 使用在线生成
   --generate_heatmap_on_fly
   ```

### 验证命令
```bash
# 完整环境验证
bash scripts/verify_environment.sh

# 测试数据集
python -c "from src.experiments.exp9_pseudo_query.datasets import build_dior_dataset; print('OK')"

# 测试热图
python src/experiments/exp9_pseudo_query/utils/check_heatmap_format.py

# 测试模块
python src/experiments/exp9_pseudo_query/test_modules.py
```

---

## 📈 下一步计划

### Week 1 (当前)
- [x] 环境准备 ✅
- [x] A0 baseline启动 ✅
- [ ] A0完成并分析
- [ ] A2 teacher运行

### Week 2
- [ ] A3 heatmap运行
- [ ] B1/B2证伪实验
- [ ] Phase A/B结果分析

### Week 3
- [ ] Phase C消融实验 (K/Pool/Use)
- [ ] 详细分析和可视化

### Week 4
- [ ] 补充实验
- [ ] 论文图表生成
- [ ] 文档整理

---

## 📚 参考资料

- [README.md](README.md) - 项目概览
- [NEXT_STEPS.md](NEXT_STEPS.md) - 4周实验计划
- [EXPERIMENT_CHECKLIST.md](EXPERIMENT_CHECKLIST.md) - 实验完整清单
- [experiment_config_v2.py](configs/experiment_config_v2.py) - 完整配置
- [Deformable DETR论文](https://arxiv.org/abs/2010.04159)

---

## 📞 联系方式

如有问题，请查看:
1. [EXPERIMENT_CHECKLIST.md](EXPERIMENT_CHECKLIST.md) - 完整清单
2. [README.md](README.md) - 项目文档
3. 训练日志: `outputs/exp9_pseudo_query/*.log`

---

**状态**: ✅ 环境准备完成，A0运行中
**下一步**: 等待A0完成 → 运行A2 → 运行A3

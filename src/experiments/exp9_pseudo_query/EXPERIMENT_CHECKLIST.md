# Exp9 Pseudo Query 实验完整清单

## 📋 实验概览

| 实验 | 类型 | 目的 | 预期结果 | 状态 |
|------|------|------|----------|------|
| **A0** | Baseline | 对照组 | 基础性能 | ✅ 运行中 |
| **A2** | Teacher | 管线自检 | 更快收敛 | ⏳ 待运行 |
| **A3** | Heatmap | 核心方法 | 小目标提升 | ⏳ 待运行 |
| **B1** | Random | 证伪 | 显著差于A3 | ⏳ 待运行 |
| **B2** | Shuffled | 证伪 | 显著差于A3 | ⏳ 待运行 |

---

## 🗂️ 文件结构

```
src/experiments/exp9_pseudo_query/
├── configs/
│   ├── experiment_config.py          # 旧版配置 (参考)
│   └── experiment_config_v2.py       # 新版配置 (完整消融)
├── datasets/
│   ├── __init__.py                   # 数据集导出
│   ├── dior_deformable.py            # DIOR基础数据集
│   └── dior_with_heatmap.py          # DIOR+热图数据集
├── models/
│   ├── __init__.py
│   ├── heatmap_query_gen.py          # Q-Gen模块 (热图→query)
│   ├── query_injection.py            # Q-Use模块 (query混合+loss)
│   └── deformable_detr_pseudo.py     # 包装模型 (未使用)
├── scripts/
│   ├── setup_env.sh                  # 环境设置
│   ├── train_a0_baseline.py          # A0训练脚本
│   ├── train_pseudo_query.py         # A2/A3/B1/B2统一训练脚本
│   ├── run_a0.sh                     # A0运行脚本
│   ├── run_a2_teacher.sh             # A2运行脚本
│   ├── run_a3_heatmap.sh             # A3运行脚本
│   ├── run_b1_random.sh              # B1运行脚本
│   └── run_b2_shuffled.sh            # B2运行脚本
├── utils/
│   ├── run_manager.py                # 训练管理器
│   └── check_heatmap_format.py       # 热图格式检查
├── test_modules.py                   # 模块单元测试
├── requirements.txt                  # 依赖清单
├── README.md                         # 项目文档
├── NEXT_STEPS.md                     # 实验计划
└── EXPERIMENT_CHECKLIST.md           # 本文件

external/
├── Deformable-DETR/                  # Deformable DETR代码库
│   ├── models/ops/                   # CUDA算子 (已编译✅)
│   └── ...
└── Pseudo-Q/                         # Pseudo-Q参考代码

outputs/
├── exp9_pseudo_query/
│   ├── a0_baseline_20260129_100049/  # A0输出目录
│   │   ├── config.json
│   │   ├── log.txt
│   │   └── checkpoints/
│   └── a0_training.log               # A0训练日志
└── heatmap_cache/
    ├── dior_trainval/                # 训练集热图缓存
    └── dior_test/                    # 测试集热图缓存
```

---

## 📦 依赖项清单

### 1. Python环境
- **Conda环境**: `samrs`
- **Python版本**: 3.8
- **CUDA版本**: 11.3
- **PyTorch**: 1.10.1+cu113
- **TorchVision**: 0.11.2+cu113

### 2. 核心依赖包
```bash
# 安装命令
conda activate samrs
pip install -r requirements.txt
```

主要包:
- `torch==1.10.1+cu113`
- `torchvision==0.11.2+cu113`
- `numpy==1.24.4`
- `opencv-python==4.8.0.74`
- `Pillow==9.5.0`
- `matplotlib==3.7.5`
- `scipy==1.10.1`
- `tqdm==4.67.1`
- `pycocotools`
- `lxml` (用于DIOR XML解析)

### 3. CUDA算子编译
```bash
# Deformable DETR的Multi-Scale Deformable Attention
cd external/Deformable-DETR/models/ops
bash make.sh
```

**状态**: ✅ 已编译完成

---

## 💾 数据集清单

### 1. DIOR数据集
**路径**: `datasets/DIOR/`

**结构**:
```
datasets/DIOR/
├── JPEGImages/              # 图像文件
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── Annotations/             # VOC XML标注
│   ├── 00001.xml
│   ├── 00002.xml
│   └── ...
└── ImageSets/
    └── Main/
        ├── train.txt        # 训练集ID列表
        ├── val.txt          # 验证集ID列表
        └── test.txt         # 测试集ID列表
```

**类别数**: 20类
**训练集**: ~11,725张
**验证集**: ~2,933张
**测试集**: ~2,933张

### 2. 热图缓存 (A3/B2需要)
**路径**: `outputs/heatmap_cache/`

**生成方式**:
- **在线生成**: `--generate_heatmap_on_fly` (慢但省空间)
- **离线缓存**: 预先生成并保存 (快但占空间)

**热图格式**:
- 类型: `numpy.ndarray`
- 形状: `(H, W)` (原图尺寸)
- 数据类型: `float32`
- 值域: `[0, 1]`
- 来源: SurgeryCLIP baseline + CAL(scene_neg)

### 3. 预训练权重
**RemoteCLIP**: `checkpoints/RemoteCLIP-ViT-B-32.pt`
- 用于生成热图 (A3/B2实验)

**Deformable DETR Backbone**: 
- ResNet50预训练权重 (自动从PyTorch Hub下载)

---

## ⚙️ 环境配置

### 1. 环境变量设置
```bash
# 方式1: 使用setup_env.sh
source src/experiments/exp9_pseudo_query/scripts/setup_env.sh

# 方式2: 手动设置
export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main:/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/external/Deformable-DETR:${PYTHONPATH}"
```

### 2. 验证环境
```bash
# 检查CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# 检查Deformable Attention
python -c "from models.ops.modules import MSDeformAttn; print('✅ Deformable Attention OK')"

# 检查数据集
python -c "from src.experiments.exp9_pseudo_query.datasets import build_dior_dataset; print('✅ DIOR Dataset OK')"

# 检查热图生成
python src/experiments/exp9_pseudo_query/utils/check_heatmap_format.py
```

---

## 🚀 运行实验

### Phase A: MVP可行性

#### A0: Baseline (无pseudo query)
```bash
# 状态: ✅ 运行中 (PID: 84120)
bash src/experiments/exp9_pseudo_query/scripts/run_a0.sh

# 监控
tail -f outputs/exp9_pseudo_query/a0_training.log
```

**配置**:
- Epochs: 50
- Batch size: 2
- Learning rate: 2e-4
- Num queries: 300 (全部learnable)
- Eval epochs: [1, 5, 10, 20, 30, 40, 50]

**成功判据**:
- Loss稳定下降
- Epoch 1 Recall@100 > 0.02
- Epoch 50 mAP@0.5 > 0.10

---

#### A2: Teacher Proposals → Pseudo Query
```bash
bash src/experiments/exp9_pseudo_query/scripts/run_a2_teacher.sh
```

**配置**:
- Pseudo queries: 100 (从GT boxes)
- Learnable queries: 200
- Mix mode: concat
- Pool mode: heatmap_weighted

**成功判据**:
- Epoch 10 Recall@100 > A0 + 3~10%
- 收敛速度明显快于A0
- 不应该比A0差

**如果A2失败**:
1. 检查query_embed结构 (content + pos)
2. 检查坐标映射 (原图 → feature map)
3. 检查mix_mode是否正确

---

#### A3: Heatmap → Pseudo Query (核心方法⭐⭐)
```bash
bash src/experiments/exp9_pseudo_query/scripts/run_a3_heatmap.sh
```

**配置**:
- Pseudo queries: 100 (从热图top-k)
- Learnable queries: 200
- Mix mode: concat
- Pool mode: heatmap_weighted
- Heatmap: SurgeryCLIP + CAL(scene_neg)

**成功判据**:
- Recall@100 ≥ A2
- 不能全指标明显劣于A2
- 密集小目标(ship/vehicle)提升明显

**如果A3失败**:
1. 检查热图坐标系对齐
2. 检查top-k采样是否分散 (需要NMS)
3. 检查pool_window大小

---

### Phase B: 证伪实验

#### B1: Random Query
```bash
bash src/experiments/exp9_pseudo_query/scripts/run_b1_random.sh
```

**预期**: 明显差于A2/A3

**如果B1≈A3**: 说明A3增益只是"多了queries"，方法不成立

---

#### B2: Shuffled Heatmap
```bash
bash src/experiments/exp9_pseudo_query/scripts/run_b2_shuffled.sh
```

**预期**: 明显差于A3

**如果B2不掉**: A3的因果链不成立

---

## 📊 评估指标

### 主要指标
1. **mAP@0.5**: 主要检测性能
2. **mAP@0.5:0.95**: COCO标准
3. **Recall@100**: 前100个预测的召回率
4. **AP_small**: 小目标性能

### 辅助指标
1. **收敛速度**: 达到某阈值的epoch数
2. **训练稳定性**: Loss曲线波动
3. **梯度范数**: 训练健康度

### 对比维度
- **Early epoch** (1, 5, 10): 收敛速度
- **Mid epoch** (20, 30): 稳定性
- **Final epoch** (50): 最终性能

---

## 🔍 调试检查清单

### 训练前检查
- [ ] CUDA可用且版本正确
- [ ] Deformable Attention编译成功
- [ ] DIOR数据集路径正确
- [ ] 热图缓存存在 (A3/B2)
- [ ] 输出目录可写
- [ ] 环境变量设置正确

### 训练中监控
- [ ] Loss稳定下降 (不应NaN/Inf)
- [ ] GPU利用率正常 (>80%)
- [ ] 内存不溢出
- [ ] 梯度范数合理 (<1000)
- [ ] 学习率正常衰减

### 训练后检查
- [ ] 生成了checkpoint
- [ ] log.txt完整
- [ ] config.json保存
- [ ] 评估指标合理

---

## 🐛 常见问题

### 1. ImportError: libc10.so
**解决**: 设置LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### 2. CUDA out of memory
**解决**: 减小batch_size
```bash
--batch_size 1  # 从2降到1
```

### 3. Loss不降/NaN
**可能原因**:
- 学习率太大
- 梯度爆炸 (检查grad_norm)
- 数据标注错误
- Box格式不对 (cxcywh vs xyxy)

### 4. Recall@100很低 (<0.01)
**可能原因**:
- Matcher不工作
- Box坐标归一化错误
- 类别映射错误

### 5. A2/A3不如A0
**可能原因**:
- Pseudo query注入失败
- query_embed格式错误
- 坐标系不对齐
- Mix mode选择不当

---

## 📝 实验日志模板

```markdown
## 实验: A2 Teacher Proposals

**日期**: 2026-01-29
**配置**: 
- K=100, learnable=200, mix=concat
- lr=2e-4, epochs=50

**结果**:
| Epoch | Loss | Recall@100 | mAP@0.5 |
|-------|------|------------|---------|
| 1     | 23.5 | 0.025      | 0.005   |
| 10    | 18.2 | 0.089      | 0.035   |
| 50    | 12.1 | 0.245      | 0.158   |

**对比A0**:
- Epoch 10: Recall +8.5% ✅
- 收敛更快 ✅

**结论**: A2成功，可以进行A3
```

---

## 🎯 下一步

1. **等待A0完成** (~14小时)
2. **分析A0曲线** (loss/recall/mAP)
3. **运行A2** (如果A2失败，先debug)
4. **运行A3** (核心方法)
5. **运行B1/B2** (证伪)
6. **对比分析** (生成表格和曲线)
7. **Phase C消融** (K/Pool/Use)

---

## 📚 参考文档

- [README.md](README.md): 项目概览
- [NEXT_STEPS.md](NEXT_STEPS.md): 4周实验计划
- [experiment_config_v2.py](configs/experiment_config_v2.py): 完整配置
- [Deformable DETR论文](https://arxiv.org/abs/2010.04159)
- [Pseudo-Q论文](https://arxiv.org/abs/xxxx.xxxxx)

---

**最后更新**: 2026-01-29
**维护者**: Exp9 Team

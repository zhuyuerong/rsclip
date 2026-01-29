# Exp9: Pseudo Query for Object Detection

## 📌 核心思想

**Pseudo-Q 本质**: 把弱线索（vv-attention热图/相似度）变成"query/训练信号"，从而让grounding/detection学起来。

> **关键原则**: Phase A 的成功不是看最终 mAP，而是看 **early-epoch 的收敛速度** + **small-object recall** 是否出现结构性差异；**A2 是管线自检，A2 失败则 A3 不得下结论**。

---

## 🔑 三个核心变量 (正交设计)

| 变量 | 含义 | 选项 | 消融实验 |
|------|------|------|----------|
| **Q-Gen** | pseudo query生成方式 | teacher / heatmap / fusion | C1 |
| **Q-Init** | query初始化模式 | replace / concat / ratio / attention | - |
| **Q-Loss** | 额外loss | align(l2/cos/nce) / prior(center/attn) | C4 |

⚠️ **Init 和 Loss 正交**: 消融C4时固定Init，只变Loss

---

## 🏗️ 项目结构

```
exp9_pseudo_query/
├── models/
│   ├── heatmap_query_gen.py      # Q-Gen: heatmap → pseudo queries ✓
│   ├── query_injection.py         # Q-Use: query混合 + loss ✓
│   └── deformable_detr_pseudo.py  # 改进的Deformable DETR
├── configs/
│   ├── experiment_config.py       # 旧版配置
│   └── experiment_config_v2.py    # 新版配置 (正交设计) ✓
├── utils/
│   └── run_manager.py             # 可审计训练协议 ✓
├── test_modules.py                # 模块测试 ✓
└── README.md
```

---

## 📊 Deformable DETR Query处理流程

```
query_embed = nn.Embedding(num_queries, hidden_dim * 2)  # [300, 512]
                         ↓
         torch.split(query_embed, c, dim=-1)
              ↓                    ↓
        tgt: [B,Q,d]         query_pos: [B,Q,d]
        (decoder input)      (positional embed)
                                   ↓
                    reference_points = Linear(query_pos).sigmoid()
                                   ↓
                    reference_points: [B, Q, 2]
```

**关键**: 
- 前半 = **content** (tgt)
- 后半 = **positional embedding**
- reference_points 从 pos 预测

---

## 📋 实验路径与预期现象

### Phase A: MVP可行性

#### A0: Baseline (无pseudo)
| 项目 | 内容 |
|------|------|
| **预期现象** | loss平稳下降; boxes从"全图乱飘"到"目标附近聚集"; Recall@K前5-10 epoch明显上升 |
| **角色** | 对照组 |
| **失败信号** | loss不降 → 检查box normalize/gt format; boxes全在边缘 → matcher错误 |

#### A2: Teacher proposals → pseudo (管线自检 ⭐)
| 项目 | 内容 |
|------|------|
| **预期现象** | ✅ **前期收敛更快** (最重要); 同epoch下Recall@K更早抬头 |
| **目标** | epoch 10时: Recall@0.5 > A0 (提升3~10个点正常); AP_small ≥ A0 |
| **允许** | 后期(50 epoch)趋同 → 说明pseudo主要提供"引导" |
| **失败排查** | ① pseudo只替换content,pos/ref没对齐 ② 坐标映射错(原图vs feature) ③ learnable全被替掉 |

> ⚠️ **A2是系统检查。A2不对，A3没资格下结论。**

#### A3: Heatmap → pseudo (核心方法 ⭐⭐)
| 项目 | 内容 |
|------|------|
| **预期现象** | 比A0更快收敛; 密集小目标(ship/vehicle)Recall上升更明显 |
| **正常情况** | 可能带来FP(背景高响应) → mAP未必立刻涨 |
| **目标** | vs A0: AP_small或Recall@0.5之一有稳定提升; vs A2: 允许略弱但不能全指标劣于A2 |
| **失败排查** | ① heatmap坐标系没对齐(patch vs原图) ② top-k全挤一个连通域(无NMS) ③ pool_window太小/太大 |

---

### Phase B: 证伪实验

#### B1: Random query (必须显著差)
| 项目 | 内容 |
|------|------|
| **预期** | 明显劣于A2/A3 (尤其early epoch); 甚至可能比A0还差 |
| **目的** | 证明不是"多加queries就行" |
| **如果B1≈A3** | 说明A3增益只是"多了queries/训练trick"，不是空间证据 → 方法不成立 |

#### B2: Shuffled heatmap (必须明显掉)
| 项目 | 内容 |
|------|------|
| **预期** | 相对A3有显著下降 (early epoch更明显) |
| **目的** | 证明是"图像相关的空间证据" |
| **如果不掉** | A3的因果链不成立 → reviewer会直接否 |

---

### Phase C: 消融实验

#### C1: Q-Gen来源
- teacher vs heatmap vs fusion
- 期望: fusion ≥ heatmap ≥ teacher (或各有长处)

#### C2: K (query数量) ⚠️
- K = 50 / 100 / 150 / 200 / 300
- **关键**: 固定 total_queries=300, 只变pseudo数量
- 正常曲线: **先升后平/下降 (U型或饱和)**
- 如果单调上涨 → 检查是否真的固定了total

#### C3: Q-Pool方式
| 方式 | 预期 |
|------|------|
| mean | 最弱但最稳 |
| heatmap_weighted | 最稳且最强的默认 |
| attn_pool | 可能更强但更容易不稳(波动/对seed敏感) |

#### C4: Q-Use方式 (阶梯增益)
| 配置 | 预期 |
|------|------|
| init_only | baseline |
| +align (L2) | 小幅稳定提升 (尤其early epoch/small objects) |
| +align+prior | 可能再涨，也可能引入FP或训练不稳 (都正常) |

---

## 🎯 推荐默认参数 (避免踩坑)

```python
# Phase A 第一版参数
backbone = "resnet50"           # 最稳
K = 100                         # 别上来300+
pool_mode = "heatmap_weighted"  # 局部窗口3×3
init_mode = "concat"            # 100 pseudo + 200 learnable
fixed_total_queries = True      # 总数固定300
total_queries = 300

# Loss (先不加)
align_loss_type = "none"
prior_loss_type = "none"
```

---

## ⚠️ 关键注意事项

### 1. 坐标系统对齐
```
heatmap: patch级别 (如16×16 for ViT)
FPN: 多尺度 (stride 8/16/32/64)
→ 需要统一坐标映射到原图尺度
```

### 2. Query结构
```python
# 必须分成 content + pos
pseudo_embed = cat([content, pos], dim=-1)  # [B, K, 2*d]

# 否则decoder会"不用/学不动/发散"
```

### 3. 调试顺序
```
A2 (teacher) → 确保pipeline正确
    ↓ 成功
A3 (heatmap) → 验证热图有效
    ↓ 失败时
检查: query结构 → 坐标映射 → NMS → pool_window
```

---

## 📈 评估指标

| 指标 | 用途 |
|------|------|
| mAP@0.5 | 主指标 |
| AP_small | 小目标性能 (关键) |
| Recall@K | proposal recall |
| 收敛epoch | 达到某阈值的epoch数 |

---

## 🔬 可审计训练协议

每次训练自动记录:

1. **数据来源**: dataset, split, 样本数, ID hash
2. **模型权重**: backbone/detr/clip checkpoint + SHA256
3. **环境**: git commit, pytorch版本, GPU型号
4. **随机性**: seed, deterministic设置
5. **超参**: 完整config dump
6. **训练过程**: loss, metrics at eval_epochs
7. **调试记录**: debug_log.md

使用 `utils/run_manager.py` 自动管理。

---

## 🔧 环境设置

### 必需环境: `samrs`
```bash
# 激活环境
conda activate samrs

# 设置库路径 (重要!)
source scripts/setup_env.sh
```

### 环境信息
| 项目 | 值 |
|------|-----|
| PyTorch | 1.10.1+cu113 |
| CUDA | 11.3 (编译) / 11.8 (运行) |
| GPU | NVIDIA GeForce RTX 4090 |
| Deformable Attn | ✅ 已编译安装 |

---

## 📅 下一步实验计划

### Week 1: 环境准备与A0/A2
- [x] 编译Deformable DETR的CUDA算子 ✅
- [x] 确认热图格式 (SurgeryCLIP + CAL scene_neg) ✅
- [x] 准备DIOR数据集dataloader ✅
- [x] A0 baseline训练脚本 ✅ (正在运行)
- [x] A2/A3/B1/B2训练脚本 ✅
- [ ] 运行A2 teacher pseudo (验证query注入机制)
- [ ] 运行A3 heatmap pseudo (核心方法)

### Week 2: A3与证伪
- [ ] 集成vv-attention热图生成
- [ ] 运行A3 heatmap pseudo
- [ ] 运行B1/B2证伪实验
- [ ] 分析early epoch收敛曲线

### Week 3: 消融实验
- [ ] C1: Q-Gen来源对比
- [ ] C2: K数量消融 (记得固定total!)
- [ ] C3: Pool方式对比
- [ ] C4: Q-Use阶梯增益

### Week 4: 结果整理
- [ ] 生成消融表格
- [ ] 绘制收敛曲线对比图
- [ ] 撰写实验章节初稿

---

## 🔥 热图格式规范

热图来源: **SurgeryCLIP baseline + CAL(scene_neg)**

| 项目 | 值 |
|------|-----|
| 类型 | `numpy.ndarray` |
| dtype | `float32` |
| shape | `(H, W)` 与原图一致 |
| 值域 | `[0, 1]` 已归一化 |
| 坐标系 | 原图像素坐标 |

```python
from src.competitors.clip_methods.surgeryclip.model_wrapper import SurgeryCLIPWrapper
from src.competitors.clip_methods.surgeryclip.cal_config import CALConfig

# 生成热图
cal_config = CALConfig(
    enable_cal=True,
    negative_mode='fixed',
    fixed_negatives=["aerial view", "satellite image", "remote sensing scene"],
    alpha=2.0,
    cal_space='similarity',
)
model = SurgeryCLIPWrapper(
    model_name="surgeryclip",
    checkpoint_path="checkpoints/RemoteCLIP-ViT-B-32.pt",
    use_surgery_single="empty",
    cal_config=cal_config
)
model.load_model()
heatmap = model.generate_heatmap(image, [class_name])  # [H, W]
```

---

## 🚀 运行实验

```bash
# 激活环境
conda activate samrs

# A0: Baseline (无pseudo query)
bash scripts/run_a0.sh

# A2: Teacher proposals → pseudo query
bash scripts/run_a2_teacher.sh

# A3: Heatmap → pseudo query (核心方法)
bash scripts/run_a3_heatmap.sh

# B1: Random query (证伪)
bash scripts/run_b1_random.sh

# B2: Shuffled heatmap (证伪)
bash scripts/run_b2_shuffled.sh
```

### 监控训练

```bash
# 查看日志
tail -f outputs/exp9_pseudo_query/a0_training.log

# 检查GPU
nvidia-smi

# 比较多个实验结果
python scripts/compare_experiments.py --exp_dirs outputs/exp9_pseudo_query/a0* outputs/exp9_pseudo_query/a3*
```

---

## 🔗 参考代码

- Pseudo-Q: `external/Pseudo-Q/`
- Deformable DETR: `external/Deformable-DETR/`
- 配置文件: `configs/experiment_config_v2.py`
- 运行管理: `utils/run_manager.py`
- 热图检查: `utils/check_heatmap_format.py`
- 训练脚本: `scripts/train_pseudo_query.py`

# Experiment2: 全局上下文引导的开放词汇检测（基于RemoteCLIP）

## 🎯 核心创新

本实验基于**RemoteCLIP遥感专用模型**，实现了一个**完全不需要手动负文本的端到端开放词汇检测流程**。核心创新是通过"局部实例"与"全局场景"的自对比来自动学习背景抑制。

### 🔥 关键思想

**无需外部负样本**: 通过全局图像上下文 $I_g$ 作为自动负样本

**全局-局部对比**: 局部特征 $f_m$ 应接近目标文本 $t_c$，远离全局上下文 $I_g$

**上下文门控**: 全局上下文引导局部查询的注意力

## 📐 架构设计

### 四阶段架构

```
输入图像 I
    ↓
┌─────────────────────────────────────────┐
│ Stage1: 特征提取 (CLIP Encoder)        │
│  - 多尺度特征 {F^s}                     │
│  - 全局上下文 I_g (CLS token)           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Stage2: 全局上下文引导的解码器          │
│  For l = 1 to L:                        │
│    1. 文本调制: q̃_m = q_m + W_t·t_c   │
│    2. 上下文门控⭐: q'_m = Gate(q̃_m, I_g) │
│    3. 局部采样: z_m = Attn(q'_m, F^s)  │
│    4. 查询更新: q_m ← Update(q_m, z_m) │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Stage3: 预测与精细化                    │
│  - 分类头: f_m (映射到CLIP空间)         │
│  - 回归头: b_m (边界框预测)             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Stage4: 监督与推理                      │
│  - 全局对比损失⭐: L_gc(f_m, t_c, I_g)  │
│  - 边界框损失: L_box(b_m, b_gt)         │
└─────────────────────────────────────────┘
```

## 🌟 核心模块详解

### 1. 全局对比损失 (Global Contrast Loss) ⭐

这是整个框架的**核心创新**。

#### 数学原理

对于匹配到 GT 的查询 $m$：
- **正对**: $(f_m, t_c)$ — 局部特征 vs 目标文本
- **负对**: $(f_m, I_g)$ — 局部特征 vs 全局上下文

$$
L_{GlobalContrast} = -\log\left[\frac{\exp(\langle f_m, t_c \rangle / \tau)}{\exp(\langle f_m, t_c \rangle / \tau) + \exp(\langle f_m, I_g \rangle / \tau)}\right]
$$

#### 直觉理解

假设我们在检测**飞机**：
- $f_m$: 某个候选框的局部特征（可能是飞机）
- $t_c$: "airplane" 的文本嵌入
- $I_g$: 整张图的全局嵌入（可能是 "天空+跑道+建筑" 的混合表示）

损失函数迫使：
1. $f_m$ **接近** $t_c$（"飞机"）
2. $f_m$ **远离** $I_g$（"天空"主导的全局场景）

这样模型会学会：
- 如果某个区域是飞机，它应该与 "airplane" 文本相似
- 如果某个区域是背景（天空/跑道），它会与全局上下文相似，从而被**自动抑制**

### 2. 上下文门控 (Context Gating) ⭐

使用全局上下文调制局部查询。

#### 两种实现

**方案1: FiLM (Feature-wise Linear Modulation)**
```python
γ, β = MLP(I_g)
q'_m = γ ⊙ q̃_m + β
```
- 优点: 参数少，计算高效
- 适用: 小规模模型

**方案2: Concat + MLP**
```python
q'_m = MLP([q̃_m, I_g])
```
- 优点: 表达能力强
- 适用: 大规模模型

## 📁 文件结构

```
experiment2/
├── config/                         # ✅ 配置文件
│   ├── default_config.py          # ✅ 默认配置（超参数）
│   └── model_config.py            # ✅ 模型架构配置
│
├── stage1_encoder/                # ✅ 阶段一：特征提取
│   ├── clip_image_encoder.py     # ✅ CLIP 图像编码器
│   ├── clip_text_encoder.py      # ✅ CLIP 文本编码器
│   └── global_context_extractor.py # ✅ 全局上下文提取
│
├── stage2_decoder/                # ✅ 阶段二：解码器
│   ├── query_initializer.py      # ✅ 位置查询初始化
│   ├── text_conditioner.py       # ✅ 文本调制
│   └── context_gating.py         # ✅ ⭐上下文门控（核心）
│
├── stage3_prediction/             # ✅ 阶段三：预测
│   ├── classification_head.py    # ✅ 分类头（映射到CLIP空间）
│   └── regression_head.py        # ✅ 回归头（边界框）
│
├── stage4_supervision/            # ✅ 阶段四：监督
│   ├── global_contrast_loss.py   # ✅ ⭐全局对比损失（核心）
│   ├── box_loss.py               # ✅ 边界框损失
│   ├── matcher.py                # ✅ 匈牙利匹配器
│   └── loss_functions.py         # ✅ 总损失函数
│
├── models/                        # ✅ 完整模型
│   └── context_guided_detector.py # ✅ 主模型（整合四阶段）
│
├── inference/                     # ✅ 推理
│   ├── inference_engine.py       # ✅ 推理引擎
│   └── post_processor.py         # ✅ 后处理（NMS）
│
└── README.md                      # 本文档
```

**实现状态**: ✅ 所有核心模块已完成！

## 🚀 快速开始

### 1. 环境准备

```bash
# 已包含在项目中：
# - RemoteCLIP权重: checkpoints/RemoteCLIP-RN50.pt
# - open_clip_torch: remoteclip环境
# - 其他依赖: torch, scipy等

# 如需额外安装：
pip install torch torchvision open_clip_torch scipy
```

### 2. 测试核心模块

```bash
# 测试全局对比损失
python experiment2/stage4_supervision/global_contrast_loss.py

# 测试上下文门控
python experiment2/stage2_decoder/context_gating.py

# 测试主模型
python experiment2/models/context_guided_detector.py
```

### 3. 推理

```bash
python experiment2/inference/inference_engine.py \
    --image assets/airport.jpg \
    --text airplane \
    --threshold 0.5 \
    --output results/detection.jpg
```

## 🧪 实验结果（占位）

### mAP 表格

| 配置 | mAP | AP50 | AP75 |
|------|-----|------|------|
| Full Model | TBD | TBD | TBD |
| - ContextGating | TBD | TBD | TBD |
| - GlobalContrast | TBD | TBD | TBD |

### 可视化示例

（待补充）

## 📊 与 Experiment1 的对比

| 特性 | Experiment1 | Experiment2 |
|------|-------------|-------------|
| **负样本来源** | WordNet 外部词汇（100类） | 全局上下文（自动） |
| **依赖** | 需要预定义负样本 | 无需外部负样本 |
| **扩展性** | 受限于 WordNet 词汇 | 完全开放词汇 |
| **核心创新** | 对比学习 + 词汇增强 | 全局-局部对比 |
| **上下文利用** | 无 | 上下文门控 |

### 核心优势

**Experiment2 的创新点**:
1. **自动负样本**: 无需手动定义负样本类别
2. **上下文感知**: 利用全局场景信息引导检测
3. **更通用**: 适用于任意开放词汇

## 🔧 技术细节

### 超参数

```python
# 模型架构
num_queries = 100          # M，查询数量
num_decoder_layers = 6     # L，解码器层数
d_model = 256              # 模型维度
d_clip = 512               # CLIP 空间维度

# 损失权重
lambda_box = 5.0           # 边界框损失
lambda_global_contrast = 1.0  # 全局对比损失⭐
temperature = 0.07         # 对比学习温度 τ

# 训练
batch_size = 16
learning_rate = 1e-4
num_epochs = 50
```

### 数学符号说明

- $I$: 输入图像
- $F^s$: 多尺度特征图
- $I_g$: 全局上下文嵌入（CLS token）
- $t_c$: 目标类别的文本嵌入
- $q_m$: 第 $m$ 个查询
- $f_m$: 第 $m$ 个查询的局部特征（映射到CLIP空间）
- $b_m$: 第 $m$ 个查询预测的边界框
- $\tau$: 对比学习温度参数

## 📚 论文参考

本实验借鉴了以下论文的思想：
- DETR: End-to-End Object Detection with Transformers
- CLIP: Learning Transferable Visual Models From Natural Language Supervision
- FiLM: Visual Reasoning with a General Conditioning Layer
- InfoNCE: Representation Learning with Contrastive Predictive Coding

## 🔄 未来工作

- [ ] 实现完整的训练流水线
- [ ] 在多个遥感数据集上评估
- [ ] 与 Experiment1 进行定量对比
- [ ] 探索不同的上下文门控机制
- [ ] 研究温度参数 τ 的影响

## 💡 使用建议

1. **调试**: 先用小数据集验证代码正确性
2. **消融**: 运行消融实验了解各组件的贡献
3. **可视化**: 使用 `test_global_contrast.py` 可视化相似度分布
4. **参数调优**: 温度参数 τ 对性能影响较大，建议在 [0.05, 0.1] 范围内调整

## 📞 问题反馈

如有问题，请查看：
1. `config/default_config.py` - 详细的配置说明
2. `stage4_supervision/global_contrast_loss.py` - 核心损失函数的数学推导
3. `stage2_decoder/context_gating.py` - 上下文门控的实现细节

---

**核心文件优先级**:
1. ⭐ `stage4_supervision/global_contrast_loss.py` - 全局对比损失（最核心）
2. ⭐ `stage2_decoder/context_gating.py` - 上下文门控（核心创新）
3. `models/context_guided_detector.py` - 主模型
4. `inference/inference_engine.py` - 推理引擎
5. `scripts/train.py` - 训练脚本

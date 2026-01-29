# 下一步实验计划 (详细版)

## 🎯 总体目标

在4周内完成Pseudo Query方法的MVP验证和消融实验，产出可写入论文的实验结果。

---

## 📅 Week 1: 环境准备与基础实验

### Day 1-2: 环境配置

#### 1.1 编译Deformable DETR CUDA算子
```bash
cd external/Deformable-DETR/models/ops
sh ./make.sh
python test.py  # 应该看到所有checking is True
```

**常见问题**:
- CUDA版本不匹配: 检查`nvcc --version`与PyTorch CUDA版本
- 编译失败: 尝试降级GCC或修改`setup.py`中的编译标志

#### 1.2 准备DIOR数据集
```python
# 需要创建的文件: datasets/dior_dataset.py
# 输出格式:
# - images: [B, 3, H, W]
# - targets: List[Dict] with keys: 'boxes', 'labels'
# - heatmaps: [B, H', W'] (可选，后续加入)
```

#### 1.3 验证热图生成
```python
# 确认你的vv-attention热图输出格式:
# - shape: [B, H, W] 或 [H, W]
# - dtype: float32
# - range: [0, 1] 或可归一化
# - 坐标系: 原图尺度 or patch尺度?
```

**今天要做的确认**:
- [ ] 打印一张热图的shape和value range
- [ ] 可视化热图与原图叠加，确认空间对应正确

### Day 3-4: A0 Baseline

#### 1.4 运行A0: 标准Deformable DETR
```bash
# 目标: 确保基础detection pipeline正确

# 训练命令 (示例)
python train.py \
    --exp_name A0_baseline \
    --dataset DIOR \
    --epochs 50 \
    --batch_size 2 \
    --lr 1e-4 \
    --use_pseudo_query False
```

**检查点**:
- [ ] Epoch 1: loss < 20 (sanity check)
- [ ] Epoch 5: Recall@300 > 0.05
- [ ] Epoch 10: 可视化pred boxes，应该开始聚集到目标附近

**如果失败**:
1. 检查GT box格式 (cxcywh vs xyxy? 归一化?)
2. 检查matcher是否正确计算cost
3. 检查loss计算是否有NaN

### Day 5-7: A2 Teacher Pseudo

#### 1.5 准备Teacher Detector
选择一个:
- Faster R-CNN (torchvision预训练)
- YOLO (ultralytics)
- 用你已有的任何detector

```python
# 需要输出:
# teacher_boxes: [B, N, 4]  # 归一化 (x1,y1,x2,y2) 或 (cx,cy,w,h)
# teacher_scores: [B, N]
```

#### 1.6 运行A2: Teacher → Pseudo Query
```bash
python train.py \
    --exp_name A2_teacher \
    --use_pseudo_query True \
    --pseudo_gen_type teacher \
    --num_pseudo_queries 100 \
    --init_mode concat \
    --total_queries 300
```

**关键检查**:
```python
# 在forward()里打印:
print(f"pseudo_embed shape: {pseudo_embed.shape}")  # 应该是 [B, 100, 512]
print(f"mixed_embed shape: {mixed_embed.shape}")    # 应该是 [B, 300, 512]
print(f"reference_points range: [{ref.min():.3f}, {ref.max():.3f}]")  # 应该在[0,1]
```

**预期结果对比 A0**:
| Metric | A0 @ epoch 10 | A2 @ epoch 10 | 判断 |
|--------|---------------|---------------|------|
| Loss | X | < X | ✓ |
| Recall@300 | Y | > Y | ✓ |
| AP_small | Z | ≥ Z | ✓ |

---

## 📅 Week 2: 核心方法与证伪

### Day 1-3: A3 Heatmap Pseudo

#### 2.1 集成热图生成到数据流
```python
# 在dataset中添加heatmap字段
class DiorDataset:
    def __getitem__(self, idx):
        image, target = self.load_image_and_target(idx)
        
        # 生成或加载预计算的热图
        heatmap = self.generate_heatmap(image)  # [H, W]
        
        return image, target, heatmap
```

#### 2.2 运行A3
```bash
python train.py \
    --exp_name A3_heatmap \
    --use_pseudo_query True \
    --pseudo_gen_type heatmap \
    --pool_mode heatmap_weighted \
    --pool_window 3 \
    --num_pseudo_queries 100
```

**调试步骤** (如果不work):
1. 可视化top-k选点位置，是否分散在目标区域?
2. 检查heatmap坐标到feature坐标的映射
3. 打印pooled features的statistics

### Day 4-5: B1/B2 证伪实验

#### 2.3 B1: Random Query
```python
# 在HeatmapQueryGenerator中添加:
if self.debug_mode == DebugMode.RANDOM_QUERY:
    # 随机坐标
    coords = torch.rand(B, K, 2, device=device)
    # 随机特征
    pooled_features = torch.randn(B, K, self.hidden_dim, device=device)
```

#### 2.4 B2: Shuffled Heatmap
```python
# 在dataloader中添加:
if self.debug_mode == DebugMode.SHUFFLE_HEATMAP:
    # 用其他图片的heatmap
    other_idx = (idx + random.randint(1, len(self)-1)) % len(self)
    heatmap = self.generate_heatmap(self.images[other_idx])
```

**预期结果**:
| Exp | vs A3 Recall | vs A3 AP_small | 结论 |
|-----|--------------|----------------|------|
| B1 | 显著低 | 显著低 | ✓ 不是"多queries"的功劳 |
| B2 | 显著低 | 显著低 | ✓ 是"空间证据"的功劳 |

### Day 6-7: 分析与可视化

#### 2.5 绘制收敛曲线
```python
import matplotlib.pyplot as plt

epochs = [1, 5, 10, 20, 30, 50]
metrics = ['loss', 'recall_300', 'ap_small']

for metric in metrics:
    plt.figure()
    for exp in ['A0', 'A2', 'A3', 'B1', 'B2']:
        plt.plot(epochs, results[exp][metric], label=exp)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f'figures/{metric}_comparison.png')
```

---

## 📅 Week 3: 消融实验

### Day 1-2: C1 Q-Gen来源

```bash
# 运行3个变体
for gen_type in teacher heatmap fusion; do
    python train.py \
        --exp_name C1_qgen_${gen_type} \
        --pseudo_gen_type ${gen_type} \
        ... # 其他参数与A3相同
done
```

**结果表格模板**:
| Q-Gen | AP@0.5 | AP_small | Recall@300 | 收敛epoch |
|-------|--------|----------|------------|-----------|
| teacher | | | | |
| heatmap | | | | |
| fusion | | | | |

### Day 3-4: C2 K消融

⚠️ **关键**: 固定 total_queries=300

```bash
for K in 50 100 150 200 300; do
    python train.py \
        --exp_name C2_K_${K} \
        --num_pseudo_queries ${K} \
        --total_queries 300 \  # 固定!
        --fixed_total_queries True
done
```

**预期曲线形状**:
```
Performance
    ^
    |     ***
    |   **   **
    |  *       *
    | *         *
    |*           *
    +-------------> K
    50  100  150  200  300
    
应该是: 先升后平或下降 (U型/饱和)
如果单调上升: 检查是否真的固定了total
```

### Day 5: C3 Pool消融

```bash
for pool in mean heatmap_weighted attn_pool; do
    python train.py \
        --exp_name C3_pool_${pool} \
        --pool_mode ${pool}
done
```

### Day 6-7: C4 Use消融

```bash
# init_only (baseline)
python train.py --exp_name C4_init_only \
    --align_loss_type none --prior_loss_type none

# +align
python train.py --exp_name C4_plus_align \
    --align_loss_type l2 --align_loss_weight 1.0 \
    --prior_loss_type none

# +align+prior  
python train.py --exp_name C4_plus_align_prior \
    --align_loss_type l2 --align_loss_weight 1.0 \
    --prior_loss_type center --prior_loss_weight 0.5
```

**预期阶梯增益**:
| Use | AP@0.5 | 增益 | 说明 |
|-----|--------|------|------|
| init_only | X | - | baseline |
| +align | X+a | +a | 小幅稳定 |
| +align+prior | X+a+b | +b | 可能涨，也可能不稳 |

---

## 📅 Week 4: 结果整理与论文撰写

### Day 1-2: 生成论文表格

#### 主表: 与baseline对比
| Method | AP@0.5 | AP_small | AP_medium | AP_large | Params | FLOPs |
|--------|--------|----------|-----------|----------|--------|-------|
| Deformable DETR | | | | | | |
| + Pseudo-Q (ours) | | | | | | |

#### 消融表1: Q-Gen
| Q-Gen Source | AP@0.5 | AP_small | Recall@0.5 |
|--------------|--------|----------|------------|
| Teacher proposals | | | |
| Heatmap regions | | | |
| Fusion | | | |

#### 消融表2: K
| K | AP@0.5 | AP_small | Training Time |
|---|--------|----------|---------------|
| 50 | | | |
| 100 | | | |
| ... | | | |

### Day 3-4: 可视化

1. **收敛曲线对比** (Figure 3)
2. **热图 vs pseudo query位置** (Figure 4)
3. **检测结果对比** (Figure 5)
4. **失败案例分析** (Figure 6)

### Day 5-7: 撰写实验章节

```
4. Experiments
4.1 Dataset and Implementation Details
4.2 Main Results
4.3 Ablation Studies
    4.3.1 Query Generation Source
    4.3.2 Number of Pseudo Queries
    4.3.3 Feature Pooling Strategy
    4.3.4 Query Usage Strategy
4.4 Analysis
    4.4.1 Convergence Speed
    4.4.2 Small Object Detection
4.5 Limitations
```

---

## ✅ Checklist (每天检查)

### 跑实验前
- [ ] git status 干净 (或已保存diff)
- [ ] 确认config正确 (打印key_vars)
- [ ] 确认数据路径存在
- [ ] 确认checkpoint路径存在

### 跑实验后
- [ ] 检查loss曲线是否正常
- [ ] 检查sanity checks是否pass
- [ ] 保存manifest.json
- [ ] 记录任何手动调整到debug_log.md

### 对比实验时
- [ ] 确认只改了一个变量
- [ ] 确认seed相同
- [ ] 使用inherit()从base config派生

---

## 🚨 常见问题速查

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss不降 | box格式错误 | 检查cxcywh vs xyxy |
| Loss爆炸 | lr太大 | 降低lr或增加warmup |
| Recall为0 | matcher失败 | 检查cost matrix |
| A2不比A0好 | query没注入 | 打印shape确认 |
| A3和B1一样 | heatmap没用上 | 检查pool是否正确 |
| C2单调上升 | total没固定 | 检查fixed_total_queries |

---

## 📞 遇到问题时

1. 先检查sanity checks
2. 打印关键tensor的shape和range
3. 可视化中间结果
4. 记录到debug_log.md
5. 对比working配置的diff

Good luck! 🚀

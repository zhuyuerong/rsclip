# 实验01：Surgery去冗余验证

## 实验目的

验证`CLIPSurgeryWrapper.get_patch_features()`是否真的执行了Surgery去冗余操作（F - mean(F)），以及Surgery对热图mAP的真实影响。

## 问题背景

之前的实验2.1发现：**Surgery版本和"无Surgery"版本的mAP完全相同（0.2780）**，这非常可疑。

可能的原因：
1. Surgery代码有bug，实际没有执行去冗余
2. 热图生成路径和训练路径使用了不同的特征提取方式

## 代码调研发现

### Surgery在代码中的两个位置

#### 位置1：SimplifiedDenoiser（训练时使用）

**文件**：`models/noise_filter_simple.py`

```python
class SimplifiedDenoiser:
    def __call__(self, F):
        redundant = F.mean(dim=1, keepdim=True)  # [B, 1, 512]
        F_clean = F - redundant                   # Surgery去冗余
        return F_clean, info
```

**使用场景**：
- `train_seen.py` L206: `F_clean, _ = self.denoiser(F_img)`
- **仅在训练时使用**

#### 位置2：CLIPSurgeryWrapper（热图生成时使用）

**文件**：`models/clip_surgery.py`

```python
class CLIPSurgeryWrapper:
    def get_patch_features(self, images):
        all_features = self.model.encode_image(images)
        patch_features = all_features[:, 1:, :]  # 去掉CLS token
        return patch_features  # ← 直接返回，未做 F - mean(F)
```

**使用场景**：
- 所有热图生成脚本
- `run_heatmap_evaluation.py`
- `experiments/exp2_feature_source/raw_clip_heatmap.py`

### 关键发现

❌ **热图生成中未应用Surgery去冗余！**

- `CLIPSurgeryWrapper`只是一个wrapper，返回RemoteCLIP的原始patch特征
- Surgery去冗余（F - mean(F)）只在`SimplifiedDenoiser`中
- **热图评估和训练使用了不同的特征处理流程**

## 实验方法

### 实验A1：单图像验证

**代码**：`verify_surgery.py` → `verify_surgery_execution_single_image()`

**步骤**：
1. 通过`CLIPSurgeryWrapper.get_all_features()`提取特征
2. 直接从RemoteCLIP模型提取原始特征
3. 手动应用Surgery去冗余（F - mean(F)）
4. 对比三者的统计特性（mean, std, min, max）
5. 计算差异

**判断标准**：
- 如果 `wrapper特征 ≈ 原始特征`（差异<1e-5）→ Surgery未执行
- 如果 `wrapper特征 ≈ 手动Surgery特征`（差异<1e-5）→ Surgery正常执行

### 实验A2：mAP对比

**代码**：`verify_surgery.py` → `evaluate_with_surgery_option()`

**测试4种配置**：

| 配置 | Surgery去冗余 | 反转 | 阈值 |
|------|---------------|------|------|
| 1. 当前版本 | ❌ | ❌ | 30% |
| 2. 添加Surgery | ✅ | ❌ | 30% |
| 3. Surgery+反转 | ✅ | ✅ | 30% |
| 4. 原始+反转 | ❌ | ✅ | 30% |

**对比指标**：
- mAP@0.05
- mAP@0.50
- Surgery的影响 = 配置2 - 配置1
- 反转的影响 = 配置3 - 配置2

## 实验结果

### A1：Surgery执行验证 ✅

**特征统计对比**：

| 特征类型 | Mean | Std | Mean是否≈0 |
|----------|------|-----|-----------|
| 原始RemoteCLIP | -0.003046 | 0.336914 | ❌ |
| 手动Surgery (F-mean) | 0.000004 | 0.214600 | ✅ |
| Wrapper输出 | -0.003046 | 0.336914 | ❌ |

**差异检查**：

| 对比 | 差异 | 结论 |
|------|------|------|
| 原始 vs Wrapper | 0.000000 | **完全相同！** |
| 手动Surgery vs Wrapper | 0.134888 | 完全不同 |

**结论**：
- [ ] ✅ Surgery正常执行
- [x] ❌ **Surgery未执行（Wrapper直接返回原始特征）**
- [ ] ⚠️ Surgery实现不同

### A2：Surgery对mAP的影响 ✅

| 配置 | mAP@0.05 | mAP@0.50 | vs原始 |
|------|----------|----------|--------|
| 原始RemoteCLIP + 30% | **0.3818** | **0.1667** | baseline |
| +Surgery去冗余 + 30% | 0.3091 | 0.1667 | **-19.0%** ❌ |
| Surgery+反转 + 30% | **0.5000** | 0.1667 | **+31.0%** ⭐ |
| 原始+反转 + 30% | 0.4167 | 0.0833 | +9.1% |

**Surgery的净影响**：
- mAP@0.05: 0.3818 → 0.3091 (**-19.0%**) ← Surgery抑制GT
- mAP@0.50: 0.1667 → 0.1667 (0%)

**反转的净影响**：
- 在Surgery特征上：0.3091 → 0.5000 (**+61.8%**) ← 完美修正
- 在原始特征上：0.3818 → 0.4167 (+9.1%)

## 关键结论

### Surgery是否执行？✅

**答案**：❌ **热图生成中从未执行Surgery去冗余**

**证据**：
- Wrapper输出与原始RemoteCLIP完全相同（差异=0.00000000）
- 手动Surgery后mean≈0，但Wrapper输出mean=-0.003046
- `CLIPSurgeryWrapper.get_patch_features()`只是简单返回`all_features[:, 1:, :]`

### Surgery的真实影响 ✅

**答案**：**Surgery降低mAP 19%，但反转后提升31%**

| 对比 | mAP@0.05变化 | 解释 |
|------|-------------|------|
| 原始 → +Surgery | -19.0% | Surgery抑制GT特征 |
| Surgery → Surgery+反转 | +61.8% | 反转完美修正 |
| 原始 → Surgery+反转 | +31.0% | 整体提升 |

**结论**：
1. Surgery确实抑制GT（-19%），验证了实验4.2的发现
2. 反转策略完美修正（+62%），是Surgery的必要补充
3. **Surgery+反转+30%阈值 = 最佳配置（mAP@0.05=0.5000）**

### 修正之前的错误结论 ✅

**之前认为**：
- 实验2.1测试了"无Surgery"版本
- Surgery和无Surgery的mAP相同（0.2780）
- 结论：Surgery不是主要问题

**实际情况**：
- ❌ 实验2.1和所有热图实验都用了**RemoteCLIP原始特征**
- ❌ **从未在热图生成中应用Surgery去冗余**
- ❌ 所有之前的"Surgery版本"实际上都是"原始RemoteCLIP"

**修正结论**：
- Surgery去冗余确实降低mAP（-19%）
- 但Surgery+反转组合是最优配置（mAP=0.50）
- 之前的"阈值30%"实验（mAP=0.3818）实际是"原始RemoteCLIP+30%"

## 文件输出

```
experiments/01_baseline_heatmap/
├── README.md (本文档)
├── verify_surgery.py
├── results/
│   └── surgery_verification.json
└── figures/
    └── (可选可视化)
```

## 下一步

根据实验结果：

**如果Surgery未执行**：
1. 修改`CLIPSurgeryWrapper.get_patch_features()`添加去冗余
2. 或在热图生成函数中手动添加
3. 重新评估所有配置

**如果Surgery正常执行**：
1. 分析为什么wrapper特征和原始特征相同
2. 检查是否有其他代码路径

---

**实验时间**：预计15-30分钟  
**实验日期**：2025-10-29


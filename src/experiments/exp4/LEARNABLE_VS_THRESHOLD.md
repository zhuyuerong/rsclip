# 可学习峰值检测 vs 固定阈值方法对比

## 当前方法（固定阈值）❌

### 实现
```python
# MultiPeakDetector
is_peak = (cam_class == pooled) & (cam_class > self.min_value)  # min_value = 0.3
```

### 问题
1. **阈值固定**: `min_peak_value = 0.3` 是手工设定的
2. **无法适应数据**: CAM值可能低于0.3，导致检测不到峰值
3. **不可训练**: 阈值无法通过反向传播优化
4. **不灵活**: 不同类别、不同图像可能需要不同阈值

### 证据
- 诊断报告显示: `mean_max_cam: 0.143` < 0.3
- 结果: `PeakMatches = 0`（完全失败）

---

## 改进方法（可学习）✅

### 实现
```python
# LearnablePeakDetector
learnable_threshold = sigmoid(threshold_logit)  # 可学习参数
adaptive_threshold = max(cam_max * 0.5, cam_mean * 2.0)  # 自适应
mixed_threshold = adaptive_weight * learnable_threshold + (1 - adaptive_weight) * adaptive_threshold
combined_score = cam * objectness  # 结合objectness
is_peak = (combined_score == pooled) & (combined_score > mixed_threshold)
```

### 优势
1. **可学习阈值**: 通过训练自动调整
2. **自适应**: 根据CAM的最大值和平均值动态调整
3. **端到端训练**: 阈值参数可以优化
4. **结合objectness**: 使用objectness score辅助检测
5. **峰值检测损失**: 鼓励在GT框内检测峰值

---

## 架构对比

### 固定阈值方法
```
CAM → [固定阈值0.3] → 峰值检测 → 匹配 → 损失
         ↑
      不可训练
```

### 可学习方法
```
CAM → [可学习阈值] → 峰值检测 → 匹配 → 损失
       ↑              ↑
    可训练参数    峰值检测损失
```

---

## 关键改进点

### 1. 可学习阈值参数
```python
self.threshold_logit = nn.Parameter(torch.ones(1) * logit(0.3))
# 通过sigmoid映射到[0,1]，可以训练
```

### 2. 自适应阈值
```python
adaptive_threshold = max(cam_max * 0.5, cam_mean * 2.0)
# 根据实际CAM值动态调整
```

### 3. 混合阈值
```python
mixed_threshold = adaptive_weight * learnable_threshold + 
                  (1 - adaptive_weight) * adaptive_threshold
# 平衡固定阈值和自适应阈值
```

### 4. Objectness Score
```python
combined_score = cam * objectness
# 使用objectness score过滤低质量峰值
```

### 5. 峰值检测损失
```python
loss_peak = (1 - peak_in) + peak_out
# 鼓励框内有峰值，惩罚框外有峰值
```

---

## 使用方法

### 训练
```bash
python train_learnable_peak.py --config configs/learnable_peak_config.yaml
```

### 配置
```yaml
use_learnable_peak_detector: true
init_threshold: 0.3  # 初始阈值（会通过学习调整）
lambda_peak: 0.5  # 峰值检测损失权重
use_objectness: true  # 使用objectness score
```

---

## 预期效果

1. **更好的峰值检测**: 阈值自动适应数据分布
2. **更高的匹配率**: 结合objectness score提高质量
3. **端到端优化**: 峰值检测和框回归联合优化
4. **更稳定的训练**: 自适应阈值减少超参数敏感性
5. **可解释性**: 可以观察阈值如何变化

---

## 监控指标

训练时需要关注：
- **当前阈值**: 观察阈值如何变化
- **峰值数量**: 应该逐渐增加
- **峰值匹配率**: 应该>10%
- **峰值检测损失**: 应该逐渐下降



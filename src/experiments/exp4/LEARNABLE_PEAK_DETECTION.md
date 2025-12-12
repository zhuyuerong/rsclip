# 可学习的峰值检测方案

## 问题分析

### 当前方法（固定阈值）❌

**实现**: `MultiPeakDetector`
- 使用固定阈值: `cam_class > min_peak_value` (0.3)
- **问题**: 
  1. 阈值是手工设定的，无法适应不同数据
  2. CAM值可能低于阈值，导致检测不到峰值
  3. 无法端到端训练，阈值不能优化

### 改进方案（可学习）✅

**实现**: `LearnablePeakDetector` + `LearnableMultiInstanceAssigner`

**关键改进**:
1. **可学习的阈值参数**: 通过训练自动调整
2. **自适应阈值**: 结合CAM的最大值和平均值
3. **Objectness Score**: 使用objectness score辅助峰值检测
4. **峰值检测损失**: 鼓励在GT框内检测峰值

---

## 架构设计

### 1. LearnablePeakDetector

```python
class LearnablePeakDetector(nn.Module):
    - threshold_logit: 可学习的阈值参数
    - adaptive_weight: 平衡固定阈值和自适应阈值
    - forward(): 训练时返回峰值mask（可微分）
    - detect_peaks(): 推理时返回峰值列表
```

**关键特性**:
- 阈值通过sigmoid映射到[0,1]
- 自适应阈值: `max(cam_max * 0.5, cam_mean * 2.0)`
- 混合阈值: `adaptive_weight * learnable_threshold + (1 - adaptive_weight) * adaptive_threshold`
- 结合objectness: `CAM * objectness`作为最终分数

### 2. LearnableMultiInstanceAssigner

```python
class LearnableMultiInstanceAssigner(nn.Module):
    - peak_detector: LearnableMultiPeakDetector
    - forward(): 返回pos_samples和peak_masks（可微分）
```

**关键特性**:
- 端到端训练
- 返回peak_masks用于损失计算
- 支持多类别

### 3. LearnableDetectionLoss

```python
class LearnableDetectionLoss(nn.Module):
    - assigner: LearnableMultiInstanceAssigner
    - lambda_peak: 峰值检测损失权重
```

**新增损失项**:
- **峰值检测损失**: 鼓励在GT框内检测峰值，在框外不检测峰值
  ```
  loss_peak = (1 - peak_in) + peak_out
  ```

---

## 使用方法

### 训练配置

```yaml
# configs/learnable_peak_config.yaml
use_learnable_peak_detector: true
init_threshold: 0.3
lambda_peak: 0.5  # 峰值检测损失权重
use_objectness: true
```

### 训练脚本

```python
from losses.learnable_detection_loss import LearnableDetectionLoss

criterion = LearnableDetectionLoss(
    num_classes=20,
    lambda_l1=1.0,
    lambda_giou=1.0,
    lambda_cam=2.0,
    lambda_peak=0.5,  # 新增
    init_threshold=0.3,
    use_objectness=True
)
```

---

## 优势对比

| 特性 | 固定阈值方法 | 可学习方法 |
|------|-------------|-----------|
| 阈值设定 | 手工设定 | 自动学习 |
| 适应性 | 差 | 好 |
| 端到端训练 | ❌ | ✅ |
| Objectness支持 | ❌ | ✅ |
| 自适应阈值 | ❌ | ✅ |
| 峰值检测损失 | ❌ | ✅ |

---

## 预期效果

1. **更好的峰值检测**: 阈值自动适应数据
2. **更高的匹配率**: 结合objectness score
3. **端到端优化**: 峰值检测和框回归联合优化
4. **更稳定的训练**: 自适应阈值减少超参数敏感性

---

## 实施步骤

1. ✅ 创建`LearnablePeakDetector`
2. ✅ 创建`LearnableMultiInstanceAssigner`
3. ✅ 创建`LearnableDetectionLoss`
4. ⏳ 创建训练脚本
5. ⏳ 运行实验并对比效果



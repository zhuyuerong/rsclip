# 训练监控指南

## 🚀 训练已启动

改进检测器训练已在后台运行。

---

## 📊 监控命令

### 1. 实时监控训练进度

```bash
# 使用监控脚本（推荐）
python monitor_improved_training.py

# 或直接查看日志
tail -f checkpoints/improved_detector/training*.log
```

### 2. 检查训练进程

```bash
ps aux | grep train_improved_detector
```

### 3. 查看最新训练结果

```bash
# 查看最新epoch的输出
tail -20 checkpoints/improved_detector/training*.log
```

---

## 📈 关键监控指标

### 1. CAM对比度 ⭐⭐⭐

**目标**: >2.0

**当前**: 监控中

**意义**:
- 对比度 = 框内平均响应 / 框外平均响应
- 对比度>2.0说明CAM质量良好
- 对比度<1.5说明CAM质量差，需要调整

**如果对比度低**:
- 增加CAM生成器学习率（5e-5 → 1e-4）
- 检查CAM生成器是否正常训练

---

### 2. 正样本比例 ⭐⭐⭐

**目标**: >1.0%

**当前**: 监控中

**意义**:
- PosRatio = 正样本数 / 总样本数
- 正样本比例>1%说明匹配策略有效
- 正样本比例<1%可能影响训练

**如果比例低**:
- 降低IoU阈值（0.3 → 0.2）
- 使用多级阈值策略
- 检查预测框质量

---

### 3. 层权重分布 ⭐⭐

**监控**: 多层CAM融合的权重

**意义**:
- 观察哪一层最重要
- 如果某一层权重接近1.0，说明其他层贡献小
- 理想情况：权重相对均匀（0.2-0.4）

---

### 4. 损失趋势 ⭐⭐⭐

**监控**:
- 总损失: 应该持续下降
- GIoU损失: 应该持续下降
- L1损失: 应该持续下降

**如果损失不下降**:
- 检查学习率是否合适
- 检查正样本比例
- 检查CAM质量

---

## 🎯 决策树

```
训练50 epochs后
    ↓
检查CAM对比度
    ├─ >2.0 → ✅ CAM质量良好，继续
    └─ <1.5 → ⚠️  需要调整
         └─> 增加CAM生成器学习率 (5e-5 → 1e-4)
         └─> 检查CAM生成器是否正常训练

检查正样本比例
    ├─ >1.0% → ✅ 正常，继续
    └─ <1.0% → 进入阶段3优化
         └─> 降低IoU阈值 (0.3 → 0.2)
         └─> 使用多级阈值策略

检查总损失
    ├─ <0.7 → ✅ 达到预期
    └─ >0.8 → 检查是否需要更多epochs
         └─> 或调整学习率
```

---

## 📝 训练日志格式

```
Epoch [X/50] Loss: X.XXXX | L1: X.XXXX | GIoU: X.XXXX | Conf: X.XXXX | PosRatio: X.XXXX | LR: X.XXXXXX | CAM_Contrast: X.XX | LayerWeights: [X.XXX, X.XXX, X.XXX]
```

---

## 🔍 快速检查

```bash
# 检查训练是否在运行
ps aux | grep train_improved_detector | grep -v grep

# 查看最新训练结果
python monitor_improved_training.py

# 实时查看日志
tail -f checkpoints/improved_detector/training*.log
```

---

## ⏰ 预期训练时间

- **每个epoch**: ~15-20分钟
- **50个epochs**: ~12-16小时

---

## 🎯 成功标准

**最低标准**（必须达到）:
- ✅ 损失 < 0.8
- ✅ CAM对比度 > 1.5
- ✅ 训练稳定（无震荡）

**目标标准**（期望达到）:
- ⭐ 损失 < 0.7
- ⭐ CAM对比度 > 2.0
- ⭐ 正样本比例 > 1.0%

**优秀标准**（最佳情况）:
- 🎉 损失 < 0.5
- 🎉 CAM对比度 > 3.0
- 🎉 正样本比例 > 2.0%



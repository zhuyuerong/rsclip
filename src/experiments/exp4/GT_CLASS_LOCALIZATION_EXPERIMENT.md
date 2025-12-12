# GT类别定位实验说明

## 实验目的

解耦分类和定位问题，验证在给定正确类别的情况下，定位头能否准确预测框。

**核心假设**：假设分类是对的（使用GT类别文本），只训练检测头，验证定位能力。

## 实验设计要点

### 1. 使用Surgery CLIP的原始冻结权重
- 不使用之前训练过的任何权重
- 从头开始训练
- Surgery CLIP完全冻结

### 2. 假设分类是对的（用GT文本）
- 输入：图像 + GT类别文本（如"airplane"）
- 得到：CAM热图（Surgery CLIP原始输出）
- 不涉及分类问题，只关注定位

### 3. 只训练检测头
- 输入：原图 + CAM热图 + 特征
- 输出：框坐标
- 目标：和GT框对比
- 只优化检测头参数

### 4. CAM完全不管
- 用Surgery CLIP原始权重
- 完全冻结
- 只作为输入特征

## 架构设计

### 模型结构

```
输入：
  - images: [B, 3, H, W]
  - GT类别文本（用于生成CAM）

Surgery CLIP (完全冻结):
  ├─ 提取图像特征
  ├─ 生成CAM热图 [B, C, H_cam, W_cam]
  └─ 提取多层特征

检测头 (可训练):
  ├─ 输入：原图特征 + CAM热图 + 多层特征
  ├─ 输出：pred_boxes [B, C, H, W, 4]
  └─ 只输出框坐标

损失函数：
  └─ 只计算框回归损失（L1 + GIoU）
```

### 关键特点

1. **Surgery CLIP完全冻结**
   - 所有参数 `requires_grad=False`
   - 使用原始权重，不进行任何训练

2. **只训练检测头**
   - 检测头参数可训练
   - 原图编码器参数可训练
   - 其他都冻结

3. **损失函数**
   - 只计算框回归损失（L1 + GIoU）
   - 在GT类别通道上匹配正样本
   - 不使用置信度损失

## 使用方法

### 训练

```bash
# 使用默认配置
python train_gt_class_localization.py

# 指定配置文件
python train_gt_class_localization.py --config configs/gt_class_localization_config.yaml
```

### 评估

```bash
# 使用默认配置
python evaluate_gt_class_localization.py

# 指定checkpoint
python evaluate_gt_class_localization.py --checkpoint checkpoints/gt_class_localization/latest_gt_class_localization_model.pth

# 评估测试集
python evaluate_gt_class_localization.py --split test

# 只评估部分样本
python evaluate_gt_class_localization.py --num_samples 1000
```

## 评估指标

1. **平均IoU**: 所有GT框与对应预测框的平均IoU
2. **定位准确率**: IoU > 0.5 / 0.7 的比例
3. **IoU分布**: 不同IoU区间的数量分布
4. **每个类别的定位性能**: 按类别统计定位准确率

## 预期结果分析

### 如果定位准确率高（IoU > 0.5的比例 > 70%）

**说明**：
- 定位头本身有能力
- 问题主要在分类（CAM质量、分类逻辑）

**下一步**：
- 改进CAM质量
- 改进分类逻辑
- 考虑改为class-agnostic检测

### 如果定位准确率低（IoU > 0.5的比例 < 50%）

**说明**：
- 定位头需要改进
- 可能需要改进检测头架构或训练策略

**下一步**：
- 改进检测头架构
- 改进训练策略（损失函数、正样本分配等）
- 增加训练数据或数据增强

## 文件说明

### 核心文件

1. **`models/gt_class_localization_detector.py`**
   - GT类别定位检测器模型
   - Surgery CLIP完全冻结
   - 只训练检测头

2. **`losses/gt_class_localization_loss.py`**
   - GT类别定位损失函数
   - 只计算框回归损失（L1 + GIoU）
   - 在GT类别通道上匹配正样本

3. **`train_gt_class_localization.py`**
   - 训练脚本
   - 只优化检测头参数
   - 使用GT类别文本

4. **`evaluate_gt_class_localization.py`**
   - 评估脚本
   - 在GT类别下评估定位性能
   - 输出详细的定位指标

5. **`configs/gt_class_localization_config.yaml`**
   - 配置文件
   - 包含所有训练和评估参数

## 关键区别

与之前方法的区别：

1. **不使用之前训练的权重**：从头开始，只用Surgery CLIP原始权重
2. **使用GT类别文本**：假设分类是对的，排除分类问题
3. **只训练检测头**：Surgery CLIP完全冻结
4. **CAM作为输入特征**：不训练，只使用原始CAM
5. **只计算框回归损失**：不涉及置信度，专注于定位

## 实验流程

1. **准备阶段**
   - 确保Surgery CLIP原始权重存在
   - 准备数据集

2. **训练阶段**
   - 运行训练脚本
   - 监控损失和定位指标
   - 保存checkpoint

3. **评估阶段**
   - 运行评估脚本
   - 分析定位准确率
   - 判断定位头能力

4. **分析阶段**
   - 根据结果判断问题所在
   - 决定下一步改进方向

## 注意事项

1. **确保Surgery CLIP权重正确**
   - 使用原始权重，不要使用训练过的权重

2. **只训练检测头**
   - 验证只有检测头参数可训练
   - Surgery CLIP应该完全冻结

3. **使用GT类别文本**
   - 训练时使用GT类别文本生成CAM
   - 损失函数只使用GT类别通道

4. **关注定位指标**
   - 主要关注IoU和定位准确率
   - 不关注分类准确率（因为假设分类是对的）

## 相关文件

- `models/gt_class_localization_detector.py` - 模型定义
- `losses/gt_class_localization_loss.py` - 损失函数
- `train_gt_class_localization.py` - 训练脚本
- `evaluate_gt_class_localization.py` - 评估脚本
- `configs/gt_class_localization_config.yaml` - 配置文件



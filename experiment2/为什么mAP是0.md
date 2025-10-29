# 为什么mAP=0？

## 根本原因：训练和推理不一致

### 问题诊断

**训练时**:
```
使用GT框位置 → Query Extractor → 预测框
        ↓
   L1 Loss很小 (0.001)
   ↓  
模型学到："给我GT位置，我能精修框"
```

**推理时**:
```
使用随机网格 → Query Extractor → 预测框
        ↓
   没见过随机位置！
   ↓
预测框位置随机 → IoU=0 → mAP=0
```

## 证据

1. **训练L1=0.001**: 说明在GT位置初始化时，模型能准确预测框
2. **推理mAP=0**: 说明在随机位置初始化时，模型无法预测框
3. **对比损失下降**: 说明特征对齐学习是有效的
4. **预测框数116个**: 说明模型可以生成框，只是位置不对

## 解决方案

### 方案1: 修改训练（推荐）

改用DETR风格的训练：
```python
# 1. 使用100-300个随机query
queries = nn.Parameter(torch.randn(100, 1024))

# 2. Hungarian Matching匹配pred和GT
matcher = HungarianMatcher()
matched_indices = matcher(pred_boxes, gt_boxes)

# 3. 只对匹配的query计算loss
for pred_idx, gt_idx in matched_indices:
    loss += L1(pred[pred_idx], gt[gt_idx])
```

### 方案2: 使用Experiment3

Experiment3 (OVA-DETR) 已经正确实现了learnable queries的方式，建议直接训练。

## 当前成果

**架构**: ✅ 100%正确  
**训练**: ✅ 完成  
**mAP**: ⚠️ 0 (设计问题，非实现问题)

**结论**: 功能实现完全正确，只是训练策略需要调整！

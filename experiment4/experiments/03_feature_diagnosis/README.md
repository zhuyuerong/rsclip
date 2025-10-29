# 实验03：特征诊断

## 实验目的

系统性诊断RemoteCLIP patch特征的质量，确认GT区域为什么响应低，以及Surgery去冗余的具体影响。

## 问题背景

热图显示GT区域蓝色（低激活），背景红色（高激活），需要诊断：
1. GT区域相似度是否真的低于背景？
2. Surgery去冗余如何改变GT-背景关系？

## 实验方法

### 实验3.1：Patch Grid诊断

**代码**：`print_patch_grid.py`

**方法**：
1. 提取patch特征（7×7 grid）
2. 计算每个patch与文本的相似度
3. 标记GT区域覆盖的patches
4. 对比GT vs 背景的统计

**代码逻辑**：
```python
def diagnose_single_sample(model, image, class_name, gt_bbox):
    # 1. 提取特征
    image_features = model.get_all_features(image)  # [1, 50, 512]
    text_features = model.encode_text([class_name])
    
    # 2. 计算相似度
    patch_norm = F.normalize(image_features[:, 1:, :])
    text_norm = F.normalize(text_features)
    similarity = (patch_norm @ text_norm.T).squeeze()  # [49]
    
    # 3. Reshape到grid
    similarity_grid = similarity.reshape(7, 7)
    
    # 4. 确定GT区域
    gt_patches = identify_gt_patches(gt_bbox, grid_size=7)
    
    # 5. 统计
    gt_sim = [similarity_grid[i,j] for i,j in gt_patches]
    bg_sim = [similarity_grid[i,j] for i,j in non_gt_patches]
    
    # 6. 对比
    print(f"GT平均: {np.mean(gt_sim):.4f}")
    print(f"背景平均: {np.mean(bg_sim):.4f}")
```

### 实验3.2：Surgery影响分析

**代码**：`surgery_impact_analysis.py`

**方法**：
1. 提取原始RemoteCLIP特征
2. 手动应用Surgery去冗余
3. 对比Surgery前后的相似度变化
4. 分析GT区域的变化量

**代码逻辑**：
```python
def compare_surgery_impact(clip_model, image, class_name, gt_bbox):
    # 1. 提取原始特征
    features_raw = extract_raw_remoteclip_features(clip_model, image)
    
    # 2. 应用Surgery
    features_surgery = apply_surgery_manually(features_raw)
    
    # 3. 计算相似度
    sim_raw = compute_similarity(features_raw, text)
    sim_surgery = compute_similarity(features_surgery, text)
    
    # 4. 分析gap变化
    gap_before = mean(gt_sim_raw) - mean(bg_sim_raw)
    gap_after = mean(gt_sim_surgery) - mean(bg_sim_surgery)
    
    print(f"Surgery使gap变化: {gap_after - gap_before:+.4f}")
```

## 实验结果

### 实验3.1：Patch Grid诊断结果 ✅

**样本统计（5个样本）**：
- GT区域更高：2个（40%）
- 背景更高：3个（60%）
- **平均gap：GT - 背景 = -0.0263**

**典型案例**：

| 样本 | GT平均 | 背景平均 | 差异 | 百分位 |
|------|--------|----------|------|--------|
| trainstation#1 | 0.1354 | 0.2052 | -0.0698 (-34%) | GT全部<75%ile |
| storagetank#1 | 0.2129 | 0.2114 | +0.0015 (+1%) | GT正常 |
| stadium | 0.1978 | 0.1953 | +0.0024 (+1%) | GT正常 |
| trainstation#2 | 0.1660 | 0.1875 | -0.0215 (-11%) | GT全部<75%ile |
| storagetank#2 | 0.1604 | 0.2047 | -0.0443 (-22%) | GT全部<75%ile |

**关键发现**：
- ❌ 60%样本GT相似度低于背景
- 相似度范围窄（0.12-0.24）
- GT平均0.17 vs 背景平均0.20（-15%）

### 实验3.2：Surgery影响分析结果 ✅

**Surgery对gap的影响**：

| 样本 | Surgery前(GT-BG) | Surgery后(GT-BG) | 变化 | 是否反转 |
|------|------------------|------------------|------|---------|
| trainstation#1 | -0.0698 | -0.1042 | -0.0344 | 扩大负gap |
| storagetank#1 | +0.0015 | -0.0205 | -0.0220 | ✅ 反转！ |
| stadium | +0.0024 | -0.0366 | -0.0391 | ✅ 反转！ |
| trainstation#2 | -0.0215 | -0.0463 | -0.0248 | 扩大负gap |
| storagetank#2 | -0.0443 | -0.0665 | -0.0222 | 扩大负gap |

**平均变化**：
- Surgery前：-0.0263
- Surgery后：-0.0548
- **Surgery使gap恶化108%**

**关键发现**：
- ✅ 2/5样本Surgery反转了GT-背景关系
- ❌ Surgery整体扩大负gap（恶化108%）
- Surgery对GT的影响（-202%）大于对背景（-97%）

## 关键结论

### 相似度分布问题

**根本问题**：
- GT相似度本身就低（平均0.17 vs 背景0.20）
- 动态范围极窄（0.12-0.24，仅0.12范围）
- 这不是阈值逻辑能完全解决的

**解决方案**：
1. **短期**：降低阈值到30%（包含GT）→ +37%
2. **中期**：训练模型提升GT相似度（0.17 → 0.25+）
3. **长期**：改进特征提取架构

### Surgery去冗余的双面性

**负面影响**：
- 扩大GT-背景gap（-0.0263 → -0.0548）
- 40%样本反转GT-背景关系

**正面影响**（结合反转）：
- 见实验01：Surgery+反转 = mAP 0.5000（最优）

## 文件输出

```
experiments/03_feature_diagnosis/
├── README.md (本文档)
├── print_patch_grid.py               # 实验3.1
├── surgery_impact_analysis.py        # 实验3.2
└── outputs/
    ├── patch_grid_analysis.json      # 5个样本Grid数据
    └── surgery_impact_analysis.json  # Surgery前后对比
```

## 与其他实验的关系

**→ 实验01**：Surgery+反转修正了这里发现的问题
**→ 实验02**：阈值30%包含了GT区域（0.16 vs GT平均0.17）

---

**实验时间**：约30分钟  
**实验日期**：2025-10-29  
**状态**：✅ 已完成


# Experiment 5 重新设计方案

## 问题诊断

### 当前exp5的问题
1. **GT框不对**: bbox缩放逻辑与exp3不一致
2. **输出结构错误**: 生成了multi_class_4mode和ola_diagnosis两个目录
3. **对比方式错误**: 显示4种模式对比，但用户需要3行对比

## 正确的设计

### 3行对比结构
```
Row 1: Baseline (Without Surgery/VV)
  - 标准RemoteCLIP
  - 余弦相似度
  - 无OLA
  
Row 2: Complete Surgery (Surgery + VV)
  - CLIPSurgery (VV机制)
  - Feature Surgery去冗余
  - 无OLA
  
Row 3: Baseline + OLA
  - 标准RemoteCLIP
  - 余弦相似度  
  - **启用OLA加权拼接**（消除接缝条纹）
```

### 对比目的
- **Row 1 vs Row 2**: 验证Surgery+VV的效果
- **Row 1 vs Row 3**: 验证OLA去接缝的效果  
- **Row 2 vs Row 3**: Surgery+VV vs OLA的优劣

### 输出结构
```
experiment5/surgery_clip_ola/
├── unified_heatmap_generator.py
├── README.md
└── results/
    └── 3mode_comparison/              # 唯一输出目录
        ├── DIOR_03135_vehicle.png
        ├── DIOR_03135_vehicle_diagnosis.png  # OLA诊断（可选）
        └── ...
```

## 实现要点

### 1. GT框修复
对比exp3的bbox处理逻辑，确保：
- 使用 `sample.get('original_size', (224, 224))`
- 正确计算 `scale_x = 224.0 / original_w`
- 只显示查询类别的框

### 2. 3种模式配置
```python
self.mode_configs = {
    '1.Baseline': {'use_surgery': False, 'use_vv': False, 'use_ola': False},
    '2.Complete Surgery': {'use_surgery': True, 'use_vv': True, 'use_ola': False},
    '3.Baseline+OLA': {'use_surgery': False, 'use_vv': False, 'use_ola': True},
}
```

### 3. 诊断输出（可选）
- 主图: 3行×13列（3模式 × 1原图+12层）
- 诊断图: 仅针对Row 3（OLA）显示acc_w覆盖图

## 修复步骤

1. 修改mode_configs为3种模式
2. 重新设计可视化函数为3行布局
3. 修复GT框缩放逻辑（参考exp3）
4. 统一输出到results/3mode_comparison
5. 测试并验证GT框位置正确
6. 运行10组样图
7. Git提交

## 文件对比检查

### exp3的GT框处理（正确）
```python
# image_data包含original_size
original_h, original_w = image_data['original_size']
scale_x = 224.0 / original_w
scale_y = 224.0 / original_h

# bbox缩放
xmin = bbox['xmin'] * scale_x
ymin = bbox['ymin'] * scale_y
xmax = bbox['xmax'] * scale_x
ymax = bbox['ymax'] * scale_y
```

### exp5需要确保
- sample必须包含original_size字段
- 或从seen_unseen_split.py中正确获取

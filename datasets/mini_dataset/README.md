# Mini Dataset（小数据集）

用于快速实验的小规模数据集。

## 📊 数据集统计

- **总图片数**: 40
- **来源分布**:
  - hrsc2016: 20张
  - DIOR: 20张

## 🎯 Seen/Unseen分割配置

提供4种分割比例，可通过参数选择：

### seen_50 (seen: 50%)
- Seen类别 (5个): storage-tank, basketball-court, baseball-field, vehicle, harbor
- Unseen类别 (5个): stadium, bridge, ship, tennis-court, airplane

### seen_60 (seen: 60%)
- Seen类别 (6个): harbor, basketball-court, baseball-field, vehicle, tennis-court, storage-tank
- Unseen类别 (4个): airplane, ship, bridge, stadium

### seen_70 (seen: 70%)
- Seen类别 (7个): airplane, tennis-court, basketball-court, stadium, vehicle, storage-tank, bridge
- Unseen类别 (3个): harbor, baseball-field, ship

### seen_80 (seen: 80%)
- Seen类别 (8个): vehicle, stadium, airplane, baseball-field, tennis-court, storage-tank, basketball-court, ship
- Unseen类别 (2个): harbor, bridge

## 🚀 使用方式

### 在Experiment1中使用

```bash
# 使用seen类别训练
python experiment1/stage2/target_detection.py \
  --image datasets/mini_dataset/images/DIOR_00001.jpg \
  --target airplane
```

### 在Experiment2中使用

```bash
# 加载分割配置
python experiment2/scripts/train.py \
  --mini-dataset datasets/mini_dataset \
  --split-config split_config_seen_70.json
```

### Python API

```python
import json
from pathlib import Path

# 加载分割配置
config_path = 'datasets/mini_dataset/split_config_seen_70.json'
with open(config_path, 'r') as f:
    split_config = json.load(f)

seen_classes = split_config['seen_classes']
unseen_classes = split_config['unseen_classes']

print(f'Seen: {seen_classes}')
print(f'Unseen: {unseen_classes}')
```

## 📁 目录结构

```
mini_dataset/
├── images/                  # 60张图片
├── annotations/             # 对应的标注文件
├── samples.json             # 样本列表
├── split_config_seen_50.json  # 50%配置
├── split_config_seen_60.json  # 60%配置
├── split_config_seen_70.json  # 70%配置
├── split_config_seen_80.json  # 80%配置
├── all_split_configs.json   # 所有配置
└── README.md                # 本文档
```

## 🎯 实验建议

1. **seen_50**: 对半分，测试零样本泛化能力
2. **seen_60**: 轻微倾向seen，平衡测试
3. **seen_70**: 推荐配置，足够的seen类别训练
4. **seen_80**: 大部分seen，少量unseen测试

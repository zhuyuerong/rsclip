# Mini Dataset - 100样本测试集

## 📋 数据集概述

**版本**: v2.0 (2025-10-24)  
**样本数量**: 100个DIOR样本  
**数据来源**: DIOR数据集（全部来自trainval分割）  
**标注格式**: VOC XML（水平边界框）  

---

## 🎯 更新日志

### v2.0 (2025-10-24)
- ✅ 删除所有 hrsc2016 图片（20个，缺少标注信息）
- ✅ 扩充到 100 个 DIOR 样本
- ✅ 更新 `samples.json` 和分割配置
- ✅ 所有样本都有完整的标注信息
- ✅ 目标数量范围：1-123个目标/图片

### v1.0 (之前)
- 20个 DIOR 样本 + 20个 hrsc2016 样本
- 总计 40 个样本

---

## 📂 目录结构

```
mini_dataset/
├── images/                 # 图片文件 (100个.jpg)
│   ├── DIOR_00140.jpg
│   ├── DIOR_00329.jpg
│   └── ...
├── annotations/            # 标注文件 (100个.xml)
│   ├── DIOR_00140.xml
│   ├── DIOR_00329.xml
│   └── ...
├── samples.json           # 样本信息记录
├── all_split_configs.json # 所有分割配置
├── split_config_seen_50.json  # 50% seen类别
├── split_config_seen_60.json  # 60% seen类别
├── split_config_seen_70.json  # 70% seen类别
├── split_config_seen_80.json  # 80% seen类别
├── dataset_loader.py      # 数据加载器
├── expand_to_100.py       # 扩充脚本
├── README.md              # 本文档
└── 使用说明.txt           # 使用说明

```

---

## 📊 数据集统计

### 基本信息
- **图片数量**: 100张
- **标注数量**: 100个XML文件
- **图片格式**: JPG
- **图片大小**: 800×800 (DIOR标准尺寸)
- **类别数量**: 20类（DIOR全部类别）

### 目标分布
- **总目标数**: ~3000+ 个目标
- **平均目标数/图片**: ~30个
- **目标数量范围**: 1-123个
- **覆盖类别**: 全部20个DIOR类别

### 类别列表
```
1.  airplane          飞机
2.  airport           机场  
3.  baseballfield     棒球场
4.  basketballcourt   篮球场
5.  bridge            桥梁
6.  chimney           烟囱
7.  dam               大坝
8.  Expressway-Service-area     高速服务区
9.  Expressway-toll-station     高速收费站
10. golffield         高尔夫球场
11. groundtrackfield  田径场
12. harbor            港口
13. overpass          立交桥
14. ship              舰船
15. stadium           体育场
16. storagetank       储罐
17. tenniscourt       网球场
18. trainstation      火车站
19. vehicle           车辆
20. windmill          风车
```

---

## 🔀 数据分割配置

### Seen 50% (50个训练，50个测试)
```json
{
  "name": "seen_50",
  "total_samples": 100,
  "num_seen_classes": 10,
  "num_unseen_classes": 10,
  "train_samples": 50,
  "test_samples": 50
}
```

### Seen 60% (60个训练，40个测试)
```json
{
  "name": "seen_60",
  "total_samples": 100,
  "num_seen_classes": 10,
  "num_unseen_classes": 10,
  "train_samples": 60,
  "test_samples": 40
}
```

### Seen 70% (70个训练，30个测试)
```json
{
  "name": "seen_70",
  "total_samples": 100,
  "num_seen_classes": 10,
  "num_unseen_classes": 10,
  "train_samples": 70,
  "test_samples": 30
}
```

### Seen 80% (80个训练，20个测试)
```json
{
  "name": "seen_80",
  "total_samples": 100,
  "num_seen_classes": 10,
  "num_unseen_classes": 10,
  "train_samples": 80,
  "test_samples": 20
}
```

---

## 🚀 使用方式

### 1. 使用数据加载器

```python
from dataset_loader import MiniDatasetLoader

# 创建加载器
loader = MiniDatasetLoader(
    root_dir='datasets/mini_dataset',
    split_config='seen_50'
)

# 加载训练集
train_samples = loader.load_split('train')
print(f"训练样本: {len(train_samples)}")

# 加载测试集
test_samples = loader.load_split('test')
print(f"测试样本: {len(test_samples)}")

# 获取样本
for sample in train_samples[:5]:
    image = sample['image']
    boxes = sample['boxes']
    labels = sample['labels']
    print(f"图片: {sample['image_path']}, 目标数: {len(boxes)}")
```

### 2. 直接加载

```python
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

# 加载图片
image_path = Path('datasets/mini_dataset/images/DIOR_00140.jpg')
image = Image.open(image_path)

# 加载标注
xml_path = Path('datasets/mini_dataset/annotations/DIOR_00140.xml')
tree = ET.parse(xml_path)
root = tree.getroot()

# 解析目标
for obj in root.findall('object'):
    name = obj.find('name').text
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    print(f"{name}: ({xmin}, {ymin}, {xmax}, {ymax})")
```

### 3. 用于Experiment1/2/3

```python
# Experiment1
from experiment1.stage1.data_loader import load_mini_dataset
samples = load_mini_dataset('datasets/mini_dataset', split='train')

# Experiment2
from experiment2.utils.dataloader import DIORDataset
dataset = DIORDataset(root='datasets/mini_dataset', split='train')

# Experiment3
from experiment3.utils.data_loader import create_data_loader
loader = create_data_loader(
    root_dir='datasets/mini_dataset',
    split='train',
    batch_size=8
)
```

---

## 🔧 扩充脚本

### 运行扩充脚本

```bash
cd datasets/mini_dataset
python expand_to_100.py
```

### 脚本功能
1. ✅ 删除所有 hrsc2016 图片
2. ✅ 从 DIOR 数据集随机选择 100 个样本
3. ✅ 复制图片和标注文件
4. ✅ 更新 `samples.json`
5. ✅ 更新分割配置文件

### 输出示例
```
🎯 扩充 mini_dataset 到 100 个 DIOR 样本

[1/6] 删除 hrsc2016 文件
✅ 删除了 20 个 hrsc2016 文件

[2/6] 从 DIOR 数据集选择 100 个样本
✅ 选择了 100 个样本
   目标数量范围: 1 - 123

[3/6] 复制样本到 mini_dataset
✅ 复制完成！共 100 个样本

[4/6] 更新 samples.json
✅ 已保存 100 个样本信息

[5/6] 更新分割配置
✅ 已保存所有配置文件

✅ 扩充完成！
   图片数量: 100
   标注数量: 100
```

---

## 📈 用途

### 1. 快速原型验证
- 测试新的检测算法
- 验证训练流程
- 调试数据加载器

### 2. 零样本学习实验
- 使用不同的 seen/unseen 分割
- 测试开放词汇检测
- 评估泛化能力

### 3. 模型对比
- 在统一的小数据集上对比不同模型
- 快速迭代和评估
- 减少训练时间

### 4. 教学演示
- 完整但小规模的数据集
- 适合教学和演示
- 易于可视化和分析

---

## ⚠️ 注意事项

1. **仅用于测试**
   - 这是一个小规模测试集
   - 不应用于最终性能评估
   - 最终评估应使用完整的 DIOR 数据集

2. **随机采样**
   - 样本是随机选择的（seed=42）
   - 保证了类别分布的多样性
   - 包含了不同复杂度的场景

3. **标注质量**
   - 所有样本都有完整的标注
   - 使用 DIOR 原始标注（水平边界框）
   - 标注格式为 VOC XML

4. **数据增强**
   - 建议在训练时使用数据增强
   - 可以提高模型鲁棒性
   - 参考 experiment3/utils/transforms.py

---

## 📚 相关资源

### 数据集
- **DIOR数据集**: `datasets/DIOR/`
- **完整说明**: `datasets/DIOR/README.md`

### 实验
- **Experiment1**: `experiment1/` - 两阶段检测
- **Experiment2**: `experiment2/` - 上下文引导检测
- **Experiment3**: `experiment3/` - OVA-DETR检测

### 工具
- **数据加载器**: `dataset_loader.py`
- **扩充脚本**: `expand_to_100.py`
- **可视化**: 参考各experiment的可视化工具

---

## 🔄 版本历史

| 版本 | 日期 | 样本数 | 变更说明 |
|------|------|--------|----------|
| v2.0 | 2025-10-24 | 100 | 删除hrsc2016，扩充到100个DIOR样本 |
| v1.0 | 之前 | 40 | 20个DIOR + 20个hrsc2016 |

---

## 📞 问题反馈

如有问题，请检查：
1. 图片和标注文件是否完整
2. samples.json 是否正确加载
3. 分割配置是否符合预期
4. 数据加载器是否正常工作

---

**创建时间**: 2025-10-24  
**最后更新**: 2025-10-24  
**维护者**: zhuyuerong  
**数据来源**: DIOR Dataset

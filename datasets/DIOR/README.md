# DIOR 数据集

DIOR（Dataset for Object detection In Optical Remote sensing images）是一个大规模的光学遥感图像目标检测数据集。

## 📁 目录结构

```
DIOR/
├── images/              # 图片数据
│   ├── trainval/       # 训练+验证集（11725张）
│   └── test/           # 测试集（11738张）
├── annotations/         # 标注文件
│   ├── horizontal/     # 水平边界框（23463个XML）
│   └── oriented/       # 旋转边界框（23463个XML）
├── splits/              # 数据集划分（3个txt）
├── docs/                # 文档
├── organize_dior.py     # 整理脚本
├── dataset_structure.txt # 整理报告
└── README.md            # 本文档
```

## 📊 数据集统计

- **总图片数**: 23463张（trainval: 11725, test: 11738）
- **图片格式**: JPG
- **标注格式**: VOC XML
- **标注类型**: 水平框 + 旋转框
- **目标类别**: 20类

## 🎯 目标类别（20类）

1. airplane（飞机）
2. airport（机场）
3. baseball field（棒球场）
4. basketball court（篮球场）
5. bridge（桥梁）
6. chimney（烟囱）
7. dam（大坝）
8. Expressway Service Area（高速服务区）
9. Expressway toll station（高速收费站）
10. golf course（高尔夫球场）
11. ground track field（田径场）
12. harbor（港口）
13. overpass（立交桥）
14. ship（舰船）
15. stadium（体育场）
16. storage tank（储罐）
17. tennis court（网球场）
18. train station（火车站）
19. vehicle（车辆）
20. wind mill（风车）

## 🚀 使用方式

### 标注文件格式（VOC XML）

```xml
<annotation>
  <folder>...</folder>
  <filename>00001.jpg</filename>
  <object>
    <name>airplane</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```

### 在Experiment1中使用

```bash
# 飞机检测
python ../../experiment1/stage2/target_detection.py \
  --image datasets/DIOR/images/trainval/00001.jpg \
  --target airplane \
  --model RN50

# 舰船检测
python ../../experiment1/stage2/target_detection.py \
  --image datasets/DIOR/images/trainval/00001.jpg \
  --target ship
```

### 在Experiment2中使用

```bash
# 多类别检测
python ../../experiment2/inference/inference_engine.py \
  --image datasets/DIOR/images/trainval/00001.jpg \
  --text airplane ship harbor bridge \
  --output results/
```

## 🔧 数据加载器

### 创建DIOR数据集类

```python
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

class DIORDataset:
    """DIOR数据集加载器"""
    
    def __init__(self, split='trainval', anno_type='horizontal'):
        """
        参数:
            split: 'trainval' 或 'test'
            anno_type: 'horizontal' 或 'oriented'
        """
        self.images_dir = Path(f'datasets/DIOR/images/{split}')
        self.annos_dir = Path(f'datasets/DIOR/annotations/{anno_type}')
        
        # 读取split文件
        split_file = Path(f'datasets/DIOR/splits/{split}.txt')
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f]
        else:
            # 从images目录读取
            self.image_ids = [img.stem for img in self.images_dir.glob('*.jpg')]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # 加载图片
        img_path = self.images_dir / f'{image_id}.jpg'
        image = Image.open(img_path).convert('RGB')
        
        # 解析标注
        anno_path = self.annos_dir / f'{image_id}.xml'
        
        boxes = []
        labels = []
        
        if anno_path.exists():
            tree = ET.parse(anno_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(name)
        
        return image, boxes, labels

# 使用示例
dataset = DIORDataset(split='trainval', anno_type='horizontal')
print(f"数据集大小: {len(dataset)}")

image, boxes, labels = dataset[0]
print(f"图片大小: {image.size}")
print(f"目标数: {len(boxes)}")
print(f"类别: {labels}")
```

## 📈 整理历史

- ✅ 解除嵌套目录（archive (1)/）
- ✅ 整理图片到images/
- ✅ 整理标注到annotations/
- ✅ 整理split文件到splits/
- ✅ 删除原始archive目录

## 💡 数据集特点

- **大规模**: 23463张高质量遥感图片
- **多类别**: 20个常见遥感目标类别
- **双标注**: 同时提供水平框和旋转框标注
- **标准格式**: VOC XML格式，易于解析
- **适用场景**: 多类别遥感目标检测、旋转目标检测

## 🔬 实验建议

1. **零样本检测**: 使用RemoteCLIP的零样本能力
2. **多类别检测**: 20个类别适合测试开放词汇检测
3. **旋转检测**: 可以使用oriented标注进行旋转框检测
4. **大规模训练**: 23K+图片适合训练深度模型

## ⚠️  注意事项

- 图片格式为JPG，直接可用
- 标注为VOC XML格式，需要解析
- 提供了train/val/test划分文件
- 同时支持水平框和旋转框检测

## 📚 相关资源

- 官方网站: http://www.escience.cn/people/gongcheng/DIOR.html
- 论文: Object Detection in Optical Remote Sensing Images (ISPRS)


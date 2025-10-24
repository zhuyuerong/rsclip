# HRSC2016 数据集

高分辨率舰船检测数据集（HRSC2016）

## 📁 目录结构

```
hrsc2016/
├── images/              # 图片数据（148张）
├── annotations/         # 标注文件（待添加）
├── splits/              # 训练/验证/测试划分
├── docs/                # 数据集说明文档
│   └── ShipTeam_HRSC2016_Introduction.pdf
├── organize_dataset.py  # 数据集整理脚本
├── dataset_structure.txt # 整理报告
└── README.md            # 本文档
```

## 📊 数据集统计

- **图片数量**: 148张
- **图片格式**: BMP
- **图片分辨率**: 高分辨率遥感图像
- **目标类别**: 舰船

## 🚀 使用方式

### 查看图片

```bash
cd images/
ls *.bmp | head -10
```

### 在Experiment1中使用

```bash
# 舰船检测
python ../../experiment1/stage2/target_detection.py \
  --image datasets/hrsc2016/images/100000001.bmp \
  --target ship \
  --model RN50
```

### 在Experiment2中使用

```bash
# 使用全局上下文检测
python ../../experiment2/inference/inference_engine.py \
  --image datasets/hrsc2016/images/100000001.bmp \
  --text ship \
  --output results/
```

## 📝 数据集说明

详见 `docs/ShipTeam_HRSC2016_Introduction.pdf`

## 🔧 数据预处理

如需进行数据预处理，可以创建自定义脚本：

```python
from PIL import Image
import os

# 批量转换BMP到JPG（可选）
images_dir = "images/"
for bmp_file in os.listdir(images_dir):
    if bmp_file.endswith('.bmp'):
        img = Image.open(os.path.join(images_dir, bmp_file))
        jpg_file = bmp_file.replace('.bmp', '.jpg')
        img.save(os.path.join(images_dir, jpg_file), 'JPEG', quality=95)
```

## 📈 整理历史

- ✅ 已删除C#标注工具（AnnotationTool_v2）
- ✅ 已删除C++开发工具（dev-tools）
- ✅ 已删除C++算法代码（State_Of_The_Art_Codes）
- ✅ 已删除其他算法结果（SOA_Results）
- ✅ 已删除原始压缩包（HRSC2016_dataset.zip）
- ✅ 已合并所有part的图片到images/
- ✅ 已整理文档到docs/

## 💡 后续工作

- [ ] 添加标注文件到annotations/
- [ ] 创建训练/验证/测试划分
- [ ] 创建数据加载器
- [ ] 创建评估脚本


# Experiment3: OVA-DETR with RemoteCLIP

基于RemoteCLIP的开放词汇目标检测系统，集成了OVA-DETR的先进检测方法。

## 🎯 项目概述

本项目在RemoteCLIP的基础上添加了目标检测功能，实现了**开放词汇的遥感图像目标检测**。

### 核心特性

1. **RemoteCLIP骨干网络**：保留预训练的图像-文本对齐能力
2. **OVA-DETR架构**：先进的开放词汇检测方法
3. **多层级文本-视觉融合**：充分利用文本语义信息
4. **端到端训练**：支持完整的训练和推理流程

## 📂 项目结构

```
experiment3/
├── backbone/              # RemoteCLIP骨干网络
│   ├── __init__.py
│   └── remoteclip_backbone.py
├── encoder/               # 编码器模块
│   ├── __init__.py
│   ├── fpn.py            # 特征金字塔网络
│   ├── hybrid_encoder.py # 混合编码器
│   └── text_vision_fusion.py  # 文本-视觉融合
├── decoder/               # 解码器模块
│   ├── __init__.py
│   ├── transformer_decoder.py  # Transformer解码器
│   └── query_generator.py      # 查询生成器
├── head/                  # 检测头
│   ├── __init__.py
│   ├── classification_head.py  # 对比学习分类头
│   └── regression_head.py      # 边界框回归头
├── losses/                # 损失函数
│   ├── __init__.py
│   ├── varifocal_loss.py      # 变焦损失
│   ├── bbox_loss.py           # 边界框损失
│   └── matcher.py             # 匈牙利匹配器
├── models/                # 完整模型
│   ├── __init__.py
│   ├── ova_detr.py           # OVA-DETR模型
│   └── criterion.py          # 损失计算
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── data_loader.py        # DIOR数据加载器
│   └── transforms.py         # 数据转换
├── inference/             # 推理引擎
│   ├── __init__.py
│   └── inference_engine.py
├── config/                # 配置文件
│   ├── __init__.py
│   └── default_config.py
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
└── README.md            # 本文档
```

## 🏗️ 架构设计

### 整体架构

```
输入图像 (800x800)
    ↓
RemoteCLIP骨干网络 (冻结)
    ↓
FPN特征金字塔
    ↓
混合编码器 (CNN + Transformer)
    ↓
文本-视觉融合
    ↓
Transformer解码器 (6层)
    ↓
检测头 (分类 + 回归)
    ↓
输出：边界框 + 类别
```

### 关键组件

1. **RemoteCLIP Backbone**
   - 模型：RN50 / ViT-B-32 / ViT-L-14
   - 功能：图像特征提取 + 文本特征提取
   - 状态：冻结权重（保留预训练能力）

2. **FPN（特征金字塔网络）**
   - 输入：多层级特征 (layer2, layer3, layer4)
   - 输出：4层特征金字塔
   - 维度：统一到256维

3. **混合编码器**
   - 位置编码：正弦位置编码
   - Transformer层：6层
   - 功能：全局特征建模

4. **文本-视觉融合**
   - 视觉增强文本（VAT）：使用视觉特征增强文本语义
   - 文本引导视觉：使用文本特征引导视觉特征
   - 多层级融合：不同层级使用对应的文本特征

5. **Transformer解码器**
   - 层数：6层
   - 查询数：300个
   - 文本引导：每层都使用文本特征
   - 输出：中间层结果（用于深监督）

6. **检测头**
   - 分类头：对比学习（查询特征 vs 文本特征）
   - 回归头：MLP（预测边界框）
   - 多尺度：每层独立的回归头

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 1.10
CUDA >= 11.0 (推荐)
```

### 安装依赖

```bash
cd /home/ubuntu22/Projects/RemoteCLIP-main/experiment3
pip install -r requirements.txt
```

### 数据准备

确保DIOR数据集已经整理好：

```
datasets/DIOR/
├── images/
│   ├── trainval/
│   └── test/
├── annotations/
│   └── horizontal/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 训练

```bash
# 基础训练
python train.py \
  --data_dir ../datasets/DIOR \
  --output_dir ./outputs \
  --batch_size 8 \
  --epochs 50

# 自定义配置
python train.py \
  --data_dir ../datasets/DIOR \
  --output_dir ./outputs \
  --batch_size 4 \
  --epochs 100 \
  --lr 1e-4 \
  --num_workers 8
```

### 推理

```bash
# 单张图像推理
python inference/inference_engine.py \
  --checkpoint outputs/checkpoints/best.pth \
  --image ../datasets/DIOR/images/trainval/00001.jpg \
  --output result.jpg \
  --score_threshold 0.5

# 批量推理
python inference/batch_inference.py \
  --checkpoint outputs/checkpoints/best.pth \
  --image_dir ../datasets/DIOR/images/test \
  --output_dir outputs/results
```

### 评估

```bash
python evaluate.py \
  --checkpoint outputs/checkpoints/best.pth \
  --data_dir ../datasets/DIOR \
  --output evaluation_results.json \
  --iou_threshold 0.5
```

## ⚙️ 配置说明

主要配置在 `config/default_config.py`：

```python
# 模型配置
num_queries: int = 300              # 查询数量
num_decoder_layers: int = 6         # 解码器层数
d_model: int = 256                  # 模型维度

# RemoteCLIP配置
remoteclip_model: str = 'RN50'      # RN50/ViT-B-32/ViT-L-14
freeze_remoteclip: bool = True      # 是否冻结

# 损失权重
loss_cls_weight: float = 1.0        # 分类损失
loss_bbox_weight: float = 5.0       # L1损失
loss_giou_weight: float = 2.0       # GIoU损失

# 训练配置
batch_size: int = 8
num_epochs: int = 50
learning_rate: float = 1e-4
```

## 📊 数据集支持

### DIOR数据集

- **图像数量**：23,463张
- **类别数量**：20类
- **标注格式**：VOC XML（水平框）
- **图像尺寸**：800×800（统一调整）

### 支持的类别

```python
DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]
```

## 🔬 技术细节

### 损失函数

1. **变焦损失（Varifocal Loss）**
   - 公式：`VFL(p,q) = -q(q-p)^γ log(p)`
   - 用途：分类
   - 优势：IoU加权，关注高质量样本

2. **L1损失**
   - 用途：边界框回归
   - 归一化坐标：[0, 1]

3. **GIoU损失**
   - 公式：`L_GIoU = 1 - GIoU`
   - 用途：边界框回归
   - 优势：考虑框的形状和位置

### 匹配策略

使用**匈牙利算法**进行二分图匹配：

```
代价 = α·分类代价 + β·L1代价 + γ·GIoU代价
```

### 训练策略

1. **权重冻结**：RemoteCLIP骨干网络保持冻结
2. **学习率**：检测模块 1e-4，骨干网络 1e-5（可选）
3. **数据增强**：随机翻转、颜色抖动、尺寸调整
4. **优化器**：AdamW (weight_decay=1e-4)
5. **学习率调度**：StepLR (step_size=20, gamma=0.1)

## 📈 性能指标

### 评估指标

- **mAP@0.5**：IoU阈值为0.5的平均精度
- **AP per class**：每个类别的平均精度
- **Precision / Recall**：精确率和召回率

### 预期性能

| 指标 | 数值 |
|------|------|
| mAP@0.5 | 待测试 |
| 推理速度 | ~10 FPS (RTX 3090) |
| 内存占用 | ~8GB |

## 🛠️ 开发指南

### 测试单个模块

```bash
# 测试FPN
python encoder/fpn.py

# 测试解码器
python decoder/transformer_decoder.py

# 测试损失函数
python losses/varifocal_loss.py

# 测试数据加载器
python utils/data_loader.py
```

### 自定义开发

1. **添加新的特征提取器**
   - 修改 `backbone/remoteclip_backbone.py`
   - 调整 `output_layers` 参数

2. **修改检测头**
   - 编辑 `head/classification_head.py`
   - 编辑 `head/regression_head.py`

3. **调整损失函数**
   - 修改 `models/criterion.py`
   - 调整权重参数

## 📝 使用示例

### Python API

```python
from models.ova_detr import OVADETR
from config.default_config import DefaultConfig
from inference.inference_engine import InferenceEngine

# 方式1：直接使用模型
config = DefaultConfig()
model = OVADETR(config)

# 方式2：使用推理引擎
engine = InferenceEngine(
    checkpoint_path='outputs/checkpoints/best.pth',
    score_threshold=0.5
)

result = engine.predict_single('test.jpg')
vis_image = engine.visualize('test.jpg', result)
vis_image.save('result.jpg')
```

## 🐛 常见问题

### Q1: CUDA内存不足

A: 减小batch_size或图像尺寸：

```python
config.batch_size = 4
config.image_size = (600, 600)
```

### Q2: 训练速度慢

A: 增加num_workers或使用混合精度训练：

```bash
python train.py --num_workers 8 --amp
```

### Q3: mAP很低

A: 检查以下几点：
1. 数据集是否正确加载
2. 文本特征是否正确提取
3. 损失权重是否合理
4. 学习率是否过大或过小

## 🔗 参考资源

### 相关论文

1. **OVA-DETR**: Open-Vocabulary DETR with Conditional Matching
2. **DETR**: End-to-End Object Detection with Transformers
3. **RemoteCLIP**: A Vision Language Foundation Model for Remote Sensing

### 代码参考

- [DETR Official](https://github.com/facebookresearch/detr)
- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)

## 📧 联系方式

如有问题，请提交Issue或联系开发者。

## 📄 许可证

本项目遵循MIT许可证。

---

**创建日期**：2025-10-24  
**最后更新**：2025-10-24  
**版本**：v1.0

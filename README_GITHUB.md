# RemoteCLIP + OVA-DETR 目标检测系统

**开放词汇的遥感图像目标检测**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 项目简介

本项目实现了基于RemoteCLIP的遥感图像目标检测系统，包含三个不同的实验方法：

1. **Experiment1**: 两阶段检测 (Region-based)
2. **Experiment2**: 上下文引导检测 (Context-Guided Transformer)
3. **Experiment3**: OVA-DETR (Open-Vocabulary DETR) ⭐ **推荐**

### 核心特性

- ✅ 集成RemoteCLIP预训练模型
- ✅ 开放词汇目标检测能力
- ✅ 多层级文本-视觉融合
- ✅ 完整的训练和评估流程
- ✅ 支持DIOR遥感数据集

---

## 📊 实验对比

| 实验 | 模型类型 | 参数量 | 完成度 | 推荐度 |
|------|----------|--------|--------|--------|
| Experiment1 | 两阶段检测 | 102M | 90% | ⭐⭐⭐ |
| Experiment2 | 上下文引导 | ~132M | 90% | ⭐⭐⭐⭐ |
| Experiment3 | OVA-DETR | 128M | 100% | ⭐⭐⭐⭐⭐ |

---

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.0 (推荐)
```

### 安装

```bash
# 克隆仓库
git clone https://github.com/zhuyuerong/RemoteCLIP-main.git
cd RemoteCLIP-main

# 激活环境
source activate_env.sh

# 下载RemoteCLIP权重
# 将权重放在 checkpoints/ 目录下
```

### 运行 Experiment3 (推荐)

```bash
cd experiment3

# 训练
python train.py \
  --data_dir ../datasets/mini_dataset \
  --batch_size 4 \
  --epochs 20

# 评估
python evaluate.py \
  --checkpoint outputs/checkpoints/best.pth \
  --data_dir ../datasets/mini_dataset

# 推理
python inference/inference_engine.py \
  --checkpoint outputs/checkpoints/best.pth \
  --image test.jpg \
  --output result.jpg
```

---

## 📂 项目结构

```
RemoteCLIP-main/
├── experiment1/          # 两阶段检测
│   ├── stage1/          # 提议生成
│   ├── stage2/          # 目标检测
│   ├── utils/           # 评估工具 ✅
│   └── evaluate.py      # 评估脚本 ✅
├── experiment2/          # 上下文引导检测
│   ├── stage1_encoder/  # 编码器
│   ├── stage2_decoder/  # 解码器
│   ├── stage3_prediction/  # 预测头
│   ├── stage4_supervision/  # 损失函数
│   ├── utils/           # 工具模块 ✅
│   ├── train.py         # 训练框架 ✅
│   └── evaluate.py      # 评估框架 ✅
├── experiment3/          # OVA-DETR ⭐
│   ├── backbone/        # RemoteCLIP骨干
│   ├── encoder/         # FPN + Transformer
│   ├── decoder/         # Transformer解码器
│   ├── head/            # 检测头
│   ├── losses/          # 损失函数
│   ├── models/          # 完整模型
│   ├── utils/           # 工具函数
│   ├── train.py         # 训练脚本
│   ├── evaluate.py      # 评估脚本
│   └── inference/       # 推理引擎
├── datasets/
│   ├── DIOR/            # DIOR数据集
│   └── mini_dataset/    # 测试数据集(100样本)
├── checkpoints/         # RemoteCLIP权重
└── 文档/                # 详细文档
```

---

## 📈 性能指标

### 模型参数

| 实验 | 总参数 | 可训练 | 冻结 |
|------|--------|--------|------|
| Exp1 | 102M | 102M | 0M |
| Exp2 | ~132M | ~30M | ~102M |
| Exp3 | 128M | 26M | 102M |

### 推理速度 (估算)

| 实验 | CPU | GPU (RTX 3090) |
|------|-----|----------------|
| Exp1 | 3.4 FPS | ~20 FPS |
| Exp2 | ~2.0 FPS | ~10-15 FPS |
| Exp3 | ~2.5 FPS | ~10-15 FPS |

---

## 📚 文档

- [三个实验详细对比报告](三个实验详细对比报告.md) - 30页完整分析
- [Experiment3 README](experiment3/README.md) - OVA-DETR说明
- [Experiment3 使用指南](experiment3/使用指南.md) - 详细教程
- [项目完成总结](项目完成总结.md) - 工作总结

---

## 🔬 技术栈

- **深度学习**: PyTorch, OpenCLIP
- **视觉-语言模型**: RemoteCLIP
- **检测架构**: DETR, Transformer
- **数据集**: DIOR (遥感图像)
- **评估**: mAP (PASCAL VOC标准)

---

## 🎯 主要贡献

1. **RemoteCLIP + OVA-DETR** 的首次结合
2. 多层级文本-视觉融合策略
3. 完整的训练评估流程
4. 统一的mAP评估系统
5. 详细的文档和使用指南

---

## 📖 引用

```bibtex
@misc{remoteclip-ovadetr2025,
  title={RemoteCLIP + OVA-DETR for Remote Sensing Object Detection},
  author={Zhu, Yuerong},
  year={2025},
  publisher={GitHub},
  url={https://github.com/zhuyuerong/RemoteCLIP-main}
}
```

---

## 📄 许可证

MIT License

---

## 📧 联系方式

**作者**: zhuyuerong  
**邮箱**: 3074143509@qq.com  

---

## 🙏 致谢

- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) - 遥感领域CLIP模型
- [OVA-DETR](https://github.com/om-ai-lab/OV-DETR) - 开放词汇DETR
- [DIOR Dataset](http://www.escience.cn/people/gongcheng/DIOR.html) - 遥感数据集

---

**⭐ 如果这个项目对你有帮助，请给个Star！**



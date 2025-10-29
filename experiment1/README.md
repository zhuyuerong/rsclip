# Experiment1: RemoteCLIP 目标检测实验

## 📋 实验概述

这是基于RemoteCLIP的遥感目标检测实验，包含完整的数据处理、候选框生成、目标检测和优化流水线。

## 🚀 快速开始

### 方法1: 使用快速开始脚本（推荐）
```bash
# 在项目根目录运行
./快速开始实验.sh
```

### 方法2: 直接运行Python脚本
```bash
# 运行完整流水线
python experiment1/inference/inference_engine.py --image assets/airport.jpg --pipeline full

# 运行单个模块
python experiment1/inference/inference_engine.py --image assets/airport.jpg --module target_detection
```

## 📁 文件结构

```
experiment1/
├── stage1/          # 数据预处理和候选框生成
├── stage2/          # 目标检测和优化
├── inference/       # 推理引擎
├── scripts/         # 实验脚本
├── outputs/         # 输出文件
└── docs/            # 文档说明
```

## 🎯 核心功能

### Stage1: 数据预处理和候选框生成
- 遥感数据加载
- 区域采样（分层、金字塔、多阈值显著性）
- 候选框生成和分类
- 区域可视化

### Stage2: 目标检测和优化
- 对比学习目标检测
- WordNet词汇增强
- 边界框微调
- 候选框打分

### 推理引擎
- 统一推理接口
- 模型加载管理
- 输出文件管理

## 📊 实验结果

所有实验结果保存在 `outputs/` 文件夹：
- `detection_results/` - 检测结果图像
- `visualizations/` - 可视化结果
- `evaluation_tables/` - 评估表格

## 📚 详细文档

- [实验结构说明](docs/实验结构说明.md)
- [文件重新组织完成总结](docs/文件重新组织完成总结.md)

## 🔧 技术栈

- **深度学习框架**: PyTorch
- **视觉-语言模型**: RemoteCLIP (OpenCLIP)
- **图像处理**: OpenCV, Pillow
- **词汇增强**: WordNet

## 💡 使用建议

1. 首次使用建议运行快速开始脚本了解系统功能
2. 查看详细文档了解各模块的使用方法
3. 根据需要修改参数进行实验
4. 查看outputs文件夹获取实验结果

## 🔄 扩展性

本实验结构设计为模块化，便于：
- 添加新的检测方法（experiment2、experiment3等）
- 修改现有模块的实现
- 进行不同方法的对比实验
- 集成其他遥感目标检测算法

## 📞 问题反馈

如有问题，请查看：
1. `docs/实验结构说明.md` - 详细的使用说明
2. `scripts/experiments/` - 示例测试脚本
3. 各模块文件的注释和文档字符串

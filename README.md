# RemoteCLIP 遥感目标检测实验平台

基于**RemoteCLIP遥感专用视觉-语言模型**的目标检测实验平台，包含两个创新实验方法。

## 🚀 快速开始

    ```bash
# 运行交互式启动脚本
./start.sh
```

## 📁 项目结构

```
RemoteCLIP-main/
├── experiment1/          # 实验1：基于WordNet的对比学习检测
│   ├── stage1/          # 数据预处理和候选框生成
│   ├── stage2/          # 目标检测和优化
│   └── inference/       # 推理引擎
│
├── experiment2/          # 实验2：全局上下文引导检测 ⭐
│   ├── stage1_encoder/  # RemoteCLIP特征提取
│   ├── stage2_decoder/  # 上下文门控（核心创新）
│   ├── stage4_supervision/ # 全局对比损失（核心创新）
│   └── models/          # 主模型
│
├── experiment3/          # 实验3：OVA-DETR with RemoteCLIP ⭐⭐
│   ├── backbone/        # RemoteCLIP骨干网络
│   ├── encoder/         # 视觉-文本融合编码器
│   ├── decoder/         # 多层级文本引导解码器
│   └── head/            # 对比学习检测头
│
├── datasets/            # 数据集（3个+小数据集）
├── checkpoints/         # RemoteCLIP权重
└── start.sh             # 启动脚本
```

## 🎯 三个实验对比

| 特性 | Experiment1 | Experiment2 | Experiment3 |
|------|-------------|-------------|-------------|
| **核心思想** | 对比学习+WordNet | 全局对比+上下文门控 | OVA-DETR+RemoteCLIP |
| **负样本来源** | WordNet (100类) | 全局上下文（自动）⭐ | 文本采样 |
| **融合方式** | 无 | 上下文门控 | 双向融合⭐⭐ |
| **层级** | 单层 | 单层 | 多层级⭐ |
| **实现状态** | ✅ 完整 | ✅ 完整 | ✅ 核心完成 |

## 💡 使用示例

### Experiment1

```bash
# 舰船检测
python experiment1/stage2/target_detection.py --image ship.jpg --target ship

# 飞机检测
python experiment1/stage2/target_detection.py --image assets/airport.jpg --target airplane

# 完整流水线
python experiment1/inference/inference_engine.py --image assets/airport.jpg --pipeline full
```

### Experiment2

```bash
# 测试核心创新模块
python experiment2/stage4_supervision/global_contrast_loss.py
python experiment2/stage2_decoder/context_gating.py
```

### Experiment3

```bash
# 测试RemoteCLIP骨干网络
python experiment3/backbone/remoteclip_backbone.py

# 查看详细文档
cat experiment3/README.md
```

## 🌟 Experiment2 核心创新

### 1. 全局对比损失 ⭐⭐⭐
**无需外部负样本**，使用全局图像上下文作为自动负样本

$$L = -\log\left[\frac{\exp(\langle f_m, t_c \rangle / \tau)}{\exp(\langle f_m, t_c \rangle / \tau) + \exp(\langle f_m, I_g \rangle / \tau)}\right]$$

### 2. 上下文门控 ⭐⭐
使用全局上下文调制局部查询，增强目标检测能力

## 📚 文档

- `experiment1/README.md` - 实验1完整说明
- `experiment2/README.md` - 实验2架构和创新点
- `experiment1/docs/` - 详细技术文档

## 🔧 环境说明

本项目使用**RemoteCLIP**（遥感专用的CLIP模型）：
- 模型权重: `checkpoints/RemoteCLIP-RN50.pt`
- Python环境: `remoteclip/` (已包含open_clip等依赖)
- 基础模型: OpenCLIP框架

```bash
# 环境已配置，直接使用即可
# 如需额外安装：
pip install torch torchvision open_clip_torch opencv-python scipy
```

## 📊 数据集

### HRSC2016（舰船检测）
- 图片: 148张（.bmp）
- 类别: 1类（ship）
- 状态: ✅ 已整理完成
- 路径: `datasets/hrsc2016/`

### DOTA-v2.0（多类别检测）
- 标注: 5215个（.txt）
- 类别: 18类（plane, ship, harbor等）
- 状态: ✅ 标注已整理（图片待下载）
- 路径: `datasets/DOTA/DOTA-v2.0/`

### DIOR（光学遥感检测）⭐
- 图片: 23463张（.jpg）
- 标注: 23463个（水平框+旋转框，.xml）
- 类别: 20类（airplane, ship, bridge等）
- 状态: ✅ 已整理完成
- 路径: `datasets/DIOR/`

详见 `datasets/数据集整理总结.txt`

## 📊 实验结果

详见各实验文件夹的README和outputs目录。

---

**原始RemoteCLIP论文**: [RemoteCLIP: A Vision Language Foundation Model for Remote Sensing](https://arxiv.org/abs/2306.11029)

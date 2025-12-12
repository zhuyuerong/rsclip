# SurgeryCLIP + AAF + p2p传播实验

本实验实现了增强版的SurgeryCLIP，通过自适应注意力融合（AAF）和patch-to-patch（p2p）传播来改进类激活图（CAM）的生成质量。

## 概述

实验通过以下方式增强SurgeryCLIP：
1. **AAF（自适应注意力融合）**：融合SurgeryCLIP后6层的双路径注意力（VV路径和原始路径）
2. **Patch-to-patch传播**：使用融合后的注意力进行空间传播，提升CAM质量
3. **轻量级训练**：仅训练AAF参数（约13个参数），冻结CLIP模型

## 架构

```
SurgeryCLIP (冻结)
    ↓
后6层 → VV注意力 + 原始注意力
    ↓
AAF层 (可训练)
    ↓
融合注意力 [B, N², N²]
    ↓
CAM生成器
    ↓
初始CAM (patch-text相似度)
    ↓
p2p传播
    ↓
最终CAM [B, C, N, N]
```

## 项目结构

```
exp1/
├── models/
│   ├── __init__.py
│   ├── aaf.py                      # AAF层实现
│   ├── cam_generator.py            # CAM生成器（p2p传播）
│   └── surgery_aaf.py              # 主模型
├── utils/
│   ├── __init__.py
│   ├── data.py                     # 数据加载（DIOR数据集）
│   ├── visualization.py            # CAM可视化
│   └── metrics.py                  # 评估指标
├── configs/
│   └── config.yaml                 # 配置文件
├── train.py                        # 训练脚本
├── eval.py                         # 评估脚本
├── checkpoints/                    # 保存的模型权重
└── outputs/                        # 评估结果和可视化
```

## 环境设置

1. **激活虚拟环境**：
   ```bash
   source remoteclip/bin/activate
   # 或者
   conda activate remoteclip
   ```

2. **安装依赖**（如果尚未安装）：
   ```bash
   pip install torch torchvision pillow matplotlib numpy pyyaml tqdm
   ```

3. **准备DIOR数据集**：
   - 将DIOR数据集放置在 `datasets/DIOR/` 或在配置文件中指定路径
   - 数据集结构应为：
     ```
     DIOR/
     ├── images/
     │   ├── trainval/
     │   └── test/
     ├── annotations/
     │   └── horizontal/
     └── splits/
     ```

4. **下载CLIP权重**：
   - 将CLIP检查点放置在 `checkpoints/` 目录
   - 在 `configs/config.yaml` 中更新 `clip_weights_path`

## 使用方法

### 训练

```bash
cd src/experiments/exp1
source ../../../remoteclip/bin/activate  # 激活虚拟环境
python train.py
```

训练脚本将：
- 加载预训练的SurgeryCLIP
- 冻结CLIP参数
- 仅训练AAF参数
- 保存检查点到 `exp1/checkpoints/`

### 评估

```bash
python eval.py
```

评估脚本将：
- 加载训练好的AAF权重
- 在测试集上评估
- 生成CAM可视化
- 保存指标到 `exp1/outputs/`

## 配置说明

编辑 `configs/config.yaml` 以自定义：

- **模型配置**：CLIP模型名称、检查点路径、层数
- **训练配置**：训练轮数、批次大小、学习率、权重衰减
- **数据配置**：数据集名称、根路径、工作进程数
- **输出配置**：检查点和输出目录

### 配置示例

```yaml
# 模型配置
clip_model: "ViT-B/32"
clip_weights_path: "checkpoints/RemoteCLIP-ViT-B-32.pt"
num_layers: 6

# 训练配置
device: "cuda"
num_epochs: 50
batch_size: 8
learning_rate: 1.0e-4
weight_decay: 0.01

# 数据配置
dataset: "DIOR"
dataset_root: null  # null表示自动查找
num_workers: 4
```

## 核心特性

### AAF层
- 学习每层的权重（VV路径和原始路径分别学习）
- 学习两条路径的混合系数
- 仅约13个可训练参数

### CAM生成流程
1. **初始CAM**：通过patch-text相似度计算
2. **p2p传播**：使用融合注意力在空间上传播激活

### 训练策略
- 冻结整个CLIP模型（约100M参数）
- 仅训练AAF（约13个参数）
- 内存占用低
- 训练速度快

## 预期效果

- **初始CAM**：可能只有部分激活
- **p2p传播后**：激活扩散到相似patches，完整覆盖目标
- **AAF作用**：学习最优的层权重组合

## 注意事项

- 实验修改了 `clip_surgery_model.py` 以暴露注意力权重
- 确保在训练前正确加载SurgeryCLIP
- CAM可视化保存在 `outputs/visualizations/`
- 评估指标保存在 `outputs/results.json`

## 代码验证

代码已通过测试验证：

✅ **模块导入**：所有模块可以正常导入  
✅ **模型创建**：SurgeryAAF模型可以成功创建  
✅ **前向传播**：模型前向传播正常工作，可以生成CAM  
✅ **数据加载**：DIOR数据集可以正常加载  

运行测试脚本：
```bash
cd src/experiments/exp1
source ../../../remoteclip/bin/activate
python test_model.py
```

## 故障排除

1. **导入错误**：
   - 确保路径正确，SurgeryCLIP在预期位置
   - 检查 `sys.path` 设置
   - 从项目根目录运行脚本

2. **数据集未找到**：
   - 在配置中更新 `dataset_root` 或将数据集放在默认位置
   - 检查DIOR数据集目录结构

3. **CUDA内存不足**：
   - 在配置中减小 `batch_size`
   - 使用更小的模型（如ViT-B/16）

4. **注意力未收集**：
   - 确保SurgeryCLIP模型已修改以存储注意力权重
   - 检查hook是否正确注册（代码已处理延迟注册问题）

5. **模型加载失败**：
   - 检查检查点路径是否正确
   - 确保使用正确的CLIP权重文件

6. **维度不匹配**：
   - CAM生成器已自动处理图像和文本特征的维度不匹配
   - 如果仍有问题，检查CLIP模型配置

## 代码修改说明

本实验对SurgeryCLIP进行了以下修改：

1. **`clip_surgery_model.py`**：
   - `Attention.forward()`: 存储 `attn_weights_vv` 和 `attn_weights_ori`
   - `VisionTransformer`: 添加 `encode_image_with_all_tokens()` 方法

2. **新增模块**：
   - `models/aaf.py`: AAF层实现
   - `models/cam_generator.py`: CAM生成器
   - `models/surgery_aaf.py`: 主模型包装器

## 实验结果

训练完成后，可以在 `outputs/` 目录查看：
- `results.json`: 评估指标（准确率、精确率、召回率、F1）
- `visualizations/`: CAM可视化图像
- `all_metrics.npy`: 所有样本的详细指标
- `all_aps.npy`: 每类的平均精度

## 引用

如果使用本实验代码，请引用相关论文：
- SurgeryCLIP: [相关论文]
- AAF: 自适应注意力融合方法
- p2p传播: Patch-to-patch传播机制


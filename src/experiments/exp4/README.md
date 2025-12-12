# SurgeryCAM Detection System

基于SurgeryCLIP + AAF + CAMGenerator的遥感目标检测系统。

## 核心思路

- **冻结**: SurgeryCLIP + AAF + CAMGenerator
- **训练**: 轻量BoxHead (CAM → BBox回归)
- **数据**: 遥感检测数据集(DIOR)
- **目标**: Dense CAM + 精准框

## 目录结构

```
exp4/
├── models/
│   ├── surgery_cam_detector.py      # 核心检测模型
│   ├── box_head.py                  # BoxHead回归头
│   ├── multi_instance_assigner.py   # 多峰匹配分配器
│   └── owlvit_baseline.py           # OWL-ViT baseline
├── datasets/
│   └── dior_detection.py            # DIOR检测数据集加载器
├── losses/
│   └── detection_loss.py            # 检测损失函数
├── utils/
│   ├── metrics.py                   # mAP评估指标
│   └── visualization.py             # 可视化工具
├── configs/
│   ├── surgery_cam_config.yaml      # SurgeryCAM配置
│   └── owlvit_config.yaml           # OWL-ViT配置
├── train_owlvit.py                  # OWL-ViT训练脚本
├── train_surgery_cam.py             # SurgeryCAM训练脚本
├── eval.py                          # 评估脚本
├── inference.py                     # 推理脚本
└── compare_experiments.py           # 对比实验脚本
```

## 使用方法

### 1. 训练OWL-ViT Baseline

```bash
python train_owlvit.py --config configs/owlvit_config.yaml
```

### 2. 训练SurgeryCAM

```bash
python train_surgery_cam.py --config configs/surgery_cam_config.yaml
```

### 3. 评估模型

```bash
# 评估SurgeryCAM
python eval.py --model surgery_cam --checkpoint checkpoints/best_model.pth --config configs/surgery_cam_config.yaml

# 评估OWL-ViT
python eval.py --model owlvit --checkpoint dummy --config configs/owlvit_config.yaml
```

### 4. 单张图片推理

```bash
python inference.py --image path/to/image.jpg --model surgery_cam --checkpoint checkpoints/best_model.pth --config configs/surgery_cam_config.yaml
```

### 5. 对比实验

```bash
python compare_experiments.py --config configs/surgery_cam_config.yaml --experiments all --checkpoints original:checkpoints/original.pth upsampled:checkpoints/upsampled.pth
```

## 关键组件

### MultiInstanceAssigner

多峰匹配分配器，包含：
- `MultiPeakDetector`: 检测CAM上的局部极大值
- `PeakToGTMatcher`: 匈牙利算法匹配峰和GT
- `FallbackAssigner`: 处理未匹配的GT

### DetectionLoss

检测损失函数，包含：
- L1损失：框坐标回归
- GIoU损失：框形状优化
- CAM监督损失：鼓励框内高响应、框外低响应

### BoxHead

轻量级回归头，从CAM预测框参数：
- 输入: CAM [B, C, H, W]
- 输出: box参数 [B, C, H, W, 4] (Δcx, Δcy, w, h)

## 配置说明

主要配置参数：

- `surgery_clip_checkpoint`: SurgeryCLIP checkpoint路径
- `cam_resolution`: CAM分辨率（默认7，ViT-B/32）
- `upsample_cam`: 是否上采样CAM
- `lambda_l1`, `lambda_giou`, `lambda_cam`: 损失权重
- `min_peak_distance`, `min_peak_value`: 多峰检测参数

## 实验设计

1. **分辨率对比**: 7×7 vs 14×14 CAM
2. **模型对比**: OWL-ViT vs SurgeryCAM
3. **策略对比**: 单峰 vs 多峰匹配

## 测试验证

### 运行测试套件

```bash
# 运行所有测试
cd src/experiments/exp4
./tests/run_all_tests.sh

# 或单独运行
python tests/test_components.py      # 单元测试
python tests/test_integration.py     # 集成测试
python tests/test_quick_train.py     # 快速训练测试
```

测试包括：
- ✅ 组件单元测试（BoxHead, MultiPeakDetector等）
- ✅ 集成测试（训练步骤、推理流程）
- ✅ 快速训练测试（验证训练能收敛）

详见 `tests/README.md`

## 注意事项

1. 确保DIOR数据集路径正确配置
2. SurgeryCLIP checkpoint需要预先下载
3. 训练时只训练BoxHead，SurgeryCLIP+AAF+CAMGenerator保持冻结
4. 运行测试前确保所有依赖已安装


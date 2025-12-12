#!/bin/bash
# 批量运行所有实验

set -e  # 遇到错误立即退出

# 设置Python路径
PYTHON_ENV="/home/ubuntu22/.cursor/worktrees/RemoteCLIP-main/nvVcv/remoteclip/bin/python"
WORK_DIR="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/src/experiments/exp4"

cd "$WORK_DIR"

echo "=========================================="
echo "开始运行所有实验"
echo "=========================================="
echo ""

# 阶段1: 问题诊断
echo "=========================================="
echo "阶段1: 问题诊断"
echo "=========================================="

echo "实验1.1: 损失组件诊断..."
$PYTHON_ENV diagnose_loss_components.py \
    --checkpoint checkpoints/best_simple_model.pth \
    --num-batches 50 \
    --output outputs/diagnosis/loss_components_report.json

echo ""
echo "实验1.2: 梯度流诊断..."
$PYTHON_ENV check_gradients.py \
    --checkpoint checkpoints/best_simple_model.pth \
    --output outputs/diagnosis/gradient_report.json

echo ""
echo "阶段1完成！"
echo ""

# 阶段2: 快速改进实验
echo "=========================================="
echo "阶段2: 快速改进实验"
echo "=========================================="

echo "实验2.1a: 增加CAM损失权重 (20 epochs)..."
$PYTHON_ENV train_simple_surgery_cam.py \
    --config configs/exp2.1a_increase_cam_loss.yaml

echo ""
echo "实验2.1b: 降低峰值阈值 (20 epochs)..."
$PYTHON_ENV train_simple_surgery_cam.py \
    --config configs/exp2.1b_lower_peak_threshold.yaml

echo ""
echo "实验2.1c: 增加CAM生成器学习率 (20 epochs)..."
$PYTHON_ENV train_simple_surgery_cam.py \
    --config configs/exp2.1c_increase_cam_lr.yaml

echo ""
echo "实验2.1d: 调整损失权重比例 (20 epochs)..."
$PYTHON_ENV train_simple_surgery_cam.py \
    --config configs/exp2.1d_adjust_loss_weights.yaml

echo ""
echo "实验2.2: 改进正样本分配 (30 epochs)..."
$PYTHON_ENV train_exp2.2_improved_matching.py \
    --config configs/surgery_cam_config.yaml

echo ""
echo "阶段2完成！"
echo ""

# 阶段3: 深度优化实验
echo "=========================================="
echo "阶段3: 深度优化实验"
echo "=========================================="
echo "注意: 阶段3的实验需要修改损失函数和模型结构"
echo "建议先完成阶段2，然后根据结果决定是否继续"
echo ""

# 阶段4: 综合优化实验
echo "=========================================="
echo "阶段4: 综合优化实验"
echo "=========================================="
echo "注意: 阶段4需要先完成阶段2和3"
echo ""

# 阶段5: 验证和对比
echo "=========================================="
echo "阶段5: 验证和对比"
echo "=========================================="
echo "注意: 阶段5需要先完成前面的所有实验"
echo ""

echo "=========================================="
echo "所有实验运行完成！"
echo "=========================================="



#!/bin/bash
# 运行直接检测实验

cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/src/experiments/exp4

PYTHON_ENV="/home/ubuntu22/.cursor/worktrees/RemoteCLIP-main/nvVcv/remoteclip/bin/python"

echo "=========================================="
echo "直接检测方法实验"
echo "=========================================="
echo ""
echo "实验目标: 验证CAM + 图像特征 → 直接预测框的方法"
echo "训练epoch: 50"
echo ""

# 创建checkpoint目录
mkdir -p checkpoints/direct_detection

# 运行训练
echo "开始训练..."
$PYTHON_ENV train_direct_detection.py \
    --config configs/direct_detection_config.yaml \
    2>&1 | tee checkpoints/direct_detection/training.log

echo ""
echo "训练完成！"
echo "查看日志: tail -f checkpoints/direct_detection/training.log"



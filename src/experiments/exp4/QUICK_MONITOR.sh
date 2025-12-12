#!/bin/bash
# 快速监控训练进度

cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/src/experiments/exp4

echo "=========================================="
echo "改进检测器训练监控"
echo "=========================================="
echo ""

# 检查训练进程
echo "训练进程状态:"
ps aux | grep "train_improved_detector" | grep -v grep | head -1
echo ""

# 查找最新日志
LATEST_LOG=$(find checkpoints/improved_detector -name "training_improved_detector_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_LOG" ]; then
    echo "⚠️  未找到训练日志"
    exit 1
fi

echo "日志文件: $LATEST_LOG"
echo ""

# 显示最新epoch
echo "最新训练结果:"
echo "----------------------------------------"
tail -5 "$LATEST_LOG" 2>/dev/null | grep "Epoch" || echo "训练可能还在初始化..."
echo ""

# 使用Python监控脚本
PYTHON_ENV="/home/ubuntu22/.cursor/worktrees/RemoteCLIP-main/nvVcv/remoteclip/bin/python"
$PYTHON_ENV monitor_improved_training.py --log "$LATEST_LOG" 2>/dev/null || echo "监控脚本运行中..."

echo ""
echo "实时查看: tail -f $LATEST_LOG"



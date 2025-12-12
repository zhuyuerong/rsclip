#!/bin/bash
# CAL实验运行脚本

cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

# 设置PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "🚀 开始运行CAL实验"
echo "📁 工作目录: $(pwd)"
echo "🐍 Python路径: $PYTHONPATH"
echo ""

# 运行实验
python src/experiments/exp3_cal/run_all_experiments_simple.py

echo ""
echo "✅ 脚本执行完成！"

#!/bin/bash
# 实时监控CAL实验运行状态

LOG_FILE="/tmp/cal_experiments_live.log"
PROJECT_DIR="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"

cd "$PROJECT_DIR"

echo "🔍 监控CAL实验运行状态"
echo "日志文件: $LOG_FILE"
echo "按 Ctrl+C 停止监控"
echo ""

# 检查进程
if ! pgrep -f "run_final.py" > /dev/null; then
    echo "⚠️  实验进程未运行，正在启动..."
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python src/experiments/exp3_cal/run_final.py 2>&1 | tee "$LOG_FILE" &
    sleep 3
fi

# 实时监控日志
tail -f "$LOG_FILE" 2>/dev/null || {
    echo "等待日志文件生成..."
    sleep 5
    tail -f "$LOG_FILE"
}







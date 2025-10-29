#!/bin/bash
# 监控VV对比脚本的运行状态

LOG_FILE="experiment4/outputs/diagnosis/vv_vs_normal_alignment.json"

echo "监控VV机制对比脚本..."
echo "===================="

# 检查进程
PID=$(pgrep -f "compare_vv_vs_normal_patch_alignment.py")
if [ -z "$PID" ]; then
    echo "⚠️ 脚本未运行"
    echo ""
    echo "检查结果文件:"
    if [ -f "$LOG_FILE" ]; then
        echo "✅ 结果文件已生成: $LOG_FILE"
        echo ""
        echo "结果预览:"
        cat "$LOG_FILE" | head -30
    else
        echo "❌ 结果文件不存在"
    fi
else
    echo "✅ 脚本正在运行 (PID: $PID)"
    echo ""
    echo "使用以下命令查看完整输出:"
    echo "  tail -f /proc/$PID/fd/1"
fi

echo ""
echo "检查GPU使用:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1


#!/bin/bash
# 快速激活remoteclip环境

echo "🐍 激活remoteclip虚拟环境..."

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/remoteclip/bin/activate"

if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "✅ remoteclip环境已激活"
    echo ""
    echo "环境信息:"
    echo "  Python: $(which python)"
    echo "  版本: $(python --version)"
    echo ""
    echo "📦 关键包:"
    python -c "import torch; print('  PyTorch:', torch.__version__)" 2>/dev/null || echo "  PyTorch: 未安装"
    python -c "import open_clip; print('  OpenCLIP:', open_clip.__version__)" 2>/dev/null || echo "  OpenCLIP: 未安装"
    python -c "import cv2; print('  OpenCV:', cv2.__version__)" 2>/dev/null || echo "  OpenCV: 未安装"
    echo ""
    python -c "import torch; print('  CUDA可用:', torch.cuda.is_available() if hasattr(torch, 'cuda') else 'N/A')" 2>/dev/null
    echo ""
    echo "🚀 现在可以运行实验了！"
    echo ""
    echo "快速命令:"
    echo "  python check_cuda.py          # 检查CUDA状态"
    echo "  python experiment1/...        # 运行实验1"
    echo "  python experiment2/...        # 运行实验2"
    echo "  python experiment3/...        # 运行实验3"
else
    echo "❌ 未找到remoteclip虚拟环境"
    echo "路径应该在: $VENV_PATH"
    echo "当前目录: $(pwd)"
fi

#!/bin/bash
# 快速激活remoteclip环境

echo "🐍 激活remoteclip虚拟环境..."

if [ -f "remoteclip/bin/activate" ]; then
    source remoteclip/bin/activate
    echo "✅ remoteclip环境已激活"
    echo ""
    echo "环境信息:"
    echo "  Python: $(which python)"
    echo "  版本: $(python --version)"
    echo ""
    echo "📦 关键包:"
    python -c "import torch; print('  PyTorch:', torch.__version__)"
    python -c "import open_clip; print('  OpenCLIP:', open_clip.__version__)"
    python -c "import cv2; print('  OpenCV:', cv2.__version__)"
    echo ""
    echo "🚀 现在可以运行实验了！"
    echo ""
    echo "快速命令:"
    echo "  ./start.sh                    # 交互式菜单"
    echo "  python experiment1/...        # 运行实验1"
    echo "  python experiment2/...        # 运行实验2"
else
    echo "❌ 未找到remoteclip虚拟环境"
    echo "路径应该在: remoteclip/bin/activate"
fi

#!/bin/bash
# ============================================================
# Exp9 Pseudo Query 环境设置脚本
# ============================================================
# 使用方法:
#   source scripts/setup_env.sh
#   或在运行前: conda activate samrs && source scripts/setup_env.sh
# ============================================================

# 1. 激活conda环境
echo "=== 设置 Pseudo Query 实验环境 ==="
echo ""

# 检查是否在samrs环境
CURRENT_ENV=$(conda info --envs | grep "*" | awk '{print $1}')
if [ "$CURRENT_ENV" != "samrs" ]; then
    echo "⚠️  当前环境: $CURRENT_ENV"
    echo "   请先激活samrs环境: conda activate samrs"
    echo ""
fi

# 2. 设置库路径 (Deformable DETR CUDA算子需要)
export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
echo "✅ LD_LIBRARY_PATH 已设置"

# 3. 设置Python路径
PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"
echo "✅ PYTHONPATH 已设置"

# 4. 验证环境
echo ""
echo "=== 环境验证 ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

try:
    import MultiScaleDeformableAttention as MSDA
    print('✅ Deformable DETR CUDA算子: 可用')
except ImportError as e:
    print(f'❌ Deformable DETR CUDA算子: 不可用 ({e})')
"

echo ""
echo "=== 环境设置完成 ==="

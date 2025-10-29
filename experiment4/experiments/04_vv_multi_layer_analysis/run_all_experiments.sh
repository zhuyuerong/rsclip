#!/bin/bash
# 运行所有VV多层分析实验

set -e  # 遇到错误立即退出

PYTHON="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/ovadetr_env/bin/python3.9"
BASE_DIR="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
EXP_DIR="$BASE_DIR/experiment4/experiments/04_vv_multi_layer_analysis"

echo "======================================"
echo "VV机制与多层特征分析 - 完整实验流程"
echo "======================================"

cd "$BASE_DIR"

# 实验1: Patch相似度矩阵分析（Layer 12）
echo ""
echo "========================================================================"
echo "实验1: Patch相似度矩阵分析 (Layer 12)"
echo "========================================================================"
$PYTHON "$EXP_DIR/patch_similarity_matrix.py" --dataset datasets/mini_dataset --layer 12

# 实验2: 多层特征分析（1/6/9/12层）
echo ""
echo "========================================================================"
echo "实验2: 多层特征分析 (Layers 1/6/9/12)"
echo "========================================================================"
$PYTHON "$EXP_DIR/layer_analysis.py" \
    --dataset datasets/mini_dataset \
    --layers 1 6 9 12 \
    --max-samples 10 \
    --use-surgery

# 实验3: 文本引导VV^T热图（1/3/6/9层）
echo ""
echo "========================================================================"
echo "实验3: 文本引导VV^T热图 (Layers 1/3/6/9)"
echo "========================================================================"
$PYTHON "$EXP_DIR/text_guided_vvt.py" \
    --dataset datasets/mini_dataset \
    --layers 1 3 6 9 \
    --max-samples 10

# 实验4: 3种模式对比（标准/Surgery/Surgery+VV）
echo ""
echo "========================================================================"
echo "实验4: 3种模式对比 (Seen/Unseen数据集)"
echo "========================================================================"
echo "注意: 此实验需要较长时间，建议单独运行"
echo "命令: $PYTHON $EXP_DIR/compare_three_modes.py --quick-test"

echo ""
echo "======================================"
echo "实验流程完成！"
echo "======================================"
echo ""
echo "结果位置:"
echo "  - Patch相似度矩阵: $EXP_DIR/outputs/layer_analysis/"
echo "  - 多层特征分析: $EXP_DIR/outputs/layer_analysis/"
echo "  - VV^T热图: $EXP_DIR/outputs/vvt_heatmaps/"
echo "  - 模式对比: $EXP_DIR/outputs/mode_comparison/"
echo ""


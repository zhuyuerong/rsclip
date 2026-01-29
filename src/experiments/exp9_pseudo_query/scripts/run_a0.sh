#!/bin/bash
# ============================================================
# A0 Baseline: Deformable DETR on DIOR
# ============================================================
# 
# 使用方法:
#   cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
#   conda activate samrs
#   source src/experiments/exp9_pseudo_query/scripts/setup_env.sh
#   bash src/experiments/exp9_pseudo_query/scripts/run_a0.sh
#
# ============================================================

set -e

# 项目路径
PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
cd $PROJECT_ROOT

# 设置环境变量
export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"

# 实验配置
EXP_NAME="a0_baseline"
OUTPUT_DIR="outputs/exp9_pseudo_query/${EXP_NAME}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR}_${TIMESTAMP}"

echo "============================================================"
echo "A0 Baseline: Deformable DETR on DIOR"
echo "============================================================"
echo "Output: ${OUTPUT_DIR}"
echo "Time: ${TIMESTAMP}"
echo "============================================================"

# 运行训练
python src/experiments/exp9_pseudo_query/scripts/train_a0_baseline.py \
    --dior_path datasets/DIOR \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --epochs 50 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --lr_drop 40 \
    --num_queries 300 \
    --hidden_dim 256 \
    --num_feature_levels 4 \
    --enc_layers 6 \
    --dec_layers 6 \
    --eval_epochs 1 5 10 20 30 40 50 \
    --save_epochs 10 20 30 40 50 \
    --num_workers 4 \
    --seed 42

echo "============================================================"
echo "✅ A0 Baseline training completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "============================================================"

#!/bin/bash
# ============================================================
# A2: Teacher Proposals → Pseudo Query (管线自检)
# ============================================================
# 
# 目的: 验证pseudo query注入机制是否正确
# 预期: 比A0更快收敛，尤其在early epoch
# 
# 使用方法:
#   conda activate samrs
#   bash src/experiments/exp9_pseudo_query/scripts/run_a2_teacher.sh
#
# ============================================================

set -e

PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
cd $PROJECT_ROOT

export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"

EXP_NAME="a2_teacher"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/exp9_pseudo_query/${EXP_NAME}_${TIMESTAMP}"

echo "============================================================"
echo "A2: Teacher Proposals -> Pseudo Query"
echo "============================================================"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

/home/ubuntu22/anaconda3/envs/samrs/bin/python \
    src/experiments/exp9_pseudo_query/scripts/train_pseudo_query.py \
    --exp_type A2 \
    --dior_path datasets/DIOR \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --epochs 50 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --lr_drop 40 \
    --num_queries 300 \
    --num_pseudo_queries 100 \
    --num_learnable_queries 200 \
    --mix_mode concat \
    --pool_mode heatmap_weighted \
    --eval_epochs 1 5 10 20 30 40 50 \
    --save_epochs 10 20 30 40 50 \
    --num_workers 4 \
    --seed 42

echo "============================================================"
echo "✅ A2 Training completed!"
echo "============================================================"

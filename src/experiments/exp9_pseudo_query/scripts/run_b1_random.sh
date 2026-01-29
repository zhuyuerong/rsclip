#!/bin/bash
# ============================================================
# B1: Random Query (证伪实验)
# ============================================================
# 
# 目的: 证明不是"多加queries就行"
# 预期: 明显劣于A2/A3，甚至可能比A0还差
# 
# 如果B1≈A3: 说明A3增益只是"多了queries"，方法不成立
#
# ============================================================

set -e

PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
cd $PROJECT_ROOT

export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"

EXP_NAME="b1_random"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/exp9_pseudo_query/${EXP_NAME}_${TIMESTAMP}"

echo "============================================================"
echo "B1: Random Query (Falsification)"
echo "============================================================"

/home/ubuntu22/anaconda3/envs/samrs/bin/python \
    src/experiments/exp9_pseudo_query/scripts/train_pseudo_query.py \
    --exp_type B1 \
    --dior_path datasets/DIOR \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --epochs 50 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --num_pseudo_queries 100 \
    --num_learnable_queries 200 \
    --mix_mode concat \
    --eval_epochs 1 5 10 20 30 40 50 \
    --save_epochs 10 50 \
    --num_workers 4 \
    --seed 42

echo "✅ B1 completed!"

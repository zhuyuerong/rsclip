#!/bin/bash
# ============================================================
# B2: Shuffled Heatmap (证伪实验)
# ============================================================
# 
# 目的: 证明是"图像相关的空间证据"
# 预期: 相对A3有显著下降
# 
# 如果B2不掉: A3的因果链不成立
#
# ============================================================

set -e

PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
cd $PROJECT_ROOT

export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"

EXP_NAME="b2_shuffled"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/exp9_pseudo_query/${EXP_NAME}_${TIMESTAMP}"

echo "============================================================"
echo "B2: Shuffled Heatmap (Falsification)"
echo "============================================================"

/home/ubuntu22/anaconda3/envs/samrs/bin/python \
    src/experiments/exp9_pseudo_query/scripts/train_pseudo_query.py \
    --exp_type B2 \
    --dior_path datasets/DIOR \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --epochs 50 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --num_pseudo_queries 100 \
    --num_learnable_queries 200 \
    --mix_mode concat \
    --pool_mode heatmap_weighted \
    --heatmap_cache_dir outputs/heatmap_cache/dior_trainval \
    --checkpoint_path checkpoints/RemoteCLIP-ViT-B-32.pt \
    --generate_heatmap_on_fly \
    --eval_epochs 1 5 10 20 30 40 50 \
    --save_epochs 10 50 \
    --num_workers 2 \
    --seed 42

echo "✅ B2 completed!"

#!/bin/bash
# ============================================================
# A3: Heatmap → Pseudo Query (核心方法)
# ============================================================
# 
# 目的: 验证vv-attention热图引导的pseudo query有效性
# 预期: 
#   - 比A0更快收敛
#   - 密集小目标(ship/vehicle)Recall上升更明显
#   - 可能带来FP(背景高响应)
# 
# 使用方法:
#   conda activate samrs
#   bash src/experiments/exp9_pseudo_query/scripts/run_a3_heatmap.sh
#
# ============================================================

set -e

PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
cd $PROJECT_ROOT

export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"

EXP_NAME="a3_heatmap"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/exp9_pseudo_query/${EXP_NAME}_${TIMESTAMP}"

echo "============================================================"
echo "A3: Heatmap -> Pseudo Query"
echo "============================================================"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

/home/ubuntu22/anaconda3/envs/samrs/bin/python \
    src/experiments/exp9_pseudo_query/scripts/train_pseudo_query.py \
    --exp_type A3 \
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
    --heatmap_cache_dir outputs/heatmap_cache/dior_trainval \
    --checkpoint_path checkpoints/RemoteCLIP-ViT-B-32.pt \
    --generate_heatmap_on_fly \
    --eval_epochs 1 5 10 20 30 40 50 \
    --save_epochs 10 20 30 40 50 \
    --num_workers 2 \
    --seed 42

echo "============================================================"
echo "✅ A3 Training completed!"
echo "============================================================"

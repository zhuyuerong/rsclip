#!/bin/bash
# ============================================================
# 顺序运行所有实验
# ============================================================

set -e

PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
cd $PROJECT_ROOT/src/experiments/exp9_pseudo_query

export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"

echo "============================================================"
echo "顺序运行实验: A0 → A2 → A3 → B1 → B2"
echo "============================================================"

# 检查A0是否完成
echo ""
echo "⏳ 等待A0完成..."
while ps aux | grep -q "[t]rain_a0_baseline"; do
    echo "   A0仍在运行，等待中... ($(date +%H:%M:%S))"
    sleep 300  # 每5分钟检查一次
done

echo "   ✅ A0已完成！"
sleep 10

# 运行A2
echo ""
echo "🚀 启动A2 (Teacher proposals)..."
bash scripts/run_a2_teacher.sh
echo "   ✅ A2已完成！"

# 运行A3
echo ""
echo "🚀 启动A3 (Heatmap pseudo)..."
bash scripts/run_a3_heatmap.sh
echo "   ✅ A3已完成！"

# 运行B1
echo ""
echo "🚀 启动B1 (Random query)..."
bash scripts/run_b1_random.sh
echo "   ✅ B1已完成！"

# 运行B2
echo ""
echo "🚀 启动B2 (Shuffled heatmap)..."
bash scripts/run_b2_shuffled.sh
echo "   ✅ B2已完成！"

echo ""
echo "============================================================"
echo "✅ 所有实验完成！"
echo "============================================================"
echo ""
echo "📊 对比分析:"
python scripts/compare_experiments.py \
    --exp_dirs outputs/exp9_pseudo_query/a0_* \
               outputs/exp9_pseudo_query/a2_* \
               outputs/exp9_pseudo_query/a3_* \
               outputs/exp9_pseudo_query/b1_* \
               outputs/exp9_pseudo_query/b2_*

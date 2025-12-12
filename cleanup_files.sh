#!/bin/bash
# 清理临时文件、缓存和不需要的文件

cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

echo "🧹 开始清理文件..."
echo ""

# 1. 清理Python缓存（不包括虚拟环境）
echo "1️⃣ 清理Python缓存文件..."
find src -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find src -name "*.pyc" -delete 2>/dev/null
find src -name "*.pyo" -delete 2>/dev/null
find src -name "*.pyd" -delete 2>/dev/null
echo "   ✅ Python缓存已清理"
echo ""

# 2. 清理临时文件
echo "2️⃣ 清理临时文件..."
find . -name ".DS_Store" -delete 2>/dev/null
find . -name "Thumbs.db" -delete 2>/dev/null
find . -name "*.tmp" -delete 2>/dev/null
find . -name "*.temp" -delete 2>/dev/null
echo "   ✅ 临时文件已清理"
echo ""

# 3. 清理训练日志文件（保留最新的）
echo "3️⃣ 清理训练日志文件..."
# 保留最新的日志，删除旧的
find src/experiments/exp4/checkpoints -name "*.log" -type f -delete 2>/dev/null
# 保留根目录下的最新日志
rm -f src/experiments/exp4/evaluation_output.log
rm -f src/experiments/exp4/evaluation_simple.log
rm -f src/experiments/exp4/training_simple_surgery_cam_live.log
# 保留最新的训练日志
# training_gt_class_localization.log 和 training_seen_classes_fixed.log 保留
echo "   ✅ 旧日志文件已清理"
echo ""

# 4. 清理旧的训练权重（保留最新的和best模型）
echo "4️⃣ 清理旧的训练权重..."
# 删除checkpoints根目录下的旧checkpoint文件（保留best）
rm -f src/experiments/exp4/checkpoints/checkpoint_epoch_*.pth
rm -f src/experiments/exp4/checkpoints/backup_*/checkpoint_epoch_*.pth
# 删除backup目录中的旧权重
rm -rf src/experiments/exp4/checkpoints/backup_20251205_153409
rm -rf src/experiments/exp4/checkpoints/backup_before_seen_training
# 删除exp2.1系列实验的旧权重（如果不再需要）
# rm -rf src/experiments/exp4/checkpoints/exp2.1*
echo "   ✅ 旧权重文件已清理"
echo ""

# 5. 清理重复/临时的MD文件
echo "5️⃣ 清理重复/临时的MD文档..."
# 保留重要的文档，删除临时状态文档
rm -f src/experiments/exp4/INFERENCE_STARTED.md
rm -f src/experiments/exp4/TRAINING_STARTED.md
rm -f src/experiments/exp4/TRAINING_STATUS.md
rm -f src/experiments/exp4/TRAINING_STATUS_FIXED.md
rm -f src/experiments/exp4/EVALUATION_IN_PROGRESS.md
rm -f src/experiments/exp4/IMPLEMENTATION_STATUS.md
rm -f src/experiments/exp4/EXPERIMENT_STATUS.md
rm -f src/experiments/exp4/CONTINUED_TRAINING.md
rm -f src/experiments/exp4/TRAINING_PROGRESS_CHECK.md
rm -f src/experiments/exp4/TRAINING_PROGRESS_ANALYSIS.md
rm -f src/experiments/exp4/TRAINING_RESTART_SUMMARY.md
rm -f src/experiments/exp4/TRAINING_FIX_SUMMARY.md
rm -f src/experiments/exp4/EVALUATION_ANALYSIS.md
rm -f src/experiments/exp4/EVALUATION_RESULTS_ANALYSIS.md
rm -f src/experiments/exp4/DIAGNOSIS_RESULTS.md
rm -f src/experiments/exp4/DIAGNOSIS_RESULTS_ANALYSIS.md
rm -f src/experiments/exp4/DIAGNOSIS_ANALYSIS.md
rm -f src/experiments/exp4/PRIORITY_FIXES_SUMMARY.md
rm -f src/experiments/exp4/IMPLEMENTATION_COMPLETE.md
rm -f src/experiments/exp4/EXPERIMENTS_SUMMARY_AND_DEFECTS.md
echo "   ✅ 临时MD文档已清理"
echo ""

# 6. 清理wandb缓存（如果不需要）
echo "6️⃣ 检查wandb缓存..."
if [ -d "wandb" ]; then
    echo "   ⚠️  wandb目录存在，保留（如需清理可手动删除）"
fi
echo ""

echo "✅ 清理完成！"
echo ""
echo "📊 清理总结："
echo "   - Python缓存文件"
echo "   - 临时文件（.DS_Store等）"
echo "   - 旧训练日志"
echo "   - 旧训练权重（保留best模型）"
echo "   - 临时状态MD文档"
echo ""
echo "💾 保留的重要文件："
echo "   - 最新的训练日志"
echo "   - best模型权重"
echo "   - 重要的MD文档（README、实验总结等）"


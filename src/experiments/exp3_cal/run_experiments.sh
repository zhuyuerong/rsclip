#!/bin/bash
# CAL实验批量运行脚本

# 激活conda环境
echo "🔧 激活conda环境: remoteclip"
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate remoteclip

# 检查环境
if [ $? -ne 0 ]; then
    echo "⚠️  无法激活conda环境，尝试使用当前Python环境"
fi

# 进入项目目录
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

# 设置测试图像和类别
# 默认使用3张测试图像
IMAGES=(
    "datasets/mini-DIOR/test/images/00679.jpg"
    "datasets/mini-DIOR/test/images/15944.jpg"
    "datasets/mini-DIOR/test/images/16323.jpg"
)

CLASSES=(
    "vehicle"
    "airplane"
    "ship"
)

# 如果图像不存在，尝试查找其他图像
if [ ! -f "${IMAGES[0]}" ]; then
    echo "🔍 查找测试图像..."
    FIRST_IMAGE=$(find datasets -name "*.jpg" -o -name "*.png" 2>/dev/null | head -1)
    if [ -n "$FIRST_IMAGE" ]; then
        IMAGES=("$FIRST_IMAGE")
        CLASSES=("vehicle")
        echo "   使用图像: $FIRST_IMAGE"
    else
        echo "❌ 未找到测试图像，请手动指定"
        exit 1
    fi
fi

# 运行所有实验
echo "🚀 开始运行所有CAL实验"
echo "📋 实验总数: 12"
echo "🖼️  测试图像: ${#IMAGES[@]}"
echo ""

python src/experiments/exp3_cal/run_all_experiments.py \
    --images "${IMAGES[@]}" \
    --classes "${CLASSES[@]}" \
    --checkpoint checkpoints/ViT-B-32.pt \
    --device cuda \
    --output-dir outputs/exp3_cal

echo ""
echo "✅ 脚本执行完成！"







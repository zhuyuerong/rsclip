#!/bin/bash
# 完整流程：运行评估 + 可视化

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "完整评估和可视化流程"
echo "=========================================="

# 1. 运行评估
echo ""
echo "步骤1: 运行实际评估..."
echo "----------------------------------------"
python run_evaluation.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/surgery_cam_config.yaml \
    --output outputs/evaluation_results.json \
    --num-samples 100  # 快速测试，可以改为None进行完整评估

if [ $? -ne 0 ]; then
    echo "❌ 评估失败"
    exit 1
fi

echo ""
echo "✅ 评估完成！"

# 2. 查找测试图像
echo ""
echo "步骤2: 查找测试图像..."
echo "----------------------------------------"

# 尝试多个可能的路径
IMAGE_PATHS=(
    "../../datasets/DIOR/images/test"
    "../../../datasets/DIOR/images/test"
    "datasets/DIOR/images/test"
)

IMAGE_FILE=""
for path in "${IMAGE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        # 查找包含ship的图像
        IMAGE_FILE=$(find "$path" -name "*.jpg" -type f | head -1)
        if [ -n "$IMAGE_FILE" ]; then
            echo "✅ 找到图像: $IMAGE_FILE"
            break
        fi
    fi
done

if [ -z "$IMAGE_FILE" ]; then
    echo "⚠️  未找到测试图像，请手动指定图像路径"
    echo "使用方法:"
    echo "  python visualize_with_real_results.py --image <path/to/image.jpg>"
    exit 0
fi

# 3. 运行可视化
echo ""
echo "步骤3: 生成可视化..."
echo "----------------------------------------"
python visualize_with_real_results.py \
    --image "$IMAGE_FILE" \
    --checkpoint checkpoints/best_model.pth \
    --config configs/surgery_cam_config.yaml \
    --results outputs/evaluation_results.json \
    --class-name ship \
    --output-dir outputs/visualizations

if [ $? -ne 0 ]; then
    echo "❌ 可视化失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 全部完成！"
echo "=========================================="
echo "评估结果: outputs/evaluation_results.json"
echo "可视化结果: outputs/visualizations/"
echo ""



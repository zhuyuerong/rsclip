#!/bin/bash
# -*- coding: utf-8 -*-
# OVA-DETR 快速启动脚本

set -e

echo "========================================"
echo "OVA-DETR with RemoteCLIP - 快速启动"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/home/ubuntu22/Projects/RemoteCLIP-main"
EXPERIMENT_DIR="$PROJECT_ROOT/experiment3"
DATA_DIR="$PROJECT_ROOT/datasets/DIOR"

cd $EXPERIMENT_DIR

# 检查数据集
check_dataset() {
    echo -e "\n${YELLOW}[1/4] 检查数据集...${NC}"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}错误: 未找到DIOR数据集${NC}"
        echo "请确保数据集位于: $DATA_DIR"
        exit 1
    fi
    
    if [ ! -d "$DATA_DIR/images/trainval" ] || [ ! -d "$DATA_DIR/annotations/horizontal" ]; then
        echo -e "${RED}错误: 数据集结构不完整${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ 数据集检查通过${NC}"
}

# 检查RemoteCLIP权重
check_weights() {
    echo -e "\n${YELLOW}[2/4] 检查RemoteCLIP权重...${NC}"
    
    CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
    
    if [ ! -f "$CHECKPOINT_DIR/RemoteCLIP-RN50.pt" ]; then
        echo -e "${RED}错误: 未找到RemoteCLIP-RN50.pt${NC}"
        echo "请确保权重文件位于: $CHECKPOINT_DIR/"
        exit 1
    fi
    
    echo -e "${GREEN}✓ RemoteCLIP权重检查通过${NC}"
}

# 测试模块
test_modules() {
    echo -e "\n${YELLOW}[3/4] 测试核心模块...${NC}"
    
    # 测试数据加载器
    echo "  - 测试数据加载器..."
    python -c "
import sys
sys.path.append('$EXPERIMENT_DIR')
from utils.data_loader import DiorDataset
dataset = DiorDataset(root_dir='$DATA_DIR', split='train')
print(f'    训练集大小: {len(dataset)}')
" || exit 1
    
    echo -e "${GREEN}✓ 模块测试通过${NC}"
}

# 显示训练命令
show_training() {
    echo -e "\n${YELLOW}[4/4] 训练命令示例${NC}"
    
    cat << EOF

🚀 开始训练:

# 快速训练（小批次）
python train.py \\
  --data_dir $DATA_DIR \\
  --output_dir ./outputs \\
  --batch_size 4 \\
  --epochs 10 \\
  --num_workers 4

# 完整训练
python train.py \\
  --data_dir $DATA_DIR \\
  --output_dir ./outputs \\
  --batch_size 8 \\
  --epochs 50 \\
  --lr 1e-4 \\
  --num_workers 8

📊 评估模型:

python evaluate.py \\
  --checkpoint outputs/checkpoints/best.pth \\
  --data_dir $DATA_DIR \\
  --output evaluation_results.json

🔍 推理示例:

python inference/inference_engine.py \\
  --checkpoint outputs/checkpoints/best.pth \\
  --image $DATA_DIR/images/trainval/00001.jpg \\
  --output result.jpg \\
  --score_threshold 0.5

EOF
}

# 主函数
main() {
    check_dataset
    check_weights
    test_modules
    show_training
    
    echo -e "\n${GREEN}========================================"
    echo "✓ 所有检查通过！"
    echo "========================================${NC}\n"
}

# 运行
main


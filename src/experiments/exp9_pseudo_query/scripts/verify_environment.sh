#!/bin/bash
# ============================================================
# 环境验证脚本
# ============================================================
# 
# 用途: 在运行实验前验证所有依赖和配置
# 使用: bash scripts/verify_environment.sh
#
# ============================================================

set -e

PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
cd $PROJECT_ROOT

echo "============================================================"
echo "🔍 Exp9 Pseudo Query 环境验证"
echo "============================================================"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# ============================================================
# 1. Python环境检查
# ============================================================
echo ""
echo "1️⃣  检查Python环境..."

if conda env list | grep -q "samrs"; then
    check_pass "Conda环境 'samrs' 存在"
else
    check_fail "Conda环境 'samrs' 不存在"
    exit 1
fi

# 激活环境
source /home/ubuntu22/anaconda3/etc/profile.d/conda.sh
conda activate samrs

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "   Python版本: $PYTHON_VERSION"

# ============================================================
# 2. PyTorch和CUDA检查
# ============================================================
echo ""
echo "2️⃣  检查PyTorch和CUDA..."

CUDA_CHECK=$(python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Version:', torch.version.cuda, '| Devices:', torch.cuda.device_count())" 2>&1)
echo "   $CUDA_CHECK"

if echo "$CUDA_CHECK" | grep -q "True"; then
    check_pass "CUDA可用"
else
    check_fail "CUDA不可用"
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>&1)
echo "   PyTorch版本: $TORCH_VERSION"

# ============================================================
# 3. Deformable Attention检查
# ============================================================
echo ""
echo "3️⃣  检查Deformable Attention编译..."

export LD_LIBRARY_PATH=/home/ubuntu22/anaconda3/envs/samrs/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"

DEFORM_CHECK=$(python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}/external/Deformable-DETR')
try:
    from models.ops.modules import MSDeformAttn
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')
" 2>&1)

if echo "$DEFORM_CHECK" | grep -q "OK"; then
    check_pass "Deformable Attention编译成功"
else
    check_fail "Deformable Attention编译失败: $DEFORM_CHECK"
    echo "   请运行: cd external/Deformable-DETR/models/ops && bash make.sh"
    exit 1
fi

# ============================================================
# 4. 数据集检查
# ============================================================
echo ""
echo "4️⃣  检查DIOR数据集..."

DIOR_PATH="${PROJECT_ROOT}/datasets/DIOR"

if [ -d "$DIOR_PATH" ]; then
    check_pass "DIOR数据集目录存在"
    
    # 检查子目录
    if [ -d "$DIOR_PATH/JPEGImages" ]; then
        IMAGE_COUNT=$(ls $DIOR_PATH/JPEGImages/*.jpg 2>/dev/null | wc -l)
        echo "   图像数量: $IMAGE_COUNT"
        if [ $IMAGE_COUNT -gt 0 ]; then
            check_pass "图像文件存在"
        else
            check_fail "图像文件不存在"
        fi
    else
        check_fail "JPEGImages目录不存在"
    fi
    
    if [ -d "$DIOR_PATH/Annotations" ]; then
        ANNO_COUNT=$(ls $DIOR_PATH/Annotations/*.xml 2>/dev/null | wc -l)
        echo "   标注数量: $ANNO_COUNT"
        if [ $ANNO_COUNT -gt 0 ]; then
            check_pass "标注文件存在"
        else
            check_fail "标注文件不存在"
        fi
    else
        check_fail "Annotations目录不存在"
    fi
    
    if [ -f "$DIOR_PATH/ImageSets/Main/train.txt" ]; then
        TRAIN_COUNT=$(wc -l < $DIOR_PATH/ImageSets/Main/train.txt)
        echo "   训练集: $TRAIN_COUNT 张"
        check_pass "训练集列表存在"
    else
        check_fail "训练集列表不存在"
    fi
    
else
    check_fail "DIOR数据集目录不存在: $DIOR_PATH"
    exit 1
fi

# ============================================================
# 5. 数据集加载测试
# ============================================================
echo ""
echo "5️⃣  测试数据集加载..."

DATASET_CHECK=$(python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
sys.path.insert(0, '${PROJECT_ROOT}/external/Deformable-DETR')
try:
    from src.experiments.exp9_pseudo_query.datasets import build_dior_dataset
    dataset = build_dior_dataset(root='${DIOR_PATH}', image_set='train', image_size=800)
    print(f'OK: {len(dataset)} samples')
except Exception as e:
    print(f'FAIL: {e}')
" 2>&1)

if echo "$DATASET_CHECK" | grep -q "OK"; then
    check_pass "数据集加载成功: $(echo $DATASET_CHECK | awk '{print $2, $3}')"
else
    check_fail "数据集加载失败: $DATASET_CHECK"
fi

# ============================================================
# 6. 热图检查 (A3/B2需要)
# ============================================================
echo ""
echo "6️⃣  检查热图相关..."

CHECKPOINT_PATH="${PROJECT_ROOT}/checkpoints/RemoteCLIP-ViT-B-32.pt"
if [ -f "$CHECKPOINT_PATH" ]; then
    check_pass "RemoteCLIP权重存在"
else
    check_warn "RemoteCLIP权重不存在 (A3/B2实验需要)"
    echo "   路径: $CHECKPOINT_PATH"
fi

HEATMAP_CACHE="${PROJECT_ROOT}/outputs/heatmap_cache/dior_trainval"
if [ -d "$HEATMAP_CACHE" ]; then
    CACHE_COUNT=$(ls $HEATMAP_CACHE/*.npy 2>/dev/null | wc -l)
    if [ $CACHE_COUNT -gt 0 ]; then
        check_pass "热图缓存存在: $CACHE_COUNT 个文件"
    else
        check_warn "热图缓存目录为空 (将在线生成)"
    fi
else
    check_warn "热图缓存目录不存在 (将在线生成)"
fi

# ============================================================
# 7. 输出目录检查
# ============================================================
echo ""
echo "7️⃣  检查输出目录..."

OUTPUT_DIR="${PROJECT_ROOT}/outputs/exp9_pseudo_query"
if [ -d "$OUTPUT_DIR" ]; then
    check_pass "输出目录存在"
else
    mkdir -p "$OUTPUT_DIR"
    check_pass "创建输出目录"
fi

# ============================================================
# 8. 模块导入测试
# ============================================================
echo ""
echo "8️⃣  测试模块导入..."

MODULE_CHECK=$(python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
sys.path.insert(0, '${PROJECT_ROOT}/external/Deformable-DETR')

modules = [
    'src.experiments.exp9_pseudo_query.models.heatmap_query_gen',
    'src.experiments.exp9_pseudo_query.models.query_injection',
    'src.experiments.exp9_pseudo_query.datasets',
]

failed = []
for mod in modules:
    try:
        __import__(mod)
    except Exception as e:
        failed.append(f'{mod}: {e}')

if failed:
    print('FAIL')
    for f in failed:
        print(f)
else:
    print('OK')
" 2>&1)

if echo "$MODULE_CHECK" | grep -q "OK"; then
    check_pass "所有模块导入成功"
else
    check_fail "模块导入失败:"
    echo "$MODULE_CHECK"
fi

# ============================================================
# 9. GPU状态
# ============================================================
echo ""
echo "9️⃣  GPU状态..."

if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
    echo ""
    check_pass "GPU信息获取成功"
else
    check_warn "nvidia-smi不可用"
fi

# ============================================================
# 总结
# ============================================================
echo ""
echo "============================================================"
echo "✅ 环境验证完成！"
echo "============================================================"
echo ""
echo "📌 下一步:"
echo "   1. 运行A0 baseline: bash scripts/run_a0.sh"
echo "   2. 监控训练: tail -f outputs/exp9_pseudo_query/a0_training.log"
echo "   3. 检查GPU: nvidia-smi"
echo ""

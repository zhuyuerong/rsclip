#!/bin/bash
# Exp9: 自动安装Deformable DETR依赖
#
# 功能:
# 1. 克隆Deformable DETR官方仓库
# 2. 编译CUDA算子
# 3. 验证安装

set -e

# ============================================================================
# 配置
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP9_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$EXP9_DIR/../../../../.." && pwd)"
EXTERNAL_DIR="$PROJECT_ROOT/external"
DETR_DIR="$EXTERNAL_DIR/Deformable-DETR"

DETR_REPO="https://github.com/fundamentalvision/Deformable-DETR.git"

# ============================================================================
# 颜色输出
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ============================================================================
# 主流程
# ============================================================================

echo ""
echo "============================================"
echo "  Exp9: 安装Deformable DETR依赖"
echo "============================================"
echo ""

# 检查是否已存在
if [ -d "$DETR_DIR" ]; then
    warning "Deformable-DETR 已存在: $DETR_DIR"
    echo ""
    echo "选项:"
    echo "  1) 跳过克隆，直接编译CUDA算子"
    echo "  2) 删除并重新克隆"
    echo "  3) 退出"
    echo ""
    read -p "请选择 (1-3): " choice
    
    case $choice in
        1)
            info "跳过克隆，准备编译..."
            ;;
        2)
            warning "删除现有目录: $DETR_DIR"
            rm -rf "$DETR_DIR"
            ;;
        3)
            info "退出安装"
            exit 0
            ;;
        *)
            error "无效选择，退出"
            exit 1
            ;;
    esac
fi

# 克隆仓库 (如果需要)
if [ ! -d "$DETR_DIR" ]; then
    info "克隆Deformable-DETR官方仓库..."
    echo "  源: $DETR_REPO"
    echo "  目标: $DETR_DIR"
    
    mkdir -p "$EXTERNAL_DIR"
    cd "$EXTERNAL_DIR"
    
    if git clone "$DETR_REPO"; then
        success "克隆完成"
    else
        error "克隆失败"
        exit 1
    fi
fi

# 检查ops目录
OPS_DIR="$DETR_DIR/models/ops"
if [ ! -d "$OPS_DIR" ]; then
    error "找不到CUDA算子目录: $OPS_DIR"
    error "Deformable-DETR仓库可能不完整"
    exit 1
fi

# 编译CUDA算子
echo ""
info "编译CUDA算子..."
cd "$OPS_DIR"

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    error "未找到nvcc，请确保CUDA已安装并在PATH中"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
info "CUDA版本: $CUDA_VERSION"

# 检查PyTorch
if ! python -c "import torch" 2>/dev/null; then
    error "未找到PyTorch，请先安装PyTorch"
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
info "PyTorch版本: $TORCH_VERSION"
info "PyTorch CUDA版本: $TORCH_CUDA"

# 编译
info "开始编译 (这可能需要几分钟)..."
if bash make.sh; then
    success "编译完成"
else
    error "编译失败"
    echo ""
    echo "故障排查:"
    echo "1. 检查CUDA版本是否与PyTorch匹配"
    echo "2. 检查gcc版本 (需要gcc 7+)"
    echo "3. 查看编译日志中的具体错误"
    echo ""
    exit 1
fi

# 测试
echo ""
info "测试CUDA算子..."
if python test.py; then
    success "测试通过"
else
    error "测试失败"
    exit 1
fi

# 完成
echo ""
echo "============================================"
success "Deformable DETR 安装完成!"
echo "============================================"
echo ""
echo "安装位置: $DETR_DIR"
echo ""
echo "下一步:"
echo "  1. 激活环境: conda activate samrs"
echo "  2. 设置环境变量: source scripts/setup_env.sh"
echo "  3. 验证环境: bash scripts/verify_environment.sh"
echo "  4. 运行实验: bash scripts/run_a0.sh"
echo ""
echo "详细文档: DEPENDENCIES.md"
echo ""

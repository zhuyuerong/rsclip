#!/bin/bash
# ============================================================
# 准备GitHub上传脚本
# ============================================================
# 
# 用途: 清理不必要的文件，准备上传到GitHub
# 使用: bash scripts/prepare_for_github.sh
#
# ============================================================

set -e

PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
cd $PROJECT_ROOT

echo "============================================================"
echo "准备GitHub上传"
echo "============================================================"

# 1. 清理Python缓存
echo ""
echo "1️⃣  清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
echo "   ✅ Python缓存已清理"

# 2. 清理日志文件
echo ""
echo "2️⃣  清理日志文件..."
find . -type f -name "*.log" -delete 2>/dev/null || true
echo "   ✅ 日志文件已清理"

# 3. 检查大文件
echo ""
echo "3️⃣  检查大文件 (>50MB)..."
find . -type f -size +50M 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "   ⚠️  大文件: $file ($size)"
done

# 4. 显示将要上传的目录结构
echo ""
echo "4️⃣  将要上传的目录结构:"
tree -L 3 -I '__pycache__|*.pyc|*.log|outputs|datasets|checkpoints|*.pth|*.pt' \
    src/experiments/exp9_pseudo_query/ 2>/dev/null || \
    find src/experiments/exp9_pseudo_query/ -type f \
    ! -path "*/outputs/*" \
    ! -path "*/__pycache__/*" \
    ! -name "*.pyc" \
    ! -name "*.log" | head -30

# 5. 统计文件数量和大小
echo ""
echo "5️⃣  统计信息:"
CODE_SIZE=$(du -sh src/experiments/exp9_pseudo_query/ 2>/dev/null | cut -f1)
CODE_FILES=$(find src/experiments/exp9_pseudo_query/ -type f \
    ! -path "*/outputs/*" \
    ! -path "*/__pycache__/*" \
    ! -name "*.pyc" \
    ! -name "*.log" | wc -l)
echo "   代码大小: $CODE_SIZE"
echo "   文件数量: $CODE_FILES"

# 6. 检查必需文件
echo ""
echo "6️⃣  检查必需文件:"
REQUIRED_FILES=(
    "src/experiments/exp9_pseudo_query/README.md"
    "src/experiments/exp9_pseudo_query/requirements.txt"
    "src/experiments/exp9_pseudo_query/.gitignore"
    "src/experiments/exp9_pseudo_query/CLOUD_DEPLOYMENT.md"
    "src/experiments/exp9_pseudo_query/scripts/run_a0.sh"
    "src/experiments/exp9_pseudo_query/models/heatmap_query_gen.py"
    "src/experiments/exp9_pseudo_query/datasets/dior_deformable.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (缺失)"
    fi
done

# 7. 创建上传清单
echo ""
echo "7️⃣  创建上传清单..."
cat > /tmp/github_upload_list.txt << 'EOF'
# 需要上传到GitHub的文件和目录

## 核心代码
src/experiments/exp9_pseudo_query/

## 外部依赖
external/Deformable-DETR/
  (排除: build/, *.so, *.egg-info)

## 必要的辅助代码
src/competitors/clip_methods/surgeryclip/
  (用于生成热图)

## 不上传（太大或不必要）
- datasets/          # 数据集 (需要单独下载)
- outputs/           # 实验输出
- checkpoints/*.pth  # 模型权重
- *.log             # 日志文件
EOF

cat /tmp/github_upload_list.txt
echo "   ✅ 清单已保存到: /tmp/github_upload_list.txt"

echo ""
echo "============================================================"
echo "✅ 准备完成！"
echo "============================================================"
echo ""
echo "📌 下一步:"
echo "   1. 查看 /tmp/github_upload_list.txt"
echo "   2. 运行: bash scripts/upload_to_github.sh"
echo ""

#!/bin/bash
# ============================================================
# GitHub上传脚本
# ============================================================
# 
# 用途: 将代码上传到GitHub
# 使用: bash scripts/upload_to_github.sh
#
# 注意: 
# 1. 需要先在GitHub创建仓库
# 2. 需要配置好Git用户信息
# 3. 建议使用SSH密钥认证
#
# ============================================================

set -e

PROJECT_ROOT="/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"
cd $PROJECT_ROOT

echo "============================================================"
echo "上传代码到GitHub"
echo "============================================================"

# 检查Git配置
echo ""
echo "1️⃣  检查Git配置..."
if ! git config user.name > /dev/null 2>&1; then
    echo "   ⚠️  未配置Git用户名"
    read -p "   请输入用户名: " username
    git config user.name "$username"
fi

if ! git config user.email > /dev/null 2>&1; then
    echo "   ⚠️  未配置Git邮箱"
    read -p "   请输入邮箱: " email
    git config user.email "$email"
fi

echo "   Git用户: $(git config user.name) <$(git config user.email)>"

# 初始化Git仓库
echo ""
echo "2️⃣  初始化Git仓库..."
if [ ! -d ".git" ]; then
    git init
    echo "   ✅ Git仓库已初始化"
else
    echo "   ℹ️  Git仓库已存在"
fi

# 添加远程仓库
echo ""
echo "3️⃣  配置远程仓库..."
read -p "   请输入GitHub仓库URL (例: https://github.com/username/repo.git): " repo_url

if git remote get-url origin > /dev/null 2>&1; then
    echo "   ℹ️  远程仓库已存在，更新URL..."
    git remote set-url origin "$repo_url"
else
    git remote add origin "$repo_url"
fi

echo "   ✅ 远程仓库: $repo_url"

# 添加文件
echo ""
echo "4️⃣  添加文件到Git..."

# 添加.gitignore
if [ ! -f ".gitignore" ]; then
    cp src/experiments/exp9_pseudo_query/.gitignore .gitignore
fi

# 添加exp9代码
git add src/experiments/exp9_pseudo_query/

# 添加Deformable DETR (排除编译产物)
git add external/Deformable-DETR/ || true

# 添加必要的辅助代码
git add src/competitors/clip_methods/surgeryclip/ || true

# 查看状态
echo ""
echo "   将要提交的文件:"
git status --short | head -20
echo "   ..."
echo "   总计: $(git status --short | wc -l) 个文件"

# 确认
echo ""
read -p "5️⃣  确认提交? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "   ❌ 已取消"
    exit 1
fi

# 提交
echo ""
echo "6️⃣  提交代码..."
git commit -m "Add Exp9 Pseudo Query experiment

Features:
- Q-Gen module: HeatmapQueryGenerator, TeacherQueryGenerator
- Q-Use module: QueryMixer, QueryAlignmentLoss, AttentionPriorLoss
- Training scripts: A0/A2/A3/B1/B2
- DIOR dataset loaders with heatmap support
- Complete documentation and deployment guide

Experiments:
- A0: Baseline (Deformable DETR)
- A2: Teacher proposals → pseudo query
- A3: Heatmap → pseudo query (core method)
- B1: Random query (falsification)
- B2: Shuffled heatmap (falsification)

Documentation:
- README.md: Project overview
- CLOUD_DEPLOYMENT.md: Cloud deployment guide
- EXPERIMENT_CHECKLIST.md: Complete experiment checklist
- QUICK_REFERENCE.md: Quick reference card
" || echo "   ℹ️  没有新的更改需要提交"

# 推送
echo ""
echo "7️⃣  推送到GitHub..."
read -p "   推送到哪个分支? (默认: main): " branch
branch=${branch:-main}

# 检查分支是否存在
if ! git show-ref --verify --quiet refs/heads/$branch; then
    echo "   创建新分支: $branch"
    git checkout -b $branch
fi

echo "   推送到 origin/$branch..."
git push -u origin $branch

echo ""
echo "============================================================"
echo "✅ 上传完成！"
echo "============================================================"
echo ""
echo "📌 GitHub仓库: $repo_url"
echo "📌 分支: $branch"
echo ""
echo "🔗 访问: ${repo_url%.git}"
echo ""

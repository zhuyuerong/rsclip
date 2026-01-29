#!/bin/bash
# ============================================================
# 更新GitHub仓库 - 添加exp9所有新文件
# ============================================================

set -e

cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

echo "============================================================"
echo "更新GitHub仓库 - Exp9 Pseudo Query"
echo "============================================================"

# 1. 查看当前状态
echo ""
echo "1️⃣  当前Git状态:"
git remote -v
echo ""
git branch -v

# 2. 添加exp9所有文件
echo ""
echo "2️⃣  添加exp9文件..."
git add src/experiments/exp9_pseudo_query/

# 3. 查看将要提交的文件
echo ""
echo "3️⃣  将要提交的文件:"
git status --short | grep "exp9"

# 4. 提交
echo ""
read -p "4️⃣  确认提交? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "   ❌ 已取消"
    exit 1
fi

git commit -m "Update Exp9 Pseudo Query: Complete deployment and automation

New Documentation:
- CLOUD_DEPLOYMENT.md: Cloud deployment guide
- DATA_PREPARATION.md: Data preparation guide
- DEPLOYMENT_CHECKLIST.md: Deployment checklist
- .gitignore: Git ignore configuration

New Scripts:
- scripts/prepare_for_github.sh: GitHub upload preparation
- scripts/upload_to_github.sh: GitHub upload automation
- scripts/run_sequential.sh: Sequential experiment runner
- scripts/verify_environment.sh: Environment verification
- scripts/compare_experiments.py: Experiment comparison

Updates:
- All training scripts (A0/A2/A3/B1/B2)
- Complete documentation updates
- Model modules and datasets
- Configuration files

Status:
- A0 training in progress (Epoch 10/50)
- Ready for cloud deployment
- All documentation complete
"

# 5. 推送
echo ""
echo "5️⃣  推送到GitHub..."
git push origin main

echo ""
echo "============================================================"
echo "✅ 更新完成！"
echo "============================================================"
echo ""
echo "🔗 GitHub仓库: https://github.com/zhuyuerong/rsclip"
echo ""

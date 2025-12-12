#!/bin/bash
echo "=== 🔍 GitHub仓库验证和推送工具 ==="
echo ""

# 获取GitHub用户名
GITHUB_USER=$(ssh -T git@github.com 2>&1 | grep -o "Hi [^!]*" | sed 's/Hi //')
echo "✅ GitHub用户名: $GITHUB_USER"
echo ""

# 检查仓库
echo "请确认以下信息："
echo "1. 仓库是否已在GitHub上创建？"
echo "2. 仓库的完整名称是什么？"
echo ""
echo "常见可能："
echo "  - RemoteCLIP-main"
echo "  - RemoteCLIP"
echo "  - remoteclip-main"
echo "  - 或其他名称"
echo ""

read -p "请输入仓库名称（直接回车使用 RemoteCLIP-main）: " REPO_NAME
REPO_NAME=${REPO_NAME:-RemoteCLIP-main}

echo ""
echo "检查仓库: git@github.com:$GITHUB_USER/$REPO_NAME.git"
git ls-remote git@github.com:$GITHUB_USER/$REPO_NAME.git > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 仓库存在！"
    echo ""
    echo "更新远程地址..."
    git remote set-url origin git@github.com:$GITHUB_USER/$REPO_NAME.git
    echo "✅ 远程地址已更新"
    echo ""
    echo "开始推送代码..."
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 代码推送成功！"
        echo "查看仓库: https://github.com/$GITHUB_USER/$REPO_NAME"
    else
        echo ""
        echo "❌ 推送失败，请检查错误信息"
    fi
else
    echo "❌ 仓库不存在"
    echo ""
    echo "请："
    echo "1. 访问 https://github.com/new 创建仓库"
    echo "2. 仓库名称: $REPO_NAME"
    echo "3. 不要勾选任何初始化选项"
    echo "4. 创建后再次运行此脚本"
fi

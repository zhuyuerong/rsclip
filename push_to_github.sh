#!/bin/bash
# GitHub推送脚本

echo "🔍 测试SSH连接..."
ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"

if [ $? -eq 0 ]; then
    echo "✅ SSH连接成功！"
    echo ""
    echo "📤 开始推送代码到GitHub..."
    cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 代码推送成功！"
        echo "查看仓库: https://github.com/zhuyuerong/RemoteCLIP-main"
    else
        echo ""
        echo "❌ 推送失败，请检查错误信息"
    fi
else
    echo "❌ SSH连接失败"
    echo ""
    echo "请确保："
    echo "1. 已将SSH公钥添加到GitHub: https://github.com/settings/keys"
    echo "2. 公钥内容："
    cat ~/.ssh/id_ed25519.pub
    echo ""
    echo "添加后再次运行此脚本"
fi


#!/bin/bash
echo "检查可能的仓库名称..."
for repo in "RemoteCLIP-main" "RemoteCLIP" "remoteclip-main" "remoteclip"; do
    echo -n "检查: $repo ... "
    git ls-remote git@github.com:zhuyuerong/$repo.git > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ 找到！"
        echo "更新远程地址..."
        git remote set-url origin git@github.com:zhuyuerong/$repo.git
        echo "✅ 已更新为: git@github.com:zhuyuerong/$repo.git"
        exit 0
    else
        echo "❌ 不存在"
    fi
done
echo "未找到匹配的仓库"

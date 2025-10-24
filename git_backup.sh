#!/bin/bash
# Git 快速备份脚本
# 作者: zhuyuerong
# 用途: 快速创建备份分支并提交

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}    Git 快速备份工具${NC}"
echo -e "${BLUE}================================${NC}\n"

# 获取当前日期和时间
DATE=$(date +"%Y%m%d_%H%M%S")
BRANCH_NAME="backup_${DATE}"

# 提示输入备份描述
echo -e "${YELLOW}请输入本次备份的描述 (直接回车使用默认描述):${NC}"
read -r DESCRIPTION

if [ -z "$DESCRIPTION" ]; then
    DESCRIPTION="自动备份 - ${DATE}"
fi

echo -e "\n${GREEN}开始备份流程...${NC}\n"

# 1. 显示当前状态
echo -e "${BLUE}[1/6] 检查当前状态...${NC}"
git status --short

# 2. 添加所有更改
echo -e "\n${BLUE}[2/6] 添加所有更改...${NC}"
git add .

# 3. 显示将要提交的文件
echo -e "\n${BLUE}[3/6] 将要提交的文件:${NC}"
git status --short

# 4. 提交更改到当前分支
echo -e "\n${BLUE}[4/6] 提交更改...${NC}"
git commit -m "${DESCRIPTION}" || echo "没有需要提交的更改"

# 5. 创建备份分支
echo -e "\n${BLUE}[5/6] 创建备份分支: ${BRANCH_NAME}${NC}"
git branch "${BRANCH_NAME}"

# 6. 显示所有分支
echo -e "\n${BLUE}[6/6] 当前所有分支:${NC}"
git branch -a

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}✅ 备份完成！${NC}"
echo -e "${GREEN}================================${NC}\n"

echo -e "备份信息:"
echo -e "  分支名称: ${GREEN}${BRANCH_NAME}${NC}"
echo -e "  备份描述: ${GREEN}${DESCRIPTION}${NC}"
echo -e "  提交时间: ${GREEN}$(date)${NC}"

echo -e "\n${YELLOW}下一步操作:${NC}"
echo -e "  推送到GitHub:"
echo -e "    ${BLUE}git push origin master${NC}              # 推送主分支"
echo -e "    ${BLUE}git push origin ${BRANCH_NAME}${NC}  # 推送备份分支"
echo -e "\n  或推送所有分支:"
echo -e "    ${BLUE}git push origin --all${NC}"

echo -e "\n  查看所有备份分支:"
echo -e "    ${BLUE}git branch | grep backup${NC}"

echo -e "\n  恢复到某个备份:"
echo -e "    ${BLUE}git checkout ${BRANCH_NAME}${NC}"

echo ""


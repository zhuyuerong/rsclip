# GitHub 使用指南

## 📝 Git配置信息

- **用户名**: zhuyuerong
- **邮箱**: 3074143509@qq.com
- **仓库**: RemoteCLIP-main

---

## 🚀 首次上传到GitHub

### 1. 在GitHub上创建仓库

访问 https://github.com，登录后：

1. 点击右上角 `+` → `New repository`
2. 仓库名称：`RemoteCLIP-main`
3. 描述：`RemoteCLIP with OVA-DETR for Remote Sensing Object Detection`
4. 选择 `Public` 或 `Private`
5. **不要**勾选 `Initialize this repository with a README`
6. 点击 `Create repository`

### 2. 连接到远程仓库

复制GitHub显示的仓库地址，然后执行：

```bash
cd /home/ubuntu22/Projects/RemoteCLIP-main

# 添加远程仓库（替换为你的实际地址）
git remote add origin https://github.com/zhuyuerong/RemoteCLIP-main.git

# 或使用SSH（推荐）
git remote add origin git@github.com:zhuyuerong/RemoteCLIP-main.git
```

### 3. 首次推送

```bash
# 查看当前分支
git branch

# 重命名主分支为main（可选）
git branch -M main

# 推送到GitHub
git push -u origin main
```

---

## 💾 日常备份工作流

### 方式1: 使用备份脚本（推荐）

```bash
# 进入项目目录
cd /home/ubuntu22/Projects/RemoteCLIP-main

# 运行备份脚本
./git_backup.sh
```

脚本会自动：
1. ✅ 添加所有更改
2. ✅ 创建提交
3. ✅ 创建带时间戳的备份分支（如 `backup_20251024_143022`）
4. ✅ 显示推送命令

然后执行推送：
```bash
# 推送主分支
git push origin main

# 推送备份分支
git push origin backup_20251024_143022

# 或推送所有分支
git push origin --all
```

### 方式2: 手动备份

```bash
# 1. 查看当前状态
git status

# 2. 添加所有更改
git add .

# 3. 提交更改
git commit -m "描述你的修改"

# 4. 创建备份分支（可选）
git branch backup_实验3完成_$(date +%Y%m%d)

# 5. 推送到GitHub
git push origin main
git push origin --all  # 推送所有分支
```

---

## 🌿 分支管理

### 查看所有分支

```bash
# 本地分支
git branch

# 远程分支
git branch -r

# 所有分支
git branch -a

# 查看备份分支
git branch | grep backup
```

### 创建备份分支

```bash
# 创建带描述的备份分支
git branch backup_实验3完成_20251024

# 创建带时间戳的备份分支
git branch backup_$(date +%Y%m%d_%H%M%S)
```

### 切换分支

```bash
# 切换到备份分支
git checkout backup_20251024_143022

# 切回主分支
git checkout main
```

### 删除分支

```bash
# 删除本地分支
git branch -d backup_20251024_143022

# 强制删除
git branch -D backup_20251024_143022

# 删除远程分支
git push origin --delete backup_20251024_143022
```

---

## 📋 常用命令速查

### 查看状态
```bash
git status                    # 查看工作区状态
git log --oneline            # 查看提交历史
git log --graph --all        # 查看分支图
```

### 添加和提交
```bash
git add .                    # 添加所有更改
git add file.py              # 添加指定文件
git commit -m "说明"         # 提交更改
git commit -am "说明"        # 添加并提交（已跟踪文件）
```

### 推送和拉取
```bash
git push origin main         # 推送到远程主分支
git push origin --all        # 推送所有分支
git push origin --tags       # 推送所有标签
git pull origin main         # 拉取远程更新
```

### 查看差异
```bash
git diff                     # 查看未暂存的更改
git diff --staged            # 查看已暂存的更改
git diff HEAD                # 查看所有更改
```

### 撤销操作
```bash
git checkout -- file.py      # 撤销文件的修改
git reset HEAD file.py       # 取消暂存
git reset --soft HEAD^       # 撤销最后一次提交（保留更改）
git reset --hard HEAD^       # 撤销最后一次提交（丢弃更改）
```

---

## 🏷️ 标签管理

### 创建标签

```bash
# 创建轻量标签
git tag v1.0

# 创建带注释的标签
git tag -a v1.0 -m "Experiment3 完成版本"

# 为某个提交创建标签
git tag -a v1.0 <commit-id> -m "说明"
```

### 推送标签

```bash
# 推送单个标签
git push origin v1.0

# 推送所有标签
git push origin --tags
```

### 查看和删除标签

```bash
# 查看所有标签
git tag

# 查看标签详情
git show v1.0

# 删除本地标签
git tag -d v1.0

# 删除远程标签
git push origin :refs/tags/v1.0
```

---

## 📦 推荐的备份策略

### 策略1: 按功能备份

```bash
# 完成重要功能后
git add .
git commit -m "完成: 添加OVA-DETR检测功能"
git branch backup_ova_detr_完成
git push origin main
git push origin backup_ova_detr_完成
```

### 策略2: 定期备份

```bash
# 每天工作结束前
./git_backup.sh
# 输入描述: 每日备份 - 完成训练脚本优化
git push origin --all
```

### 策略3: 版本里程碑

```bash
# 完成重要版本
git tag -a v1.0 -m "Experiment3 第一个完整版本"
git push origin main
git push origin v1.0
```

---

## 🔐 SSH密钥配置（推荐）

使用SSH可以避免每次都输入密码：

### 1. 生成SSH密钥

```bash
ssh-keygen -t ed25519 -C "3074143509@qq.com"
# 或使用RSA
ssh-keygen -t rsa -b 4096 -C "3074143509@qq.com"
```

### 2. 添加到GitHub

```bash
# 复制公钥
cat ~/.ssh/id_ed25519.pub
# 或
cat ~/.ssh/id_rsa.pub
```

然后：
1. 访问 GitHub → Settings → SSH and GPG keys
2. 点击 `New SSH key`
3. 粘贴公钥
4. 保存

### 3. 测试连接

```bash
ssh -T git@github.com
```

---

## 📊 .gitignore 说明

项目已配置 `.gitignore`，以下内容不会上传：

- ✅ Python缓存文件（`__pycache__`、`*.pyc`）
- ✅ 虚拟环境（`remoteclip/`、`venv/`）
- ✅ 训练输出（`outputs/`、`runs/`）
- ✅ 大型数据集图像（`datasets/*/images/`）
- ✅ 临时文件和缓存

保留上传：
- ✅ 所有源代码（`.py`）
- ✅ 配置文件
- ✅ 文档（`.md`）
- ✅ 数据集说明和分割文件
- ✅ 脚本文件（`.sh`）

---

## ⚠️ 注意事项

1. **不要上传大文件**
   - GitHub单文件限制：100MB
   - 仓库建议大小：< 1GB
   - 大文件使用Git LFS

2. **保护敏感信息**
   - 不要提交密码、API密钥
   - 使用环境变量或配置文件

3. **编写有意义的提交信息**
   ```bash
   # 好的提交信息
   git commit -m "添加: OVA-DETR推理引擎"
   git commit -m "修复: 数据加载器边界框转换错误"
   git commit -m "优化: 减少训练内存占用"
   
   # 不好的提交信息
   git commit -m "更新"
   git commit -m "修改代码"
   ```

4. **定期推送**
   - 每完成一个功能就提交
   - 每天至少推送一次
   - 重要修改立即备份

---

## 🎯 快速命令速记

```bash
# 日常工作流
git add .
git commit -m "说明"
git push origin main

# 快速备份
./git_backup.sh

# 查看状态
git status
git log --oneline -10

# 创建版本标签
git tag -a v1.0 -m "版本说明"
git push origin v1.0
```

---

## 📞 遇到问题？

### 常见问题

**Q: 推送失败，提示认证错误？**
```bash
# 检查远程地址
git remote -v

# 使用SSH地址
git remote set-url origin git@github.com:zhuyuerong/RemoteCLIP-main.git
```

**Q: 文件太大无法推送？**
```bash
# 检查大文件
find . -type f -size +50M

# 从历史中删除大文件
git filter-branch --tree-filter 'rm -f path/to/largefile' HEAD
```

**Q: 如何恢复到之前的版本？**
```bash
# 查看历史
git log --oneline

# 恢复到指定提交
git checkout <commit-id>

# 或创建新分支
git checkout -b recovery_branch <commit-id>
```

---

**创建时间**: 2025-10-24  
**作者**: zhuyuerong


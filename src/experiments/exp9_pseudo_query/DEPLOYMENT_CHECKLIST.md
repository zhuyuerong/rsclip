# 云端部署检查清单

完整的部署步骤检查清单，确保不遗漏任何步骤。

---

## 📋 部署前准备

### ✅ 本地准备

- [ ] 代码已清理（运行 `bash scripts/prepare_for_github.sh`）
- [ ] 已创建GitHub仓库
- [ ] Git用户信息已配置
- [ ] SSH密钥已添加到GitHub（推荐）
- [ ] DIOR数据集已打包 (`DIOR.tar.gz`)
- [ ] RemoteCLIP权重已准备
- [ ] 数据已上传到云盘
- [ ] 数据下载URL已记录

### ✅ GitHub准备

- [ ] 代码已推送到GitHub
- [ ] README.md已创建
- [ ] .gitignore已配置
- [ ] 所有文档已上传
- [ ] 仓库可公开访问（或配置好私有访问）

---

## 🖥️ 云服务器准备

### ✅ 服务器选择

- [ ] GPU: RTX 3090 / A100 (24GB+)
- [ ] 内存: 32GB+
- [ ] 存储: 100GB+ SSD
- [ ] 系统: Ubuntu 20.04/22.04
- [ ] CUDA: 11.1/11.3
- [ ] 已开通SSH访问

### ✅ 基础环境

```bash
# 检查清单
- [ ] SSH登录成功
- [ ] sudo权限正常
- [ ] 网络连接正常
- [ ] GPU驱动已安装
- [ ] CUDA已安装
- [ ] Conda/Miniconda已安装
```

验证命令:
```bash
# GPU
nvidia-smi

# CUDA
nvcc --version

# Conda
conda --version
```

---

## 📥 代码部署

### Step 1: 克隆代码

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/RemoteCLIP-Exp9.git
cd RemoteCLIP-Exp9

# 检查
- [ ] 代码克隆成功
- [ ] 目录结构正确
- [ ] 所有文件完整
```

### Step 2: 创建环境

```bash
# 创建Conda环境
conda create -n exp9 python=3.8 -y
conda activate exp9

# 检查
- [ ] 环境创建成功
- [ ] Python版本正确 (3.8)
```

### Step 3: 安装PyTorch

```bash
# CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 检查
- [ ] PyTorch安装成功
- [ ] CUDA可用
- [ ] 版本匹配
```

### Step 4: 安装依赖

```bash
cd src/experiments/exp9_pseudo_query
pip install -r requirements.txt

# 检查
- [ ] 所有依赖安装成功
- [ ] 无错误信息
```

### Step 5: 编译CUDA算子

```bash
cd ../../../external/Deformable-DETR/models/ops
bash make.sh

# 验证
python -c "from models.ops.modules import MSDeformAttn; print('OK')"

# 检查
- [ ] 编译成功
- [ ] 无错误
- [ ] 导入测试通过
```

### Step 6: 配置环境变量

```bash
# 创建配置脚本
cat > ~/setup_exp9.sh << 'EOF'
#!/bin/bash
export PROJECT_ROOT="/path/to/RemoteCLIP-Exp9"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"
cd $PROJECT_ROOT
EOF

chmod +x ~/setup_exp9.sh

# 检查
- [ ] 脚本创建成功
- [ ] 路径配置正确
```

---

## 💾 数据准备

### Step 7: 下载DIOR数据集

```bash
cd /path/to/RemoteCLIP-Exp9
mkdir -p datasets
cd datasets

# 下载
wget YOUR_CLOUD_URL/DIOR.tar.gz

# 解压
tar -xzf DIOR.tar.gz

# 验证
ls DIOR/JPEGImages/ | wc -l

# 检查
- [ ] 数据集下载成功
- [ ] 解压无错误
- [ ] 图像数量正确 (~17,591)
- [ ] 标注文件完整
- [ ] ImageSets存在
```

### Step 8: 下载RemoteCLIP权重

```bash
cd /path/to/RemoteCLIP-Exp9
mkdir -p checkpoints
cd checkpoints

# 下载
wget https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt

# 验证
ls -lh RemoteCLIP-ViT-B-32.pt

# 检查
- [ ] 权重下载成功
- [ ] 文件大小正确 (~350MB)
```

### Step 9: 热图缓存（可选）

```bash
# 方案A: 下载缓存
mkdir -p outputs/heatmap_cache
cd outputs/heatmap_cache
wget YOUR_CLOUD_URL/heatmap_cache.tar.gz
tar -xzf heatmap_cache.tar.gz

# 方案B: 在线生成
# 训练时添加 --generate_heatmap_on_fly

# 检查
- [ ] 选择了一种方案
- [ ] 如果下载，缓存完整
```

---

## ✅ 环境验证

### Step 10: 运行验证脚本

```bash
cd /path/to/RemoteCLIP-Exp9/src/experiments/exp9_pseudo_query

# 激活环境
conda activate exp9
source ~/setup_exp9.sh

# 验证
bash scripts/verify_environment.sh

# 检查清单
- [ ] Conda环境正确
- [ ] Python版本正确
- [ ] CUDA可用
- [ ] Deformable Attention编译成功
- [ ] DIOR数据集存在
- [ ] 数据集加载成功
- [ ] RemoteCLIP权重存在
- [ ] 所有模块导入成功
- [ ] GPU状态正常
```

---

## 🚀 运行实验

### Step 11: 测试运行

```bash
# 小规模测试（1个epoch）
bash scripts/run_a0.sh --epochs 1

# 检查
- [ ] 训练启动成功
- [ ] 无错误信息
- [ ] GPU利用率正常
- [ ] 内存使用正常
- [ ] Loss正常下降
```

### Step 12: 正式运行

```bash
# 使用tmux/screen
tmux new -s exp9_a0

# 运行A0
bash scripts/run_a0.sh

# 分离会话
# Ctrl+B D

# 检查
- [ ] 训练正常运行
- [ ] 日志正常输出
- [ ] GPU使用正常
```

### Step 13: 监控训练

```bash
# 查看日志
tail -f outputs/exp9_pseudo_query/a0_*/log.txt

# 查看GPU
watch -n 1 nvidia-smi

# 检查
- [ ] Loss稳定下降
- [ ] 无OOM错误
- [ ] 无NaN/Inf
- [ ] 梯度范数正常
```

---

## 📊 多机部署

### 机器1配置

```bash
# 检查清单
- [ ] 代码部署完成
- [ ] 环境配置完成
- [ ] 数据准备完成
- [ ] 验证通过
- [ ] 运行A0实验
```

### 机器2配置

```bash
# 检查清单
- [ ] 代码部署完成
- [ ] 环境配置完成
- [ ] 数据准备完成
- [ ] 验证通过
- [ ] 运行A3实验
```

### 机器3配置（可选）

```bash
# 检查清单
- [ ] 代码部署完成
- [ ] 环境配置完成
- [ ] 数据准备完成
- [ ] 验证通过
- [ ] 运行B1/B2实验
```

---

## 🔄 结果同步

### Step 14: 定期备份

```bash
# 创建备份脚本
cat > ~/backup_results.sh << 'EOF'
#!/bin/bash
cd /path/to/RemoteCLIP-Exp9
DATE=$(date +%Y%m%d_%H%M)
tar -czf backup_${DATE}.tar.gz outputs/exp9_pseudo_query/
# 上传到云盘
# rclone copy backup_${DATE}.tar.gz remote:exp9_backups/
EOF

chmod +x ~/backup_results.sh

# 设置定时任务
crontab -e
# 添加: 0 */6 * * * ~/backup_results.sh

# 检查
- [ ] 备份脚本创建
- [ ] 定时任务设置
```

### Step 15: 结果下载

```bash
# 训练完成后
cd /path/to/RemoteCLIP-Exp9

# 打包结果
tar -czf results_a0.tar.gz outputs/exp9_pseudo_query/a0_*

# 下载到本地
# 方案1: scp
scp user@cloud-server:/path/to/results_a0.tar.gz .

# 方案2: 云盘
# 上传到云盘后下载

# 检查
- [ ] 结果打包成功
- [ ] 下载到本地
- [ ] 文件完整
```

---

## 📝 文档记录

### Step 16: 记录部署信息

创建 `deployment_log.md`:

```markdown
# 部署日志

## 服务器信息
- 平台: AutoDL / 阿里云 / AWS
- GPU: RTX 3090
- 内存: 32GB
- 存储: 100GB
- IP: xxx.xxx.xxx.xxx

## 部署时间
- 开始: 2026-01-29 10:00
- 完成: 2026-01-29 12:00

## 实验状态
- A0: 运行中 (Epoch 5/50)
- A2: 待运行
- A3: 待运行

## 问题记录
- 无

## 备注
- 使用在线生成热图
- batch_size=2
```

检查:
- [ ] 部署信息已记录
- [ ] 实验状态已更新

---

## ✅ 最终检查

### 部署完成检查清单

#### 代码
- [ ] GitHub仓库已创建
- [ ] 代码已推送
- [ ] 文档完整
- [ ] .gitignore配置正确

#### 环境
- [ ] Conda环境正常
- [ ] PyTorch + CUDA正常
- [ ] Deformable Attention编译成功
- [ ] 所有依赖安装完成

#### 数据
- [ ] DIOR数据集完整
- [ ] RemoteCLIP权重存在
- [ ] 数据路径正确

#### 实验
- [ ] 验证脚本通过
- [ ] 测试运行成功
- [ ] 正式实验启动
- [ ] 监控正常

#### 备份
- [ ] 备份脚本配置
- [ ] 定时任务设置
- [ ] 结果可下载

---

## 🎯 下一步

部署完成后:

1. **监控训练**: 定期检查日志和GPU状态
2. **记录问题**: 遇到问题及时记录
3. **备份结果**: 定期备份实验输出
4. **运行后续实验**: A0完成后运行A2/A3
5. **分析结果**: 使用对比脚本分析

---

## 📞 支持

遇到问题？

1. 查看 [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md)
2. 查看 [DATA_PREPARATION.md](DATA_PREPARATION.md)
3. 查看 [EXPERIMENT_CHECKLIST.md](EXPERIMENT_CHECKLIST.md)
4. 提交Issue到GitHub

---

**部署检查清单版本**: v1.0  
**最后更新**: 2026-01-29

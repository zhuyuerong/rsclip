# Exp9 Pseudo Query 云端部署指南

本文档详细说明如何在云端服务器（如AutoDL、阿里云、AWS等）上部署和运行实验。

---

## 📋 目录

1. [准备工作](#准备工作)
2. [代码上传GitHub](#代码上传github)
3. [云端环境配置](#云端环境配置)
4. [数据集准备](#数据集准备)
5. [运行实验](#运行实验)
6. [多机并行](#多机并行)
7. [故障排查](#故障排查)

---

## 1. 准备工作

### 本地准备

#### 1.1 整理代码
```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

# 确保所有必需文件存在
ls src/experiments/exp9_pseudo_query/
```

#### 1.2 创建代码压缩包（备用方案）
```bash
# 如果GitHub上传慢，可以打包上传到云盘
tar -czf exp9_pseudo_query.tar.gz \
    src/experiments/exp9_pseudo_query/ \
    external/Deformable-DETR/ \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='outputs/' \
    --exclude='*.log'

# 查看大小
du -h exp9_pseudo_query.tar.gz
```

---

## 2. 代码上传GitHub

### 2.1 初始化Git仓库（如果还没有）

```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

# 初始化（如果是新仓库）
git init

# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/RemoteCLIP-Exp9.git

# 或使用SSH
git remote add origin git@github.com:YOUR_USERNAME/RemoteCLIP-Exp9.git
```

### 2.2 准备提交

```bash
# 检查状态
git status

# 添加exp9相关文件
git add src/experiments/exp9_pseudo_query/
git add external/Deformable-DETR/

# 添加必要的依赖文件
git add src/competitors/clip_methods/surgeryclip/

# 查看将要提交的文件
git status

# 提交
git commit -m "Add Exp9 Pseudo Query experiment code

- Q-Gen and Q-Use modules
- A0/A2/A3/B1/B2 training scripts
- DIOR dataset loaders with heatmap support
- Complete documentation and tools
"

# 推送到GitHub
git push -u origin main
```

### 2.3 创建README（GitHub首页）

在GitHub仓库根目录创建 `README.md`:

```markdown
# RemoteCLIP Exp9: Pseudo Query for Remote Sensing Object Detection

Pseudo Query方法在遥感目标检测中的应用实验。

## 🚀 快速开始

详见: [src/experiments/exp9_pseudo_query/CLOUD_DEPLOYMENT.md](src/experiments/exp9_pseudo_query/CLOUD_DEPLOYMENT.md)

## 📚 文档

- [实验清单](src/experiments/exp9_pseudo_query/EXPERIMENT_CHECKLIST.md)
- [快速参考](src/experiments/exp9_pseudo_query/QUICK_REFERENCE.md)
- [云端部署](src/experiments/exp9_pseudo_query/CLOUD_DEPLOYMENT.md)

## 📊 实验矩阵

| ID | 名称 | 状态 |
|----|------|------|
| A0 | Baseline | ✅ |
| A2 | Teacher | ⏳ |
| A3 | Heatmap | ⏳ |
| B1 | Random | ⏳ |
| B2 | Shuffled | ⏳ |
```

---

## 3. 云端环境配置

### 3.1 选择云服务器

推荐配置:
- **GPU**: NVIDIA RTX 3090 / A100 (24GB显存)
- **内存**: 32GB+
- **存储**: 100GB+ SSD
- **系统**: Ubuntu 20.04 / 22.04

推荐平台:
- **AutoDL** (国内，便宜): https://www.autodl.com/
- **阿里云PAI**: https://pai.aliyun.com/
- **AWS EC2**: https://aws.amazon.com/ec2/

### 3.2 克隆代码

```bash
# SSH登录云服务器后

# 克隆代码
git clone https://github.com/YOUR_USERNAME/RemoteCLIP-Exp9.git
cd RemoteCLIP-Exp9

# 或使用代码压缩包
# wget YOUR_CLOUD_STORAGE_URL/exp9_pseudo_query.tar.gz
# tar -xzf exp9_pseudo_query.tar.gz
```

### 3.3 创建Conda环境

```bash
# 创建环境
conda create -n exp9 python=3.8 -y
conda activate exp9

# 安装PyTorch (根据CUDA版本选择)
# CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 或 CUDA 11.1
# pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 \
#     -f https://download.pytorch.org/whl/cu111/torch_stable.html

# 安装依赖
cd src/experiments/exp9_pseudo_query
pip install -r requirements.txt
```

### 3.4 编译Deformable DETR CUDA算子

```bash
cd /path/to/RemoteCLIP-Exp9/external/Deformable-DETR/models/ops

# 编译
bash make.sh

# 验证
python -c "from models.ops.modules import MSDeformAttn; print('✅ OK')"
```

### 3.5 配置环境变量

```bash
# 创建环境配置脚本
cat > ~/setup_exp9.sh << 'EOF'
#!/bin/bash
export PROJECT_ROOT="/path/to/RemoteCLIP-Exp9"  # 修改为实际路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external/Deformable-DETR:${PYTHONPATH}"
cd $PROJECT_ROOT
EOF

chmod +x ~/setup_exp9.sh

# 使用
source ~/setup_exp9.sh
```

---

## 4. 数据集准备

### 4.1 DIOR数据集

**方案1: 从云盘下载**

```bash
# 假设你已上传到云盘
cd /path/to/RemoteCLIP-Exp9

# 下载并解压
wget YOUR_CLOUD_URL/DIOR.tar.gz
tar -xzf DIOR.tar.gz -C datasets/

# 验证
ls datasets/DIOR/
# 应该看到: JPEGImages/ Annotations/ ImageSets/
```

**方案2: 从官方下载**

```bash
# DIOR官方下载地址
# https://gcheng-nwpu.github.io/#Datasets

# 下载后解压到 datasets/DIOR/
```

**数据集结构**:
```
datasets/DIOR/
├── JPEGImages/          # 图像文件
│   ├── 00001.jpg
│   └── ...
├── Annotations/         # VOC XML标注
│   ├── 00001.xml
│   └── ...
└── ImageSets/
    └── Main/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

### 4.2 RemoteCLIP权重（A3/B2需要）

```bash
# 下载RemoteCLIP权重
mkdir -p checkpoints
cd checkpoints

# 从Hugging Face下载
wget https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt

# 或从你的云盘
# wget YOUR_CLOUD_URL/RemoteCLIP-ViT-B-32.pt
```

### 4.3 热图缓存（可选）

```bash
# 如果你已预生成热图，可以上传
# 否则使用 --generate_heatmap_on_fly 在线生成

# 上传热图缓存
mkdir -p outputs/heatmap_cache
cd outputs/heatmap_cache

# 下载
wget YOUR_CLOUD_URL/dior_trainval.tar.gz
tar -xzf dior_trainval.tar.gz
```

---

## 5. 运行实验

### 5.1 验证环境

```bash
conda activate exp9
source ~/setup_exp9.sh

cd src/experiments/exp9_pseudo_query
bash scripts/verify_environment.sh
```

### 5.2 运行A0 Baseline

```bash
# 后台运行
nohup bash scripts/run_a0.sh > logs/a0.log 2>&1 &

# 查看日志
tail -f logs/a0.log

# 或使用tmux/screen
tmux new -s exp9_a0
bash scripts/run_a0.sh
# Ctrl+B D 分离会话
```

### 5.3 运行A2/A3

```bash
# A2: Teacher proposals
nohup bash scripts/run_a2_teacher.sh > logs/a2.log 2>&1 &

# A3: Heatmap pseudo (核心)
nohup bash scripts/run_a3_heatmap.sh > logs/a3.log 2>&1 &

# B1/B2: 证伪实验
nohup bash scripts/run_b1_random.sh > logs/b1.log 2>&1 &
nohup bash scripts/run_b2_shuffled.sh > logs/b2.log 2>&1 &
```

### 5.4 监控训练

```bash
# 查看GPU使用
watch -n 1 nvidia-smi

# 查看日志
tail -f outputs/exp9_pseudo_query/a0_*/log.txt

# 查看进程
ps aux | grep train
```

---

## 6. 多机并行

### 6.1 实验分配策略

**单机方案** (1个GPU):
```
Day 1-2: A0 (50 epochs, ~14小时)
Day 3:   A2 (50 epochs, ~14小时)
Day 4:   A3 (50 epochs, ~14小时)
Day 5:   B1 + B2 (各50 epochs, 共~28小时)
```

**双机方案** (2个GPU):
```
机器1: A0 + A2 + B1
机器2: A3 + B2
```

**三机方案** (3个GPU):
```
机器1: A0 (baseline)
机器2: A2 + A3 (主要实验)
机器3: B1 + B2 (证伪实验)
```

### 6.2 配置多机

**机器1**:
```bash
# 克隆代码
git clone https://github.com/YOUR_USERNAME/RemoteCLIP-Exp9.git
cd RemoteCLIP-Exp9

# 配置环境
conda create -n exp9 python=3.8 -y
conda activate exp9
pip install -r src/experiments/exp9_pseudo_query/requirements.txt

# 准备数据
# ... (同上)

# 运行A0
cd src/experiments/exp9_pseudo_query
nohup bash scripts/run_a0.sh > logs/a0.log 2>&1 &
```

**机器2**:
```bash
# 同样步骤
# 运行A3
nohup bash scripts/run_a3_heatmap.sh > logs/a3.log 2>&1 &
```

### 6.3 结果同步

```bash
# 在每台机器上，训练完成后上传结果
cd /path/to/RemoteCLIP-Exp9

# 打包结果
tar -czf results_a0.tar.gz outputs/exp9_pseudo_query/a0_*

# 上传到云盘或GitHub Release
# 方案1: 使用rclone同步到云盘
rclone copy results_a0.tar.gz remote:exp9_results/

# 方案2: 使用scp传回本地
scp results_a0.tar.gz user@local_machine:/path/to/results/

# 方案3: 上传到GitHub Release
gh release create v1.0-a0 results_a0.tar.gz
```

---

## 7. 故障排查

### 7.1 常见问题

#### 问题1: CUDA out of memory
```bash
# 解决: 减小batch_size
# 修改 scripts/run_*.sh 中的 --batch_size 参数
--batch_size 1  # 从2降到1
```

#### 问题2: ImportError: libc10.so
```bash
# 解决: 设置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
```

#### 问题3: Deformable Attention编译失败
```bash
# 解决: 检查CUDA版本
nvcc --version

# 重新安装匹配的PyTorch
# 然后重新编译
cd external/Deformable-DETR/models/ops
python setup.py clean
bash make.sh
```

#### 问题4: 数据集路径错误
```bash
# 解决: 修改路径
# 编辑 scripts/run_*.sh
--dior_path /absolute/path/to/datasets/DIOR
```

### 7.2 调试技巧

```bash
# 1. 测试小批量
# 修改脚本，只训练1个epoch
--epochs 1

# 2. 启用详细日志
# 在Python脚本中添加
import logging
logging.basicConfig(level=logging.DEBUG)

# 3. 使用pdb调试
# 在代码中添加
import pdb; pdb.set_trace()

# 4. 检查GPU内存
nvidia-smi -l 1

# 5. 查看系统资源
htop
```

---

## 8. 成本估算

### 8.1 云服务器成本（参考）

**AutoDL** (RTX 3090):
- 价格: ~2.5元/小时
- A0训练: 14小时 × 2.5 = 35元
- 全部实验(A0+A2+A3+B1+B2): ~70小时 × 2.5 = 175元

**阿里云PAI** (V100):
- 价格: ~10元/小时
- 全部实验: ~70小时 × 10 = 700元

**AWS EC2** (p3.2xlarge, V100):
- 价格: ~$3/小时
- 全部实验: ~70小时 × $3 = $210

### 8.2 存储成本

- 代码: ~50MB (GitHub免费)
- DIOR数据集: ~3GB
- 热图缓存: ~5GB (可选)
- 实验输出: ~10GB (5个实验)
- **总计**: ~20GB

---

## 9. 最佳实践

### 9.1 实验管理

```bash
# 使用tmux管理多个实验
tmux new -s exp9

# 创建多个窗口
Ctrl+B C  # 新窗口
Ctrl+B 0  # 切换到窗口0
Ctrl+B 1  # 切换到窗口1

# 窗口0: A0
bash scripts/run_a0.sh

# 窗口1: 监控
watch -n 1 nvidia-smi

# 窗口2: 日志
tail -f outputs/exp9_pseudo_query/a0_*/log.txt
```

### 9.2 自动化脚本

创建 `scripts/run_all_experiments.sh`:

```bash
#!/bin/bash
# 自动运行所有实验

set -e

PROJECT_ROOT="/path/to/RemoteCLIP-Exp9"
cd $PROJECT_ROOT/src/experiments/exp9_pseudo_query

# A0
echo "Starting A0..."
bash scripts/run_a0.sh
wait

# A2
echo "Starting A2..."
bash scripts/run_a2_teacher.sh
wait

# A3
echo "Starting A3..."
bash scripts/run_a3_heatmap.sh
wait

# B1
echo "Starting B1..."
bash scripts/run_b1_random.sh
wait

# B2
echo "Starting B2..."
bash scripts/run_b2_shuffled.sh
wait

echo "All experiments completed!"
```

### 9.3 结果备份

```bash
# 定期备份
crontab -e

# 添加定时任务（每6小时备份一次）
0 */6 * * * cd /path/to/RemoteCLIP-Exp9 && tar -czf backup_$(date +\%Y\%m\%d_\%H\%M).tar.gz outputs/exp9_pseudo_query/
```

---

## 10. 快速部署脚本

创建 `scripts/cloud_setup.sh`:

```bash
#!/bin/bash
# 云端一键部署脚本

set -e

echo "============================================================"
echo "Exp9 Pseudo Query 云端部署"
echo "============================================================"

# 1. 克隆代码
echo "1. 克隆代码..."
git clone https://github.com/YOUR_USERNAME/RemoteCLIP-Exp9.git
cd RemoteCLIP-Exp9

# 2. 创建环境
echo "2. 创建Conda环境..."
conda create -n exp9 python=3.8 -y
conda activate exp9

# 3. 安装依赖
echo "3. 安装PyTorch..."
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

echo "4. 安装其他依赖..."
cd src/experiments/exp9_pseudo_query
pip install -r requirements.txt

# 4. 编译CUDA算子
echo "5. 编译Deformable Attention..."
cd ../../../external/Deformable-DETR/models/ops
bash make.sh

# 5. 下载数据集
echo "6. 下载数据集..."
cd ../../../../
mkdir -p datasets
cd datasets
wget YOUR_CLOUD_URL/DIOR.tar.gz
tar -xzf DIOR.tar.gz

# 6. 下载权重
echo "7. 下载RemoteCLIP权重..."
cd ../checkpoints
wget YOUR_CLOUD_URL/RemoteCLIP-ViT-B-32.pt

# 7. 验证环境
echo "8. 验证环境..."
cd ../src/experiments/exp9_pseudo_query
bash scripts/verify_environment.sh

echo "============================================================"
echo "✅ 部署完成！"
echo "============================================================"
echo ""
echo "下一步:"
echo "  conda activate exp9"
echo "  cd src/experiments/exp9_pseudo_query"
echo "  bash scripts/run_a0.sh"
```

---

## 📞 支持

遇到问题？
1. 查看 [EXPERIMENT_CHECKLIST.md](EXPERIMENT_CHECKLIST.md)
2. 查看 [README.md](README.md)
3. 提交Issue: https://github.com/YOUR_USERNAME/RemoteCLIP-Exp9/issues

---

**最后更新**: 2026-01-29

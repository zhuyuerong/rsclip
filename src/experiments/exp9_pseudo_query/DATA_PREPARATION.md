# 数据准备指南

本文档说明如何准备和上传数据集，以便在云端服务器上使用。

---

## 📋 数据清单

### 必需数据
1. **DIOR数据集** (~3GB)
2. **RemoteCLIP权重** (~350MB)

### 可选数据
3. **热图缓存** (~5GB, 可在线生成)
4. **预训练Backbone** (ResNet50, 自动下载)

---

## 1. DIOR数据集准备

### 1.1 本地打包

```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

# 检查数据集结构
ls datasets/DIOR/
# 应该看到: JPEGImages/ Annotations/ ImageSets/

# 打包数据集
tar -czf DIOR.tar.gz -C datasets DIOR/

# 查看大小
du -h DIOR.tar.gz
# 预期: ~2.5-3GB
```

### 1.2 上传到云盘

**方案1: 百度网盘**
```bash
# 使用网页上传 DIOR.tar.gz
# 获取分享链接
```

**方案2: 阿里云OSS**
```bash
# 安装ossutil
wget http://gosspublic.alicdn.com/ossutil/1.7.15/ossutil64
chmod +x ossutil64

# 配置
./ossutil64 config

# 上传
./ossutil64 cp DIOR.tar.gz oss://your-bucket/exp9/
```

**方案3: AWS S3**
```bash
# 安装awscli
pip install awscli

# 配置
aws configure

# 上传
aws s3 cp DIOR.tar.gz s3://your-bucket/exp9/
```

**方案4: 使用scp直接传到云服务器**
```bash
# 如果云服务器已开通
scp DIOR.tar.gz user@cloud-server:/path/to/datasets/
```

### 1.3 云端下载

```bash
# 在云服务器上

# 从百度网盘 (需要手动下载)
# 或使用 BaiduPCS-Go
# https://github.com/qjfoidnh/BaiduPCS-Go

# 从阿里云OSS
wget https://your-bucket.oss-cn-hangzhou.aliyuncs.com/exp9/DIOR.tar.gz

# 从AWS S3
aws s3 cp s3://your-bucket/exp9/DIOR.tar.gz .

# 解压
tar -xzf DIOR.tar.gz -C datasets/
```

---

## 2. RemoteCLIP权重准备

### 2.1 下载权重

```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/checkpoints

# 方案1: 从Hugging Face下载
wget https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt

# 方案2: 如果已有，直接使用
ls RemoteCLIP-ViT-B-32.pt
```

### 2.2 上传到云盘

```bash
# 打包
tar -czf RemoteCLIP-weights.tar.gz checkpoints/RemoteCLIP-ViT-B-32.pt

# 上传 (同DIOR数据集方法)
```

### 2.3 云端下载

```bash
# 在云服务器上
mkdir -p checkpoints
cd checkpoints

# 直接从Hugging Face下载
wget https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt

# 或从你的云盘下载
wget YOUR_CLOUD_URL/RemoteCLIP-weights.tar.gz
tar -xzf RemoteCLIP-weights.tar.gz
```

---

## 3. 热图缓存准备（可选）

### 3.1 生成热图缓存

```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

# 运行热图生成脚本
python src/experiments/exp9_pseudo_query/utils/generate_heatmap_cache.py \
    --dior_path datasets/DIOR \
    --checkpoint_path checkpoints/RemoteCLIP-ViT-B-32.pt \
    --output_dir outputs/heatmap_cache/dior_trainval \
    --split trainval

# 查看生成的热图
ls outputs/heatmap_cache/dior_trainval/
# 应该看到很多 .npy 文件
```

### 3.2 打包上传

```bash
# 打包
tar -czf heatmap_cache.tar.gz outputs/heatmap_cache/

# 查看大小
du -h heatmap_cache.tar.gz
# 预期: ~4-5GB

# 上传 (同上)
```

### 3.3 云端使用

**方案A: 使用缓存（快但占空间）**
```bash
# 下载并解压
wget YOUR_CLOUD_URL/heatmap_cache.tar.gz
tar -xzf heatmap_cache.tar.gz

# 训练时不需要 --generate_heatmap_on_fly
```

**方案B: 在线生成（慢但省空间）**
```bash
# 训练时添加参数
--generate_heatmap_on_fly

# 不需要预先下载热图缓存
```

---

## 4. 创建数据下载脚本

### 4.1 本地创建下载脚本

创建 `scripts/download_data.sh`:

```bash
#!/bin/bash
# 云端数据下载脚本

set -e

echo "============================================================"
echo "下载实验数据"
echo "============================================================"

# 配置（修改为你的实际URL）
DIOR_URL="YOUR_CLOUD_URL/DIOR.tar.gz"
WEIGHTS_URL="https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt"
HEATMAP_URL="YOUR_CLOUD_URL/heatmap_cache.tar.gz"  # 可选

PROJECT_ROOT=$(pwd)

# 1. 下载DIOR数据集
echo ""
echo "1️⃣  下载DIOR数据集..."
mkdir -p datasets
cd datasets
if [ ! -f "DIOR.tar.gz" ]; then
    wget $DIOR_URL
    tar -xzf DIOR.tar.gz
    echo "   ✅ DIOR数据集已下载并解压"
else
    echo "   ℹ️  DIOR数据集已存在"
fi
cd $PROJECT_ROOT

# 2. 下载RemoteCLIP权重
echo ""
echo "2️⃣  下载RemoteCLIP权重..."
mkdir -p checkpoints
cd checkpoints
if [ ! -f "RemoteCLIP-ViT-B-32.pt" ]; then
    wget $WEIGHTS_URL
    echo "   ✅ RemoteCLIP权重已下载"
else
    echo "   ℹ️  RemoteCLIP权重已存在"
fi
cd $PROJECT_ROOT

# 3. 下载热图缓存（可选）
echo ""
read -p "3️⃣  是否下载热图缓存? (y/n, 默认n): " download_heatmap
if [ "$download_heatmap" = "y" ]; then
    mkdir -p outputs/heatmap_cache
    cd outputs/heatmap_cache
    if [ ! -f "heatmap_cache.tar.gz" ]; then
        wget $HEATMAP_URL
        tar -xzf heatmap_cache.tar.gz
        echo "   ✅ 热图缓存已下载并解压"
    else
        echo "   ℹ️  热图缓存已存在"
    fi
    cd $PROJECT_ROOT
else
    echo "   ℹ️  跳过热图缓存下载（将在线生成）"
fi

# 4. 验证数据
echo ""
echo "4️⃣  验证数据..."
if [ -d "datasets/DIOR/JPEGImages" ]; then
    IMG_COUNT=$(ls datasets/DIOR/JPEGImages/*.jpg 2>/dev/null | wc -l)
    echo "   ✅ DIOR图像: $IMG_COUNT 张"
else
    echo "   ❌ DIOR图像目录不存在"
fi

if [ -f "checkpoints/RemoteCLIP-ViT-B-32.pt" ]; then
    WEIGHT_SIZE=$(du -h checkpoints/RemoteCLIP-ViT-B-32.pt | cut -f1)
    echo "   ✅ RemoteCLIP权重: $WEIGHT_SIZE"
else
    echo "   ❌ RemoteCLIP权重不存在"
fi

echo ""
echo "============================================================"
echo "✅ 数据下载完成！"
echo "============================================================"
```

### 4.2 上传脚本到GitHub

```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
git add src/experiments/exp9_pseudo_query/scripts/download_data.sh
git commit -m "Add data download script"
git push
```

---

## 5. 数据URL配置

### 5.1 创建配置文件

创建 `data_urls.txt`:

```txt
# 数据下载URL配置
# 使用前请修改为你的实际URL

# DIOR数据集
DIOR_URL=https://your-storage.com/DIOR.tar.gz

# RemoteCLIP权重
REMOTECLIP_URL=https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt

# 热图缓存（可选）
HEATMAP_URL=https://your-storage.com/heatmap_cache.tar.gz

# 备用下载地址
# 百度网盘: https://pan.baidu.com/s/xxxxx
# 阿里云盘: https://www.aliyundrive.com/s/xxxxx
```

### 5.2 不要上传到GitHub

```bash
# 添加到.gitignore
echo "data_urls.txt" >> .gitignore
```

---

## 6. 完整部署流程

### 6.1 在云服务器上

```bash
# 1. 克隆代码
git clone https://github.com/YOUR_USERNAME/RemoteCLIP-Exp9.git
cd RemoteCLIP-Exp9

# 2. 下载数据
bash src/experiments/exp9_pseudo_query/scripts/download_data.sh

# 3. 配置环境
conda create -n exp9 python=3.8 -y
conda activate exp9
pip install -r src/experiments/exp9_pseudo_query/requirements.txt

# 4. 编译CUDA算子
cd external/Deformable-DETR/models/ops
bash make.sh

# 5. 验证环境
cd ../../../src/experiments/exp9_pseudo_query
bash scripts/verify_environment.sh

# 6. 运行实验
bash scripts/run_a0.sh
```

---

## 7. 数据大小估算

| 数据 | 大小 | 必需 | 说明 |
|------|------|------|------|
| DIOR数据集 | ~3GB | ✅ | 必需 |
| RemoteCLIP权重 | ~350MB | ✅ | A3/B2需要 |
| 热图缓存 | ~5GB | ❌ | 可在线生成 |
| 实验输出 | ~2GB/实验 | ❌ | 训练产生 |
| **总计** | **~8-15GB** | - | 取决于是否缓存热图 |

---

## 8. 网络优化

### 8.1 使用镜像加速

```bash
# PyPI镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Conda镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```

### 8.2 使用代理

```bash
# 设置代理
export http_proxy=http://proxy-server:port
export https_proxy=http://proxy-server:port

# 下载完成后取消
unset http_proxy
unset https_proxy
```

### 8.3 断点续传

```bash
# 使用wget的断点续传
wget -c $URL

# 使用aria2（更快）
aria2c -x 16 -s 16 $URL
```

---

## 9. 故障排查

### 问题1: 下载速度慢
```bash
# 解决: 使用多线程下载
aria2c -x 16 -s 16 YOUR_URL

# 或使用国内云盘
```

### 问题2: 解压失败
```bash
# 检查文件完整性
md5sum DIOR.tar.gz

# 重新下载
rm DIOR.tar.gz
wget -c YOUR_URL
```

### 问题3: 空间不足
```bash
# 检查空间
df -h

# 清理不必要的文件
conda clean --all
pip cache purge

# 不下载热图缓存，使用在线生成
```

---

## 10. 最佳实践

1. **使用版本控制**: 记录数据集版本和MD5
2. **分块上传**: 大文件分块上传，避免超时
3. **多个备份**: 上传到多个云盘，防止失效
4. **文档化**: 记录所有URL和访问方式
5. **自动化**: 使用脚本自动下载和验证

---

**最后更新**: 2026-01-29

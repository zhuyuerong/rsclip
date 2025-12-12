# BERT 模型下载解决方案

## 当前情况
- ✅ Tokenizer 已下载（268KB）
- ❌ BERT 模型权重缺失（需要约 440MB）
- ❌ SSH 连接被拒绝（可能是同一台机器或 SSH 未配置）

## 解决方案

### 方案 1：直接在当前机器下载（如果网络可用）

如果你之前能在 base 环境下载 tokenizer，可以尝试下载完整模型：

```bash
# 在 base 环境或 remoteclip 环境
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
source remoteclip/bin/activate  # 或使用 base 环境

# 尝试下载完整 BERT 模型
python -c "from transformers import BertModel; print('Downloading...'); model = BertModel.from_pretrained('bert-base-uncased'); print('✅ Done!')"
```

**如果网络不稳定，可以设置重试：**
```bash
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1
python -c "
from transformers import BertModel
import os
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
print('Downloading BERT model (this may take a while)...')
model = BertModel.from_pretrained('bert-base-uncased')
print('✅ BERT model downloaded successfully!')
"
```

### 方案 2：使用 U盘/移动硬盘传输

如果你有另一台有网的机器：

1. **在有网的机器上下载：**
   ```bash
   python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
   ```

2. **打包文件：**
   ```bash
   cd ~/.cache/huggingface/hub/
   tar -czf bert-base-uncased-cache.tar.gz models--bert-base-uncased/
   ```

3. **复制到 U盘，然后在当前机器解压：**
   ```bash
   # 假设 U盘挂载在 /media/ubuntu22/...
   # 先找到 U盘路径
   ls /media/ubuntu22/
   
   # 解压到缓存目录
   mkdir -p ~/.cache/huggingface/hub/
   tar -xzf /media/ubuntu22/<U盘路径>/bert-base-uncased-cache.tar.gz -C ~/.cache/huggingface/hub/
   ```

### 方案 3：配置 SSH（如果有两台机器）

如果确实有两台机器，需要先配置 SSH：

**在服务器上（192.168.233.198）启动 SSH 服务：**
```bash
# 安装 SSH 服务（如果未安装）
sudo apt update
sudo apt install openssh-server -y

# 启动 SSH 服务
sudo systemctl start ssh
sudo systemctl enable ssh

# 检查状态
sudo systemctl status ssh
```

**然后从有网的机器执行 scp：**
```bash
scp -r ~/.cache/huggingface/hub/models--bert-base-uncased \
    ubuntu22@192.168.233.198:~/.cache/huggingface/hub/
```

### 方案 4：使用 Python 脚本手动下载（支持断点续传）

创建一个下载脚本：

```python
# download_bert.py
import os
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

from transformers import BertModel
import time

max_retries = 5
retry_delay = 10

for i in range(max_retries):
    try:
        print(f"Attempt {i+1}/{max_retries}: Downloading BERT model...")
        model = BertModel.from_pretrained('bert-base-uncased')
        print("✅ BERT model downloaded successfully!")
        break
    except Exception as e:
        if i < max_retries - 1:
            print(f"❌ Error: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print(f"❌ Failed after {max_retries} attempts: {e}")
            raise
```

运行：
```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
source remoteclip/bin/activate
python download_bert.py
```

### 方案 5：使用 HuggingFace 镜像（如果可用）

如果可以使用镜像源：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
```

## 推荐方案

**如果只有一台机器：**
- 优先尝试方案 1（直接下载）
- 如果网络不稳定，使用方案 4（带重试的脚本）

**如果有两台机器但 SSH 未配置：**
- 使用方案 2（U盘传输）最简单
- 或配置 SSH 后使用方案 3

## 验证下载

下载完成后验证：

```bash
# 检查文件大小（应该约 440MB+）
du -sh ~/.cache/huggingface/hub/models--bert-base-uncased/

# 检查模型权重文件
find ~/.cache/huggingface/hub/models--bert-base-uncased -name "pytorch_model.bin" -o -name "model.safetensors"

# 运行测试
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
source remoteclip/bin/activate
python src/experiments/exp2/run_gdino_sanity.py
```



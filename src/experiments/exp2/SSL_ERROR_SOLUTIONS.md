# SSL 错误解决方案

## 问题
```
SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol
```

## 解决方案

### 方案 1：使用 HuggingFace 镜像源（推荐）

```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 然后下载
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
```

### 方案 2：使用带重试的脚本

```bash
python /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/src/experiments/exp2/download_bert_with_retry.py
```

### 方案 3：禁用 SSL 验证（不推荐，仅用于测试）

```bash
export PYTHONHTTPSVERIFY=0
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
unset PYTHONHTTPSVERIFY  # 下载后恢复
```

### 方案 4：手动下载文件

如果网络问题持续，可以手动下载：

1. **访问 HuggingFace 网站**（如果有浏览器访问）：
   - https://huggingface.co/bert-base-uncased/tree/main
   - 下载以下文件：
     - `config.json` (已有)
     - `pytorch_model.bin` (约 440MB) 或 `model.safetensors`
     - `tokenizer.json` (已有)
     - `vocab.txt` (已有)

2. **放置到缓存目录**：
   ```bash
   # 找到快照目录
   SNAPSHOT_DIR=$(ls -d ~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/*/ | head -1)
   
   # 复制文件到快照目录
   cp pytorch_model.bin "$SNAPSHOT_DIR/"
   ```

### 方案 5：使用 wget/curl 下载（绕过 Python SSL）

```bash
# 获取下载链接（需要从 HuggingFace 网站获取）
# 然后使用 wget 下载
cd ~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/*/
wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
```

## 推荐操作顺序

1. **先尝试镜像源**（最简单）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
   ```

2. **如果镜像源不行，使用重试脚本**：
   ```bash
   python /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main/src/experiments/exp2/download_bert_with_retry.py
   ```

3. **如果都不行，考虑手动下载或使用 U盘传输**



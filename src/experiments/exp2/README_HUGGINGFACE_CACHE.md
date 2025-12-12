# HuggingFace 缓存设置说明

## 问题
GroundingDINO 需要下载 `bert-base-uncased` tokenizer，但服务器可能没有网络连接。

## 解决方案：从有网络的机器复制缓存

### 步骤 1：在有网络的机器上下载 tokenizer

在有网络的机器上（本地电脑）执行：

```bash
python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased')"
```

这会将文件下载到：
- **Linux/Mac**: `~/.cache/huggingface/hub/`
- **Windows**: `C:\Users\<username>\.cache\huggingface\hub\`

下载后会生成类似这样的目录结构：
```
~/.cache/huggingface/hub/
  models--bert-base-uncased/
    snapshots/
      <hash>/
        vocab.txt
        tokenizer_config.json
        tokenizer.json
        config.json
        ...
```

### 步骤 2：复制到服务器

将整个 `models--bert-base-uncased` 文件夹复制到服务器的：

```bash
~/.cache/huggingface/hub/
```

**方法 1：使用 scp（从本地到服务器）**
```bash
scp -r ~/.cache/huggingface/hub/models--bert-base-uncased user@server:~/.cache/huggingface/hub/
```

**方法 2：使用 rsync**
```bash
rsync -avz ~/.cache/huggingface/hub/models--bert-base-uncased user@server:~/.cache/huggingface/hub/
```

**方法 3：手动复制**
1. 将 `models--bert-base-uncased` 文件夹打包：
   ```bash
   cd ~/.cache/huggingface/hub/
   tar -czf bert-base-uncased-cache.tar.gz models--bert-base-uncased/
   ```
2. 传输到服务器
3. 在服务器上解压：
   ```bash
   mkdir -p ~/.cache/huggingface/hub/
   tar -xzf bert-base-uncased-cache.tar.gz -C ~/.cache/huggingface/hub/
   ```

### 步骤 3：验证

在服务器上检查：
```bash
ls -la ~/.cache/huggingface/hub/models--bert-base-uncased/
```

应该能看到 `snapshots/` 目录。

### 步骤 4：重新运行测试

```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
source remoteclip/bin/activate
python src/experiments/exp2/run_gdino_sanity.py
```

## 当前状态

✅ Checkpoint 文件已找到：
- `groundingdino_swint_ogc.pth`
- `groundingdino_swinb_cogcoor.pth`

✅ 配置文件已找到：
- `GroundingDINO_SwinT_OGC.py`
- `GroundingDINO_SwinB_cfg.py`

❌ HuggingFace 缓存缺失：
- 需要 `~/.cache/huggingface/hub/models--bert-base-uncased/`

## 测试脚本功能

`run_gdino_sanity.py` 现在可以：
1. 自动检测可用的 checkpoint 文件
2. 自动匹配对应的配置文件
3. 检查 HuggingFace 缓存状态
4. 加载模型并进行前向推理（如果有缓存）



# 设置 BERT 模型缓存完整指南

## 问题分析

GroundingDINO 需要：
1. ✅ **Tokenizer**（已下载）：`vocab.txt`, `tokenizer_config.json`, `tokenizer.json`
2. ❌ **BERT 模型权重**（缺失）：`pytorch_model.bin` 或 `model.safetensors`

当前错误：
```
bert-base-uncased does not appear to have a file named pytorch_model.bin, model.safetensors
```

## 解决方案

### 情况 1：你有两台机器（有网的机器 + 没网的服务器）

#### 步骤 1：在有网的机器上下载完整的 BERT 模型

```bash
# 在有网的机器上执行
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
```

这会下载：
- Tokenizer 文件（如果还没有）
- **BERT 模型权重文件**（`pytorch_model.bin` 或 `model.safetensors`）

#### 步骤 2：确认下载的文件

```bash
# 检查缓存目录
ls -lh ~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/*/

# 应该能看到：
# - vocab.txt
# - tokenizer_config.json
# - config.json
# - pytorch_model.bin (或 model.safetensors)  ← 这个很重要！
```

#### 步骤 3：使用 scp 复制到服务器

**正确的 scp 命令格式：**

```bash
# 在有网的机器上执行
scp -r ~/.cache/huggingface/hub/models--bert-base-uncased \
    <服务器用户名>@<服务器IP或主机名>:~/.cache/huggingface/hub/
```

**示例：**
```bash
# 假设：
# - 服务器用户名：ubuntu22
# - 服务器 IP：192.168.1.123

scp -r ~/.cache/huggingface/hub/models--bert-base-uncased \
    ubuntu22@192.168.1.123:~/.cache/huggingface/hub/
```

**如果服务器用户名和当前机器相同，且在同一网络：**
```bash
# 如果服务器主机名是 server-ubuntu22
scp -r ~/.cache/huggingface/hub/models--bert-base-uncased \
    ubuntu22@server-ubuntu22:~/.cache/huggingface/hub/

# 或者使用 IP
scp -r ~/.cache/huggingface/hub/models--bert-base-uncased \
    ubuntu22@192.168.1.123:~/.cache/huggingface/hub/
```

#### 步骤 4：在服务器上验证

```bash
# SSH 登录到服务器
ssh ubuntu22@<服务器IP或主机名>

# 检查文件是否存在
ls -lh ~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/*/

# 应该能看到 pytorch_model.bin 或 model.safetensors
```

### 情况 2：只有一台机器（当前机器就是服务器）

如果当前机器 `ubuntu22` 就是运行 GroundingDINO 的服务器，且**有网络连接**：

```bash
# 直接下载完整的 BERT 模型
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
```

如果**没有网络连接**，你需要：
1. 在有网的机器上下载完整的 BERT 模型
2. 将整个 `models--bert-base-uncased` 文件夹复制到当前机器

### 情况 3：使用其他传输方式

如果 scp 不可用，可以使用：

**方法 A：使用 rsync**
```bash
rsync -avz ~/.cache/huggingface/hub/models--bert-base-uncased \
    ubuntu22@<服务器IP>:~/.cache/huggingface/hub/
```

**方法 B：打包传输**
```bash
# 在有网的机器上打包
cd ~/.cache/huggingface/hub/
tar -czf bert-base-uncased-cache.tar.gz models--bert-base-uncased/

# 传输 tar 文件（使用 scp、ftp、U盘等）
scp bert-base-uncased-cache.tar.gz ubuntu22@<服务器IP>:~/

# 在服务器上解压
ssh ubuntu22@<服务器IP>
mkdir -p ~/.cache/huggingface/hub/
tar -xzf ~/bert-base-uncased-cache.tar.gz -C ~/.cache/huggingface/hub/
```

## 验证安装

复制完成后，运行测试脚本：

```bash
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
source remoteclip/bin/activate
python src/experiments/exp2/run_gdino_sanity.py
```

如果看到：
- ✅ "load tokenizer done."
- ✅ "Model loaded successfully."
- ✅ 输出包含 "pred_logits" 和 "pred_boxes"

说明安装成功！

## 当前状态检查

检查当前机器的缓存：

```bash
# 检查 tokenizer 文件（应该存在）
ls ~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/*/vocab.txt

# 检查模型权重文件（可能缺失）
ls ~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/*/pytorch_model.bin
ls ~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/*/model.safetensors
```

## 常见问题

**Q: 如何知道服务器的 IP 或主机名？**

A: 在服务器上执行：
```bash
hostname -I  # 显示 IP 地址
hostname     # 显示主机名
```

**Q: scp 需要密码怎么办？**

A: 可以使用 `-o` 选项指定 SSH 配置，或者设置 SSH 密钥免密登录。

**Q: 文件很大，传输很慢怎么办？**

A: BERT 模型大约 440MB，可以使用：
- `rsync` 支持断点续传
- 压缩传输：`tar -czf` 打包后传输
- 使用更快的网络连接



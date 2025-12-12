#!/bin/bash
echo "尝试使用 HuggingFace 镜像源下载 BERT 模型..."
echo ""

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1

python -c "
from transformers import BertModel
print('Using mirror:', 'https://hf-mirror.com')
print('Downloading bert-base-uncased model...')
print('This may take several minutes (model size ~440MB)...')
print('')
model = BertModel.from_pretrained('bert-base-uncased')
print('')
print('✅ BERT model downloaded successfully!')
print(f'Model config: {model.config.model_type}')
"

echo ""
echo "检查下载的文件..."
du -sh ~/.cache/huggingface/hub/models--bert-base-uncased/ 2>/dev/null
find ~/.cache/huggingface/hub/models--bert-base-uncased -name "pytorch_model.bin" -o -name "model.safetensors" 2>/dev/null | head -3

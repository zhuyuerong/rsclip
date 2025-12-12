#!/bin/bash
# 在 base 环境下载完整的 BERT 模型

echo "=========================================="
echo "Downloading BERT model in base environment"
echo "=========================================="
echo ""

# 确保在 base 环境（不激活任何虚拟环境）
python -c "
from transformers import BertModel
import os

print('Downloading bert-base-uncased model...')
print('This may take several minutes (model size ~440MB)...')
print('')

try:
    model = BertModel.from_pretrained('bert-base-uncased')
    print('')
    print('✅ BERT model downloaded successfully!')
    print(f'Model type: {type(model)}')
    print(f'Model config: {model.config.model_type}')
    print(f'Hidden size: {model.config.hidden_size}')
except Exception as e:
    print(f'❌ Error: {e}')
    raise
"

echo ""
echo "=========================================="
echo "Checking downloaded files..."
echo "=========================================="
du -sh ~/.cache/huggingface/hub/models--bert-base-uncased/ 2>/dev/null || echo "Cache directory not found"

echo ""
find ~/.cache/huggingface/hub/models--bert-base-uncased -name "pytorch_model.bin" -o -name "model.safetensors" 2>/dev/null | head -3

echo ""
echo "Done!"



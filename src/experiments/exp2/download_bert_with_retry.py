#!/usr/bin/env python
"""
带重试和错误处理的 BERT 模型下载脚本
"""
import os
import time
import sys

# 禁用实验性警告
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

# 尝试禁用 SSL 验证（仅用于测试，不推荐生产环境）
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['REQUESTS_CA_BUNDLE'] = ''

from transformers import BertModel

def download_with_retry(max_retries=10, retry_delay=5):
    """带重试的下载函数"""
    for attempt in range(max_retries):
        try:
            print(f"\n{'='*60}")
            print(f"Attempt {attempt + 1}/{max_retries}")
            print(f"{'='*60}")
            print("Downloading bert-base-uncased model...")
            print("This may take several minutes (model size ~440MB)...")
            print("")
            
            # 尝试使用本地文件优先模式
            local_files_only = False
            if attempt > 3:
                # 如果多次失败，尝试使用本地缓存
                print("Trying to use local cache if available...")
                local_files_only = False  # 仍然允许下载
            
            model = BertModel.from_pretrained(
                'bert-base-uncased',
                local_files_only=local_files_only
            )
            
            print("\n✅ BERT model downloaded successfully!")
            print(f"Model type: {type(model)}")
            print(f"Model config: {model.config.model_type}")
            print(f"Hidden size: {model.config.hidden_size}")
            return model
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error on attempt {attempt + 1}:")
            print(f"   {error_msg[:200]}...")  # 只显示前200个字符
            
            if attempt < max_retries - 1:
                print(f"\n⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                # 逐渐增加等待时间
                retry_delay = min(retry_delay * 1.5, 30)
            else:
                print(f"\n❌ Failed after {max_retries} attempts")
                print("\n建议:")
                print("1. 检查网络连接")
                print("2. 尝试使用镜像源: export HF_ENDPOINT=https://hf-mirror.com")
                print("3. 或者手动下载文件并放置到缓存目录")
                raise

if __name__ == "__main__":
    try:
        model = download_with_retry(max_retries=10, retry_delay=5)
        print("\n" + "="*60)
        print("✅ Download completed successfully!")
        print("="*60)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Download failed: {e}")
        sys.exit(1)



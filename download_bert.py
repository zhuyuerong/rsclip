import os
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

from transformers import BertModel
import time

max_retries = 5
retry_delay = 10

for i in range(max_retries):
    try:
        print(f"Attempt {i+1}/{max_retries}: Downloading BERT model...")
        print("This may take several minutes (model size ~440MB)...")
        model = BertModel.from_pretrained('bert-base-uncased')
        print("✅ BERT model downloaded successfully!")
        print(f"Model config: {model.config}")
        break
    except Exception as e:
        if i < max_retries - 1:
            print(f"❌ Error: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print(f"❌ Failed after {max_retries} attempts: {e}")
            raise

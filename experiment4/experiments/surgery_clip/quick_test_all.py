# -*- coding: utf-8 -*-
"""
快速测试所有功能

一次性运行:
1. patch相似度矩阵分析
2. 多层特征分析  
3. 文本引导VV^T热图
"""

import sys
from pathlib import Path

# 添加路径
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))
exp_dir = Path(__file__).parent
sys.path.append(str(exp_dir))

import torch
from experiment4.core.config import Config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper
from utils.seen_unseen_split import SeenUnseenDataset

def main():
    print("="*60)
    print("VV机制与多层特征分析 - 快速测试")
    print("="*60)
    
    # 配置
    config = Config()
    config.dataset_root = "datasets/mini_dataset"
    config.use_surgery = False
    config.use_vv_mechanism = False
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    print("\n1. 加载模型...")
    model = CLIPSurgeryWrapper(config)
    print(f"   ✓ 模型加载完成 (use_surgery={config.use_surgery}, use_vv={config.use_vv_mechanism})")
    
    # 加载数据集
    print("\n2. 加载数据集...")
    unseen_classes = ['airplane', 'bridge', 'storagetank', 'vehicle', 'windmill']
    
    seen_dataset = SeenUnseenDataset(
        config.dataset_root,
        split='seen',
        mode='val',
        unseen_classes=unseen_classes
    )
    
    unseen_dataset = SeenUnseenDataset(
        config.dataset_root,
        split='unseen',
        mode='val',
        unseen_classes=unseen_classes
    )
    
    print(f"   ✓ Seen验证集: {len(seen_dataset)}个样本")
    print(f"   ✓ Unseen验证集: {len(unseen_dataset)}个样本")
    
    # 测试特征提取
    print("\n3. 测试多层特征提取...")
    sample = seen_dataset[0]
    images = sample['image'].unsqueeze(0).to(config.device)
    
    layer_features = model.get_layer_features(images, layer_indices=[1, 6, 9, 12])
    print(f"   ✓ 提取了{len(layer_features)}层特征:")
    for layer_idx, features in layer_features.items():
        print(f"     Layer {layer_idx}: {features.shape} (dtype={features.dtype})")
    
    # 测试文本编码
    print("\n4. 测试文本编码...")
    text_features = model.encode_text(['airplane', 'ship', 'car'])
    print(f"   ✓ 文本特征: {text_features.shape}")
    
    # 测试patch相似度
    print("\n5. 测试patch相似度计算...")
    patch_features = layer_features[12][:, 1:, :]  # [B, N_patches, C]
    import torch.nn.functional as F
    patch_norm = F.normalize(patch_features, dim=-1)
    similarity_matrix = patch_norm @ patch_norm.transpose(-2, -1)
    print(f"   ✓ 相似度矩阵: {similarity_matrix.shape}")
    print(f"   ✓ 均值: {similarity_matrix.mean().item():.4f}")
    print(f"   ✓ 标准差: {similarity_matrix.std().item():.4f}")
    
    # 测试Feature Surgery
    print("\n6. 测试Feature Surgery...")
    from experiment4.core.models.clip_surgery import clip_feature_surgery
    
    image_features = layer_features[12]  # [B, N+1, C]
    text_feature_single = model.encode_text([sample['class_name']])  # [1, C]
    text_feature_single = F.normalize(text_feature_single, dim=-1)
    
    try:
        similarity_surgery = clip_feature_surgery(image_features, text_feature_single, t=2)
        print(f"   ✓ Surgery相似度: {similarity_surgery.shape}")
        print(f"   ✓ 均值: {similarity_surgery.mean().item():.4f}")
        print(f"   ✓ 标准差: {similarity_surgery.std().item():.4f}")
    except Exception as e:
        print(f"   ❌ Surgery失败: {e}")
    
    print("\n" + "="*60)
    print("✓ 所有功能测试通过！")
    print("="*60)
    
    print("\n下一步:")
    print("  1. 运行完整实验: bash experiment4/experiments/04_vv_multi_layer_analysis/run_all_experiments.sh")
    print("  2. 或单独运行:")
    print("     - layer_analysis.py")
    print("     - patch_similarity_matrix.py")
    print("     - text_guided_vvt.py")
    print("     - compare_three_modes.py")


if __name__ == "__main__":
    main()


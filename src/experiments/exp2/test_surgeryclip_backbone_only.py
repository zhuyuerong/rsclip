# src/experiments/exp2/test_surgeryclip_backbone_only.py
"""
测试 SurgeryCLIPBackbone 是否能独立运行
"""
import os
import sys
import torch

# 添加路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

gdino_root = os.path.join(project_root, "src", "experiments", "exp2", "Open-GroundingDino-main")
if gdino_root not in sys.path:
    sys.path.insert(0, gdino_root)

from groundingdino.util.misc import NestedTensor
from surgeryclip_backbone import SurgeryCLIPBackbone

def main():
    # ====== 配置 ======
    checkpoint_path = input("请输入 SurgeryCLIP checkpoint 路径: ").strip()
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint 文件不存在: {checkpoint_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ====== 创建 backbone ======
    print("=> Creating SurgeryCLIPBackbone...")
    try:
        backbone = SurgeryCLIPBackbone(
            checkpoint_path=checkpoint_path,
            device=device,
            train_backbone=False,
        ).to(device).eval()
        print("✅ Backbone created successfully!")
        print(f"   Embed dim: {backbone.num_channels}")
    except Exception as e:
        print(f"❌ Error creating backbone: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ====== 测试 forward ======
    print("\n=> Testing forward pass...")
    try:
        # 创建测试输入（CLIP 通常使用 224x224）
        B = 1
        H, W = 224, 224
        x = torch.randn(B, 3, H, W).to(device)
        mask = torch.zeros(B, H, W, dtype=torch.bool).to(device)
        samples = NestedTensor(x, mask)
        
        print(f"Input shape: {x.shape}")
        print(f"Input mask shape: {mask.shape}")
        
        with torch.no_grad():
            out = backbone(samples)
        
        print("\n✅ Forward pass successful!")
        print("Output:")
        for k, v in out.items():
            print(f"  Level '{k}':")
            print(f"    Features: {v.tensors.shape}")
            print(f"    Mask: {v.mask.shape if v.mask is not None else None}")
        
        # 测试位置编码
        print("\n=> Testing position embedding...")
        pos = backbone.get_position_embedding(samples)
        print(f"Position embedding shape: {pos.shape}")
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    main()



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试改进检测器的各个组件
"""

import torch
import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent.parent))

def test_image_encoder():
    """测试原图编码器"""
    print("=" * 60)
    print("测试原图编码器")
    print("=" * 60)
    
    from models.image_encoder import SimpleImageEncoder
    
    encoder = SimpleImageEncoder(output_dim=128, output_size=7)
    print(f"参数量: {encoder.get_param_count():,}")
    
    # 测试forward
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    assert out.shape == (2, 128, 7, 7), f"输出shape错误: {out.shape}"
    print("✅ 原图编码器测试通过\n")


def test_cam_fusion():
    """测试CAM融合"""
    print("=" * 60)
    print("测试多层CAM融合")
    print("=" * 60)
    
    from models.multi_layer_cam_fusion import MultiLayerCAMFusion
    
    fusion = MultiLayerCAMFusion(num_layers=3)
    print(f"参数量: {fusion.layer_weights.numel()}")
    
    # 测试forward
    multi_cams = [
        torch.randn(2, 20, 7, 7),
        torch.randn(2, 20, 7, 7),
        torch.randn(2, 20, 7, 7)
    ]
    fused = fusion(multi_cams)
    print(f"输入: 3个 [{multi_cams[0].shape}]")
    print(f"输出: {fused.shape}")
    assert fused.shape == (2, 20, 7, 7), f"输出shape错误: {fused.shape}"
    
    # 测试权重
    weights = fusion.get_layer_weights()
    print(f"层权重: {weights}")
    assert len(weights) == 3, f"权重数量错误: {len(weights)}"
    print("✅ CAM融合测试通过\n")


def test_detection_head():
    """测试多输入检测头"""
    print("=" * 60)
    print("测试多输入检测头")
    print("=" * 60)
    
    from models.multi_input_detection_head import MultiInputDetectionHead
    
    head = MultiInputDetectionHead(
        num_classes=20,
        img_feat_dim=128,
        cam_dim=20,
        layer_feat_dim=768,
        num_layers=3,
        hidden_dim=256,
        cam_resolution=7
    )
    print(f"参数量: {head.get_param_count():,}")
    
    # 测试forward
    img_features = torch.randn(2, 128, 7, 7)
    fused_cam = torch.randn(2, 20, 7, 7)
    multi_features = [
        torch.randn(2, 49, 768),  # [B, N², D]
        torch.randn(2, 49, 768),
        torch.randn(2, 49, 768)
    ]
    
    outputs = head(img_features, fused_cam, multi_features)
    print(f"输入:")
    print(f"  原图特征: {img_features.shape}")
    print(f"  融合CAM: {fused_cam.shape}")
    print(f"  多层特征: {[f.shape for f in multi_features]}")
    print(f"输出:")
    print(f"  框坐标: {outputs['boxes'].shape}")
    print(f"  置信度: {outputs['confidences'].shape}")
    
    assert outputs['boxes'].shape == (2, 20, 7, 7, 4), f"框shape错误: {outputs['boxes'].shape}"
    assert outputs['confidences'].shape == (2, 20, 7, 7), f"置信度shape错误: {outputs['confidences'].shape}"
    print("✅ 检测头测试通过\n")


def test_full_model():
    """测试完整模型（需要SurgeryCLIP checkpoint）"""
    print("=" * 60)
    print("测试完整模型")
    print("=" * 60)
    
    checkpoint_path = "checkpoints/RemoteCLIP-ViT-B-32.pt"
    if not Path(checkpoint_path).exists():
        project_root = Path(__file__).parent.parent.parent.parent
        checkpoint_path = project_root / checkpoint_path
        if not checkpoint_path.exists():
            print("⚠️  Skipping: SurgeryCLIP checkpoint not found")
            return
    
    from models.improved_direct_detection_detector import create_improved_direct_detection_detector
    
    print("创建模型...")
    model = create_improved_direct_detection_detector(
        surgery_clip_checkpoint=str(checkpoint_path),
        num_classes=20,
        cam_resolution=7,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        unfreeze_cam_last_layer=True
    )
    
    # 测试forward
    images = torch.randn(2, 3, 224, 224).to(model.device)
    # 使用正确的20个类别
    text_queries = [
        'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'dam', 'expressway-service-area', 'expressway-toll-station', 'golffield',
        'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
        'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
    ]
    
    print("Forward pass...")
    outputs = model(images, text_queries)
    
    print(f"输出keys: {list(outputs.keys())}")
    print(f"  预测框: {outputs['pred_boxes'].shape}")
    print(f"  置信度: {outputs['confidences'].shape}")
    print(f"  融合CAM: {outputs['fused_cam'].shape}")
    print(f"  层权重: {outputs['layer_weights']}")
    
    assert outputs['pred_boxes'].shape == (2, 20, 7, 7, 4)
    assert outputs['confidences'].shape == (2, 20, 7, 7)
    assert len(outputs['multi_layer_cams']) == 3
    print("✅ 完整模型测试通过\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("改进检测器组件测试")
    print("=" * 60 + "\n")
    
    try:
        test_image_encoder()
        test_cam_fusion()
        test_detection_head()
        test_full_model()
        
        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


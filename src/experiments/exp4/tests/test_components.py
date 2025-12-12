# -*- coding: utf-8 -*-
"""
单元测试: 验证每个组件独立工作
运行: python tests/test_components.py
"""

import torch
import sys
import os
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent.parent))


def test_box_head():
    """测试BoxHead能否正常前向传播"""
    print("=" * 50)
    print("Test 1: BoxHead")
    print("=" * 50)
    
    from models.box_head import BoxHead
    
    # 创建
    box_head = BoxHead(
        num_classes=20,
        hidden_dim=256,
        cam_resolution=7
    )
    
    # 假输入
    B, C, H, W = 2, 20, 7, 7
    fake_cam = torch.randn(B, C, H, W)
    
    # 前向
    box_outputs = box_head(fake_cam)
    
    # 验证
    assert 'box_params' in box_outputs, "box_params should be in outputs"
    box_params = box_outputs['box_params']
    assert box_params.shape == (B, C, H, W, 4), f"Expected shape (2,20,7,7,4), got {box_params.shape}"
    
    # 解码测试
    pred_boxes = box_head.decode_boxes(box_params, cam_resolution=H)
    assert pred_boxes.shape == (B, C, H, W, 4)
    assert (pred_boxes >= 0).all() and (pred_boxes <= 1).all(), "Boxes should be in [0,1]"
    
    print("✅ BoxHead works!")
    print(f"   Input CAM: {fake_cam.shape}")
    print(f"   Output boxes: {pred_boxes.shape}")
    print()


def test_multi_peak_detector():
    """测试峰值检测器"""
    print("=" * 50)
    print("Test 2: MultiPeakDetector")
    print("=" * 50)
    
    from models.multi_instance_assigner import MultiPeakDetector
    
    # 创建检测器
    detector = MultiPeakDetector(
        min_peak_distance=2,
        min_peak_value=0.3
    )
    
    # 构造测试CAM (有明显的峰)
    cam = torch.zeros(7, 7)
    cam[1, 2] = 0.9  # 峰1
    cam[5, 5] = 0.8  # 峰2
    cam[3, 3] = 0.7  # 峰3
    cam[0, 0] = 0.2  # 太弱,不应该被检测
    
    # 检测
    peaks = detector.detect_peaks(cam)
    
    # 验证
    assert len(peaks) >= 2, f"Expected at least 2 peaks, got {len(peaks)}"
    # 检查峰值是否按分数排序（降序）
    scores = [p[2] for p in peaks]
    assert scores == sorted(scores, reverse=True), "Peaks should be sorted by score (descending)"
    
    print("✅ MultiPeakDetector works!")
    print(f"   Detected peaks: {len(peaks)}")
    for i, (row, col, score) in enumerate(peaks):
        print(f"   Peak {i+1}: ({row},{col}) score={score:.2f}")
    print()


def test_peak_matcher():
    """测试峰值匹配器"""
    print("=" * 50)
    print("Test 3: PeakToGTMatcher")
    print("=" * 50)
    
    from models.multi_instance_assigner import PeakToGTMatcher
    
    matcher = PeakToGTMatcher(iou_threshold=0.3)
    
    # 假设7×7的CAM
    H, W = 7, 7
    
    # 峰值
    peaks = [
        (1, 2, 0.9),  # 应该匹配GT 0
        (5, 5, 0.8),  # 应该匹配GT 1
    ]
    
    # GT boxes (归一化坐标)
    gt_boxes = torch.tensor([
        [0.2, 0.1, 0.4, 0.3],  # GT 0 (峰1在这里面)
        [0.6, 0.6, 0.9, 0.9],  # GT 1 (峰2在这里面)
    ])
    
    # 匹配
    matches, unmatched_peaks, unmatched_gts = matcher.match(
        peaks, gt_boxes, H, W
    )
    
    # 验证
    assert len(matches) >= 1, f"Expected at least 1 match, got {len(matches)}"
    
    print("✅ PeakToGTMatcher works!")
    print(f"   Matches: {matches}")
    print(f"   Unmatched peaks: {unmatched_peaks}")
    print(f"   Unmatched GTs: {unmatched_gts}")
    print()


def test_multi_instance_assigner():
    """测试完整的多实例分配器"""
    print("=" * 50)
    print("Test 4: MultiInstanceAssigner")
    print("=" * 50)
    
    from models.multi_instance_assigner import MultiInstanceAssigner
    
    assigner = MultiInstanceAssigner(
        min_peak_distance=2,
        min_peak_value=0.3
    )
    
    # 构造测试数据: 2个类别,每个类别2个实例
    C, H, W = 5, 7, 7
    cam = torch.zeros(C, H, W)
    
    # 类别0: 2个峰
    cam[0, 1, 2] = 0.9
    cam[0, 5, 5] = 0.8
    
    # 类别1: 2个峰
    cam[1, 2, 1] = 0.85
    cam[1, 4, 6] = 0.75
    
    # GT boxes
    gt_boxes = torch.tensor([
        [0.2, 0.1, 0.4, 0.3],  # 类别0 实例1
        [0.6, 0.6, 0.9, 0.9],  # 类别0 实例2
        [0.1, 0.2, 0.3, 0.4],  # 类别1 实例1
        [0.8, 0.5, 0.95, 0.7], # 类别1 实例2
    ])
    gt_labels = torch.tensor([0, 0, 1, 1])
    
    # 分配
    pos_samples = assigner.assign(cam, gt_boxes, gt_labels)
    
    # 验证
    assert len(pos_samples) > 0, f"Expected positive samples, got {len(pos_samples)}"
    
    print("✅ MultiInstanceAssigner works!")
    print(f"   Total positive samples: {len(pos_samples)}")
    for i, sample in enumerate(pos_samples[:5]):  # 只显示前5个
        print(f"   Sample {i}: GT={sample['gt_idx']}, "
              f"Class={sample['class']}, "
              f"Pos=({sample['i']},{sample['j']}), "
              f"Type={sample['match_type']}")
    print()


def test_detection_loss():
    """测试检测损失"""
    print("=" * 50)
    print("Test 5: DetectionLoss")
    print("=" * 50)
    
    from losses.detection_loss import DetectionLoss
    
    criterion = DetectionLoss(
        lambda_l1=1.0,
        lambda_giou=2.0,
        lambda_cam=0.5
    )
    
    # 假输出
    B, C, H, W = 2, 5, 7, 7
    outputs = {
        'cam': torch.rand(B, C, H, W),
        'pred_boxes': torch.rand(B, C, H, W, 4),
        'scores': torch.rand(B, C, H, W)
    }
    
    # 假目标
    targets = [
        {
            'boxes': torch.tensor([[0.2, 0.3, 0.5, 0.6], [0.6, 0.7, 0.8, 0.9]]),
            'labels': torch.tensor([0, 1])
        },
        {
            'boxes': torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            'labels': torch.tensor([2])
        }
    ]
    
    # 计算损失
    loss_dict = criterion(outputs, targets)
    
    # 验证
    assert 'loss_total' in loss_dict
    assert 'loss_box_l1' in loss_dict
    assert 'loss_box_giou' in loss_dict
    
    assert loss_dict['loss_total'].item() >= 0, "Loss should be non-negative"
    
    print("✅ DetectionLoss works!")
    print(f"   Total loss: {loss_dict['loss_total'].item():.4f}")
    print(f"   L1 loss: {loss_dict['loss_box_l1'].item():.4f}")
    print(f"   GIoU loss: {loss_dict['loss_box_giou'].item():.4f}")
    if 'loss_cam' in loss_dict:
        print(f"   CAM loss: {loss_dict['loss_cam'].item():.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Component Tests")
    print("="*50 + "\n")
    
    try:
        test_box_head()
        test_multi_peak_detector()
        test_peak_matcher()
        test_multi_instance_assigner()
        test_detection_loss()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


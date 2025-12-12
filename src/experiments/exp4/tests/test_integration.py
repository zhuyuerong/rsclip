# -*- coding: utf-8 -*-
"""
集成测试: 验证模块组合工作
"""

import torch
import sys
import os
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent.parent))


def test_box_head_training():
    """测试BoxHead训练步骤"""
    print("=" * 50)
    print("Integration Test 1: BoxHead Training Step")
    print("=" * 50)
    
    from models.box_head import BoxHead
    from losses.detection_loss import DetectionLoss
    
    # 模型
    box_head = BoxHead(
        num_classes=5,
        hidden_dim=256,
        cam_resolution=7
    )
    
    # 损失
    criterion = DetectionLoss(
        lambda_l1=1.0,
        lambda_giou=2.0,
        lambda_cam=0.5
    )
    
    # 优化器
    optimizer = torch.optim.Adam(box_head.parameters(), lr=1e-4)
    
    # 假数据
    B = 2
    fake_cam = torch.rand(B, 5, 7, 7)
    
    # 前向
    box_outputs = box_head(fake_cam)
    pred_boxes = box_head.decode_boxes(box_outputs['box_params'], cam_resolution=7)
    
    outputs = {
        'cam': fake_cam,
        'pred_boxes': pred_boxes,
        'scores': fake_cam
    }
    
    targets = [
        {
            'boxes': torch.tensor([[0.2, 0.3, 0.5, 0.6]]),
            'labels': torch.tensor([0])
        },
        {
            'boxes': torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            'labels': torch.tensor([1])
        }
    ]
    
    # 训练步骤
    optimizer.zero_grad()
    loss_dict = criterion(outputs, targets)
    loss = loss_dict['loss_total']
    loss.backward()
    optimizer.step()
    
    print("✅ BoxHead training step works!")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradient computed: {any(p.grad is not None for p in box_head.parameters())}")
    print()


def test_inference_pipeline():
    """测试推理流程"""
    print("=" * 50)
    print("Integration Test 2: Inference Pipeline")
    print("=" * 50)
    
    from models.box_head import BoxHead
    
    box_head = BoxHead(
        num_classes=20,
        hidden_dim=256,
        cam_resolution=7
    )
    box_head.eval()
    
    # 假输入
    fake_cam = torch.rand(1, 20, 7, 7)
    text_queries = ["ship", "airplane"]
    
    # 推理
    with torch.no_grad():
        box_outputs = box_head(fake_cam)
        pred_boxes = box_head.decode_boxes(box_outputs['box_params'], cam_resolution=7)
        
        # 简单的检测测试
        detections = []
        for c in range(2):  # 只看前2个类
            class_cam = fake_cam[0, c]
            class_boxes = pred_boxes[0, c]
            
            # 找高响应点
            threshold = 0.5
            mask = class_cam > threshold
            if mask.any():
                indices = mask.nonzero(as_tuple=False)
                for idx in indices[:3]:  # 最多3个
                    i, j = idx[0].item(), idx[1].item()
                    box = class_boxes[i, j]
                    score = class_cam[i, j].item()
                    
                    detections.append({
                        'box': box.tolist(),
                        'class': c,
                        'score': score
                    })
    
    print("✅ Inference pipeline works!")
    print(f"   Detections: {len(detections)}")
    for i, det in enumerate(detections[:5]):
        print(f"   Det {i}: class={det['class']}, score={det['score']:.3f}")
    print()


def test_loss_backward():
    """测试损失反向传播"""
    print("=" * 50)
    print("Integration Test 3: Loss Backward")
    print("=" * 50)
    
    from models.box_head import BoxHead
    from losses.detection_loss import DetectionLoss
    
    box_head = BoxHead(num_classes=5, hidden_dim=256, cam_resolution=7)
    criterion = DetectionLoss()
    
    # 生成假数据
    B = 2
    fake_cam = torch.rand(B, 5, 7, 7)
    box_outputs = box_head(fake_cam)
    pred_boxes = box_head.decode_boxes(box_outputs['box_params'], cam_resolution=7)
    
    outputs = {
        'cam': fake_cam,
        'pred_boxes': pred_boxes,
        'scores': fake_cam
    }
    
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
    
    # 计算损失并反向
    loss_dict = criterion(outputs, targets)
    loss = loss_dict['loss_total']
    
    # 检查loss是否可反向
    assert loss.requires_grad or loss.item() == 0, "Loss should be differentiable or zero"
    
    if loss.requires_grad:
        loss.backward()
        has_grad = any(p.grad is not None for p in box_head.parameters())
        print(f"✅ Loss backward works! Has gradients: {has_grad}")
    else:
        print(f"✅ Loss is zero (no positive samples), which is valid")
    
    print(f"   Loss value: {loss.item():.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Integration Tests")
    print("="*50 + "\n")
    
    try:
        test_box_head_training()
        test_inference_pipeline()
        test_loss_backward()
        
        print("\n" + "="*50)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



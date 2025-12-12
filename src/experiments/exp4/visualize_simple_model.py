#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化简化SurgeryCAM模型的检测结果
"""

import matplotlib
matplotlib.use('Agg')

import torch
import sys
import json
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.dior_detection import get_detection_dataloader
from models.simple_surgery_cam_detector import create_simple_surgery_cam_detector

# DIOR类别列表
DIOR_CLASSES = [
    "airplane", "airport", "baseball field", "basketball court",
    "bridge", "chimney", "dam", "expressway service area",
    "expressway toll station", "golf course", "ground track field",
    "harbor", "overpass", "ship", "stadium", "storage tank",
    "tennis court", "train station", "vehicle", "wind mill"
]

def load_image(image_path: str, image_size: int = 224):
    """加载和预处理图片"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image, original_size

def visualize_results(image_pil, cam, detections_list, text_queries, class_idx, output_path):
    """可视化结果"""
    fig = plt.figure(figsize=(20, 6))
    
    # 图1: 原始图像
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image_pil)
    ax1.set_title("原始图像", fontsize=16, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # 图2: CAM热力图
    ax2 = plt.subplot(1, 3, 2)
    if class_idx is not None and class_idx < cam.shape[0]:
        cam_vis = cam[class_idx].cpu().numpy()
        class_name = text_queries[class_idx] if class_idx < len(text_queries) else f"Class {class_idx}"
    else:
        cam_vis = cam.max(dim=0)[0].cpu().numpy()
        class_name = "All Classes"
    
    h_img, w_img = image_pil.size[1], image_pil.size[0]
    cam_resized = cv2.resize(cam_vis, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-6)
    
    image_np = np.array(image_pil)
    cam_colored = plt.cm.jet(cam_normalized)[:, :, :3]
    cam_colored = (cam_colored * 255).astype(np.uint8)
    overlay = (0.5 * cam_colored + 0.5 * image_np).astype(np.uint8)
    
    ax2.imshow(overlay)
    ax2.set_title(f"CAM热力图\n({class_name})", fontsize=16, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # 图3: 检测结果
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(image_pil)
    
    # detections_list是一个列表，包含该图像的所有检测结果
    colors = plt.cm.tab20(np.linspace(0, 1, len(text_queries)))
    
    for det in detections_list:
        box = det['box'].cpu().numpy() if isinstance(det['box'], torch.Tensor) else det['box']
        label = det['class']
        score = det['score']
        class_name = det.get('class_name', text_queries[label] if label < len(text_queries) else f"class_{label}")
        
        xmin = box[0] * w_img
        ymin = box[1] * h_img
        xmax = box[2] * w_img
        ymax = box[3] * h_img
        
        color = colors[label % len(colors)]
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2.5, edgecolor=color, facecolor='none'
        )
        ax3.add_patch(rect)
        
        ax3.text(
            xmin, ymin - 5,
            f'{class_name}: {score:.2f}',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
            fontsize=11, color='white', weight='bold'
        )
    
    ax3.set_title(f"最终检测结果\n({len(detections_list)} 个目标)", fontsize=16, fontweight='bold', pad=10)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 已保存: {output_path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载配置
    with open('configs/surgery_cam_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载数据集
    print("\n加载数据集...")
    loader = get_detection_dataloader(
        root=config.get('dataset_root'),
        split='test',
        batch_size=1,
        num_workers=0,
        image_size=config.get('image_size', 224),
        shuffle=False,
        augment=False
    )
    dataset = loader.dataset
    dataset_root = Path(dataset.root)
    print(f"✅ 数据集: {len(dataset)} 张图像")
    print(f"   根目录: {dataset_root}")
    
    # 选择10张图像
    image_ids = dataset.image_ids[:10]
    print(f"\n选择图像: {image_ids}")
    
    # 加载模型
    print("\n加载简化SurgeryCAM模型...")
    surgery_checkpoint = config.get('surgery_clip_checkpoint', 'checkpoints/RemoteCLIP-ViT-B-32.pt')
    if not Path(surgery_checkpoint).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        surgery_checkpoint = project_root / surgery_checkpoint
        surgery_checkpoint = str(surgery_checkpoint)
    
    model = create_simple_surgery_cam_detector(
        surgery_clip_checkpoint=surgery_checkpoint,
        num_classes=config.get('num_classes', 20),
        cam_resolution=config.get('cam_resolution', 7),
        upsample_cam=config.get('upsample_cam', False),
        device=device,
        unfreeze_cam_last_layer=True
    )
    
    # 加载checkpoint
    checkpoint_path = 'checkpoints/best_simple_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 只加载可训练部分的权重
    model_state = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    trainable_state = {}
    for key, value in model_state.items():
        if key in model_dict and ('box_head' in key or 'cam_generator.learnable_proj' in key):
            trainable_state[key] = value
    model_dict.update(trainable_state)
    model.load_state_dict(model_dict, strict=False)
    
    model.eval()
    print(f"✅ 使用checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"   已加载 {len(trainable_state)} 个可训练参数")
    print("✅ 模型加载完成")
    
    # 创建输出目录
    output_dir = Path('outputs/simple_model_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每张图像
    text_queries = DIOR_CLASSES.copy()
    success_count = 0
    
    print(f"\n开始处理 {len(image_ids)} 张图像...")
    for idx, image_id in enumerate(image_ids):
        print(f"\n[{idx+1}/{len(image_ids)}] {image_id}")
        
        # 获取图像路径
        image_path = dataset_root / 'images' / 'test' / f'{image_id}.jpg'
        if not image_path.exists():
            image_path = dataset_root / 'images' / 'trainval' / f'{image_id}.jpg'
        
        if not image_path.exists():
            print(f"  ⚠️  图像不存在: {image_path}")
            continue
        
        try:
            # 加载图像
            image_tensor, image_pil, _ = load_image(str(image_path))
            image_tensor = image_tensor.to(device)
            
            # 获取该图像的类别（用于CAM可视化）
            # 显示响应最高的类别，而不是第一个GT类别
            try:
                sample = dataset[dataset.image_ids.index(image_id)]
                labels = sample['labels'].numpy()
                # 先获取CAM，然后找到响应最高的类别
                with torch.no_grad():
                    temp_outputs = model(image_tensor, text_queries)
                    temp_cam = temp_outputs['cam'][0]  # [C, H, W]
                    # 计算每个类别的平均CAM响应
                    cam_means = temp_cam.mean(dim=(1, 2))  # [C]
                    class_idx = int(cam_means.argmax().item())  # 响应最高的类别
            except:
                class_idx = None
            
            # 推理
            with torch.no_grad():
                outputs = model(image_tensor, text_queries)
                cam = outputs['cam'][0]
                
                detections = model.inference(
                    image_tensor, text_queries,
                    conf_threshold=0.1,  # 使用较低的阈值以显示更多检测
                    nms_threshold=0.5,
                    topk=50,
                    max_peaks_per_class=10
                )
                detections_list = detections[0]  # 第一张图像的检测结果
            
            num_detections = len(detections_list)
            print(f"  检测到 {num_detections} 个目标")
            
            # 生成可视化
            output_path = output_dir / f"{image_id}_simple_model.jpg"
            visualize_results(image_pil, cam, detections_list, text_queries, class_idx, str(output_path))
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ 完成！成功处理 {success_count}/{len(image_ids)} 张图像")
    print(f"输出目录: {output_dir}")

if __name__ == '__main__':
    main()


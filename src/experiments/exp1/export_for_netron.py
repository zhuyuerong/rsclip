#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出模型为ONNX格式，用于Netron可视化
"""

import torch
import sys
import os
from pathlib import Path
import yaml

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.surgery_aaf import create_surgery_aaf_model


def export_onnx():
    """导出模型为ONNX格式"""
    print("=" * 80)
    print("导出模型为ONNX格式（用于Netron可视化）")
    print("=" * 80)
    
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 查找检查点
    project_root = Path(__file__).parent.parent.parent.parent
    checkpoint_path = config['clip_weights_path']
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = project_root / checkpoint_path
        checkpoint_path = str(checkpoint_path)
    
    device = 'cpu'  # ONNX导出通常在CPU上进行
    print(f"\n使用设备: {device}")
    print(f"检查点: {checkpoint_path}\n")
    
    # 创建模型
    model, preprocess = create_surgery_aaf_model(
        checkpoint_path=checkpoint_path,
        device=device,
        num_layers=config.get('num_layers', 6)
    )
    
    # 冻结CLIP参数
    for param in model.clip.parameters():
        param.requires_grad = False
    
    model.eval()
    
    # 创建示例输入
    batch_size = 1
    num_classes = 20
    images = torch.randn(batch_size, 3, 224, 224)
    
    # DIOR的20个类别
    text_queries = [
        "airplane", "airport", "baseball field", "basketball court",
        "bridge", "chimney", "dam", "expressway service area",
        "expressway toll station", "golf course", "ground track field",
        "harbor", "overpass", "ship", "stadium", "storage tank",
        "tennis court", "train station", "vehicle", "wind mill"
    ]
    
    print(f"输入图像形状: {images.shape}")
    print(f"文本查询数量: {len(text_queries)}")
    
    # 由于模型forward需要text_queries作为列表，我们需要创建一个包装器
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model, text_queries):
            super().__init__()
            self.base_model = base_model
            self.text_queries = text_queries
        
        def forward(self, images):
            # 调用原始模型的forward
            cam, aux = self.base_model(images, self.text_queries)
            return cam
    
    wrapped_model = ModelWrapper(model, text_queries)
    wrapped_model.eval()
    
    # 测试前向传播
    print("\n测试前向传播...")
    with torch.no_grad():
        output = wrapped_model(images)
        print(f"输出CAM形状: {output.shape}")
    
    # 导出ONNX
    output_path = Path(__file__).parent / 'model_for_netron.onnx'
    print(f"\n导出ONNX模型到: {output_path}")
    
    try:
        torch.onnx.export(
            wrapped_model,
            images,
            str(output_path),
            input_names=['images'],
            output_names=['cam'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'cam': {0: 'batch_size'}
            },
            opset_version=14,  # 使用较新的opset版本
            do_constant_folding=True,
            verbose=False
        )
        print(f"✅ ONNX模型导出成功: {output_path}")
        print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 导出TorchScript格式（Netron也支持，且不需要额外依赖）
    print(f"\n导出TorchScript模型...")
    try:
        traced_model = torch.jit.trace(wrapped_model, images)
        torchscript_path = Path(__file__).parent / 'model_for_netron.pt'
        traced_model.save(str(torchscript_path))
        print(f"✅ TorchScript模型导出成功: {torchscript_path}")
        print(f"   文件大小: {torchscript_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   Netron可以直接打开此文件！")
    except Exception as e:
        print(f"⚠️  TorchScript导出失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 也导出PyTorch模型（Netron也支持）
    pytorch_path = Path(__file__).parent / 'model_for_netron.pth'
    print(f"\n导出PyTorch模型到: {pytorch_path}")
    
    try:
        # 保存完整模型（包含状态）
        torch.save({
            'model': wrapped_model,
            'model_state_dict': wrapped_model.state_dict(),
            'text_queries': text_queries,
            'config': config
        }, pytorch_path)
        print(f"✅ PyTorch模型导出成功: {pytorch_path}")
        print(f"   文件大小: {pytorch_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ PyTorch导出失败: {e}")
    
    # 导出简化版本（只包含AAF部分，更轻量）
    print(f"\n导出简化模型（仅AAF部分）...")
    aaf_model = model.aaf
    aaf_model.eval()
    
    # 创建示例输入（模拟6层注意力）
    dummy_vv_attentions = [
        torch.randn(1, 12, 50, 50) for _ in range(6)
    ]
    dummy_ori_attentions = [
        torch.randn(1, 12, 50, 50) for _ in range(6)
    ]
    
    # AAF的forward需要列表输入，需要包装
    class AAFWrapper(torch.nn.Module):
        def __init__(self, aaf):
            super().__init__()
            self.aaf = aaf
        
        def forward(self, vv_attn_0, vv_attn_1, vv_attn_2, vv_attn_3, vv_attn_4, vv_attn_5,
                   ori_attn_0, ori_attn_1, ori_attn_2, ori_attn_3, ori_attn_4, ori_attn_5):
            vv_attentions = [vv_attn_0, vv_attn_1, vv_attn_2, vv_attn_3, vv_attn_4, vv_attn_5]
            ori_attentions = [ori_attn_0, ori_attn_1, ori_attn_2, ori_attn_3, ori_attn_4, ori_attn_5]
            return self.aaf(vv_attentions, ori_attentions)
    
    aaf_wrapped = AAFWrapper(aaf_model)
    
    aaf_onnx_path = Path(__file__).parent / 'aaf_for_netron.onnx'
    try:
        torch.onnx.export(
            aaf_wrapped,
            (*dummy_vv_attentions, *dummy_ori_attentions),
            str(aaf_onnx_path),
            input_names=[f'vv_attn_{i}' for i in range(6)] + [f'ori_attn_{i}' for i in range(6)],
            output_names=['attn_p2p'],
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )
        print(f"✅ AAF ONNX模型导出成功: {aaf_onnx_path}")
        print(f"   文件大小: {aaf_onnx_path.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print(f"⚠️  AAF ONNX导出失败: {e}")
    
    print("\n" + "=" * 80)
    print("导出完成！")
    print("=" * 80)
    print("\n生成的文件：")
    print(f"1. {output_path} - 完整模型（ONNX格式）")
    print(f"2. {pytorch_path} - 完整模型（PyTorch格式）")
    if aaf_onnx_path.exists():
        print(f"3. {aaf_onnx_path} - AAF层（ONNX格式，轻量级）")
    
    print("\n使用方法：")
    print("1. 打开 https://netron.app/ 或下载Netron桌面版")
    print("2. 拖拽上述文件到Netron中即可查看模型结构")
    print("3. 推荐使用 model_for_netron.onnx 查看完整模型")
    print("4. 使用 aaf_for_netron.onnx 查看AAF层详细结构")
    
    return True


if __name__ == '__main__':
    export_onnx()


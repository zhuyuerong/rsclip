# -*- coding: utf-8 -*-
"""
实验4 Demo - 快速测试
展示完整的推理流程
"""

import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.config import get_config
from experiment4.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.models.decomposer import ImageOnlyDecomposer
from experiment4.models.noise_filter import RuleBasedDenoiser
from torchvision import transforms


class Experiment4Demo:
    """实验4演示"""
    
    def __init__(self, checkpoint_path=None):
        print("=" * 60)
        print("实验4 Demo - Surgery + 文本引导稀疏分解 + 规则去噪")
        print("=" * 60)
        
        self.config = get_config()
        self.device = self.config.device
        
        print(f"\n设备: {self.device}")
        
        # 加载模型
        self._load_models(checkpoint_path)
        
        # 准备transform
        self.transform = self._get_transform()
        
        # 准备类别
        self._prepare_classes()
    
    def _load_models(self, checkpoint_path):
        """加载模型"""
        print("\n加载模型...")
        
        # Surgery模型
        self.surgery_model = CLIPSurgeryWrapper(self.config)
        print("  ✓ CLIP Surgery")
        
        # 背景特征
        self.bg_features = self.surgery_model.encode_text(self.config.background_words)
        print("  ✓ 背景词表")
        
        # 去噪器
        self.denoiser = RuleBasedDenoiser(self.bg_features, self.config)
        print("  ✓ 规则去噪器")
        
        # 图像分解器
        self.img_decomposer = ImageOnlyDecomposer(self.config).to(self.device)
        print("  ✓ 图像分解器")
        
        # 加载权重（如果有）
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.img_decomposer.load_state_dict(checkpoint['img_decomposer'])
            print(f"  ✓ 加载权重: {checkpoint_path}")
        else:
            print("  ! 使用随机初始化权重（仅用于演示）")
        
        self.img_decomposer.eval()
    
    def _get_transform(self):
        """获取图像变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def _prepare_classes(self):
        """准备类别列表"""
        # 遥感常见类别
        self.classes = [
            'airplane', 'airport', 'baseballfield', 'basketballcourt',
            'beach', 'bridge', 'chaparral', 'church',
            'circularfarmland', 'cloud', 'denseresidential', 'forest',
            'freeway', 'golfcourse', 'harbor', 'intersection',
            'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot'
        ]
        
        # 编码文本
        print("\n准备类别...")
        self.class_features = self.surgery_model.encode_text(self.classes).to(self.device)
        print(f"  ✓ {len(self.classes)} 个类别")
    
    @torch.no_grad()
    def predict(self, image_path_or_tensor):
        """
        预测图像
        
        Args:
            image_path_or_tensor: 图像路径或tensor
        
        Returns:
            result: dict
        """
        # 加载图像
        if isinstance(image_path_or_tensor, str):
            image = Image.open(image_path_or_tensor).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            image_np = np.array(image.resize((224, 224)))
        else:
            image_tensor = image_path_or_tensor
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # 反归一化用于显示
            img = image_tensor[0].cpu().permute(1, 2, 0).numpy()
            mean = np.array(self.config.normalize_mean)
            std = np.array(self.config.normalize_std)
            image_np = np.clip(img * std + mean, 0, 1)
        
        # ===== 核心推理流程 =====
        
        # 1. Surgery特征提取
        F_img = self.surgery_model.get_patch_features(image_tensor)  # [1, 196, 512]
        
        # 2. 规则去噪
        F_clean, denoise_info = self.denoiser(F_img)
        
        # 3. 图像分解
        Q_img = self.img_decomposer(F_clean)  # [1, 196, 512, 20]
        
        # 4. 聚合特征
        import torch.nn.functional as F
        attention = Q_img.abs().sum(dim=2)  # [1, 196, 20]
        attention = F.softmax(attention, dim=1)
        
        global_features = []
        for m in range(self.config.n_components):
            attn_m = attention[:, :, m].unsqueeze(-1)
            feat_m = (F_clean * attn_m).sum(dim=1)
            global_features.append(feat_m)
        
        global_feat = torch.stack(global_features, dim=1).mean(dim=1)  # [1, 512]
        global_feat = F.normalize(global_feat, dim=1)
        
        # 5. 分类
        logits = global_feat @ self.class_features.T * 100
        probs = logits.softmax(dim=1)[0]
        
        # Top-5预测
        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_classes = [self.classes[idx] for idx in top5_indices.cpu().numpy()]
        
        # Attention map
        attention_max = attention.max(dim=-1)[0]  # [1, 196]
        attention_map = attention_max.reshape(14, 14).cpu().numpy()
        
        # 结果
        result = {
            'top1_class': top5_classes[0],
            'top1_prob': top5_probs[0].item(),
            'top5_classes': top5_classes,
            'top5_probs': top5_probs.cpu().numpy(),
            'attention_map': attention_map,
            'image': image_np,
            'denoise_info': {
                'fg_ratio': denoise_info['fg_ratio'],
                'noise_reduction': denoise_info['noise_reduction_ratio']
            },
            'sparsity': (Q_img.abs() > 1e-6).float().mean().item()
        }
        
        return result
    
    def visualize_result(self, result, save_path=None):
        """可视化结果"""
        fig = plt.figure(figsize=(15, 5))
        
        # 1. 原图
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(result['image'])
        ax1.set_title(f"输入图像", fontsize=12)
        ax1.axis('off')
        
        # 2. Attention Map
        ax2 = plt.subplot(1, 3, 2)
        im = ax2.imshow(result['attention_map'], cmap='hot', interpolation='bilinear')
        ax2.set_title(f"注意力图", fontsize=12)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        
        # 3. 预测结果
        ax3 = plt.subplot(1, 3, 3)
        ax3.axis('off')
        
        # 标题
        title_text = f"预测: {result['top1_class']}\n"
        title_text += f"置信度: {result['top1_prob']:.2%}\n\n"
        title_text += "Top-5:\n"
        for i, (cls, prob) in enumerate(zip(result['top5_classes'], result['top5_probs'])):
            title_text += f"{i+1}. {cls}: {prob:.2%}\n"
        
        title_text += f"\n稀疏度: {result['sparsity']:.2%}"
        title_text += f"\n前景比例: {result['denoise_info']['fg_ratio']:.2%}"
        
        ax3.text(0.1, 0.9, title_text, 
                transform=ax3.transAxes,
                fontsize=11,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n可视化保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def demo_single_image(self, image_path, save_path=None):
        """演示单张图像"""
        print(f"\n{'='*60}")
        print(f"推理图像: {image_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(image_path):
            print(f"错误: 图像不存在 {image_path}")
            return
        
        # 预测
        result = self.predict(image_path)
        
        # 打印结果
        print(f"\n预测结果:")
        print(f"  Top-1: {result['top1_class']} ({result['top1_prob']:.2%})")
        print(f"\n  Top-5:")
        for i, (cls, prob) in enumerate(zip(result['top5_classes'], result['top5_probs'])):
            print(f"    {i+1}. {cls}: {prob:.2%}")
        
        print(f"\n  稀疏度: {result['sparsity']:.2%}")
        print(f"  前景比例: {result['denoise_info']['fg_ratio']:.2%}")
        print(f"  噪声降低: {result['denoise_info']['noise_reduction']:.2%}")
        
        # 可视化
        if save_path is None:
            save_path = image_path.replace('.jpg', '_result.png').replace('.png', '_result.png')
        
        self.visualize_result(result, save_path)
        
        return result


def main():
    """主函数"""
    # 检查点路径（可选）
    checkpoint_path = "experiment4/checkpoints/best.pth"
    
    # 创建demo
    demo = Experiment4Demo(checkpoint_path)
    
    # 测试图像
    test_images = [
        "assets/airport.jpg",
        "datasets/mini_dataset/images/airplane_001.jpg",
    ]
    
    # 遍历测试
    for img_path in test_images:
        if os.path.exists(img_path):
            demo.demo_single_image(img_path)
            break
    else:
        print("\n没有找到测试图像，请提供图像路径：")
        print("  python experiment4/demo.py <image_path>")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 命令行指定图像
        demo = Experiment4Demo("experiment4/checkpoints/best.pth")
        demo.demo_single_image(sys.argv[1])
    else:
        main()


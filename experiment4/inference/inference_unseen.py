# -*- coding: utf-8 -*-
"""
Unseen类推理脚本 - Zero-shot评估
关键：只用image-only分解器（不依赖seen类知识）
"""

import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.core.config import get_config
from experiment4.core.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.core.models.decomposer import ImageOnlyDecomposer
from experiment4.core.models.noise_filter import RuleBasedDenoiser
from experiment4.core.data.dataset import MiniDataset, UnseenDataset


class UnseenInference:
    """Unseen类推理器 - Zero-shot"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = config.device
        self.checkpoint_path = checkpoint_path
        
        # 加载模型
        self._load_models()
        
        # 加载数据
        self._load_data()
    
    def _load_models(self):
        """加载模型"""
        print("加载模型...")
        
        # Surgery模型
        self.surgery_model = CLIPSurgeryWrapper(self.config)
        
        # 背景特征
        self.bg_features = self.surgery_model.encode_text(self.config.background_words)
        
        # 去噪器
        self.denoiser = RuleBasedDenoiser(self.bg_features, self.config)
        
        # 图像分解器（关键：只用这个，不用text分解器）
        self.img_decomposer = ImageOnlyDecomposer(self.config).to(self.device)
        
        # 加载权重
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.img_decomposer.load_state_dict(checkpoint['img_decomposer'])
            print(f"  ✓ 加载权重: {self.checkpoint_path}")
            print(f"  ✓ Epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            print(f"  ✗ 权重文件不存在: {self.checkpoint_path}")
            print(f"  使用随机初始化权重")
        
        # 评估模式
        self.img_decomposer.eval()
    
    def _load_data(self):
        """加载数据"""
        print("\n加载数据...")
        
        # 创建完整数据集
        full_dataset = MiniDataset(
            root_dir=self.config.dataset_root,
            split='test',
            seen_ratio=0.75,
            config=self.config
        )
        
        # 创建unseen子集
        unseen_dataset = UnseenDataset(full_dataset)
        
        self.test_loader = torch.utils.data.DataLoader(
            unseen_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        self.unseen_classes = full_dataset.unseen_classes
        self.dataset = full_dataset
        
        # Unseen类文本特征
        self.unseen_text_features = self.surgery_model.encode_text(self.unseen_classes).to(self.device)
        
        print(f"  ✓ 测试集: {len(self.test_loader.dataset)} 样本")
        print(f"  ✓ Unseen类: {len(self.unseen_classes)}")
        print(f"  类别: {self.unseen_classes}")
    
    @torch.no_grad()
    def infer_single(self, image):
        """
        推理单张图像
        
        Args:
            image: [1, 3, 224, 224] or [3, 224, 224]
        
        Returns:
            result: dict
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # 提取特征
        F_img = self.surgery_model.get_patch_features(image)  # [1, 196, 512]
        
        # 去噪
        F_clean, _ = self.denoiser(F_img)
        
        # 纯图像分解（不用text引导）
        Q_img = self.img_decomposer(F_clean)  # [1, 196, 512, 20]
        
        # 聚合全局特征
        attention = Q_img.abs().sum(dim=2)  # [1, 196, 20]
        attention = F.softmax(attention, dim=1)
        
        global_features = []
        for m in range(self.config.n_components):
            attn_m = attention[:, :, m].unsqueeze(-1)
            feat_m = (F_clean * attn_m).sum(dim=1)
            global_features.append(feat_m)
        
        global_feat = torch.stack(global_features, dim=1).mean(dim=1)  # [1, 512]
        global_feat = F.normalize(global_feat, dim=1)
        
        # 与unseen类文本匹配
        logits = global_feat @ self.unseen_text_features.T * 100  # [1, N_unseen]
        
        # 预测
        pred_idx = logits.argmax(dim=1).item()
        pred_class = self.unseen_classes[pred_idx]
        confidence = logits.softmax(dim=1)[0, pred_idx].item()
        
        # 定位图
        attention_max = attention.max(dim=-1)[0]  # [1, 196]
        loc_map = attention_max.reshape(14, 14).cpu().numpy()
        
        result = {
            'pred_class': pred_class,
            'pred_idx': pred_idx,
            'confidence': confidence,
            'logits': logits.cpu().numpy(),
            'loc_map': loc_map,
            'all_probs': logits.softmax(dim=1).cpu().numpy()
        }
        
        return result
    
    @torch.no_grad()
    def evaluate(self):
        """评估unseen类性能"""
        print("\n" + "=" * 50)
        print("评估Unseen类性能 - Zero-shot")
        print("=" * 50)
        
        all_preds = []
        all_labels = []
        all_confidences = []
        all_class_names = []
        
        for batch in tqdm(self.test_loader, desc="推理中"):
            images = batch['image'].to(self.device)
            labels = batch['label']
            class_names = batch['class_name']
            
            # 批量推理
            batch_size = len(images)
            
            # 提取特征
            F_img = self.surgery_model.get_patch_features(images)
            F_clean, _ = self.denoiser(F_img)
            
            # 分解
            Q_img = self.img_decomposer(F_clean)
            
            # 聚合
            attention = Q_img.abs().sum(dim=2)
            attention = F.softmax(attention, dim=1)
            
            global_features = []
            for m in range(self.config.n_components):
                attn_m = attention[:, :, m].unsqueeze(-1)
                feat_m = (F_clean * attn_m).sum(dim=1)
                global_features.append(feat_m)
            
            global_feat = torch.stack(global_features, dim=1).mean(dim=1)
            global_feat = F.normalize(global_feat, dim=1)
            
            # 匹配
            logits = global_feat @ self.unseen_text_features.T * 100
            
            # 预测
            preds = logits.argmax(dim=1).cpu()
            confidences = logits.softmax(dim=1).max(dim=1)[0].cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.numpy())
            all_class_names.extend(class_names)
        
        # 转换预测索引到原始标签空间
        # unseen_classes在完整类别列表中的索引
        unseen_indices = [self.dataset.class_to_idx[cls] for cls in self.unseen_classes]
        
        # 将预测从unseen索引转换到原始索引
        all_preds_original = [unseen_indices[p] for p in all_preds]
        
        all_preds_original = np.array(all_preds_original)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        # 计算准确率
        accuracy = (all_preds_original == all_labels).mean()
        avg_confidence = all_confidences.mean()
        
        # 每个类别的准确率
        class_acc = {}
        class_counts = {}
        for class_name in self.unseen_classes:
            class_idx = self.dataset.class_to_idx[class_name]
            mask = all_labels == class_idx
            class_counts[class_name] = mask.sum()
            if mask.sum() > 0:
                class_acc[class_name] = (all_preds_original[mask] == all_labels[mask]).mean()
            else:
                class_acc[class_name] = 0.0
        
        # 打印结果
        print(f"\nZero-shot准确率: {accuracy:.4f}")
        print(f"平均置信度: {avg_confidence:.4f}")
        print(f"\n各类别准确率:")
        for class_name in sorted(class_acc.keys()):
            acc = class_acc[class_name]
            count = class_counts[class_name]
            print(f"  {class_name}: {acc:.4f} ({count} 样本)")
        
        # 保存结果
        results = {
            'overall_accuracy': float(accuracy),
            'average_confidence': float(avg_confidence),
            'class_accuracy': {k: float(v) for k, v in class_acc.items()},
            'class_counts': {k: int(v) for k, v in class_counts.items()},
            'num_samples': len(all_preds),
            'num_classes': len(self.unseen_classes),
            'unseen_classes': self.unseen_classes
        }
        
        output_path = os.path.join(self.config.output_dir, 'unseen_inference_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n结果保存至: {output_path}")
        
        return results
    
    def visualize_sample(self, sample_idx=0, save_path=None):
        """可视化单个样本"""
        # 获取样本
        sample = self.test_loader.dataset[sample_idx]
        image = sample['image'].unsqueeze(0).to(self.device)
        true_label = sample['label']
        true_class = sample['class_name']
        
        # 推理
        result = self.infer_single(image)
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 原图
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        # 反归一化
        mean = np.array(self.config.normalize_mean)
        std = np.array(self.config.normalize_std)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        axes[0].imshow(img_np)
        axes[0].set_title(f"True: {true_class}\nPred: {result['pred_class']}")
        axes[0].axis('off')
        
        # 定位图
        loc_map = result['loc_map']
        axes[1].imshow(loc_map, cmap='hot')
        axes[1].set_title(f"Attention Map\nConfidence: {result['confidence']:.4f}")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """主函数"""
    # 配置
    config = get_config()
    
    # 检查点路径
    checkpoint_path = os.path.join(config.checkpoint_dir, 'best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"警告: 最佳模型不存在 {checkpoint_path}")
        checkpoint_path = os.path.join(config.checkpoint_dir, 'latest.pth')
        print(f"尝试使用最新模型: {checkpoint_path}")
    
    # 创建推理器
    inferencer = UnseenInference(config, checkpoint_path)
    
    # 评估
    results = inferencer.evaluate()
    
    # 可视化几个样本
    print("\n生成可视化...")
    os.makedirs(os.path.join(config.output_dir, 'visualizations'), exist_ok=True)
    
    for i in range(min(5, len(inferencer.test_loader.dataset))):
        save_path = os.path.join(config.output_dir, 'visualizations', f'unseen_sample_{i}.png')
        inferencer.visualize_sample(i, save_path)


if __name__ == "__main__":
    main()


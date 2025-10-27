# -*- coding: utf-8 -*-
"""
Seen类推理脚本
对seen类进行推理和评估
"""

import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.config import get_config
from experiment4.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.models.decomposer import TextGuidedDecomposer, ImageOnlyDecomposer
from experiment4.models.noise_filter import RuleBasedDenoiser
from experiment4.data.dataset import get_dataloaders


class SeenInference:
    """Seen类推理器"""
    
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
        
        # 分解器
        self.text_decomposer = TextGuidedDecomposer(self.config).to(self.device)
        self.img_decomposer = ImageOnlyDecomposer(self.config).to(self.device)
        
        # 加载权重
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.text_decomposer.load_state_dict(checkpoint['text_decomposer'])
            self.img_decomposer.load_state_dict(checkpoint['img_decomposer'])
            print(f"  ✓ 加载权重: {self.checkpoint_path}")
            print(f"  ✓ Epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            print(f"  ✗ 权重文件不存在: {self.checkpoint_path}")
            print(f"  使用随机初始化权重")
        
        # 设置为评估模式
        self.text_decomposer.eval()
        self.img_decomposer.eval()
    
    def _load_data(self):
        """加载数据"""
        print("\n加载数据...")
        
        _, self.val_loader, _, self.dataset = get_dataloaders(self.config)
        
        self.seen_classes = self.dataset.seen_classes
        
        # Seen类文本特征
        self.seen_text_features = self.surgery_model.encode_text(self.seen_classes).to(self.device)
        
        print(f"  ✓ 验证集: {len(self.val_loader.dataset)} 样本")
        print(f"  ✓ Seen类: {len(self.seen_classes)}")
    
    @torch.no_grad()
    def infer_batch(self, images, use_text_decomposer=True):
        """
        推理一个batch
        
        Args:
            images: [B, 3, 224, 224]
            use_text_decomposer: 使用text或img分解器
        
        Returns:
            logits: [B, N_classes]
            attention_maps: [B, 14, 14]
        """
        B = len(images)
        
        # 提取特征
        F_img = self.surgery_model.get_patch_features(images)  # [B, 196, D]
        
        # 去噪
        F_clean, _ = self.denoiser(F_img)
        
        # 分解
        if use_text_decomposer:
            # 使用text分解器（需要WordNet特征，这里简化为类别名）
            # 实际应用中，可以为每个样本提供WordNet词表
            Q = self.img_decomposer(F_clean)  # 简化：用img分解器
        else:
            Q = self.img_decomposer(F_clean)
        
        # 聚合特征
        attention = Q.abs().sum(dim=2)  # [B, 196, 20]
        attention = F.softmax(attention, dim=1)
        
        # 加权池化
        global_features = []
        for m in range(self.config.n_components):
            attn_m = attention[:, :, m].unsqueeze(-1)  # [B, 196, 1]
            feat_m = (F_clean * attn_m).sum(dim=1)  # [B, 512]
            global_features.append(feat_m)
        
        global_feat = torch.stack(global_features, dim=1).mean(dim=1)  # [B, 512]
        global_feat = F.normalize(global_feat, dim=1)
        
        # 与文本匹配
        text_features_norm = F.normalize(self.seen_text_features, dim=1)
        logits = global_feat @ text_features_norm.T * 100  # [B, N_classes]
        
        # Attention map
        attention_max = attention.max(dim=-1)[0]  # [B, 196]
        attention_maps = attention_max.reshape(B, 14, 14)
        
        return logits, attention_maps
    
    def evaluate(self):
        """评估seen类性能"""
        print("\n" + "=" * 50)
        print("评估Seen类性能")
        print("=" * 50)
        
        all_preds = []
        all_labels = []
        all_class_names = []
        
        for batch in tqdm(self.val_loader, desc="推理中"):
            images = batch['image'].to(self.device)
            labels = batch['label']
            class_names = batch['class_name']
            
            # 推理
            logits, _ = self.infer_batch(images, use_text_decomposer=False)
            
            # 预测
            preds = logits.argmax(dim=1).cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_class_names.extend(class_names)
        
        # 计算准确率
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        # 每个类别的准确率
        class_acc = {}
        for class_name in self.seen_classes:
            class_idx = self.dataset.class_to_idx[class_name]
            mask = all_labels == class_idx
            if mask.sum() > 0:
                class_acc[class_name] = (all_preds[mask] == all_labels[mask]).mean()
            else:
                class_acc[class_name] = 0.0
        
        # 打印结果
        print(f"\n整体准确率: {accuracy:.4f}")
        print(f"\n各类别准确率:")
        for class_name, acc in sorted(class_acc.items()):
            print(f"  {class_name}: {acc:.4f}")
        
        # 保存结果
        results = {
            'overall_accuracy': float(accuracy),
            'class_accuracy': {k: float(v) for k, v in class_acc.items()},
            'num_samples': len(all_preds),
            'num_classes': len(self.seen_classes)
        }
        
        output_path = os.path.join(self.config.output_dir, 'seen_inference_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n结果保存至: {output_path}")
        
        return results


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
    inferencer = SeenInference(config, checkpoint_path)
    
    # 评估
    results = inferencer.evaluate()


if __name__ == "__main__":
    main()


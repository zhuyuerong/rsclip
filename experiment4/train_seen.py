# -*- coding: utf-8 -*-
"""
实验4训练脚本 - Seen类训练
Surgery + 文本引导稀疏分解 + 规则去噪
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment4.config import get_config
from experiment4.models.clip_surgery import CLIPSurgeryWrapper
from experiment4.models.decomposer import TextGuidedDecomposer, ImageOnlyDecomposer
from experiment4.models.noise_filter import RuleBasedDenoiser
from experiment4.data.dataset import get_dataloaders
from experiment4.data.wordnet_utils import get_wordnet_words
from experiment4.losses import compute_total_loss


class Experiment4Trainer:
    """实验4训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 创建保存目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 初始化模型
        self._init_models()
        
        # 初始化数据
        self._init_data()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def _init_models(self):
        """初始化模型"""
        print("初始化模型...")
        
        # Surgery模型（预训练，frozen）
        self.surgery_model = CLIPSurgeryWrapper(self.config)
        print(f"  ✓ CLIP Surgery加载完成")
        
        # 背景词表特征（预计算）
        self.bg_features = self.surgery_model.encode_text(self.config.background_words)
        print(f"  ✓ 背景词表: {len(self.config.background_words)} 个词")
        
        # 去噪器（规则，无参数）
        self.denoiser = RuleBasedDenoiser(self.bg_features, self.config)
        print(f"  ✓ 规则去噪器初始化完成")
        
        # 文本引导分解器（需要训练）
        self.text_decomposer = TextGuidedDecomposer(self.config).to(self.device)
        print(f"  ✓ 文本引导分解器: {sum(p.numel() for p in self.text_decomposer.parameters())} 参数")
        
        # 纯图像分解器（用于unseen泛化）
        self.img_decomposer = ImageOnlyDecomposer(self.config).to(self.device)
        print(f"  ✓ 图像分解器: {sum(p.numel() for p in self.img_decomposer.parameters())} 参数")
    
    def _init_data(self):
        """初始化数据"""
        print("\n加载数据...")
        
        # 获取数据加载器
        self.train_loader, self.val_loader, self.test_loader, self.dataset = get_dataloaders(self.config)
        
        # 获取seen类别名
        self.seen_classes = self.dataset.seen_classes
        self.unseen_classes = self.dataset.unseen_classes
        
        print(f"  ✓ 训练集: {len(self.train_loader.dataset)} 样本")
        print(f"  ✓ 验证集: {len(self.val_loader.dataset)} 样本")
        print(f"  ✓ Seen类: {len(self.seen_classes)}")
        print(f"  ✓ Unseen类: {len(self.unseen_classes)}")
        
        # 预计算seen类文本特征
        self.seen_text_features = self.surgery_model.encode_text(self.seen_classes).to(self.device)
        print(f"  ✓ Seen类文本特征: {self.seen_text_features.shape}")
        
        # 预计算WordNet词表
        self.wordnet_cache = {}
        for class_name in self.seen_classes:
            words = get_wordnet_words(class_name, k=self.config.wordnet_k)
            word_features = self.surgery_model.encode_text(words).to(self.device)
            # 如果需要，投影到正确的维度
            if word_features.shape[-1] != self.config.embed_dim:
                word_features = self._project_text_features(word_features)
            self.wordnet_cache[class_name] = word_features
        print(f"  ✓ WordNet词表缓存: {len(self.wordnet_cache)} 个类别")
    
    def _project_text_features(self, text_features):
        """将512维文本特征投影到embed_dim维度"""
        import torch.nn.functional as F
        # 使用线性插值
        projected = F.interpolate(
            text_features.unsqueeze(0).unsqueeze(0),
            size=(text_features.shape[0], self.config.embed_dim),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        return projected
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 优化器
        self.optimizer = torch.optim.AdamW([
            {'params': self.text_decomposer.parameters(), 'name': 'text_decomposer'},
            {'params': self.img_decomposer.parameters(), 'name': 'img_decomposer'}
        ], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.min_lr
        )
        
        print(f"\n优化器: AdamW, LR={self.config.learning_rate}")
        print(f"调度器: CosineAnnealing, min_lr={self.config.min_lr}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.text_decomposer.train()
        self.img_decomposer.train()
        
        total_loss = 0
        total_acc = 0
        loss_items = {
            'cls_text': 0,
            'cls_img': 0,
            'loc': 0,
            'sparse': 0,
            'ortho': 0,
            'align': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            class_names = batch['class_name']
            has_bbox = batch['has_bbox']
            
            # bbox（只用有标注的）
            if has_bbox.any():
                bboxes = batch['bbox'][has_bbox].to(self.device)
            else:
                bboxes = None
            
            # ===== 前向传播 =====
            # 1. 提取Surgery特征
            with torch.no_grad():
                F_img = self.surgery_model.get_patch_features(images)  # [B, 196, D]
            
            # ========== 插桩1：Surgery输出验证 ==========
            if batch_idx == 0 and epoch == 1:  # 只在第一个batch打印
                print(f"\n{'='*60}")
                print(f"[插桩1] Surgery输出验证 (Batch {batch_idx})")
                print(f"{'='*60}")
                print(f"F_img.shape: {F_img.shape}")
                print(f"F_img.dtype: {F_img.dtype}")
                print(f"F_img.mean(): {F_img.mean().item():.4f}")
                print(f"F_img.std(): {F_img.std().item():.4f}")
                print(f"F_img.max(): {F_img.max().item():.4f}")
                print(f"F_img.min(): {F_img.min().item():.4f}")
                
                # 检查是否有异常值
                nan_count = torch.isnan(F_img).sum().item()
                inf_count = torch.isinf(F_img).sum().item()
                print(f"NaN数量: {nan_count}")
                print(f"Inf数量: {inf_count}")
                print(f"{'='*60}\n")
            
            # 2. 规则去噪
            F_clean, denoise_info = self.denoiser(F_img)
            
            # ========== 插桩2：去噪效果验证 ==========
            if batch_idx == 0 and epoch == 1:
                print(f"\n{'='*60}")
                print(f"[插桩2] 去噪效果验证")
                print(f"{'='*60}")
                print(f"去噪前 F_img:")
                print(f"  shape: {F_img.shape}")
                print(f"  mean: {F_img.mean().item():.4f}")
                print(f"  std: {F_img.std().item():.4f}")
                
                print(f"\n去噪后 F_clean:")
                print(f"  shape: {F_clean.shape}")
                print(f"  mean: {F_clean.mean().item():.4f}")
                print(f"  std: {F_clean.std().item():.4f}")
                print(f"  max: {F_clean.max().item():.4f}")
                print(f"  min: {F_clean.min().item():.4f}")
                
                print(f"\n去噪信息:")
                print(f"  前景比例: {denoise_info.get('fg_ratio', 0):.2%}")
                print(f"  噪声降低: {denoise_info.get('noise_reduction_ratio', 0):.4f}")
                
                # 统计0值比例（稀疏度）
                zero_ratio = (F_clean.abs() < 1e-6).float().mean().item()
                print(f"  0值比例: {zero_ratio:.2%}")
                print(f"{'='*60}\n")
            
            # 3. 文本引导分解
            # 为每个样本获取WordNet特征
            wordnet_features_batch = []
            for class_name in class_names:
                wordnet_features_batch.append(self.wordnet_cache[class_name])
            
            # 批量分解
            Q_text_list = []
            for b in range(len(images)):
                Q_b, _ = self.text_decomposer(
                    F_clean[b:b+1],
                    wordnet_features_batch[b]
                )
                Q_text_list.append(Q_b)
            Q_text = torch.cat(Q_text_list, dim=0)
            
            # 4. 图像分解
            Q_img = self.img_decomposer(F_clean)
            
            # ===== 计算损失 =====
            loss, loss_dict = compute_total_loss(
                Q_text, Q_img, F_clean,
                self.seen_text_features, labels,
                bboxes, self.config
            )
            
            # ===== 反向传播 =====
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.text_decomposer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.img_decomposer.parameters(), 1.0)
            
            self.optimizer.step()
            
            # ===== 统计 =====
            total_loss += loss_dict['total']
            total_acc += loss_dict['acc_text']
            
            for key in loss_items:
                if key in loss_dict:
                    loss_items[key] += loss_dict[key]
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'acc': f"{loss_dict['acc_text']:.4f}"
            })
        
        # 平均
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        for key in loss_items:
            loss_items[key] /= num_batches
        
        return avg_loss, avg_acc, loss_items
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.text_decomposer.eval()
        self.img_decomposer.eval()
        
        total_loss = 0
        total_acc_text = 0
        total_acc_img = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            class_names = batch['class_name']
            
            # 提取特征
            F_img = self.surgery_model.get_patch_features(images)
            F_clean, _ = self.denoiser(F_img)
            
            # 分解（动态生成缺失的wordnet特征）
            wordnet_features_batch = []
            for cn in class_names:
                if cn not in self.wordnet_cache:
                    # 动态生成wordnet特征
                    words = get_wordnet_words(cn, k=self.config.wordnet_k)
                    word_features = self.surgery_model.encode_text(words).to(self.device)
                    self.wordnet_cache[cn] = word_features
                wordnet_features_batch.append(self.wordnet_cache[cn])
            Q_text_list = []
            for b in range(len(images)):
                Q_b, _ = self.text_decomposer(F_clean[b:b+1], wordnet_features_batch[b])
                Q_text_list.append(Q_b)
            Q_text = torch.cat(Q_text_list, dim=0)
            
            Q_img = self.img_decomposer(F_clean)
            
            # 计算损失
            loss, loss_dict = compute_total_loss(
                Q_text, Q_img, F_clean,
                self.seen_text_features, labels,
                None, self.config
            )
            
            total_loss += loss_dict['total']
            total_acc_text += loss_dict['acc_text']
            total_acc_img += loss_dict['acc_img']
        
        # 平均
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_acc_text = total_acc_text / num_batches
        avg_acc_img = total_acc_img / num_batches
        
        return avg_loss, avg_acc_text, avg_acc_img
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'text_decomposer': self.text_decomposer.state_dict(),
            'img_decomposer': self.img_decomposer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'history': self.history,
            'config': vars(self.config)
        }
        
        # 保存最新
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 保存最佳模型: {best_path}")
        
        # 定期保存
        if epoch % self.config.save_freq == 0:
            epoch_path = os.path.join(self.config.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
            print(f"  ✓ 保存检查点: {epoch_path}")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 50)
        print("开始训练 - 实验4")
        print("=" * 50)
        print(self.config)
        
        best_val_acc = 0.0
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_loss, train_acc, loss_items = self.train_epoch(epoch)
            
            print(f"\n训练结果:")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  Acc: {train_acc:.4f}")
            print(f"  分项损失:")
            for key, val in loss_items.items():
                print(f"    {key}: {val:.4f}")
            
            # 验证
            if epoch % self.config.eval_freq == 0:
                val_loss, val_acc_text, val_acc_img = self.validate(epoch)
                
                print(f"\n验证结果:")
                print(f"  Loss: {val_loss:.4f}")
                print(f"  Acc (text): {val_acc_text:.4f}")
                print(f"  Acc (img): {val_acc_img:.4f}")
                
                # 记录历史
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc_text)
                
                # 保存最佳
                is_best = val_acc_text > best_val_acc
                if is_best:
                    best_val_acc = val_acc_text
                    print(f"  ✓ 新的最佳验证准确率: {best_val_acc:.4f}")
                
                self.save_checkpoint(epoch, is_best)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n学习率: {current_lr:.6f}")
        
        print("\n" + "=" * 50)
        print("训练完成！")
        print("=" * 50)
        print(f"最佳验证准确率: {best_val_acc:.4f}")
        
        # 保存训练历史
        history_path = os.path.join(self.config.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"训练历史保存至: {history_path}")


def main():
    """主函数"""
    # 获取配置
    config = get_config()
    
    # 创建训练器
    trainer = Experiment4Trainer(config)
    
    # 训练
    trainer.train()


if __name__ == "__main__":
    main()


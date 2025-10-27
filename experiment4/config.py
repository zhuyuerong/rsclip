# -*- coding: utf-8 -*-
"""
实验4配置文件
Surgery + 文本引导稀疏分解 + 规则去噪
"""

import torch


class Config:
    """实验4核心配置"""
    
    # ===== 模型配置 =====
    backbone = "ViT-B/32"  # 匹配RemoteCLIP-ViT-B-32.pt
    embed_dim = 512  # CLIP投影后的特征维度（768->512投影）
    n_patches = 49  # 7x7 for 224x224 images with patch_size=32
    n_components = 20  # 原子模式数量
    sparsity_ratio = 0.1  # 稀疏度（90%为0）
    
    # ===== 训练配置 =====
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-2
    epochs = 50
    warmup_epochs = 5
    
    # 学习率调度
    lr_scheduler = "cosine"
    min_lr = 1e-6
    
    # ===== 损失权重 =====
    w_cls = 1.0  # 分类损失
    w_loc = 1.0  # 定位损失
    w_sparse = 0.1  # 稀疏性损失
    w_ortho = 0.05  # 正交性损失
    w_align = 0.3  # text-img对齐损失
    
    # ===== WordNet配置 =====
    wordnet_k = 20  # 每个类别取20个相关词
    
    # 背景词表（用于去噪）
    background_words = [
        "background", "texture", "plain surface", "blur",
        "sky", "ground", "wall", "floor", "shadow",
        "empty space", "void", "nothing", "blank",
        "uniform color", "solid color"
    ]
    
    # ===== 数据集配置 =====
    # Mini dataset配置
    dataset_root = "datasets/mini_dataset"
    seen_classes = 15  # mini数据集的seen类数量
    unseen_classes = 5  # mini数据集的unseen类数量
    
    # 图像配置
    image_size = 224
    normalize_mean = [0.48145466, 0.4578275, 0.40821073]  # CLIP标准化
    normalize_std = [0.26862954, 0.26130258, 0.27577711]
    
    # ===== 去噪配置 =====
    # 背景阈值（分位数）
    bg_threshold_quantile = 0.7
    
    # 低通滤波kernel大小
    lowpass_kernel_size = 3
    
    # 异常值检测（MAD倍数）
    outlier_mad_multiplier = 3.0
    
    # ===== 稀疏分解配置 =====
    # Cross attention配置
    cross_attn_heads = 8
    cross_attn_dropout = 0.1
    
    # 稀疏化策略
    sparsity_method = "topk"  # "topk" or "threshold"
    sparsity_schedule = "constant"  # "constant" or "increasing"
    
    # ===== 设备配置 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    pin_memory = True
    
    # ===== 保存配置 =====
    checkpoint_dir = "experiment4/checkpoints"
    output_dir = "experiment4/outputs"
    log_dir = "experiment4/logs"
    
    # 保存频率
    save_freq = 5  # 每5个epoch保存一次
    eval_freq = 1  # 每个epoch评估一次
    
    # ===== 评估配置 =====
    eval_batch_size = 64
    
    # Top-k accuracy
    topk_values = [1, 3, 5]
    
    # 可视化配置
    visualize_num_samples = 10
    visualize_freq = 10  # 每10个epoch可视化一次
    
    # ===== Zero-shot配置 =====
    # Unseen类推理时使用image-only分解器
    use_img_decomposer_for_unseen = True
    
    # 是否使用集成（text + img两条路径）
    use_ensemble = False
    ensemble_weight_text = 0.7
    ensemble_weight_img = 0.3
    
    def __repr__(self):
        """打印配置信息"""
        config_str = "=" * 50 + "\n"
        config_str += "实验4配置\n"
        config_str += "=" * 50 + "\n"
        config_str += f"主干网络: {self.backbone}\n"
        config_str += f"嵌入维度: {self.embed_dim}\n"
        config_str += f"原子模式数: {self.n_components}\n"
        config_str += f"稀疏度: {self.sparsity_ratio}\n"
        config_str += f"Batch大小: {self.batch_size}\n"
        config_str += f"学习率: {self.learning_rate}\n"
        config_str += f"训练轮数: {self.epochs}\n"
        config_str += f"Seen类数: {self.seen_classes}\n"
        config_str += f"Unseen类数: {self.unseen_classes}\n"
        config_str += f"设备: {self.device}\n"
        config_str += "=" * 50 + "\n"
        return config_str


def get_config():
    """获取默认配置"""
    return Config()


if __name__ == "__main__":
    # 测试配置
    config = get_config()
    print(config)


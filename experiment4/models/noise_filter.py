# -*- coding: utf-8 -*-
"""
规则去噪模块
去除冗余、背景、结构化噪声
"""

import torch
import torch.nn.functional as F


class RuleBasedDenoiser:
    """
    规则去噪：去除冗余、背景、结构化噪声
    
    基于Surgery思想和信号处理方法
    """
    
    def __init__(self, bg_text_features, config):
        """
        Args:
            bg_text_features: [N_bg, 512] 背景词表的CLIP特征
            config: 配置对象
        """
        # 如果embed_dim != 512，需要投影
        if config.embed_dim != 512:
            # 使用简单的线性插值投影
            import torch.nn.functional as F_func
            self.bg_features = F_func.interpolate(
                bg_text_features.unsqueeze(0).unsqueeze(0),  # [1, 1, N_bg, 512]
                size=(bg_text_features.shape[0], config.embed_dim),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # [N_bg, embed_dim]
        else:
            self.bg_features = bg_text_features
            
        self.config = config
        
        # 去噪参数
        self.bg_threshold_quantile = config.bg_threshold_quantile
        self.lowpass_kernel_size = config.lowpass_kernel_size
        self.outlier_mad_multiplier = config.outlier_mad_multiplier
    
    def __call__(self, F_img):
        """
        执行规则去噪
        
        Args:
            F_img: [B, 196, 512] Surgery V-V attention特征
        
        Returns:
            F_clean: [B, 196, 512] 去噪后的特征
            denoise_info: dict, 去噪过程信息
        """
        B, N, D = F_img.shape
        
        denoise_info = {}
        
        # ===== 步骤1: 去冗余噪音（Surgery思想） =====
        # 所有patch共享的部分 = 冗余
        redundant = F_img.mean(dim=1, keepdim=True)  # [B, 1, 512]
        F_step1 = F_img - redundant
        denoise_info['redundant'] = redundant
        
        # ===== 步骤2: 去背景噪音 =====
        F_step2, bg_info = self._remove_background_noise(F_step1)
        denoise_info.update(bg_info)
        
        # ===== 步骤3: 去结构化噪音（位置编码泄露） =====
        F_step3 = self._remove_structural_noise(F_step2)
        
        # ===== 步骤4: 鲁棒性增强（去异常值） =====
        F_clean = self._remove_outliers(F_step3)
        
        # 记录去噪统计
        denoise_info['noise_reduction_ratio'] = self._compute_noise_reduction(F_img, F_clean)
        
        return F_clean, denoise_info
    
    def _remove_background_noise(self, F):
        """
        去除背景噪音
        
        Args:
            F: [B, 196, 512]
        
        Returns:
            F_clean: [B, 196, 512]
            info: dict
        """
        B, N, D = F.shape
        
        # 确保float类型（用于quantile等操作）
        F = F.float()
        bg_features = self.bg_features.float()
        
        # 归一化特征
        F_norm = torch.nn.functional.normalize(F, dim=-1)
        bg_norm = torch.nn.functional.normalize(bg_features, dim=-1)
        
        # 计算每个patch对背景词的响应
        bg_scores = F_norm @ bg_norm.T  # [B, 196, N_bg]
        bg_score_avg = bg_scores.mean(dim=-1).float()  # [B, 196]，确保float类型
        
        # 背景分数的阈值（动态计算）
        bg_threshold = torch.quantile(
            bg_score_avg.float(), 
            self.bg_threshold_quantile, 
            dim=1, 
            keepdim=True
        )  # [B, 1]
        
        # 前景mask
        fg_mask = (bg_score_avg < bg_threshold).float().unsqueeze(-1)  # [B, 196, 1]
        
        # 软抑制背景（不是完全去除，保留10%）
        F_clean = F * (0.1 + 0.9 * fg_mask)
        
        info = {
            'fg_mask': fg_mask,
            'bg_scores': bg_score_avg.float(),
            'bg_threshold': bg_threshold,
            'fg_ratio': fg_mask.mean().item()
        }
        
        return F_clean, info
    
    def _remove_structural_noise(self, F):
        """
        去除结构化噪音（位置编码泄露）
        
        使用低通滤波去除高频噪声
        
        Args:
            F: [B, N, D]
        
        Returns:
            F_clean: [B, N, D]
        """
        B, N, D = F.shape
        
        # 计算grid大小 (N = H*W, 需要是完全平方数)
        import math
        grid_size = int(math.sqrt(N))
        if grid_size * grid_size != N:
            # 如果不是完全平方数，跳过这一步
            return F
        
        # Reshape到2D空间
        F_2d = F.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # 低通滤波去除高频噪声
        kernel_size = self.lowpass_kernel_size
        padding = kernel_size // 2
        
        F_filtered = torch.nn.functional.avg_pool2d(
            F_2d, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )  # [B, D, H, W]
        
        # Reshape回1D
        F_clean = F_filtered.permute(0, 2, 3, 1).reshape(B, N, D)
        
        return F_clean
    
    def _remove_outliers(self, F):
        """
        去除异常值
        
        使用MAD (Median Absolute Deviation)方法
        
        Args:
            F: [B, 196, 512]
        
        Returns:
            F_clean: [B, 196, 512]
        """
        B, N, D = F.shape
        
        # 计算中位数
        median = F.median(dim=1, keepdim=True)[0]  # [B, 1, 512]
        
        # 计算MAD
        mad = (F - median).abs().median(dim=1, keepdim=True)[0]  # [B, 1, 512]
        
        # 3-sigma规则（使用MAD代替std）
        outlier_threshold = self.outlier_mad_multiplier * (mad + 1e-6)
        outlier_mask = (F - median).abs() < outlier_threshold
        
        # 应用mask
        F_clean = F * outlier_mask.float()
        
        return F_clean
    
    def _compute_noise_reduction(self, F_original, F_clean):
        """
        计算噪声降低比例
        
        Args:
            F_original: [B, 196, 512]
            F_clean: [B, 196, 512]
        
        Returns:
            ratio: float
        """
        noise = F_original - F_clean
        noise_power = (noise ** 2).mean().item()
        signal_power = (F_original ** 2).mean().item()
        
        if signal_power > 0:
            ratio = noise_power / signal_power
        else:
            ratio = 0.0
        
        return ratio


def test_denoiser():
    """测试去噪器"""
    print("测试规则去噪器...")
    
    # 模拟配置
    class DummyConfig:
        bg_threshold_quantile = 0.7
        lowpass_kernel_size = 3
        outlier_mad_multiplier = 3.0
    
    config = DummyConfig()
    
    # 模拟背景特征
    bg_features = torch.randn(10, 512)
    
    # 创建去噪器
    denoiser = RuleBasedDenoiser(bg_features, config)
    
    # 模拟输入
    F_img = torch.randn(4, 196, 512)
    
    # 去噪
    F_clean, info = denoiser(F_img)
    
    print(f"输入形状: {F_img.shape}")
    print(f"输出形状: {F_clean.shape}")
    print(f"前景比例: {info['fg_ratio']:.4f}")
    print(f"噪声降低比例: {info['noise_reduction_ratio']:.4f}")
    print("测试通过！")


if __name__ == "__main__":
    test_denoiser()


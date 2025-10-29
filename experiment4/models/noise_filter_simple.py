"""简化版去噪器 - 只保留Surgery去冗余"""
import torch
import torch.nn as nn
from typing import Tuple, Dict

class SimplifiedDenoiser(nn.Module):
    """只做Surgery去冗余: F̃ = F - mean(F)"""
    
    def __init__(self, bg_features=None, config=None):
        super().__init__()
        # 不使用bg_features和config，但需要接受参数以保持接口兼容
    
    def __call__(self, F: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if F.dtype == torch.float16:
            F = F.float()
        
        # Surgery去冗余（唯一操作）
        redundant = F.mean(dim=1, keepdim=True)
        F_clean = F - redundant
        
        info = {
            'fg_ratio': 1.0,
            'noise_reduction_ratio': 0.0,
            'redundant_norm': redundant.norm().item(),
        }
        return F_clean, info


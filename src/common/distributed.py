"""
分布式训练和随机种子管理
"""
import random
import numpy as np
import torch
import torch.distributed as dist
import os
from typing import Optional


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed(
    backend: str = 'nccl',
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None
) -> tuple:
    """
    初始化分布式训练
    
    Args:
        backend: 后端 ('nccl' for GPU, 'gloo' for CPU)
        init_method: 初始化方法（如 'env://'）
        rank: 进程rank（如果为None则从环境变量读取）
        world_size: 总进程数（如果为None则从环境变量读取）
    
    Returns:
        (rank, world_size) 元组
    """
    if init_method is None:
        init_method = 'env://'
    
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    
    return rank, world_size


def is_distributed() -> bool:
    """
    检查是否在分布式模式下运行
    
    Returns:
        是否分布式
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """
    获取当前进程的rank
    
    Returns:
        rank值，如果不是分布式则返回0
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    获取总进程数
    
    Returns:
        world_size值，如果不是分布式则返回1
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


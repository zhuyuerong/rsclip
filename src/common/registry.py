"""
简单的注册器系统
用于注册预处理、后处理、模型等组件
"""
from typing import Dict, Callable, Any, Optional


class Registry:
    """
    通用注册器
    
    用法:
        # 创建注册器
        PREPROCESS_REGISTRY = Registry('preprocess')
        
        # 注册组件
        @PREPROCESS_REGISTRY.register('resize')
        def resize_func(image, size):
            return cv2.resize(image, size)
        
        # 获取组件
        func = PREPROCESS_REGISTRY.get('resize')
    """
    
    def __init__(self, name: str):
        """
        初始化注册器
        
        Args:
            name: 注册器名称
        """
        self.name = name
        self._registry: Dict[str, Callable] = {}
    
    def register(self, name: Optional[str] = None):
        """
        注册装饰器
        
        Args:
            name: 注册名称，如果为None则使用函数名
        
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            key = name if name is not None else func.__name__
            if key in self._registry:
                raise ValueError(f"{key} already registered in {self.name}")
            self._registry[key] = func
            return func
        return decorator
    
    def get(self, name: str) -> Callable:
        """
        获取注册的组件
        
        Args:
            name: 组件名称
        
        Returns:
            注册的函数或类
        
        Raises:
            KeyError: 如果组件未注册
        """
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self.name}. Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def list(self) -> list:
        """
        列出所有已注册的组件名称
        
        Returns:
            组件名称列表
        """
        return list(self._registry.keys())
    
    def has(self, name: str) -> bool:
        """
        检查组件是否已注册
        
        Args:
            name: 组件名称
        
        Returns:
            是否已注册
        """
        return name in self._registry


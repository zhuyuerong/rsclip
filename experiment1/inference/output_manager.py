#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出管理器
统一管理所有生成文件的输出路径
"""

import os
from datetime import datetime
from typing import Optional


class OutputManager:
    """输出管理器"""
    
    def __init__(self, base_dir: str = "extensions/outputs"):
        """
        初始化输出管理器
        
        参数:
            base_dir: 基础输出目录
        """
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录结构
        self.dirs = {
            'detection_results': os.path.join(base_dir, 'detection_results'),
            'visualizations': os.path.join(base_dir, 'visualizations'),
            'test_images': os.path.join(base_dir, 'test_images'),
            'notebooks': os.path.join(base_dir, 'notebooks'),
            'logs': os.path.join(base_dir, 'logs'),
            'temp': os.path.join(base_dir, 'temp'),
        }
        
        # 创建所有目录
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def get_detection_result_path(self, target_class: str, model_name: str = None, suffix: str = "") -> str:
        """
        获取检测结果文件路径
        
        参数:
            target_class: 目标类别
            model_name: 模型名称（可选）
            suffix: 后缀（可选）
        
        返回:
            文件路径
        """
        base_name = target_class.replace(' ', '_').replace('/', '_')
        if model_name:
            base_name += f"_{model_name}"
        if suffix:
            base_name += f"_{suffix}"
        
        filename = f"{base_name}_detection_result.jpg"
        return os.path.join(self.dirs['detection_results'], filename)
    
    def get_visualization_path(self, strategy: str, image_name: str = "airport") -> str:
        """
        获取可视化文件路径
        
        参数:
            strategy: 采样策略
            image_name: 图像名称
        
        返回:
            文件路径
        """
        filename = f"{image_name}_{strategy}_visualization.jpg"
        return os.path.join(self.dirs['visualizations'], filename)
    
    def get_test_image_path(self, test_name: str, extension: str = "jpg") -> str:
        """
        获取测试图像路径
        
        参数:
            test_name: 测试名称
            extension: 文件扩展名
        
        返回:
            文件路径
        """
        filename = f"{test_name}_test.{extension}"
        return os.path.join(self.dirs['test_images'], filename)
    
    def get_log_path(self, log_name: str) -> str:
        """
        获取日志文件路径
        
        参数:
            log_name: 日志名称
        
        返回:
            文件路径
        """
        filename = f"{log_name}_{self.timestamp}.log"
        return os.path.join(self.dirs['logs'], filename)
    
    def get_temp_path(self, temp_name: str, extension: str = "tmp") -> str:
        """
        获取临时文件路径
        
        参数:
            temp_name: 临时文件名称
            extension: 文件扩展名
        
        返回:
            文件路径
        """
        filename = f"{temp_name}_{self.timestamp}.{extension}"
        return os.path.join(self.dirs['temp'], filename)
    
    def create_subdir(self, subdir_name: str) -> str:
        """
        创建子目录
        
        参数:
            subdir_name: 子目录名称
        
        返回:
            子目录路径
        """
        subdir_path = os.path.join(self.base_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        return subdir_path
    
    def list_outputs(self, subdir: Optional[str] = None) -> dict:
        """
        列出输出文件
        
        参数:
            subdir: 子目录名称（可选）
        
        返回:
            文件列表字典
        """
        if subdir:
            target_dir = os.path.join(self.base_dir, subdir)
            if os.path.exists(target_dir):
                return {subdir: os.listdir(target_dir)}
            else:
                return {subdir: []}
        else:
            outputs = {}
            for dir_name, dir_path in self.dirs.items():
                if os.path.exists(dir_path):
                    outputs[dir_name] = os.listdir(dir_path)
                else:
                    outputs[dir_name] = []
            return outputs
    
    def clean_temp_files(self):
        """清理临时文件"""
        temp_dir = self.dirs['temp']
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    os.remove(file_path)
                except:
                    pass


# 全局输出管理器实例
output_manager = OutputManager()


def get_output_manager() -> OutputManager:
    """获取全局输出管理器实例"""
    return output_manager


if __name__ == "__main__":
    # 测试输出管理器
    om = OutputManager()
    
    print("输出目录结构:")
    for name, path in om.dirs.items():
        print(f"  {name}: {path}")
    
    print("\n示例文件路径:")
    print(f"检测结果: {om.get_detection_result_path('ship', 'RN50')}")
    print(f"可视化: {om.get_visualization_path('pyramid', 'airport')}")
    print(f"测试图像: {om.get_test_image_path('bbox_refinement')}")
    
    print("\n当前输出文件:")
    outputs = om.list_outputs()
    for dir_name, files in outputs.items():
        print(f"  {dir_name}: {len(files)} 个文件")

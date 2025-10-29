# -*- coding: utf-8 -*-
"""
3种模式的实验配置
"""

import sys
from pathlib import Path

# 添加项目根目录
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

from experiment4.core.config import Config


class ExperimentConfig:
    """实验配置管理"""
    
    @staticmethod
    def get_mode_configs():
        """
        获取3种模式的配置
        
        Returns:
            dict: {mode_key: config_dict}
        """
        return {
            'mode1_baseline': {
                'name': '标准RemoteCLIP',
                'use_surgery': False,
                'use_vv_mechanism': False,
                'description': '直接使用RemoteCLIP特征，无Surgery，无VV'
            },
            'mode2_surgery': {
                'name': 'Surgery去冗余',
                'use_surgery': True,
                'use_vv_mechanism': False,
                'description': '使用Feature Surgery去冗余，无VV机制'
            },
            'mode3_surgery_vv': {
                'name': 'Surgery+VV机制',
                'use_surgery': True,
                'use_vv_mechanism': True,
                'num_vv_blocks': 6,
                'description': '使用Feature Surgery + VV自注意力机制'
            }
        }
    
    @staticmethod
    def get_config_for_mode(mode_key, base_config=None):
        """
        为指定模式创建配置对象
        
        Args:
            mode_key: 模式键（'mode1_baseline', 'mode2_surgery', 'mode3_surgery_vv'）
            base_config: 基础配置对象（可选）
        
        Returns:
            Config: 配置对象
        """
        if base_config is None:
            config = Config()
        else:
            config = base_config
        
        mode_configs = ExperimentConfig.get_mode_configs()
        if mode_key not in mode_configs:
            raise ValueError(f"未知模式: {mode_key}, 可选: {list(mode_configs.keys())}")
        
        mode_config = mode_configs[mode_key]
        
        # 更新配置
        config.use_surgery = mode_config['use_surgery']
        config.use_vv_mechanism = mode_config['use_vv_mechanism']
        if 'num_vv_blocks' in mode_config:
            config.num_vv_blocks = mode_config['num_vv_blocks']
        
        return config
    
    @staticmethod
    def print_all_configs():
        """打印所有模式配置"""
        print("="*60)
        print("实验模式配置")
        print("="*60)
        
        mode_configs = ExperimentConfig.get_mode_configs()
        for mode_key, mode_config in mode_configs.items():
            print(f"\n{mode_key}:")
            print(f"  名称: {mode_config['name']}")
            print(f"  Surgery去冗余: {mode_config['use_surgery']}")
            print(f"  VV机制: {mode_config['use_vv_mechanism']}")
            if 'num_vv_blocks' in mode_config:
                print(f"  VV层数: {mode_config['num_vv_blocks']}")
            print(f"  描述: {mode_config['description']}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # 测试配置
    ExperimentConfig.print_all_configs()
    
    # 测试创建配置对象
    print("\n测试创建配置对象:")
    for mode_key in ['mode1_baseline', 'mode2_surgery', 'mode3_surgery_vv']:
        config = ExperimentConfig.get_config_for_mode(mode_key)
        print(f"\n{mode_key}:")
        print(f"  use_surgery={config.use_surgery}")
        print(f"  use_vv_mechanism={config.use_vv_mechanism}")
        print(f"  num_vv_blocks={config.num_vv_blocks}")


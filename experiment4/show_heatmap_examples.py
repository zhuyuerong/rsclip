# -*- coding: utf-8 -*-
"""
展示热图示例
读取生成的PNG图片并创建对比总览
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_comparison_grid(output_dir, sample_idx=0, save_path=None):
    """
    创建四种方法的对比网格图
    
    Args:
        output_dir: 输出目录
        sample_idx: 样本索引（0-9）
        save_path: 保存路径（可选）
    """
    methods = ['standard', 'vv_qk', 'vv_vv', 'vv_mixed']
    method_names = ['标准Surgery', 'VV-QK路径', 'VV-VV路径', 'VV-混合路径']
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle(f'CLIP Surgery 热图对比（样本{sample_idx}）', fontsize=16, y=0.995)
    
    for i, (method, name) in enumerate(zip(methods, method_names)):
        img_path = output_dir / method / f"sample_{sample_idx:03d}.png"
        
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            axes[i].imshow(img)
            axes[i].set_title(name, fontsize=14)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"未找到: {img_path}", 
                        ha='center', va='center', fontsize=12)
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 对比图已保存: {save_path}")
    else:
        plt.show()
    
    return fig


def main():
    """主函数"""
    print("="*70)
    print("热图示例可视化")
    print("="*70)
    
    output_dir = Path("experiment4/outputs/heatmap_evaluation")
    
    if not output_dir.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        print("请先运行: python experiment4/run_heatmap_evaluation.py")
        return
    
    # 检查可用的样本
    standard_dir = output_dir / "standard"
    if not standard_dir.exists():
        print(f"❌ 标准Surgery结果不存在")
        return
    
    png_files = list(standard_dir.glob("sample_*.png"))
    num_samples = len(png_files)
    
    print(f"\n发现 {num_samples} 个样本的可视化结果")
    
    # 为前3个样本生成对比图
    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(exist_ok=True)
    
    for idx in range(min(3, num_samples)):
        print(f"\n生成样本{idx}的对比图...")
        save_path = comparison_dir / f"comparison_{idx:03d}.png"
        create_comparison_grid(output_dir, sample_idx=idx, save_path=save_path)
    
    print(f"\n" + "="*70)
    print(f"✅ 对比图已保存至: {comparison_dir}")
    print(f"   共 {min(3, num_samples)} 张对比图")
    print("="*70)
    
    # 打印文件列表
    print(f"\n生成的文件:")
    for f in sorted(comparison_dir.glob("*.png")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.2f} MB")
    
    print(f"\n💡 提示:")
    print(f"  打开图片查看四种方法的热图对比")
    print(f"  每张图包含4行，分别对应4种方法")


if __name__ == "__main__":
    main()


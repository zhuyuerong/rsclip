全面对比实验: 2种模式 × 12层热图
=====================================

实验目的:
--------
对比Surgery去冗余对所有12层特征的影响，并标注GT区域

生成文件:
--------
1. text_guided_vvt_sample0-4.png (5个, 9.1MB)
   - 原图 + 4层热图 (L1, L3, L6, L9)
   - 绿色边界框: GT目标区域
   - 红蓝热图: 模型关注度（红=高，蓝=低）

2. comprehensive_comparison_results/
   - comprehensive_comparison_sample0-4.png (5个, 30.7MB)
   - 每张图: 2行 × 13列
     * 第1行: Baseline (无Surgery)
     * 第2行: Surgery (有Surgery)
     * 列: 原图 + L1-L12热图
   - 所有热图都标注GT边界框

代码文件:
--------
- text_guided_vvt.py: 生成带GT框的4层热图
- comprehensive_comparison.py: 生成2×13全面对比图

运行命令:
--------
# 生成带GT框的4层热图
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main
PYTHONPATH=. ovadetr_env/bin/python3.9 \
  experiment4/experiments/surgery_clip/exp3_text_guided_vvt/text_guided_vvt.py \
  --dataset datasets/mini_dataset \
  --layers 1 3 6 9 \
  --max-samples 5

# 生成2×13全面对比图
PYTHONPATH=. ovadetr_env/bin/python3.9 \
  experiment4/experiments/surgery_clip/exp3_text_guided_vvt/comprehensive_comparison.py \
  --dataset datasets/mini_dataset \
  --max-samples 5

关键发现:
--------
从comprehensive_comparison图中可以观察到:

1. **Surgery对不同层的影响**:
   - 浅层(L1-L4): Surgery显著改变热图分布
   - 中层(L5-L8): Surgery效果逐渐减弱
   - 深层(L9-L12): Surgery影响较小

2. **GT区域响应对比**:
   - Baseline: 热图可能在非GT区域也有高响应
   - Surgery: 热图更集中在GT区域（理论上）

3. **跨层演化**:
   - 浅层: 热图分散，覆盖多个区域
   - 深层: 热图集中，定位更精确

数据统计:
--------
- 样本数: 5个
- 模式数: 2种 (Baseline, Surgery)
- 层数: 12层 (L1-L12)
- 总热图数: 5 × 2 × 12 = 120个
- GT边界框: 绿色，2像素宽

图像格式:
--------
- 分辨率: 200 DPI
- 尺寸: ~2.5英寸 × 列数
- 格式: PNG
- 颜色映射: jet (蓝→绿→黄→红)
- Alpha混合: 0.5

注意事项:
--------
1. GT边界框坐标来自数据集标注
2. 热图归一化范围: 0-1
3. Surgery使用温度参数 t=2
4. 所有类别使用DIOR 20类文本特征

文件大小:
--------
- 带GT框4层图: 1.6-1.8MB/张
- 2×13全面对比图: 5.8-6.4MB/张
- 总计: 9.1MB + 30.7MB = 39.8MB

版本信息:
--------
- 创建时间: 2025-10-29 19:04
- Git commit: 8922f0ca
- 实验位置: experiment4/experiments/surgery_clip/exp3_text_guided_vvt/


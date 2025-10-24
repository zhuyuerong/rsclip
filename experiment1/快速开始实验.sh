#!/bin/bash
# -*- coding: utf-8 -*-
"""
RemoteCLIP 实验快速开始脚本
"""

echo "=" * 70
echo "RemoteCLIP 实验快速开始"
echo "=" * 70

# 检查Python环境
echo "🔍 检查Python环境..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3未安装或不在PATH中"
    exit 1
fi

# 检查必要的Python包
echo "🔍 检查必要的Python包..."
python3 -c "import torch, open_clip, cv2, numpy, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少必要的Python包，请安装：torch, open_clip, opencv-python, numpy, Pillow"
    exit 1
fi

echo "✅ Python环境检查通过"

# 检查测试图像
echo "🔍 检查测试图像..."
if [ ! -f "assets/airport.jpg" ]; then
    echo "❌ 测试图像不存在: assets/airport.jpg"
    echo "请确保测试图像存在"
    exit 1
fi

echo "✅ 测试图像检查通过"

# 检查模型权重
echo "🔍 检查模型权重..."
if [ ! -f "checkpoints/RemoteCLIP-RN50.pt" ]; then
    echo "❌ 模型权重不存在: checkpoints/RemoteCLIP-RN50.pt"
    echo "请确保模型权重文件存在"
    exit 1
fi

echo "✅ 模型权重检查通过"

# 显示可用选项
echo ""
echo "🚀 可用的实验选项："
echo "1. 运行完整流水线"
echo "2. 运行Stage1（数据预处理和候选框生成）"
echo "3. 运行Stage2（目标检测和优化）"
echo "4. 运行单个模块测试"
echo "5. 运行实验脚本"
echo "6. 查看文件结构"
echo ""

# 获取用户选择
read -p "请选择要运行的实验 (1-6): " choice

case $choice in
    1)
        echo "🚀 运行完整流水线..."
        python3 experiment1/inference/inference_engine.py --image assets/airport.jpg --pipeline full
        ;;
    2)
        echo "🚀 运行Stage1流水线..."
        python3 experiment1/inference/inference_engine.py --image assets/airport.jpg --pipeline stage1
        ;;
    3)
        echo "🚀 运行Stage2流水线..."
        python3 experiment1/inference/inference_engine.py --image assets/airport.jpg --pipeline stage2
        ;;
    4)
        echo "🚀 运行单个模块测试..."
        echo "可用的模块："
        echo "- sampling: 区域采样"
        echo "- target_detection: 目标检测"
        echo "- wordnet_enhancement: WordNet增强"
        echo "- bbox_refinement: 边界框微调"
        echo ""
        read -p "请输入模块名称: " module
        python3 experiment1/inference/inference_engine.py --image assets/airport.jpg --module $module
        ;;
    5)
        echo "🚀 运行实验脚本..."
        echo "可用的脚本："
        echo "- test_bbox_refinement.py: 边界框微调测试"
        echo "- test_file_organization.py: 文件组织测试"
        echo "- test_remoteclip.py: RemoteCLIP测试"
        echo ""
        read -p "请输入脚本名称: " script
        python3 experiment1/scripts/experiments/$script
        ;;
    6)
        echo "📁 文件结构："
        echo ""
        echo "experiment1/"
        echo "├── stage1/             # 数据预处理和候选框生成"
        echo "├── stage2/             # 目标检测和优化"
        echo "├── inference/          # 推理模块"
        echo "├── scripts/experiments/# 实验脚本"
        echo "├── outputs/            # 输出文件"
        echo "└── docs/               # 文档说明"
        echo ""
        echo "详细说明请查看：experiment1/docs/实验结构说明.md"
        ;;
    *)
        echo "❌ 无效选择，请重新运行脚本"
        exit 1
        ;;
esac

echo ""
echo "✅ 实验完成！"
echo "📖 更多信息请查看：experiment1/docs/实验结构说明.md"

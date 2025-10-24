#!/bin/bash
# RemoteCLIP 实验启动脚本

set -e

echo "================================================================"
echo "RemoteCLIP 遥感目标检测实验平台"
echo "================================================================"

# 激活虚拟环境
echo ""
echo "🐍 激活remoteclip虚拟环境..."
if [ -f "remoteclip/bin/activate" ]; then
    source remoteclip/bin/activate
    echo "✅ remoteclip环境已激活"
    echo "   Python: $(which python)"
    echo "   版本: $(python --version)"
else
    echo "⚠️  未找到remoteclip虚拟环境"
fi

# 检查Python环境
echo ""
echo "🔍 检查环境..."
if ! python -c "import torch, open_clip, cv2, numpy, PIL" 2>/dev/null; then
    echo "❌ 缺少必要的Python包"
    echo "请运行: pip install torch open_clip_torch opencv-python numpy pillow scipy"
    exit 1
fi
echo "✅ 环境检查通过"

# 显示菜单
echo ""
echo "请选择实验："
echo "1. Experiment1 - 对比学习检测（完整实现）"
echo "2. Experiment2 - 全局上下文检测（核心创新）"
echo "3. 查看项目结构"
echo "4. 运行测试"
echo "0. 退出"
echo ""

read -p "请输入选项 (0-4): " choice

case $choice in
    1)
        echo ""
        echo "=== Experiment1 选项 ==="
        echo "1. 舰船检测"
        echo "2. 飞机检测"
        echo "3. 完整流水线"
        echo "4. 单个模块测试"
        read -p "请选择 (1-4): " exp1_choice
        
        case $exp1_choice in
            1)
                python3 experiment1/stage2/target_detection.py \
                    --image assets/ship.jpg --target ship --model RN50
                ;;
            2)
                python3 experiment1/stage2/target_detection.py \
                    --image assets/airport.jpg --target airplane --model RN50
                ;;
            3)
                python3 experiment1/inference/inference_engine.py \
                    --image assets/airport.jpg --pipeline full
                ;;
            4)
                echo "可用模块: sampling, target_detection, bbox_refinement"
                read -p "输入模块名: " module
                python3 experiment1/inference/inference_engine.py \
                    --image assets/airport.jpg --module $module
                ;;
        esac
        ;;
    
    2)
        echo ""
        echo "=== Experiment2 核心模块测试 ==="
        echo "1. 全局对比损失（核心创新）"
        echo "2. 上下文门控"
        echo "3. 查看配置"
        echo "4. 查看README"
        read -p "请选择 (1-4): " exp2_choice
        
        case $exp2_choice in
            1)
                python3 experiment2/stage4_supervision/global_contrast_loss.py
                ;;
            2)
                python3 experiment2/stage2_decoder/context_gating.py
                ;;
            3)
                python3 experiment2/config/default_config.py
                ;;
            4)
                cat experiment2/README.md | less
                ;;
        esac
        ;;
    
    3)
        echo ""
        echo "📁 项目结构："
        echo ""
        echo "experiment1/             # 实验1（完整）"
        echo "├── stage1/             # 数据预处理"
        echo "├── stage2/             # 目标检测"
        echo "├── inference/          # 推理引擎"
        echo "└── outputs/            # 输出文件"
        echo ""
        echo "experiment2/             # 实验2（核心创新）⭐"
        echo "├── config/             # 配置文件"
        echo "├── stage2_decoder/     # 上下文门控"
        echo "├── stage4_supervision/ # 全局对比损失"
        echo "└── README.md           # 详细说明"
        echo ""
        echo "详细文档："
        echo "- experiment1/README.md"
        echo "- experiment2/README.md"
        ;;
    
    4)
        echo ""
        echo "运行测试..."
        if [ -f experiment1/scripts/experiments/test_remoteclip.py ]; then
            python3 experiment1/scripts/experiments/test_remoteclip.py
        else
            echo "测试文件不存在"
        fi
        ;;
    
    0)
        echo "退出"
        exit 0
        ;;
    
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "✅ 完成！"
echo "================================================================"


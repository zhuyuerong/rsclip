#!/bin/bash

# 实验4快速启动脚本

echo "=========================================="
echo "实验4 快速启动"
echo "Surgery + 文本引导稀疏分解 + 规则去噪"
echo "=========================================="

# 进入项目根目录
cd "$(dirname "$0")/.." || exit 1

echo ""
echo "当前目录: $(pwd)"
echo ""

# 激活环境（如果存在）
if [ -f "activate_env.sh" ]; then
    echo "激活虚拟环境..."
    source activate_env.sh
fi

# 检查Python
echo "检查Python环境..."
python --version
echo ""

# 菜单
echo "请选择操作:"
echo "  1) 训练Seen类"
echo "  2) 评估Seen类"
echo "  3) Zero-shot评估Unseen类"
echo "  4) 运行Demo"
echo "  5) 测试所有模块"
echo "  0) 退出"
echo ""

read -p "请输入选项 [0-5]: " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "开始训练Seen类"
        echo "=========================================="
        python -m experiment4.train_seen
        ;;
    2)
        echo ""
        echo "=========================================="
        echo "评估Seen类"
        echo "=========================================="
        python -m experiment4.inference_seen
        ;;
    3)
        echo ""
        echo "=========================================="
        echo "Zero-shot评估Unseen类"
        echo "=========================================="
        python -m experiment4.inference_unseen
        ;;
    4)
        echo ""
        echo "=========================================="
        echo "运行Demo"
        echo "=========================================="
        
        # 查找测试图像
        if [ -f "assets/airport.jpg" ]; then
            python experiment4/demo.py assets/airport.jpg
        elif [ -f "datasets/mini_dataset/images/airplane_001.jpg" ]; then
            python experiment4/demo.py datasets/mini_dataset/images/airplane_001.jpg
        else
            echo "请提供图像路径："
            read -p "图像路径: " img_path
            python experiment4/demo.py "$img_path"
        fi
        ;;
    5)
        echo ""
        echo "=========================================="
        echo "测试所有模块"
        echo "=========================================="
        
        echo ""
        echo "[1/6] 测试配置..."
        python experiment4/config.py
        
        echo ""
        echo "[2/6] 测试CLIP Surgery..."
        python experiment4/models/clip_surgery.py
        
        echo ""
        echo "[3/6] 测试去噪器..."
        python experiment4/models/noise_filter.py
        
        echo ""
        echo "[4/6] 测试分解器..."
        python experiment4/models/decomposer.py
        
        echo ""
        echo "[5/6] 测试损失函数..."
        python experiment4/losses.py
        
        echo ""
        echo "[6/6] 测试数据集..."
        python experiment4/data/dataset.py
        
        echo ""
        echo "✓ 所有模块测试完成！"
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项: $choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="


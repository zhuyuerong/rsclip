#!/bin/bash
# 自动等待训练完成并评估

echo "🔍 等待训练完成..."
echo "  目标: 50 epochs"
echo ""

while true; do
    if [ -f "outputs/logs/correct_train_history.json" ]; then
        epochs=$(python3.9 -c "import json; f=open('outputs/logs/correct_train_history.json'); h=json.load(f); print(len(h))")
        
        if [ "$epochs" -ge 50 ]; then
            echo ""
            echo "✅ 训练完成！(50 epochs)"
            echo ""
            echo "开始评估..."
            /media/ubuntu22/新加卷1/anaconda_envs/ovadetr/bin/python3.9 evaluate_correct_version.py
            break
        else
            echo -ne "\r  当前进度: $epochs/50 epochs"
            sleep 10
        fi
    else
        echo -ne "\r  等待训练开始..."
        sleep 5
    fi
done


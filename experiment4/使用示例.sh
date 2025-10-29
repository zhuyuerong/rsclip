#!/bin/bash
# CLIP Surgery 热图评估使用示例

echo "========================================="
echo "CLIP Surgery 热图评估工具"
echo "========================================="

cd "/media/ubuntu22/新加卷1/Projects/RemoteCLIP-main"

# 1. 完整评估（生成四种热图，计算mAP@0.5）
echo -e "\n1. 运行完整评估..."
ovadetr_env/bin/python3.9 experiment4/run_heatmap_evaluation.py

# 2. 热图质量分析
echo -e "\n2. 分析热图质量..."
ovadetr_env/bin/python3.9 experiment4/analyze_heatmap_quality.py

# 3. 多IoU阈值验证
echo -e "\n3. 测试多个IoU阈值..."
ovadetr_env/bin/python3.9 experiment4/quick_verify_with_low_iou.py

echo -e "\n========================================="
echo "✅ 评估完成！"
echo "========================================="
echo "查看结果："
echo "  数值: experiment4/outputs/heatmap_evaluation/map_results.json"
echo "  报告: experiment4/outputs/heatmap_evaluation/evaluation_report.md"
echo "  可视化: experiment4/outputs/heatmap_evaluation/*/*.png"

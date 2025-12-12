#!/bin/bash

echo "======================================"
echo "Running All Tests"
echo "======================================"
echo ""

# 切换到exp4目录
cd "$(dirname "$0")/.." || exit 1

# Level 1: 单元测试
echo ">>> Level 1: Component Tests"
python tests/test_components.py
if [ $? -ne 0 ]; then
    echo "❌ Component tests failed!"
    exit 1
fi
echo ""

# Level 2: 集成测试
echo ">>> Level 2: Integration Tests"
python tests/test_integration.py
if [ $? -ne 0 ]; then
    echo "❌ Integration tests failed!"
    exit 1
fi
echo ""

# Level 3: 数据测试
echo ">>> Level 3: Dataset Tests"
python tests/test_dataset.py
echo ""

# Level 4: 快速训练测试
echo ">>> Level 4: Quick Training Test"
python tests/test_quick_train.py
if [ $? -ne 0 ]; then
    echo "❌ Training test failed!"
    exit 1
fi
echo ""

echo "======================================"
echo "✅ All Tests Passed!"
echo "======================================"



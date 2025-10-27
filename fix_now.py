#!/usr/bin/env python3
"""一键修复实验4的bug"""

# 读取文件
filepath = "experiment4/models/noise_filter.py"
with open(filepath, 'r') as f:
    content = f.read()

# 备份
with open(filepath + '.backup', 'w') as f:
    f.write(content)
print("✓ 已备份原文件")

# 修复1: quantile的float问题
content = content.replace(
    'bg_score_avg,',
    'bg_score_avg.float(),'
)
print("✓ 修复Bug 1: quantile dtype")

# 修复2: 动态grid大小
old_code = 'F_2d = F.reshape(B, 14, 14, D).permute(0, 3, 1, 2)  # [B, 512, 14, 14]'
new_code = '''# 动态计算grid大小
        grid_size = int(N ** 0.5)
        assert grid_size * grid_size == N, f"Patches={N}不是完全平方数"
        F_2d = F.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)'''

content = content.replace(old_code, new_code)
print("✓ 修复Bug 2: 动态reshape")

# 写回
with open(filepath, 'w') as f:
    f.write(content)

print("\n✅ 修复完成！\n")
print("现在运行: python experiment4/test_train_1epoch.py")


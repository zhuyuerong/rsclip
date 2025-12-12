#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行CAL实验并实时监控输出
"""
import sys
import os
import subprocess
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
os.chdir(project_root)

print("🚀 启动CAL实验（实时监控模式）")
print("=" * 80)

# 运行实验脚本
cmd = [
    sys.executable,
    str(project_root / "src/experiments/exp3_cal/run_final.py")
]

env = os.environ.copy()
env['PYTHONPATH'] = str(project_root) + ':' + env.get('PYTHONPATH', '')

# 使用subprocess实时输出
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=env,
    universal_newlines=True,
    bufsize=1
)

print(f"📊 进程ID: {process.pid}")
print("=" * 80)
print()

# 实时输出
error_count = 0
line_count = 0

try:
    for line in process.stdout:
        line_count += 1
        print(line, end='', flush=True)
        
        # 检测错误
        if '❌' in line or 'Error' in line or 'Exception' in line or 'Traceback' in line:
            error_count += 1
            print(f"\n⚠️  检测到错误 #{error_count}")
        
        # 每100行显示一次进度
        if line_count % 100 == 0:
            print(f"\n📊 已处理 {line_count} 行输出\n")
    
    # 等待进程结束
    return_code = process.wait()
    
    print("\n" + "=" * 80)
    print(f"✅ 实验完成")
    print(f"   返回码: {return_code}")
    print(f"   总输出行数: {line_count}")
    print(f"   检测到的错误数: {error_count}")
    print("=" * 80)
    
    if return_code != 0:
        print(f"\n❌ 实验异常退出（返回码: {return_code}）")
        sys.exit(return_code)
        
except KeyboardInterrupt:
    print("\n\n⚠️  用户中断，正在终止进程...")
    process.terminate()
    process.wait()
    sys.exit(1)







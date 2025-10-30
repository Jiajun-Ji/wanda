#!/usr/bin/env python3
"""
测试 lm_eval 导入是否成功
"""

import os
import sys

# 添加 lm-evaluation-harness 到 Python 路径
LM_EVAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lm-evaluation-harness')
print(f"LM_EVAL_PATH: {LM_EVAL_PATH}")
print(f"Path exists: {os.path.exists(LM_EVAL_PATH)}")

if os.path.exists(LM_EVAL_PATH) and LM_EVAL_PATH not in sys.path:
    sys.path.insert(0, LM_EVAL_PATH)
    print(f"Added to sys.path")

print(f"\nPython path:")
for p in sys.path[:5]:
    print(f"  {p}")

print(f"\nTrying to import lm_eval...")
try:
    import lm_eval
    print(f"✓ Success! lm_eval imported from: {lm_eval.__file__}")

    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    print(f"✓ Success! evaluator and TaskManager imported")

    # 创建 TaskManager 实例
    task_manager = TaskManager()
    print(f"✓ Success! TaskManager initialized")

    print(f"\nAvailable tasks (first 20):")
    all_tasks = task_manager.all_tasks[:20]
    for task in all_tasks:
        print(f"  - {task}")

    print(f"\nTotal tasks available: {len(task_manager.all_tasks)}")

    # 检查我们需要的任务是否存在
    required_tasks = ['boolq', 'rte', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
    print(f"\nChecking required tasks:")
    for task in required_tasks:
        if task in task_manager.all_tasks:
            print(f"  ✓ {task}")
        else:
            print(f"  ✗ {task} NOT FOUND")

except ImportError as e:
    print(f"✗ Failed to import lm_eval: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n✓ All imports successful!")


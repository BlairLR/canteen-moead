# -*- coding: utf-8 -*-
"""校园选址 MOEA/D 多目标优化 —— 主入口。

用法（在项目根目录 canteen-moead/ 下）：
  python main.py              # 默认：南湖 21 点快速实验（5×100 代 → output/moead_nanhu_fast/demo_5x100/）
  python main.py --baseline   # 旧版基线实验（虚拟校园，3 种子 × 200 代 → output/moead/）
  python main.py --nanhu      # 南湖 21 点完整实验（10 种子 × 200 代 → output/moead_nanhu/）
  python main.py --quick      # 南湖 1 种子 × 50 代快速验证（→ output/moead_nanhu_fast/quick_1x50/）
  python main.py --demo       # 南湖 5 种子 × 100 代快速实验（→ output/moead_nanhu_fast/demo_5x100/）
  python main.py --tests      # 只跑正确性自检 + 冒烟测试
  python main.py --plots      # 只重新出图（用已保存的实验结果）

输出目录：
  output/moead/                   旧版基线实验（虚拟校园，3 种子 × 200 代）
  output/moead_nanhu/             南湖 21 点完整实验（10 种子 × 200 代）
  output/moead_nanhu_fast/        demo_5x100/ + quick_1x50/

数据版本：
  - [真实数据 real] 21 点位坐标（4 竞争 + 10 宿舍 + 7 教学 + A3 边界多边形，WGS-84，上层收口）
  - [旧版虚拟校园] src/campus_model.py 内的硬编码坐标
"""

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _run_script(rel_path: str):
    """用子进程运行项目根目录下的指定脚本。"""
    py = sys.executable
    script = os.path.join(HERE, rel_path)
    print(f"\n>>> python {rel_path}")
    r = subprocess.run([py, script], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"{rel_path} 异常退出 (code={r.returncode})")


def run_tests():
    _run_script("tests/sanity_tests.py")
    _run_script("tests/smoke_test.py")


def run_baseline():
    _run_script("experiments/run_experiments.py")
    _run_script("visualization/moead_plots.py")


def run_nanhu_full():
    _run_script("experiments/run_nanhu.py")


def run_nanhu_quick():
    _run_script("experiments/run_nanhu_fast.py --quick")


def run_nanhu_demo():
    _run_script("experiments/run_nanhu_fast.py")


def run_plots():
    _run_script("visualization/moead_plots.py")


if __name__ == "__main__":
    args = set(sys.argv[1:])

    if "--tests" in args:
        run_tests()
        sys.exit(0)

    if "--plots" in args:
        run_plots()
        sys.exit(0)

    if "--baseline" in args:
        run_baseline()
    elif "--nanhu" in args:
        run_nanhu_full()
    elif "--quick" in args:
        run_nanhu_quick()
    elif "--demo" in args:
        run_nanhu_demo()
    else:
        # 默认：南湖 21 点快速实验（demo_5x100）
        run_nanhu_demo()
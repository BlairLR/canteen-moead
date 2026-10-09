# -*- coding: utf-8 -*-
"""校园选址 MOEA/D 多目标优化 —— 主入口。

直接在 PyCharm 中点击运行即可完成一次完整的优化实验并生成可视化。

流程:
  1. 阶段 0 采样（6000 点在 A3 多边形内）
  2. MOEA/D 优化（91 子问题 × 200 代 × 1 次运行 × seed 42）
  3. 结果保存至 output/moead_nanhu/
  4. 运行统一可视化（plot_all.py）生成 7 张图表至 output/

可选 --tests 参数运行正确性自检 + 多边形冒烟测试（不进行完整优化）。
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
    _run_script("tests/polygon_smoke_test.py")


if __name__ == "__main__":
    args = set(sys.argv[1:])

    if "--tests" in args:
        run_tests()
        sys.exit(0)

    print("=" * 55)
    print("▌ 学生城食堂 · MOEA/D 选址优化")
    print("▌ 配置: 91 子问题 × 200 代 × 1 次运行 × seed 42")
    print("=" * 55)

    _run_script("experiments/run_nanhu.py")
    _run_script("visualization/plot_all.py")
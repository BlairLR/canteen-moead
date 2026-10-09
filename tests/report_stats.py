# -*- coding: utf-8 -*-
"""报告补充统计（一次性脚本）。"""
import json

import numpy as np

r = json.load(open("output/moead/experiment_results.json", encoding="utf-8"))
runs = r["runs"]
print("ref_survivors:", [x["ref_survivors"] for x in runs])
print("mean ref_survivors:", np.mean([x["ref_survivors"] for x in runs]))
best = max(runs, key=lambda x: x["hv"])
print("best seed:", best["seed"], "hv:", best["hv"])
print("hv_trajectory:", [round(t[1], 4) for t in best["hv_trajectory"]])

Ax = np.load(f"output/moead/archive_seed{best['seed']}_X.npy")
edge = ((Ax[:, 0] < 100) | (Ax[:, 0] > 900) | (Ax[:, 1] < 70) | (Ax[:, 1] > 630)).mean()
print(f"best seed archive 边缘带比例: {edge:.1%}, "
      f"x范围[{Ax[:, 0].min():.0f},{Ax[:, 0].max():.0f}], "
      f"y范围[{Ax[:, 1].min():.0f},{Ax[:, 1].max():.0f}]")

# 决策空间二维直方图：解密度最高的区域
Hh, xe, ye = np.histogram2d(Ax[:, 0], Ax[:, 1], bins=[10, 7])
print("决策空间 10x7 网格解计数（行=y 700→0 自上而下打印）:")
for row in Hh.T[::-1]:
    print("  " + " ".join(f"{int(v):5d}" for v in row))

# -*- coding: utf-8 -*-
"""A3 接入冒烟测试：5 代 MOEA/D + 外层约束评价，验证评价流程正常且档案候选满足 A3 多边形约束。

约束机制（moead/ 核心零改动）：build_constrained_evaluate（罚值法）
  - 界外候选返回有界罚值 caps = 可行域各目标最大估计 × 1.1；
  - 界外解被任意可行解严格支配 → 外部档案保持干净（约束输出）；
  - 种群 pop_X 在外包矩形上做 SBX/变异，约 30% 为界外点（仅消耗评价预算，不进入档案）。

判定标准（适配罚值法）：
  - 罚值自检：界外点必须返回 caps（外层约束逻辑本身有效）；
  - 档案候选（约束输出）必须全部位于 A3 多边形内；
  - 种群界外比例为信息项（设计预期 < 100%，不计入 PASS/FAIL）。
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from moead import MOEAD, MOEADConfig
from src.nanhu_campus import build_constrained_evaluate, build_model, is_inside_polygon

model = build_model()
evaluate, caps = build_constrained_evaluate(model)
poly = model.polygon_local

# ---- 外层约束自检：界外点应返回罚值 caps ----
lo, hi = poly.min(axis=0), poly.max(axis=0)
out_pt = np.array([[hi[0] + 500.0, hi[1] + 500.0]])   # 明确界外（外包矩形之外）
F_out = evaluate(out_pt)
penalty_ok = bool(np.allclose(F_out[0], caps))
print(f"罚值自检: caps={np.round(caps, 2).tolist()}")
print(f"  界外点 {np.round(out_pt[0], 1).tolist()} → F={np.round(F_out[0], 2).tolist()} "
      f"{'✔ 等于罚值' if penalty_ok else '✗ 与罚值不符'}")

# ---- 5 代短程 MOEA/D（外层约束评价，无修复钩子）----
cfg = MOEADConfig(x_bounds=model.bounds_x, y_bounds=model.bounds_y, seed=99, max_gen=5)
res = MOEAD(cfg, evaluate).run()

ins_pop = is_inside_polygon(poly, res["pop_X"])
ins_arch = is_inside_polygon(poly, res["archive_X"])
print(f"5 代冒烟: 评价 {res['n_evals']} 次 | 替换 {res['total_replacements']} 次")
print(f"种群 {int(ins_pop.sum())}/{len(ins_pop)} 界内（界外候选仅消耗评价预算，不进入档案）")
print(f"档案 {int(ins_arch.sum())}/{len(ins_arch)} 界内（约束输出）")
print(f"决策变量范围 x({res['pop_X'][:, 0].min():.0f}, {res['pop_X'][:, 0].max():.0f}) "
      f"y({res['pop_X'][:, 1].min():.0f}, {res['pop_X'][:, 1].max():.0f})")

# ---- 约束判定：罚值有效 + 档案候选全部界内 ----
ok = bool(penalty_ok and len(ins_arch) > 0 and ins_arch.all())
print("PIPELINE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

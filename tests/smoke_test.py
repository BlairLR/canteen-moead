# -*- coding: utf-8 -*-
"""冒烟测试：小代数（max_gen=30）单次 MOEA/D 运行，验证主循环可跑通。

检查：评价次数、档案规模增长、无 NaN、替换发生、标量化输出有限。
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from moead import MOEAD, MOEADConfig
from src.campus_model import evaluate

cfg = MOEADConfig(max_gen=30, seed=42)
t0 = time.time()
res = MOEAD(cfg, evaluate).run()
dt = time.time() - t0

F = res["archive_F"]
print(f"冒烟测试 max_gen={cfg.max_gen}  耗时 {dt:.2f}s")
print(f"评价次数     : {res['n_evals']}（期望 91 + 91*30 = 2821）")
print(f"总替换次数   : {res['total_replacements']}")
print(f"档案规模     : {len(F)}")
print(f"档案 f 范围  : f1[{F[:,0].min():.1f},{F[:,0].max():.1f}] "
      f"f2[{F[:,1].min():.4f},{F[:,1].max():.4f}] f3[{F[:,2].min():.1f},{F[:,2].max():.1f}]")
print(f"决策变量范围 : x[{res['archive_X'][:,0].min():.1f},{res['archive_X'][:,0].max():.1f}] "
      f"y[{res['archive_X'][:,1].min():.1f},{res['archive_X'][:,1].max():.1f}]")

ok = (
    res["n_evals"] == 91 + 91 * 30
    and res["total_replacements"] > 0
    and len(F) > 10
    and np.isfinite(F).all()
    and np.isfinite(res["archive_X"]).all()
)
print("冒烟测试:", "PASS ✔" if ok else "FAIL ✘")
sys.exit(0 if ok else 1)

# 校园新增食堂三目标 MOEA/D（学习项目）

用 MOEA/D（MOEA based on Decomposition, Zhang & Li 2007）求解**校园新增食堂选址**三目标优化问题。

## 数据版本

本仓库包含两个版本的实验数据和对应的入口：

| 模型 | 数据源 | 需求点 | 竞争点 | 搜索空间 | 使用场景 |
|------|--------|--------|--------|---------|---------|
| **南湖 21 点模型**（当前主模型） | `data/nanhu_points.csv`（WGS-84 坐标，上层收口） | 17 个（D01-D10 宿舍 + T01-T07 教学区） | 4 个（C01-C04） | A3 Polygon（高德底图描边 + 四门配准，工程近似 ±20-40m，用户已批准） | 替代旧版虚拟校园的正式模型 |
| **旧版虚拟校园**（baseline） | `src/campus_model.py` 内硬编码坐标 | 约 6 个宿舍点（等权） | 3 个已有食堂 | `[0,1000] × [0,700]` 矩形校园 | 第一版学习/验证原型 |

## 入口与用法

在项目根目录 `canteen-moead/` 下执行：

```bash
# ── 默认：南湖 21 点快速实验（演示） ──
python main.py                          # 5 种子 × 100 代 → output/moead_nanhu_fast/demo_5x100/

# ── 入口参数 ──
python main.py --quick                  # 1 种子 × 50 代快速验证 → output/moead_nanhu_fast/quick_1x50/
python main.py --demo                   # 同默认，5×100 代
python main.py --nanhu                  # 南湖 10 种子 × 200 代完整实验 → output/moead_nanhu/
python main.py --baseline               # 旧版虚拟校园 3 种子 × 200 代 → output/moead/
python main.py --tests                  # 正确性自检 + 冒烟测试
python main.py --plots                  # 用已保存的实验结果重新出图
```

## 输出目录

```
output/
├── moead/                     # 旧版基线实验（虚拟校园，3 种子 × 200 代）
│   ├── archive_seed{1-3}_X.npy / _F.npy   # Pareto 解集
│   ├── experiment_results.json            # 全部统计 + 聚合
│   └── fig{1-9}_*.png                     # 9 张结果图
├── moead_nanhu/               # 南湖 21 点完整实验（10 种子 × 200 代）
│   ├── archive_seed{1-10}_X.npy / _F.npy  # 10 个独立种子
│   ├── experiment_results.json            # 全部统计
│   ├── stage0_stats.json / *.npy          # 阶段 0 采样数据
│   └── figures/                           # 5 张可视化图
└── moead_nanhu_fast/          # 南湖 21 点快速实验（独立输出）
    ├── quick_1x50/                        # 1 种子 × 50 代
    │   ├── archive_seed1_X.npy / _F.npy
    │   └── experiment_results.json
    ├── demo_5x100/                        # 5 种子 × 100 代
    │   ├── archive_seed{1-5}_X.npy / _F.npy
    │   └── experiment_results.json
    └── experiment_log.txt                 # 实验运行日志
```

## 项目结构

```
canteen-moead/
├── data/                     # 南湖校区真实数据层
│   ├── nanhu_points.csv      # C01-C04（竞争点）、D01-D10（宿舍）、T01-T07（教学）共 21 点（全部有坐标，WGS-84）
│   ├── nanhu_reference.json  # 校区信息/参考锚点/边界与成本场状态
│   └── nanhu_boundary_polygon.json  # A3 搜索多边形（21 顶点，用户已批准）
├── src/                      # 问题建模 + 可行性验证
│   ├── campus_model.py       # [旧版] 校园数据 + 三目标函数（硬编码虚拟校园）
│   ├── nanhu_campus.py       # [当前] 南湖 21 点加载/结构校验/A3 约束/评价函数
│   ├── validate.py           # 6000 随机采样验证
│   └── make_figures.py       # 阶段一 5 张图
├── moead/                    # MOEA/D 算法核心（从零实现，无黑箱库）
│   ├── config.py             # 全部参数集中配置
│   ├── weights.py            # 单纯形格点权重（H=12 → 91 子问题）+ 邻域 T=18
│   ├── normalization.py      # 动态归一化（z* + z_nad）
│   ├── scalarization.py      # Tchebycheff（默认）/ PBI（预留）
│   ├── operators.py          # SBX(ηc=15, pc=0.9) + 多项式变异(ηm=20, pm=1/2)
│   ├── dominance.py          # Pareto 支配判定（向量化）
│   ├── archive.py            # 外部 Pareto 档案（非支配过滤 + 去重）
│   └── moead.py              # MOEA/D 主循环
├── metrics/                  # HV（精确 3D 阶梯法）/ IGD / Spacing
├── experiments/              # 实验运行脚本
│   ├── run_experiments.py    # [旧版基线] 虚拟校园 3 种子 × 200 代 → output/moead/
│   ├── run_nanhu.py           # [南湖完整] 10 种子 × 200 代 → output/moead_nanhu/
│   └── run_nanhu_fast.py     # [南湖快速] quick(1×50) + demo(5×100) → output/moead_nanhu_fast/
├── visualization/            # 结果可视化
├── tests/                    # 正确性自检 + 冒烟测试
├── main.py                   # 主入口（默认南湖快速实验）
└── output/                   # 实验产物（见上表）
```

## 第一版关键参数

| 参数 | 值 |
|---|---|
| 权重向量 | 单纯形格点 H=12 → N=91 个子问题 |
| 邻域 T | 18（≈0.2N，含自身） |
| 标量化 | Tchebycheff（归一化目标空间） |
| 交叉 | SBX，pc=0.9，ηc=15 |
| 变异 | 多项式，pm=1/2，ηm=20 |
| 归一化 | 动态：z*=运行最小，z_nad=运行最大 |
| 档案 | 外部非支配档案，决策/目标空间双去重 |
| 阶段 0 采样 | 6000 随机点（A3 多边形内拒绝采样）/ 旧版矩形内均匀采样 |

## 约束机制（南湖 A3 Polygon）

- 阶段 0/参考前沿采样：`src.nanhu_campus.sample_in_polygon` — A3 多边形内部拒绝法均匀点（无投影、无边界聚集）
- MOEA/D 运行时：`build_constrained_evaluate` — 界外候选返回有界罚值（可行域各目标最大估计 ×1.1），罚解被一切可行解严格支配，外部档案保持干净

## 依赖

Python 3.13 验证。依赖库：numpy、pandas、matplotlib、scipy。

## 报告

- 阶段 1：《校园新增食堂三目标MOEA-D问题建模与可行性验证报告.md》
- 阶段 2：《校园新增食堂三目标MOEA-D第一版工程实现报告.md》
- 南湖校区分册：《校园新增食堂MOEA-D南湖校区第一版计算报告.md》
- 南湖坐标数据详情：`data/nanhu_coordinate_research.md`
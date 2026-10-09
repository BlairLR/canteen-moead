# 校园新增食堂三目标 MOEA/D（学习项目）

用 MOEA/D（MOEA based on Decomposition, Zhang & Li 2007）求解**校园新增食堂选址**三目标优化问题。

## 数据版本

**南湖 21 点模型**（当前主模型）
- 数据源：`data/nanhu_points.csv`（WGS-84 坐标，上层收口）
- 需求点：17 个（D01-D10 宿舍 + T01-T07 教学区）
- 竞争点：4 个（C01-C04）
- 搜索空间：A3 Polygon（高德底图描边 + 四门配准，工程近似 ±20-40m，用户已批准）

## 入口与用法

在项目根目录 `canteen-moead/` 下执行：

```bash
# ── 默认（优化 + 可视化） ──
python main.py                              # 91 子问题 × 200 代 × seed 42 → output/moead_nanhu/

# ── 可选参数 ──
python main.py --tests                      # 只运行正确性自检 + 多边形冒烟测试（不进行优化）
```

## 输出目录

```
output/
├── moead_nanhu/               # 南湖校区实验（1 种子 × 200 代）
│   ├── archive_seed42_X.npy / _F.npy       # Pareto 候选解（决策变量 + 目标值）
│   ├── experiment_results.json             # 实验统计
│   ├── stage0_stats.json / *.npy           # 阶段 0 采样数据
│   └── hv_convergence.png                  # HV 收敛曲线图
└── *.png                      # 可视化图表（由 plot_all.py 生成）
```

## 项目结构

```
canteen-moead/
├── data/                     # 南湖校区真实数据层
│   ├── nanhu_points.csv      # C01-C04（竞争点）、D01-D10（宿舍）、T01-T07（教学）共 21 点（全部有坐标，WGS-84）
│   ├── nanhu_reference.json  # 校区信息/参考锚点/边界与成本场状态
│   └── nanhu_boundary_polygon.json  # A3 搜索多边形（21 顶点，用户已批准）
├── src/                      # 问题建模 + 可行性验证
│   └── nanhu_campus.py       # 南湖 21 点加载/结构校验/A3 约束/评价函数
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
│   └── run_nanhu.py           # 南湖校区实验（1 种子 × 200 代 → output/moead_nanhu/）
├── visualization/            # 结果可视化
│   └── plot_all.py           # 统一可视化入口（只读已有实验结果，不自动运行 MOEA/D）
├── tests/                    # 正确性自检 + 冒烟测试
├── main.py                   # 唯一推荐入口（默认：优化 + 可视化）
└── output/                   # 实验产物（见上表）
```

## 关键参数

| 参数 | 值 |
|---|---|
| 权重向量 | 单纯形格点 H=12 → N=91 个子问题 |
| 邻域 T | 18（≈0.2N，含自身） |
| 标量化 | Tchebycheff（归一化目标空间） |
| 交叉 | SBX，pc=0.9，ηc=15 |
| 变异 | 多项式，pm=1/2，ηm=20 |
| 归一化 | 动态：z*=运行最小，z_nad=运行最大 |
| 档案 | 外部非支配档案，决策/目标空间双去重 |
| 阶段 0 采样 | 6000 随机点（A3 多边形内拒绝采样） |

## 约束机制（南湖 A3 Polygon）

- 阶段 0/参考前沿采样：`src.nanhu_campus.sample_in_polygon` — A3 多边形内部拒绝法均匀点（无投影、无边界聚集）
- MOEA/D 运行时：`build_constrained_evaluate` — 界外候选返回有界罚值（可行域各目标最大估计 ×1.1），罚解被一切可行解严格支配，外部档案保持干净

## 依赖

Python 3.13 验证。依赖库：numpy、pandas、matplotlib、scipy。

## 数据溯源

- **南湖 21 点坐标**：`data/nanhu_points.csv`（WGS-84，上层收口）。每个点的经纬度来源、核对过程、置信度记录于 `nanhu_coordinate_research.md`，补充信息见 `data/nanhu_reference.json`。
- **A3 多边形**：`data/nanhu_boundary_polygon.json`（21 顶点，高德底图描边 + 四门配准，工程近似 ±20-40m，用户已批准）。
- **底图**：`data/nanhu_campus_map.png`。
- **边界配准说明**：`data/nanhu_boundary.md`。
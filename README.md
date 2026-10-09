# 校园新增食堂三目标 MOEA/D（学习项目）

用 MOEA/D（MOEA based on Decomposition, Zhang & Li 2007）求解"校园新增 1 个食堂位置"的三目标优化问题。问题定义见 `src/campus_model.py`（唯一事实来源）。

## 问题定义（第一版，禁止随手修改）

- 决策变量：X = (x, y) ∈ [0,1000] × [0,700]（米）
- 目标（全部最小化）：
  - f1：16 栋宿舍（等权）到新食堂的平均欧氏距离（米）
  - f2：与 3 个已有食堂的高斯服务场重叠度，σ_s=150 m（无量纲 [0,1]）
  - f3：土地成本 = 80 + 240·exp(热点1) + 170·exp(热点2)（万元）
- 约束：仅盒约束（校园边界）

## 项目结构

```
canteen-moead/
├── data/                     # 南湖校区真实数据层（坐标收口后接入选址计算）
│   ├── nanhu_points.csv      # C01-C04（竞争点）、D01-D10（宿舍）、T01-T07（教学）共 21 点（全部有坐标，WGS-84）
│   └── nanhu_reference.json  # 校区信息/参考锚点/边界与成本场状态
├── src/                      # 阶段一：问题建模 + 可行性验证
│   ├── campus_model.py       # 校园数据 + 三目标函数（唯一事实来源）
│   ├── nanhu_campus.py       # 南湖点位加载/结构校验/就绪检查（python src/nanhu_campus.py）
│   ├── validate.py           # 6000 随机采样验证（相关性/成对前沿/3D前沿）
│   └── make_figures.py       # 阶段一 5 张图
├── moead/                   # 阶段二：MOEA/D 算法核心（从零实现，无黑箱库）
│   ├── config.py            # 全部参数集中配置
│   ├── weights.py           # 单纯形格点权重（H=12 → 91 子问题）+ 邻域 T=18
│   ├── normalization.py     # 动态归一化（z* 理想点 + z_nad 最低点估计）
│   ├── scalarization.py     # Tchebycheff（默认）/ PBI（预留接口）
│   ├── operators.py         # SBX(ηc=15, pc=0.9) + 多项式变异(ηm=20, pm=1/2)
│   ├── dominance.py         # Pareto 支配判定（向量化）
│   ├── archive.py           # 外部 Pareto 档案（非支配过滤 + 去重）
│   └── moead.py             # MOEA/D 主循环
├── metrics/                 # HV（精确 3D 阶梯法）/ IGD / Spacing
├── visualization/
│   └── moead_plots.py       # 阶段二 9 张结果图
├── experiments/
│   └── run_experiments.py   # 10 次独立运行 + 指标统计
├── tests/
│   ├── sanity_tests.py      # 组件正确性自检（HV vs 蒙特卡洛等）
│   └── smoke_test.py        # 小代数冒烟测试
├── main.py                  # 主入口
└── output/
    ├── ...                  # 阶段一产物
    └── moead/               # 阶段二产物（档案 npy / experiment_results.json / 图）
```

## 运行

```bash
python main.py --tests    # 自检 + 冒烟（应全 PASS）
python main.py            # 10 次独立实验 + 全部图表
python main.py --plots    # 只出图
python src/validate.py    # 阶段一验证（先行产物）
python src/make_figures.py
```

依赖：numpy、pandas、matplotlib（Python 3.13 验证）。

## 第一版关键参数

| 参数 | 值 |
|---|---|
| 权重向量 | 单纯形格点 H=12 → N=91 个子问题 |
| 邻域 T | 18（≈0.2N，含自身） |
| 标量化 | Tchebycheff（归一化目标空间） |
| 交叉 | SBX，pc=0.9，ηc=15 |
| 变异 | 多项式，pm=1/2（按变量），ηm=20 |
| 代数 | 500（每次运行 91+91×500=45591 次评价） |
| 归一化 | 动态：z*=运行最小，z_nad=运行最大（分量） |
| 档案 | 外部非支配档案，决策/目标空间双去重 |
| 种子 | 1..10 各独立运行 |

## 报告

- 阶段一：《校园新增食堂三目标MOEA-D问题建模与可行性验证报告.md》
- 阶段二：《校园新增食堂三目标MOEA-D第一版工程实现报告.md》

# NOG distributed first-order experiments

本仓库在真实 CPU processes/Gloo 环境中实现并比较 NOG-FO 与 ME-DOL-FO，用于研究
nonsmooth nonconvex stochastic optimization 中精度要求 epsilon 对通信深度（depth）
和随机一阶 oracle 工作量（work）的影响。

当前有两套互不混用的正式结果：宽范围 `epsilon=0.2--0.01` 使用
[theory_validation_v4](results/theory_validation_v4/)；低 epsilon `0.01000--0.00400`
使用新的 [low_epsilon_v5_symmetric](results/low_epsilon_v5_symmetric/)。v5 采用双方对称调参、
新 seeds、共同 evaluation grid 和更高预算。历史
[epsilon_scaling_v2](results/epsilon_scaling_v2/) 仅用于追溯。

## 当前低 epsilon 对称验证：v5

v5 在 25 个预注册 epsilon、30 个 formal/confirmation seeds 上得到双方全部 `30/30`
confirmed hits。算法参数由全新 pilot seeds `110--114` 自动冻结为：

| 方法 | 冻结算法参数 | 最大训练 rounds | 最大记录 depth |
|---|---|---:|---:|
| NOG-FO | `M=4, eta=2, smooth_B=1` | 15360 | 15362（含两次初始化通信） |
| ME-DOL-FO | `H=12, multiplier=200, smooth_B=1` | 15360 | 15360 |

双方各使用 6 个算法候选、相同总 batch 128 和相同 pilot 预算；随后都从总 batch
`32,64,96,128,192,256` 中独立选择 5/5 hit 后 first-hit work 最小的非递减 schedule。
Formal seeds `20--39` 与 pilot 严格隔离。检测到相邻 depth-ratio 下降超过 20% 后，按冻结
协议追加 seeds `40--49` 和中点 `epsilon=0.007375`，没有重新选择参数。

### v5 严格结论

- 预注册 schedule 的 `ME-DOL/NOG depth` 从 `0.462x` 增至 `1.087x`，但
  Spearman `rho=0.373 < 0.7`，未通过单调趋势门槛。
- `NOG/ME-DOL work` 均值 `1.456x`，范围 `0.799x--2.176x`，CV `0.348`；超出
  预注册范围/CV 门槛。
- `epsilon=0.00750 -> 0.00725` 时 ME-DOL batch 从 32 切到 64，depth ratio 从
  `0.679x` 降到 `0.436x`；增加到 30 seeds 后下降仍为 35.7%。
- 因此 v5 的预注册总 verdict 是 **not fully supported**。只有 hit-rate 门槛通过，
  不能写成对称实验已经验证了完整理论主张。
- 固定共同 batch 诊断中，batch 32/64/96/128 的 depth ratio 都随 epsilon 收紧而上升，
  Spearman 为 `0.992--1.000`；但同 batch 下 work ratio 会相应下降，所以该诊断只说明
  主曲线跳变主要来自 schedule 切换，不能替代主 verdict。

### v5 全部 25 点绝对结果

比例先在相同 seed 内计算，再对 30 seeds 求均值；depth 列为均值 ± 样本标准差。

| ε | N batch | ME batch | N hit | ME hit | N depth | ME depth | ME/N depth | N work | ME work | N/ME work |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01000 | 32 | 32 | 30/30 | 30/30 | 626.8 ± 23.1 | 289.6 ± 25.2 | 0.462x | 20,058 | 9,267 | 2.176x |
| 0.00975 | 32 | 32 | 30/30 | 30/30 | 631.6 ± 23.3 | 295.2 ± 28.3 | 0.468x | 20,211 | 9,446 | 2.156x |
| 0.00950 | 32 | 32 | 30/30 | 30/30 | 641.2 ± 27.1 | 301.6 ± 31.9 | 0.471x | 20,518 | 9,651 | 2.145x |
| 0.00925 | 32 | 32 | 30/30 | 30/30 | 647.6 ± 27.0 | 305.6 ± 33.9 | 0.472x | 20,723 | 9,779 | 2.139x |
| 0.00900 | 32 | 32 | 30/30 | 30/30 | 650.8 ± 25.6 | 313.6 ± 35.6 | 0.482x | 20,826 | 10,035 | 2.096x |
| 0.00875 | 32 | 32 | 30/30 | 30/30 | 658.0 ± 30.4 | 324.0 ± 37.2 | 0.492x | 21,056 | 10,368 | 2.053x |
| 0.00850 | 32 | 32 | 30/30 | 30/30 | 665.2 ± 28.5 | 337.6 ± 44.1 | 0.508x | 21,286 | 10,803 | 2.000x |
| 0.00825 | 32 | 32 | 30/30 | 30/30 | 670.8 ± 32.0 | 358.4 ± 54.2 | 0.536x | 21,466 | 11,469 | 1.912x |
| 0.00800 | 32 | 32 | 30/30 | 30/30 | 681.2 ± 37.9 | 388.8 ± 75.5 | 0.572x | 21,798 | 12,442 | 1.811x |
| 0.00775 | 32 | 32 | 30/30 | 30/30 | 690.8 ± 39.9 | 420.0 ± 94.7 | 0.608x | 22,106 | 13,440 | 1.717x |
| 0.00750 | 32 | 32 | 30/30 | 30/30 | 705.2 ± 40.4 | 477.6 ± 132.1 | 0.679x | 22,566 | 15,283 | 1.571x |
| 0.00725 | 32 | 64 | 30/30 | 30/30 | 721.2 ± 43.4 | 314.4 ± 31.1 | 0.436x | 23,078 | 20,122 | 1.154x |
| 0.00700 | 32 | 64 | 30/30 | 30/30 | 730.0 ± 42.4 | 323.2 ± 30.7 | 0.444x | 23,360 | 20,685 | 1.137x |
| 0.00675 | 32 | 64 | 30/30 | 30/30 | 737.2 ± 42.5 | 340.8 ± 38.0 | 0.463x | 23,590 | 21,811 | 1.093x |
| 0.00650 | 32 | 64 | 30/30 | 30/30 | 752.4 ± 53.1 | 356.8 ± 44.0 | 0.475x | 24,077 | 22,835 | 1.068x |
| 0.00625 | 32 | 64 | 30/30 | 30/30 | 777.2 ± 56.4 | 389.6 ± 62.0 | 0.502x | 24,870 | 24,934 | 1.019x |
| 0.00600 | 32 | 96 | 30/30 | 30/30 | 815.6 ± 90.1 | 345.6 ± 48.2 | 0.427x | 26,099 | 33,178 | 0.799x |
| 0.00575 | 32 | 96 | 30/30 | 30/30 | 851.6 ± 104.6 | 357.6 ± 51.3 | 0.426x | 27,251 | 34,330 | 0.808x |
| 0.00550 | 32 | 96 | 30/30 | 30/30 | 928.4 ± 175.1 | 380.0 ± 56.4 | 0.420x | 29,709 | 36,480 | 0.828x |
| 0.00525 | 64 | 128 | 30/30 | 30/30 | 633.2 ± 67.0 | 359.2 ± 43.9 | 0.573x | 40,525 | 45,978 | 0.893x |
| 0.00500 | 64 | 128 | 30/30 | 30/30 | 707.6 ± 97.4 | 373.6 ± 50.3 | 0.537x | 45,286 | 47,821 | 0.961x |
| 0.00475 | 96 | 128 | 30/30 | 30/30 | 578.8 ± 92.7 | 424.0 ± 81.3 | 0.744x | 55,565 | 54,272 | 1.050x |
| 0.00450 | 128 | 128 | 30/30 | 30/30 | 550.8 ± 110.6 | 468.8 ± 96.4 | 0.871x | 70,502 | 60,006 | 1.207x |
| 0.00425 | 256 | 192 | 30/30 | 30/30 | 389.2 ± 54.8 | 394.4 ± 52.2 | 1.026x | 99,635 | 75,725 | 1.333x |
| 0.00400 | 256 | 192 | 30/30 | 30/30 | 430.0 ± 84.6 | 456.0 ± 73.8 | 1.087x | 110,080 | 87,552 | 1.283x |

- [v5 完整中文报告](results/low_epsilon_v5_symmetric/analysis/low_epsilon_report.md)
- [v5 主 depth/work ratio 图](results/low_epsilon_v5_symmetric/analysis/figures/low_epsilon_ratios.png)
- [v5 固定共同 batch 诊断图](results/low_epsilon_v5_symmetric/analysis/figures/fixed_batch_diagnostics.png)
- [v5 hit-rate 图](results/low_epsilon_v5_symmetric/analysis/figures/low_epsilon_hit_rates.png)
- [v5 每 seed 绝对数据](results/low_epsilon_v5_symmetric/analysis/formal_per_seed.csv)
- [v5 两阶段冻结参数与哈希](results/low_epsilon_v5_symmetric/frozen_parameters.json)
- [v5 复现命令](results/low_epsilon_v5_symmetric/REPRODUCE.md)

## v4 结论摘要

- 27 个 primary 点上，两种方法均为 20/20 confirmed hits。
- paired-seed `ME-DOL/NOG depth` 均值从 0.49x 上升到 1.92x，随 epsilon 变小严格
  上升，Spearman rho=1.000。
- paired-seed `NOG/ME-DOL work` 均值为 1.49x，范围 0.98x--2.05x，CV=0.208，
  属于相同常数量级，但不是严格不变。
- 预先冻结的 work-ratio 上限是 2.00x；epsilon=0.2 的正式值为 2.049x，超出约
  2.5%。因此预注册总 verdict 是 **not fully supported**，不能改写成完全通过。
- epsilon=0.01025 时 NOG batch 从 8 切换为 16。该处 depth 和 work 的跳变同时包含
  参数切换效应，不能解释为只由 epsilon 引起。
- v4 是固定 d、delta 和单一 problem family 上的有限区间、定性 scaling evidence，
  没有精确验证 worst-case 渐进指数。

## 指标和统计口径

| 名称 | 定义 |
|---|---|
| hit | stationarity proxy 连续两个高精度 checkpoints 不超过目标 epsilon |
| depth | 第一次 confirmed hit 对应的通信/算法深度 |
| total work | 到 confirmed hit 为止累计的 stochastic first-order oracle evaluations |
| ME/NOG depth | 先在相同 formal seed 内计算 `ME depth / NOG depth`，再对 20 seeds 求均值 |
| NOG/ME work | 先在相同 formal seed 内计算 `NOG work / ME work`，再对 20 seeds 求均值 |
| capped value | 未命中时使用预注册最大预算对应的 depth/work，不把 non-hit 当成真正 hit |

表中的 depth/work 是跨 seed 的均值，比例是 paired-seed ratio 的均值，因此比例不一定
严格等于两个均值相除。每个 seed 的第一次 reach 值见
[formal_per_seed.csv](results/theory_validation_v4/analysis/formal_per_seed.csv)。

## v4 问题与运行设置

### SyntheticMaxSinL1 问题

| 项目 | v4 正式设置 |
|---|---|
| problem | `SyntheticMaxSinL1` |
| dimension `d` | 100 |
| number of data points `n_data` | 4096 |
| constraint radius `R` | 1 |
| L1 coefficient `lambda` | 0.001 |
| feature scale | 1.0 |
| common feature bias | 0.25 |
| phase mode | `zero` |
| smoothing radius `delta` | 0.1 |
| evaluation bank | `eval_smooth_B=256`, `eval_data_B=512` |
| evaluation cost | 131,072 SFO calls/checkpoint |
| evaluation seed mode | one fixed bank |

### 分布式、seed 和预算

| 项目 | v4 正式设置 |
|---|---|
| workers | `m=8` real CPU processes |
| backend/topology | Gloo / complete topology / exact mean |
| partition mode | total batch fixed, shuffled partitions |
| RNG mode | rank schedule |
| pilot seeds | 100--104 |
| formal seeds | 0--19，与 pilot 严格不重叠 |
| evaluation interval | 24 |
| confirmed hit | 连续 2 个 checkpoints |
| NOG maximum rounds | 960（有效 censoring depth 约 962） |
| ME-DOL maximum rounds | 3840 |
| CPU 上限 | 4 physical tasks × 8 workers = 32 worker processes |
| task timeout | 129,600 seconds |

同一条 trajectory 可同时判断多个 epsilon，所以增加 threshold 点不会使训练成本按点数
线性增加。Raw trajectories 较大，不提交到 Git；冻结文件和分析 manifest 保留了全部
输入的 SHA256。

### 冻结后的算法参数

| method | parameter | value |
|---|---|---:|
| NOG-FO | `M` | 2 |
| NOG-FO | `eta` | 1.0 |
| NOG-FO | `smooth_B` | 1 |
| NOG-FO | rounds | 960 |
| ME-DOL-FO | `epoch_length` | 6 |
| ME-DOL-FO | `theory_multiplier` | 100.0 |
| ME-DOL-FO | rounds | 3840 |

NOG 的 global data batch 是 pilot-only、matched-work 冻结规则选出的单调 schedule：

- epsilon=0.2 到 0.0105：`data_B_total=8`；
- epsilon=0.01025 到 0.002：`data_B_total=16`。

ME-DOL 使用 8 workers，每层总 work 为 8。NOG 在 batch=8 区间每层 work 为 8，
在 batch=16 区间每层 work 为 16。这解释了为什么 batch 切换会同时改变 depth 和
work，也是解读 0.01025 跳变时必须保留的限制。

完整配置和冻结记录：

- [v4 完整运行配置](configs/distributed_cpu_fo_theory_validation_v4.yaml)
- [结果包内配置](results/theory_validation_v4/config.yaml)
- [冻结参数与输入哈希](results/theory_validation_v4/frozen_parameters.json)
- [正式结果审计](results/theory_validation_v4/audit/formal_result_audit.json)

## v4 primary 完整结果

下表同时给出 hit 数、绝对 depth/work 均值和 paired ratios。全部 primary 点均为
20/20 hits，因此这些绝对值是真实 first-hit 均值，不是预算上限替代值。

| epsilon | NOG batch | NOG hit | ME hit | NOG depth | ME depth | ME/NOG depth | NOG work | ME work | NOG/ME work |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20000 | 8 | 20/20 | 20/20 | 135.8 | 66.3 | 0.49x | 1,086.4 | 530.4 | 2.05x |
| 0.18000 | 8 | 20/20 | 20/20 | 162.6 | 82.5 | 0.51x | 1,300.8 | 660.0 | 1.97x |
| 0.16000 | 8 | 20/20 | 20/20 | 187.2 | 96.0 | 0.51x | 1,497.6 | 768.0 | 1.95x |
| 0.14000 | 8 | 20/20 | 20/20 | 211.2 | 111.0 | 0.53x | 1,689.6 | 888.0 | 1.90x |
| 0.12000 | 8 | 20/20 | 20/20 | 235.6 | 127.5 | 0.54x | 1,884.8 | 1,020.0 | 1.85x |
| 0.10000 | 8 | 20/20 | 20/20 | 260.5 | 146.1 | 0.56x | 2,084.0 | 1,168.8 | 1.78x |
| 0.09000 | 8 | 20/20 | 20/20 | 274.1 | 156.9 | 0.57x | 2,192.8 | 1,255.2 | 1.75x |
| 0.08000 | 8 | 20/20 | 20/20 | 288.6 | 168.6 | 0.58x | 2,308.8 | 1,348.8 | 1.71x |
| 0.07000 | 8 | 20/20 | 20/20 | 304.9 | 182.4 | 0.60x | 2,439.2 | 1,459.2 | 1.67x |
| 0.06000 | 8 | 20/20 | 20/20 | 323.3 | 198.0 | 0.61x | 2,586.4 | 1,584.0 | 1.63x |
| 0.05000 | 8 | 20/20 | 20/20 | 345.7 | 217.2 | 0.63x | 2,765.6 | 1,737.6 | 1.59x |
| 0.04000 | 8 | 20/20 | 20/20 | 372.4 | 241.5 | 0.65x | 2,979.2 | 1,932.0 | 1.54x |
| 0.03000 | 8 | 20/20 | 20/20 | 408.4 | 275.1 | 0.67x | 3,267.2 | 2,200.8 | 1.49x |
| 0.02500 | 8 | 20/20 | 20/20 | 429.8 | 297.6 | 0.69x | 3,438.4 | 2,380.8 | 1.45x |
| 0.02000 | 8 | 20/20 | 20/20 | 459.9 | 330.9 | 0.72x | 3,679.2 | 2,647.2 | 1.39x |
| 0.01800 | 8 | 20/20 | 20/20 | 476.6 | 347.7 | 0.73x | 3,812.8 | 2,781.6 | 1.38x |
| 0.01600 | 8 | 20/20 | 20/20 | 492.4 | 369.0 | 0.75x | 3,939.2 | 2,952.0 | 1.34x |
| 0.01500 | 8 | 20/20 | 20/20 | 499.2 | 381.3 | 0.77x | 3,993.6 | 3,050.4 | 1.31x |
| 0.01400 | 8 | 20/20 | 20/20 | 508.0 | 394.8 | 0.78x | 4,064.0 | 3,158.4 | 1.29x |
| 0.01300 | 8 | 20/20 | 20/20 | 518.9 | 418.8 | 0.81x | 4,151.2 | 3,350.4 | 1.25x |
| 0.01200 | 8 | 20/20 | 20/20 | 538.9 | 446.1 | 0.83x | 4,311.2 | 3,568.8 | 1.22x |
| 0.01150 | 8 | 20/20 | 20/20 | 546.0 | 468.3 | 0.86x | 4,368.0 | 3,746.4 | 1.18x |
| 0.01100 | 8 | 20/20 | 20/20 | 566.2 | 535.8 | 0.95x | 4,529.6 | 4,286.4 | 1.09x |
| 0.01075 | 8 | 20/20 | 20/20 | 574.1 | 590.4 | 1.04x | 4,592.8 | 4,723.2 | 1.02x |
| 0.01050 | 8 | 20/20 | 20/20 | 587.2 | 642.9 | 1.10x | 4,697.6 | 5,143.2 | 0.98x |
| 0.01025 | 16 | 20/20 | 20/20 | 400.4 | 685.2 | 1.72x | 6,406.4 | 5,481.6 | 1.25x |
| 0.01000 | 16 | 20/20 | 20/20 | 408.9 | 783.0 | 1.92x | 6,542.4 | 6,264.0 | 1.14x |

机器可读数据：

- [formal_summary.csv](results/theory_validation_v4/analysis/formal_summary.csv)：绝对
  depth/work、标准差和 hit rate；
- [formal_ratios.csv](results/theory_validation_v4/analysis/formal_ratios.csv)：paired
  ratios、标准差和 bootstrap CI；
- [formal_per_seed.csv](results/theory_validation_v4/analysis/formal_per_seed.csv)：每个
  seed 的第一次 reach；
- [formal_trends.json](results/theory_validation_v4/analysis/formal_trends.json)：趋势、
  斜率和预注册 verdict。

![Depth/work ratios](results/theory_validation_v4/analysis/figures/depth_work_ratios.png)

![Depth/work versus epsilon](results/theory_validation_v4/analysis/figures/depth_work_vs_epsilon.png)

## v4 在 0.01 以下的 exploratory/censored 结果

这些点不参与 primary 参数选择或主趋势 verdict。命中不足时不报告伪造的有限
first-hit mean，也不留空；下表明确报告 hit 数以及把 non-hit 计到预算上限后的 capped
depth/work。

| epsilon | NOG hit | ME hit | NOG capped depth | ME capped depth | NOG capped work | ME capped work | 解释 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0095 | 20/20 | 20/20 | 418.7 | 1,316.4 | 6,699.2 | 10,531.2 | full |
| 0.0090 | 20/20 | 12/20 | 451.4 | 2,497.8 | 7,222.4 | 19,982.4 | ME censored |
| 0.0080 | 18/20 | 0/20 | 659.7 | 3,840.0 | 10,555.2 | 30,720.0 | both censored |
| 0.0070 | 2/20 | 0/20 | 950.1 | 3,840.0 | 15,201.6 | 30,720.0 | both censored |
| 0.0060 | 0/20 | 0/20 | 962.0 | 3,840.0 | 15,392.0 | 30,720.0 | no finite ratio |
| 0.0050 | 0/20 | 0/20 | 962.0 | 3,840.0 | 15,392.0 | 30,720.0 | no finite ratio |
| 0.0040 | 0/20 | 0/20 | 962.0 | 3,840.0 | 15,392.0 | 30,720.0 | no finite ratio |
| 0.0030 | 0/20 | 0/20 | 962.0 | 3,840.0 | 15,392.0 | 30,720.0 | no finite ratio |
| 0.0020 | 0/20 | 0/20 | 962.0 | 3,840.0 | 15,392.0 | 30,720.0 | no finite ratio |

![Hit rate](results/theory_validation_v4/analysis/figures/hit_rate_vs_epsilon.png)

当任一方法存在 non-hit 时，capped ratio 只是固定预算下的描述性量，不是双方真实
first-hit depth/work 的无偏估计。尤其是 ME-DOL 0/20 hit 时，只能给出下界或删失结论，
不能声称测得了有限比例。

## 2026 wide-epsilon 实验：当前主结果（历史 v2）

这一标题为旧报告和测试保留；当前正式讨论应优先使用上方 v4/v5，v2 只用于追溯。

### 为什么 v4 与历史 v2 差别很大

v4 不是在 v2 完全相同的设置下仅增加 epsilon 点和 seeds。问题实例、评估精度、算法
参数、每层 work 和统计口径都发生了实质变化，因此两版不是 apples-to-apples
replication。

| 项目 | 历史 v2 | 当前 v4 |
|---|---|---|
| `R` | 4 | 1 |
| phase | random | zero |
| common feature bias | 0 | 0.25 |
| evaluation bank | `64 × 128 = 8,192` | `256 × 512 = 131,072` |
| NOG coarse/medium | `M=4, eta=0.3, smooth_B=1, batch=64` | `M=2, eta=1, smooth_B=1, batch=8/16` |
| NOG fine | `M=24, eta=0.3, smooth_B=8, batch=64` | 同一 global 参数，batch=16 |
| ME-DOL coarse/medium | `epoch=6, multiplier=0.3/3` | `epoch=6, multiplier=100` |
| ME-DOL fine | `epoch=24, multiplier=10` | `epoch=6, multiplier=100` |
| formal maximum rounds | 960 或 61,440 | NOG 960、ME-DOL 3,840 |
| NOG work/depth | coarse/medium 64，fine 512 | 8 或 16 |
| ME-DOL work/depth | 8 | 8 |
| 主要 ratio 口径 | ratio of capped means | mean of paired-seed ratios |

v2 在 epsilon=0.2 到 0.015 时，两种算法均在最早记录深度 6 达到目标，导致 depth
ratio 全为 1；这个区间出现 checkpoint saturation，无法展示更宽松阈值下的真实轨迹
差异。v2 的 4--6x 主要出现在 epsilon=0.01 及其不同删失统计口径中，并不是严格
epsilon>0.01 的正式结果。

### v2 代表性绝对值

以下是 v2 的 capped means；full 行等于真实 first-hit mean。

| epsilon | NOG hit | ME hit | NOG depth | ME depth | ME/NOG depth | NOG work | ME work | NOG/ME work | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.200 | 20/20 | 20/20 | 6.0 | 6.0 | 1.00x | 384.0 | 48.0 | 8.00x | full |
| 0.100 | 20/20 | 20/20 | 6.0 | 6.0 | 1.00x | 384.0 | 48.0 | 8.00x | full |
| 0.050 | 20/20 | 20/20 | 6.0 | 6.0 | 1.00x | 384.0 | 48.0 | 8.00x | full |
| 0.030 | 20/20 | 20/20 | 6.0 | 6.0 | 1.00x | 384.0 | 48.0 | 8.00x | full |
| 0.020 | 20/20 | 20/20 | 6.0 | 6.0 | 1.00x | 384.0 | 48.0 | 8.00x | full |
| 0.015 | 20/20 | 20/20 | 6.0 | 6.0 | 1.00x | 384.0 | 48.0 | 8.00x | full |
| 0.010 | 20/20 | 14/20 | 116.4 | 570.6 | 4.90x | 7,449.6 | 4,564.8 | 1.63x | ME censored |
| 0.009 | 20/20 | 20/20 | 131.6 | 1,632.0 | 12.40x | 67,379.2 | 13,056.0 | 5.16x | full |
| 0.008 | 20/20 | 20/20 | 508.4 | 7,761.6 | 15.27x | 260,300.8 | 62,092.8 | 4.19x | full |

v2 epsilon=0.01 的 `ME/NOG depth` 会随口径变化：

- ratio of capped means：4.90x；
- capped paired ratios 的均值：6.19x；
- 只对双方均 hit 的 paired ratios 求均值：5.73x。

所以旧文档中出现 4x、5x、6x 并不一定是三次不同运行，也可能是同一批删失数据的
不同统计量。v2 的完整文件：

- [v2 正式报告](results/epsilon_scaling_v2/analysis/epsilon_scaling_report.md)
- [v2 绝对值](results/epsilon_scaling_v2/analysis/formal_summary.csv)
- [v2 比例](results/epsilon_scaling_v2/analysis/formal_ratios.csv)
- [v2 分区冻结参数](results/epsilon_scaling_v2/audit/frozen_region_configs.json)

若要严格归因 v2/v4 的差异，需要做逐项控制变量的 bridge/ablation：固定 problem
只换算法参数、固定算法只换 problem，再逐项改变 evaluation bank 和 batch。当前 v4
结果不能单独回答哪个变化贡献了全部差异。

## Pilot、冻结和防止事后挑选

1. 使用 pilot seeds 100--104 测试 NOG 的 `M, eta, smooth_B` 和 ME-DOL 的
   `epoch_length, theory_multiplier`，formal seeds 不参与参数选择。
2. 对 NOG global batch `8,16,...,64` 做 dense-epsilon pilot 校准。
3. 冻结要求所有 primary epsilon 在 pilot 为 5/5 hits，且 batch 随 epsilon 变小
   不能下降；目标是最小化 matched-work 偏差。
4. 冻结参数并确认 formal 输出目录为空后，才运行 seeds 0--19。
5. 60/60 physical formal tasks 通过 SHA256、task fingerprint、rank/shard、trajectory
   和精确 SFO work-accounting 审计。
6. formal 结果没有用于重新选择最有利的参数；预注册门槛也没有在看到结果后修改。

## 理论参照和结论边界

| method | depth | work |
|---|---:|---:|
| ME-DOL | `O(delta^-1 epsilon^-3)` | `O(delta^-1 epsilon^-3)` |
| NOG | `O(d^(1/3) delta^-1 epsilon^(-5/3))` | `O(delta^-1 epsilon^-3)` |

| metric | theory exponent | observed log-log slope |
|---|---:|---:|
| NOG depth | 1.667 | 0.372 |
| ME-DOL depth | 3.000 | 0.636 |
| NOG work | 3.000 | 0.429 |
| ME-DOL work | 3.000 | 0.636 |

观测斜率明显小于 worst-case theory exponent。合适的表述是：在当前固定问题和有限
epsilon 区间中，ME-DOL/NOG depth ratio 随精度要求增强而上升，work 保持同一常数
量级。不能表述为已经精确验证 epsilon^-5/3 或 epsilon^-3，也不能声称所有有限
epsilon 上 NOG 都更快。

## 安装和复现 v5

激活按照 `requirements.txt` 配置的 `NOG` 环境后，可以运行完整的可恢复流水线：

```bash
conda activate NOG
bash scripts/run_low_epsilon_v5.sh
```

流水线依次执行：

1. 双方各 6 个算法候选的对称 pilot；
2. 冻结 NOG/ME-DOL 全局算法参数；
3. 双方相同 6 档总 batch 的对称 pilot；
4. 冻结各自非递减 batch schedule；
5. formal、artifact/work 审计和相邻异常确认；
6. 30-seed 统计、固定 batch 诊断图、中文报告和紧凑结果包。

runner 会复用 fingerprint/SHA256 一致的完整 partial；最多同时运行 4 个任务、每任务
8 workers。各条独立命令、seed 集合和输出目录见
[v5 REPRODUCE.md](results/low_epsilon_v5_symmetric/REPRODUCE.md)。Raw trajectories 位于
`outputs/distributed_cpu_fo_v5/`，不提交 Git；可审计的紧凑结果包位于
`results/low_epsilon_v5_symmetric/`。

## 安装和复现 v4

仓库已有名为 `NOG` 的 Conda 环境时，只需激活并确认依赖：

```bash
conda activate NOG
pip install -r requirements.txt
```

从仓库根目录依次执行：

```bash
conda run -n NOG python -m src.distributed.theory_validation_runner pilot-batch-grid
conda run -n NOG python -m src.distributed.theory_validation_freeze
conda run -n NOG python -m src.distributed.theory_validation_runner formal
conda run -n NOG python -m src.distributed.theory_validation_audit
conda run -n NOG python -m src.distributed.theory_validation_analysis
conda run -n NOG python -m src.distributed.theory_validation_report
conda run -n NOG python -m src.distributed.theory_validation_package
```

运行阶段：

1. `pilot-batch-grid`：只使用 pilot seeds 搜索 batch schedule；
2. `freeze`：根据 pilot 结果冻结参数和输入哈希；
3. `formal`：运行独立 formal seeds，支持原子 partial 和安全 resume；
4. `audit`：检查任务完整性、trajectory 和 work accounting；
5. `analysis`：生成绝对值、paired ratios、bootstrap CI 和趋势；
6. `report`：生成 Markdown 报告和 PNG/PDF 图；
7. `package`：生成可提交 Git 的紧凑结果包。

详细说明见 [REPRODUCE.md](results/theory_validation_v4/REPRODUCE.md)。完整测试基线为
**67 passed, 8 subtests passed**。

## 代码和结果索引

- [low_epsilon_runner.py](src/distributed/low_epsilon_runner.py)：对称 algorithm/batch
  pilot、formal/extra 调度、32-worker 上限和 resume；
- [low_epsilon_freeze.py](src/distributed/low_epsilon_freeze.py)：两阶段自动选择、seed
  隔离、非递减 batch schedule 和冻结哈希；
- [low_epsilon_audit.py](src/distributed/low_epsilon_audit.py)：formal/extra 的哈希、进程、
  checkpoint 和精确 work 审计；
- [low_epsilon_analysis.py](src/distributed/low_epsilon_analysis.py)：30-seed paired ratios、
  bootstrap、异常确认与固定 batch 诊断；
- [low_epsilon_report.py](src/distributed/low_epsilon_report.py)：25 点绝对值、比例和图表；
- [low_epsilon_package.py](src/distributed/low_epsilon_package.py)：v5 紧凑结果包；
- [v5 正式中文报告](results/low_epsilon_v5_symmetric/analysis/low_epsilon_report.md)；
- [v5 结果包清单](results/low_epsilon_v5_symmetric/package_manifest.json)；
- [theory_validation_runner.py](src/distributed/theory_validation_runner.py)：32-process
  调度、pilot/formal 和 resume；
- [theory_validation_freeze.py](src/distributed/theory_validation_freeze.py)：pilot-only
  参数选择、单调 batch schedule 和冻结哈希；
- [theory_validation_audit.py](src/distributed/theory_validation_audit.py)：artifact 与
  work-accounting 审计；
- [theory_validation_analysis.py](src/distributed/theory_validation_analysis.py)：
  censoring-aware 统计、bootstrap 和趋势；
- [theory_validation_report.py](src/distributed/theory_validation_report.py)：结果表和图；
- [theory_validation_package.py](src/distributed/theory_validation_package.py)：紧凑结果包；
- [v4 正式中文报告](results/theory_validation_v4/analysis/theory_validation_report.md)；
- [v4 结果包清单](results/theory_validation_v4/package_manifest.json)；
- [实验计划和完成记录](plan.md)。

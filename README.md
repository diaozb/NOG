# NOG distributed first-order experiments

本仓库实现并比较 NOG-FO 与 ME-DOL-FO 在 nonsmooth nonconvex stochastic
optimization 上的真实 CPU-process/Gloo 实验。当前应优先引用的是
`theory_validation_v4`：它针对导师提出的窄 epsilon、seed 太少、未命中留空和
0.010 异常跳变问题，重新设计了 problem、pilot、预算、统计和审计流程。

## 当前结果：27 个 primary epsilon、20 个 formal seeds

Primary epsilon 从 `0.2` 覆盖到 `0.01`，包含
`0.011, 0.01075, 0.0105, 0.01025, 0.01` 等局部加密点。每个点两种方法均为
20/20 confirmed hits；hit 要求连续两个高精度 checkpoints 达标。

| epsilon | NOG batch | NOG hit | ME-DOL hit | ME/NOG depth | NOG/ME work |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 8 | 20/20 | 20/20 | 0.49x | 2.05x |
| 0.1 | 8 | 20/20 | 20/20 | 0.56x | 1.78x |
| 0.05 | 8 | 20/20 | 20/20 | 0.63x | 1.59x |
| 0.02 | 8 | 20/20 | 20/20 | 0.72x | 1.39x |
| 0.011 | 8 | 20/20 | 20/20 | 0.95x | 1.09x |
| 0.01075 | 8 | 20/20 | 20/20 | 1.04x | 1.02x |
| 0.0105 | 8 | 20/20 | 20/20 | 1.10x | 0.98x |
| 0.01025 | 16 | 20/20 | 20/20 | 1.72x | 1.25x |
| 0.01 | 16 | 20/20 | 20/20 | 1.92x | 1.14x |

完整 27 点上，`ME-DOL/NOG depth` 随 epsilon 变小严格上升，Spearman
`rho=1.000`；比例从 `0.49x` 增至 `1.92x`，表明 NOG 的有限区间优势在更严格精度下
出现并增强。`NOG/ME-DOL work` 的均值为 `1.49x`、CV 为 `0.208`，范围
`0.98x--2.05x`。

预先冻结的 work 上限为 `2.00x`，而 epsilon=0.2 的正式结果为 `2.049x`，超出约
2.5%。因此严格总 verdict 是 **not fully supported**：depth 趋势得到强支持，work
保持常数量级且波动较小，但没有完全通过事先设定的 matched-work 门槛。这个门槛没有在
看到 formal 结果后修改。

- [完整中文报告](results/theory_validation_v4/analysis/theory_validation_report.md)
- [Depth/work ratio 图](results/theory_validation_v4/analysis/figures/depth_work_ratios.png)
- [Depth/work scaling 图](results/theory_validation_v4/analysis/figures/depth_work_vs_epsilon.png)
- [全 epsilon hit-rate 图](results/theory_validation_v4/analysis/figures/hit_rate_vs_epsilon.png)
- [每 seed 数据](results/theory_validation_v4/analysis/formal_per_seed.csv)
- [冻结参数与输入哈希](results/theory_validation_v4/frozen_parameters.json)
- [结果审计](results/theory_validation_v4/audit/formal_result_audit.json)

## 0.01 以下结果不会留空

`0.0095` 到 `0.002` 的 9 个点作为 exploratory/censored 结果完整保留：

- epsilon=0.0095：NOG 20/20，ME-DOL 20/20；
- epsilon=0.009：NOG 20/20，ME-DOL 12/20；
- epsilon=0.008：NOG 18/20，ME-DOL 0/20；
- 更严格点按最大已测 depth/work 报告 hit rate 与下界，不静默舍弃 non-hit seed。

详见完整报告和 `formal_summary.csv`。这些点不参与 primary 参数选择或主趋势 verdict。

## 为什么旧的 0.010 结果会异常

旧 problem 在初始点的 stationarity proxy 已接近 0.01，使 0.011、0.010、0.009 落在
初始化/首个 checkpoint 的边界附近；少量 seed 和稀疏 evaluation 会把配置边界放大成
跳变。v4 使用同一类 `sin + L1` 非凸非光滑问题，但设置：

| 项目 | v4 设置 |
|---|---|
| Problem | `SyntheticMaxSinL1` |
| `d, n_data, R, lambda` | `100, 4096, 1, 0.001` |
| feature scale / common bias / phase | `1.0 / 0.25 / zero` |
| smoothing radius | `delta=0.1` |
| evaluation bank | `256 x 512 = 131072` SFO calls/checkpoint |
| workers | `m=8`, real processes, Gloo exact mean |
| pilot / formal seeds | `100--104` / `0--19`, strictly disjoint |
| CPU cap | 4 concurrent tasks x 8 workers = 32 processes |

默认 problem 行为保持向后兼容；新参数只有在 v4 config 中显式启用。

## Pilot、冻结与正式实验

1. 先测试 NOG 的 `M, eta, smooth_B` 和 ME-DOL 的 `epoch_length,
   theory_multiplier`，并对未命中配置扩展预算。
2. 对 NOG global data batch `8,16,...,64` 使用 5 个 pilot seeds 做 dense-epsilon
   校准。
3. 冻结规则要求每个 primary epsilon 为 5/5 hits，batch 随 epsilon 变小不能下降，
   并最小化 matched-work 偏差；正式 schedule 只使用 batch 8 和 16。
4. 冻结完成且确认 formal 目录为空后，才运行 20 个独立 formal seeds。
5. 60/60 physical tasks 通过 SHA256、task fingerprint、rank/shard、trajectory 和精确
   SFO work accounting 审计。

同一 trajectory 可复用多个 epsilon，因此增加 threshold 点不会线性增加训练成本。
Raw trajectories 较大，不提交 Git；结果包中的 manifests 保存了所有原始输入的 SHA256。

## 理论口径与结论边界

论文 first-order 结果给出：

| method | depth | work |
|---|---:|---:|
| ME-DOL | `O(delta^-1 epsilon^-3)` | `O(delta^-1 epsilon^-3)` |
| NOG | `O(d^(1/3) delta^-1 epsilon^(-5/3))` | `O(delta^-1 epsilon^-3)` |

v4 支持的是固定 `d, delta` 和单一 problem family 上的定性趋势。观测 log-log slopes
明显小于 worst-case theory exponents，因此不能写成已经精确验证 `epsilon^-5/3` 或
`epsilon^-3`，也不能声称所有有限 epsilon 上 NOG 都更快。

## 复现

安装依赖后，从仓库根目录运行：

```bash
conda activate NOG
python -m src.distributed.theory_validation_runner pilot-batch-grid
python -m src.distributed.theory_validation_freeze
python -m src.distributed.theory_validation_runner formal
python -m src.distributed.theory_validation_audit
python -m src.distributed.theory_validation_analysis
python -m src.distributed.theory_validation_report
python -m src.distributed.theory_validation_package
```

所有长任务支持原子 partial 和安全 resume。更详细的命令和结果包说明见
[REPRODUCE.md](results/theory_validation_v4/REPRODUCE.md)。

## 核心文件

- `src/distributed/theory_validation_runner.py`：32-process 上限、pilot/formal 调度和 resume。
- `src/distributed/theory_validation_freeze.py`：pilot-only 选择、单调 batch schedule 和冻结哈希。
- `src/distributed/theory_validation_audit.py`：正式 artifact 与 work accounting 审计。
- `src/distributed/theory_validation_analysis.py`：censoring-aware 统计、bootstrap 和趋势。
- `src/distributed/theory_validation_report.py`：27 点主表、censored 表和图表。
- `src/distributed/theory_validation_package.py`：紧凑、可版本控制的结果包。
- `configs/distributed_cpu_fo_theory_validation_v4.yaml`：完整实验配置。
- `plan.md`：逐步计划、决策和完成状态。

完整测试结果：**67 passed, 8 subtests passed**。

## 2026 wide-epsilon 实验：当前主结果（历史 v2）

该标题对应此前 `results/epsilon_scaling_v2/` 的 17 点实验，保留用于历史测试与结果追溯；当前论文讨论和数值应优先使用上方 v4。

旧的窄 epsilon、runtime 和 worker-robustness 结果仍保存在
`results/epsilon_scaling_v2/` 与 `outputs/distributed_cpu_fo/`，仅作为历史基线，不应与
v4 formal 数字混合引用。

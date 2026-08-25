# NOG optimistic / non-optimistic 消融实验

这是针对学长提出的问题做的独立 CPU 消融，不替换论文主实验。

## 消融问题

固定 SyntheticMaxSinL1 的 FO v4 主图口径，只替换 NOG 的方向更新：

$$
\Delta_t^{\rm opt}=\Pi_{\mathbb B_D}(\Delta_{t-1}-2\eta g_{t-1}+\eta g_{t-2}),
\qquad
\Delta_t^{\rm nonopt}=\Pi_{\mathbb B_D}(\Delta_{t-1}-\eta g_{t-1}).
$$

两种方法使用完全相同的函数、数据分区、随机 oracle 流、评价 bank、worker 数、batch、$M$、eta、初始化 oracle 次数、work/depth 账本和 960 训练轮次。唯一改变的是 optimistic correction。

## 运行设置

- `SyntheticMaxSinL1`：`d=100`、`n=4096`、`R=1`、`lambda=0.001`；
- 8 个 CPU workers，每个 rank 1 thread，4 个并发 task；
- `M=2`、`eta=1.0`、`smooth_B=1`、global data batch=8；
- pilot seeds：600--604；formal seeds：620--639；完全分离；
- formal 共 40 个任务（2 methods × 20 seeds），全部完成；
- 每个方法保留两次初始化 oracle，之后每轮只新增一次 oracle；
- evaluation work 不计入 training SFO work；
- 每个阈值要求连续两个 checkpoint 达标；未命中 seed 不删除，按 capped 结果保留。

## Formal 结果

表中 depth 是首次 confirmed hit；若未命中，则 capped depth 使用该 seed 的最大记录 depth，并由 `hit_count` 明确标记。

| epsilon | NOG-opt hit | NOG-opt capped depth | NOG-non-opt hit | NOG-non-opt capped depth |
|---:|---:|---:|---:|---:|
| 0.20 | 20/20 | 146.0 | 20/20 | 74.0 |
| 0.10 | 20/20 | 267.2 | 20/20 | 146.0 |
| 0.05 | 20/20 | 358.4 | 20/20 | 183.2 |
| 0.03 | 20/20 | 418.4 | 20/20 | 206.0 |
| 0.02 | 20/20 | 464.0 | 20/20 | 230.0 |
| 0.015 | 20/20 | 513.2 | 20/20 | 273.2 |
| 0.01 | 20/20 | 653.6 | 0/20 | 962.0 (censored) |

在 `epsilon=0.05, 0.03, 0.02, 0.015` 处，non-opt 的 paired capped depth/work 约为 opt 的 `0.49--0.54` 倍；在 `epsilon=0.01`，non-opt 在 960 轮内没有形成连续两次命中，而 opt 为 20/20 命中。这个结果说明：在当前固定有限预算和 eta 下，optimistic correction 没有带来实践优势，普通更新反而更快到达中等 epsilon；这不等价于理论定理错误，因为理论是渐近复杂度上界，并不保证每个有限问题和每个固定参数都优于 non-opt。

## 结果文件

- `analysis/trajectory_depth_work.png/pdf`：20-seed epsilon 曲线（depth/work）；
- `analysis/threshold_depth_work.png/pdf`：阈值曲线；
- `analysis/formal_trajectories.csv`：逐 checkpoint formal 原始轨迹；
- `analysis/threshold_first_hit.csv`：逐 seed、逐阈值 hit/censored；
- `analysis/threshold_summary.csv`：均值和 95% t CI；
- `analysis/paired_comparison.csv`：同 seed opt/non-opt 对比；
- `formal/partials/` 和 `formal/task_manifests/`：原始 CPU 任务结果和审计信息；
- `protocol_*.json`、`completion.json`：参数、seed 和运行记录；
- `analysis/SHA256SUMS.txt`：分析输出哈希。

该消融用于判断 optimistic 机制是否在当前有限实验中体现优势，不应被写成对 NOG 理论复杂度定理的直接否定或证明。

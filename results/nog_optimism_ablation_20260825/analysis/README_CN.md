# NOG optimistic / non-optimistic CPU ablation (2026-08-25)

本目录是一个机制消融，不替换论文主实验。它固定 SyntheticMaxSinL1 的 FO v4 主图口径，只替换 NOG 更新式：

- NOG-opt: `Delta_t = Proj(Delta_{t-1} - 2 eta g_{t-1} + eta g_{t-2})`；
- NOG-non-opt: `Delta_t = Proj(Delta_{t-1} - eta g_{t-1})`。

两者使用相同的 `M=2`、`eta=1`、`smooth_B=1`、global data batch=8、8 workers、固定评价 bank、960 rounds 和相同的 problem/partition/oracle random streams。两者都保留并计入两次初始化 oracle，因此 work/depth 账本一致。

## Seed 与运行

- pilot seeds: 600--604；formal seeds: 620--639；完全分离；
- formal: 20 paired seeds × 2 methods = 40 CPU tasks；
- 4 个并发 task，每个 task 8 个 CPU worker，每个 rank 1 thread；
- evaluation work 不计入 training SFO work；所有任务均成功完成。

## 结果解释

在相同参数下，pilot 和 formal 都用于检查 optimistic correction 的独立作用。若 NOG-non-opt 更快，只能说明当前有限函数、batch 和 eta 下 optimistic 的实践收益没有体现；这不等价于理论定理错误，因为定理给出的是假设条件下的渐近复杂度上界，而不是每个有限预算点都优于普通更新。

`threshold_first_hit.csv` 同时保留每个 seed 的 hit/censored 状态；未命中没有删除，而在 capped 汇总中使用该 seed 的最大记录 depth/work，并明确标记 hit_count。

## 文件

- `formal_trajectories.csv`：逐 checkpoint formal 轨迹；
- `threshold_first_hit.csv`：逐 seed 阈值首次 confirmed hit；
- `threshold_summary.csv`：20-seed 汇总和 95% t CI；
- `paired_comparison.csv`：同 seed opt/non-opt 对比；
- `trajectory_depth_work.png/pdf`：epsilon 曲线；
- `threshold_depth_work.png/pdf`：阈值曲线；
- `protocol_formal.json`、`completion.json`：参数、seed、CPU 运行记录；
- `SHA256SUMS.txt`：输出哈希。

## Formal first-hit summary

| epsilon | opt hit | opt capped depth | non-opt hit | non-opt capped depth |
|---:|---:|---:|---:|---:|
| 0.2 | 20/20 | 146.0 | 20/20 | 74.0 |
| 0.1 | 20/20 | 267.2 | 20/20 | 146.0 |
| 0.05 | 20/20 | 358.4 | 20/20 | 183.2 |
| 0.03 | 20/20 | 418.4 | 20/20 | 206.0 |
| 0.02 | 20/20 | 464.0 | 20/20 | 230.0 |
| 0.015 | 20/20 | 513.2 | 20/20 | 273.2 |
| 0.01 | 20/20 | 653.6 | 0/20 | 962.0 |

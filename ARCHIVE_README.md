# NOG 历史实验归档

本分支保存已经完成但不作为当前论文主结果的实验、原始输出和版本对比。它们用于
复盘参数选择、删失现象和实验演进，不应与 `main` 分支中的正式 FO v4 与 ZO frozen
protocol 结果混合引用。

## 归档内容

| 路径或分支 | 内容 | 当前使用方式 |
|---|---|---|
| `results/epsilon_scaling_v2/` | 早期 wide-epsilon FO 协议 | 历史参考，不作为论文主结果 |
| `results/low_epsilon_v5_symmetric/` | FO v5 对称预算与低 epsilon 重测 | 参数与结果不优于 v4，仅用于诊断 |
| `results/v4_v7_comparison/` | FO v4、v6、v7 绝对值和比例对比 | 解释版本差异，不进入当前论文 |
| `fo_experiments/V4_V7_EXPERIMENT_COMPARISON.md` | v4--v7 详细说明 | 历史版本对照 |
| `SHARED_DISTRIBUTED_FO_ZO_BASELINE.md` | 早期 FO/ZO 共用分布式基线 | 实现演进参考 |
| `outputs/` | 早期提交过的原始轨迹和中间输出 | 不在 `main` 跟踪，正式紧凑结果见 `results/` 和 `zo_experiments/` |

## 相关历史分支

- `agent/wide-epsilon-results`
- `agent/theory-validation-retest`
- `agent/low-epsilon-v5`
- `agent/low-epsilon-v6-retune`
- `agent/low-epsilon-v7-theory-scaling`
- `agent/publish-experiment-results`

## 使用限制

1. v5--v7 使用了不同的冻结参数、预算或调参目标，不能与 v4 拼成同一条正式曲线。
2. 命中率不足 20/20 的 conditional first-hit 均值存在 survivor bias，不能当作无删失比例。
3. `outputs/` 可能包含早期或中间轨迹；论文引用应以带 audit、manifest 和 frozen
   parameters 的紧凑结果包为准。
4. 当前正式入口始终以 `main` 分支根目录的 `README.md` 为准。

该分支只做历史保存，不再接受新的正式结果。

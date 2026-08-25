# ZO NOG optimistic / non-optimistic 消融

本目录针对 SyntheticMaxSinL1、a9a 和 ijcnn1 做 NOG-ZO optimistic / non-optimistic 消融。所有新任务强制使用 CPU；只改变 NOG 的更新公式，其他 frozen 参数、随机流、评价设置和 SZO work/depth 账本保持一致。

## 固定设置

optimistic 版本使用

$$
\Delta_t=\Pi(\Delta_{t-1}-2\eta g_{t-1}+\eta g_{t-2}),
$$

non-optimistic 版本使用

$$
\Delta_t=\Pi(\Delta_{t-1}-\eta g_{t-1}).
$$

两种版本都保留两次初始化 oracle，并计入 training SZO work；评价 work 单独记录，不计入 training work。每个 epsilon 需要连续两个已保存 checkpoint 达标。

| 数据集 | (M) | η | smooth_B | data_B_total | eval_every | formal budget |
|---|---:|---:|---:|---:|---:|---:|
| SyntheticMaxSinL1 | 2 | 0.01 | 8 | 64 | 4 | 983,040 |
| a9a | 1 | 9e-5 | 1 | 64 | 96 | 983,040 |
| ijcnn1 | 1 | 1e-4 | 1 | 64 | 96 | 983,040 |

所有任务均使用 8 个 logical workers、CPU，正式新 seed 为 720--739；pilot seed 为 700--704。新 seed 与原 ZO formal 0--19、原 pilot 100--104/200--204 以及 SVM batch-retuned search/validation 400--404/500--504 分离。

参数没有用新 formal 结果重新选择，而是直接移植最新冻结版本：

- SyntheticMaxSinL1：`zo_experiments/frozen_parameters.json` 的 NOG-ZO 参数；
- a9a、ijcnn1：`results/advisor_cpu_batch_retuned_svm/frozen_parameters.json` 的 batch-retuned NOG-ZO 参数。

## 对比基线

- SyntheticMaxSinL1 的 latest NOG-ZO：`outputs/distributed_zo/zo_theory_validation/formal/fixed_work_983040`。该历史目录的 environment 记录为 CUDA；本目录的新消融按当前要求强制 CPU，因此它们适合作为算法/轨迹参考，不是硬件时间 benchmark。
- a9a、ijcnn1 的 latest NOG-ZO：`results/advisor_cpu_batch_retuned_svm/merged/formal_trajectories.csv`，该正式结果本身为 CPU。

## 正式结果（20 seeds）

下表是每个 seed 首次连续两个 checkpoint 达到目标 epsilon 的均值；括号内是命中率。`--` 表示在 983,040 training work 内没有足够 checkpoint 命中。完整逐 seed 数据见 `threshold_first_hit.csv`，因此没有删除失败 seed 或截取有利区间。

### SyntheticMaxSinL1

| 方法 | ε=0.05 depth/work | ε=0.03 depth/work | ε=0.02 depth/work | ε=0.015 depth/work |
|---|---:|---:|---:|---:|
| latest NOG-ZO | 214.4 / 219,546 (100%) | 229.2 / 234,701 (100%) | 237.2 / 242,893 (100%) | 244.0 / 249,856 (55%) |
| current NOG-opt | 214.2 / 219,341 (100%) | 229.0 / 234,496 (100%) | 237.0 / 242,688 (100%) | 243.0 / 248,832 (60%) |
| current NOG-non-opt | **209.8 / 214,835 (100%)** | **224.4 / 229,786 (100%)** | **232.8 / 238,387 (100%)** | **238.0 / 243,712 (70%)** |

### SVM（a9a、ijcnn1）

| 数据集/方法 | ε=0.05 depth/work | ε=0.03 depth/work | ε=0.02 depth/work | ε=0.015 depth/work | ε=0.01 depth/work |
|---|---:|---:|---:|---:|---:|
| a9a latest NOG-ZO | 2,095.8 / 268,262 (100%) | 3,118.2 / 399,130 (100%) | -- | -- | -- |
| a9a current NOG-opt | 2,076.6 / 265,805 (100%) | 3,094.2 / 396,058 (100%) | -- | -- | -- |
| a9a current NOG-non-opt | **2,033.4 / 260,275 (100%)** | **3,060.6 / 391,757 (100%)** | -- | -- | -- |
| ijcnn1 latest NOG-ZO | 3,363.0 / 430,464 (100%) | 3,468.6 / 443,981 (100%) | **3,852.6 / 493,133 (100%)** | 5,398.2 / 690,970 (100%) | 6,887.6 / 881,609 (35%) |
| ijcnn1 current NOG-opt | 3,363.0 / 430,464 (100%) | **3,459.0 / 442,752 (100%)** | 4,155.0 / 531,840 (100%) | **5,359.8 / 686,054 (100%)** | **6,703.8 / 858,086 (25%)** |
| ijcnn1 current NOG-non-opt | 3,363.0 / 430,464 (100%) | 3,454.2 / 442,138 (100%) | 4,145.4 / 530,611 (100%) | **5,359.8 / 686,054 (100%)** | 6,851.0 / 876,928 (30%) |

结论是：在 SyntheticMaxSinL1 上 optimistic 项并未带来优势，non-opt 在这些阈值反而略早；在 SVM 上两种更新几乎重合，a9a 的 non-opt 略早，ijcnn1 则没有稳定赢家。因而这组严格配对消融不支持“optimistic 是 NOG 实际优势来源”的说法；理论算法与实现保持一致，但该机制的经验收益依赖目标函数。

正式任务共 120 个（3 数据集 × 2 方法 × 20 seeds），全部成功；paired `problem_seed`、`partition_seed`、`method_seed`、iteration、depth 和 training work 均一致，`stat_proxy` 非有限行数为 0。机器为 Intel Xeon Gold 5220R，实验使用 4 路进程并发、每个任务 8 logical workers，Python 3.10.20，强制 CPU。正式任务记录的串行任务时间总和约 6,779 秒，实际墙钟约 26 分 53 秒（`formal_run.log`）。

## 结果文件

- `svm_nog_opt_nonopt_equal_budget_epsilon.png/pdf`：仿照四算法 SVM 图绘制的三曲线版本（latest NOG-ZO、NOG-opt、NOG-non-opt），含 a9a/ijcnn1 的 same-depth、same-work 面板、20-seed 均值和 95% CI；对应数据为 `svm_nog_opt_nonopt_equal_budget_epsilon_data.csv`；绘图脚本为 `plot_svm_ablation_equal_budget.py`；
- `zo_ablation_vs_latest_all_datasets.png/pdf`：三行（SyntheticMaxSinL1、a9a、ijcnn1）× 两列（depth、work）的总对比曲线；
- `zo_ablation_synthetic_maxsinl1.png/pdf`、`zo_ablation_a9a.png/pdf`、`zo_ablation_ijcnn1.png/pdf`：分数据集曲线；
- `threshold_first_hit.csv`：逐 seed、逐 epsilon 的命中和 censor 信息；
- `threshold_summary.csv`：20-seed 命中率、首次命中均值、capped 均值；
- `formal_trajectories_summary.csv`：逐 checkpoint 的 20-seed 均值和 95% t 区间；
- `formal_trajectories_synthetic_and_ablation.csv`、`formal_trajectories_svm_and_ablation.csv`：机器可读原始 formal 对比数据；
- `raw/`：pilot/formal 每个任务的原始 CSV；`completion_*.json`：任务审计；
- `protocol.json`、`configs/`：参数、seed、输入配置和哈希；
- `analysis_audit.json`：数据集、seed 数、非有限值和基线来源检查。

图中空心点表示命中率低于 100%，零命中点不绘制。首次命中均值只对成功命中的 seed 计算；capped 结果保留在 CSV 中，不能当作成功命中。

## 复现

```bash
/root/miniconda3/envs/NOG/bin/python \
  zo_optimism_ablation_20260825/run_ablation.py --stage both --concurrency 4

/root/miniconda3/envs/NOG/bin/python \
  zo_optimism_ablation_20260825/analyze_ablation.py
```

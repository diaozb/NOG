# SVM NOG-ZO 联合 batch/M/eta 重选结果

本目录是独立于旧 `advisor_cpu_retuned_svm` 的 CPU 复现实验。目的为在不改变正式 work 上限（983,040）的前提下，联合重选 NOG 的 `batch`、`M` 和 `eta`，检验 SVM 前期 epsilon 平台期是否由 batch 过小造成。旧结果未覆盖。

## 协议与冻结参数

- search seeds：400--404；validation seeds：500--504；formal seeds：0--19，三组完全分离。
- 每个数据集 180 个候选，完整覆盖 983,040 work；指标为四个 work 分位点上 `mean(log(stat_proxy))`，非有限/发散候选淘汰，随后用独立 validation 排名第一者冻结。
- 网格：`M={1,2,4}`；`smooth_B={1,2,4,8}`；`data_B_total={64,128,256}`；eta 为旧值乘 `{0.3,1,3,10,30}`。
- 冻结结果：a9a=`M=1, eta=9e-5, smooth_B=1, data_B_total=64`；ijcnn1=`M=1, eta=1e-4, smooth_B=1, data_B_total=64`。也就是说，独立验证没有支持增大 batch；ijcnn1 的 eta 未变，a9a 仅升至旧值 3 倍。

## 正式结果（20 seeds 均值）

最终 epsilon（work=983,040）：a9a NOG 0.02517、ME-DOL-ZO 0.01923、DGFM 0.01588、DGFM+ 0.02617；ijcnn1 NOG 0.01233、ME-DOL-ZO 0.01479、DGFM 0.01496、DGFM+ 0.02071。故 a9a 同等 work 仍由 DGFM 最好，ijcnn1 由 NOG 最好。NOG 的 a9a 结果没有改善旧 formal 的 0.0182；这不是通过删除 seed 或有利截取得到的，所有正式 seed 均保留。

同等 communication depth=3840 的 epsilon：a9a NOG 0.02837（DGFM 0.01725），ijcnn1 NOG 0.02796（ME-DOL-ZO 0.01479、DGFM 0.01688）。NOG 在同等 depth 下两个数据集都不领先；其后段仍会下降，但前期平台明显存在（a9a 在约 depth 2,000 前、ijcnn1 在约 depth 3,300 前仍高于 0.05）。

## 图与数据

- `svm_equal_budget_epsilon_full.png/pdf`：完整线性纵轴四算法曲线。
- `svm_equal_budget_epsilon_zoom_0_005.png/pdf`：严格 `epsilon=0--0.05` 线性局部图；高于 0.05 的前期原生点被裁剪并在图中注明，不表示从 0.05 初始化。
- `plot_data_full.csv`、`plot_data_zoom_0_005.csv`：最近原生 checkpoint、20-seed 均值和 95% CI；不插值。
- `threshold_first_hit.csv` 与 `threshold_first_hit_per_seed.csv`：进入 0.05、0.03、0.02、0.015 的 depth/work 首次命中（未命中保留为空）。
- `formal_trajectories.csv`、`formal_summary.csv`、`formal_summary_ci95.csv`：逐 checkpoint 原始轨迹和汇总。

所有 baseline 来自未改变的旧 CPU formal shards；`merged/baseline_reuse_audit.csv` 记录了复用审计。NOG 新 formal 为 40 个任务（2 数据集 × 20 seeds），CPU、Torch 线程均为 1，20 个并行进程；无 GPU、无失败任务。

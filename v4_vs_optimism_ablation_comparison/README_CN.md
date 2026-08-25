# SyntheticMaxSinL1：FO v4 与 optimistic 消融的细 epsilon 对比

本目录把之前 FO v4 的细粒度 epsilon 结果与刚完成的 NOG optimistic / non-optimistic 消融放在同一张表和同一组曲线中。结果没有重新运行，也没有插值；v4 的 epsilon 网格直接来自冻结协议，当前消融的阈值从逐 checkpoint 原始轨迹按相同的“两次连续 checkpoint 达标”规则重算。

## 数据来源

- FO v4：`results/theory_validation_v4/analysis/formal_summary.csv`，正式 seeds 0--19，NOG-FO 的最终冻结参数为 `M=2, eta=1, smooth_B=1`。
- 当前消融：`results/nog_optimism_ablation_20260825/analysis/formal_trajectories.csv`，formal seeds 620--639；`NOG-FO` 是 optimistic，`NOG-FO-NONOPT` 是普通更新。
- 两组实验均为 `SyntheticMaxSinL1`、8 workers、CPU。当前消融固定 `data_B_total=8`。

## epsilon 网格和统计口径

表格使用 v4 实验实际测试的全部 epsilon 点：

`0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.018, 0.016, 0.015, 0.014, 0.013, 0.012, 0.0115, 0.011, 0.01075, 0.0105, 0.01025, 0.01, 0.0095, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002`。

`first_hit_*` 是 20 个 seed 中成功命中的 seed 的首次确认 checkpoint 均值；`hit` 给出命中数/seed 数。若某个 epsilon 在预算内没有命中，则 first-hit 均值为空，不能把预算终点误当成成功结果。原始长表同时保留 `capped_*`，表示预算截断位置，仅用于审计。

当前消融只跑到 960 轮，因此在较小 epsilon 处会出现部分命中或 0/20 命中。图中空心点表示命中率低于 100%，0 命中点不绘制。

## 一个重要的公平性说明

FO v4 在 epsilon=0.01025 和 0.01 时将 `data_B_total` 从 8 调为 16；其他 v4 点主要使用 8。当前 optimistic / non-optimistic 消融始终固定 batch=8。因此，曲线可用于查看细 epsilon 区域的趋势和 optimistic 消融结果，但不能把 v4 尾部点解读为严格的“同 batch”比较。`epsilon_threshold_comparison.csv` 末尾保留了 `v4_data_B_total` 和 `current_data_B_total` 供检查。

另外，v4 与当前消融的“核心算法参数”相同，并不代表完整运行设置相同。实际 formal 运行配置中：

- v4 的 `config_used.json` 使用 `eval_every=2`，且没有 `strict_eval_grid`；因为 `M=2`，基本每个 NOG block 都会保存一个评价点，960 轮共有约 480 个 checkpoint。
- 当前消融使用 `eval_every=24` 和 `strict_eval_grid=true`，只保存第 24、48、...、960 轮，共 40 个 checkpoint。

训练更新本身不因评价频率而改变，但阈值判定要求“连续两个已保存 checkpoint 达标”。因此当前消融可能在两个稀疏 checkpoint 之间跨过一个短暂低点，却不会被记为 confirmed hit；v4 的密集评价更容易捕捉到这种下降。两组 formal seeds 也不同（v4 为 0--19，当前为 620--639），所以还存在正常的 seed 方差。此前表格中的“命中不了”首先应理解为“在当前 960 轮、当前评价网格下没有被确认命中”，不能直接理解为底层迭代轨迹一定没有短暂低于该 epsilon。

## 文件

- `epsilon_threshold_comparison.csv`：宽格式机器可读总表，包含全部 epsilon、命中数/命中率、首次命中 depth/work、预算截断 depth/work 和 batch 记录。
- `epsilon_threshold_long.csv`：长格式统计表，保留 v4 的 `primary` / `exploratory_censored` scope。
- `epsilon_threshold_comparison.md`：适合直接阅读的首次命中表。
- `v4_vs_current_epsilon_depth_work.png/pdf`：两幅曲线，左为 communication depth，右为 training work；横轴为对数 epsilon，纵轴为线性 depth/work。
- `compare_v4_and_ablation.py`：重新生成上述 CSV、Markdown 和图片的脚本。

生成命令：

```bash
/root/miniconda3/envs/NOG/bin/python \
  v4_vs_optimism_ablation_comparison/compare_v4_and_ablation.py
```

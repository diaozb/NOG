# 论文实验图与复现说明（2026-08-24）

## 这张图是什么

用户所示图片对应本目录中的
`figures/fo_first_hit_depth_work.pdf`（PNG 版本同名）。它不是 SVM、a9a 或
ijcnn1 实验，而是 `SyntheticMaxSinL1` 上的正式 FO theory-validation v4
实验：左图是首次 confirmed hit 的 communication depth，右图是首次
confirmed hit 的训练 SFO work；曲线为 20 个 formal seeds 的均值，阴影为
95% t 置信区间。

该图在论文正文中作为 Figure 1 使用，正文引用的文件为
`nog_iclr2027_complete_source/figures/fo_first_hit_depth_work.pdf`。
对应的 ZO 主图为 `figures/zo_first_hit_depth_work.pdf`，比较 NOG-ZO、
ME-DOL-ZO、DGFM 和 DGFM+。

## FO 实验设置

- 目标函数：
  `F(x;ξ)=max_r sin(a_{ξ,r}^T x+b_{ξ,r}) + λ||x||_1`；
- `d=100`、`n=4096`、`R=1`、`λ=10^-3`，8 个 logical workers；
- pilot seeds：100--104；formal seeds：0--19，二者完全分离；
- NOG-FO 与 ME-DOL-FO 参数在 formal seeds 运行前冻结；
- 主图只保留双方 batch 均为 8 的同质区间 `ε∈[0.0105,0.2]`；
- 25 个 epsilon 阈值、每个算法每个阈值均为 20/20 confirmed hits；
- confirmed hit 要求连续两个有效 evaluation checkpoints 的 stationarity
  proxy 不超过目标 epsilon；evaluation calls 不计入训练 work。

NOG-FO 在最严格展示阈值 `ε=0.0105` 的平均 depth/work 为
`587.2 / 4697.6`，ME-DOL-FO 为 `642.9 / 5143.2`。整个展示区间反映的是
NOG 的相对 depth 随 epsilon 变小而改善；work 仍是同数量级，不能据此声称
NOG 在所有精度下全面优于基线。

## 数据与生成关系

图片由 `scripts/generate_theory_validation_figures.py` 生成，FO 原始汇总为
`data/fo_formal_summary.csv`，ZO 原始汇总为 `data/zo_formal_summary.csv`。
冻结参数、运行配置和正式任务审计分别位于 `data/` 与 `audit/`。脚本使用的
源数据来自仓库中的 `results/theory_validation_v4/` 和 `zo_experiments/formal/`，
不是从单个 seed 或有利区间手工截取。

## GitHub 状态

本目录用于保存论文实际采用的图片和最小可审计实验说明；提交后位于 GitHub
仓库 `main` 分支的
`results/paper_experiments_20260824/`。论文源码和正文引用的图片同时保留在
`nog_iclr2027_complete_source/`。旧图没有删除实验数据；旧图的完整历史归档
保留在本地工作区的 `results/paper_figures_unused_archive_20260824/`，不作为本次
论文主结果上传。

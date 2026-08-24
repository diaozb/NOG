# NOG：高并行非光滑非凸优化

本仓库实现 Nonconvex Optimistic Gradient（NOG）的 first-order（FO）和
zeroth-order（ZO）方法，并以通信深度（`depth`）和总 oracle 工作量（`work`）为
主要指标进行可审计实验。仓库同时包含当前 ICLR 论文源码、正式实验数据、冻结参数、
审计文件、论文图片和 CPU 复现脚本。

当前论文版本的结论边界是：SyntheticMaxSinL1 实验支持 NOG 的相对
communication-depth 趋势；有限实例并不支持 NOG 在所有目标精度或总 work 下普遍领先。
a9a、ijcnn1 上的 SVM 实验作为真实数据补充和适用性边界保存，不作为正文主结果。

## 快速入口

| 内容 | 入口 |
|---|---|
| 论文实验图片、数据快照、审计与复现 | [`results/paper_experiments_20260824/`](results/paper_experiments_20260824/) |
| 论文实验图中文说明 | [`results/paper_experiments_20260824/README_CN.md`](results/paper_experiments_20260824/README_CN.md) |
| FO v4 正式结果包 | [`results/theory_validation_v4/`](results/theory_validation_v4/) |
| ZO 20-seed 正式结果包 | [`zo_experiments/formal/`](zo_experiments/formal/) |
| SVM 真实数据完整结果包 | [`svm_cpu_joint_retuning_2026/`](svm_cpu_joint_retuning_2026/) |
| SVM 中文实验报告 | [`SVM_CPU_EXPERIMENT_REPORT_CN.md`](SVM_CPU_EXPERIMENT_REPORT_CN.md) |
| ICLR 论文源码与当前图片 | [`nog_iclr2027_complete_source/`](nog_iclr2027_complete_source/) |
| FO 实验说明 | [`fo_experiments/README.md`](fo_experiments/README.md) |
| ZO 实验说明 | [`zo_experiments/README.md`](zo_experiments/README.md) |
| 历史实验索引 | [`archive/legacy-experiments`](https://github.com/diaozb/NOG/tree/archive/legacy-experiments) |

## 研究问题和复杂度口径

主要 synthetic 目标为

```text
F(x; ξ) = max_r sin(a_{ξ,r}^T x + b_{ξ,r}) + λ ||x||_1
```

它同时包含非凸的正弦项和非光滑的 ℓ1 项。正式 SyntheticMaxSinL1 配置为
`d=100`、`n=4096`、`R=1`、`λ=10^-3`、8 个 logical workers。所有方法从相同初始点
出发，并使用固定 evaluation bank 计算高精度 stationarity proxy。

- `confirmed hit`：连续两个有效 checkpoint 的 stationarity proxy 都不超过目标
  `epsilon`；
- `depth`：首次 confirmed hit 对应的依赖图/通信层数；
- `work`：到达首次 confirmed hit 前累计的训练 oracle calls，evaluation calls 单独记录；
- `paired ratio`：先在相同 formal seed 内计算 baseline/NOG，再跨 seed 汇总；
- 未命中结果保留为 right-censored，不用 conditional mean 伪装成完整样本。

## 论文主实验

### FO：NOG-FO vs. ME-DOL-FO

来源是 [FO theory-validation v4](results/theory_validation_v4/)。pilot seeds 为
100--104，formal seeds 为 0--19，参数在 formal 实验前冻结。论文 Figure 1 使用双方
per-layer batch 均为 8 的同质区间 `epsilon ∈ [0.0105, 0.2]`：

- 25 个 epsilon 阈值，两个算法均为 20/20 confirmed hits；
- `ME-DOL/NOG depth` 从 `0.488` 单调增加到 `1.102`，Spearman `rho=1`；
- `NOG/ME-DOL work` 为 `0.980--2.049`，仍是同数量级；
- 因此论文只报告相对 depth 趋势，不宣称粗精度下 NOG 全面优于 ME-DOL。

论文实际使用的图：[FO depth/work PDF](results/paper_experiments_20260824/figures/fo_first_hit_depth_work.pdf)。

### ZO：NOG-ZO vs. ME-DOL-ZO、DGFM、DGFM+

来源是 [ZO formal package](zo_experiments/formal/)。四种方法共享每个 seed 的
`983,040` training SZO work cap，pilot seeds 为 100--104，formal seeds 为 0--19。
论文 Figure 2 只绘制四种方法都达到 20/20 的共同区间 `epsilon ∈ [0.03, 0.2]`：

- 13 个共同 epsilon 阈值，四个算法均为 20/20 confirmed hits；
- NOG-ZO 在该区间的 mean communication depth 均最低；
- depth ratios 随 epsilon 变小而增大；
- NOG-ZO 不是所有点的 work 最低，论文将 work 作为描述性次要指标。

论文实际使用的图：[ZO depth/work PDF](results/paper_experiments_20260824/figures/zo_first_hit_depth_work.pdf)。

两张图均为 20 formal seeds 均值、95% t 置信区间、无插值的原生 first-hit checkpoint。

## SVM 真实数据补充实验

SVM 结果位于 [svm_cpu_joint_retuning_2026](svm_cpu_joint_retuning_2026/)，覆盖 a9a 和
ijcnn1，并保留 batch、`M`、`eta` 联合 pilot、validation、20-seed formal trajectories、
完整纵轴图和 `epsilon=0--0.05` 放大图。完整中文讨论见
[`SVM_CPU_EXPERIMENT_REPORT_CN.md`](SVM_CPU_EXPERIMENT_REPORT_CN.md)。

当前解释是：SVM 上 NOG 的 stationarity proxy 存在前期平台期，重新联合选参后仍没有形成
可替代 SyntheticMaxSinL1 主结果的稳定 depth/work 优势。因此 SVM 不放入论文正文的主
实验图，只作为附录/适用性边界材料保存；不删除失败或不利 seed。

## 参数选择与公平性

FO 和 ZO 均采用 pilot-then-freeze：

1. 只用 pilot seeds 100--104 搜索候选参数；
2. 在 formal seeds 0--19 运行前冻结参数、预算、阈值和统计规则；
3. formal seeds 不参与重新调参；
4. 保存冻结 JSON、配置、逐 seed 数据、审计结果和 SHA256 manifest；
5. 图表只使用预先定义的共同完整支持区间。

当前代表性参数：

| Method | Frozen parameters |
|---|---|
| NOG-FO | `M=2`, `eta=1.0`, one smoothing sample；主图 batch=8 |
| ME-DOL-FO | `epoch_length=6`, `theory_multiplier=100` |
| NOG-ZO | `M=2`, `eta=0.01`, 8 smoothing directions |
| ME-DOL-ZO | `epoch_length=12`, `theory_multiplier=60` |
| DGFM / DGFM+ | `eta=0.05` |

## 代码和结果结构

| 路径 | 内容 |
|---|---|
| `src/distributed/` | FO/ZO 算法、runner、统计和审计代码 |
| `configs/` | 正式、pilot 和扩展实验配置 |
| `fo_experiments/` | FO 实验文档和计划 |
| `zo_experiments/` | ZO 文档、formal 表格、图和审计 |
| `results/theory_validation_v4/` | FO v4 正式汇总、冻结参数和审计 |
| `results/paper_experiments_20260824/` | 论文实际采用图片和最小复现包 |
| `svm_cpu_joint_retuning_2026/` | SVM CPU 联合重选参结果包 |
| `nog_iclr2027_complete_source/` | 论文 LaTeX、附录和主图 |
| `tests/` | 协议、runner、计数和等价性测试 |

## CPU 环境和复现

论文正式结果和绘图均使用 CPU。推荐直接使用现成解释器，不依赖 `conda activate`：

```bash
cd /data/diaozb/NOG
/root/miniconda3/envs/NOG/bin/python \
  results/paper_experiments_20260824/scripts/generate_theory_validation_figures.py
```

该命令从审计过的 FO/ZO 汇总 CSV 重新生成论文两张图。完整说明见
[`results/paper_experiments_20260824/REPRODUCE.md`](results/paper_experiments_20260824/REPRODUCE.md)。

编译当前论文：

```bash
cd /data/diaozb/NOG/nog_iclr2027_complete_source
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/nog_iclr2027_build nog_iclr2027.tex
```

当前版本正文和结论在第 9 页内，参考文献从第 9 页开始，量子理论和实验细节在附录。

## 测试

```bash
/root/miniconda3/envs/NOG/bin/python -m pytest -q
```

正式结果中的 raw trajectories 和临时日志默认不作为主结果提交；紧凑 CSV、冻结配置、
审计 JSON 和图像保存在上述版本化目录中。工作区中可能存在未提交的本地实验文件，不能
与 `main` 中的正式结果混合使用。

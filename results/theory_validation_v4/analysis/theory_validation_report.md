# NOG 宽 epsilon 理论验证实验（v4）

## 结论

在 27 个 primary epsilon（0.2 到 0.01）和每点 20 个独立 formal seeds 上，ME-DOL/NOG depth 比例的 Spearman rho 为 1.000。
NOG/ME-DOL work 比例均值为 1.49x，范围 0.98--2.05x，CV=0.208。
预先冻结判据的总 verdict：**not fully supported**。

这里的正确解释是有限区间中的定性 scaling evidence。固定 d、delta 和单一 problem family，不能据此声称已经精确验证渐进指数。

## Primary 结果（完整保留 27 个点）

| epsilon | NOG batch | NOG hit | ME-DOL hit | ME/NOG depth | NOG/ME work |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 8 | 20/20 | 20/20 | 0.49x | 2.05x |
| 0.18 | 8 | 20/20 | 20/20 | 0.51x | 1.97x |
| 0.16 | 8 | 20/20 | 20/20 | 0.51x | 1.95x |
| 0.14 | 8 | 20/20 | 20/20 | 0.53x | 1.90x |
| 0.12 | 8 | 20/20 | 20/20 | 0.54x | 1.85x |
| 0.1 | 8 | 20/20 | 20/20 | 0.56x | 1.78x |
| 0.09 | 8 | 20/20 | 20/20 | 0.57x | 1.75x |
| 0.08 | 8 | 20/20 | 20/20 | 0.58x | 1.71x |
| 0.07 | 8 | 20/20 | 20/20 | 0.60x | 1.67x |
| 0.06 | 8 | 20/20 | 20/20 | 0.61x | 1.63x |
| 0.05 | 8 | 20/20 | 20/20 | 0.63x | 1.59x |
| 0.04 | 8 | 20/20 | 20/20 | 0.65x | 1.54x |
| 0.03 | 8 | 20/20 | 20/20 | 0.67x | 1.49x |
| 0.025 | 8 | 20/20 | 20/20 | 0.69x | 1.45x |
| 0.02 | 8 | 20/20 | 20/20 | 0.72x | 1.39x |
| 0.018 | 8 | 20/20 | 20/20 | 0.73x | 1.38x |
| 0.016 | 8 | 20/20 | 20/20 | 0.75x | 1.34x |
| 0.015 | 8 | 20/20 | 20/20 | 0.77x | 1.31x |
| 0.014 | 8 | 20/20 | 20/20 | 0.78x | 1.29x |
| 0.013 | 8 | 20/20 | 20/20 | 0.81x | 1.25x |
| 0.012 | 8 | 20/20 | 20/20 | 0.83x | 1.22x |
| 0.0115 | 8 | 20/20 | 20/20 | 0.86x | 1.18x |
| 0.011 | 8 | 20/20 | 20/20 | 0.95x | 1.09x |
| 0.01075 | 8 | 20/20 | 20/20 | 1.04x | 1.02x |
| 0.0105 | 8 | 20/20 | 20/20 | 1.10x | 0.98x |
| 0.01025 | 16 | 20/20 | 20/20 | 1.72x | 1.25x |
| 0.01 | 16 | 20/20 | 20/20 | 1.92x | 1.14x |

比例是对 paired formal seeds 计算的均值；若任一方法未命中则使用预注册上限的 capped 值，并在 formal_ratios.csv 中标记。

![Depth/work ratios](figures/depth_work_ratios.png)

![Depth/work versus epsilon](figures/depth_work_vs_epsilon.png)

## 0.01 以下 exploratory/censored 结果

这些点不参与 primary 参数选择或主趋势 verdict。未命中不会留空，而是保留 hit rate 和最大预算下界。

| epsilon | NOG hit | ME-DOL hit | NOG capped depth | ME capped depth |
|---:|---:|---:|---:|---:|
| 0.0095 | 20/20 | 20/20 | 418.7 | 1316.4 |
| 0.009 | 20/20 | 12/20 | 451.4 | 2497.8 |
| 0.008 | 18/20 | 0/20 | 659.7 | 3840.0 |
| 0.007 | 2/20 | 0/20 | 950.1 | 3840.0 |
| 0.006 | 0/20 | 0/20 | 962.0 | 3840.0 |
| 0.005 | 0/20 | 0/20 | 962.0 | 3840.0 |
| 0.004 | 0/20 | 0/20 | 962.0 | 3840.0 |
| 0.003 | 0/20 | 0/20 | 962.0 | 3840.0 |
| 0.002 | 0/20 | 0/20 | 962.0 | 3840.0 |

![Hit rate](figures/hit_rate_vs_epsilon.png)

## 理论参照与观测斜率

| metric | theory exponent | observed log-log slope |
|---|---:|---:|
| NOG depth | 1.667 | 0.372 |
| ME-DOL depth | 3.000 | 0.636 |
| NOG work | 3.000 | 0.429 |
| ME-DOL work | 3.000 | 0.636 |

论文的 first-order 结论是 NOG depth 为 O(epsilon^-5/3)、ME-DOL depth 为 O(epsilon^-3)，两者 work 同为 O(epsilon^-3)。本实验采用 pilot-calibrated matched-work batch schedule 来检验有限区间中的方向性，而不是把理论常数当成已知。

## 可复现性与防止事后挑结果

- Pilot seeds 100--104；formal seeds 0--19，严格不重叠。
- NOG batch grid 为 8,16,...,64；只允许 epsilon 变小时 batch 不下降。
- 参数冻结发生在任何 formal partial 生成之前。
- 每个 hit 要求连续两个 high-precision checkpoints 达标。
- 所有 raw pilot/formal 输入的 SHA256 均记录在 frozen_parameters.json 和 analysis_manifest.json。
- 并发上限是 4 个任务 × 8 workers = 32 个 worker processes。

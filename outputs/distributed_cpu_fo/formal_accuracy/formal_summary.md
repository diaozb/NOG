# Step 7C Formal accuracy 审计与汇总

本报告只分析已冻结配置的 formal seeds，不包含任何重新调参或新训练。

## 完整性审计

- 状态：`passed`
- Tasks：`30/30` passed
- Unique configs：`6`
- Formal seeds：`[0, 1, 2, 3, 4]`；pilot seeds 已排除
- Raw checkpoint rows：`1130`
- SFO training work、evaluation work、depth、SHA256、PID/rank、seed、finite metrics 与单调性均逐 task 审计。

## Confirmed-hit 结果

First hit 要求连续 `2` 个 evaluation checkpoints 满足 `stat_proxy <= epsilon`，并把第一个 checkpoint 记为命中位置。均值和 sample std 只在命中 seeds 上计算，hit rate/censoring 单独报告。

| Method | epsilon | hit | Depth mean±std | Total work mean±std | Per-worker work mean±std | Training time mean±std (s) | Final stat mean±std |
|---|---:|---:|---:|---:|---:|---:|---:|
| ME-DOL-FO | 0.011 | 5/5 | 111.60±236.13 | 892.80±1889.03 | 111.60±236.13 | 3.19±6.23 | 0.008439±0.000739 |
| NOG-FO | 0.011 | 5/5 | 15.60±21.47 | 3993.60±5495.36 | 499.20±686.92 | 0.66±0.58 | 0.007686±0.000443 |
| ME-DOL-FO | 0.01 | 5/5 | 213.60±228.69 | 1708.80±1829.55 | 213.60±228.69 | 7.58±8.22 | 0.008261±0.000443 |
| NOG-FO | 0.01 | 5/5 | 34.80±26.29 | 17817.60±13460.83 | 2227.20±1682.60 | 1.38±0.86 | 0.007529±0.000432 |
| ME-DOL-FO | 0.009 | 5/5 | 578.40±218.39 | 4627.20±1747.09 | 578.40±218.39 | 19.91±8.52 | 0.008261±0.000443 |
| NOG-FO | 0.009 | 5/5 | 54.00±0.00 | 13824.00±0.00 | 1728.00±0.00 | 2.50±0.52 | 0.007686±0.000443 |
| ME-DOL-FO | 0.008 | 1/5 | 1164.00±0.00 | 9312.00±0.00 | 1164.00±0.00 | 39.55±0.00 | 0.008261±0.000443 |
| NOG-FO | 0.008 | 5/5 | 67.60±21.47 | 34611.20±10990.72 | 4326.40±1373.84 | 2.44±0.73 | 0.007199±0.000393 |
| ME-DOL-FO | 0.0075 | 3/5 | 3368.00±337.14 | 26944.00±2697.13 | 3368.00±337.14 | 112.88±12.91 | 0.007653±0.000536 |
| NOG-FO | 0.0075 | 4/5 | 94.00±45.96 | 48128.00±23529.73 | 6016.00±2941.22 | 3.26±1.43 | 0.007199±0.000393 |

## NOG-FO / ME-DOL-FO 对比

`Depth improvement = ME-DOL depth / NOG depth`，大于 1 表示 NOG communication depth 更小；`Work ratio = NOG work / ME-DOL work`，大于 1 表示 NOG 使用更多 SFO work。

| epsilon | NOG hit | ME-DOL hit | Full-hit comparison | Depth improvement | Total-work ratio | Per-worker-work ratio | Training-time improvement |
|---:|---:|---:|:---:|---:|---:|---:|---:|
| 0.011 | 100% | 100% | yes | 7.15 | 4.47 | 4.47 | 4.81 |
| 0.01 | 100% | 100% | yes | 6.14 | 10.43 | 10.43 | 5.51 |
| 0.009 | 100% | 100% | yes | 10.71 | 2.99 | 2.99 | 7.96 |
| 0.008 | 100% | 20% | no* | 17.22 | 3.72 | 3.72 | 16.18 |
| 0.0075 | 80% | 60% | no* | 35.83 | 1.79 | 1.79 | 34.59 |

`*` 非 full-hit 行的 ratio 仅基于成功 seeds，存在 censoring bias，不能作为无条件算法比较。

Evaluation work 来自共同 fixed high-precision sample bank，已单独审计；上表 Work 指 training SFO calls，不包含 evaluation calls。

# Step 7：NOG-FO vs ME-DOL-FO Formal Accuracy 最终结果

## 一句话结论

在双方均 `5/5` formal seeds confirmed hit 的 `epsilon = 0.011, 0.010, 0.009` 上，NOG-FO 达到相同 stationarity threshold 所需的 mean communication depth 比 ME-DOL-FO 小 `7.15x, 6.14x, 10.71x`，与论文 Section 5 所预测的 communication advantage 定性一致；但 NOG-FO 的 finite total SFO work 高 `4.47x, 10.43x, 2.99x`。因此当前证据支持 communication-depth advantage，并显示 first-hit training time 更短；它不支持 finite-work advantage，也不能替代 Step 8 的受控 runtime-scaling 结论。

## 实验协议

- Target：`(2 delta, epsilon)`-Goldstein stationarity，其中 `delta=0.1`，即固定 neighborhood 为 `2 delta=0.2`；
- Methods：`NOG-FO` 与 `ME-DOL-FO`；每个 logical worker 是一个真实 CPU process，`m=8`，Gloo exact all-reduce；
- Problem：`SyntheticMaxSinL1`，`d=100, n_data=4096, R=4, lambda=0.001`；
- Formal seeds：`[0,1,2,3,4]`；pilot seeds `[100,101,102]` 严格隔离；
- 每个 `method x epsilon` 的 hyperparameters 只用 pilot seeds 选择并在 formal run 前冻结；
- 使用共同的 fixed high-precision evaluation sample bank；连续两个 checkpoints 满足 `stat_proxy <= epsilon` 才记为 confirmed hit；
- Work 指 training SFO calls，不包含单列审计的 evaluation work。

## 与 Section 5 理论结果的关系

本结果包采用论文当前的 `(2 delta, epsilon)` 定义及以下 SFO 复杂度口径：

| Method | Communication | Work |
|---|---:|---:|
| ME-DOL (SFO) | `O(delta^{-1} epsilon^{-3})` | `O(delta^{-1} epsilon^{-3})` |
| NOG (SFO) | `O(d^{1/3} delta^{-1} epsilon^{-5/3})` | `O(delta^{-1} epsilon^{-3})` |

理论预测 NOG 改善 communication exponent，同时与 ME-DOL 保持相同 asymptotic work order。本实验固定了 `d` 和 `delta`，且只有三个双方 full-hit thresholds，因此只能检验 qualitative depth advantage，不能可靠拟合或宣称验证 `epsilon^{-5/3}` 与 `epsilon^{-3}` 的 asymptotic slopes。实验中的 constant、batch 和 tuned hyperparameters 会影响 finite work；NOG work 高 3–10 倍不否定同阶理论，但也不能被表述为验证了 work 同阶。

## Formal results

下表 mean ± sample std 只对成功 seeds 计算；带 `†` 的行存在 right censoring，不能使用 hit-only ratio 做无条件比较。

| epsilon | NOG hit | ME hit | NOG depth | ME depth | Depth gain | NOG total work | ME total work | NOG/ME work |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.011 | 5/5 | 5/5 | 15.6 ± 21.5 | 111.6 ± 236.1 | 7.15x | 3994 ± 5495 | 893 ± 1889 | 4.47x |
| 0.01 | 5/5 | 5/5 | 34.8 ± 26.3 | 213.6 ± 228.7 | 6.14x | 17818 ± 13461 | 1709 ± 1830 | 10.43x |
| 0.009 | 5/5 | 5/5 | 54.0 ± 0.0 | 578.4 ± 218.4 | 10.71x | 13824 ± 0 | 4627 ± 1747 | 2.99x |
| 0.008 | 5/5 | 1/5† | 67.6 ± 21.5 | 1164.0 ± 0.0† | — | 34611 ± 10991 | 9312 ± 0† | — |
| 0.0075 | 4/5† | 3/5† | 94.0 ± 46.0† | 3368.0 ± 337.1† | — | 48128 ± 23530† | 26944 ± 2697† | — |

`†` `epsilon=0.008`：NOG `5/5`、ME-DOL `1/5`；`epsilon=0.0075`：NOG `4/5`、ME-DOL `3/5`。这些条件均值存在 censoring bias。

## 如何解释结果

1. **Communication depth**：三个 full-hit thresholds 上均有稳定的大幅优势，且在更严格 epsilon 上优势没有消失；这是当前最强、最适合写入论文的证据。
2. **Finite SFO work**：三个可公平比较的 thresholds 上，NOG 分别多使用约 `4.47x, 10.43x, 2.99x` total/per-worker SFO work。由于两种方法都在 `m=8`，total-work ratio 与 per-worker-work ratio 相同。
3. **Training time**：对应三个 full-hit thresholds，formal checkpoint training time 的 ME-DOL/NOG mean ratios 为 `4.81x, 5.51x, 7.96x`。这说明真实 CPU-process 实现中低 depth 已转化为较短训练时间，但它不是受控 systems benchmark；worker scaling、warm-up 和重复计时属于 Step 8。
4. **严格 thresholds**：`epsilon=0.008` 明显有利于 NOG 的 hit rate，但 ME-DOL 只有一个成功 seed；`epsilon=0.0075` 两者都 censored。因此可报告 hit rates，不能用这些行的成功-seed ratios 证明 speedup。
5. **Same stationary point**：这里指相同 problem、`delta` 和 empirical stationarity threshold，不表示两个算法到达完全相同的 parameter vector。

## Frozen configurations

| epsilon | NOG-FO | ME-DOL-FO |
|---:|---|---|
| 0.011 | `M=4, eta=1, smooth_B=4, data_B_total=64`; rounds `960` | `epoch_length=6, theory_multiplier=10`; rounds `1920` |
| 0.01 | `M=4, eta=1, smooth_B=8, data_B_total=64`; rounds `960` | `epoch_length=12, theory_multiplier=10`; rounds `1920` |
| 0.009 | `M=4, eta=1, smooth_B=4, data_B_total=64`; rounds `960` | `epoch_length=12, theory_multiplier=10`; rounds `1920` |
| 0.008 | `M=8, eta=1, smooth_B=8, data_B_total=64`; rounds `960` | `epoch_length=12, theory_multiplier=10`; rounds `1920` |
| 0.0075 | `M=8, eta=1, smooth_B=8, data_B_total=64`; rounds `960` | `epoch_length=24, theory_multiplier=10`; rounds `3840` |

配置随 epsilon 变化是预注册设计：每个 threshold 在独立 pilot seeds 上选择配置；formal seeds 没有参与调参。图中的 trajectory panels 也分别使用对应 frozen config，不能解释为一组 universal hyperparameters 的 epsilon scaling。

## Figures

- [`depth_vs_epsilon.pdf`](../figures/depth_vs_epsilon.pdf)：论文主图候选，最直接展示 communication advantage；
- [`work_vs_epsilon.pdf`](../figures/work_vs_epsilon.pdf)：与主图配套，揭示 finite-work tradeoff；
- [`stat_proxy_vs_depth.pdf`](../figures/stat_proxy_vs_depth.pdf)：五个 epsilon-specific panels 的完整收敛轨迹；
- [`stat_proxy_vs_work.pdf`](../figures/stat_proxy_vs_work.pdf)：相同轨迹按 training SFO work 展示。

Threshold plots 中 filled marker 表示 `5/5` hit，hollow marker 表示 censored conditional mean；所有 error bars/bands 均为 mean ± one sample std。

## 建议写入论文的英文表述

> On the three stationarity tolerances for which both methods achieved a 5/5 confirmed-hit rate, NOG-FO required 6.14--10.71 times fewer communication rounds than ME-DOL-FO. This qualitative advantage is consistent with the improved communication dependence predicted by our theory. NOG-FO used 2.99--10.43 times more finite-sample SFO work, highlighting a practical constant-factor tradeoff that is not captured by the shared asymptotic work order. At the two strictest tolerances, at least one method was right-censored; we therefore report hit rates and omit unconditional speedup ratios for those settings.

这里建议使用 `consistent with`，不要使用 `verifies the epsilon^{-5/3} rate`、`same work in practice` 或在 censored thresholds 上写无条件 `speedup`。

## 当前限制与后续动作

- 只有三个双方 full-hit epsilon，epsilon 范围也较窄，不足以拟合 asymptotic exponent；
- `d` 与 `delta` 固定，当前实验不验证 `d^{1/3}` 或 `delta^{-1}` dependence；
- epsilon-specific tuning 提高了每个 threshold 的实际表现，但不适合用单一 log-log slope 解释理论；
- Synthetic problem 不能替代 real-data generalization evidence；
- Step 8 仍需做预注册的 CPU runtime scaling，才能正式讨论 parallel speedup/efficiency；
- 是否扩展更多 epsilon、delta、dimension 或 ZO/DGFM/DGFM+，建议先把本结果包发给学长确认，不在当前 formal results 上事后追加有利配置。

## 复现

```bash
/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_formal_analysis \
  --config configs/distributed_cpu_fo_pilot.yaml
/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_formal_figures
/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_formal_report
```

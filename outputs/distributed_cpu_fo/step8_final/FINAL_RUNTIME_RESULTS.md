# Step 8：NOG-FO vs ME-DOL-FO CPU Runtime 最终结果

## 一句话结论

Step 7 的 formal accuracy experiment 显示，在双方均 `5/5` confirmed hit 的 `epsilon=0.011,0.010,0.009` 上，NOG-FO 达到相同 empirical stationarity threshold 所需 mean communication depth 少 `6.14--10.71x`；Step 8 的真实 CPU-process repeated benchmark 则显示，NOG-FO 在当前单机 CPU/Gloo 环境下没有获得 positive strong scaling，增加 workers 反而变慢。NOG 的 frozen full-budget median training time 是 ME-DOL 的 `0.539--0.821x`，但两者没有 empirical-work matching，不能把该 wall-clock ratio 解释为 finite-work parity、time-to-epsilon speedup 或 Section 5 asymptotic Work complexity 的验证。

## Step 8 protocol

- Methods：`NOG-FO`、`ME-DOL-FO`；真实 CPU processes + Gloo exact all-reduce；
- Tolerances：`epsilon=[0.010,0.009,0.008]`；worker counts：`m=[1,2,4,8,16,32]`；benchmark seed：`0`；
- 每个 unique config × worker setting 先运行 one-update-unit warm-up，再运行 3 个 measured repeats；
- 24 个 warm-ups 不进入统计；72 个 physical measured runs 展开为 108 个 method-epsilon-worker-repeat rows；
- NOG 每个 epsilon 使用 Step 6 pilot-frozen config、960 rounds；ME-DOL 三个 epsilon 映射到同一个 frozen config、1920 rounds；
- `training_time` 排除 process startup、evaluation 和 serialization；`end_to_end_time` 包含这些开销；
- 每个点报告 3 repeats 的 median 和完整 `[min,max]`，不删除 outliers；
- 该 benchmark 运行完整 frozen budget，不是 first-hit time-to-epsilon。

## Runtime results at m=8

选择 `m=8` 单列，是为了与 Step 7 formal accuracy 的 worker count 对齐；仍然是 full-budget comparison。

| epsilon | Accuracy status | NOG training (s) | ME training (s) | NOG/ME time | NOG/ME end-to-end | NOG/ME work |
|---:|---|---:|---:|---:|---:|---:|
| 0.01 | both 5/5 confirmed hit | 37.96 | 54.10 | 0.702 | 0.728 | 32.067 |
| 0.009 | both 5/5 confirmed hit | 37.93 | 54.10 | 0.701 | 0.726 | 16.033 |
| 0.008 | NOG 5/5; ME 1/5† | 34.49 | 54.10 | 0.638 | 0.664 | 32.067 |

`†` `epsilon=0.008` 的 ME-DOL formal accuracy 为 `1/5`，所以该行只能说明完整预算的执行时间，不能说明达到 epsilon 的 runtime。

## Scaling result

- 两个 methods、三个 epsilon 的最低 median training time 全部出现在 `m=1`；
- NOG 是 fixed-total-work workload，但 `m=32` 相对 `m=1` 的 training speedup 仅为 `0.118--0.139x`，即不是加速而是约 `7.2--8.5x` slowdown；
- ME-DOL 每个 worker 每轮各执行一个 SFO call，total work 随 m 线性增长，因此 ME-DOL 的 `T1/Tm` 不能解释为 strong-scaling efficiency；
- median communication/training-time fraction 从 `m=1` 的约 `0.9--1.3%` 增至 `m=32` 的约 `13.7--14.8%`；
- 结果只说明当前 single-host CPU process/Gloo implementation；不能外推为多机、真实网络或 GPU collective 的 scaling behavior。

## Full-budget method comparison

跨全部 epsilon 和 worker settings，NOG/ME-DOL median training-time ratio 为 `0.539--0.821`，end-to-end ratio 为 `0.553--0.828`。因此 NOG 确实更快完成了各自的 frozen full budget，但对应 NOG/ME-DOL total SFO-work ratio 为 `4.008--256.533`。

这个组合结果并不矛盾：wall-clock cost 取决于 update structure、communication depth、batch vectorization、process synchronization 和 Python/PyTorch implementation，而 SFO work 只计 oracle samples。当前结果支持的是 implementation-level full-budget runtime difference；它不支持相同 finite work 下 NOG 更快。

## 与 Step 7 和 Section 5 的联合解释

| Question | Evidence | Conclusion |
|---|---|---|
| 达到相同 empirical threshold 的 depth 是否更小？ | Step 7，三个 mutual-full-hit thresholds | 是，NOG mean depth 少 `6.14--10.71x`；与 predicted communication advantage 定性一致 |
| finite SFO work 是否相近或更小？ | Step 7 first-hit work；Step 8 full-budget work | 否，当前 tuned configs 下 NOG 使用更多 finite SFO work |
| 单机增加 CPU workers 是否带来 acceleration？ | Step 8 repeated scaling benchmark | 否，两种方法均在 `m=1` 最快 |
| NOG 是否更快完成 frozen full budget？ | Step 8 repeated timing | 是，但 workload 未 work matched，且不是 time-to-epsilon |
| 是否验证了 `epsilon^{-5/3}`、`epsilon^{-3}`、`d^{1/3}` 或 `delta^{-1}`？ | epsilon/d/delta 覆盖不足 | 否，只能使用 `consistent with` 的 qualitative wording |

## Figures 与使用建议

### 推荐作为论文主结果

- [`depth_vs_epsilon.pdf`](../figures/depth_vs_epsilon.pdf)：Step 7 communication-depth 主图；
- [`work_vs_epsilon.pdf`](../figures/work_vs_epsilon.pdf)：必须与 depth evidence 配套，呈现 finite-work tradeoff。

### 可放 supplement 或发给学长

- [`runtime_vs_workers.pdf`](../runtime/figures/runtime_vs_workers.pdf)：完整 timing、range 和 long-tail behavior；
- [`full_budget_method_comparison.pdf`](../runtime/figures/full_budget_method_comparison.pdf)：runtime ratio 与 work mismatch 必须成对展示；
- [`communication_fraction_vs_workers.pdf`](../runtime/figures/communication_fraction_vs_workers.pdf)：systems diagnostic。

### 仅作为 negative-scaling diagnostic

- [`nog_strong_scaling_speedup.pdf`](../runtime/figures/nog_strong_scaling_speedup.pdf)：不能作为 positive scaling evidence；如果篇幅有限，不建议放正文。

## 建议写入论文的英文表述

> On the three tolerances for which both methods achieved a 5/5 confirmed-hit rate, NOG-FO required 6.14--10.71 times fewer communication rounds than ME-DOL-FO, a qualitative advantage consistent with the improved communication dependence predicted by our theory. In a separate single-host CPU/Gloo benchmark, NOG-FO completed its pilot-frozen full budget in 0.539--0.821 times the median training time of ME-DOL-FO. This benchmark did not exhibit positive process-level strong scaling, and the frozen workloads were not matched by empirical SFO work; accordingly, we do not interpret the timing ratio as a work-matched time-to-stationarity speedup or as verification of the asymptotic work complexity.

## 不应使用的 claims

- `NOG scales linearly with the number of workers`；
- `NOG reaches epsilon 1.2--1.9x faster in the runtime benchmark`；
- `NOG and ME-DOL use the same work in practice`；
- `The experiments verify the epsilon^{-5/3} communication rate`；
- 把 `epsilon=0.008` 的 ME-DOL full-budget timing 写成 uncensored time-to-epsilon；
- 只展示 runtime ratio 而隐藏相邻的 work-ratio panel。

## Reproduction

```bash
/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_runtime_analysis
/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_runtime_figures
/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_runtime_report
```

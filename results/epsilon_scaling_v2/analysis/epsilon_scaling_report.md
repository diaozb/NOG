# Wide-epsilon NOG-FO vs ME-DOL-FO report

## Protocol

- epsilon: 0.2 down to 0.002 (17 values; log-scale figures)
- primary: m=8, 20 formal seeds; pilot seeds 100-104 are disjoint
- non-hits remain right-censored; capped mean, KM restricted mean, and lower bounds are always reported
- robustness: m in {1,2,4,8}, epsilon in {0.1,0.01,0.005}, 10 seeds

## Selected primary results

| epsilon | NOG hits | ME-DOL hits | capped depth ME/NOG | status |
|---:|---:|---:|---:|---|
| 0.1 | 20/20 | 20/20 | 1 | full |
| 0.03 | 20/20 | 20/20 | 1 | full |
| 0.01 | 20/20 | 14/20 | 4.9 | censored (14/20 paired) |
| 0.009 | 20/20 | 20/20 | 12.4 | full |
| 0.008 | 20/20 | 20/20 | 15.3 | full |
| 0.007 | 19/20 | 11/20 | 5.9 | censored (11/20 paired) |
| 0.006 | 7/20 | 0/20 | 1.3 | censored (0/20 paired) |
| 0.005 | 0/20 | 0/20 | 1 | censored (0/20 paired) |
| 0.002 | 0/20 | 0/20 | 1 | censored (0/20 paired) |

## Hypothesis decisions

| Claim | Decision | Evidence |
|---|---|---|
| NOG depth advantage grows for demanding epsilon | **Partially supported** | At epsilon=0.009 and 0.008 both methods hit 20/20 and depth ME/NOG is about 12.4 and 15.3. The full trend is not monotone because frozen region changes and censoring create discontinuities. |
| NOG/ME-DOL work ratio remains approximately constant | **Not supported** | The all-epsilon capped work-ratio coefficient of variation is about 1.11. |
| ME-DOL non-hits are reported without blanks | **Supported** | Every non-hit contributes a censoring limit; capped and KM summaries are always present. |
| Depth advantage is robust to worker count | **Supported at epsilon=0.01** | NOG hits 10/10 for every m; ME-DOL hits 0/10, 4/10, 7/10, 8/10. Capped depth ME/NOG is 6.84, 7.26, 5.39, 4.26. |
| epsilon=0.005 has finite time-to-hit ratios | **Not supported** | Neither method hits for any worker count; only censoring lower bounds are valid. |

## Recommended wording

> Across 20 formal seeds, NOG-FO shows a substantial empirical communication-depth advantage at the fully observed thresholds epsilon=0.009 and 0.008, while maintaining higher hit rates at several more demanding thresholds. Worker-count robustness supports the advantage at epsilon=0.01. Results below the observed hitting range are reported as right-censored lower bounds rather than omitted or treated as hits.

Do not claim constant work ratio, monotone growth across all 17 epsilon values, or hits at epsilon<=0.005.

## Figures

- [formal_hit_rate_vs_epsilon.png](figures/formal_hit_rate_vs_epsilon.png)
- [formal_depth_work_ratios.png](figures/formal_depth_work_ratios.png)
- [formal_capped_depth_work.png](figures/formal_capped_depth_work.png)
- [robustness_hit_rate_by_workers.png](figures/robustness_hit_rate_by_workers.png)
- [robustness_depth_ratio_by_workers.png](figures/robustness_depth_ratio_by_workers.png)

## 中文结论

结果支持 NOG 在若干可观测 epsilon 上具有明显通信深度优势，且 epsilon=0.01 的优势对 m=1,2,4,8 稳健。结果不支持 work 比例在完整 epsilon 范围内基本不变；epsilon<=0.005 时两种方法均未命中，必须按删失下界报告。

学长好，FO 的真实 CPU-process runtime/scaling benchmark 也完成并审计好了。我们在 epsilon=0.010/0.009/0.008、m=1/2/4/8/16/32 上，每个 setting 做了 warm-up 和 3 次正式重复，统计 median[min,max]，没有删除 outlier。

这台单机 CPU + Gloo 环境下没有观察到 positive scaling：NOG 和 ME-DOL 都是 m=1 最快，NOG 到 m=32 时相对 m=1 只有 0.118--0.139x speedup（实际是变慢约 7--8.5 倍）。所以不能说增加 CPU workers 后 NOG 跑得更快，也不能把这个结果外推到多机/GPU。

另一方面，在相同 m 下，NOG 完成各自 frozen full budget 的 median training time 是 ME-DOL 的 0.539--0.821x，确实更短。但两种方法没有按 empirical SFO work 匹配，NOG/ME-DOL full-budget work ratio 是 4.008--256.533x；epsilon=0.008 的 ME-DOL accuracy 还是 censored。因此这部分只能写 implementation-level full-budget runtime difference，不能写成 work-matched time-to-epsilon speedup。

和 Step 7 合起来，最稳妥的论文结论仍然是：在双方 5/5 命中的三个 epsilon 上，NOG 达到相同 threshold 的 communication depth 少 6.14--10.71x，与理论 communication advantage 定性一致；finite SFO work 没有显示优势。建议正文优先放 depth 图并配套 work 图，runtime/scaling 图放 supplement 或先作为内部 diagnostic。
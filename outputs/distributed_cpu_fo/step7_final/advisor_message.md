学长好，FO 的真实 CPU-process formal experiment 已经完成并审计过了。我们固定 delta=0.1、m=8，用 5 个 formal seeds，对每个 epsilon 使用独立 pilot seeds 选出的 frozen config；命中要求连续两个 evaluation checkpoints 都低于 epsilon。

在 epsilon=0.011/0.010/0.009 上，两种方法都是 5/5 命中。NOG-FO 相比 ME-DOL-FO 的平均 communication depth 分别减少约 7.15x、6.14x、10.71x；对应实际 training time 快约 4.81x、5.51x、7.96x。这个结果定性支持论文里 NOG communication complexity 更优的结论。

不过 NOG 的 finite total SFO work 在这三组上分别高约 4.47x、10.43x、2.99x，所以目前不能说实际 work 区别不大，只能说理论上 asymptotic order 相同、实验中 constant-factor gap 较明显。epsilon=0.008 时 NOG 5/5、ME-DOL 1/5；epsilon=0.0075 时 NOG 4/5、ME-DOL 3/5，这两组有 censoring，我没有用成功 seeds 的 ratio 做无条件 speedup 结论。

表格、四组 PNG/PDF 图、可复现配置和限制说明都已整理好。建议论文里写“consistent with the predicted communication advantage”，暂时不要写“verified epsilon^{-5/3} scaling”。如果您认可这个方向，下一步可以继续做 CPU worker scaling/runtime benchmark，或者再决定是否补更多 epsilon/delta/dimension。
# NOG 宽 epsilon 理论验证计划与完成记录

## 目标

针对旧实验 epsilon 过窄、只有 5 次重复、ME-DOL non-hit 留空以及 0.010 异常跳变，
重新建立一组可复现、可审计的 NOG-FO vs ME-DOL-FO 实验。主要检验：epsilon 变小时，
`ME-DOL/NOG depth` 是否总体上升，同时 `NOG/ME-DOL work` 是否保持常数量级。

原则：pilot 可以用于选参数，formal seeds 不能参与选择；不删除异常 epsilon、non-hit
seed 或不利结果；精确渐进指数未被数据支持时不得宣称已经验证。

## 冻结协议

- Primary epsilon：27 点，从 `0.2` 到 `0.01`；在 0.011 到 0.01 区间加密。
- Exploratory/censored epsilon：`0.0095,0.009,0.008,...,0.002`。
- Pilot seeds：`100--104`；formal seeds：`0--19`；两者严格不相交。
- Worker count：`m=8`；真实 CPU processes + Gloo exact all-reduce。
- 并发上限：4 physical tasks，每个 8 workers，总计不超过 32 worker processes。
- Confirmed hit：连续两个 high-precision checkpoints 的 stat proxy 不高于 epsilon。
- Evaluation：固定 bank，`256 x 512`；evaluation work 不混入 training SFO work。
- NOG：`M=2, eta=1, smooth_B=1, rounds=960, eval_every=2`。
- ME-DOL：`epoch_length=6, theory_multiplier=100, rounds=3840, eval_every=6`。
- NOG batch grid：`8,16,24,32,40,48,56,64`。
- Batch 选择：5/5 pilot hits；epsilon 变小时 batch 不下降；最小化
  `sum(abs(log(NOG_work/ME_work)))`，另加固定 switch penalty 0.05。

正式成功门槛在 formal 运行前写入 freeze：每个 primary method/epsilon 至少 18/20 hits；
depth-ratio Spearman 至少 0.7 且末端大于起点；work ratio 在 0.5--2.0 内且 CV 不超过
0.25。

## Step 8：问题诊断、pilot 与参数冻结

### Step 8A：恢复与旧结果核验 — 已完成

- 确认昨晚的 NOG/ME-DOL high-resolution pilot 各 5/5 完整落盘。
- 定位旧 0.010 跳变：初始 stationarity proxy 接近 threshold，加上 5 seeds 和稀疏
  checkpoints，产生了配置边界效应。

### Step 8B：论文理论参数核对 — 已完成

- 核对 Theorem 4.1/4.3：first-order NOG 使用 epsilon-dependent oracle variance/batch；
  理论 depth 为 `epsilon^(-5/3)`，work 为 `epsilon^-3`。
- 明确固定 batch pilot 只能检验有限区间趋势，不能直接宣称 work exponent 相同。

### Step 8C：batch-grid calibration — 已完成

- 8 batches x 5 seeds = 40/40 tasks 完成，0 失败。
- 选择仅依赖 pilot seeds；保存所有输入 SHA256 和完整 `pilot_calibration.csv`。

### Step 8D：freeze — 已完成

- 27 点 schedule 冻结为 batch 8 和 16。
- 冻结前确认 formal 目录没有 partial；之后不允许根据 formal 结果重选参数。

## Step 9：独立正式实验

### Step 9A：runner 与审计实现 — 已完成

- 实现 ThreadPool 调度、32-process 硬上限、atomic partial、resume 和 failure record。
- 实现 SHA256、fingerprint、rank、trajectory、depth/work formula 审计。

### Step 9B：20-seed formal — 已完成

- NOG batch 8：20/20 tasks。
- NOG batch 16：20/20 tasks。
- ME-DOL extended budget：20/20 tasks。
- 合计 60/60，0 失败；结果审计 60/60 passed。

## Step 10：统计、图表、报告与结果包

### Step 10A：formal statistics — 已完成

- 27/27 primary epsilon 上双方均为 20/20 hits。
- `ME-DOL/NOG depth` Spearman `rho=1.000`，从 `0.488x` 上升到 `1.919x`。
- `NOG/ME-DOL work` 均值 `1.489x`，CV `0.208`，范围 `0.980x--2.049x`。
- Depth 趋势门槛通过，hit 门槛通过；work 门槛因 epsilon=0.2 超出 2.0 约 2.5%
  而未通过。因此严格总 verdict 为 `not fully supported`。
- 0.01 以下 9 个点保留 hit rate 和 capped depth/work，不留空、不删除 non-hit。

### Step 10B：文档与 package — 已完成

- 生成 3 组 PNG/PDF 图、per-seed/summary/ratio CSV、trend JSON 和完整中文报告。
- 生成 `results/theory_validation_v4`，包含 19 个带 SHA256 manifest 的文件。
- README 更新为 v4 当前结果，旧结果明确标注为历史基线。

## Step 11：验证与发布

### Step 11A：完整验证 — 已完成

- 完整测试：67 passed，8 subtests passed。
- 正式结果审计：60/60 passed。
- Package 19、freeze 46、analysis 63 个 manifest entries 的 SHA256 全部复核通过。
- README、正式报告和 REPRODUCE 的本地链接检查无缺失；`git diff --check` 通过。

### Step 11B：GitHub 发布 — 已完成

- 仅提交 final v4 config、代码、测试、README/plan 和紧凑结果包；raw trajectories 保持
  在本机并由 `.gitignore` 排除。
- 主提交 `8dba5af` 已推送到 `origin/agent/theory-validation-retest`。
- Draft PR 已创建：`https://github.com/diaozb/NOG/pull/3`。
- 未自动 merge，保留给用户/导师审阅。

## 最终结论口径

可以写：在固定 d、delta 和同一 problem family 的宽 epsilon、20-seed 实验中，NOG 的
相对 depth 表现随目标精度收紧而严格改善；work ratio 保持在约 1--2 倍常数量级，且
formal 结果对 seed 噪声稳定。

不应写：实验精确验证了 `epsilon^-5/3` 或 `epsilon^-3`；NOG 在所有 epsilon 上都优于
ME-DOL；work ratio 完全不变；未命中点可以忽略。

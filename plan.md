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


## Step 12：`epsilon < 0.01` 非对称探索 — 已由 Step 13 取代

### Step 12A：现有低 epsilon 诊断 — 已完成

- v4 的 `0.0095, 0.009, 0.008, ...` 是固定最大 batch 和预算下的
  exploratory/censored 结果，不足以验证完整 ratio 趋势。
- `epsilon=0.0095` 的现有 20-seed depth ratio 约 `3.14x`，work ratio 约
  `0.77x`；`epsilon <= 0.009` 时 ME-DOL 单样本 oracle 出现方差平台。

### Step 12B：低 epsilon 冻结前协议 — 已完成

- 21 个预注册阈值从 `0.01000` 到 `0.00500`，间隔 `0.00025`；同一轨迹同时判断
  所有阈值，因此点数加密不会线性增加训练时间。
- Pilot seeds `100--104` 与 formal seeds `0--19` 严格分离。
- ME-DOL 在 pilot 中比较 epoch、multiplier 和明确计入 work 的每 worker mini-batch；
  选择最长的双方 5/5 命中且 work 可匹配连续区间，只冻结一个全局配置。
- NOG 固定 `M=2, eta=1, smooth_B=1`，batch 随 epsilon 变小不得下降，通过 pilot
  做 matched-work 动态规划。
- 若 formal 相邻 depth ratio 下降超过 20%，不重新选参数：插入 `0.000125` 中点并
  用冻结配置追加 seeds `20--29`，合并全部 seeds 报告。
- 最多 4 个并行任务、每任务 8 workers，总计不超过 32 CPU worker processes。

### Step 12C：pilot — 已停止，不进入正式结论

- ME-DOL 单样本 7680-round pilot 只能稳定覆盖到 `epsilon=0.00900`，因此在 freeze
  前按协议加入可审计 mini-batch 候选；已完成任务继续通过 SHA/fingerprint 缓存复用。

### Step 12D：正式发布 — 不执行（由 Step 13 对称协议替代）


## Step 13：对称低 epsilon 验证（方案 B）— 已完成

### Step 13A：协议重新冻结 — 已完成

- Step 12 保留为探索数据，不进入正式结论；原因是其 NOG 与 ME-DOL 调参范围不完全对称。
- 预注册 25 个阈值：`0.01000` 到 `0.00400`，统一间隔 `0.00025`。
- 双方各 6 个算法参数候选、相同 pilot seeds `110--114`、相同总 batch 128、
  相同 15360 training rounds。
- 双方使用相同总 batch 网格 `32,64,96,128,192,256`，独立选择 5/5 hit 后
  first-hit work 最小的非递减 schedule。
- 统一每 24 iterations 评估；NOG 两次初始化通信明确计入 depth，因此实际最大记录
  depth 为 NOG 15362、ME-DOL 15360。
- 正式 seeds 为 `20--39`，异常确认 seeds 为 `40--49`，均与 pilot 和 v4 formal 隔离。

### Step 13B：真实进程验证与对称 pilot — 已完成

- 8-process smoke test 已通过；双方 checkpoint iterations 均为 24、48。
- algorithm pilot 60/60、batch pilot 60/60、formal 200/200，均为 0 failures。
- 自动冻结 NOG `M=4, eta=2`；ME-DOL `H=12, multiplier=200`。
- 最多 4 个并行任务，每任务 8 workers，总计不超过 32 worker processes。

### Step 13C：formal、统计与报告 — 已完成

- Base formal 已通过 200/200 artifact/work 审计。
- 预注册 schedule 在 0.00750 到 0.00725 出现超过 20% 的相邻 depth-ratio 下降，
  已按冻结协议追加 seeds 40--49 和中点 0.007375；不重新选择参数。
- 固定共同 batch 诊断作为补充切片，主 verdict 仍由预注册 schedule 决定。
- Extra formal 100/100、0 failures，artifact/work 审计 100/100 passed；base 与 extra
  合计 300 条正式/确认轨迹。
- 25 点上双方均为 30/30 hits；schedule depth ratio 从 0.462x 到 1.087x，
  Spearman 0.373；work ratio 均值 1.456x、范围 0.799--2.176x、CV 0.348。
- 严格 verdict 为 not fully supported；固定 batch 诊断和不利主结果同时报告。
- 已生成三组 PNG/PDF、完整 CSV/JSON、中文报告和 22 项 hash-verified 结果包。

### Step 13D：完整测试、README 与 GitHub 发布 — 已完成

- README 已加入 v5 全 25 点 absolute depth/work、比例、参数、复现和严格 verdict。
- 完整测试 71 passed、8 subtests passed；v5 专项 4 passed。
- Base/extra 审计分别 200/200 和 100/100 passed。
- 发布提交：`df2c956`（分支 `agent/low-epsilon-v5`）。
- GitHub 草稿 PR：<https://github.com/diaozb/NOG/pull/5>。
- 最新 freeze/analysis/package 依赖链共 455 个 SHA256 校验通过；22 项结果包、
  文档链接、脚本语法和 diff 检查通过。


## Step 14：v6 逐 epsilon 联合重调参 — 进行中

### Step 14A：审计 v5 单参数冻结 — 已完成

- v5 先在共同 batch 128 上选一个覆盖全区间的算法参数，再单独选择 batch；这不是
  `algorithm parameter × batch` 的联合最优。
- NOG `M=2, eta=1` 在 pilot `epsilon=0.010` 的平均 depth 为 170，优于最终冻结
  `M=4, eta=2` 的 362；前者因为不能覆盖最小 epsilon 而被全局淘汰。
- ME-DOL 也存在区域最优切换，因此 v5 的全区间固定参数不能继续解释为逐 epsilon 最优。

### Step 14B：v6 协议重新冻结 — 进行中

- 25 个 epsilon 保持 `0.01000--0.00400`，联合搜索算法参数与 batch。
- screen seeds `210--212`、confirmation pilot seeds `213--214`、formal seeds
  `50--69` 完全隔离。
- pilot 最大 depth 7680，formal 最大 depth 15360，统一每 24 rounds 评估；并发仍为
  4 tasks × 8 workers，不超过 32 worker processes。
- 同时冻结 `work-optimal` 与 `depth-optimal` 两套 Pareto 口径，不用正式比例反向挑参数。

### Step 14C：联合 pilot 与边界检查 — 待执行

### Step 14D：pilot-only 逐 epsilon 冻结 — 待执行

### Step 14E：全新 formal 重测、审计与统计 — 待执行

### Step 14F：README、结果包与 GitHub 发布 — 待执行

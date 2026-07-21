# NOG distributed 实验计划

## 0. 当前执行状态（2026-07-15）

第一阶段 single-process simulation 已完成：六个 algorithm variants、correctness checks、独立 pilot tuning 和 5-seed formal runs 共完成 `60/60` tasks，生成 `1,260` 行 checkpoint records；5 个 unit tests 和全部 work/depth accounting audit 均通过。

根据 `Wechat picture 2` 中与学长的最新讨论，下一阶段改为真实 CPU Process FO experiment：

- 每个 logical worker 对应一个独立 CPU process；
- 第一优先级只比较 `NOG-FO` 和 `ME-DOL-FO`；
- 对多个 stationarity tolerance `epsilon` 独立选择配置；
- NOG 不再固定 `M=12, eta=0.3`，而是在 pilot seeds 上运行多组 `(M, eta)`；
- 同时报告 first-hit communication depth、SFO work、真实 CPU training time 和 end-to-end time；
- correctness/profile 已测试 `m=[1,2,4,8,16,32,64]`；后续正式 grid 使用 `m=[1,2,4,8,16,32]`；
- 用户已明确批准第二阶段计划；
- Step 1 CPU Process/Gloo common layer 已完成，2-process smoke check 和仓库全部 7 个 unit tests 均通过；
- Step 2 NOG-FO/ME-DOL-FO real-process migration 已完成，2-process short-trajectory equivalence 与仓库全部 8 个 unit tests 均通过；
- Step 3 atomic partial/resume/manifest/failure cleanup 已完成，故障注入与仓库全部 10 个 tests 均通过；
- Step 4 m=1/2/8 CPU correctness matrix 已完成，6/6 tasks 和全部 invariants 通过，重复执行 6/6 tasks 均安全 resume；
- Step 5 12/48/96-round profile 已完成，42/42 tasks 成功；m=64 可运行但超过 32-CPU cgroup quota 且持续慢于 m=32，因此正式上限冻结为 32；
- 当前没有启动 pilot 或 formal experiment。

第一阶段结果仍保存在：

- `outputs/distributed/distributed_baselines_d100/pilot/config_selected.yaml`
- `outputs/distributed/distributed_baselines_d100/formal/results.csv`
- `outputs/distributed/distributed_baselines_d100/formal/summary.md`
- `outputs/distributed/distributed_baselines_d100/formal/`

## A. 第二阶段：真实 CPU Process FO 实验

### A.1 目标与理论变量

论文目标是找到 `(2 delta, epsilon)`-Goldstein stationary point：

- `delta` 是 Goldstein neighborhood / smoothing radius；
- `epsilon` 是 stationarity norm tolerance；
- 当前 problem 固定 `delta=0.1`，所以不同 empirical thresholds 近似对应 `(0.2, epsilon)`；
- 本阶段主要改变 `epsilon`，因为 Section 5 的 communication improvement 是
  `epsilon^{-5/3}` 对比 `epsilon^{-3}`，而不是不同的 `delta` exponent。

本阶段回答三个问题：

1. 对同一个 empirical `epsilon` threshold，NOG-FO 是否比 ME-DOL-FO 使用更小的 communication depth？
2. 在达到相同 threshold 时，两者的 total/per-worker SFO work 差多少，能否形成 depth/work Pareto improvement？
3. 将 local oracle 真实分配到多个 CPU processes 后，NOG-FO 是否能把较大的 oracle batch 转化为实际 wall-clock speedup？

不把“达到同一个 stationary point”理解成到达完全相同的参数向量，而是达到同一个 problem、`delta` 和 stationarity criterion。

### A.2 本阶段范围

包含：

- `NOG-FO`
- `ME-DOL-FO`
- `d=100, n_data=4096, R=4, lambda=0.001, delta=0.1`
- complete-graph exact averaging / all-reduce
- CPU Process correctness、pilot tuning、formal accuracy 和 runtime scaling

暂不包含：

- NOG-ZO、ME-DOL-ZO、DGFM、DGFM+
- 多 GPU / DDP
- ring、mesh 或其他 sparse topology
- 多个 `delta` 或多个 dimension `d`
- CIFAR 或新的实际数据集

ZO 的 noise、batch 和 step-size 问题留到 FO 结果通过验收并与学长再次确认之后。

### A.3 CPU Process 执行模型

计划使用 `torch.multiprocessing` + `torch.distributed` 的 CPU/Gloo backend：

- 一个 worker 对应一个 OS process 和一个 rank；
- 每个 rank 只保存自己的 deterministic data shard、local RNG 和 local optimizer state；
- NOG 的 local oracle 先在各 rank 独立计算，再用 `all_reduce(SUM)/m` 得到 exact mean；
- ME-DOL 将可合并的 action/state payload 合并后执行一次 collective，对应一个 communication depth；
- rank 0 只负责调度、统一 evaluation 和结果写盘，不替其他 workers 计算 local training oracle；
- 每个 process 设置 `torch.set_num_threads(1)`，并设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，避免 `m` processes 各自再开启大量线程；
- 使用动态 free port 和异常清理，任一 rank 失败时整组 task 标记失败，不保留不完整 formal result；
- 记录 CPU model、NUMA information、process count、thread settings、backend 和 hostname。

correctness/profile worker grid：

```text
m = [1, 2, 4, 8, 16, 32, 64]
```

机器和 affinity 可见 128 logical CPUs，但实测 cgroup quota 为 32 CPUs。Step 5 中 m=64 能运行，但在 48/96 rounds 上两种方法都持续慢于 m=32，且一次 96-round ME-DOL task 超时后才在 retry 成功。因此保留全部 m=64 profile/failure evidence，后续 pilot/formal/runtime grid 冻结为：

```text
m = [1, 2, 4, 8, 16, 32]
```

### A.4 与旧 simulation 的等价性要求

CPU Process runner 必须首先与现有 single-process simulator 做数值和计数对照：

- 相同 problem/partition/method seeds；
- 相同 initial point；
- 相同 local samples 和 smoothing samples；
- 相同 all-reduce mean；
- 相同 output/evaluation point；
- 相同 communication depth；
- 相同 total/per-worker training work；
- 浮点 reduction 顺序可能不同，因此 trajectory 采用 `allclose`，不要求逐 bit 一致。

只有短程等价性测试通过后才能进入 pilot。

## B. 参数选择与 epsilon protocol

### B.1 Candidate epsilon

预声明 candidate thresholds：

```text
epsilon_candidates = [0.011, 0.010, 0.009, 0.008, 0.0075]
```

它们都使用同一个 `delta=0.1`，近似对应：

```text
(2 delta, epsilon) =
(0.2, 0.011), (0.2, 0.010), (0.2, 0.009),
(0.2, 0.008), (0.2, 0.0075)
```

这些值比第一阶段覆盖更明确的 epsilon scaling。Pilot 可以判断某个 threshold 是否在最大预算内不可达，但 formal threshold list 在 formal seeds 运行前冻结，不能根据 formal results 删除不利 threshold。

Evaluation 应使用共同、固定的 high-precision evaluation sample bank，避免不同 method/checkpoint 的 Monte Carlo noise 改变 first-hit 判断。Training RNG 与 evaluation RNG 继续隔离。

### B.2 NOG-FO pilot grid

不直接使用理论公式给出的常数，也不继续只固定 `M=12, eta=0.3`。

第一阶段 coarse grid：

```yaml
M:   [4, 8, 12, 16, 24]
eta: [0.03, 0.1, 0.3, 1.0]
```

所有 candidate rounds 必须同时能被对应 `M` 整除，最大 budgets 使用 `960, 1920, 3840`。预算按阶段递进：先运行到 `960`，只有尚未确认达标或仍位于 Pareto 前沿的 candidate 才延长到 `1920` 和 `3840`，避免每组配置都自动跑满。

为了处理第一阶段 NOG work 过大的问题，不能只选择最低 depth 而忽略 batch。采用两阶段 NOG pilot：

1. 在中等 batch preset（`smooth_B=2, data_B_total=64`）下扫描全部 20 组 `(M,eta)`；
2. 对每个 epsilon 的 top-3 non-dominated configurations，再测试
   `smooth_B=[1,2,4,8]`，保持 `data_B_total=64`，使所有
   `m<=64` 都至少有一个 local data sample。

这样 NOG 每次 global SFO aggregation 的 total work 分别为：

```text
64, 128, 256, 512 SFO calls
```

而不是永远固定为第一阶段的 512。

如果 `eta=1.0` 数值不稳定，只将其记录为 failed candidate；不能在 formal seeds 上临时换值。若 best eta 位于 grid 边界，允许在 pilot 阶段增加一个相邻值，但必须把扩展及理由写入 pilot manifest。

### B.3 ME-DOL-FO 的公平 tuning

不能只调 NOG 而让 baseline 固定。ME-DOL-FO 使用相近的 pilot budget：

```yaml
epoch_length:     [6, 12, 24]
theory_multiplier: [0.3, 1.0, 3.0, 10.0]
```

总计 12 个 primary candidates。若实现中增加 local mini-batch 选项，则其 batch candidate 数必须计入 tuning budget，并同步报告真实 SFO work。

### B.4 Pilot seeds 与逐 epsilon 选择

- Pilot seeds：`[100,101,102]`
- Formal seeds：`[0,1,2,3,4]`
- 两组严格隔离；
- 参数选择参考 worker count：`m_ref=8`；
- 每个 epsilon 可以得到不同的 NOG/ME-DOL frozen config；
- formal seeds 不参与任何 grid expansion、threshold selection 或 config selection。

每个 `method x epsilon` 的选择规则：

1. 首先最大化 pilot hit rate；
2. 对 full-hit candidates 计算 first-hit depth、total work 和 training time 的 Pareto frontier；
3. 排除 first-hit work 明显异常、只依赖巨大 batch 换取低 depth 的 dominated candidates；
4. 在非 dominated candidates 中，以 median training time 为主指标、depth 为 secondary tie-breaker；
5. 同时保留 min-depth 和 min-work candidates 供审计，不只保存一个 best row；
6. 若没有 candidate 在所有 pilot seeds 上命中，则选择最高 hit rate 后 final stat_proxy 最低者，但将该 epsilon 标为 pilot-infeasible/censored，不声称 speedup。

正式配置中必须记录：

```text
epsilon -> method -> selected parameters
pilot score / hit rate / depth / work / time
candidate grid hash
selection rule version
```

## C. 正式实验矩阵

### C.1 Accuracy/depth/work study

在 `m_ref=8` 上运行：

```text
methods       = [NOG-FO, ME-DOL-FO]
epsilons      = [0.011, 0.010, 0.009, 0.008, 0.0075]
formal_seeds  = [0, 1, 2, 3, 4]
config        = 每个 method/epsilon 的 frozen pilot config
```

first hit 使用 confirmed-hit 规则，避免把单次噪声 crossing 当成收敛：

- 只有连续两个 evaluation checkpoints 都满足 `stationarity <= epsilon` 才视为 confirmed hit；
- 这两个 checkpoints 中的第一个记为 first-hit depth；
- 若未 confirmed hit，则运行到统一 max rounds/work budget。

正式结果必须同时给出：

- hit rate；
- first-hit depth mean/std；
- first-hit total/per-worker work mean/std；
- final stat_proxy；
- right-censored seeds；
- depth vs epsilon 的 log-log plot；
- work vs epsilon 的 log-log plot；
- 每个 epsilon 的 depth/work Pareto table。

不能只展示 NOG 命中的 thresholds，也不能只对成功 seeds 报均值而隐藏 hit rate。

### C.2 CPU scaling/runtime study

默认对三个代表性 thresholds：

```text
epsilon_runtime = [0.010, 0.009, 0.008]
m               = [1, 2, 4, 8, 16, 32]
methods         = [NOG-FO, ME-DOL-FO]
```

使用对应 epsilon 的 frozen config。Runtime benchmark 使用固定 problem/method seed 保持计算轨迹一致；每个 setting：

1. 执行一个不计入结果的 short warm-up；
2. 正式重复 3 次；
3. 报告 median、min、max，主表使用 median；
4. run order 在 method 间交错，避免系统负载随时间产生单向偏差。

Step 5 显示 runtime matrix 成本不可忽略，因此保持上述三个预注册 epsilon，不扩展到全部五个。m=64 只保留为 profile diagnostic，不进入正式重复计时。

### C.3 计时定义

同时记录以下时间：

1. `training_time`
   - barrier 后开始；
   - 排除 process startup、problem/data generation、evaluation 和 CSV 写盘；
   - 所有 ranks 结束后取 `max(rank_elapsed)`，而不是 rank 0 单独时间。
2. `communication_time`
   - 统计 all-reduce/collective 周围的同步 elapsed；
   - 用于解释 CPU scaling 在大 m 下的退化。
3. `evaluation_time`
   - 单独记录，不进入 training speedup。
4. `end_to_end_time`
   - parent process 从 spawn 前到所有 processes join、结果落盘后的总时间；
   - 包含 startup、evaluation 和 serialization。

派生指标：

```text
speedup(m)   = median_time(1) / median_time(m)
efficiency(m)= speedup(m) / m
```

真实 speedup 只依据 CPU Process timing，旧 single-process `time_sec` 不进入新结论。

## D. Work、depth 与 runtime 的解释规则

本阶段不能预设 work 一定接近。对每个 epsilon 如实报告：

```text
depth_ratio = depth_NOG / depth_MEDOL
work_ratio  = work_NOG  / work_MEDOL
time_ratio  = time_NOG  / time_MEDOL
```

允许出现三种不同结论：

1. NOG depth 更低、work 接近、runtime 更低：最符合 Section 5 的 empirical narrative；
2. NOG depth 更低、work 更高，但多 CPU runtime 更低：支持 parallel latency 优势，但不能声称 finite-work 优势；
3. NOG depth 更低但 runtime 不低：说明 communication-round theory 没有在当前 CPU/Gloo/problem size 下转化为 wall-clock speedup。

“work 接近”必须给出预先声明的数值定义。建议在正式报告中同时给 ratio，不使用模糊形容词；若需要文字分类，可暂定 `0.5 <= work_ratio <= 2` 为同一常数量级，但原始 ratio 必须始终展示。

## E. 实施步骤与预计耗时

以下步骤只有在用户批准计划后才能开始：

| Step | 内容 | 验收产物 | 人工耗时 | 机器耗时 |
|---|---|---|---:|---:|
| 1 | 新建 CPU Process/Gloo common runner 与 timing recorder | process/rank/shard/all-reduce 可审计 | 3–5 h | 0 |
| 2 | 移植 NOG-FO、ME-DOL-FO，不改数学 update | 与旧 simulator 短程 allclose | 2–4 h | <15 min |
| 3 | 增加 multiprocessing failure cleanup、resume、manifest | rank failure 可安全退出和恢复 | 1–2 h | <15 min |
| 4 | CPU correctness tests：m=1/2/8、work/depth/seed | unit/integration tests 全通过 | 2–3 h | 15–30 min |
| 5 | 12–96 rounds profile，确认 m=64 和计时口径 | 已完成：42/42 tasks，约 26.3 min | 1 h | 26.3 min |
| 6 | NOG/ME-DOL pilot，seeds 100–102 | Pareto tables、epsilon-specific config | 1–2 h | 6–18 h |
| 7 | m=8 accuracy/depth/work formal，5 seeds | raw results、threshold/audit tables | 0.5–1 h | 1–4 h |
| 8 | m=1..32 runtime scaling，warm-up + 3 repeats | runtime/speedup/efficiency tables | 0.5–1 h | 6–12 h |
| 9 | 汇总、画图、理论对照与可发给学长的结论 | summary.md、figures、manifest | 2–3 h | <30 min |

Step 5 实测 m=8 的 provisional 960-round end-to-end estimate 约为 NOG 224.5 s、ME-DOL 163.4 s；m=1..32 的完整 runtime repetition 也明显高于初始估计。当前剩余机器墙钟修订为约 13–35 小时，其中 pilot grid expansion 是最大不确定项。该估计仍会在 pilot 第一阶段 960-round candidates 完成后再次校正。

## F. 验收与停止规则

进入 formal 前必须满足：

- 每个 logical worker 是独立 PID/rank，而不是单进程 for-loop；
- shards disjoint/exhaustive，seed mapping 可复现；
- m=1 与旧 runner 的 objective/stat_proxy/work/depth 在容差内一致；
- m=2/8 all-reduce mean 与 centralized mean 一致；
- training/evaluation work 分账；
- process startup/evaluation 不混入 training_time；
- training_time 使用 max-rank elapsed；
- 每个 epsilon 的 config 只由 pilot seeds 决定；
- NOG 和 ME-DOL 获得可审计、相近规模的 tuning budget；
- formal coverage、finite metrics、work sum 和 depth monotonicity audit 全通过。

停止规则：

- m=64 profile evidence 已触发正式上限 32：cgroup quota=32 CPUs，且 48/96 rounds 的四个 m64/m32 ratios 均超过 1.5；
- 某 epsilon 在 pilot max budget 内不可达时保留 censored 记录，不无限扩展 rounds；
- formal seeds 上不重新选择 `M,eta,batch` 或 epsilon；
- NOG work 仍明显更高时如实报告，不通过增加 CPU 掩盖 work ratio；
- 若 CPU runtime 没有加速，也如实区分 algorithmic depth 与 systems overhead。

## G. 第二阶段预期输出

```text
outputs/distributed_cpu_fo/<run_name>/
  base_config.yaml
  pilot/
    candidate_manifest.json
    candidate_results.csv
    pareto_by_epsilon.csv
    selected_config_by_epsilon.yaml
  formal_accuracy/
    partials/
    results.csv
    threshold_per_seed.csv
    threshold_summary.csv
    work_accounting_audit.csv
  runtime/
    raw_repeats.csv
    runtime_summary.csv
    speedup_summary.csv
  figures/
    depth_vs_epsilon.png
    work_vs_epsilon.png
    stat_proxy_vs_depth.png
    stat_proxy_vs_work.png
    runtime_vs_workers.png
    speedup_efficiency_vs_workers.png
  environment.json
  completion_manifest.json
  summary.md
```

最终给学长的主材料建议包括：

1. FO stat_proxy vs communication depth；
2. first-hit depth/work ratio across epsilon；
3. CPU runtime vs worker count；
4. speedup/parallel efficiency；
5. 一段明确区分 theoretical communication、finite SFO work 和 real runtime 的结论。

## H. 执行进度与阶段门

用户已批准执行第二阶段。当前进度：

- Step 1：CPU Process/Gloo common layer、独立 PID/rank、deterministic shard、all-reduce mean、max-rank timing 和 timeout cleanup 已完成；
- Step 1 verification：2-process CLI smoke check 通过，仓库全部 7 个 unit tests 通过；
- Step 2：NOG-FO 与 ME-DOL-FO 已迁移，数学 update 不变；rank_schedule 模式下 2-process checkpoint trajectory、depth 和 work 与 simulator 对齐；
- Step 2 verification：仓库全部 8 个 unit tests 通过；
- Step 3：task-level atomic partial、SHA256-validated resume、manifest recovery、completion manifest 和 failure cleanup audit 已完成；
- Step 3 verification：正常 resume、manifest recovery、partial tamper rejection 和 rank failure cleanup 均通过，仓库全部 10 个 tests 通过；
- Step 4：NOG-FO/ME-DOL-FO x m=1/2/8 correctness matrix 已完成；trajectory、seed、PID、shard、thread、work、depth、eval work 和 timing partition audits 全通过；
- Step 4 verification：6/6 tasks passed，m=8 最大 trajectory difference 为 1.49e-8，failure records 为 0，仓库全部 10 个 tests 通过；
- Step 5：正式问题规模、12/48/96 rounds、两种方法和 m=1..64 profile 已完成，42/42 tasks 最终成功；
- Step 5 verification：timing invariants 全通过；m=64 可运行但 oversubscribed，四个预注册 m64/m32 ratios 为 3.17、1.87、3.66、3.23；一次 timeout 清理后 alive_after_cleanup 为空；
- Step 5 decision：后续 worker 上限冻结为 32，runtime epsilon 保持三个，不扩展为五个；
- Step 6A（prepare）已完成：新增固定 high-precision evaluation sample bank、32 个 coarse candidates、confirmed-hit/Pareto selection utilities 和可恢复的 CPU-process pilot runner；
- Step 6A verification：仓库全部 14 个 tests 通过；dry-run 得到 NOG 20 + ME-DOL 12 个 candidates、pilot seeds 100–102 上共 96 个 960-round tasks，raw profile estimate 为 5.38 h，`launches_started=0`；
- Step 6A outputs：`configs/distributed_cpu_fo_pilot.yaml`、`outputs/distributed_cpu_fo/pilot/candidate_manifest.json` 和 `outputs/distributed_cpu_fo/pilot/config_used.json`；
- Step 6B（coarse sweep）已完成：32/32 candidates、96/96 tasks、96/96 atomic partials，failure records 为 0；
- Step 6C（coarse analysis）已完成：生成 160 条 candidate-epsilon summaries、480 条 seed-epsilon confirmed-hit records 和 Pareto/selection/advancement manifests，仓库全部 15 个 tests 通过；
- Step 6C preliminary result：NOG-FO 对五个 epsilon 均为 3/3 full-hit；ME-DOL-FO 对 0.011/0.010/0.009 为 full-hit，对 0.008/0.0075 为 0/3 censored；
- Step 6C advancement：按 `full-hit Pareto frontier or top-3 censored` 规则，暂定 11 个 candidates（NOG 8、ME-DOL 3）进入 1920 rounds，manifest 中 `launches_started=0`；
- Step 6D（1920-round staged extension）已完成：11/11 candidates、33/33 tasks、33/33 atomic partials，failure records 为 0；
- Step 6E（1920 analysis）已完成：33/33 shared-prefix audits 通过，共享 checkpoints 的最大数值差异为 0；960-run 独有的 `iteration=960` 被确认只是预算边界强制 final checkpoint；
- Step 6E result：NOG-FO 五个 epsilon 继续全部 full-hit；ME-DOL-FO 在 epsilon=0.008 已达到 3/3 full-hit，在 epsilon=0.0075 仍只有最佳 candidates 的 1/3 hit，保持 censored；
- Step 6E advancement：仅 3 个 ME-DOL (`epoch_length=[6,12,24]`, `theory_multiplier=10`) 因未解决的 epsilon=0.0075 晋级 3840 rounds，共 9 tasks；NOG 不重复延长，manifest 中 `launches_started=0`；
- Step 6E verification：仓库全部 17 个 tests 通过；
- Step 6F（3840-round max-budget extension）已完成：3/3 candidates、9/9 tasks、9/9 atomic partials，failure records 为 0；
- Step 6G（max-budget analysis）已完成：9/9 shared-prefix audits 通过，共享 checkpoints 的最大数值差异为 0；
- Step 6G result：ME-DOL 在 epsilon=0.0075 的最佳配置为 `epoch_length=24, theory_multiplier=10`，hit rate 从 1920 rounds 的 1/3 提升为 3840 rounds 的 2/3，但仍非 full-hit；最大预注册预算已到，保留 `max-budget censored`，不继续扩展 rounds；
- Step 6G refinement gate：NOG top Pareto union 得到 3 个 base configs；对 `smooth_B=[1,2,4,8]` 共 12 variants，其中 3 个 `smooth_B=2` 复用已有结果，剩余 9 variants × 3 seeds = 27 个 960-round tasks，manifest 中 `launches_started=0`；
- Step 6G verification：仓库全部 19 个 tests 通过；
- Step 6H（NOG batch refinement）已完成：9/9 new variants、27/27 tasks、27/27 atomic partials，failure records 为 0；3 个 `smooth_B=2` variants 直接复用，没有重跑；
- Step 6I（final pilot selection）已完成：12 个 batch variants、五个 epsilon、36/36 work-audit rows 全部通过；生成 60 条 candidate-epsilon 和 180 条 seed-epsilon records；
- Step 6I frozen result：NOG 在五个 epsilon 上均为 3/3 full-hit；ME-DOL 仅 epsilon=0.0075 在 max budget 3840 下为 2/3 censored，其余均 full-hit；最终 config hash 为 `25dc7fe50e1c3205798f1ddb5cc8e9804e8f3135fde794bbd3cdbb16684a66de`；
- Step 6I artifacts：`pilot_final_report.json`、`selected_config_by_epsilon.yaml`、batch-refinement hits/results/Pareto/work-audit CSVs；`pilot_complete=true`、`formal_runs_started=0`；
- Step 6I verification：仓库全部 20 个 tests 通过；
- Step 6 complete：所有参数选择只使用 pilot seeds 100–102，formal seeds 0–4 未参与；
- Step 7A（formal prepare/dry-run）已完成：10 个 method-epsilon pairs 去重为 6 个 unique configs（NOG 3、ME-DOL 3），formal seeds 0–4 上共 30 tasks；
- Step 7A validation：frozen config hash 验证通过，pilot/formal seeds 隔离、五个 epsilon 和两种 methods 覆盖完整；formal manifest hash 为 `9889a57cf7e3e2432b8a12d264ccc25be1c99211e9a0c319775faaa654b89f80`；
- Step 7A timing：根据对应 pilot trajectories 的 median end-to-end time，raw estimate 为 0.48 h，1.5× conservative estimate 为 0.72 h；
- Step 7A outputs：`formal_accuracy/base_config.json` 和 `formal_accuracy/formal_task_manifest.json`；目录中没有 partials，`launches_started=0`；
- Step 7A verification：仓库全部 22 个 tests 通过；
- Step 7B（formal accuracy run）已完成：6/6 frozen configs、30/30 tasks、30/30 atomic partials，failure records 为 0；所有配置均使用 `m=8` 和 formal seeds 0–4；
- Step 7B completion：formal manifest hash 保持为 `9889a57cf7e3e2432b8a12d264ccc25be1c99211e9a0c319775faaa654b89f80`，frozen config hash 保持为 `25dc7fe50e1c3205798f1ddb5cc8e9804e8f3135fde794bbd3cdbb16684a66de`；
- Step 7C（formal audit and threshold summary）已完成：30/30 task-level SHA256、config、seed、rank/PID、finite metric、monotonic depth/work/time 与 analytical SFO work audits 全部通过，共汇总 1,130 条 raw checkpoints、50 条 method-epsilon-seed records、10 条 threshold summaries 和 5 条 method comparisons；
- Step 7C full-hit result：在 `epsilon=[0.011,0.010,0.009]` 上两种方法均为 5/5 hit；NOG 相对 ME-DOL 的 mean communication-depth improvement 分别为 `7.15x, 6.14x, 10.71x`，对应 NOG/ME-DOL total SFO work ratio 为 `4.47x, 10.43x, 2.99x`；
- Step 7C censored result：`epsilon=0.008` 时 NOG 为 5/5、ME-DOL 为 1/5；`epsilon=0.0075` 时 NOG 为 4/5、ME-DOL 为 3/5。对应 hit-only ratios 存在 censoring bias，不能作为无条件算法比较；
- Step 7C outputs：`formal_results.csv`、`threshold_per_seed.csv`、`threshold_summary.csv`、`method_comparison.csv`、`work_accounting_audit.csv`、`formal_audit_report.json`、`formal_analysis_completion.json` 和 `formal_summary.md`；
- Step 7C verification：仓库全部 25 个 tests 通过；
- Step 7D（formal accuracy figures）已完成：基于 Step 7C 已审计且 SHA256 验证通过的 CSV，生成 `depth_vs_epsilon`、`work_vs_epsilon`、`stat_proxy_vs_depth` 和 `stat_proxy_vs_work` 四组 paper-candidate figures，每组同时提供 300-DPI PNG 和 vector PDF，共 8 个文件；
- Step 7D protocol：threshold plots 使用 confirmed-hit mean ± sample std；5/5 full-hit 使用 filled marker，right-censored points 使用 hollow marker 并直接标出 hit count；trajectory plots 对五个 epsilon 分 panel，分别使用该 epsilon 的 pilot-frozen config，曲线和 band 为 5 formal seeds 的 mean ± sample std；
- Step 7D integrity：`figures/figure_manifest.json` 记录全部 source/result hashes、plot protocol、3 个 censored method-epsilon pairs 与 8 个 figure hashes；视觉检查确认 axes、error bars、censoring markers、target epsilon lines 和 legends 可读；
- Step 7D verification：仓库全部 27 个 tests 通过；
- Step 7E（final accuracy result package）已完成：生成 `step7_final/FINAL_RESULTS.md`、`paper_table.tex`、`figure_captions.md`、`advisor_message.md` 和 hash-verified `step7_completion.json`；论文理论对比采用已纠正的 `(2 delta, epsilon)` 与 Section 5 SFO complexity 口径；
- Step 7E claim boundary：只在双方 5/5 hit 的 `epsilon=[0.011,0.010,0.009]` 上报告 unconditional ratios；结论使用 `consistent with the predicted communication advantage`，不声称验证 `epsilon^{-5/3}` exponent，不声称 finite-work parity，不在 censored thresholds 上报告无条件 speedup，也不提前声称 parallel scaling；
- Step 7E paper assets：LaTeX table 对 method-specific right censoring 使用 `dagger` 并删除 censored ratios；四组 English captions 明确 fixed evaluation、confirmed-hit、epsilon-specific frozen configs、training/evaluation work 分账和 conditional means；
- Step 7E verification：4/4 report deliverables 与 8/8 figures 的 SHA256/coverage 验证通过，仓库全部 28 个 tests 通过；
- Step 7 complete：formal accuracy/depth/work 的 30 tasks、审计、汇总、figures 和论文/学长结果包均已完成，不需要再运行 accuracy training；
- Step 8A（runtime prepare/dry-run）已完成：runtime protocol 冻结为 `epsilon=[0.010,0.009,0.008]`、methods `[NOG-FO, ME-DOL-FO]`、`m=[1,2,4,8,16,32]`、benchmark seed 0、one-update-unit warm-up 和 3 measured repeats；m=64 继续排除；
- Step 8A workload deduplication：三个 epsilon 的 ME-DOL 选择完全相同的 `epoch_length=12, theory_multiplier=10, rounds=1920`，因此只测一次再映射回三个 epsilon；NOG 保留三个不同 frozen configs。最终为 4 unique configs、24 unique settings、24 warm-ups、72 measured physical runs，汇总时展开为 108 method-epsilon-worker-repeat rows；
- Step 8A order：workers 在 repeats 间 forward/reverse 交替，四个 unique configs 使用 rotating insertion 改变位置；由于 unique configs 为 3 NOG 对 1 ME-DOL，无法保证每两个相邻 tasks 都切换 method，此限制已写入 manifest；
- Step 8A timing estimate：用 Step 7 exact-config 的 m=8 formal median end-to-end time 作锚点，并乘 Step 5 的 96-round `m/m8` ratio；24 short warm-ups 使用 Step 5 12-round time。raw estimate 为 `0.93 h`，1.5x conservative estimate 为 `1.40 h`；
- Step 8A artifacts：`configs/distributed_cpu_fo_runtime.yaml`、`runtime/base_config.json` 和 `runtime/runtime_task_manifest.json`；manifest 记录 CPU quota 32、96-task order、source hashes、task/config hashes 与 time-estimate caveats，`launches_started=0`；
- Step 8A verification：runtime protocol/config/task-order/manifest tampering tests 全部通过，仓库全部 32 个 tests 通过；没有创建 runtime raw partials，也没有启动 worker；
- Step 8B（formal CPU runtime run）已完成：严格按 frozen manifest 顺序完成 96/96 physical tasks，包括 24/24 one-update-unit warm-ups 和 72/72 measured runs；4 unique configs × 6 workers × 3 repeats coverage 完整，最终失败任务为 0；
- Step 8B timing：实际 sequential wall time 为 `4917.24 s = 81.95 min = 1.366 h`，落在 Step 8A 的 1.40 h conservative estimate 内；
- Step 8B reliability：96/96 atomic partials、96/96 task manifests、96/96 timing invariants 全部通过；第 44 个 NOG-FO `m=16` task 首次发生一次 `CpuProcessLaunchError`，16 个 child processes 全部清理、`alive_after_cleanup=[]`，第二次尝试成功；保留 1 条 failure evidence，最终 completion failures 为 0；
- Step 8B artifacts：`runtime/runtime_progress.json|csv`、`runtime/runtime_completion.json|csv` 和各 setting/repeat 下的 raw partial/task/completion manifests；executor 支持 SHA256-validated resume；
- Step 8B verification：runtime targeted tests 6/6 通过，仓库全部 34 个 tests 通过；
- Step 8C（runtime audit and scaling summary）已完成：96/96 task artifacts 的 manifest/config/task identity、SHA256、rank/PID/thread/shard coverage、finite metric、monotonic depth/work/time 与 analytical SFO work accounting 全部通过；24/24 settings 的 3-repeat trajectory consistency 全部通过，最大 numerical difference 为 0；历史 1 次 process failure 的 cleanup evidence 通过审计；
- Step 8C expansion：72 条 physical measured runs 按 deduplicated ME-DOL mapping 展开为 108 条 method-epsilon-worker-repeat rows；生成 36 条 runtime summaries、36 条 speedup summaries 和 18 条 full-budget method comparisons；24 个 warm-ups 不进入统计；
- Step 8C scaling result：对两个 methods 和三个 epsilon，最低 median training time 均在 `m=1`。NOG 的 `m=32` training speedup vs `m=1` 为 `0.118--0.139x`；当前单机 CPU/Gloo process overhead 超过 local parallelism 收益，因此没有观察到 positive strong scaling。ME-DOL 的 total work 随 m 线性增加，其 m1 ratio 不能解释为 strong-scaling efficiency；
- Step 8C full-budget runtime result：在同一 `m` 下，NOG/ME-DOL median training-time ratio 为 `0.539--0.821`，end-to-end ratio 为 `0.553--0.828`，即本实现和本机条件下 NOG 的 frozen full-budget run 较快；communication fraction 随 m 大致从约 `1%` 增至 `14--15%`；
- Step 8C claim boundary：该对比使用 accuracy-pilot frozen full budgets（NOG 960 rounds、ME-DOL 1920 rounds），不是 first-hit time-to-epsilon；`epsilon=0.008` 的 ME-DOL accuracy 仍 censored。配置没有做 empirical work matching，NOG/ME-DOL full-budget SFO work ratio 为 `4.008--256.533`，因此 runtime 优势不能表述为 finite-work parity，也不能直接验证 Section 5 asymptotic Work complexity；
- Step 8C variability：采用 3 repeats 的 median `[min,max]`，保留 ME-DOL `m=1` 首次 measured run 和部分 NOG `m=16` run 的明显 long-tail timing；不删除 outliers，后续图表必须展示 range/error information；
- Step 8C artifacts：`runtime_work_audit.csv`、`runtime_repeat_consistency.csv`、`runtime_audit_report.json`、`raw_repeats.csv`、`runtime_summary.csv`、`speedup_summary.csv`、`method_runtime_comparison.csv`、`runtime_summary.md` 和 hash-verified `runtime_analysis_completion.json`；
- Step 8C verification：新增 analysis tests 2/2 通过，仓库全部 36 个 tests 通过；
- Step 8D（runtime figures）已完成：只读取 Step 8C hash-verified inputs，生成 `runtime_vs_workers`、`nog_strong_scaling_speedup`、`communication_fraction_vs_workers` 和 `full_budget_method_comparison` 四组 figures；每组同时提供 300-DPI PNG 和 vector PDF，共 8 个 files；
- Step 8D protocol：runtime panels 同时展示 training/end-to-end median 与完整 observed `[min,max]`，不删除 long-tail repeats；speedup panel 只绘制 fixed-total-work 的 NOG，并提供 ideal scaling reference；communication panel 使用 communication/training-time fraction；method comparison 将 training-time ratio 与 unmatched SFO-work ratio 并列，避免只展示有利的 wall-clock ratio；
- Step 8D claim boundary：所有图均明确为 frozen full-budget diagnostic、不是 first-hit time-to-epsilon；单机 CPU/Gloo negative scaling 不外推到 multi-node cluster；ME-DOL 不标注 strong-scaling efficiency；work mismatch 与 `epsilon=0.008` ME-DOL accuracy censoring 均写入 figure notes 和 manifest warnings；
- Step 8D integrity：`runtime/figures/runtime_figure_manifest.json` 记录 4 个 Step 8C source hashes、8 个 figure hashes、plot protocol、worker/epsilon coverage 和 4 条 warnings；`runtime_figure_notes.md` 提供可直接整理为 paper/report captions 的英文说明；
- Step 8D visual verification：四组 PNG 已逐张检查；首次检查发现 runtime 总标题与 legend 重叠，调整顶部留白后复查通过，axes、error ranges、legends、ratio reference lines 和 log scales 均清晰；
- Step 8D verification：新增 figure tests 2/2 通过，仓库全部 38 个 tests 通过；没有启动新的 training experiment；
- Step 8E（final joint result package）已完成：生成 `step8_final/FINAL_RUNTIME_RESULTS.md`、`runtime_figure_captions.md`、`advisor_message.md` 和 `asset_recommendations.md`；报告将 Step 7 accuracy/depth/work 与 Step 8 runtime/scaling 分开解释，并给出 `m=8` aligned comparison table；
- Step 8E paper recommendation：正文优先使用 Step 7 `depth_vs_epsilon`，并强制配套 `work_vs_epsilon`；Step 8 `runtime_vs_workers` 与双 panel `full_budget_method_comparison` 可放 supplement/学长讨论，`communication_fraction` 为 systems diagnostic，negative `nog_strong_scaling_speedup` 不作为 positive result；
- Step 8E joint claim：支持三个 mutual-full-hit thresholds 上 NOG qualitative communication-depth advantage；支持本机实现中 NOG 更快完成 unmatched frozen full budget；明确报告没有 positive single-host CPU-process strong scaling。禁止声称 empirical asymptotic exponent verification、finite-work parity/advantage、work-matched time-to-epsilon speedup、multi-node/GPU scaling 或 censored threshold 的 unconditional runtime-to-epsilon；
- Step 8E integrity：`step8_final/step8_completion.json` 验证论文 PDF、Step 7 completion、Step 8C analysis completion、Step 8D figure manifest 和 4 个 final deliverables 的 SHA256，记录 `verified_runtime_figure_count=8`、`step8_complete=true`；
- Step 8E verification：新增 joint-report test 1/1 通过，仓库全部 39 个 tests 通过，0 failures、0 errors；没有启动新实验；
- Step 8 complete：formal runtime 的 prepare、96 tasks execution、audit/summary、8 figures 和联合 final package 全部完成。下一步默认先将 `step8_final/advisor_message.md`、推荐的 Step 7 figures 和必要的 runtime diagnostic 发给学长确认；在收到新反馈前不事后扩展 epsilon/delta/dimension、GPU scaling、ZO 或 DGFM/DGFM+。

在 Step 6 pilot selection 完成并冻结 config 之前：

- 不启动 formal accuracy 或 formal runtime experiments；
- 不进行 W&B 上传；
- 不把 correctness/profile timing 解释为论文 performance result。

---

## 第一阶段计划与记录（已完成）

以下内容保留为第一阶段 single-process simulation 的历史 specification。

## 1. 目标与结论口径

本轮只完成学长在最新聊天中提出的 distributed validation，不扩展到 quantum experiment 或新的 CIFAR experiment。

实验目标是验证以下 hypothesis：在相同的 nonsmooth nonconvex problem、相同的 worker/data partition 和可核算的 oracle budget 下，NOG 能以更小的 communication depth 达到给定的 stationarity threshold，同时保持有竞争力的 total work。

“NOG 跑得更快”在本文中必须具体表述为：

- 主要含义：达到同一 `stat_proxy` threshold 所需的 `communication_round / depth` 更少；
- 同时报告：`total_work`、`per_worker_work` 和最终 `objective`；
- `wall-clock time` 只作为 implementation diagnostic。因为采用单卡、单进程 sequential simulation，它不能作为真实 distributed speedup 的证据；
- 正式结果不保证一定支持 hypothesis。如果 NOG 没有胜出，应如实报告，而不是在正式 seeds 上继续挑 threshold 或反复调参。

## 2. 已确认的实验范围

### 2.1 执行方式

- 使用一张 NVIDIA A100 80GB；不需要额外 GPU。
- 使用单进程 sequential simulation：逻辑上维护多个 workers、各自的 local state/data shard，并显式模拟 aggregation 或 mixing。
- 不恢复 PyTorch DDP，也不把模拟运行时间解释为真实多卡加速。
- 不再运行仓库中旧的 `distributed_synthetic_smoke` 作为单独阶段；新增 baseline 仍必须通过很小的 deterministic/unit sanity checks，否则无法确认实现正确。

### 2.2 对比算法必须按 oracle 类型分组

不能把 first-order NOG 和 zeroth-order DGFM/DGFM+ 放在同一条“谁更快”的曲线上。正式结果拆为两条赛道：

1. **First-order / SFO track**
   - `NOG-FO`
   - `ME-DOL-FO`

2. **Zeroth-order / SZO track**
   - `NOG-ZO`
   - `ME-DOL-ZO`
   - `DGFM`
   - `DGFM+`

其中 `ME-DOL` 的 first-order/zero-order 版本依据其 Algorithm 1–4；`DGFM` 依据 Algorithm 1；`DGFM+` 依据 Algorithm 2（包括 SPIDER-style variance reduction、periodic large batch 和 restart communication）。

### 2.3 主实验矩阵

- **Algorithm comparison**：固定 `worker_count = 8`，分别运行上述 SFO 和 SZO tracks。这是论文中展示 NOG 与前人方法对比的主结果。
- **NOG scaling**：仅对 NOG 运行 `worker_count = [1, 2, 4, 8]`，验证 fixed total batch 下 `per_worker_work` 是否近似按 `1/m` 缩放，同时检查 convergence curve 没有明显恶化。
- 正式 seeds 使用 `[0, 1, 2, 3, 4]`，报告 mean ± standard deviation；参数选择只使用独立 pilot seeds（建议 `[100, 101]`）。
- 暂不加入 ring topology sensitivity、真实多卡 DDP、CIFAR-10 或额外公开数据集；这些留给和学长看完本轮结果后的下一轮讨论。

## 3. 公平对比协议

### 3.1 Problem 与共同设置

沿用仓库已有、已经调通的 synthetic problem：

```text
F(x; xi) = max_r sin(a_{xi,r}^T x + b_{xi,r}) + lambda * ||x||_1
d = 100, n_data = 4096, R = 4, lambda = 0.001, delta = 0.1
```

- 每个 formal seed 先生成一次 problem 和 `x0`，同一 seed 下所有 methods 共享完全相同的 problem instance 与 data partition。
- method-specific randomness 使用可复现但彼此独立的 RNG streams，不能通过不同 worker 数量意外生成不同 problem。
- 沿用先前 parameter sweep 选出的 NOG `M = 12, eta = 0.3`；不在 formal seeds 上重新选择 NOG 参数。
- 暂定 `rounds = 960`。pilot 若显示多数方法尚未接近预先声明的 thresholds，只允许在 formal run 前统一增加 budget，并把修改写入 `config_used.yaml`。

### 3.2 Communication model

- 主实验对所有方法使用同一个 complete-graph averaging matrix `P = (1/m)11^T`，与当前 NOG 的 exact mean aggregation 对齐。
- 一次应用 `P` 或一次 global aggregation 记为一个 `communication_round`。
- DGFM+ 在 restart 阶段的多次 mixing 必须按实际次数累计 communication rounds，不能只按 outer iteration 计一次。
- 该设置比较的是 algorithmic communication/work，不声称复现 DGFM/ME-DOL 论文中的 ring-network wall time。

### 3.3 Oracle 与 work accounting

- 一次 stochastic gradient sample 计为 `1 SFO call`。
- 一次 two-point zeroth-order estimator 使用 `F(x + delta*u; xi)` 和 `F(x - delta*u; xi)`，计为 `2 SZO calls`。
- DGFM+ 的 periodic large batch、inner small batch 和 restart 均按真实调用数累计。
- 统一记录：`total_work`、`per_worker_work_max`、`communication_round`、`eval_work`、`time_sec`。
- evaluation 使用统一、高精度的 centralized `stat_proxy = ||estimated grad f_delta(x)||`；其开销只进入 `eval_work`，不混入 training work。
- SFO 和 SZO 的 work 数值不跨赛道直接比较。

### 3.4 参数选择规则

- NOG 使用已有结果 `M = 12, eta = 0.3`。
- ME-DOL 的 `D`、epoch length `T` 与 learning rate 以论文公式为中心，只扫描少量常数倍。
- DGFM 只扫描 learning rate；DGFM+ 扫描 learning rate、`b`、`b0`、restart period 和 restart mixing 次数的一个小型、预先写入 YAML 的 grid。
- 所有 baselines 获得相近的 tuning budget；选择标准统一为 pilot seeds 上、给定 work budget 内最小的 mean final `stat_proxy`。
- grid 和选择标准必须在 formal run 前冻结。formal seeds 只运行选定配置，禁止根据正式结果回头挑参数。

## 4. 实施步骤、产物与预计耗时

下表中的人工耗时是专注开发/检查时间；机器耗时按当前 A100 和现有 NOG smoke/formal 记录外推，baseline 实现完成后再用短 profile 修正。

| Step | 工作内容 | 主要产物/验收条件 | 人工耗时 | 单卡机器耗时 |
|---|---|---|---:|---:|
| 1 | 冻结实验 specification；逐项对照 NOG Section 4–5、ME-DOL Algorithms 1–4、DGFM Algorithms 1–2 | 一份 machine-readable config schema；明确每个 method 的 update、output point、oracle 与计数规则 | 1.5–2.5 h | 0 |
| 2 | 重构 common simulator | 统一的 problem、data sharding、mixing matrix、RNG、evaluation、logging；旧 NOG 数值行为不发生非预期变化 | 2–3 h | <5 min |
| 3 | 补齐 `NOG-ZO` | two-point estimator 与论文 Proposition 4.1 一致；`2 SZO/sample` 计数正确 | 1–2 h | <5 min |
| 4 | 实现 `ME-DOL-FO/ZO` | 独立实现论文算法；作者代码仅作行为参考，记录来源和差异 | 2–4 h | <5 min |
| 5 | 实现 `DGFM` | local `x/y/g` state、gradient tracking、mixing 和随机输出/评估点正确 | 1.5–3 h | <5 min |
| 6 | 实现 `DGFM+` | SPIDER recursion、large/small batch、restart tracking 与额外 communication accounting 正确 | 3–5 h | <10 min |
| 7 | Correctness checks（不是旧 smoke experiment） | 固定 seed 可复现；`m=1` 退化行为合理；`P` doubly stochastic；work/depth 手算与日志一致；无 NaN/Inf；shared problem/partition 检查通过 | 1.5–2.5 h | 5–15 min |
| 8 | Pilot 参数选择与 profile，仅使用 seeds `[100,101]` | 冻结每个 baseline 的最佳配置、正式 rounds 和预计总时长；生成 `pilot_summary.csv` | 1–2 h | 1–3 h |
| 9 | Formal runs：m=8 algorithm comparison + NOG m-scaling，seeds `[0..4]` | 所有 raw results、config、environment 和 completion manifest 齐全；任一 method 失败则整组不进入最终图 | 0.5–1 h | 1–2.5 h |
| 10 | 汇总、画图和审计 | 两条 oracle tracks 分图；threshold table；mean±std；work/depth accounting audit；可发给学长的简短结论 | 2–3 h | 5–15 min |

**总计：人工约 16–26 小时，单张 A100 机器时间约 2.5–6 小时。** 实际跨度预计 2–4 个工作日，主要不确定性来自 DGFM+ 的独立实现与 baseline tuning，而不是 NOG 本身。若之后临时增加 GPU，可并行不同 method/seed 来缩短 Step 8–9 的墙钟时间，但本计划不依赖多卡。

## 5. 正式输出

建议新增统一目录（具体 run name 在实施时加入 timestamp/hash）：

```text
outputs/distributed/formal_baselines_d100/
  config_used.yaml
  environment.json
  results.csv
  pilot_summary.csv
  final_summary.csv
  threshold_per_seed.csv
  threshold_summary.csv
  work_accounting_audit.csv
  sfo_stat_proxy_vs_depth.png
  sfo_stat_proxy_vs_total_work.png
  szo_stat_proxy_vs_depth.png
  szo_stat_proxy_vs_total_work.png
  nog_worker_scaling.png
  summary.json
  run.log
```

其中论文/汇报优先使用：

1. `SFO: NOG-FO vs ME-DOL-FO` 的 `stat_proxy vs communication depth`；
2. `SZO: NOG-ZO vs ME-DOL-ZO vs DGFM vs DGFM+` 的 `stat_proxy vs communication depth`；
3. 两条赛道各自的 `stat_proxy vs total work`，防止用更大 oracle budget 换取表面上的低 depth；
4. first-hit threshold table：预先固定 `[0.01, 0.009, 0.008, 0.0075]`，报告 hit rate、round/work mean±std；
5. NOG 在 `m=[1,2,4,8]` 下的 `per_worker_work` scaling 图。

## 6. 验收标准与停止规则

只有同时满足下列条件，结果才可以交给学长：

- 四类 baselines 的 update 与原论文伪代码逐项核对完成；
- 同一 formal seed 的所有 methods 使用同一个 problem/data partition；
- 每条曲线至少包含 5 个 seeds，并清楚标注 error band 是 standard deviation；
- SFO/SZO 分图，training/evaluation work 分账；
- DGFM+ 的 restart communication 和 large-batch SZO calls 没有漏计；
- formal config、raw CSV、日志和 plotting code 足以完全复现图片；
- “更快”只依据预先声明的 threshold/depth/work 规则判断。

停止规则：如果 pilot 后某个 baseline 按论文更新仍然数值不稳定，先检查实现与参数尺度；确认实现无误后，可在预先限定的 grid 内选稳定配置。若 grid 全部失败，则保留 failure record 并暂停该 baseline，不无限扩大 grid。如果 formal 结果不支持 NOG 更低 depth，则如实汇报并和学长讨论下一轮设计。

## 7. 当前代码的已知差距

- `src/distributed/run_distributed_synthetic.py` 当前只有 distributed NOG，没有 ME-DOL、DGFM、DGFM+ 或 NOG-ZO。
- 当前 simulation 是逐 worker 顺序计算，已有 `time_sec` 会随 worker 数增大，不能展示真实并行 speedup。
- 当前 smoke 已经显示 fixed total batch 下 `total_work` 不随 m 改变、`per_worker_work` 约按 `1/m` 下降；这只是 accounting sanity evidence，不能替代 baseline comparison。
- 当前 seed 日志使用了 method/worker-dependent 的派生 seed；正式实现需要额外记录 `problem_seed`、`partition_seed` 和 `method_seed`，以保证 paired comparison 可审计。
- ME-DOL 作者提供了 simulation repository，但仓库主页未显示明确 license；实施时只把它用于核对行为，优先依据论文独立实现，避免直接复制无许可证代码。

## 8. 主要参考材料

- 本地论文：`NeurIPS_NOG.pdf`，重点为 Algorithm 1、Theorems 4.2–4.3、Proposition 5.1 和 Theorems 5.1–5.2。
- DGFM/DGFM+：Z. Lin, J. Xia, Q. Deng, L. Luo, *Decentralized Gradient-Free Methods for Stochastic Non-smooth Non-convex Optimization*, AAAI 2024: https://doi.org/10.1609/aaai.v38i16.29697
- ME-DOL：E. Sahinoglu, S. Shahrampour, *Online Optimization Perspective on First-Order and Zero-Order Decentralized Nonsmooth Nonconvex Stochastic Optimization*, ICML 2024: https://proceedings.mlr.press/v235/sahinoglu24a.html
- ME-DOL 作者 simulation code（只作参考）：https://github.com/emreesahinoglu/Decentralized-Nonsmooth

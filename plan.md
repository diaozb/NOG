# NOG 宽范围 epsilon 实验计划

## 0. 目标与动机

本计划根据 2026-07-21 导师反馈，替换此前仅覆盖
`epsilon=[0.011,0.010,0.009,0.008,0.0075]` 的方案。旧实验已完成真实 CPU
process/Gloo correctness、pilot、5-seed formal accuracy 和 runtime audit，但存在：

1. epsilon 范围过窄，无法观察 scaling 趋势；
2. formal seeds 只有 5 个，first-hit depth 方差较大；
3. ME-DOL 在严格 epsilon 上未命中时结果为空；
4. 每个 epsilon 独立调参，使趋势混入超参数变化。

新实验检验：随 epsilon 下降，`NOG/ME-DOL work ratio` 是否保持常数量级，而
`ME-DOL/NOG communication-depth ratio` 是否总体增长。结果不符合预期时如实报告，
不事后删除 epsilon、seed 或配置。

## 1. 冻结范围

- Problem：`SyntheticMaxSinL1`
- `d=100, n_data=4096, R=4, lambda=0.001, delta=0.1`
- Methods：`NOG-FO, ME-DOL-FO`
- 真实独立 CPU processes + Gloo exact all-reduce
- fixed high-precision evaluation bank；training/evaluation RNG 隔离
- 本轮不改变 dimension、delta、problem family，不加入 ZO、DGFM、CIFAR、GPU 或多机

正式 epsilon：

```text
[0.2, 0.15, 0.1, 0.075, 0.05, 0.03, 0.02, 0.015,
 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002]
```

同一 trajectory 在每个 checkpoint 产生一个 stat-proxy，可复用于全部阈值；增加 epsilon
点本身不会线性增加训练时间，主要成本来自严格 epsilon 的预算扩展。所有预注册阈值均须
出现在最终图表中。

## 2. Seeds 与超参数

- pilot seeds：`[100,101,102,103,104]`
- formal seeds：`[0,...,19]`
- worker robustness seeds：`[0,...,9]`
- pilot/formal 严格隔离

不再逐 epsilon 调参。按三个区间为每个 method 分别冻结一套配置：

| 区间 | 范围 | pilot 代表阈值 |
|---|---|---|
| coarse | `epsilon >= 0.05` | `0.1, 0.05` |
| medium | `0.01 <= epsilon < 0.05` | `0.03, 0.02, 0.01` |
| fine | `epsilon < 0.01` | `0.008, 0.005, 0.002` |

Candidate grids：

```yaml
NOG-FO:
  M: [4, 8, 12, 16, 24]
  eta: [0.03, 0.1, 0.3, 1.0]
  smooth_B: [1, 2, 4, 8]
  data_B_total: 64
ME-DOL-FO:
  epoch_length: [6, 12, 24]
  theory_multiplier: [0.3, 1.0, 3.0, 10.0]
```

选择顺序：代表阈值 confirmed-hit coverage、跨 pilot seeds 稳定性、depth/work/time
Pareto frontier。保存全部候选、选择理由、grid hash；formal seeds 不参与扩网格或重选。

## 3. 分阶段预算与 hit

```text
960 -> 3,840 -> 15,360 -> 61,440 communication/update steps
```

- 只扩展服务于尚未解决代表阈值的候选；
- full-hit 后仅保留该区间 Pareto frontier；
- 无 full-hit 时保留 hit rate 最高、final stat-proxy 最低的少量候选；
- 每次扩展验证 trajectory prefix 一致；
- 单 task 最长 36 小时，整体目标一周内；
- `61,440` 为预注册上限，仍未命中则报告 censoring/lower bound。

Confirmed hit 要求连续两个 evaluation checkpoints 满足 `stat_proxy <= epsilon`，first-hit
取第一个 checkpoint。单次 transient crossing 不算；evaluation work 不计入 training work。

## 4. 正式矩阵

### 4.1 主 epsilon-scaling

```text
methods      = [NOG-FO, ME-DOL-FO]
workers      = 8
epsilons     = 17 个预注册阈值
formal seeds = 20
parameters   = method x epsilon-region frozen config
max budget   = 61,440
```

物理 task 数不等于 `2 x 17 x 20`：同一 method/region/config/seed 的一条最长 trajectory
复用于该 region 全部 epsilon，manifest 必须记录 deduplication 映射。

### 4.2 Worker-count 稳健性

```text
methods = [NOG-FO, ME-DOL-FO]
workers = [1,2,4,8]
epsilon = [0.1,0.01,0.005]
seeds   = [0,...,9]
```

此实验检查主要 epsilon 趋势是否依赖 `m=8`，不用于声称 strong scaling。总并发 worker
processes 不超过 32。不重跑 `m=16/32`，旧实验已显示单机 Gloo 在高 worker 下负扩展。

## 5. Censoring 与统计

每个 method x epsilon 报告：

- hit count/rate；
- hit-only conditional mean/std（明确标注为条件统计）；
- median first-hit depth/work；
- deterministic bootstrap 95% CI；
- capped mean：未命中按最大 depth/work 计入；
- right-censoring-aware restricted mean；
- final stat-proxy distribution和 censoring horizon。

不得静默舍弃未命中 seed，即使只有 1 个未命中也保留 hit rate。

重点比例：

```text
depth_ratio = ME-DOL depth / NOG depth
work_ratio  = NOG total SFO work / ME-DOL total SFO work
per_worker_work_ratio = NOG per-worker work / ME-DOL per-worker work
```

报告 paired-seed ratio median/bootstrap CI、ratio of capped/restricted means、双方命中的
paired seed 数及一方未命中时的 ratio bound。不将 hit-only ratio解释为无条件结果。

趋势分析：

- 对 `log(1/epsilon)` 与 log-ratio 做加权稳健回归；
- 报 slope、bootstrap CI 和 Spearman correlation；
- 比较 depth-ratio slope 与 work-ratio slope；
- 仅在 CI 支持时使用 “increases” 或 “approximately constant”；
- 本实验不宣称验证精确理论 exponent，因为 d、delta 和 problem family 固定。

## 6. 图表与交付物

生成 PNG/PDF：

1. hit rate vs epsilon；
2. first-hit depth vs epsilon（CI+censored markers）；
3. first-hit total work vs epsilon；
4. `ME-DOL/NOG depth ratio` vs epsilon；
5. `NOG/ME-DOL work ratio` vs epsilon；
6. depth/work ratio 同图；
7. capped/restricted mean sensitivity；
8. worker-count robustness panels。

x 轴为 log scale，覆盖 `[0.002,0.2]`。Censored 点、下界和 conditional estimates 使用
不同符号。另交付 per-seed/summary/paired-ratio CSV、trend JSON、accounting audit、figure
manifest、双语报告、README 更新和复现命令。

## 7. 实施步骤与门槛

### Step 1：协议实现（已完成）

- 新建独立配置/output root，不覆盖旧 Step 7/8；
- 加入 17 epsilon、20 formal seeds、5 pilot seeds和三个 regions；
- 将固定预算阶段泛化为任意 stages；
- 保留 atomic partial、hash identity、resume、failure cleanup；
- 调度器限制总 CPU processes `<=32`。

### Step 2：统计实现（已完成）

- 实现 capped/restricted mean、paired censoring bounds、deterministic bootstrap、trend stats；
- 对全命中、部分命中、零命中编写测试；
- 未命中不得产生误导性空白主结果或被当作真实 hit。

### Step 3：dry-run 与 pilot（已完成）

- 小 problem、2 seeds、短预算 dry-run；
- 验证 trajectory 多 epsilon 复用；
- 复用旧 artifacts 或扩展 pilot；
- 冻结三个 region configs 并生成选择审计。

### Step 4：主 formal（已完成：120/120 通过审计）

- `m=8`、20 seeds，逐 stage 后台运行；
- 每阶段生成 progress/completion/ETA；
- 所有完成 task 通过 trajectory、rank/shard、work/depth/hash audit。

### Step 5：worker robustness（已完成：240/240 通过审计）

- `[1,2,4,8] x [0.1,0.01,0.005] x 10 seeds`；
- 不与高负载主实验争用 CPU。

### Step 6：最终分析（已完成：Step 6A–6D）

- 生成图表、统计和报告；
- 对预注册假设逐项给出 supported/not supported；
- 更新 README 并保留旧结果作为历史证据。

长阶段启动门槛：全部测试通过；dry-run 和 audit 通过；输出 task 数、CPU-hours、磁盘和
最长 task 估计；预计单阶段不超过 36 小时；无冲突实验；output root 与旧结果隔离。

## 8. 资源与后台运行

- CPU process 硬上限 32，并为系统保留余量；
- 后台 runner 写 PID、stdout/stderr、heartbeat 和 completion manifest；
- 启动前检查磁盘、内存、load 和现有 NOG processes；
- 失败时保留 evidence，清理全部 child ranks 后再 retry；
- 总运行目标一周内；若资源不足，优先保留 17 epsilon 和统计完整性；将 formal seeds
  从 20 降到 10 只能作为明确记录的资源降级，不优先删除严格 epsilon。

# NOG distributed baseline 实验说明与复现指南

> 本文件是早期 SFO/SZO 共同基线的历史归档。当前 FO 主结论见 [fo_experiments/README.md](fo_experiments/README.md)；后续 ZO 实验可复用这里的 SZO 实现与审计口径，但应使用独立配置和结果目录。

本文档记录 2026-07-14 完成的 distributed synthetic experiment：做了哪些代码修改、实验如何设置、如何复现、结果如何解释，以及后续应从哪里验证或调整。

> 本实验是单张 GPU 上的 **single-process logical distributed simulation**。代码维护独立的 worker data shards 和 optimizer states，并显式计算 communication depth，但 workers 实际上顺序执行。因此，本文中的 wall-clock time 只用于诊断，不能作为真实多机或多卡 speedup 的证据。

## 1. 实验目的和比较口径

本轮实验仅处理学长提出的 distributed validation：

- SFO track：`NOG-FO` 对比 `ME-DOL-FO`；
- SZO track：`NOG-ZO` 对比 `ME-DOL-ZO`、`DGFM` 和 `DGFM+`；
- 固定 `m=8` workers 做算法主对比；
- 对 NOG 额外运行 `m=[1,2,4,8]`，检查 fixed-total-batch 条件下的 per-worker work scaling。

SFO 和 SZO 使用不同类型的 oracle，不能放在一起直接比较 oracle work。“NOG 更快”在这组实验中应具体解释为：

1. 达到同一 stationarity threshold 所需的 communication depth 更低；
2. 或者在相近 communication depth 下获得更低的 stationarity proxy；
3. 同时必须报告 total/per-worker oracle work，防止把更大的 batch 隐藏起来；
4. 不使用 single-process wall-clock time 声称真实 distributed speedup。

完整的事前实验计划见 [plan.md](fo_experiments/PLAN.md)。

## 2. 修改和新增的文件

### 2.1 配置

- [configs/distributed_baselines_d100.yaml](configs/distributed_baselines_d100.yaml)
  - 原始实验配置和 pilot search grid；
  - 包含 problem、oracle、NOG、baseline、worker、metric 和 W&B 设置。
- [outputs/distributed/distributed_baselines_d100/pilot/config_selected.yaml](outputs/distributed/distributed_baselines_d100/pilot/config_selected.yaml)
  - pilot 完成后冻结的正式配置；
  - formal run 实际使用该文件；
  - `run.pilot_selection_complete: true` 是 formal preflight gate。

### 2.2 公共模拟器

- [src/distributed/common.py](src/distributed/common.py)
  - 可复现的 problem/partition/method seed；
  - deterministic、disjoint、exhaustive data sharding；
  - complete-graph mixing matrix；
  - SFO randomized-smoothing estimator；
  - SZO two-point estimator；
  - centralized evaluation；
  - training/evaluation/oracle/communication accounting；
  - config、shard 和 mixing matrix validation。

### 2.3 算法实现

- [src/distributed/algorithms.py](src/distributed/algorithms.py)
  - `run_nog(..., oracle_type="sfo"|"szo")`；
  - `run_me_dol(..., oracle_type="sfo"|"szo")`；
  - `run_dgfm(...)`；
  - `run_dgfm_plus(...)`。

这些实现依据论文伪代码独立编写。ME-DOL 作者仓库只用于行为核对，没有直接复制其未明确授权的代码。

### 2.4 运行和汇总

- [src/distributed/run_distributed_baselines.py](src/distributed/run_distributed_baselines.py)
  - 统一的 check/pilot/formal CLI；
  - 根据 method 自动选择 SFO/SZO；
  - NOG 自动运行 scaling workers，baselines 默认只运行 `m=8`；
  - 每个 method/worker/seed task 完成后立即保存 partial CSV；
  - 使用相同命令重启时自动 resume；
  - 保存 `config_used.yaml`、`environment.json` 和最终 `results.csv`。
- [src/distributed/run_pilot.py](src/distributed/run_pilot.py)
  - 在独立 pilot seeds 上执行小型 grid search；
  - DGFM+ 分 phase 1/2 搜索，减少组合数量；
  - 按 method 内的 common oracle-work budget 评分；
  - 输出冻结后的 `config_selected.yaml`；
  - 可选 W&B 上传。
- [src/distributed/summarize_results.py](src/distributed/summarize_results.py)
  - 检查 formal method/seed coverage；
  - 生成 final、threshold 和 curve tables；
  - 检查 work/depth monotonicity、finite metrics 和 worker work 求和；
  - 绘制 SFO/SZO depth/work curves 和 NOG scaling 图。

### 2.5 测试和仓库辅助文件

- [tests/test_distributed_simulation.py](tests/test_distributed_simulation.py)
  - data shards 与 complete mixing；
  - work/depth accounting；
  - evaluation RNG isolation；
  - 六种 algorithm variants 的手工计数；
  - 忽略 wall time 后的 deterministic trajectory。
- [.gitignore](.gitignore)
  - 保留/整理了 `wandb/`、`__pycache__/`、`data/` 和旧 synthetic archive 的忽略项。
- [plan.md](fo_experiments/PLAN.md)
  - 记录实验范围、公平性规则、预计步骤和当前完成状态。

原有 [src/synthetic/run_synthetic.py](src/synthetic/run_synthetic.py) 没有为本轮实验重写；新代码复用了其中的 `SyntheticMaxSinL1`、`sample_ball`、`project_l2_ball` 和 seed/device utilities。

## 3. 公共实验设置

### 3.1 运行环境

正式运行记录在 [environment.json](outputs/distributed/distributed_baselines_d100/formal/environment.json)：

| 项目 | 值 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.1.2+cu121 |
| CUDA runtime | 12.1 |
| GPU | NVIDIA A100 80GB PCIe |
| Device | cuda |
| 执行方式 | single-process simulation |

使用的 Python 路径为：

```bash
/root/miniconda3/envs/NOG/bin/python
```

如果已经激活环境，也可以把后续命令中的完整路径替换成 `python`。

### 3.2 Synthetic problem

```text
F(x; xi) = max_r sin(a_{xi,r}^T x + b_{xi,r}) + lambda * ||x||_1
d = 100
n_data = 4096
R = 4
lambda = 0.001
delta = 0.1
```

- 所有方法从零向量开始；
- 同一个 formal seed 的所有方法共享同一个 problem instance；
- 同一个 seed 和 worker count 使用同一 deterministic data partition；
- shards 不重叠且完整覆盖 `[0, n_data)`。

Seed 规则：

```text
problem_seed   = 100000 + formal_seed
partition_seed = 200000 + formal_seed
method_seed    = stable SHA-256 hash(formal_seed, method, worker_count)
```

正式 seeds 为 `[0,1,2,3,4]`，pilot seeds 为 `[100,101]`，两者不重叠。

### 3.3 Distributed/communication model

- 主比较使用 `m=8`；
- NOG scaling 使用 `m=[1,2,4,8]`；
- topology 为 complete graph：

```text
P = (1/m) 11^T
```

- 每应用一次 `P` 或一次 exact global aggregation，communication depth 加一；
- 当前 topology 是在算法中通过 `complete_graph_mixing()` 明确构造的。只修改 YAML 中的 `topology` 字符串不会产生 ring graph；如需 ring/mesh，必须修改代码。

### 3.4 Oracle 和 evaluation

训练设置：

| 配置 | 值 |
|---|---:|
| rounds | 960 |
| eval_every | 48 |
| smooth_B | 8 |
| data_B_total | 64 |
| split_mode | total_batch_fixed |
| SFO sample cost | 1 |
| two-point SZO sample cost | 2 |

对 NOG，`data_B_total=64` 被均匀拆到 workers，因此每个 worker 的 local data batch 是 `64/m`，所有 workers 合计的 batch 始终是 64。

统一 evaluation 使用：

- `eval_smooth_B=64`；
- `eval_data_B=128`；
- centralized first-order smoothed gradient norm 作为 `stat_proxy`；
- 全数据 objective 作为 `objective`；
- 每次 evaluation 产生 `64*128=8192` 个 evaluation SFO calls；
- evaluation work 单独进入 `eval_work`，不混入 training `total_work`。

Evaluation 在隔离的 RNG context 中运行，不会改变训练随机轨迹。同一个 problem seed 和 iteration 使用相同 evaluation seed，便于 paired comparison。

## 4. 各算法的实际设置与计数

### 4.1 NOG-FO / NOG-ZO

```yaml
M: 12
eta: 0.3
delta: 0.1
rounds: 960
```

- update projection radius 为 `delta/M`；
- 每次训练 oracle 使用 `smooth_B=8` 和 fixed total data batch 64；
- 正式循环前计算两个独立初始 oracle，它们都计入 training work 和 communication depth；
- 因此最终 depth 是 `960+2=962`；
- 每次 global oracle 的 SFO work 为 `8*64=512`；
- SZO two-point work 是其两倍，即 1024。

最终计数：

| 方法 | total work | depth |
|---|---:|---:|
| NOG-FO | `962*512 = 492544` SFO calls | 962 |
| NOG-ZO | `962*1024 = 985088` SZO calls | 962 |

不同 worker counts 下 total work 不变，per-worker work 理想情况下按 `1/m` 缩放。

### 4.2 ME-DOL-FO / ME-DOL-ZO

```yaml
epoch_length: 12
theory_multiplier:
  sfo: 3.0
  szo: 1.0
```

代码中的参数关系为：

```text
theory_radius = delta / (4 * epoch_length * sqrt(m))
domain_radius = theory_multiplier * theory_radius
learning_rate = domain_radius / sqrt(epoch_length)
```

- 每个 inner iteration、每个 worker 使用一个 local sample；
- FO 计 1 SFO call，ZO two-point 计 2 SZO calls；
- action mixing 和 state mixing 可以合并为同一个通信 payload，因此每个 inner iteration 计一个 communication depth；
- 960 iterations 后 depth 为 960；
- m=8 时 FO total work 为 `8*960=7680`，ZO 为 `2*8*960=15360`。

### 4.3 DGFM

```yaml
eta: 0.001
batch_size: 1
```

- 每个 worker 每轮使用一个 two-point SZO estimator；
- gradient tracker mixing 计一次通信；
- parameter mixing 再计一次通信；
- 每轮 depth 增加 2。

m=8、960 rounds 后：

```text
total work = 2 * 8 * 960 = 15360 SZO calls
depth      = 2 * 960 = 1920
```

### 4.4 DGFM+

```yaml
eta: 0.001
small_batch: 8
large_batch: 64
restart_period: 48
restart_mixing_rounds: 1
```

- restart iteration 使用 large batch，per-worker work 为 `2*large_batch`；
- ordinary SPIDER iteration 在相同 samples/directions 上计算 current 和 previous point，per-worker work 为 `4*small_batch`；
- restart mixing 和 ordinary tracker mixing 都显式计数；
- parameter mixing 每轮另计一次；
- 当前 `restart_mixing_rounds=1`，所以每轮总 depth 仍是 2。

960 rounds 中有 20 个 restart iterations 和 940 个 ordinary iterations，因此：

```text
per-worker work = 20*(2*64) + 940*(4*8) = 32640
total work      = 8*32640 = 261120 SZO calls
depth           = 1920
```

## 5. Pilot tuning 和冻结配置

Pilot 使用 `m=8`、seeds `[100,101]`、480 rounds，正式 seeds 没有用于调参。

原始 grid 位于 [configs/distributed_baselines_d100.yaml](configs/distributed_baselines_d100.yaml)：

- ME-DOL multiplier：`[0.3,1.0,3.0]`；
- DGFM eta：`[0.001,0.003,0.01,0.03]`；
- DGFM+ 先选 eta，再搜索 small/large batch、restart period 和 mixing rounds；
- candidate 在同一 method 的 common final oracle-work budget 下比较 mean final `stat_proxy`。

最终选择见 [selected_candidates.csv](outputs/distributed/distributed_baselines_d100/pilot/selected_candidates.csv)：

| 方法 | 选中参数 | pilot mean stat_proxy |
|---|---|---:|
| ME-DOL-FO | multiplier=3.0 | 0.009580 |
| ME-DOL-ZO | multiplier=1.0 | 0.010184 |
| DGFM | eta=0.001 | 0.010593 |
| DGFM+ | eta=0.001, small=8, large=64, period=48, mixing=1 | 0.010517 |

这些值写入 [config_selected.yaml](outputs/distributed/distributed_baselines_d100/pilot/config_selected.yaml)，并在 formal run 前冻结。

## 6. 实际运行命令

以下命令均在仓库根目录 `/data/diaozb/NOG` 执行。

### 6.1 Unit tests

```bash
/root/miniconda3/envs/NOG/bin/python \
  -m unittest tests.test_distributed_simulation -q
```

结果：5 tests passed。

### 6.2 Pilot phase 1

```bash
/root/miniconda3/envs/NOG/bin/python -u \
  src/distributed/run_pilot.py \
  --config configs/distributed_baselines_d100.yaml \
  --rounds 480 \
  --phase phase1 \
  --no-wandb
```

### 6.3 Pilot phase 2

```bash
/root/miniconda3/envs/NOG/bin/python -u \
  src/distributed/run_pilot.py \
  --config configs/distributed_baselines_d100.yaml \
  --rounds 480 \
  --phase phase2 \
  --no-wandb
```

Pilot candidate CSV 会增量保存；重新运行相同命令会 resume 已完成 candidates。

### 6.4 Formal run

```bash
/root/miniconda3/envs/NOG/bin/python -u \
  src/distributed/run_distributed_baselines.py \
  --config outputs/distributed/distributed_baselines_d100/pilot/config_selected.yaml \
  --stage formal
```

任务总数为：

```text
每个 seed:
  NOG-FO:    4 worker settings
  ME-DOL-FO: 1
  NOG-ZO:    4 worker settings
  ME-DOL-ZO: 1
  DGFM:      1
  DGFM+:     1
合计 12 tasks/seed * 5 seeds = 60 tasks
```

正式运行完成 `60/60` tasks，得到 1,260 行 checkpoint records。

### 6.5 汇总和画图

```bash
/root/miniconda3/envs/NOG/bin/python -u \
  src/distributed/summarize_results.py \
  --results outputs/distributed/distributed_baselines_d100/formal/results.csv \
  --config outputs/distributed/distributed_baselines_d100/pilot/config_selected.yaml \
  --out-dir outputs/distributed/distributed_baselines_d100/formal
```

脚本输出：

```text
summary_written=outputs/distributed/distributed_baselines_d100/formal
audit_all_pass=True
```

## 7. 输出文件

正式输出目录为 [outputs/distributed/distributed_baselines_d100/formal](outputs/distributed/distributed_baselines_d100/formal)。

主要文件：

| 文件 | 用途 |
|---|---|
| `results.csv` | 全部 raw checkpoint records |
| `partials/*.csv` | 每个 method/worker/seed 的断点文件 |
| `config_used.yaml` | runner 实际读取并补充 stage/device 后的配置 |
| `environment.json` | Python/PyTorch/CUDA/GPU 环境 |
| `final_per_seed.csv` | 每个 seed 的最终结果 |
| `final_summary.csv` | mean/std 汇总 |
| `threshold_per_seed.csv` | 每个 seed 的 first-hit threshold |
| `threshold_summary.csv` | hit rate 和 first-hit mean/std |
| `curve_summary.csv` | 绘图用 mean/std 曲线 |
| `work_accounting_audit.csv` | work/depth/finite metrics 审计 |
| `summary.md` / `summary.json` | 自动汇总 |

图片：

- [sfo_stat_proxy_vs_depth.png](outputs/distributed/distributed_baselines_d100/formal/sfo_stat_proxy_vs_depth.png)
- [sfo_stat_proxy_vs_total_work.png](outputs/distributed/distributed_baselines_d100/formal/sfo_stat_proxy_vs_total_work.png)
- [szo_stat_proxy_vs_depth.png](outputs/distributed/distributed_baselines_d100/formal/szo_stat_proxy_vs_depth.png)
- [szo_stat_proxy_vs_total_work.png](outputs/distributed/distributed_baselines_d100/formal/szo_stat_proxy_vs_total_work.png)
- [nog_worker_scaling.png](outputs/distributed/distributed_baselines_d100/formal/nog_worker_scaling.png)

## 8. 正式结果

### 8.1 m=8 最终结果

下面是 5 formal seeds 的 mean ± sample standard deviation：

| 方法 | Oracle | Final stat_proxy | Total work | Per-worker work | Depth |
|---|---|---:|---:|---:|---:|
| NOG-FO | SFO | 0.007311 ± 0.000153 | 492544 | 61568 | 962 |
| ME-DOL-FO | SFO | 0.009525 ± 0.000522 | 7680 | 960 | 960 |
| NOG-ZO | SZO | 0.008976 ± 0.000623 | 985088 | 123136 | 962 |
| ME-DOL-ZO | SZO | 0.010829 ± 0.000808 | 15360 | 1920 | 960 |
| DGFM+ | SZO | 0.011023 ± 0.000678 | 261120 | 32640 | 1920 |
| DGFM | SZO | 0.011325 ± 0.000513 | 15360 | 1920 | 1920 |

### 8.2 First-hit thresholds

SFO：

- 对 threshold `0.01`，两种方法 hit rate 都是 100%；
- NOG-FO mean first-hit depth 为 33.2，ME-DOL-FO 为 307.2；
- 即 NOG-FO 在这个 threshold 上使用约 `307.2/33.2 = 9.3x` 更少的 communication depth；
- 但对应 first-hit total work 分别约为 16998 和 2458，NOG 使用了约 6.9x 更多 SFO work；
- 对 `0.009`，NOG-FO hit rate 100%、depth 62；ME-DOL-FO hit rate 60%，其 604 depth 只是在命中 seeds 上的条件均值。

SZO：

- 对 threshold `0.009`，只有 NOG-ZO 在所有 seeds 上命中，mean depth 为 628.4；
- ME-DOL-ZO、DGFM 和 DGFM+ 的 hit rate 均为 0；
- 对 `0.01`，NOG-ZO hit rate 100%；另外三种方法均为 60%；
- 因为 baseline 的 first-hit depth 只对命中的 3/5 seeds 求平均，不能把该条件均值直接当作完整公平的 speedup ratio。

### 8.3 NOG worker scaling

| 方法 | m | Per-worker work | Final stat_proxy |
|---|---:|---:|---:|
| NOG-FO | 1 | 492544 | 0.007404 |
| NOG-FO | 2 | 246272 | 0.007252 |
| NOG-FO | 4 | 123136 | 0.007390 |
| NOG-FO | 8 | 61568 | 0.007311 |
| NOG-ZO | 1 | 985088 | 0.009203 |
| NOG-ZO | 2 | 492544 | 0.009380 |
| NOG-ZO | 4 | 246272 | 0.009326 |
| NOG-ZO | 8 | 123136 | 0.008976 |

per-worker work 精确按 `1/m` 下降，最终 stationarity proxy 没有随 worker count 明显恶化。

### 8.4 可以和不可以得出的结论

当前结果支持：

- NOG-FO 在显著更少的 communication depth 内达到 SFO threshold `0.01`；
- NOG-ZO 在约 962 depth 时优于同 depth 的 ME-DOL-ZO，也优于运行到 1920 depth 的 DGFM/DGFM+；
- NOG 在 fixed total batch 下实现理想的 `1/m` per-worker work scaling；
- NOG 以更高 oracle work 换取了更低 stationarity proxy 和较低 communication depth。

当前结果不支持：

- NOG 的 oracle complexity 在该有限实验中优于所有 baselines；
- 单卡 sequential wall-clock time 等价于真实 distributed runtime；
- 当前 complete-graph 结果自动推广到 ring、sparse graph 或真实网络延迟；
- 仅凭本 synthetic problem 得出普遍的 empirical superiority。

## 9. 如何验证

### 9.1 最小验证顺序

1. 运行 unit tests；
2. 对配置做 dry run；
3. 运行 12-round short check；
4. 运行汇总脚本，确认 coverage/audit；
5. 查看 depth 和 total-work 两组图，而不是只看一组。

Dry run：

```bash
/root/miniconda3/envs/NOG/bin/python -u \
  src/distributed/run_distributed_baselines.py \
  --config outputs/distributed/distributed_baselines_d100/pilot/config_selected.yaml \
  --stage check \
  --dry-run
```

短测试：

```bash
/root/miniconda3/envs/NOG/bin/python -u \
  src/distributed/run_distributed_baselines.py \
  --config outputs/distributed/distributed_baselines_d100/pilot/config_selected.yaml \
  --stage check \
  --methods NOG-FO,ME-DOL-FO,NOG-ZO,ME-DOL-ZO,DGFM,DGFM+ \
  --seeds 100 \
  --workers 8 \
  --rounds 12 \
  --name manual_check_12
```

短测试输出位于：

```text
outputs/distributed/manual_check_12/check/
```

### 9.2 查看 accounting audit

打开 [work_accounting_audit.csv](outputs/distributed/distributed_baselines_d100/formal/work_accounting_audit.csv)，每行应满足：

- `total_work_monotone=True`；
- `depth_monotone=True`；
- `eval_work_monotone=True`；
- `total_equals_sum_workers=True`；
- `finite_metrics=True`；
- `all_checks_pass=True`。

### 9.3 Resume 和 stale partial 注意事项

`run_distributed_baselines.py` 会优先读取已经存在的：

```text
<out_dir>/<run.name>/<stage>/partials/<method>__m<workers>__seed<seed>.csv
```

这使中断恢复很方便，但也意味着：

> 修改 config 或算法代码后，不要继续使用旧 `run.name`，否则可能直接复用旧 partials。

推荐每次修改实验时通过 `--name` 使用新目录，例如：

```bash
--name distributed_baselines_d100_eta_test
```

只有在 config、代码和 task identity 完全未变时，才应复用原 run name 做 resume。

## 10. 如何修改和扩展

### 10.1 只修改 threshold 或画图

修改 config 的：

```yaml
metrics:
  thresholds: [0.01, 0.009, 0.008, 0.0075]
```

然后重新运行 `summarize_results.py` 即可，不需要重跑训练。建议输出到新目录，避免覆盖原正式汇总。

### 10.2 修改训练 rounds / evaluation frequency

修改：

```yaml
train:
  rounds: 960
  eval_every: 48
```

约束：

- `rounds % nog.M == 0`；
- ME-DOL 还要求 `rounds % epoch_length == 0`；
- 修改 `eval_every`、evaluation batch 或 delta 后必须重跑训练，因为 raw results 中的 evaluation metrics 会改变。

### 10.3 修改 worker counts 或 batch

修改：

```yaml
distributed:
  comparison_worker: 8
  scaling_workers: [1, 2, 4, 8]

oracle:
  data_B_total: 64
```

在 `total_batch_fixed` 模式下，`data_B_total` 必须能被每一个 worker count 整除。

### 10.4 调整 NOG

修改：

```yaml
nog:
  M: 12
  eta: 0.3
```

`M` 同时影响 block output、projection radius `delta/M` 和合法 rounds。修改后应先运行 unit tests 和 short check，再使用新的 run name。

### 10.5 重新调 baseline

修改 base config 中的 `pilot` grid，然后重新运行 phase 1/2。需要注意：

- 仍然只用 pilot seeds；
- 不要根据 formal seeds 反复选参数；
- 如果修改了算法/problem/evaluation code，应改 `run.name`，避免 pilot candidate CSV resume 旧结果；
- formal 必须使用新生成、带 `pilot_selection_complete: true` 的 selected config。

### 10.6 修改 topology

当前算法直接调用 `complete_graph_mixing()`。要加入 ring 或 sparse graph，需要：

1. 在 `common.py` 增加 mixing matrix builder；
2. 用 `validate_mixing_matrix()` 检查 doubly stochastic；
3. 在 ME-DOL/DGFM/DGFM+ 中按 config 选择 matrix；
4. 明确 NOG 的 exact aggregation 如何与 graph mixing 对齐；
5. 增加 topology-specific tests；
6. 重新定义一次 communication round 的含义。

只把 YAML 的 `topology: complete` 改成 `ring` 不会改变当前算法。

### 10.7 改为真实多 GPU

当前代码没有使用 DDP 或 RPC。真实多卡实验需要另外实现：

- 每卡一个 process/worker；
- local data shard 和 local RNG；
- all-reduce 或 graph neighbor communication；
- barrier/timing warm-up；
- communication volume、latency 和 wall-clock measurement；
- failure-safe distributed checkpoint。

在此之前，不应把 `time_sec` 用作多 worker 加速结果。

## 11. W&B 状态

冻结 config 中保留：

```yaml
wandb:
  enabled: true
  project: nog-distributed-baselines
  mode: online
```

但本次 pilot 命令显式使用了 `--no-wandb`，formal runner 当前也没有自动上传逻辑。因此：

- 本轮结果只保存在本机；
- 登录 W&B 不等价于已经上传；
- 如需上传 pilot，省略 `--no-wandb` 即会触发外部写入；
- formal results 若要上传，需要新增/运行明确的上传步骤，并在上传前确认 project/entity 和数据范围。

## 12. 推荐的下一步

建议按以下顺序继续：

1. 先把 [summary.md](outputs/distributed/distributed_baselines_d100/formal/summary.md)、两张 depth 图和两张 work 图发给学长确认叙事；
2. 明确论文正文主张是 communication efficiency，而不是 wall-clock 或 oracle efficiency；
3. 若学长认可，再把图表整理成论文统一 style，并决定使用 standard deviation 还是 confidence interval；
4. 若学长认为 oracle gap 过大，先讨论 matched-work 或 matched-target 的补充实验设计，再预注册新的 tuning/config；
5. 若需要展示真实“跑得快”，另行设计多 GPU wall-clock experiment，不应复用当前 sequential timing。


# NOG：高并行非光滑非凸优化实验仓库

本仓库实现并验证 Nonconvex Optimistic Gradient（NOG）在非光滑非凸随机优化中的
first-order（FO）与 zeroth-order（ZO）版本，重点比较通信/并行 `depth` 和总 oracle
`work`。仓库同时包含 ICLR 论文源码、冻结配置、正式实验的机器可读结果、审计清单和
复现脚本。

> 当前论文实验以 **FO theory-validation v4** 和 **ZO frozen 20-seed protocol** 为准。
> 早期 v2、FO v5--v7、旧共享基线和原始输出保存在
> [`archive/legacy-experiments`](https://github.com/diaozb/NOG/tree/archive/legacy-experiments)，
> 不应与正式结果拼接或替换。

## 1. 快速导航

| 内容 | 入口 |
|---|---|
| FO 主结果、参数与运行说明 | [`fo_experiments/README.md`](fo_experiments/README.md) |
| FO 完整实验解释 | [`fo_experiments/EXPERIMENT_DETAILED_EXPLANATION.md`](fo_experiments/EXPERIMENT_DETAILED_EXPLANATION.md) |
| FO 计划与执行记录 | [`fo_experiments/PLAN.md`](fo_experiments/PLAN.md) |
| ZO 当前状态与正式入口 | [`zo_experiments/README.md`](zo_experiments/README.md) |
| ZO 完整实验说明 | [`zo_experiments/EXPERIMENT_DETAILED_EXPLANATION.md`](zo_experiments/EXPERIMENT_DETAILED_EXPLANATION.md) |
| ZO 全部数值、比例与图表 | [`zo_experiments/ALL_EXPERIMENT_RESULTS.md`](zo_experiments/ALL_EXPERIMENT_RESULTS.md) |
| ZO 计划与执行记录 | [`zo_experiments/PLAN.md`](zo_experiments/PLAN.md) |
| ICLR 论文源码 | [`nog_iclr2027_complete_source/`](nog_iclr2027_complete_source/) |
| 原始论文 PDF | [`NeurIPS_NOG.pdf`](NeurIPS_NOG.pdf) |
| 历史实验索引 | [`archive/legacy-experiments`](https://github.com/diaozb/NOG/blob/archive/legacy-experiments/ARCHIVE_README.md) |

## 2. 实验问题与比较方法

主要 synthetic 实验使用 `SyntheticMaxSinL1`：

```text
F(x; ξ) = max_r sin(a_{ξ,r}^T x + b_{ξ,r}) + λ ||x||_1
```

默认设置为 `d=100`、`n=4096`、约束半径 `R=1`、`λ=0.001` 和 Goldstein
smoothing radius `δ=0.1`。所有方法从同一初始点出发，并使用方法独立的 evaluation
bank 计算高精度 stationarity proxy。

统一报告口径：

- `confirmed hit`：连续两个有效 checkpoints 的 stationarity proxy 均不超过目标
  `epsilon`；
- `depth`：第一次 confirmed hit 对应的依赖图/通信深度；
- `work`：达到 confirmed hit 前累计的训练 oracle calls，evaluation work 单独计算；
- `paired ratio`：先在相同 formal seed 内计算 baseline/NOG，再跨 seed 求均值；
- 未命中的点按 right-censored 结果报告，conditional mean 不当作完整样本比例。

### 算法范围

| Oracle | NOG | Baselines | 主要实现 |
|---|---|---|---|
| FO | NOG-FO | ME-DOL-FO | `src/distributed/cpu_fo_algorithms.py` |
| ZO | NOG-ZO | ME-DOL-ZO、DGFM、DGFM+ | `src/distributed/algorithms.py`、`cpu_zo_algorithms.py` |

## 3. 当前正式结果

### 3.1 FO：NOG-FO 与 ME-DOL-FO

FO 正式实验采用 8 个真实 CPU/Gloo workers、pilot seeds 100--104 和独立 formal
seeds 0--19。论文主图只使用 batch 保持为 8 的同质区间
`epsilon in [0.0105, 0.2]`：

- 25 个 epsilon 点均为 NOG/ME-DOL 双方 20/20 confirmed hits；
- `ME-DOL/NOG depth` 从 `0.488x` 单调增加至 `1.102x`，Spearman `rho=1`；
- `NOG/ME-DOL work` 位于 `0.980x--2.049x` 的常数量级范围；
- 更小 epsilon 的 batch=16 与扩预算结果属于探索性/删失诊断，详见 FO 文档。

![FO v4 depth/work ratios](results/theory_validation_v4/analysis/figures/depth_work_ratios.png)

完整数值、冻结参数、审计和结论边界见
[`fo_experiments/README.md`](fo_experiments/README.md)。

### 3.2 ZO：NOG-ZO 与三种 baseline

ZO 主实验比较 NOG-ZO、ME-DOL-ZO、DGFM 和 DGFM+。四种方法使用相同的
`983,040` training SZO work cap、8 个 logical workers、pilot seeds 100--104 和独立
formal seeds 0--19。只在 20/20 same-seed paired hits 的区间拟合趋势：

| Baseline/NOG-ZO | 完整区间 | Depth ratio 端点 | Work ratio 端点 |
|---|---|---:|---:|
| ME-DOL-ZO/NOG-ZO | 0.200--0.018 | 1.596 -> 3.547 | 0.399 -> 0.887 |
| DGFM/NOG-ZO | 0.200--0.018 | 3.556 -> 8.791 | 0.444 -> 1.099 |
| DGFM+/NOG-ZO | 0.200--0.030 | 3.520 -> 11.387 | 0.471 -> 1.514 |

三条 depth ratio 曲线在各自完整区间均满足 Spearman `rho=1`。这提供有限样本下与
理论方向一致的 communication-depth evidence；work 只作为描述性次要指标，不声称
恢复精确 worst-case exponent 或普遍优越性。

![ZO formal paired ratios](zo_experiments/formal/figures/formal_ratios.png)

ZO 还包含以下审计和扩展：

- 低 epsilon right censoring 与 NOG-ZO trajectory rebound 的独立 seeds 复核；
- `d in {25,50,100,200}` 的 fixed-configuration dimension sensitivity；
- `m in {1,2,4,8}` 的 logical-worker accounting；
- NOG-ZO/ME-DOL-ZO 的 1/2/8-worker CPU/Gloo 数值等价审计；
- a9a 与 ijcnn1 的 20-seed 真实数据实验。真实数据没有复现 synthetic depth advantage，
  因而明确作为 applicability boundary 报告。

完整结果见
[`zo_experiments/ALL_EXPERIMENT_RESULTS.md`](zo_experiments/ALL_EXPERIMENT_RESULTS.md)。

## 4. 参数选择与公平性

FO 和 ZO 均遵循 pilot-then-freeze 协议：

1. 只使用 pilot seeds 100--104 搜索候选参数；
2. 在运行 formal seeds 0--19 前冻结算法参数、预算、epsilon 网格和统计规则；
3. formal seeds 不参与重新调参；
4. 同一实验内使用共同的最大 work 预算或明确记录 matched-work 规则；
5. 结果包保存 frozen parameters、输入 SHA256、audit 和 analysis manifest。

代表性冻结参数：

| Method | Frozen parameters |
|---|---|
| NOG-FO | `M=2`, `eta=1.0`, `smooth_B=1`; batch schedule 见 FO README |
| ME-DOL-FO | `epoch_length=6`, `theory_multiplier=100` |
| NOG-ZO | `M=2`, `eta=0.01`, `smooth_B=8` |
| ME-DOL-ZO | `epoch_length=12`, `theory_multiplier=60` |
| DGFM | `eta=0.05` |
| DGFM+ | `eta=0.05` |

ZO 的冻结配置与三份哈希输入位于
[`zo_experiments/frozen_parameters.json`](zo_experiments/frozen_parameters.json) 和
[`zo_experiments/pilot_inputs/`](zo_experiments/pilot_inputs/)。
加载器优先读取本机原始 pilot 路径；在全新 clone 中路径不存在时，会使用这里的同哈希
副本，同时保持正式运行使用的 frozen JSON 及其下游审计哈希不变。

## 5. 代码结构

| 路径 | 内容 |
|---|---|
| `src/distributed/` | FO/ZO 算法、Gloo runner、simulation、统计与审计 |
| `src/synthetic/` | synthetic objective 与通用训练组件 |
| `src/cifar/` | CIFAR 示例代码 |
| `configs/` | 正式、pilot、扩预算与等价性配置 |
| `fo_experiments/` | FO 文档入口 |
| `zo_experiments/` | ZO 文档、冻结输入、正式表格、图和审计 |
| `results/theory_validation_v4/` | FO v4 紧凑正式结果包 |
| `results/theory_validation_v4_extended_budget/` | FO v4 扩预算诊断包 |
| `nog_iclr2027_complete_source/` | ICLR LaTeX、附录和论文图 |
| `scripts/` | 长实验调度、结果汇总和绘图辅助脚本 |
| `tests/` | 计数、冻结协议、runner、dimension/worker 等测试 |
| `outputs/` | 本机 raw trajectories；被 Git 忽略，可断点续跑 |

## 6. 环境安装

当前 `NOG` Conda 环境使用 Python 3.10.20、PyTorch 2.1.2+cu121、NumPy 1.24.1
和 Pandas 2.3.3。精确依赖记录在 [`requirements.txt`](requirements.txt)。

```bash
conda create -n NOG python=3.10 -y
conda activate NOG
pip install -r requirements.txt
```

`requirements.txt` 是当前机器的完整环境快照，包含 CUDA 相关 wheels。若机器 CUDA
版本不同，建议先按照 PyTorch 官方方式安装与本机匹配的 PyTorch，再安装其余依赖。
FO CPU/Gloo 实验不要求 GPU；ZO dependency-graph simulation 会在 CUDA 可用时使用 GPU。

## 7. 测试与快速检查

从仓库根目录运行：

```bash
conda run -n NOG python -m pytest -q \
  tests/test_theory_validation_freeze.py \
  tests/test_theory_validation_runner.py \
  tests/test_theory_validation_v4_extension.py \
  tests/test_cpu_zo_equivalence.py \
  tests/test_zo_dimension_analysis.py \
  tests/test_zo_worker_analysis.py
```

完整测试：

```bash
conda run -n NOG python -m pytest -q
```

## 8. 复现实验

### 8.1 FO v4

以下命令支持合法 partials 的 resume，最多同时运行 4 个 8-worker tasks，即不超过
32 个 worker processes：

```bash
conda run -n NOG python -m src.distributed.theory_validation_runner pilot-batch-grid
conda run -n NOG python -m src.distributed.theory_validation_freeze
conda run -n NOG python -m src.distributed.theory_validation_runner formal
conda run -n NOG python -m src.distributed.theory_validation_audit
conda run -n NOG python -m src.distributed.theory_validation_analysis
conda run -n NOG python -m src.distributed.theory_validation_report
conda run -n NOG python -m src.distributed.theory_validation_package
```

更详细的资源限制与扩预算命令见
[`results/theory_validation_v4/REPRODUCE.md`](results/theory_validation_v4/REPRODUCE.md) 和
[`results/theory_validation_v4_extended_budget/REPRODUCE.md`](results/theory_validation_v4_extended_budget/REPRODUCE.md)。

### 8.2 ZO formal epsilon-scaling

先检查冻结任务，不执行训练：

```bash
conda run -n NOG python -m src.distributed.zo_formal --dry-run
```

运行/恢复正式 80 个 method-seed tasks，并生成正式统计与理论解释：

```bash
conda run -n NOG python -m src.distributed.zo_formal
conda run -n NOG python -m src.distributed.zo_formal_analysis
conda run -n NOG python -m src.distributed.zo_theory_interpretation
conda run -n NOG python -m src.distributed.zo_anomaly_audit
```

已完成轨迹的 dimension、worker 和 real-data 分析入口：

```bash
conda run -n NOG python -m src.distributed.zo_dimension_analysis
conda run -n NOG python -m src.distributed.zo_worker_analysis
conda run -n NOG python -m src.distributed.zo_real_data_analysis
```

完整分阶段运行流程见
[`zo_experiments/EXPERIMENT_DETAILED_EXPLANATION.md`](zo_experiments/EXPERIMENT_DETAILED_EXPLANATION.md)。

## 9. 论文源码

主文件：

- [`nog_iclr2027_complete_source/nog_iclr2027.tex`](nog_iclr2027_complete_source/nog_iclr2027.tex)
- [`nog_iclr2027_complete_source/apx.tex`](nog_iclr2027_complete_source/apx.tex)
- [`nog_iclr2027_complete_source/figures/`](nog_iclr2027_complete_source/figures/)

正文中的 FO 与 ZO 实验用于佐证理论方向，不从有限样本反推理论，也不把删失点当作真实
first-hit ratios。论文正文、两张实验图和结论控制在第 10 页内，参考文献和附录随后开始。

## 10. 数据、输出与归档约定

- `data/`、`outputs/`、`wandb/`、日志和 checkpoints 默认只保留在本机；
- GitHub `main` 只提交紧凑、审计过的正式结果和复现所需冻结输入；
- 原始 trajectory 的 SHA256 记录在相应 manifest 中；
- 历史但不再使用的结果见
  [`archive/legacy-experiments`](https://github.com/diaozb/NOG/tree/archive/legacy-experiments)；
- FO 与 ZO 的 oracle work 定义不同，禁止直接把两类 work 数值合并比较。

如需引用实验结果，请优先引用 `main` 中的 FO v4、ZO formal 报告及论文中的审慎表述。

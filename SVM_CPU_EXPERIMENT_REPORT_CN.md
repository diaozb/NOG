# SVM CPU 联合选参与正式实验报告

> 更新时间：2026-08-23<br>
> 结果包：[`svm_cpu_joint_retuning_2026/`](svm_cpu_joint_retuning_2026/)<br>
> 结论先行：联合重选 `batch + M + eta` 后，大 batch 没有得到独立验证支持。NOG-ZO 在两个 SVM 数据集的同等 communication depth 下均不领先；在同等 training work 下仅在 ijcnn1 最终领先，在 a9a 上仍由 DGFM 最好。

## 1. 实验目的

此前 NOG-ZO 在 `SyntheticMaxSinL1` 合成问题上表现出通信优势，但在 a9a 和 ijcnn1 的 SVM 实验中，前期 stationarity proxy（下文记为 epsilon）长时间不下降。旧 pilot 只运行了正式 work 预算的十分之一，因此可能在平台期尚未结束时就选择了过于保守的参数。

本轮实验专门回答以下问题：

1. NOG 的 SVM 平台期是否主要由 `batch` 太小造成；
2. 联合重选 `batch`、块长 `M` 和步长 `eta` 后，平台期能否缩短；
3. 新参数下，NOG 在同等通信深度和同等训练计算量下能否领先 ME-DOL-ZO、DGFM 和 DGFM+。

为避免用正式结果反复调参，协议在运行 search 前写入 JSON；search、validation、formal 三组种子完全分离，最终参数在 formal 前冻结。

## 2. 优化问题与数据

实验使用带 capped-\(\ell_1\) 正则项的 hinge-loss SVM：

\[
F(x)=\frac{1}{n}\sum_{i=1}^{n}\max(0,1-y_i a_i^\top x)
+\frac{10^{-5}}{n}\sum_{j=1}^{d}\min\{|x_j|,2\}.
\]

这是非光滑、非凸的分段结构目标。训练样本逐行做 \(\ell_2\) 归一化；官方 test split 只计算 test accuracy，不参与参数选择。

| 数据集 | 训练样本 | 测试样本 | 维度 |
|---|---:|---:|---:|
| a9a | 32,561 | 16,281 | 123 |
| ijcnn1 | 49,990 | 91,701 | 22 |

共同设置：

- 8 个 worker，complete-graph mixing，单进程模拟分布式算法；
- CPU-only，Python 3.10.20，未使用 GPU；
- smoothing delta=`0.001`，evaluation delta=`0.002`；
- epsilon 定义为共同高精度评估器给出的平滑一阶估计范数：`eval_smooth_B=32`、`eval_data_B=256`；
- evaluation work 不计入 training work；所有算法使用相同 fixed evaluation bank；
- formal training work 上限统一为 983,040 次训练函数评价；
- 精确保存 \(x=0\) 的初始评价，不再用近零 checkpoint 近似初始化。

## 3. 比较算法

### NOG-ZO

使用理论部分对应的 block-average NOG 实现。算法保存两个独立的初始零阶 oracle，迭代中使用由最近两个 oracle 构成的更新，并在每个长度为 `M` 的 block 上评价平均点。零阶 oracle 使用双点估计器。

### ME-DOL-ZO

使用 decentralized online gradient descent 结构；每个 epoch 内进行局部更新和网络混合，在 epoch 的平均点上评价。正式参数为 epoch length=6、theory multiplier=30,000、每 worker data batch=16、smooth batch=1。

### DGFM

使用零阶梯度跟踪；每次迭代分别混合梯度 tracker 和参数，因此一轮迭代包含两个通信层。batch size=16；a9a 的 eta=0.5，ijcnn1 的 eta=2.0。

### DGFM+

使用带 SPIDER 型递推和周期性大 batch restart 的零阶梯度跟踪。small batch=8、large batch=64、restart period=24、restart mixing rounds=1；a9a 的 eta=0.2，ijcnn1 的 eta=0.1。

## 4. Work 与 communication depth 口径

- `training work`：所有 worker 的训练函数评价次数总和；双点零阶估计的一对 \(f(x+\delta u),f(x-\delta u)\) 计2次。
- `communication depth`：算法必须顺序执行的网络混合/聚合层数，不把可并行的本地计算重复计入。
- `evaluation work` 单独记录，不计入公平训练预算。

冻结配置下的精确计数如下：

| 算法 | 每轮/阶段 training work | 每轮通信 depth | work=983,040 时最终 depth |
|---|---:|---:|---:|
| NOG-ZO | 每个 oracle 为 \(2\times1\times64=128\)；另有2个初始 oracle | 每个 oracle 1层 | 7,680 |
| ME-DOL-ZO | \(2\times1\times16\times8=256\) | 1层 | 3,840 |
| DGFM | \(2\times16\times8=256\) | 2层 | 7,680 |
| DGFM+ | restart轮为1,024，普通轮为256 | 每轮2层 | 6,822 |

因此，大 batch 会减少固定 work 下可执行的 NOG 轮数和通信深度；它只有在降低估计方差的收益超过轮数损失时才可能改善曲线，不能假定 batch 越大越好。

## 5. NOG 联合选参协议

### 5.1 Search 网格

| 参数 | a9a 候选 | ijcnn1 候选 |
|---|---|---|
| `M` | 1, 2, 4 | 1, 2, 4 |
| `eta` | 9e-6, 3e-5, 9e-5, 3e-4, 9e-4 | 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 |
| `smooth_B` | 1, 2, 4, 8 | 1, 2, 4, 8 |
| `data_B_total` | 64, 128, 256 | 64, 128, 256 |

每个数据集共有 \(3\times5\times4\times3=180\) 个候选。每个候选都运行完整 983,040 work，而不是旧实验的 98,304 work。

- Search seeds：400--404，共 1,800 个候选-seed 任务；
- Validation seeds：500--504，对每个数据集 search 前3名独立验证，共30个任务；
- Formal seeds：0--19，严禁参与选参。

### 5.2 预先固定的评分规则

在 25%、50%、75%、100% work 处，选择不超过目标 work 的最近原生 checkpoint，计算：

\[
\operatorname{score}=\operatorname{mean}_{\text{seed},\,q\in\{.25,.5,.75,1\}}
\log(\epsilon_q).
\]

分数越小越好；非有限或发散配置淘汰。分数接近时依次参考最终 epsilon、更小的 seed 方差和更低的 depth。主要指标考察整条轨迹，不能只选择最后一个偶然低点。

### 5.3 冻结结果

| 数据集 | M | eta | smooth_B | data_B_total |
|---|---:|---:|---:|---:|
| a9a | 1 | 9e-5 | 1 | 64 |
| ijcnn1 | 1 | 1e-4 | 1 | 64 |

a9a search 第一名是 eta=3e-5，但独立 validation 的整轨迹评分选择 eta=9e-5；ijcnn1 的 search 与 validation 均选择 eta=1e-4。两个数据集最终都选择最小 batch 和 `M=1`，所以“大 batch 能稳定解决平台期”没有得到支持。

## 6. 实验运行步骤

1. 在 [`protocol.json`](svm_cpu_joint_retuning_2026/audit/protocol.json) 中预登记网格、预算、种子和评分规则；
2. 对两个数据集运行全部180个候选和5个 search seeds，保留全部候选，包括不利结果；
3. 汇总整轨迹评分，各数据集选择前3名；
4. 使用5个全新 validation seeds 独立复跑前3名；
5. 将 validation 第一名写入 [`frozen_parameters.json`](svm_cpu_joint_retuning_2026/audit/frozen_parameters.json)；
6. 只用冻结参数运行 NOG formal seeds 0--19，共40个新 CPU 任务；
7. ME-DOL-ZO、DGFM、DGFM+ 的代码、数据、预算和 evaluation bank 未改变，因此复用原20-seed formal 轨迹；复用条件见 [`baseline_reuse_audit.csv`](svm_cpu_joint_retuning_2026/audit/baseline_reuse_audit.csv)；
8. 在共同 depth/work 目标上选择最近原生 checkpoint，不插值，计算20-seed均值和95% t 置信区间；
9. 生成完整线性纵轴图、严格0--0.05放大图以及 epsilon 阈值首次命中表；
10. 审计 seed、work、非有限值、失败任务和 SHA256，并运行完整测试。

## 7. 正式实验结果

### 7.1 同等 training work=983,040

表中为 epsilon 的20-seed均值，括号内为95%置信区间；越小越好。

| 数据集 | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |
|---|---:|---:|---:|---:|
| a9a | 0.02517 (0.02374, 0.02661) | 0.01923 (0.01582, 0.02264) | **0.01588 (0.01306, 0.01869)** | 0.02617 (0.02176, 0.03059) |
| ijcnn1 | **0.01233 (0.01135, 0.01330)** | 0.01479 (0.01327, 0.01630) | 0.01496 (0.01281, 0.01711) | 0.02071 (0.01796, 0.02346) |

结论：a9a 由 DGFM 最好；ijcnn1 由 NOG-ZO 最好。置信区间重叠时不应把均值排序解释成已经完成显著性检验。

### 7.2 同等 communication depth 约为3,840

NOG 的最近原生 checkpoint 为 depth=3,843，其余方法为3,840。

| 数据集 | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |
|---|---:|---:|---:|---:|
| a9a | 0.02837 | 0.01923 | **0.01725** | 0.02885 |
| ijcnn1 | 0.02796 | **0.01479** | 0.01688 | 0.02859 |

结论：在相同通信深度下，NOG 在两个数据集上都不领先。

### 7.3 NOG 平台期与阈值首次命中

下表为20个 formal seeds 的首次命中均值：

| 数据集 | epsilon阈值 | 命中数 | 平均depth | 平均work |
|---|---:|---:|---:|---:|
| a9a | 0.05 | 20/20 | 2,028.6 | 259,660.8 |
| a9a | 0.03 | 20/20 | 3,070.2 | 392,985.6 |
| a9a | 0.02 / 0.015 | 0/20 | — | — |
| ijcnn1 | 0.05 | 20/20 | 3,363.0 | 430,464.0 |
| ijcnn1 | 0.03 | 20/20 | 3,468.6 | 443,980.8 |
| ijcnn1 | 0.02 | 20/20 | 3,708.6 | 474,700.8 |
| ijcnn1 | 0.015 | 20/20 | 5,148.6 | 659,020.8 |

ijcnn1 的 NOG 前期平台更长，但进入下降阶段后速度较快，最终在同等 work 下取得最低均值；a9a 虽然也离开平台，却没有追上 DGFM。

### 7.4 目标值与分类准确率

终点 test accuracy 均值：

| 数据集 | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |
|---|---:|---:|---:|---:|
| a9a | 0.8011 | 0.8343 | **0.8438** | 0.8308 |
| ijcnn1 | 0.90535 | 0.91996 | **0.92449** | 0.90520 |

stationarity proxy 最低并不必然对应 test accuracy 最高。尤其在 ijcnn1 上，NOG 的 epsilon 最低，但分类准确率低于 ME-DOL-ZO 和 DGFM；论文中应把优化指标和统计泛化指标分开陈述。

## 8. 结果讨论

### 8.1 平台期不是画图错误

新实验保存了精确 \(x=0\) 评价，并提高 checkpoint 密度。平台仍在各 seed 中出现，因此不是黑色初始化星号、插值或稀疏画图造成的假象。

### 8.2 为什么 hinge-loss SVM 容易出现该现象

hinge loss 是分段线性的。当一段时间内满足 margin 条件的活跃样本集合基本不变时，目标值和参数可能缓慢变化，但平滑梯度范数的变化很小；较小 eta 会延长跨越分段边界所需时间。因此 epsilon 可能先形成平台，随后在活跃集合改变时明显下降。该解释与轨迹一致，但仍属于机制解释，不等同于已经证明的理论结论。

### 8.3 增大 batch 为什么没有解决问题

更大的 batch 能降低单次 oracle 方差，但也会按比例增加每轮 work，从而在固定983,040预算下减少更新轮数和通信层数。联合网格与独立 validation 最终仍选择 `smooth_B=1, data_B_total=64`，说明在本实验范围内，方差降低不足以补偿轮数损失。

### 8.4 与旧 NOG 结果的关系

旧参数的 a9a 终点约为0.0182，而新冻结参数的 formal 均值为0.02517；ijcnn1 旧、新结果都约为0.0123。新结果没有被筛选或删除：20个 formal seeds 全部保留。a9a 的差异说明5个 validation seeds 对相近候选的排序仍有不确定性，也说明不能查看 formal 后继续调参来追回旧结果。

### 8.5 当前可支持的结论

- NOG 在 `SyntheticMaxSinL1` 上展示出的通信优势，没有在这两个真实 SVM 数据集上稳定复现；
- 联合重选 `batch/M/eta` 后，SVM 前期 epsilon 平台期仍然存在；
- 同等 depth 下 NOG 在 a9a、ijcnn1 均不领先；
- 同等 work 下 NOG 在 ijcnn1 最终领先，但在 a9a 落后于 DGFM；
- 因此 SVM 结果更适合被描述为算法的实验适用边界，而不是普遍优势证据；
- 只有两个真实数据集，不能据此断言 NOG 对所有 SVM 或所有非光滑问题都无效。

## 9. 公平性和完整性检查

- 四种算法、两个数据集均为20/20 formal seeds；
- search=400--404、validation=500--504、formal=0--19，完全分离；
- 没有删除失败或不利 seed；search、validation、formal 均无失败任务；
- 最大 training work 为983,040；逐 checkpoint epsilon 均为有限值；
- 公共曲线使用最近原生 checkpoint，不插值、不截取有利区间；
- 图为曲线和95% CI 阴影，没有柱状图，也没有对数纵轴；
- baseline 仅在公平性输入完全不变的条件下复用；
- 完整测试：92 tests + 14 subtests 全部通过；`git diff --check` 通过；
- CPU：Intel Xeon Gold 5220R，20个 formal 并行进程，每进程1个 Torch 线程；CUDA unavailable；
- 所有交付文件提供 SHA256 manifest。

## 10. 图和机器可读结果

- [完整线性纵轴图（PNG）](svm_cpu_joint_retuning_2026/figures/svm_equal_budget_epsilon_full.png)
- [严格 epsilon=0--0.05 放大图（PNG）](svm_cpu_joint_retuning_2026/figures/svm_equal_budget_epsilon_zoom_0_005.png)
- [正式逐 checkpoint 轨迹](svm_cpu_joint_retuning_2026/data/formal_trajectories.csv)
- [正式终点汇总及95% CI](svm_cpu_joint_retuning_2026/data/formal_summary_ci95.csv)
- [完整 search 候选汇总](svm_cpu_joint_retuning_2026/data/pilot_grid.csv)
- [Search 逐 seed 结果](svm_cpu_joint_retuning_2026/data/pilot_grid_per_seed.csv)
- [独立 validation 汇总](svm_cpu_joint_retuning_2026/data/pilot_validation.csv)
- [阈值首次命中汇总](svm_cpu_joint_retuning_2026/data/threshold_first_hit.csv)

0--0.05 放大图会裁剪高于0.05的前期点，图中已明确说明；这不表示算法从0.05开始。

## 11. 复现

固定使用：

```bash
/root/miniconda3/envs/NOG/bin/python
```

不需要 `conda activate`。配置为：

- `configs/distributed_zo_batch_svm_a9a.yaml`
- `configs/distributed_zo_batch_svm_ijcnn1.yaml`

冻结后的单 seed formal 示例：

```bash
/root/miniconda3/envs/NOG/bin/python scripts/run_retuned_nog_formal.py \
  --freeze results/advisor_cpu_batch_retuned_svm/frozen_parameters.json \
  --output results/advisor_cpu_batch_retuned_svm/formal_shards/seed-00 \
  --seeds 0 --cpu-threads 1
```

合并与画图：

```bash
/root/miniconda3/envs/NOG/bin/python scripts/merge_retuned_svm_results.py \
  --retuned-root results/advisor_cpu_batch_retuned_svm \
  --retuned-shards results/advisor_cpu_batch_retuned_svm/formal_shards \
  --baseline-root outputs/distributed_zo/zo_theory_validation/real_data/supplement_formal_cpu_v2_seed_shards \
  --output results/advisor_cpu_batch_retuned_svm/merged

/root/miniconda3/envs/NOG/bin/python scripts/plot_retuned_svm_equal_budget.py \
  --source results/advisor_cpu_batch_retuned_svm/merged/formal_trajectories.csv \
  --output results/advisor_cpu_batch_retuned_svm
```

结果包 `reproduction/` 中保存了本轮配置与脚本快照；真正复跑时使用仓库根目录中的同名配置、脚本和 `src/` 实现。

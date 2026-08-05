# NOG-FO 与 ME-DOL-FO：v4 理论验证实验完整说明

本文档系统整理当前仓库中 v4 主实验及其低 $\epsilon$ 扩展实验的研究问题、理论依据、参数选择、运行流程、统计方法、实验结果、局限性和论文写作边界。

> 核心结论：v4 实验在有限精度区间内支持理论所预测的**定性方向**——随着 $\epsilon$ 变小，ME-DOL-FO 相对 NOG-FO 的首次命中 depth 比例总体上升，而两者 work 保持在同一数量级。但该实验不能精确验证渐近复杂度指数，也不能证明 NOG 在所有 $\epsilon$ 上都优于 ME-DOL。

## 1. 原文理论与待验证命题

### 1.1 优化问题

本项目依据论文 [Highly-Parallel Algorithms for Non-Smooth Non-Convex Optimization](NeurIPS_NOG.pdf)，考虑随机非光滑、非凸优化问题

$$
f(x)=\mathbb{E}_{\xi}[F(x;\xi)]，
$$

并寻找 $(\delta,\epsilon)$-Goldstein stationary point。直观地说，算法需要在以 $x$ 为中心、半径为 $\delta$ 的邻域内，找到广义梯度足够小的点；$\epsilon$ 越小，要求的驻点精度越高。

### 1.2 depth 与 work

- **Depth**：并行算法的串行依赖层数，也就是即使拥有足够多处理器仍必须依次执行的轮数。
- **Work**：所有并行工作节点完成的随机一阶 oracle 调用总量。

论文中 NOG 的 Theorem 4.3 给出的量级为

$$
D_{\mathrm{NOG}}
=O\!\left(
d^{1/3}(\gamma+\delta L)L^{2/3}
\delta^{-1}\epsilon^{-5/3}
\right),
$$

$$
W_{\mathrm{NOG}}
=O\!\left(
(\gamma+\delta L)L^2
\delta^{-1}\epsilon^{-3}
\right).
$$

作为对照的 ME-DOL 具有

$$
D_{\mathrm{ME}}=O(\delta^{-1}\epsilon^{-3}),
\qquad
W_{\mathrm{ME}}=O(\delta^{-1}\epsilon^{-3}).
$$

ME-DOL 的公开原文见 [arXiv:2406.01484](https://arxiv.org/abs/2406.01484)。固定 $d,\delta,L,\gamma$ 后，理论上预期

$$
\frac{D_{\mathrm{ME}}}{D_{\mathrm{NOG}}}
\propto \epsilon^{-4/3},
\qquad
\frac{W_{\mathrm{NOG}}}{W_{\mathrm{ME}}}
=O(1).
$$

因此本实验主要检验以下定性命题：

1. 当 $\epsilon$ 变小时，$D_{\mathrm{ME}}/D_{\mathrm{NOG}}$ 是否总体上升；
2. 两者的 work 比例是否仍保持在同一数量级；
3. 在足够小的 $\epsilon$ 上，NOG 是否表现出更明显的 depth 优势。

这里检验的是有限样本、有限预算下的经验趋势，并非直接证明复杂度上界。

## 2. 实验问题与数据生成

### 2.1 合成目标函数

实验采用 `SyntheticMaxSinL1`：

$$
F(x;\xi)
=\max_{1\le r\le R}\sin(a_{\xi,r}^{\top}x+b_{\xi,r})
+\lambda\lVert x\rVert_1.
$$

主实验参数如下：

| 参数 | 数值 | 含义 |
|---|---:|---|
| $d$ | 100 | 参数维数 |
| `n_data` | 4096 | 合成样本数 |
| $R$ | 1 | 正弦分支数 |
| $\lambda$ | 0.001 | $\ell_1$ 非光滑项系数 |
| `feature_scale` | 1.0 | 特征尺度 |
| `bias_scale` | 0.25 | 偏置尺度 |
| `phase_scale` | 0.0 | 相位尺度 |
| $\delta$ | 0.1 | Goldstein 邻域半径 |

需要特别注意：当前设定为 $R=1$，所以 `max` 本身不会引入多分支切换；主要非光滑性来自 $\ell_1$ 项。它仍是合法的非光滑非凸测试问题，但比 $R>1$ 的多分支问题简单。

### 2.2 分布式执行环境

- 每个正式任务使用 8 个真实 CPU 进程；
- 通信后端为 Gloo；
- 使用 complete topology 和精确 all-reduce；
- 每个 rank 限制为 1 个 CPU 线程；
- 主实验最多同时执行 4 个任务，因此最高约 32 个进程；
- 4096 个样本被划分为 8 个互不重叠的 shard，每个 rank 持有 512 个样本。

该设计确实执行了多进程通信与同步，但所有进程位于同一台机器，因此它验证的是算法层面的并行 depth/work 计数，不是跨机器网络延迟或强扩展性能测试。

## 3. 随机种子与可复现性

### 3.1 Pilot 与 formal seeds

- Pilot 参数校准：seeds 100–104，共 5 个 seed；
- Formal 正式实验：seeds 0–19，共 20 个 seed。

对于 formal seed $s$：

- 问题实例 seed 为 $100000+s$；
- 数据划分 seed 为 $200000+s$；
- 各算法内部随机数使用方法特定的哈希 seed。

同一个 formal seed 下，NOG 与 ME-DOL 使用相同的问题实例、相同的数据划分和相同的评估样本库，但算法内部随机流相互独立。这样既保留了方法自身随机性，又允许进行 paired comparison。

## 4. 驻点代理、首次命中和评估规则

### 4.1 驻点代理

实验没有直接精确计算 Goldstein subdifferential 到原点的距离，而是使用平滑后目标的 Monte Carlo 梯度范数作为 stationarity proxy。

每次评估使用：

$$
B_{\mathrm{smooth}}^{\mathrm{eval}}=256,
\qquad
B_{\mathrm{data}}^{\mathrm{eval}}=512,
$$

所以一次 checkpoint 评估包含

$$
256\times512=131072
$$

次评估 oracle 样本。相同 seed 下，评估 bank 对方法独立固定，降低了 NOG 与 ME-DOL 比较中的评估噪声。

该指标应准确表述为“Goldstein stationarity 的高精度经验代理”，不能写成精确的 Goldstein 距离。

### 4.2 Confirmed first hit

为避免单个 checkpoint 因 Monte Carlo 噪声偶然低于阈值，只有连续两个 checkpoint 都满足

$$
\widehat{\operatorname{stat}}(x)\le\epsilon
$$

时，才确认该阈值已被命中；报告的 first-hit depth 是这两个 checkpoint 中第一个 checkpoint 的位置。

### 4.3 实际评估间隔和预算

正式 runner 的覆盖值决定了实际运行设置：

- NOG 每 2 个 depth layer 评估一次，共 480 行轨迹，最大 960 rounds；
- ME-DOL 每 6 个 depth layer 评估一次，共 640 行轨迹，最大 3840 rounds。

基础 YAML 中出现的 `eval_interval: 24` 不是正式实验最终使用的有效评估间隔；正式结果应以 runner 覆盖和已保存的轨迹为准。

## 5. 两种算法的具体运行方式

### 5.1 NOG-FO

正式冻结参数为：

| 参数 | 数值 |
|---|---:|
| $M$ | 2 |
| $\eta$ | 1.0 |
| `smooth_B` | 1 |
| 最大 rounds | 960 |
| global batch | 8 或 16 |

平滑扰动半径为

$$
D=\delta/M=0.05.
$$

每一步采用乐观梯度型更新，核心形式为

$$
\Delta_t
=\Pi\!\left(
\Delta_{t-1}-2\eta g_{t-1}+\eta g_{t-2}
\right),
$$

其中 $\Pi$ 表示投影操作。算法在局部扰动点 $y_t$ 处查询随机梯度，更新内部点 $x_t$，并以每个长度为 $M$ 的 block 的平均点 $\bar y$ 作为输出候选。

实现中 NOG 在正式迭代前还有两次初始化 oracle 调用，因此达到预算上限时记录的最大 depth 约为 962，而不是恰好 960。

### 5.2 ME-DOL-FO

正式冻结参数为：

| 参数 | 数值 |
|---|---:|
| `epoch_length` | 6 |
| `radius_multiplier` | 100 |
| per-worker batch | 1 |
| worker 数 | 8 |
| 最大 rounds | 3840 |

每个 depth layer 的总 work 为 8。其有效半径为

$$
r
=\frac{100\times0.1}{4\times6\times\sqrt 8}
\approx0.1473,
$$

对应学习率再除以 $\sqrt 6$，约为

$$
\eta_{\mathrm{ME}}\approx0.0601.
$$

ME-DOL 每 6 步结束一个 epoch，epoch 内各 rank 在本地计算随机梯度，通过 all-reduce 得到全局平均梯度，完成 online-learning 更新，并在 epoch 结束时重启局部状态和产生输出点。

## 6. 参数选择过程

### 6.1 第一阶段粗网格

初始参数网格包括：

- NOG：$M\in\{2,4,8,12\}$，$\eta\in\{0.003,0.01,0.03,0.1,0.3\}$，`smooth_B=2`，`global_batch=64`；
- ME-DOL：`epoch_length` $\in\{6,12,24\}$，`radius_multiplier` $\in\{0.3,1,3,10\}$。

Pilot 代表精度区域为：

- coarse：$0.2,0.1,0.05$；
- medium：$0.04,0.02,0.015,0.01$；

- fine：$0.009,0.008,0.005,0.002$。

初始搜索大致偏好：

- coarse 区域：NOG $M=2,\eta=0.1$；
- medium 区域：NOG $M=4,\eta=0.3$；
- fine 区域：NOG $M=8,\eta=0.3$；
- ME-DOL 的较优候选逐渐到达 `radius_multiplier=10` 的搜索边界，提示原网格太窄。

### 6.2 边界扩展与精细搜索

随后扩展了参数范围：

- NOG 加入 $\eta\in\{1,3,10\}$ 和 `smooth_B` $\in\{1,4,8\}$；
- ME-DOL 加入 `radius_multiplier` $\in\{30,100,150,200,300\}$。

最终在 pilot 结果基础上冻结：

- NOG：$M=2,\eta=1,\text{smooth\_B}=1$；
- ME-DOL：`epoch_length=6, radius_multiplier=100`。

必须诚实说明：这些是经过 pilot 校准后在已搜索网格内表现较好的固定配置，可以称为“pilot-calibrated locally good configurations”，但不能称为经过数学证明的全局最优参数。早期分区搜索和最终统一冻结之间也不是一次完全穷举的联合最优化。

### 6.3 NOG batch schedule

NOG 另外在

$$
B\in\{8,16,24,32,40,48,56,64\}
$$

中选择 global batch。选择规则要求：

1. pilot 的 5 个 seed 全部命中；
2. $\epsilon$ 变小时 batch 不下降；
3. 最小化各点 work ratio 对 1 的对数距离，并对 batch 切换次数施加 0.05 的惩罚。

最终得到两段式 schedule：

- $0.2$ 至 $0.0105$：batch 8；
- $0.01025$ 及以下：batch 16。

理论建议的 batch 随 $\epsilon^{-4/3}$ 增长，但当前实验只使用两档离散 batch，因此不能把该 schedule 当作理论连续标度律的直接实现。

## 7. 正式实验流程

### 7.1 任务组成

正式实验实际只需运行 60 条独立轨迹：

- NOG，batch 8：20 个 seed；
- NOG，batch 16：20 个 seed；
- ME-DOL：20 个 seed。

同一条轨迹会被复用于多个 $\epsilon$ 阈值：对每个阈值，从完整 trajectory 中寻找首次 confirmed hit。因而表格中的 27 个 $\epsilon$ 点不是 27 倍独立训练成本，但同一 seed 内不同 $\epsilon$ 点高度相关。

### 7.2 统计量

表中的绝对 depth/work 是 20 个 seed 的均值。比值采用逐 seed 配对后再取均值：

$$
R_D(\epsilon)
=\frac1{20}\sum_{s=1}^{20}
\frac{D_{\mathrm{ME},s}(\epsilon)}
{D_{\mathrm{NOG},s}(\epsilon)},
$$

$$
R_W(\epsilon)
=\frac1{20}\sum_{s=1}^{20}
\frac{W_{\mathrm{NOG},s}(\epsilon)}
{W_{\mathrm{ME},s}(\epsilon)}.
$$

因此“平均比值”一般不等于“两个绝对均值相除”。置信区间采用 2000 次 bootstrap。例如在 $\epsilon=0.01$ 上，depth ratio 为 1.919，95% bootstrap CI 为 $[1.650,2.226]$。

### 7.3 work 与 depth 的结构性耦合

当前实现中：

- ME-DOL 每层 work 固定为 8，因此 $W_{\mathrm{ME}}=8D_{\mathrm{ME}}$；
- NOG batch 8 时，$W_{\mathrm{NOG}}=8D_{\mathrm{NOG}}$；
- NOG batch 16 时，$W_{\mathrm{NOG}}=16D_{\mathrm{NOG}}$。

所以逐 seed 有精确关系：

$$
R_W=\frac1{R_D}
\quad\text{（NOG batch 8）},
$$

$$
R_W=\frac2{R_D}
\quad\text{（NOG batch 16）}.
$$

这意味着本实验是一个“近似 matched-work 的 depth 对比设计”。work ratio 与 depth ratio 并不是两条统计独立的证据，所以不能仅凭当前 work 曲线独立证明两种算法具有相同的 $\epsilon^{-3}$ work 指数。

## 8. v4 主实验结果：$0.2\ge\epsilon\ge0.01$

下表列出所有 27 个主实验点。所有点上 NOG 与 ME-DOL 都是 20/20 confirmed hits。

| $\epsilon$ | NOG batch | NOG depth | ME depth | ME/NOG depth | NOG work | ME work | NOG/ME work |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20000 | 8 | 135.8 | 66.3 | 0.49x | 1,086.4 | 530.4 | 2.05x |
| 0.18000 | 8 | 162.6 | 82.5 | 0.51x | 1,300.8 | 660.0 | 1.97x |
| 0.16000 | 8 | 187.2 | 96.0 | 0.51x | 1,497.6 | 768.0 | 1.95x |
| 0.14000 | 8 | 211.2 | 111.0 | 0.53x | 1,689.6 | 888.0 | 1.90x |
| 0.12000 | 8 | 235.6 | 127.5 | 0.54x | 1,884.8 | 1,020.0 | 1.85x |
| 0.10000 | 8 | 260.5 | 146.1 | 0.56x | 2,084.0 | 1,168.8 | 1.78x |
| 0.09000 | 8 | 274.1 | 156.9 | 0.57x | 2,192.8 | 1,255.2 | 1.75x |
| 0.08000 | 8 | 288.6 | 168.6 | 0.58x | 2,308.8 | 1,348.8 | 1.71x |
| 0.07000 | 8 | 304.9 | 182.4 | 0.60x | 2,439.2 | 1,459.2 | 1.67x |
| 0.06000 | 8 | 323.3 | 198.0 | 0.61x | 2,586.4 | 1,584.0 | 1.63x |
| 0.05000 | 8 | 345.7 | 217.2 | 0.63x | 2,765.6 | 1,737.6 | 1.59x |
| 0.04000 | 8 | 372.4 | 241.5 | 0.65x | 2,979.2 | 1,932.0 | 1.54x |
| 0.03000 | 8 | 408.4 | 275.1 | 0.67x | 3,267.2 | 2,200.8 | 1.49x |
| 0.02500 | 8 | 429.8 | 297.6 | 0.69x | 3,438.4 | 2,380.8 | 1.45x |
| 0.02000 | 8 | 459.9 | 330.9 | 0.72x | 3,679.2 | 2,647.2 | 1.39x |
| 0.01800 | 8 | 476.6 | 347.7 | 0.73x | 3,812.8 | 2,781.6 | 1.38x |
| 0.01600 | 8 | 492.4 | 369.0 | 0.75x | 3,939.2 | 2,952.0 | 1.34x |
| 0.01500 | 8 | 499.2 | 381.3 | 0.77x | 3,993.6 | 3,050.4 | 1.31x |
| 0.01400 | 8 | 508.0 | 394.8 | 0.78x | 4,064.0 | 3,158.4 | 1.29x |
| 0.01300 | 8 | 518.9 | 418.8 | 0.81x | 4,151.2 | 3,350.4 | 1.25x |
| 0.01200 | 8 | 538.9 | 446.1 | 0.83x | 4,311.2 | 3,568.8 | 1.22x |
| 0.01150 | 8 | 546.0 | 468.3 | 0.86x | 4,368.0 | 3,746.4 | 1.18x |
| 0.01100 | 8 | 566.2 | 535.8 | 0.95x | 4,529.6 | 4,286.4 | 1.09x |
| 0.01075 | 8 | 574.1 | 590.4 | 1.04x | 4,592.8 | 4,723.2 | 1.02x |
| 0.01050 | 8 | 587.2 | 642.9 | 1.10x | 4,697.6 | 5,143.2 | 0.98x |
| 0.01025 | 16 | 400.4 | 685.2 | 1.72x | 6,406.4 | 5,481.6 | 1.25x |
| 0.01000 | 16 | 408.9 | 783.0 | **1.92x** | 6,542.4 | 6,264.0 | **1.14x** |

### 8.1 主趋势

1. **命中率完整。** 27 个点上双方均为 20/20 hits，因此这些比例均为真实 first-hit 配对比例，不涉及删失值。
2. **depth ratio 严格单调上升。** 从 $\epsilon=0.2$ 的 0.488 上升到 $\epsilon=0.01$ 的 1.919，Spearman 相关系数为 1.0。
3. **发生优势交叉。** 在大 $\epsilon$ 区域 ME-DOL 的 depth 更小；约从 $\epsilon=0.01075$ 开始，NOG 的 depth 才更小。因此不能写“NOG 在所有精度下都优于 ME-DOL”。
4. **work 保持同一数量级。** NOG/ME work 的平均值约 1.489，范围 0.980–2.049，变异系数约 0.208。它不是严格常数，但整体没有随精度提高而发生数量级爆炸。

### 8.2 batch 切换的影响

从 $\epsilon=0.0105$ 到 $0.01025$ 时，NOG batch 从 8 切换为 16。因此：

- NOG 的 first-hit depth 从 587.2 降到 400.4；
- NOG 的 work 从 4697.6 上升到 6406.4；
- depth ratio 从 1.10 跳到 1.72；
- work ratio 从 0.98 回到 1.25。

这个跳变同时包含“精度改变”和“算法 batch 改变”两种因素，不能全部归因于 $\epsilon$。论文图中应标出 batch 切换边界，或分别拟合两个 batch 区域。

### 8.3 预注册判据

| 判据 | 目标 | 实际 | 结论 |
|---|---:|---:|---|
| 每方命中数 | $\ge18/20$ | 20/20 | 通过 |
| depth ratio 与精度难度的 Spearman $\rho$ | $\ge0.7$ | 1.0 | 通过 |
| 小 $\epsilon$ 端点 depth 优势 | $>1$ | 1.919 | 通过 |
| work ratio 下界 | $\ge0.5$ | 0.980 | 通过 |
| work ratio 上界 | $\le2$ | 2.049 | **未通过** |
| work ratio CV | $\le0.25$ | 0.208 | 通过 |

因为 work 上界轻微超过预注册的 2.0，严格的综合判定不是“全部通过”，而应写成“大部分预注册趋势判据得到支持，但 work 上界判据轻微失败”。

## 9. 经验幂指数与原文理论对照

在当前有限区间内进行 log-log 拟合，结果如下：

| 量 | 理论 $\epsilon$ 指数 | 观测指数 |
|---|---:|---:|
| NOG depth | 1.667 | 0.372 |
| ME-DOL depth | 3.000 | 0.636 |
| ME/NOG depth ratio | 1.333 | 0.266 |
| NOG work | 3.000 | 0.429 |
| ME-DOL work | 3.000 | 0.636 |
| NOG/ME work ratio | 0.000 | -0.193 |

观测指数与最坏情形理论指数差距很大，原因至少包括：

1. 理论给出的是最坏情形上界，不要求一个有限维合成实例达到该上界；
2. $\epsilon\in[0.01,0.2]$ 未必进入渐近区间；
3. 测试问题较简单，尤其 $R=1$；
4. NOG 的 $M,\eta$ 被固定，没有随理论最优规律连续缩放；
5. batch 只使用 8 和 16 两档，而非严格按 $\epsilon^{-4/3}$ 缩放；
6. 实验测量的是 stationarity proxy，而不是精确 Goldstein distance；
7. confirmed-hit checkpoint 离散化会影响首次命中深度。

所以本实验能够支持“relative depth advantage 随精度提高而增强”这一方向，但不能声称已经数值验证 $5/3$、$3$ 或 $4/3$ 的理论指数。

## 10. $\epsilon<0.01$ 的扩展预算实验

### 10.1 为什么必须扩展预算

原 v4 上限为：

- NOG：960 rounds；
- ME-DOL：3840 rounds。

在 $\epsilon<0.01$ 时，越来越多轨迹无法在该预算内 confirmed hit。因此后续实验保持 v4 算法参数、数据、seed、评估规则和 batch schedule 不变，只扩大最大运行预算。

### 10.2 两阶段扩展

| 阶段 | NOG 最大 rounds | ME-DOL 最大 rounds |
|---|---:|---:|
| 原 v4 | 960 | 3,840 |
| Stage 1 | 3,840 | 15,360 |
| Stage 2 | 15,360 | 61,440 |

Stage 2 实际记录的 NOG censoring depth 约为 15,362，原因是两次初始化调用；对应最大 work 为 245,792。ME-DOL 最大 depth 为 61,440，最大 work 为 491,520。

Stage 2 为控制资源改成单任务并发；这只改变调度速度，不改变单条轨迹的算法语义。

### 10.3 前缀一致性验证

扩展轨迹在旧 v4 预算范围内的前缀，与原 v4 轨迹逐行比较。40 条 NOG 前缀任务、共 22,400 行旧预算内记录，在忽略 wall-clock timing 字段后完全一致。

这说明扩展实验确实是原 v4 轨迹的预算延长，而不是改参数后生成的另一组实验。

### 10.4 扩展实验命中情况

| $\epsilon$ | NOG hit | ME-DOL hit | 可解释性 |
|---:|---:|---:|---|
| 0.0095 | 20/20 | 20/20 | 可报告完整 first-hit 比例 |
| 0.0090 | 20/20 | 20/20 | 可报告完整 first-hit 比例 |
| 0.0085 | 20/20 | 16/20 | ME-DOL 存在删失 |
| 0.0080 | 20/20 | 1/20 | ME-DOL 严重删失 |
| 0.0075 | 20/20 | 0/20 | ME-DOL 全部删失 |
| 0.0070 | 16/20 | 0/20 | 双方均有删失 |
| 0.0065 | 3/20 | 0/20 | 双方严重删失 |
| 0.0060 及以下 | 0/20 | 0/20 | 无有限 first-hit 比例 |

### 10.5 两个完整命中点

#### $\epsilon=0.0095$

- NOG depth：$418.7\pm38.8$；
- ME-DOL depth：$1316.4\pm658.8$；
- paired ME/NOG depth ratio：$3.135\pm1.461$；
- NOG work：$6699.2\pm620.4$；
- ME-DOL work：$10531.2\pm5270.7$；
- paired NOG/ME work ratio：$0.772\pm0.349$。

#### $\epsilon=0.0090$

- NOG depth：$451.4\pm68.7$；
- ME-DOL depth：$3823.2\pm3316.7$；
- paired ME/NOG depth ratio：$8.792\pm8.102$；
- NOG work：$7222.4\pm1099.7$；
- ME-DOL work：$30585.6\pm26533.6$；
- paired NOG/ME work ratio：$0.478\pm0.388$。

这两个点延续了 depth ratio 上升的方向，但 $\epsilon=0.0090$ 上 ME-DOL 的 seed 间方差已经很大。因此它适合用于说明“小 $\epsilon$ 下 NOG 的 depth 优势更加明显”，不适合用于精确拟合理论指数。

### 10.6 如何解释删失结果

从 $\epsilon=0.0085$ 起，至少一方不能在预算内对所有 seed 命中。此时必须区分：

- **真实 first-hit ratio**：双方同一 seed 均命中时才能计算；
- **conditional-on-hit mean**：只对命中的 seed 求均值，会产生 survivor bias；
- **capped mean**：把 non-hit 记为预算上限，只能反映预算内的下界或删失情况，不是真实 first-hit 均值；
- **无有限比例**：一方或双方 0/20 hits 时，不能报告普通 first-hit ratio。

因此 $0.0085$ 以下结果主要用于展示 hit rate、预算压力和 censoring boundary，不能与主表中的完整命中比例等价混用。

## 11. 哪些结果与理论一致，哪些不能过度解释

### 11.1 与理论方向一致的部分

1. 主区间内 ME/NOG depth ratio 从 0.49 严格上升到 1.92；
2. $0.0095$ 和 $0.0090$ 的完整命中扩展点分别达到约 3.14 和 8.79；
3. 精度提高后出现从 ME-DOL depth 更优到 NOG depth 更优的交叉；
4. 主区间 work ratio 保持在约 1–2 的同一数量级，没有出现随 $\epsilon$ 缩小而数量级恶化的现象；
5. 扩展实验中 ME-DOL 比 NOG 更早出现大量 non-hit，与 NOG 具有更好 depth scaling 的理论方向相符。

### 11.2 不能声称已经验证的部分

1. 不能声称观测到了精确的 $\epsilon^{-5/3}$、$\epsilon^{-3}$ 或 $\epsilon^{-4/3}$ 标度；
2. 不能声称 NOG 在所有 $\epsilon$ 上都优于 ME-DOL；
3. 不能声称当前参数是两种算法的全局最优参数；
4. 不能把删失点的 capped mean 当成真实 first-hit mean；
5. 不能把 work ratio 曲线当作独立于 depth ratio 的第二条证据；
6. 不能把单机多进程结果表述为真实多机 wall-clock 加速；
7. 不能把 stationarity proxy 表述为精确 Goldstein distance。

## 12. 公平性与可复现性评估

### 12.1 公平性较强的地方

- 相同 seed 使用相同问题、相同数据划分和相同评估 bank；
- 两种方法使用相同 worker 数、同一种通信后端和相同硬件环境；
- 参数在 formal seeds 之前由独立 pilot seeds 校准并冻结；
- 正式表采用逐 seed paired ratios；
- 每个点使用 20 个 formal seeds；
- confirmed-hit 规则降低了单次评估噪声；
- 扩展实验通过 trajectory prefix validation 证明没有偷偷改动 v4 配置；
- 原始轨迹、per-seed 表、summary、ratio、audit 和重现实验说明均保存在仓库中。

### 12.2 仍然存在的限制

- 参数搜索不是严格全局穷举；
- NOG 单独使用了 batch schedule，而 ME-DOL 的 per-worker batch 固定；
- batch schedule 的目标函数显式偏向 work matching，因此 work 结果带有设计因素；
- 主区间有一个 batch 切换点，使 depth 曲线产生结构性跳变；
- 测试问题只有一个、且 $R=1$；
- 只在单机 CPU 多进程环境中运行；
- 极小 $\epsilon$ 区域存在严重右删失；
- 评估的是代理指标而非精确 Goldstein stationarity。

综合判断：该实验是**可复现、配对设计清楚、适合展示理论定性趋势的验证实验**；若要作为论文中强有力的复杂度实证，还需要增加更多问题实例、$R>1$ 的更强非光滑性、更多维度、独立的参数敏感性实验，以及针对删失数据的生存分析或更大预算。

## 13. 推荐的论文表述

### 13.1 推荐中文表述

> 在统一的合成非光滑非凸问题、固定的 pilot 校准参数和 20 个配对随机种子上，ME-DOL-FO 相对 NOG-FO 的首次命中 depth 比例随目标精度提高而单调上升。在完整命中的主区间内，该比例从 $\epsilon=0.2$ 时的 0.49 增加到 $\epsilon=0.01$ 时的 1.92；扩展预算后，在 $\epsilon=0.0095$ 和 $0.0090$ 上进一步达到 3.14 和 8.79。与此同时，主区间内两者 work 保持在同一数量级。该现象与理论所预测的 NOG 更优并行 depth scaling 方向一致，但有限区间拟合未复现最坏情形渐近指数，且极小精度处存在明显右删失。

### 13.2 Recommended English wording

> Under a common synthetic nonsmooth nonconvex problem, pilot-calibrated frozen configurations, and 20 paired random seeds, the first-hit depth ratio of ME-DOL-FO to NOG-FO increases monotonically as the target tolerance becomes smaller. Over the fully observed primary range, the ratio rises from 0.49 at $\epsilon=0.2$ to 1.92 at $\epsilon=0.01$; with extended budgets, it further reaches 3.14 and 8.79 at $\epsilon=0.0095$ and $0.0090$, respectively. The two methods remain within the same order of total oracle work over the primary range. These observations are qualitatively consistent with the predicted depth-scaling advantage of NOG, but they do not recover the worst-case asymptotic exponents, and the smaller-tolerance regime is subject to substantial right censoring.

### 13.3 不推荐的表述

不要写：

- “实验严格证明了 NOG 的 $\epsilon^{-5/3}$ depth complexity”；
- “NOG 在所有精度上都优于 ME-DOL”；
- “两种算法都使用了全局最优参数”；
- “$\epsilon\le0.0085$ 的 capped ratio 就是真实 first-hit ratio”；
- “work ratio 恒定，因此已经独立验证了 work 理论”；
- “该实验验证了多机并行加速”。

## 14. 复现实验

建议首先阅读：

- [v4 主实验复现说明](results/theory_validation_v4/REPRODUCE.md)
- [扩展预算复现说明](results/theory_validation_v4_extended_budget/REPRODUCE.md)
- [主配置](results/theory_validation_v4/config.yaml)
- [冻结参数](results/theory_validation_v4/frozen_parameters.json)

在仓库根目录、`NOG` conda 环境中，可以按阶段执行：

```bash
conda run -n NOG python -m src.distributed.theory_validation_runner pilot-batch-grid
conda run -n NOG python -m src.distributed.theory_validation_runner pilot
conda run -n NOG python -m src.distributed.theory_validation_runner formal
conda run -n NOG python -m src.distributed.theory_validation_runner analyze
conda run -n NOG python -m src.distributed.theory_validation_runner audit
```

具体参数和扩展预算命令应以两个 `REPRODUCE.md` 中的当前内容为准，因为扩展实验使用独立输出目录，并包含前缀验证步骤。

## 15. 原始证据与结果文件索引

### 15.1 v4 主实验

- [README 总览](README.md)
- [论文 PDF](NeurIPS_NOG.pdf)
- [实验配置](results/theory_validation_v4/config.yaml)
- [冻结参数](results/theory_validation_v4/frozen_parameters.json)
- [Pilot 校准记录](results/theory_validation_v4/pilot_calibration.csv)
- [Formal per-seed 结果](results/theory_validation_v4/analysis/formal_per_seed.csv)
- [Formal 绝对量汇总](results/theory_validation_v4/analysis/formal_summary.csv)
- [Formal paired ratios](results/theory_validation_v4/analysis/formal_ratios.csv)
- [趋势统计](results/theory_validation_v4/analysis/formal_trends.json)
- [Formal 审计结果](results/theory_validation_v4/audit/formal_result_audit.json)

### 15.2 低 $\epsilon$ 扩展实验

- [扩展预算分析报告](results/theory_validation_v4_extended_budget/stage2/analysis/extended_budget_report.md)
- [扩展绝对量汇总](results/theory_validation_v4_extended_budget/stage2/analysis/extended_summary.csv)
- [扩展 paired ratios](results/theory_validation_v4_extended_budget/stage2/analysis/extended_ratios.csv)
- [轨迹前缀一致性验证](results/theory_validation_v4_extended_budget/stage2/analysis/prefix_validation.json)

## 16. 最终结论

当前 v4 及其扩展结果最可靠的结论不是“实验精确复现了理论幂指数”，而是：

1. 在完整命中的主区间内，随着 $\epsilon$ 减小，ME-DOL 相对 NOG 的 depth 成本呈稳定、严格单调的恶化趋势；
2. 在 $\epsilon=0.01075$ 附近出现 depth 优势交叉，之后 NOG 开始占优；
3. 扩展预算后，$0.0095$ 和 $0.0090$ 两个完整命中点继续强化这一方向；
4. 主区间内 work 仍处于相同数量级，但该现象部分来自 batch matching 设计，不能作为独立的指数验证；
5. 更小 $\epsilon$ 下的 non-hit 是右删失证据，说明预算压力迅速增大，而不是可以随意替代成有限 first-hit ratio 的普通数据点。

因此，该实验可以作为论文中的**理论趋势验证与机制性证据**，但应同时披露 batch 切换、参数搜索边界、代理驻点指标、work-depth 耦合和低精度删失等限制。

# ZO 分布式实验计划

状态：Step ZO-2A～2E、ZO-3A～3F、ZO-4A～4C、ZO-5A～5C、ZO-6A～6C、ZO-7A～7C、ZO-8A～8C 已完成；当前无后台 ZO 实验。
适用论文：[NeurIPS_NOG.pdf](NeurIPS_NOG.pdf) Section 5  
主要比较方法：`NOG-ZO`、`ME-DOL-ZO`、`DGFM`、`DGFM+`

## 1. 实验目标

本文 Section 5 给出的 zeroth-order 分布式复杂度主阶如下。表中省略了 Lipschitz 常数、初始函数差距及网络相关常数，只保留维度 \(d\)、Goldstein 邻域半径 \(\delta\) 和精度 \(\epsilon\) 的依赖。

| 方法 | communication depth | total SZO work |
|---|---:|---:|
| NOG-ZO | \(O(d^{1/3}\delta^{-1}\epsilon^{-5/3})\) | \(O(d\delta^{-1}\epsilon^{-3})\) |
| ME-DOL-ZO | \(O(d\delta^{-1}\epsilon^{-3})\) | \(O(d\delta^{-1}\epsilon^{-3})\) |
| DGFM | \(O(d^{3/2}\delta^{-1}\epsilon^{-4})\) | \(O(d^{3/2}\delta^{-1}\epsilon^{-4})\) |
| DGFM+ | \(O(d^{3/2}\delta^{-1}\epsilon^{-3})\) | \(O(d^{3/2}\delta^{-1}\epsilon^{-3})\) |

主实验需要验证：

1. 在固定 \(d\) 和 \(\delta\) 时，NOG-ZO 的 depth 随 \(\epsilon\) 减小增长得比 ME-DOL-ZO、DGFM 和 DGFM+ 更慢；
2. ME-DOL-ZO/NOG-ZO 的 depth 比例整体满足随 \(\epsilon\) 减小而上升的趋势，理论参考为
   \[
   \frac{D_{\mathrm{ME}}}{D_{\mathrm{NOG}}}
   \propto d^{2/3}\epsilon^{-4/3};
   \]
3. NOG-ZO 和 ME-DOL-ZO 都具有 \(O(d\delta^{-1}\epsilon^{-3})\) work，因此两者 work 比例可以相差常数，但不应随 \(\epsilon\) 呈明显幂次增长；
4. NOG-ZO 的维度依赖优于 DGFM/DGFM+，并且在增加 workers 后保持 total work 基本稳定、per-worker work 近似按 \(1/m\) 缩放。

实验的目标是检验这些渐近趋势，而不是要求有限样本下每两个相邻 \(\epsilon\) 点都严格单调。

## 2. 四种算法和统一计数口径

### 2.1 Two-point SZO estimator

四种方法统一使用两点估计器

\[
g(x;v,\xi)=\frac{d}{2h}
\left(F(x+hv;\xi)-F(x-hv;\xi)\right)v,
\qquad v\sim\operatorname{Unif}(\mathbb S^{d-1}).
\]

正式实现必须满足：

- 正负两次函数查询共享同一个数据样本 \(\xi\) 和方向 \(v\)；
- batch 中不同 estimator samples 使用独立的 \((v,\xi)\)；
- 一个 two-point sample 严格计为 2 次 SZO calls；
- DGFM+ 普通 SPIDER difference 同时估计当前位置和上一位置，严格计为 4 次 SZO calls；
- training SZO work 与 evaluation work 分开记录；
- total work 是所有 workers 上函数查询的总和，per-worker work 单独报告。

### 2.2 Communication depth

- NOG-ZO 的两次初始化 oracle 及其 global aggregation 均计入；
- ME-DOL-ZO 按算法依赖关系统计每个 inner iteration 的 communication depth；
- DGFM 的 gradient-tracker mixing 和 parameter mixing 分别审计；
- DGFM+ 的 restart mixing、tracker mixing 和 parameter mixing 分别审计；
- 同时记录 dependency depth、raw collective calls 和通信 payload，主理论图使用 dependency depth。

### 2.3 共同 Goldstein 半径

四种算法必须对应相同的最终 \((\delta_G,\epsilon)\)-Goldstein 目标。NOG 定理 4.2 严格输出 \((2h,\epsilon)\)-Goldstein stationary point，因此若目标半径为 \(\delta_G\)，NOG 内部 smoothing radius 应为 \(h=\delta_G/2\)。ME-DOL-ZO、DGFM 和 DGFM+ 的内部 smoothing radius 也必须根据各自原论文统一映射，不能仅让四个配置字段具有相同数值。

## 3. 主问题设置

计划使用与 FO v4 相同的合成问题，使 FO/ZO 的问题尺度可以对照，但 ZO 配置、输出和结论完全独立保存：

```yaml
problem:
  name: SyntheticMaxSinL1
  d: 100
  n_data: 4096
  R: 1
  lam: 0.001
  feature_scale: 1.0
  common_feature_bias: 0.25
  phase_mode: zero
```

主设置：

- reference workers：\(m=8\)；
- topology：complete graph/global aggregation；
- 初始点：\(x_0=0\)；
- 共同 problem seeds 和固定 evaluation sample bank；
- 主实验测量算法的 depth/work，不将单机 wall-clock time 解释为真实集群加速。

## 4. 完整 epsilon 网格

ZO 使用和 FO v4 完全相同的 threshold 列表：

\[
\begin{aligned}
\epsilon\in\{&0.2,0.18,0.16,0.14,0.12,0.1,0.09,0.08,0.07,0.06,\\
&0.05,0.04,0.03,0.025,0.02,0.018,0.016,0.015,0.014,0.013,\\
&0.012,0.0115,0.011,0.01075,0.0105,0.01025,0.01,0.0095,\\
&0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002\}.
\end{aligned}
\]

从同一训练轨迹提取更多 first-hit thresholds 通常不会显著增加训练时间。真正昂贵的是为每个 \(\epsilon\) 单独调参和重新训练，因此采用以下分层策略：

- 全部 36 个 \(\epsilon\) 都进入最终 threshold、hit-rate 和 ratio 表；
- 只用少数代表点进行参数缩放和预算校准；
- 初定代表点为
  \[
  \{0.2,0.1,0.05,0.02,0.01,0.009,0.008,0.005,0.002\};
  \]
- 如果大 \(\epsilon\) 在初始 evaluation 已命中，则保留并标注为 trivial hit，不用于复杂度斜率拟合；
- 如果低 \(\epsilon\) 全部 non-hit，则先扩展预算，不删除这些点。

## 5. Seed、evaluation 和统计规则

### 5.1 Seed 划分

- pilot seeds：`100,101,102,103,104`；
- formal seeds：`0` 至 `19`，共 20 个；
- anomaly seeds：额外 5 个未使用 seeds，仅在满足异常复核规则时使用；
- formal seeds 不得参与参数选择。

### 5.2 Evaluation

- 使用独立、高精度的 first-order smoothed-gradient norm 作为 Goldstein stationarity 的经验代理；
- evaluation 不能读取算法内部 ZO estimator；
- 对所有方法使用相同的 fixed evaluation bank；
- evaluation work 单独记录，不计入训练 SZO work；
- first hit 必须连续两个 evaluation checkpoints 满足
  \[
  \widehat{\operatorname{stat}}(x)\leq\epsilon;
  \]
- 该指标只能称为高精度经验 proxy，不能称为精确 Goldstein subdifferential distance。

### 5.3 统计结果

报告：

- mean、sample standard deviation、median；
- bootstrap 95% confidence interval，默认重复 2000 次；
- 相同 problem seed 下的 paired depth/work ratios；
- hit rate；
- hit-only first-hit 条件均值；
- capped/restricted mean 作为 censoring 敏感性分析；
- hit rate 太低时只报告 lower bound，不给虚假的有限比例。

## 6. 参数选择和冻结协议

### 6.1 NOG-ZO

根据定理 4.2，目标方差和 batch 主阶为

\[
\sigma=O(d^{1/6}L^{1/3}\epsilon^{2/3}),
\qquad
B=O(d^{2/3}L^{4/3}\epsilon^{-4/3}).
\]

根据定理 4.1 设置或缩放：

- block length \(M\)；
- projection radius \(D=h/M\)；
- learning rate \(\eta\)；
- total depth budget；
- SZO batch \(B\)。

Pilot 仅搜索理论公式外的少量常数乘子；常数冻结后跨所有 formal seeds 和 \(\epsilon\) 使用。

### 6.2 ME-DOL-ZO

从原论文重新核对并按理论设置：

- epoch length；
- domain radius；
- learning rate；
- smoothing radius；
- local ZO batch；
- mixing 参数。

### 6.3 DGFM

Pilot 搜索理论允许范围内的：

- learning rate；
- ZO batch size；
- smoothing-radius constant。

### 6.4 DGFM+

Pilot 搜索：

- learning rate；
- small batch；
- large batch；
- restart period；
- restart mixing rounds。

### 6.5 公平性约束

- 四种方法获得相近数量的 pilot candidate evaluations；
- 候选首先要求 pilot hit rate 达标，再比较 first-hit depth，depth 接近时选择 work 更少者；
- 同时保存完整 Pareto frontier；
- 正式运行开始前保存冻结配置、参数表和配置哈希；
- 禁止根据 formal 结果重新选取有利参数；
- 若理论缩放配置和逐点局部最优配置存在明显差距，两套结果分开报告，不能混成一条曲线。

## 7. 分阶段实验流程

### Step ZO-2：实现与理论审计

- ZO-2A：验证 two-point estimator 的公式、独立采样和数值正确性；
- ZO-2B：审计四种算法的 SZO work；
- ZO-2C：审计 communication dependency depth；
- ZO-2D：统一 \(\delta_G\) 和内部 smoothing radius；
- ZO-2E：完成短程 simulator/多进程等价测试。

### Step ZO-3：尺度与运行时间校准

- 测量 \(x_0\) 的高精度 proxy；
- 确定有信息量的 \(\epsilon\) 区间；
- 测量 960-round 运行时间；
- 验证预算阶梯 `960, 3840, 15360, 61440`；
- 评估是否需要最高 `245760` 的额外预算。

### Step ZO-4：Pilot 和冻结

- 在 5 个 pilot seeds 上完成粗搜索；
- 对 Pareto 前沿做局部细化；
- 冻结四种算法参数和理论缩放常数；
- 生成 pilot 报告及 formal manifest。

### Step ZO-5：正式 epsilon-scaling

在 20 个 formal seeds 上运行四种算法，生成：

1. stationarity proxy vs communication depth；
2. stationarity proxy vs total SZO work；
3. 全部 36 个 \(\epsilon\) 的 first-hit depth/work 和 hit rate；
4. ME-DOL-ZO/NOG-ZO depth ratio；
5. ME-DOL-ZO/NOG-ZO work ratio；
6. DGFM/NOG-ZO、DGFM+/NOG-ZO ratios；
7. log-log slope 和 bootstrap 95% CI。

执行状态：

- ZO-5A：80/80 个冻结配置任务完成，每种方法 20 个 formal seeds；
- ZO-5B：结果审计、全部 epsilon 的 first-hit/censoring 汇总、同 seed 配对 ratio、bootstrap 95% CI 和正式图已完成；
- ZO-5C：完成理论 ratio 指数对照、跨 epsilon 联合 seed bootstrap、删失边界和论文安全结论；
- 审计结果：80 个任务参数与 manifest 一致，depth/work 严格递增，最终 training work 均为 983,040；
- 正式结果入口：[zo_experiments/formal/README.md](zo_experiments/formal/README.md)；
- 理论解释入口：[zo_experiments/formal/STEP_ZO_5C_THEORY_COMPARISON.md](zo_experiments/formal/STEP_ZO_5C_THEORY_COMPARISON.md)；
- ZO-6A 诊断入口：[zo_experiments/formal/STEP_ZO_6A_ANOMALY_AUDIT.md](zo_experiments/formal/STEP_ZO_6A_ANOMALY_AUDIT.md)；
- ZO-6B：20/20 个冻结配置诊断任务完成（4 methods × seeds 200–204）；
- 运行进度保留在本地 `outputs/distributed_zo/zo_theory_validation/diagnostic/anomaly_seeds_fixed_work_983040/progress.json`；
- ZO-6C：完成 formal-vs-anomaly 对照；决定当前论文实验不继续扩展相同冻结配置的预算；
- 复现与决策：[zo_experiments/formal/STEP_ZO_6C_REPLICATION_DECISION.md](zo_experiments/formal/STEP_ZO_6C_REPLICATION_DECISION.md)；
- Step ZO-7A：dimension-scaling 设计与运行时间校准已完成；
- Step ZO-7B：240/240 个新 dimension tasks 已完成；
- Step ZO-7C：320/320 tasks 合并审计、paired ratios、bootstrap CI、图表和报告已完成；
- Dimension 结果入口：[zo_experiments/dimension/README.md](zo_experiments/dimension/README.md)。
- Step ZO-8A：16/16 个独立 worker calibration tasks 已完成；
- Step ZO-8B：240/240 个 m=1、2、4 正式 tasks 已完成，m=8 复用原 80 个正式 tasks；
- Step ZO-8C：320/320 tasks、194,080 行轨迹审计、bootstrap CI、图表和报告已完成；
- Worker 结果入口：[zo_experiments/worker/README.md](zo_experiments/worker/README.md)。

Step ZO-7A calibration 设置：

- dimensions：25、50、100、200；
- methods：NOG-ZO、ME-DOL-ZO、DGFM、DGFM+；
- calibration seed：300，与 pilot/formal/anomaly seeds 均不重叠；
- 每任务 training work：983,040；
- 参数策略：复用 d=100 冻结配置，不重新调参，仅用于命中范围、稳定性和运行时间校准；
- 任务数：16，原子 partial 支持断点续跑；
- 进度保留在本地 `outputs/distributed_zo/zo_theory_validation/dimension/calibration_fixed_params_work983040/progress.json`。

Step ZO-7B/7C 完成协议：

- primary epsilons：0.05、0.03；0.02及以下仅报告删失；
- d=25、50、200各运行四算法 × 20 formal seeds，共240个新任务；
- d=100复用已审计的原始20-seed formal结果；
- 参数不随dimension重新调优，结论限定为定性dimension sensitivity；
- 冻结manifest：[zo_experiments/dimension_scaling_manifest.json](zo_experiments/dimension_scaling_manifest.json)；
- ZO-7B 进度为 240/240，原始进度保留在本地 output 目录；
- ZO-7C 合并 d=100 后审计 320/320 tasks、194,000 行轨迹；
- primary epsilon 在四维度、四方法上均为20/20 hits；
- 仅2/6个 relative depth slopes 的95% CI严格大于零，0/6包含理论参考幂次；
- 结论限定为 fixed-configuration dimension sensitivity，不声称恢复精确维度指数。

理论参考斜率：

| 方法 | depth 对 \(\epsilon\) 的斜率 | work 对 \(\epsilon\) 的斜率 |
|---|---:|---:|
| NOG-ZO | \(-5/3\) | \(-3\) |
| ME-DOL-ZO | \(-3\) | \(-3\) |
| DGFM | \(-4\) | \(-4\) |
| DGFM+ | \(-3\) | \(-3\) |

### Step ZO-6：异常复核

若某一区间的比例或斜率出现超出置信区间的突变：

1. 不修改 formal 冻结配置；
2. 检查 non-hit/censoring 和 checkpoint resolution；
3. 使用相同配置复跑；
4. 使用 anomaly seeds；
5. 必要时在异常区间加入中点；
6. 将异常和解释完整保留在报告中。

### Step ZO-7：dimension scaling

使用

\[
d\in\{25,50,100,200\}
\]

和 primary 精度

\[
\epsilon\in\{0.05,0.03\}
\]

其中 \(\epsilon=0.02\) 及以下只作删失描述。当前固定 d=100 参数的协议只检查
qualitative dimension sensitivity；精确检验 \(d^{1/3}\)、\(d\) 和 \(d^{3/2}\)
需要另行冻结 dimension-aware 参数缩放协议。

### Step ZO-8：worker scaling

使用

\[
m\in\{1,2,4,8\}
\]

和 primary 精度 \(0.05\)，检查：

- total work 是否基本稳定；
- per-worker work 是否接近 \(1/m\)；
- final stationarity 是否稳定；
- depth 是否没有随 worker 数量异常增长。

由于现有固定预算下部分方法不能稳定命中 \(0.01\) 和 \(0.005\)，这两个阈值不再作为
worker scaling 的 primary endpoint；\(0.02\) 及以下仅作删失描述。参数沿用
\(d=100,m=8\) 的预冻结配置，不针对 worker 数重新调参。评估 checkpoint 按每种方法的
训练 work 对齐：NOG-ZO 在所有 \(m\) 下使用间隔 4；ME-DOL-ZO 使用
\(\{96,48,24,12\}\)；DGFM/DGFM+ 使用 \(\{32,16,8,4\}\)，顺序均对应
\(m=\{1,2,4,8\}\)。ME-DOL 只能在完整 12-round epoch 末评估。

这里的 workers 是单进程中的逻辑 worker，结果用于检查算法和计费对 worker 数的敏感性，
不能解释为真实多进程 wall-clock speedup。

完成结果：

- ZO-8A 校准 seed 301 上 epsilon 0.05 为 16/16 method-worker hits；epsilon 0.03
  为 13/16，因此正式协议只保留 0.05 为 primary，0.03 及以下作为删失描述；
- ZO-8B 新运行 m=1、2、4 的四算法 × 20 seeds，共 240/240 tasks；
- ZO-8C 合并 m=8 后审计通过 320/320 tasks 和 194,080 行轨迹；
- NOG-ZO 的 mean first-hit depth 为 214.2、214.2、214.0、214.4，total work 为
  219,340.8、219,340.8、219,136.0、219,545.6；
- NOG-ZO per-worker work 从 219,340.8 降至 27,443.2，log-log slope 为
  -0.9997，95% CI 为 [-1.0011,-0.9984]；
- DGFM+/m=2 在正式 seeds 上为 18/20 hits，所有条件均值和比例明确标记为 censored。

### Step ZO-9：真实进程和可选真实数据

- 主复杂度实验使用可恢复的逻辑分布式模拟；
- 选取少量配置运行 8-process Gloo，验证与模拟器等价；
- 不把单机 Gloo wall-clock time 当作真实集群 speedup；
- 合成实验通过后，再决定是否加入 `a9a` 或 `ijcnn1` capped-\(\ell_1\) SVM 实验。

### Step ZO-10：整理、文档和上传

正式内容放入独立的 `zo_experiments/`，与 FO 隔离，包括：

- 冻结配置和配置哈希；
- pilot 参数表和 Pareto 结果；
- raw CSV、每 seed partial 和 resume manifest；
- environment 信息；
- depth/work/ratio/slope 图片；
- censoring 和 lower-bound 结果；
- 运行命令、结果解释和局限性；
- 最终 README 和 GitHub 提交。

## 8. 资源与后台运行约束

- 总 CPU 使用不超过 32 cores/processes；
- 逻辑模拟优先使用单 GPU；
- 每个后台任务最长不超过 36 小时；
- 所有 method/seed/config 单元独立保存 partial，支持断点续跑；
- 不依赖本地终端保持连接；
- 远程主机睡眠、暂停或被平台关机仍会中断进程，因此必须依靠 partials 和 resume，而不能只依靠 `nohup`；
- 合成正式实验预计在 5–7 天内完成，具体时间在 Step ZO-3 校准后更新。

## 9. 正式验收标准

1. 四种算法通过 oracle、work 和 depth 审计；
2. 代表 \(\epsilon\) 上多数方法具有可解释的 hit rate；
3. NOG-ZO 的 depth-vs-\(\epsilon\) 增长显著慢于至少 ME-DOL-ZO；
4. ME-DOL-ZO/NOG-ZO depth ratio 的整体趋势随 \(\epsilon\) 减小而上升；
5. NOG-ZO/ME-DOL-ZO work ratio 的 log-log 斜率接近 0，或其置信区间能够解释有限样本偏差；
6. 维度和 worker scaling 结果与理论方向一致，或对偏差给出可复现诊断；
7. non-hit、失败运行和不利结果全部保留，不通过正式 seed 事后调参；
8. 所有结论能够由冻结配置、原始结果和自动汇总脚本复现。

## 10. 启动前待确认项

启动 Step ZO-2A 前需要确认：

1. 主合成问题是否正式采用 FO v4 的 `R=1, common_feature_bias=0.25, phase_mode=zero` 设置；
2. 是否采用理论严格对齐的共同目标半径 \(\delta_G=0.1\)，并允许各算法按原定理使用不同的内部 smoothing radius；
3. 本轮范围是否先完成全部合成实验，再根据结果决定是否加入真实数据实验。


# NOG 实验仓库

本仓库用于 NOG 在非光滑非凸随机优化中的分布式实验。为避免 first-order（FO）与 zeroth-order（ZO）实验的文档和结论混杂，当前材料按 oracle 类型分开管理。

## 实验入口

- [FO 实验归档与主结果](fo_experiments/README.md)
- [FO 完整实验解释](fo_experiments/EXPERIMENT_DETAILED_EXPLANATION.md)
- [FO v4–v7 设定与结果对比](fo_experiments/V4_V7_EXPERIMENT_COMPARISON.md)
- [FO 实验计划与完成记录](fo_experiments/PLAN.md)
- [ZO 实验完整说明、正式结果与复现步骤](ZO-README.md)
- [ZO 正式结果与审计材料](zo_experiments/formal/README.md)
- [ZO 实验计划与执行记录](ZO_plan.md)
- [早期 SFO/SZO 共同分布式基线](SHARED_DISTRIBUTED_FO_ZO_BASELINE.md)

ZO 的固定参数 epsilon-scaling、异常复核与理论解释已经完成；固定配置的
dimension-scaling（Step ZO-7B）仍在运行。原始逐 checkpoint 轨迹保留在本地
`outputs/distributed_zo/`，GitHub 提交的是 `zo_experiments/` 中经过审计的紧凑表格、
图像、参数 manifest 和报告。

## 目录约定

| 路径 | 用途 |
|---|---|
| `fo_experiments/` | 已完成 FO 实验的报告、解释、版本对比与计划 |
| `zo_experiments/` | ZO 参数冻结、正式结果、审计报告与可提交图表 |
| `nog_iclr2027_complete_source/` | ICLR 论文 LaTeX 源文件与论文图像 |
| `results/` | 可提交、可审计的紧凑结果包和图像 |
| `outputs/` | 原始运行轨迹与中间输出，通常不提交 Git |
| `configs/` | FO、ZO 和公共实验配置 |
| `src/` | 算法、分布式 runner、分析和审计代码 |
| `scripts/` | 复现、绘图及辅助入口 |
| `tests/` | 实现和结果口径测试 |

## FO 与 ZO 的隔离原则

1. FO 正式文档统一从 `fo_experiments/` 进入；
2. 新 ZO 实验使用带 `zo` 的 run name、配置名和输出目录；
3. FO 与 ZO 使用不同类型的 oracle work，不能把数值直接合并比较；
4. 不移动现有 `results/`、`configs/` 和 `src/` 内的稳定路径，以保持历史复现命令和测试有效；
5. 早期同时包含 SFO/SZO 的逻辑分布式基线保留为共享历史材料，不纳入 FO v4 主结论。

环境安装方式见 [requirements.txt](requirements.txt)。当前实验通常使用 `NOG` conda 环境。

# SVM CPU joint-retuning result package

本目录整理了 NOG-ZO 在 a9a、ijcnn1 上联合搜索 `batch + M + eta` 后的完整可审计结果。实验设置、步骤、结果与讨论见根目录的 [`SVM_CPU_EXPERIMENT_REPORT_CN.md`](../SVM_CPU_EXPERIMENT_REPORT_CN.md)。

目录结构：

- `figures/`：完整线性纵轴图和严格 epsilon=0--0.05 放大图（PNG/PDF）；
- `data/`：pilot、validation、formal 逐 checkpoint 数据、95% CI、阈值首次命中和绘图数据；
- `audit/`：预登记协议、冻结参数、baseline 复用审计、资源与失败任务说明；
- `reproduction/`：本次实验使用的配置与脚本快照。

全部 search、validation 和 formal seeds 均保留，没有删除不利结果。包内文件哈希见 `SHA256SUMS.txt`。

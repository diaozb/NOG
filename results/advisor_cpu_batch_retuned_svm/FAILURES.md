# 任务状态与失败说明

- NOG formal：40/40 任务成功（a9a、ijcnn1 各 20 个 seed）。
- search：1,800/1,800 候选-seed 任务成功，非有限/发散候选数为 0。
- validation：30/30 候选-seed 任务成功，非有限/发散候选数为 0。
- 未删除任何正式 seed；formal seeds 固定为 0--19，search/validation 使用 400--404 和 500--504。
- 没有失败任务需要重跑或从汇总中排除。

每个 formal shard 保留 `partials/`、`environment.json`、`progress.json` 和汇总，原始 pilot partials 位于 `outputs/distributed_zo/zo_theory_validation/batch_retuned_svm/`。

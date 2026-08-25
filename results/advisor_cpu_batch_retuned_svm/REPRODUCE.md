# CPU 复现命令

工作目录为 `/data/diaozb/NOG`，Python 固定为 `/root/miniconda3/envs/NOG/bin/python`；不需要 `conda activate`，也不使用 GPU。

## 1. 协议、search 与 validation

协议和候选网格见 `protocol.json`。已完成的结果文件为 `pilot_grid.csv`（每个数据集 180 候选、search seeds 400--404）和 `pilot_validation.csv`（validation seeds 500--504）。冻结文件为 `frozen_parameters.json`。如需从头复跑，可使用：

```bash
/root/miniconda3/envs/NOG/bin/python scripts/zo_refine_pilot.py --help
/root/miniconda3/envs/NOG/bin/python scripts/score_svm_retuned_pilot.py --help
/root/miniconda3/envs/NOG/bin/python scripts/freeze_batch_retuned_svm.py --help
```

运行时必须使用 `configs/distributed_zo_batch_svm_a9a.yaml` 和 `configs/distributed_zo_batch_svm_ijcnn1.yaml`，CPU device，target work=983040，search/validation/formal seeds 不得混用。

## 2. NOG formal（断点可续跑）

每个 seed 单独一个 shard；以下示例运行 seed 0，正式实验曾以 20 个并行进程执行，每个进程 `--cpu-threads 1`：

```bash
/root/miniconda3/envs/NOG/bin/python scripts/run_retuned_nog_formal.py \
  --freeze results/advisor_cpu_batch_retuned_svm/frozen_parameters.json \
  --output results/advisor_cpu_batch_retuned_svm/formal_shards/seed-00 \
  --seeds 0 --cpu-threads 1
```

正式 shard 输出 `partials/*.csv`、`formal_trajectories.csv`、`formal_summary.csv` 和 `progress.json`。已有 partial 会自动跳过。

## 3. 合并 baseline、画图和阈值表

```bash
/root/miniconda3/envs/NOG/bin/python scripts/merge_retuned_svm_results.py \
  --retuned-root results/advisor_cpu_batch_retuned_svm \
  --retuned-shards results/advisor_cpu_batch_retuned_svm/formal_shards \
  --baseline-root outputs/distributed_zo/zo_theory_validation/real_data/supplement_formal_cpu_v2_seed_shards \
  --output results/advisor_cpu_batch_retuned_svm/merged

cp results/advisor_cpu_batch_retuned_svm/merged/formal_trajectories.csv \
   results/advisor_cpu_batch_retuned_svm/formal_trajectories.csv
cp results/advisor_cpu_batch_retuned_svm/merged/formal_summary.csv \
   results/advisor_cpu_batch_retuned_svm/formal_summary.csv

/root/miniconda3/envs/NOG/bin/python scripts/plot_retuned_svm_equal_budget.py \
  --source results/advisor_cpu_batch_retuned_svm/merged/formal_trajectories.csv \
  --output results/advisor_cpu_batch_retuned_svm
```

图只连接最近的原生 checkpoint，不插值；zoom 图固定 y 轴 0--0.05。最终 PDF 可在确认后复制到论文 figures 目录，旧 advisor 目录保持不变。

## 4. 审计

检查每个 dataset/method 的 `formal_seed` 是否为 0--19、work 是否不超过 983040、search/validation/formal seed 是否不相交，并保存 `merged/baseline_reuse_audit.csv`。生成文件哈希：

```bash
find results/advisor_cpu_batch_retuned_svm -type f -not -name SHA256SUMS.txt -print0 \
  | sort -z | xargs -0 sha256sum > results/advisor_cpu_batch_retuned_svm/SHA256SUMS.txt
```

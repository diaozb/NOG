# ZO real-data experiment: a9a and ijcnn1

This directory contains the compact, version-controlled evidence for Steps
ZO-9C1--ZO-9C3. The experiment compares NOG-ZO, ME-DOL-ZO, DGFM, and DGFM+ on
official LIBSVM training sets using a nonsmooth, nonconvex capped-l1 SVM.

The synthetic epsilon-scaling experiment remains the primary test of the
paper's communication-complexity direction. This real-data experiment is an
independent external-validity check; it was not tuned to force the same result.

## 1. Problem and data

For samples `(a_i, y_i)`, the optimized finite-sum objective is

$$
F(x)=\frac{1}{n}\sum_{i=1}^n\max\{1-y_i a_i^\top x,0\}
+\lambda\sum_{j=1}^d\min\{|x_j|,2\},
\qquad \lambda=\frac{10^{-5}}{n}.
$$

Every feature row is normalized to unit l2 norm. The parser accepts the
official one-based LIBSVM format and verifies the decompressed training-file
SHA256 before use.

| dataset | rows | dimension | raw SHA256 |
|---|---:|---:|---|
| a9a | 32,561 | 123 | `f5d5ffd8d865ff41328e7ee043e4b020816914ff6843ff15b98905ddbedce906` |
| ijcnn1 | 49,990 | 22 | `16506cad788cf7c9607454150ed1994788204bac2ff4c9cb3b320036b6950d3f` |

## 2. Common protocol

| item | value |
|---|---|
| logical workers | 8 |
| topology | complete mixing graph |
| execution | single-process dependency simulation on CPU |
| SZO estimator | two-point estimator; each pair costs 2 function calls |
| training-work cap | 983,040 SZO calls per dataset/method/seed |
| internal smoothing radius | 0.001 |
| evaluation radius | 0.002 |
| evaluation bank | 32 smoothing samples x 256 data samples |
| formal seeds | 0--19, paired across methods |
| calibration seeds | 303--311, disjoint from formal seeds |
| metric ordering for calibration | final objective, then stationarity proxy |

Training work and evaluation work are stored separately. Depth is dependency
communication depth, not wall-clock time. All four methods receive the same
training-work cap; DGFM+ ends exactly at the cap in this protocol.

The stationarity value is a method-independent Monte Carlo smoothed-gradient
proxy. It is not the exact distance to the Goldstein subdifferential.

## 3. Parameter calibration and freezing

Calibration was completed before any formal seed was used:

1. compatibility smoke on seed 302;
2. broad equal-work calibration on seeds 303--305;
3. boundary expansion on seeds 306--308 because optima hit grid boundaries;
4. final boundary confirmation on seeds 309--311 with
   `stop_after_this_grid: true`;
5. freeze the selected parameters and hashes of the config, dataset manifest,
   calibration summaries, algorithms, problem implementation, work-budget
   logic, and formal runner;
6. run formal seeds 0--19 without further parameter changes.

The compact selection evidence is in `calibration_audit/`; the immutable
manifest is `frozen_parameters.json`.

| dataset | method | selected parameters |
|---|---|---|
| a9a | NOG-ZO | M=1, eta=0.00003, smooth_B=1 |
| a9a | ME-DOL-ZO | epoch=6, multiplier=30,000 |
| a9a | DGFM | eta=0.5 |
| a9a | DGFM+ | eta=0.2 |
| ijcnn1 | NOG-ZO | M=1, eta=0.0001, smooth_B=1 |
| ijcnn1 | ME-DOL-ZO | epoch=6, multiplier=30,000 |
| ijcnn1 | DGFM | eta=2.0 |
| ijcnn1 | DGFM+ | eta=0.1 |

These are local choices within the preregistered search grids, not proofs of
global parameter optimality.

## 4. Formal results

Values are mean plus/minus sample standard deviation over 20 formal seeds.

| dataset | method | objective | proxy | train accuracy | depth | work |
|---|---|---:|---:|---:|---:|---:|
| a9a | DGFM | 0.3740 +/- 0.0039 | 0.0155 +/- 0.0056 | 0.8412 +/- 0.0015 | 7,680 | 983,040 |
| a9a | DGFM+ | 0.4052 +/- 0.0107 | 0.0248 +/- 0.0096 | 0.8292 +/- 0.0051 | 6,822 | 983,040 |
| a9a | ME-DOL-ZO | 0.4059 +/- 0.0086 | 0.0192 +/- 0.0073 | 0.8338 +/- 0.0032 | 3,840 | 983,040 |
| a9a | NOG-ZO | 0.3871 +/- 0.0012 | 0.0182 +/- 0.0030 | 0.8333 +/- 0.0009 | 7,680 | 983,040 |
| ijcnn1 | DGFM | 0.1859 +/- 0.0042 | 0.0150 +/- 0.0046 | 0.9256 +/- 0.0025 | 7,680 | 983,040 |
| ijcnn1 | DGFM+ | 0.1971 +/- 0.0029 | 0.0207 +/- 0.0059 | 0.9039 +/- 0.0010 | 6,822 | 983,040 |
| ijcnn1 | ME-DOL-ZO | 0.1821 +/- 0.0027 | 0.0148 +/- 0.0032 | 0.9232 +/- 0.0027 | 3,840 | 983,040 |
| ijcnn1 | NOG-ZO | 0.1920 +/- 0.0010 | 0.0123 +/- 0.0021 | 0.9045 +/- 0.0006 | 7,680 | 983,040 |

![a9a formal curves](formal/a9a_formal_curves.png)

![ijcnn1 formal curves](formal/ijcnn1_formal_curves.png)

Vector versions are available as `formal/a9a_formal_curves.pdf` and
`formal/ijcnn1_formal_curves.pdf`.

## 5. Interpretation and theory boundary

- On a9a, NOG-ZO improves both final objective and stationarity proxy relative
  to ME-DOL-ZO under equal training work, but consumes twice its communication
  depth. DGFM has the best objective and proxy.
- On ijcnn1, NOG-ZO has the lowest proxy, while ME-DOL-ZO has the best objective
  and substantially better accuracy. NOG-ZO again consumes twice the ME-DOL
  communication depth.
- NOG-ZO therefore achieves competitive optimization quality on both datasets,
  but this experiment does **not** reproduce the synthetic communication-depth
  advantage.
- This does not contradict a worst-case Big-O theorem: the theorem is an upper
  bound with epsilon-dependent batching and problem constants, whereas this
  experiment freezes finite-budget batches and compares local pilot-selected
  constants on two particular datasets.
- The real-data result must not be used to claim universal empirical dominance.
  It documents where the synthetic theory-aligned trend does and does not
  transfer.

## 6. Reproduction

```bash
# Download/verify official data and run compatibility smoke.
conda run -n NOG python -m src.distributed.zo_real_data_smoke

# Reproduce the final calibration grid and freeze its selection.
conda run -n NOG python -m src.distributed.zo_real_data_calibration \
  --config configs/distributed_zo_real_data_calibration_final_boundary.yaml \
  --output outputs/distributed_zo/zo_theory_validation/real_data/calibration_final_boundary_work98304
conda run -n NOG python -m src.distributed.zo_real_data_freeze

# Run four disjoint five-seed shards (40 tasks each).
conda run -n NOG python -m src.distributed.zo_real_data_formal --seeds 0,1,2,3,4 \
  --output outputs/distributed_zo/zo_theory_validation/real_data/formal_fixed_work983040_shards/shard_0
conda run -n NOG python -m src.distributed.zo_real_data_formal --seeds 5,6,7,8,9 \
  --output outputs/distributed_zo/zo_theory_validation/real_data/formal_fixed_work983040_shards/shard_1
conda run -n NOG python -m src.distributed.zo_real_data_formal --seeds 10,11,12,13,14 \
  --output outputs/distributed_zo/zo_theory_validation/real_data/formal_fixed_work983040_shards/shard_2
conda run -n NOG python -m src.distributed.zo_real_data_formal --seeds 15,16,17,18,19 \
  --output outputs/distributed_zo/zo_theory_validation/real_data/formal_fixed_work983040_shards/shard_3
conda run -n NOG python -m src.distributed.zo_real_data_analysis
```

The formal run used four disjoint seed shards only to reduce wall-clock time;
each shard read the same frozen manifest and wrote a separate output directory.
No task settings changed. The analysis verifies exactly 160 unique
`(dataset, method, seed)` identities before producing results.

## 7. Files

- `frozen_parameters.json`: immutable protocol and source/input hashes;
- `calibration_audit/`: final three-seed calibration summary and data manifest;
- `formal/results.csv`: complete checkpoint trajectories;
- `formal/final_per_seed.csv`: 160 final rows;
- `formal/summary.csv`: the eight mean/std rows shown above;
- `formal/audit.json`: completeness and claim-boundary audit;
- `formal/*.png` and `formal/*.pdf`: formal figures.

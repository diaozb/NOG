# v4 extended-budget continuation: stage2

- Budgets: NOG=15360 rounds; ME-DOL=61440 rounds.
- All problem, algorithm, batch, evaluation, worker, and seed settings are frozen from v4.
- Original v4 deterministic prefix validation: **passed** (40 tasks).
- A finite ratio is reported only when both methods hit on all 20 paired seeds.

| epsilon | NOG hit | ME hit | ME/NOG depth | NOG/ME work |
|---:|---:|---:|---:|---:|
| 0.0095 | 20/20 | 20/20 | 3.135x | 0.772x |
| 0.0090 | 20/20 | 20/20 | 8.792x | 0.478x |
| 0.0085 | 20/20 | 16/20 | -- | -- |
| 0.0080 | 20/20 | 1/20 | -- | -- |
| 0.0075 | 20/20 | 0/20 | -- | -- |
| 0.0070 | 16/20 | 0/20 | -- | -- |
| 0.0065 | 3/20 | 0/20 | -- | -- |
| 0.0060 | 0/20 | 0/20 | -- | -- |
| 0.0055 | 0/20 | 0/20 | -- | -- |
| 0.0050 | 0/20 | 0/20 | -- | -- |
| 0.0045 | 0/20 | 0/20 | -- | -- |
| 0.0040 | 0/20 | 0/20 | -- | -- |
| 0.0035 | 0/20 | 0/20 | -- | -- |
| 0.0030 | 0/20 | 0/20 | -- | -- |
| 0.0025 | 0/20 | 0/20 | -- | -- |
| 0.0020 | 0/20 | 0/20 | -- | -- |

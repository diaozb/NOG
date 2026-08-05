# v4 extended-budget continuation

This compact package contains the auditable low-epsilon continuation of
`theory_validation_v4`.  The original v4 problem, algorithm parameters,
batches, evaluation bank, workers, and formal seeds are frozen; only the
maximum number of training rounds is increased.

- Stage 1: NOG 3,840 rounds; ME-DOL 15,360 rounds.
- Stage 2: NOG 15,360 rounds; ME-DOL 61,440 rounds.
- Formal seeds: 0--19.
- Full finite paired ratios are reported only when both methods hit on all
  20 seeds.
- Both stages passed the 40-task artifact/work audit and exact numerical v4
  prefix validation.

The final Stage 2 result has two fully observed targets: epsilon 0.0095 and
0.0090.  Smaller targets remain censored under the extended budgets.

See [REPRODUCE.md](REPRODUCE.md) for commands and the stage-specific
`analysis/extended_budget_report.md` files for result tables.

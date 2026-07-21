# Step 7 + Step 8 Asset Recommendations

## Main paper

1. `../figures/depth_vs_epsilon.pdf` — primary evidence for the qualitative communication-depth advantage.
2. `../figures/work_vs_epsilon.pdf` — required companion showing the finite-SFO-work tradeoff.
3. The Step 7 formal table in `../step7_final/paper_table.tex`.

## Supplement or advisor discussion

1. `../runtime/figures/runtime_vs_workers.pdf` — complete median/min/max timing evidence.
2. `../runtime/figures/full_budget_method_comparison.pdf` — use both panels together; never crop away the work ratio.
3. `../runtime/figures/communication_fraction_vs_workers.pdf` — systems-overhead diagnostic.

## Diagnostic only

- `../runtime/figures/nog_strong_scaling_speedup.pdf` — transparent negative result, not evidence of positive scaling.

## Required wording boundaries

- Use “consistent with the predicted communication advantage.”
- State that runtime uses complete frozen budgets and is not first-hit time-to-epsilon.
- State that frozen configurations are not empirical-work matched.
- State that the single-host CPU/Gloo benchmark does not show positive strong scaling.
- Preserve the right-censoring warning for ME-DOL at epsilon 0.008.

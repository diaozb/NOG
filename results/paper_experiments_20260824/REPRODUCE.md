# Reproduce the paper figures

All commands are CPU-only. Use the repository's Python interpreter directly:

```bash
cd /data/diaozb/NOG
/root/miniconda3/envs/NOG/bin/python \
  nog_iclr2027_complete_source/figures/generate_theory_validation_figures.py
```

The command reads the audited summaries in
`results/theory_validation_v4/analysis/formal_summary.csv` and
`zo_experiments/formal/formal_summary.csv`, then writes the two paper figures to
`nog_iclr2027_complete_source/figures/`. The same generator is copied under
`scripts/` in this directory for provenance.

The plotted support is deliberately fixed before plotting:

- FO: `0.0105 <= epsilon <= 0.2`, 25 thresholds, both methods 20/20 hits;
- ZO: `0.03 <= epsilon <= 0.2`, 13 thresholds, all four methods 20/20 hits.

The formal seeds are 0--19. Pilot seeds 100--104 were not used as formal points.
The audit JSON files record task completion, seed/config identity, finite values,
and exact depth/work accounting.

To compile the current paper source after generating the figures:

```bash
cd /data/diaozb/NOG/nog_iclr2027_complete_source
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/nog_iclr2027_build nog_iclr2027.tex
```

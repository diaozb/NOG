# Paper figure captions

## `depth_vs_epsilon.pdf`

**Communication depth to a confirmed stationarity threshold.** Mean first-hit communication depth and one sample standard deviation over five formal seeds for NOG-FO and ME-DOL-FO at $m=8$ and $\delta=0.1$. A confirmed hit requires two consecutive evaluation checkpoints with stationarity proxy at most $\epsilon$. Filled markers indicate a 5/5 hit rate; hollow markers are conditional means over successful seeds in right-censored settings, with hit counts shown next to the marker. Configurations were selected independently for each method and tolerance using disjoint pilot seeds.

## `work_vs_epsilon.pdf`

**Training SFO work to a confirmed stationarity threshold.** Mean first-hit total training SFO calls and one sample standard deviation under the same protocol as the communication-depth figure. Evaluation calls from the shared high-precision sample bank are audited separately and excluded. Hollow markers denote right-censored conditional means and should not be interpreted as unconditional method comparisons.

## `stat_proxy_vs_depth.pdf`

**Stationarity trajectories versus communication depth.** Lines and shaded bands show the mean and one sample standard deviation across five formal seeds. Each panel compares the two pilot-frozen configurations selected for the displayed tolerance; the dashed line is the target $\epsilon$. Panel titles report confirmed-hit counts. Because configurations vary across tolerances, the panels are separate empirical operating points rather than a single-hyperparameter scaling curve.

## `stat_proxy_vs_work.pdf`

**Stationarity trajectories versus total training SFO work.** The trajectories and aggregation protocol match the communication-depth panels, with the horizontal axis replaced by cumulative training SFO calls. Evaluation work is excluded. This view exposes the finite-work tradeoff accompanying NOG-FO's lower communication depth.

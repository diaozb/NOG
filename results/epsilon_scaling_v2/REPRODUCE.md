# Reproduce wide-epsilon results

Environment: the repository requirements in the `NOG` conda environment.

```bash
python -m unittest discover -s tests -p "test*.py"
python -m src.distributed.epsilon_scaling_formal_result_audit
python -m src.distributed.epsilon_scaling_robustness_audit
python -m src.distributed.epsilon_scaling_formal_statistics
python -m src.distributed.epsilon_scaling_robustness_statistics
python -m src.distributed.epsilon_scaling_report
```

The audit/statistics commands require the local raw trajectories under
`outputs/distributed_cpu_fo_v2/epsilon_scaling_v2/`. Raw trajectories are not
committed because they total about 215 MiB. This compact package contains their
completion manifests, SHA256 accounting audits, summaries, figures, and report.

Long-task runner guard (operational only; excluded from numerical identity):

```bash
NOG_CPU_LAUNCH_TIMEOUT_SECONDS=7200 python -m src.distributed.epsilon_scaling_robustness_runner --phase run --worker 4
```

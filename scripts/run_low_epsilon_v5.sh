#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/NOG

CONDA_PY=/root/miniconda3/envs/NOG/bin/python

"${CONDA_PY}" -m src.distributed.low_epsilon_runner pilot-algorithms
"${CONDA_PY}" -m src.distributed.low_epsilon_freeze freeze-algorithms
"${CONDA_PY}" -m src.distributed.low_epsilon_runner pilot-batches
"${CONDA_PY}" -m src.distributed.low_epsilon_freeze freeze-final
"${CONDA_PY}" -m src.distributed.low_epsilon_runner formal
"${CONDA_PY}" -m src.distributed.low_epsilon_audit
"${CONDA_PY}" -m src.distributed.low_epsilon_analysis
"${CONDA_PY}" -m src.distributed.low_epsilon_audit --formal-root outputs/distributed_cpu_fo_v5/epsilon_low_extension_v5_symmetric/formal_extra --extra
"${CONDA_PY}" -m src.distributed.low_epsilon_report
"${CONDA_PY}" -m src.distributed.low_epsilon_package

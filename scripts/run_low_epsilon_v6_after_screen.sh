#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/NOG

CONDA_PY=/root/miniconda3/envs/NOG/bin/python

for retry_attempt in 1 2 3; do
  if "${CONDA_PY}" -m src.distributed.low_epsilon_v6_runner screen; then
    break
  fi
  if [[ "${retry_attempt}" -eq 3 ]]; then
    echo "v6 screen remained incomplete after 3 recovery attempts" >&2
    exit 1
  fi
  sleep 15
done

"${CONDA_PY}" -m src.distributed.low_epsilon_v6_freeze shortlist
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_runner confirmation
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_freeze freeze
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_runner formal
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_analysis

#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/NOG

CONDA_PY=/root/miniconda3/envs/NOG/bin/python

run_with_retries() {
  local stage_name=$1
  shift
  local retry_attempt
  for retry_attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    if [[ "${retry_attempt}" -eq 3 ]]; then
      echo "v6 ${stage_name} remained incomplete after 3 recovery attempts" >&2
      return 1
    fi
    sleep 15
  done
}

run_with_retries screen \
  "${CONDA_PY}" -m src.distributed.low_epsilon_v6_runner screen

"${CONDA_PY}" -m src.distributed.low_epsilon_v6_freeze shortlist
run_with_retries confirmation \
  "${CONDA_PY}" -m src.distributed.low_epsilon_v6_runner confirmation
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_freeze freeze
run_with_retries formal \
  "${CONDA_PY}" -m src.distributed.low_epsilon_v6_runner formal
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_analysis

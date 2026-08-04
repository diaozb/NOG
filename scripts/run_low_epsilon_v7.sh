#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/NOG

CONDA_PY=/root/miniconda3/envs/NOG/bin/python
EXTENSION_CONFIG=configs/distributed_cpu_fo_low_epsilon_v7_extension.yaml
EXTENSION_ROOT=outputs/distributed_cpu_fo_v7/epsilon_low_extension_v7_theory_scaling/pilot_extension

run_with_retries() {
  local stage_name=$1
  shift
  local retry_attempt
  for retry_attempt in 1 2 3 4 5; do
    if "$@"; then
      return 0
    fi
    if [[ "${retry_attempt}" -eq 5 ]]; then
      echo "v7 ${stage_name} remained incomplete after 5 recovery attempts" >&2
      return 1
    fi
    sleep 15
  done
}

run_with_retries pilot \
  "${CONDA_PY}" -m src.distributed.low_epsilon_v7_runner pilot

run_with_retries pilot-extension \
  "${CONDA_PY}" -m src.distributed.low_epsilon_v7_runner pilot \
  --config "${EXTENSION_CONFIG}" --output-root "${EXTENSION_ROOT}"

"${CONDA_PY}" -m src.distributed.low_epsilon_v7_freeze_extension

run_with_retries formal \
  "${CONDA_PY}" -m src.distributed.low_epsilon_v7_runner formal

"${CONDA_PY}" -m src.distributed.low_epsilon_v7_analysis

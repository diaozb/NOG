#!/usr/bin/env bash
set -euo pipefail

cd /data/diaozb/NOG

CONDA_PY=/root/miniconda3/envs/NOG/bin/python

"${CONDA_PY}" -m src.distributed.low_epsilon_v6_freeze shortlist
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_runner confirmation
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_freeze freeze
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_runner formal
"${CONDA_PY}" -m src.distributed.low_epsilon_v6_analysis

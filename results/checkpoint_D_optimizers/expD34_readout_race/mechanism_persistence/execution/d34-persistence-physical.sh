#!/usr/bin/env bash
set -euo pipefail
base=/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923
out=$base/evidence/persistence_1bf7138
py=/workspace/junmiaoh/experiments/precision-mlps/.venv-feedback/bin/python
cd "$base/code"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for cohort in development confirmation; do
  for panel in N128 late; do
    pred=$out/physical_${cohort}_${panel}
    "$py" -m experiments.expD34_readout_race.mechanism_persistence run --predictions "$pred" --output "${pred}_run" --backend gpu --steps 20000
  done
done

#!/usr/bin/env bash
set -euo pipefail
base=/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923
out=$base/evidence/persistence_1bf7138
py=/workspace/junmiaoh/experiments/precision-mlps/.venv-feedback/bin/python
cd "$base/code"
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
kind=$1
panels='N128 N512 N1024 late'
if [ "$kind" = physical ]; then panels='N128 late'; fi
for cohort in development confirmation; do
  for panel in $panels; do
    pred=$out/${kind}_${cohort}_${panel}
    "$py" -m experiments.expD34_readout_race.mechanism_persistence analyze --predictions "$pred" --run "${pred}_run" --output "${pred}_analysis"
    if [ "$kind" = physical ]; then
      "$py" -m experiments.expD34_readout_race.mechanism_persistence_pulses analyze --predictions "$pred" --run "${pred}_run" --output "${pred}_paired"
    fi
  done
done
"$py" -m experiments.expD34_readout_race.mechanism_persistence_summary --root "$out" --output "$out/${kind}_summary"

#!/usr/bin/env bash
set -euo pipefail
base=/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923
out=$base/evidence/persistence_1bf7138
py=/workspace/junmiaoh/experiments/precision-mlps/.venv-feedback/bin/python
cd "$base/code"
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$py" -m pytest -q tests/test_mechanism_persistence_coupled_response.py
for cohort in development confirmation; do
  for panel in N128 late; do
    pred=$out/physical_${cohort}_${panel}
    "$py" -m experiments.expD34_readout_race.mechanism_persistence_coupled_response --predictions "$pred" --actual-run "${pred}_run" --output "$out/coupled_${cohort}_${panel}" --max-seconds 250
  done
done

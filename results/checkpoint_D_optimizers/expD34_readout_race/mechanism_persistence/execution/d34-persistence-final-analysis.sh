#!/usr/bin/env bash
set -euo pipefail
base=/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923
out=$base/evidence/persistence_1bf7138
py=/workspace/junmiaoh/experiments/precision-mlps/.venv-feedback/bin/python
cd "$base/code"
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$py" -m pytest -q tests/test_mechanism_persistence.py tests/test_mechanism_persistence_runner.py tests/test_mechanism_persistence_pulses.py tests/test_mechanism_persistence_pulse_summary.py
"$py" "$base/code/d34-persistence-overview.py"
for cohort in development confirmation; do
  for panel in N128 late; do
    "$py" -m experiments.expD34_readout_race.mechanism_persistence_pulse_summary --pulses "$out/kicks_${cohort}_${panel}" --analysis "$out/physical_${cohort}_${panel}_paired" --output "$out/matching_${cohort}_${panel}"
  done
done
"$py" -m experiments.expD34_readout_race.mechanism_persistence_path_audit --root "$out" --output "$out/path_audit" --max-seconds 600

#!/usr/bin/env bash
set -euo pipefail
base=/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923
ev=$base/evidence
out=$ev/persistence_1bf7138
py=/workspace/junmiaoh/experiments/precision-mlps/.venv-feedback/bin/python
cd "$base/code"
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$py" -m pytest -q tests/test_mechanism_persistence.py tests/test_mechanism_persistence_pulses.py tests/test_mechanism_persistence_runner.py
for cohort in development confirmation; do
  for nref in 128 512 1024; do
    if [ "$cohort" = development ]; then
      inp=$ev/width_inputs/N${nref}_fork20000.npz
      seed=30
    else
      inp=$ev/polynomial_confirmation/fork_inputs/N${nref}.npz
      seed=32
    fi
    "$py" -m experiments.expD34_readout_race.mechanism_persistence prepare --inputs "$inp" --output "$out/feedback_${cohort}_N${nref}" --seed "$seed" --nref "$nref" --cohort "$cohort"
  done
  seed=0
  if [ "$cohort" = confirmation ]; then seed=20; fi
  "$py" -m experiments.expD34_readout_race.mechanism_persistence prepare --inputs "$ev/inputs/$cohort.npz" --output "$out/feedback_${cohort}_late" --seed "$seed" --targets sine,moment9 --nref 128 --cohort "$cohort"
  "$py" -m experiments.expD34_readout_race.mechanism_persistence audit --inputs "$ev/inputs/$cohort.npz" --output "$out/audit_$cohort"
done

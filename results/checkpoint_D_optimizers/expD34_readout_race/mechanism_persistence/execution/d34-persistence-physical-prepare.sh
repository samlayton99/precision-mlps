#!/usr/bin/env bash
set -euo pipefail
base=/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923
ev=$base/evidence
out=$ev/persistence_1bf7138
py=/workspace/junmiaoh/experiments/precision-mlps/.venv-feedback/bin/python
cd "$base/code"
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$py" -m pytest -q tests/test_mechanism_persistence_pulses.py
for cohort in development confirmation; do
  if [ "$cohort" = development ]; then
    inp=$ev/width_inputs/N128_fork20000.npz
    seed=30
    late_seed=0
  else
    inp=$ev/polynomial_confirmation/fork_inputs/N128.npz
    seed=32
    late_seed=20
  fi
  "$py" -m experiments.expD34_readout_race.mechanism_persistence_pulses --inputs "$inp" --output "$out/kicks_${cohort}_N128" --seed "$seed"
  "$py" -m experiments.expD34_readout_race.mechanism_persistence prepare --inputs "$out/kicks_${cohort}_N128/inputs.npz" --output "$out/physical_${cohort}_N128" --physical --cohort "$cohort" --nref 128
  "$py" -m experiments.expD34_readout_race.mechanism_persistence_pulses --inputs "$ev/inputs/$cohort.npz" --output "$out/kicks_${cohort}_late" --seed "$late_seed" --targets sine,moment9
  "$py" -m experiments.expD34_readout_race.mechanism_persistence prepare --inputs "$out/kicks_${cohort}_late/inputs.npz" --output "$out/physical_${cohort}_late" --physical --cohort "$cohort" --nref 128
done

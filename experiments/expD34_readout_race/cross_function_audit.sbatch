#!/usr/bin/env bash
#SBATCH --job-name=d34-cross-audit
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
set -euo pipefail

# Arguments: checkout python evidence output pilot|full
checkout=$1
python_bin=$2
evidence=$3
output=$4
mode=${5:-full}
cd "${checkout}"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
if [[ "${mode}" == pilot ]]; then
  srun "${python_bin}" -m pytest -q tests/test_d34_cross_function_budget.py \
    tests/test_d34_cross_function_forces.py tests/test_d34_cross_function_archives.py
  extra=(--limit 1)
elif [[ "${mode}" == full ]]; then
  extra=()
else
  exit 2
fi
for cohort in existing fresh heldout; do
  srun "${python_bin}" -m experiments.expD34_readout_race.cross_function_audit \
    --evidence "${evidence}" --output "${output}/${cohort}" \
    --cohort "${cohort}" "${extra[@]}"
done

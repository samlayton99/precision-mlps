#!/bin/bash
#SBATCH --job-name=d34-force-probe
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
campaign_root=$1
source_root=$2
stage=$3
cd "${campaign_root}/code"
export RACE_SOURCE_COMMIT="$(cat .source_commit)"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python_bin=/workspace/junmiaoh/experiments/precision-mlps/.venv-race/bin/python
index=${SLURM_ARRAY_TASK_ID}
case "${stage}" in
  discovery)
    seed=$((index / 2)); start=$((100000 + (index % 2) * 500000))
    srun "${python_bin}" -u -m experiments.expD34_readout_race.plateau_probes \
      --source "${source_root}/raw" --output "${campaign_root}/probes/seed${seed}_${start}" \
      --seed "${seed}" --start "${start}" --max-seconds 1650
    ;;
  confirm)
    seed=$((20 + index))
    srun "${python_bin}" -u -m experiments.expD34_readout_race.plateau_run \
      --optimizer confirm --confirm-seed "${seed}" --output "${campaign_root}/confirm" \
      --end-step 1100000 --max-seconds 620
    ;;
  confirm-probes)
    seed=$((20 + index))
    srun "${python_bin}" -u -m experiments.expD34_readout_race.plateau_probes \
      --source "${campaign_root}/confirm" --output "${campaign_root}/confirm-probes/seed${seed}_600000" \
      --seed "${seed}" --start 600000 --max-seconds 620
    ;;
  checks)
    start=$((100000 + (index % 2) * 500000))
    extra=(--half); label=half
    if (( index >= 2 )); then extra=(--samples 4096); label=grid; fi
    srun "${python_bin}" -u -m experiments.expD34_readout_race.plateau_probes \
      --source "${source_root}/raw" --output "${campaign_root}/checks/${label}_${start}" \
      --seed 0 --start "${start}" --max-seconds 1100 "${extra[@]}"
    ;;
  *) exit 2 ;;
esac

#!/bin/bash
#SBATCH --job-name=d34-adam-analysis
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
campaign_root=$1
cd "${campaign_root}/code"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR="${campaign_root}/mpl-cache"
python_bin=/workspace/junmiaoh/experiments/precision-mlps/.venv-race/bin/python
evidence="${campaign_root}/results/adam_force_extension"
mkdir -p "${evidence}/analysis" "${evidence}/verification"
for source in "${evidence}"/raw/*; do
    bundle=${source##*/}
    srun --ntasks=1 "$python_bin" -u -m experiments.expD34_readout_race.adam_analyze \
        --source "$source" --output "${evidence}/analysis/${bundle}" > "${evidence}/verification/analysis_${bundle}.log" 2>&1
done
srun --ntasks=1 "$python_bin" -m experiments.expD34_readout_race.adam_analyze --construction --output "${evidence}/analysis"
srun --ntasks=1 "$python_bin" -m experiments.expD34_readout_race.adam_verify --root "$evidence"
srun --ntasks=1 "$python_bin" -m experiments.expD34_readout_race.adam_verify --root "$evidence" --curate
srun --ntasks=1 "$python_bin" -m experiments.expD34_readout_race.adam_summarize --root "$evidence"

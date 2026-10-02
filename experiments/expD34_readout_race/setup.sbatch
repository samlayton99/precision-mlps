#!/bin/bash
#SBATCH --job-name=race-setup
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=lab-gpu2
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/workspace/junmiaoh/experiments/precision-mlps/logs/race-setup-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/race-mpl
if [[ ${SLURM_STEP_ID:-} == '' ]]; then
    exec srun --ntasks=1 bash "$0"
fi
experiment_root=/workspace/junmiaoh/experiments/precision-mlps
cd "${experiment_root}/race-code"
export RACE_SOURCE_COMMIT="$(cat .source_commit)"
if [[ ! -f ${experiment_root}/.venv-race/bin/python ]]; then
    "${experiment_root}/.venv/bin/python" -m venv "${experiment_root}/.venv-race"
fi
"${experiment_root}/.venv-race/bin/python" - <<'PY'
from pathlib import Path
import site
Path(site.getsitepackages()[0], 'existing_jax_environment.pth').write_text('/workspace/junmiaoh/experiments/precision-mlps/.venv/lib/python3.11/site-packages\n')
PY
"${experiment_root}/.venv-race/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.4.1+cpu'
"${experiment_root}/.venv-race/bin/python" -m pip install threadpoolctl
"${experiment_root}/.venv-race/bin/python" -m pytest -q tests/test_expD34_readout_race.py \
    tests/test_expD24_gd_residual_spectrum.py tests/test_expD25_scale_barrier.py \
    tests/test_expD28_loss_gradient_decomposition.py

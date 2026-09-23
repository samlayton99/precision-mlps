#!/bin/sh
# This script was created by sbatch --wrap.

cd /workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/code && srun env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/junmiaoh/experiments/precision-mlps/.venv/bin/python d34_arb_probe.py --inputs /workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence/arb_width512_mixed_sine_seed30/input.npz --enclosure /workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence/arb_width512_mixed_sine_seed30/chord_input.npz --chunk 100 --steps 20000 --output /workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence/arb_width512_mixed_sine_seed30/certificate_chunk100.json

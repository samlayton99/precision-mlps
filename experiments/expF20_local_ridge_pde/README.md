# Local ridge PDE feasibility

Isolated preliminary prototype; existing production solvers are unchanged. See the [measured report](../../results/checkpoint_F_applications/expF20_local_ridge_pde/expF20_results.md).

Use `/Users/sam/venv/quill-radon/bin/python` below. NumPy, SciPy, pytest, matplotlib and a C compiler are required; reduced controls also import the existing Torch-based feature module. First use compiles the kernel into the gitignored result directory.

```sh
python experiments/expF20_local_ridge_pde/run.py benchmark --m 128 --n 8193 --k 1024
python experiments/expF20_local_ridge_pde/run.py benchmark --m 128 --n 32769 --k 1024
python experiments/expF20_local_ridge_pde/run.py free --m 128 --n 8193 --k 1024 --iterations 100 --coordinates raw
python experiments/expF20_local_ridge_pde/run.py free --m 128 --n 8193 --k 1024 --iterations 200 --coordinates smooth
python experiments/expF20_local_ridge_pde/run.py free --m 128 --n 8193 --k 8192 --iterations 200 --coordinates smooth
python experiments/expF20_local_ridge_pde/run.py free --m 128 --n 8193 --k 1024 --iterations 200 --coordinates smooth --nonlinear
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/expF20_local_ridge_pde/run_reduced.py --case all
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/expF20_local_ridge_pde/run_reduced.py --case refinement
python experiments/expF20_local_ridge_pde/verify_results.py
python experiments/expF20_local_ridge_pde/plot_results.py
python -m pytest -q tests/test_local_ridge_operator.py tests/test_route2_directional_operator.py
```

The recorded operator grid is `(M,N)=(128,257),(128,1025),(128,8193),(128,32769),(16,65537),(512,2049)`, all at 1,024 points. Extra free checks use `(16,257)` at 4,096 points, `--seed 814`, `--smooth-power 2`, and `--smooth-power 2 --iterations 400 --restart 400`. JSON files record exact parameters. Main free solves use 50-step LSMR chunks; reduced runs use dense QR in the small prescribed coordinate space.

`LocalBank.bind(points)` retains M-by-K projections. `forward` sums profiles; `profiles` returns direction outputs. `adjoint` and `adjoint_profiles` apply matching transposes. Derivative direction factors can be passed to `forward`/`adjoint`. Biases are handled separately by the solver. No production integration or general error certificate is included.

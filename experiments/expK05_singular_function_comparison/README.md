# Two singular-function QUILL constructions

Apply the existing multiscale C17 construction to `1/(sqrt(x)+x)` and
`exp(-x)/sqrt(x)` on `[0.0001,1]`. The 1×2 figure uses the previous inverse-root
figure's typography, logarithmic width axis, viridis colors, white sample markers,
and bottom legend. No smoothing is applied.

For each function f, define H(u)=I_(clip(u-1,0,1))(17,17), s_j=0.00005·2^j,
j=0,…,14, and s_15=1.6384. The extension derivative is

    F'(x) = f'(x) [H(x/s_0) - H(x/s_15)],  F(0.0001)=f(0.0001).

It vanishes outside `[0.00005,3.2768]`; F is bounded and C17, and equals f on
the requested domain. The derivative window telescopes into 15 components
`f'(x)[H(x/s_j)-H(x/(2s_j))]`. On u=x/s_j∈[0,4], normalize each primitive
by sqrt(s_j), then apply cardinal QI. Unlike the original inverse-root case,
these two targets need 15 distinct derivative sample sets, not one reusable
homogeneous prototype. Cardinal coefficients remain target-independent.

Use lambda=.25, Kc=160, halo=32 per side, 30-decimal construction arithmetic,
and Float64 derivative samples, coefficients and inference. All blocks expand
into one ordinary centered Linear/Tanh/Linear network. Only affine input
normalization is used. There is no training or least-squares readout refit.
The stable left-endpoint export is inherited from expK02 and asserted exactly.

The same 22 N values as the previous 1k–5k sweep give widths 990–5055.
The preselected N=256 model has 4815 neurons and 14446 parameters. Error is
physical-dx relative L2, with both numerator and denominator computed for the
actual function. The inverse-root-only shortcut `f² dx=d(log x)` is not used.

The figure's horizontal NPU references come from the saved expK04 screen:
32 hidden neurons, 97 trainable parameters, standard Glorot initialization,
20k Adam steps followed by up to 500 L-BFGS iterations, seed 0, selected on
validation. They train on the original positive interval, with importance
weights for physical dx, without QUILL's artificial halo. These are fixed-width
references, not a matched-capacity comparison. The algebraic function is
exactly representable by a small compositional NPU; observed training errors
do not prove an approximation limit. The shallow positive-input NPU with zero
imaginary initialization remains on its RealNPU branch.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK05_singular_function_comparison/run.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK05_singular_function_comparison/verify.py
~/venv/quill-radon/bin/python experiments/expK05_singular_function_comparison/plot.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python -m pytest -q tests/test_singular_function_comparison.py
```

Outputs live in `results/checkpoint_K_practical_recipes/expK05_singular_function_comparison/`:
all 44 networks and measurements, selected Torch networks, dense predictions,
halo checks, source hashes, verification results, and PNG/PDF figure. Tests
check derivatives against autograd, the exact partition, physical-L2 weighting,
and achieved construction precision for both functions. Verification separately
checks dense endpoints, quadrature refinement, Torch export, wider halo,
60-decimal evaluation of saved weights, and reloaded NPU checkpoints.

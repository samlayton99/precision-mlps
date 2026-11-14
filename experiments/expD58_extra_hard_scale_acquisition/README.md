# D58 — Extra-hard target scale acquisition

Exploratory extension of the original D56 **tanh, full-batch GD** experiment. "Extra hard" is a provisional category chosen for unusual structure, not a claim that every target is harder for GD. No target is selected or excluded based on the resulting training curves.

## Fixed target suite

All functions are defined on [-1,1] and divided by the RMS on the same 1,024-point training grid used in D56.

| Target | Function | Feature |
|---|---|---|
| Erf transition | erf(12(x-0.15)) | Steep analytic special function; plain erf(x) would be comparatively gentle |
| Gamma | Gamma(0.05+2.475(x+1)) | Gamma argument stays in [0.05,5], near a pole but never crossing it |
| Airy | Ai(-30x) | Rapid decay on one side, oscillation on the other |
| Square-root cusp | sign(x-0.137) sqrt(abs(x-0.137)) | Continuous with unbounded derivative at an off-grid cusp |
| Smooth bump | exp(-1/(1-(x/0.35)^2)) for abs(x)<0.35; zero otherwise | Infinitely differentiable but nonanalytic, with exactly flat tails |
| Oscillatory cusp | x sin(1/x), defined as zero at x=0 | Continuous with arbitrarily high local frequency near zero |
| Shifted step | 1 if x>=0.137, otherwise 0 | Off-grid discontinuity |
| Square wave | sign(sin(7*pi*x+0.37)) | Multiple shifted discontinuities |

The shift in Gamma is intentional: Gamma(x) directly on [-1,1] has poles and would not define the same bounded-target experiment. The analytic construction theorem does not cover the nonsmooth or nonanalytic targets. A continuous tanh network cannot uniformly approximate a jump to arbitrarily small error. We study scale acquisition and sampled RMS fitting, not a universal high-precision approximation claim for these targets.

## Matched protocol

Exactly 72 runs: 8 targets × 3 learning rates (0.01, 0.001, 0.0001) × 3 first-layer initialization multipliers (1, 2, 4). Each run has 196 neurons, seed 30, float64, 200,000 GD updates, half-MSE loss, and no clipping, momentum, schedule, or coefficient solve. Initialization, update code, grids, RMS normalization, and all structural diagnostics are reused from D56. The baseline tanh results remain separate and unchanged.

The same QUILL slope reference (18.375) is shown as a geometric comparison only, not a required accuracy threshold for nonsmooth functions. Standard initialization has random centers; the reference's 196-neuron budget includes its 48 halo centers.

## Sampling and records

Save all parameters at dense early checkpoints and every 500 updates. Four workers advance all cases in 5,000-step rounds, writing atomic checkpoints and regenerating the real-data plot after each round. Each function has its own median absolute slope and central 98% interval across neurons, with the approved thin lines, transparent bands, dotted boundaries, and three-panel format.

At each completed run, additionally evaluate on 65,536 independent midpoint samples and on fixed one-sided probes near the jumps, cusp, bump boundaries, and domain endpoints. Save dense-grid relative L2, sampled maximum error, and separate special-probe maximum error. These are numerical sampling checks, not uniform-error certificates. The oscillatory cusp has frequencies beyond any fixed grid near zero; its amplitude vanishes there, and denser-grid error is still important.

## Execution

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_expD58_extra_hard_scale_acquisition.py
.venv/bin/python experiments/expD58_extra_hard_scale_acquisition/plot.py --targets-only
.venv/bin/python experiments/expD58_extra_hard_scale_acquisition/run.py pilot
.venv/bin/python experiments/expD58_extra_hard_scale_acquisition/run.py launch
.venv/bin/python experiments/expD58_extra_hard_scale_acquisition/run.py status
```

Artifacts live under `results/checkpoint_D_optimizers/expD58_extra_hard_scale_acquisition/`. The target gallery is in `targets/`, the short pilot in `pilot/`, and the full sweep in `sweep/`. `sweep/figures/scale_acquisition.png` is refreshed with explicitly labeled progress, then complete data. Training-source hashes prevent incompatible resumes. Scientific conclusions await the completed sweep.

# QUILL on two singular functions — data-obvious

## Findings

- The same C17 multiscale recipe reaches about 5.5e-14 physical relative L2 on both targets, using 4815 neurons.
- Saved, validation-selected 32-neuron NPU references reach about 1e-4 after Adam and L-BFGS. These are different-size networks, not a matched-capacity experiment.
- Both constructions remain accurate after export to ordinary Float64 Torch MLPs. All models, measurements and the 1×2 figure are saved.

## Question

Does the inverse-square-root construction extend to two simple modifications that retain a nearby singularity, using the same scale geometry and numerical settings?

## Procedure

On [0.0001,1], use the known derivative of each target, the same beta(17,17) cutoff, 15 dyadic scales, lambda=.25, stencil half-width 160, and 32 halo neurons per side per block. Each scale now has its own derivative samples. No training or readout fit is used. Construction uses 30-decimal arithmetic and exports Float64 weights. Sweep the same 22 widths, 990–5055, used for inverse sqrt; retain N=256 (width 4815) as the reference model independently of the observed minimum.

NPU references reuse expK04's seed-0 validation-selected shallow-32 recipes: 20k Adam steps, rates .01 and .001 respectively, then up to 500 full-batch L-BFGS iterations. NPU receives the original function on the positive interval, without the artificial extension. Importance-weighted log sampling targets physical-dx error. No new NPU training was needed.

## Measurements

| Target | QUILL, width 4815 | NPU, width 32 | NPU same recipe, fresh seed |
|---|---:|---:|---:|
| 1/(sqrt(x)+x) | 5.5390e-14 | 1.03711e-4 | 1.28298e-4 |
| exp(-x)/sqrt(x) | 5.4248e-14 | 1.22795e-4 | 6.68616e-4 |

All numbers are relative L2 in physical dx. The NPU references have 97 trainable parameters; the selected QUILL networks have 14446 parameters. The figure plots QUILL's width curve and fixed-width NPU horizontal references, with that distinction explicit in the axis and legend. Unsmooth lines connect actual measurements; white markers show sampled widths.

## Verification

Six tests passed. Analytic derivatives match autograd; the partition's maximum relative discrepancy is below 6.6e-16. Order-256 and order-512 QUILL errors are 5.54e-14/5.61e-14 and 5.42e-14/5.49e-14. Ordinary Torch inference gives 4.33e-14 and 4.30e-14; low-level floating-point evaluation changes the floor slightly. Increasing halo to 64 gives 5.54e-14 and 5.39e-14.

On 45,056 endpoint-inclusive log, uniform and random points per function, maximum relative errors are 3.56e-13 and 3.52e-13. Evaluating saved Float64 weights at 60 decimal digits on 25 log-spaced points gives maximum relative errors 1.99e-13 and 2.88e-13. Reloaded NPU checkpoints reproduce their saved errors under independent orders 256/512.

## Interpretation and limits

The construction generalizes beyond the pure power without changing its scale geometry. It uses analytic derivative information and an explicit multiscale partition. This is a multiscale extension of QUILL, not the unchanged global uniform-grid construction, and the C17 continuation is not globally analytic.

The NPU screen is illustrative, not exhaustive. In particular, 1/(sqrt(x)+x) has an exact small compositional NPU representation; the saved residual reflects the tested training procedure, not an expressivity impossibility. The graph establishes the measured precision gap for these saved runs, not that NPU can never reach high precision.

## Artifacts

`singular_functions_1x2.png` and `.pdf`; `scaling.json` (44 measurements);
`specification.json`; `verification.json`; `figure_provenance.json`; per-target
`N*_R32.npz` and measurement JSON, `selected_torch.pt`, `selected_predictions.npz`,
and `halo_check_N256_R64.npz`. Source and reproduction instructions:
`experiments/expK05_singular_function_comparison/README.md`.

## Next decision

The requested two-panel construction comparison is complete. A capacity-matched NPU sweep would be a separate experiment; the present figure labels its references explicitly.

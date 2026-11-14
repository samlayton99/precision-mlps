# Identical-width NPU and QUILL scaling

NPU and RealNPU use all 22 hidden widths from the saved QUILL construction. At every point their training nodes and C17 target values match that construction's own mathematical halo-center grid exactly. One fixed seed, Adam lr=0.001, batch 100, 20k updates, float64; no extra LR/seed search.

- Best NPU: 0.10441468 relative L2 (10.441%), width 2,175, 6,526 parameters, selected step 5,750.
- Best RealNPU: 4.4552594 relative L2 (445.53%), width 5,055, 10,111 parameters, selected step 3,250.

| Width | QUILL relative L2 | NPU best trained relative L2 | RealNPU best trained relative L2 |
|---:|---:|---:|---:|
| 990 | 32.3053 | 0.2425674 | 11.61347 |
| 1,005 | 2936.957 | 0.2716682 | 11.5741 |
| 1,035 | 751.3936 | 0.2120105 | 11.38502 |
| 1,095 | 21.05117 | 0.2362618 | 15.07577 |
| 1,215 | 1.301729 | 0.1738128 | 12.34661 |
| 1,335 | 0.2465354 | 0.1594513 | 10.62286 |
| 1,455 | 0.03336647 | 0.4335478 | 9.657188 |
| 1,695 | 0.0001295891 | 0.1185805 | 8.28674 |
| 1,935 | 1.145204e-07 | 0.11637 | 7.293045 |
| 2,175 | 2.394427e-09 | 0.1044147 | 6.707248 |
| 2,415 | 5.577986e-11 | 0.1714788 | 6.039421 |
| 2,655 | 7.061612e-13 | 0.6674871 | 5.99648 |
| 2,895 | 1.805481e-13 | 0.1695346 | 5.668437 |
| 3,135 | 7.580931e-14 | 0.1778277 | 5.459947 |
| 3,375 | 5.709563e-14 | 0.2099142 | 5.366152 |
| 3,615 | 4.927926e-14 | 0.1906178 | 5.343534 |
| 3,855 | 4.556356e-14 | 0.2390096 | 5.262591 |
| 4,095 | 4.796373e-14 | 0.1086124 | 4.977107 |
| 4,335 | 5.203951e-14 | 0.2375378 | 4.691345 |
| 4,575 | 5.096887e-14 | 0.155876 | 4.600073 |
| 4,815 | 4.953145e-14 | 0.1975725 | 4.482062 |
| 5,055 | 4.625644e-14 | 0.1561238 | 4.455259 |

## What matches

The same `Extension(eps=1e-4, order=16)` supplies targets; score on [1e-4,1] with physical dx relative L2. For QUILL interval count N and halo R=32, each block uses physical nodes x[j,n]=4*s[j]*n/N, n=-32,...,N+32. NPU/RealNPU train on values at exactly those nodes, with the same sample-index RNG per model/width. The physical halo changes with N, exactly as in the original construction sweep. All points use the unchanged C17 function.

Training minimizes unweighted minibatch MSE over the halo-center indices. The reported physical-domain error has a different domain and weighting; training is not directly minimizing that quadrature-weighted evaluation loss. The results therefore describe this specified halo-fitting protocol, not an architecture-only approximation lower bound.

The grid above contains the network centers and R=32 halo centers on each side. QI also uses Kc=160 extra derivative-stencil padding locations; all those extra prototype derivative values are exactly zero, outside its compact support. They are not added as extra constant-tail value observations to the training set. Bitwise checks compare baseline arrays generated from the shared mathematical center grid, not distinct floating-point evaluation paths inside the original mpmath convolution.

For each trained model, select the minimum logged order-32 validation error after at least one update; logs are every 250 steps. Independent orders 256 and 512 verify the test score. Every checkpoint, parameter count, selection step and per-width target array was checked.

The 4,815-unit runs are reused from the verified previous sweep because their grid/target and full training recipe are identical. Forty-two new runs supply the other points. The source data include widths 990 and 5,055 to bracket the displayed 1k–5k window.

## Interpretation

Width is matched by explicit user choice. NPU and QUILL each have 3h+1 parameters; RealNPU has 2h+1. They have different operations and information access: QUILL constructs weights from analytic derivatives and an anchor, whereas NPU/RealNPU fit function values. This is one seed and a fixed optimization budget, not a guarantee of best possible NPU accuracy.

The dotted **Adam baseline** uses only conventional initialization and training. The user disallowed the multiscale initializer; that result is excluded from the current figure. Twelve new 100k-step trials tested PyTorch default and standard Xavier initialization, rates .001/.01/.03, and batches 100/1024, with cosine decay after 50k. Four earlier eligible standard 128-neuron runs were also considered. The selected model has **128 neurons / 385 parameters**, **unmodified PyTorch initialization**, batch **1024**, and Adam LR **0.03**, decayed to 1e-5 by 100k. Its selected checkpoint is step **94,000**, and test relative L2 is **0.753494153**. Initial weights exactly match the standard library and are independent of the target domain. No special neuron centers/slopes, output gains, feature transforms or changed loss weights are used. The C17 target and fixed N=256/R=32 halo-center dataset are unchanged. See `../adam_standard/results.md` and `validation.json`. All older experiments remain archived. NPU/RealNPU curves retain their earlier recipe and 20k-step budgets.

Raw data and checkpoints are preserved. The figure is unsmoothed; white centers mark the actual sampled widths. See `summary.csv`, `validation.json`, `halo_datasets.json`, `sweep_specification.json`, and `figure_provenance.json`.

# Raw inverse square root with standard QUILL halo

The raw sample-fitted recipe is well-defined without a C17 extension, but does not reach machine precision over the tested width range. The best measured relative L2 is 1.0753e-4 at width 4225.

One uniform grid, lambda=.25, default halo max(20,ceil(sqrt(N))) per side, pivoted QR at machine-epsilon cutoff, and physical-domain centering. Only raw function values on [1e-4,1] are fitted. No exterior labels, derivatives, nonlinear input maps, multiscale blocks, or tuning.

| N | Width | Relative L2 | Effective width order |
|---:|---:|---:|---:|
| 64 | 105 | 1.603283e-01 | — |
| 128 | 169 | 9.427469e-02 | 1.12 |
| 256 | 297 | 4.709676e-02 | 1.23 |
| 512 | 559 | 2.795275e-02 | 0.82 |
| 1024 | 1089 | 6.786722e-03 | 2.12 |
| 2048 | 2141 | 1.317675e-03 | 2.42 |
| 4096 | 4225 | 1.075258e-04 | 3.69 |

Effective order is log(E_previous/E_current)/log(W_current/W_previous), a local empirical slope, not an asymptotic guarantee. Training sample count also follows the standard width-dependent rule. The last two resolution doublings reduce error by factors 5.15 and 12.25.

Physical-dx quadrature on logarithmic panels resolves the nearby singularity without changing the metric. Orders 64/128/256 agree; all saved models pass an independent order-384 reload check. Ordinary Float64 Torch evaluates the exported scalar MLPs.

The singularity at x=0 is only 1e-4 outside the domain. The uniform bank must resolve rapid variation near the left endpoint, and readout coefficients are large (L1 norm about 1.65e9 at the largest width). This is substantially harder numerically than a function with a well-separated singularity. No claim of the eventual convergence rate or machine-precision reach is made.

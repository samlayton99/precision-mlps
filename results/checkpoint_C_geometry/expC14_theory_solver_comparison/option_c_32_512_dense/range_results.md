# Dense original-recipe comparison, N=32–512 — draft-pending-Sam

The figure contains 31 independently computed widths: $N=32,48,64,\ldots,512$. Each width evaluates all six functions with both solvers, giving 186 paired cases. Lines connect measurements; no spline or fitted smoothing is used. The primary independent midpoint grid has 16,384 points; the validation grid has 32,768 points. Their error estimates differ by at most 0.282%.

## Fixed configuration for this range

Both arms use $a_H=11/8$, $\delta=4$, $\Delta=9/16$, $\lambda=1/4$, $R=\lceil a_H\sqrt N\rceil$, and $M=8N+1$. These constants were selected to satisfy the original geometric hypotheses starting at $N=32$ and held fixed across all functions and widths. This is a new configuration, not additional points inserted into either earlier configuration. In particular, the smaller halo multiplier makes the unchanged analytic cutoff bound more conservative and delays the theoretical curves' convergence.

The original reference envelopes, reference scaling, $E_{\rm rep}$, and A.65 cutoff are unchanged. The native FP64 Jacobi driver and validated numerical-bound implementation are the same as the preceding Option C run. All 186 geometry/recovery/cutoff checks pass. Nine relevant tests pass.

## Intermediate widths and original theorem scope

Most intermediate $N$ have nonrepresentable rational cell endpoints. Consequently this experiment uses the original A.81 uniform-error statement, which does not require cellwise input-rounding preservation. It does not claim the A.80 specialized RMS input-rounding premise for these widths. The plotted relative RMS errors are measured independently.

A.81's sampled maximum-residual input is bounded by $\sqrt M R_X$, where $R_X$ is supplied by the original A.68 recovery bound. The original A.81 bound is stored for every run, including its input-rounding and activation-evaluation allowances. It is often loose and is not substituted for the plotted measured error.

For arbitrary integer $N$, the native input weight $\gamma=N/8$ and hidden biases $\beta_j=N/8-j/4$ are exactly representable dyadic FP64 values. They are stored directly; rounded center coordinates are not used for inference. At nominal training nodes $z_i=-1+2i/(Ns)$, the preactivation is exactly $\lambda(i/s-j)$. Computing this identity gives the theorem's prescribed matrix without introducing geometric rounding errors. Target labels are evaluated at the same nominal rational nodes and rounded once. Independent dyadic evaluation grids also have exactly representable affine preactivations. Lookup tables return correctly rounded 53-bit tanh values. Readout fitting and summation remain FP64.

The larger analytic neighborhood makes the chirp's global class bound exceed FP64's exponent range. Analytic bounds and their scalar verification remain in 256-bit MPFR with extended exponent range and are serialized as decimal strings when needed. No bound is dropped or clipped. Network arrays, model SVD, projections, divisions, coefficients and evaluation remain FP64. Scalar bound verification should not be described as FP64-only.

## Files and reproduction

- Figure: `comparison.png`, `comparison.pdf`, `comparison.svg`.
- Data: `N*.json`, `coefficients_N*.npz`, `summary.csv`, `summary.json`. Reconstruct exported networks using `gamma` and `hidden_biases`; the saved rounded centers are informational only in this affine mode.
- Configuration: `experiments/expC14_theory_solver_comparison/option_c_32_512_dense.yaml`.
- Run `option_c.py --config` with that configuration and `--output` set to a fresh results directory; plot it with `plot_option_c.py`.

The theoretical recipe is highly conservative at small $N$ in this fixed family. The plot retains that result, including zero-rank fits with relative error one, rather than replacing the original cutoff or tuning settings to evaluation errors.

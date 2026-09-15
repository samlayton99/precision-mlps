# Checks for the gamma interpretation discussion

These are read-only calculations on existing expD24 data and elementary quadrature checks. No training was repeated and no optimizer was changed. Interpretations of training curves remain conditional on their objective, coordinates, and numerical readout solve.

## Existing fixed-residual probe

Source: `experiments/expD24_gd_residual_spectrum/direct_scale_test.py::matched_probe`; data: `results/checkpoint_D_optimizers/expD24_gd_residual_spectrum/direct_scale_test/data/matched_residual.npz`.

For the carrier 14*pi, use a unit-L2 residual proportional to `exp(-x*x/(2*0.4**2))*sin(14*pi*x)`, center zero, readout one. The global sign of the residual does not affect the magnitudes below. Let

\[
B(\gamma)=\int_{\mathbb R}|e(x)|\,|x|\operatorname{sech}^2(\gamma x)\,dx,
\qquad
G(\gamma)=\left|\int_{\mathbb R}e(x)x\operatorname{sech}^2(\gamma x)\,dx\right|.
\]

G is taken from the saved analytic Fourier-pairing calculation. B was integrated on [-4,4], splitting at every sine zero; the omitted Gaussian/tanh tails are negligible for these displayed values. B is a spatial absolute-overlap bound, not the Fourier magnitude-overlap bound used in the separate training frequency-pairing figure.

| Gamma | B | G | G/B |
|---|---:|---:|---:|
| 1 | 0.263903682 | 1.22853574e-24 | 4.65524e-24 |
| 4 | 0.0689329065 | 5.15723849e-7 | 7.48153e-6 |
| 16 | 0.00591117489 | 0.00184591700 | 0.312276 |
| 64 | 0.000357419010 | 0.000356843184 | 0.998389 |

For a fixed residual, coefficient and center, B decreases monotonically with gamma because sech²(gamma*(x-z)) decreases pointwise in gamma. G need not decrease monotonically: the signed cancellation changes. In this example, the transition from gamma 16 to 64 reduces the available spatial overlap even while the fraction surviving cancellation increases.

Least-squares fits of log G against log gamma for saved gamma >=128 (through 256) give slopes:

| Carrier omega/pi | Slope |
|---|---:|
| 2 | -2.9978767 |
| 6 | -2.9863921 |
| 14 | -2.9297283 |

This supports the fixed-smooth-residual large-gamma asymptotic in this controlled probe, not a cubic law for raw GD trajectories.

## Centered derivative and raw derivative

Write H(x)=p(x)e(x), t=x-z, and consider a positive raw slope a=gamma, b=-gamma*z. At a given state the centered derivative is

\[
g_\gamma=c\int H(x)t\operatorname{sech}^2(\gamma t)\,dx
=\frac{c}{\gamma^2}\int uH(z+u/\gamma)\operatorname{sech}^2u\,du.
\]

Bounded H gives

\[
|g_\gamma|\leq\frac{2\log2\,|c|\|H\|_\infty}{\gamma^2}.
\]

If H is globally Lipschitz with constant K, subtract H(z) using the oddness of u*sech²u to obtain

\[
|g_\gamma|\leq\frac{\pi^2|c|K}{6\gamma^3}.
\]

For a fixed sufficiently smooth H, the corresponding leading asymptotic is `g_gamma ~ c*pi²*H'(z)/(6*gamma³)` when H'(z) is nonzero. These are already the mechanisms in note equations (5.7) and (5.9). Uniform boundedness and smoothness cannot be assumed automatically along training; H and its derivatives change with the network. A hard finite-interval density also requires care near its endpoints.

Raw SGD uses a different gradient:

\[
g_a=c\int H(x)x\operatorname{sech}^2(\gamma(x-z))\,dx
=g_\gamma+zg_b,
\qquad g_b=c\int H(x)\operatorname{sech}^2(\gamma(x-z))\,dx.
\]

For a fixed smooth H at a center away from boundaries,

\[
g_a=\frac{2czH(z)}{\gamma}+O(\gamma^{-3}).
\]

The leading coefficient can vanish (for example z=0 or H(z)=0). A bounded-H envelope without that smoothness assumption is

\[
|g_a|\leq |c|\|H\|_\infty\left(\frac{2|z|}{\gamma}+\frac{2\log2}{\gamma^2}\right).
\]

Numerical quadrature independently verified the moments integral |u|sech²u=2log2 and integral u²sech²u=pi²/6. A fixed Gaussian-sine H with center z=0.3 and gamma 128,256,512,1024 also approached both predicted leading coefficients after multiplying centered gradients by gamma³ and raw gradients by gamma. This was an illustrative integration check, not training.

## Saved raw-GD travel

For the positive-slope control trajectories, the measured total path equals

\[
\text{mean total gamma travel}
=0.002\sum_{t=0}^{1999}\operatorname{mean}_k|g_{a,k}(t)|.
\]

Recomputing this from saved `controls.npz` agrees with the stored path to at most 1.1e-13 absolute difference across all 16 cases.

| Target | Travel, gamma0=16 | Travel, gamma0=64 | Reduction factor |
|---|---:|---:|---:|
| Sine | 9.13463e-5 | 2.06292e-5 | 4.428 |
| Mixed sine | 2.70762e-4 | 9.89620e-5 | 2.736 |
| Runge | 6.65019e-6 | 1.37833e-6 | 4.825 |
| Gaussian whole line | 1.47721e-4 | 4.18419e-5 | 3.530 |

These reductions are not a controlled estimate of an exponent: residuals, readouts, and centers evolve differently across runs. The empirical path sums use raw slope gradients, not the centered gradients in the Fourier diagnostic.

At gamma0=64, final mean|raw g_a| divided by mean|centered g_gamma| is approximately 184 for sine, 52.4 for mixed sine, 39.0 for Runge, and 25.7 for the Gaussian case. Thus using the centered gradient directly as the observed raw scale update would be a substantial error here.

## Interpretation of approximately flat readout-refit error

For exact unregularized least squares with the same weighted norm, the current squared loss is the best readout loss plus the squared prediction distance to a best readout fit:

\[
L(\theta,v)=L_*(\theta)+\tfrac12\|A(\theta)(v-v_*)\|^2.
\]

If L_* is unchanged, reduced training loss closes this optimization gap. This does not uniquely identify improved conditioning of readout GD: geometry steps can directly reduce that gap as well. The actual numerical readout diagnostics use a relative singular-value cutoff of 1e-13 and target-specific evaluation errors. The frozen control updates both a and b in its joint arm, so its benefit does not isolate scale motion from center motion.

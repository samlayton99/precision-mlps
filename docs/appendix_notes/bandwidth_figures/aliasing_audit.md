# Independent audit of the bandwidth refinement and rightward drift

Audited on 2026-09-26. No training or replacement selector was run. This review uses the supplied note, the supplied appendix, the implemented scalar rule, the saved FP64 sweeps, and independent evaluations of the Fourier replica formula. The figures and selector remain under the parent agent's ownership.

## Finding

The rightward movement with increasing width is a genuine prediction of the stated rule, not a sign error. At fixed bandwidth and target frequency, the replica error decreases as the center spacing decreases. The rule can therefore allow a larger bandwidth while maintaining its chosen scalar budget.

However, the current refinement substitutes one average frequency for the entire target spectrum. That shortcut can substantially underestimate aliasing. It is not an error certificate and does not reliably locate the measured error minimum. This is a real limitation in the current figures, not merely a warning about possible numerical conditioning.

A direct check identifies it: for the Gaussian target at N=32, evaluating the replica formula over the full Fourier spectrum predicts the observed errors for all four activations to better than 0.01%. The centroid version assigns those same selections a score of 2^-52, while their measured errors are about 10^-12. The Fourier mechanism works in this check; reducing the target to one frequency does not.

## Sources and scope

- Supplied note: `/Users/sam/.codex/attachments/063e37bb-84f9-47ce-8b97-321a4883c9c9/choosing_optimal_lambda_overleaf (2).pdf`, especially equations (3), (A.5)–(A.14), and pages 6–8.
- Supplied appendix: `/Users/sam/.codex/attachments/ec79f374-507d-411f-9ec0-b0e6e1024ae1/Pasted text.txt`, especially the weighted bound, the frequency-input discussion, and the width derivative.
- Current appendix: `docs/appendix_notes/bandwidth_figures/bandwidth_appendix_complete.tex`.
- Implementation: `experiments/expC09_bandwidth_figures/appendix_bandwidth.py` and its imported `docs/lambda_theorem_compatibility/choosing_optimal_lambda/reproduce_figures.py`.
- Data: `results/checkpoint_C_geometry/expC09_bandwidth_figures/appendix_bandwidth/data/error_curves.json` (3,200 curve observations) and `refined_marker_fits_4x4.json` (80 directly evaluated selected points).

The comparisons below concern five values of the interior interval count N: 32, 64, 128, 256, 512. They are not comparisons at fixed total neuron count. The measurement protocol has 32 halo centers on each side, so total hidden width is W=N+65.

## 1. The formula and its implementation agree

Write h=2/N, theta=h*omega, and let K be the localized r-th derivative of the activation. The signed output multiplier for replica m is

\[
t_m(\lambda,\theta)=
\left(\frac{\theta}{\theta+2\pi m}\right)^r
\frac{\widehat K((\theta+2\pi m)/\lambda)}
     {\widehat K(\theta/\lambda)}.
\]

The plotted practical score is the magnitude sum of the nearest pair,

\[
T_{K,r}(\lambda,\theta)=|t_{-1}|+|t_1|,
\qquad T_{K,r}(\lambda,2\bar\omega/N)=2^{-52}.
\]

The code uses the original central denominator, the frequency factors with the correct derivative order, and stable logarithmic arithmetic. It does not replace the central transform by one. Tanh and erf use r=1; GELU and SiLU use r=2. The implemented transforms are correct for exact GELU and erf(x), rather than a tanh approximation to GELU or a rescaled erf.

For tanh, putting a=pi^2/lambda and chi=2*pi*omega/(N*lambda), the note's controlled approximation is

\[
T_{K,1}\simeq 2e^{-a}\sinh\chi.
\]

This is the same compact prescription as the supplied appendix. There is no derivative-order or missing-frequency-factor bug explaining the trend.

## 2. Why increasing N moves the selection right

At fixed lambda and small theta,

\[
T_{K,r}(\lambda,\theta)
\sim 2\left(\frac{\theta}{2\pi}\right)^r
\frac{\widehat K(2\pi/\lambda)}{\widehat K(0)}.
\]

For a fixed target frequency, theta=2*omega/N. Thus this alias score falls as N^-r, provided the small-angle approximation applies. Increasing lambda raises the transform ratio, compensating for the width factor. It is therefore consistent to choose a larger lambda at larger N while preserving the same score.

The exact derivative for the compact tanh root is already in the supplied appendix:

\[
\frac{d\log\lambda}{d\log N}
=\frac{\chi\coth\chi}{a-\chi\coth\chi}>0
\]

on its increasing branch. At small chi, this is approximately 1/(a-1). Near the FP64 regime it is a small positive number, not zero.

For the mixture's centroid 30*pi/7:

| N | Selected tanh lambda | Pair score at fixed lambda=.25 |
|---:|---:|---:|
| 32 | .200478 | 2.80041e-13 |
| 64 | .237151 | 1.41570e-15 |
| 128 | .255528 | 1.00152e-16 |
| 256 | .265292 | 2.49327e-17 |
| 512 | .271894 | 1.01644e-17 |
| 1024 | .277711 | 4.81681e-18 |
| 16384 | .302122 | 2.95662e-19 |

The note explicitly rejects convergence of the fixed-precision refined root to the general-rule constant (page 6). Its useful claim is slow finite-range variation. The function enters the scalar root only through omega/N, so plots in N are horizontal rescalings for different representative frequencies. Conversion to W adds the halo correction and breaks exact horizontal rescaling in W.

This does not mean an arbitrary rightward movement is harmless. The argument preserves the score at the chosen representative frequency. Whether that score controls the actual target is the separate issue below.

## 3. The centroid shortcut can miss the aliasing error by many orders

Consider the measured mixture

\[
f(x)=\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(10\pi x).
\]

Its amplitude-weighted frequency is 30*pi/7. At N=32 and tanh lambda=.200478, the centroid score is 2.22045e-16 by construction. But the score increases sharply with frequency, and the 10*pi component matters much more than evaluation at the mean suggests.

For amplitudes A=(1,1/2,1/4) and frequencies omega=(2*pi,6*pi,10*pi), I separately computed:

\[
\frac{\sum_j A_jT(\lambda,h\omega_j)}{\sum_j A_j},
\]

and the ideal unanchored first-replica relative RMS

\[
E_{\rm pair}=
\left[
\frac{\sum_j A_j^2\bigl(|t_{-1}(\lambda,h\omega_j)|^2+
                              |t_1(\lambda,h\omega_j)|^2\bigr)}
     {\sum_j A_j^2}
\right]^{1/2}.
\]

The RMS calculation uses orthogonality of distinct Fourier replicas. For these three frequencies and the five powers-of-two N, their first aliases do not collide. It is a calculation for the central-amplitude-matched infinite lattice, not a theorem about the FP64 finite least-squares solve. No rate or scale was fitted to the measurements.

| N | Tanh centroid lambda | Whole-mode mean pair score / 2^-52 | Ideal pair relative RMS | Measured selected error |
|---:|---:|---:|---:|---:|
| 32 | .200478 | 6.17716e6 | 2.09514e-9 | 1.90814e-9 |
| 64 | .237151 | 243.923 | 8.18274e-14 | 5.18077e-13 |
| 128 | .255528 | 5.45910 | 1.55278e-15 | 1.05866e-13 |
| 256 | .265292 | 1.53479 | 3.05851e-16 | 2.40278e-14 |
| 512 | .271894 | 1.10766 | 1.74231e-16 | 3.36527e-14 |

The understatement decreases with width as the individual frequencies enter the small-angle regime. For the other activations, the whole-mode mean score divided by 2^-52 is:

| Activation | N=32 | N=64 | N=128 | N=256 | N=512 |
|---|---:|---:|---:|---:|---:|
| GELU | 7.15198e6 | 465.546 | 15.6789 | 3.78161 | 1.97965 |
| SiLU | 6.17717e6 | 245.394 | 6.13341 | 2.04450 | 1.57842 |
| erf | 7.30453e6 | 364.848 | 9.02330 | 2.13608 | 1.24938 |

For r=2 the arithmetic centroid does not even recover the leading frequency moment. The small-angle weighted score involves mean(omega^2), while the centroid substitution uses mean(omega)^2. Their ratio for this mixture is 1.46222. The corresponding quadratic frequency scale is 16.28095, compared with the arithmetic mean 13.46397. This is an additional limitation for GELU and SiLU, not an implementation mismatch with the requested note.

For tanh the supplied appendix already identifies the issue correctly: convexity of sinh implies that evaluating at the mean is smaller than the weighted mean of the scores. That warning is materially relevant to these figures.

## 4. An especially clean check: the Gaussian target

For f(x)=exp(-20x^2), its whole-line Fourier magnitude is proportional to exp(-omega^2/80). Its arithmetic amplitude centroid is sqrt(80/pi), which is the value supplied to the selector. Rather than replacing the spectrum by this centroid, I integrated the squared first-replica multipliers over the Fourier power distribution:

\[
E_{\rm pair}^2=
\frac{
\int_{-\Omega}^{\Omega}e^{-\omega^2/40}
\left(|t_{-1}(\lambda,h\omega)|^2+|t_1(\lambda,h\omega)|^2\right)d\omega}
{\int_{\mathbb R}e^{-\omega^2/40}d\omega},
\qquad \Omega=\pi/h.
\]

This is the Parseval calculation for aliases of the retained band. Its assumptions and limitations are explicit: it is a whole-line, central-amplitude-matched lattice calculation, retaining the first pair and excluding the target beyond the Nyquist band. At N=32 the excluded target relative L2 norm is only 5.10e-15, far below the observed errors. The physical Gaussian is also strongly localized inside [-1,1].

At N=32:

| Activation | Selected lambda | Full-spectrum first-pair RMS | Measured relative error | Measured/predicted |
|---|---:|---:|---:|---:|
| Tanh | .246336062 | 1.8565225e-12 | 1.8566607e-12 | 1.00007 |
| GELU | .720240770 | 3.5706429e-12 | 3.5709235e-12 | 1.00008 |
| SiLU | .493164351 | 1.8838968e-12 | 1.8838586e-12 | .99998 |
| erf | .517889278 | 2.8445078e-12 | 2.8445432e-12 | 1.00001 |

These are independent evaluations of the formula, not calibrated predictions. The agreement strongly supports the interpretation that aliasing explains this particular discrepancy. There is no need to invoke halo error or SVD error to explain an error of this size here.

The selected tanh Gaussian first-pair RMS at N=64,128,256,512 is respectively 8.14e-16, 2.11e-16, 1.54e-16, 1.43e-16. The measured values are 1.61e-14, 1.03e-14, 1.29e-14, 1.63e-14. The spectral estimate no longer accounts for the observed floor. Finite geometry, finite sampling, floating-point evaluation, coefficient recovery, and the solver cutoff remain possible contributors. This audit did not vary those factors and cannot identify which dominates.

## 5. Analytic targets can also have important frequencies beyond the retained band

For exp(sin(3*pi*x)), the coefficients have magnitudes I_|j|(1) at frequencies 3*pi*j. For sech(5x), the whole-line transform is (pi/5)*sech(pi*omega/10). Both allow a calculation of the actual target energy beyond Omega=pi/h.

The following are diagnostics for tanh at the centroid-selected lambda. “Target tail” is the target's relative L2 norm above that Fourier cutoff. It is not asserted to be a finite-interval network approximation lower bound.

| Target | N | First-alias RMS from retained target band | Target tail above cutoff | Measured selected error |
|---|---:|---:|---:|---:|
| exp(sin(3*pi*x)) | 32 | 1.92980e-5 | 2.11179e-5 | 1.60821e-5 |
| exp(sin(3*pi*x)) | 64 | 2.74010e-11 | 1.17090e-11 | 2.54528e-11 |
| exp(sin(3*pi*x)) | 128 | 5.22334e-16 | 2.00901e-28 | 4.31963e-14 |
| sech(5x) | 32 | 1.62031e-7 | 1.96070e-7 | 2.16978e-7 |
| sech(5x) | 64 | 6.35703e-14 | 2.71837e-14 | 8.20697e-14 |
| sech(5x) | 128 | 1.62648e-16 | 5.22517e-28 | 1.44660e-14 |

Unlike the N=32 Gaussian check, the target tail here is comparable to the first-alias estimate at low N. Their interaction and the finite interval matter; the separate columns must not be added in quadrature and advertised as a certified total error. What they establish is that the representative frequency omits numerically significant parts of these targets. At larger N both idealized spectral quantities are far smaller than the measured FP64 errors.

## 6. Do the points become systematically worse at the higher widths?

I compared each of the 80 selected-point errors with the smallest error on its saved 40-point lambda grid. This grid minimum is an empirical comparator, not the continuous global minimum. The additional selected fit can sometimes be better than every grid point.

| N | Selected lambda lies right of grid-minimizing lambda | Median selected error / grid minimum | Maximum ratio | Cases above 10x grid minimum |
|---:|---:|---:|---:|---:|
| 32 | 14/16 | 7.27 | 852.50 | 6/16 |
| 64 | 16/16 | 2.71 | 18.26 | 2/16 |
| 128 | 12/16 | 3.59 | 24.04 | 3/16 |
| 256 | 14/16 | 1.48 | 29.49 | 2/16 |
| 512 | 15/16 | 1.65 | 9.38 | 0/16 |

The points do usually lie to the right of the measured minimum. But these data do not show increasing failure as N grows. At N=512 all 16 points are within a factor ten of their sampled minima; the median factor is 1.65. The median selected absolute error there is 2.32e-14. A minimum taken across a broad, noisy FP64 plateau can occur well left of the chosen lambda without implying a large error penalty.

There are still clear failures of an “optimal lambda” interpretation. For Gaussian erf at N=32, the selected error is 2.84454e-12 versus a grid minimum 3.33672e-15 (852x); for Gaussian GELU it is 3.57092e-12 versus 9.40533e-15 (380x). For tanh sech at N=256 the selected error is 3.55712e-14 versus 1.20639e-15 (29.5x). The latter is a small absolute error but a real relative penalty.

The worst small-width errors are not explainable by a harmless plotting convention. Conversely, the high-width results do not support a claim that drifting right necessarily destroys accuracy.

## 7. What can be retained, and what should be stated differently

1. Retain the replica formula, the correct derivative orders, and the implemented root. They agree with the note, and the full-spectrum calculations give direct evidence for the mechanism.
2. Retain the observed rightward drift. The declining theta^r factor predicts it; forcing a constant asymptote would misrepresent the stated refinement.
3. Label the points as representative-frequency selections. Do not claim that setting their scalar score to machine epsilon guarantees a full-target error at epsilon or finds the total-error minimum.
4. Keep the full weighted tanh bound as the rigorous statement. The general-r replica identity is rigorous under its Fourier assumptions, but the note's finite-contour tanh proof is not automatically a finite-domain theorem for GELU, SiLU, or erf.
5. The target spectrum, anchoring constant, norm conversion, neglected replicas, finite halo, and numerical recovery are separate qualifications. Merely restoring the factor two or the tiny geometric tail correction does not repair the centroid understatement of millions at N=32.
6. If a later revision seeks a dependable target-specific alias budget, use a weighted spectrum or an appropriate controlled tail description. Choosing the maximum target frequency would be conservative; a mean is not. For r=2 even the leading moment is quadratic. These are proposed follow-ups, not modifications made in this audit.
7. Do not infer the source of the high-width numerical floor from the alias calculation alone. The data establish a remainder beyond this calculation; isolating SVD, arithmetic, and halo contributions would require their own controls.

The present figures are useful as a test of the practical rule, including its failures. They support “the Fourier ratio supplies a meaningful bandwidth scale.” They do not support “the centroid refinement accurately predicts the error-minimizing bandwidth for every analytic target.”

## Numerical method for the independent checks

Roots used the existing stable log transforms, but the full-target comparisons evaluated each actual Fourier component separately. For the sine mixture the formulas above are finite sums. For the Gaussian and sech calculations, adaptive scalar quadrature integrated first-replica squared magnitudes with absolute tolerance 1e-45 and relative tolerance 1e-10 on [0,pi*N/2]; evenness removes the duplicated negative half. Gaussian power normalization is sqrt(10*pi), and sech power normalization is 10/pi on this half-line. The exp-sine calculation sums the modified Bessel coefficients up to the frequency cutoff; its full period-averaged squared norm is I_0(2). No fitted network was trained or refit for these calculations, and no measured error entered any spectral prediction.

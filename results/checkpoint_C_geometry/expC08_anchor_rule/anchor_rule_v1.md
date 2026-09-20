# Lambda anchor: version 1

Finalized 2026-09-10 from `anchor_rule_2026-09-09.zip`.

**Decision.** Use the absolute-coefficient-weighted mean frequency and the full coefficient magnitude B as the default inputs. Preserve the tested kernel formula, denominator, tail correction, precision convention, and search interval. Report the raw anchor. Offer a 5% reduction as an optional practical margin, recorded separately.

**Claim.** This is a theoretically motivated, empirically tested predictor for choosing the shared slope of a fixed-grid network. The mean-frequency substitution is an approximation. The predictor does not certify machine-epsilon error for the complete finite least-squares network, and does not always locate the error minimum.

## 1. The rule to freeze

For the interval [-1,1], N is the number of interior grid intervals:

\[
h=\frac{2}{N},\qquad \lambda=\gamma h,\qquad \gamma=\frac{N\lambda}{2}.
\]

The activation is \(\psi\), and \(K=\psi^{(r)}\) is the integrable kernel used in the Fourier calculation. Define the normalized transform magnitude

\[
H(\xi)=\frac{|\widehat K(\xi)|}{|\widehat K(0)|},\qquad H(0)=1.
\]

Given a spectral representation \(f(x)=\sum_j b_j e^{i\omega_jx}\), use

\[
B=\sum_j|b_j|,\qquad
\bar\omega=\frac{\sum_j|b_j|\,|\omega_j|}{B},\qquad
\theta=\frac{2\bar\omega}{N}.
\]

Frequencies are angular frequencies, in radians per unit of x. The coefficients describe the original target f, not its derivative. The powers involving r below already account for the derivative-to-output conversion.

For \(0<\theta<\pi\), keep exactly the studied expression:

\[
\mathcal R_{K,r}(\lambda,N,\bar\omega)=
\frac{
\left(\frac{\theta}{2\pi-\theta}\right)^rH\!\left(\frac{2\pi-\theta}{\lambda}\right)
+\left(\frac{\theta}{2\pi+\theta}\right)^rH\!\left(\frac{2\pi+\theta}{\lambda}\right)
}{\min\{1,H(\theta/\lambda)\}}
\frac{1}{1-\rho_K},
\qquad
\rho_K=
\frac{H((4\pi-\theta)/\lambda)}{H((2\pi-\theta)/\lambda)}.
\]

The mathematical selection criterion is

\[
\boxed{\lambda_{\rm anchor}
=\sup\{\lambda:B\,\mathcal R_{K,r}(\lambda,N,\bar\omega)\le\varepsilon\}.}
\]

Version 1 evaluates this criterion on the original **600-point logarithmic grid from 0.03 to 1.5** and returns the largest feasible grid point. Adjacent prediction points differ by approximately 0.66%. This preserves the numerical experiment; it does not pretend to solve the continuous equation exactly.

The default precision budget is

\[
\varepsilon_p=2^{1-p}.
\]

This is machine epsilon, the spacing above one, for a binary significand with p bits. For p=53, the value is approximately 2.220446049250313e-16. Unit roundoff under rounding to nearest is half this value. A user-specified budget such as 5e-16 overrides the default explicitly.

Report the raw anchor even when using the optional practical choice

\[
\lambda_{\rm use}=0.95\lambda_{\rm anchor}.
\]

The reference implementation defaults to the raw anchor (`buffer=1.0`). Request the margin with `buffer=0.95`. The margin was examined after seeing these experiments; it is not an independently validated performance guarantee.

If the criterion remains satisfied at 1.5, return `upper_search_limit`. If no grid point is feasible, return `no_feasible_point`. If the representative theta reaches pi, return `frequency_outside_domain`. Never silently clip theta. A zero or constant target can use the network bias and does not require this rule.

## 2. Why this formula, and where the approximation enters

The policy is to make bumps sufficiently narrow for a practical finite construction while stopping before the modeled alias contribution becomes too large. Increasing lambda narrows the physical bumps relative to the grid and broadens the kernel transform. The broader transform leaves more unwanted spectral content at frequencies separated by grid harmonics.

For arbitrary fixed-grid weights, the Fourier calculation separates the coefficient-dependent periodic factor from the kernel-dependent factor. At nonzero frequencies, differentiating the activation r times gives the relative output amplitude

\[
\left|\frac{\widehat Q(\theta+2\pi m)}{\widehat Q(\theta)}\right|
=\left|\frac{\theta}{\theta+2\pi m}\right|^r
\frac{H((\theta+2\pi m)/\lambda)}{H(\theta/\lambda)}.
\]

The equality concerns a nonzero central response, and a real activation gives the even transform magnitude used here. The same periodic coefficient factor appears at the central and shifted frequencies and cancels. Choosing least-squares weights does not change this structural identity. Finite-interval fitting does change what can be inferred about the approximation error from a global spectral identity.

The rule retains the nearest shifted frequencies, m=-1 and m=1. The derivative-order powers describe how integration attenuates high-frequency output relative to the kernel response. The denominator `min{1,H}` is no larger than the exact central denominator H, so this replacement is conservative for this single-frequency calculation. Keeping this denominator also preserves the tested treatment of GELU and Swish, whose transform magnitudes need not decrease immediately away from zero.

The geometric factor includes the more distant shifts. A sufficient condition for the displayed rho to bound their successive ratios is that log H be concave over the alias arguments. Equally spaced increments then produce nonincreasing transform ratios. The additional derivative-order factors also decrease along each sequence. A convergent geometric series bounds the remaining contributions by the first contribution divided by 1-rho. This condition holds on the alias tails of the four transforms below within the version 1 search range. The paper's general activation assumptions alone do not establish this condition for every possible activation.

For a specified ideal response that matches each desired frequency, adding component error magnitudes leads to a bound involving

\[
\sum_j |b_j|\,\mathcal R_{K,r}(\lambda,N,|\omega_j|).
\]

That step needs the individual frequencies to lie in the domain of the calculation, along with the applicable tail bounds and a specified construction. The corresponding bound for a prescribed cardinal operator can have an additional central-amplitude deficit. A bound on an ideal spectral response also leaves finite centers, the coefficient solve, and numerical evaluation to be controlled.

**Version 1 makes the practical approximation**

\[
\sum_j |b_j|\,\mathcal R_{K,r}(\lambda,N,|\omega_j|)
\;\approx\;
B\,\mathcal R_{K,r}(\lambda,N,\bar\omega).
\]

Evaluating a nonlinear function at the mean of its inputs generally does not equal, or upper-bound, the mean of its values. The average frequency therefore cannot inherit the worst-frequency guarantee. A small mean also does not establish that every appreciable frequency is below Nyquist.

This approximation is nevertheless useful in the experiments: a maximum frequency can force lambda far to the left in small-width cases, where numerical fitting becomes much less accurate. The mean avoids imposing the highest-frequency requirement on the whole target. The same relaxation can underestimate the importance of a high-frequency tail, which is visible particularly with the Gaussian activation.

The connection to the QI paper is thus specific: the spectral ratios explain and estimate the alias contribution, and the rule allocates a tolerance to that contribution. Version 1 compresses the target spectrum into B and one representative frequency and applies the estimate to a different finite least-squares implementation. The QI paper does not, by itself, prove the complete mean-frequency predictor or a machine-precision guarantee for this implementation.

## 3. Target inputs and activation conventions

For a continuous Fourier representation
\(f(x)=\int e^{i\omega x}\,d\nu(\omega)\), the same practical inputs are

\[
B=\int d|\nu|(\omega),\qquad
\bar\omega=\frac{\int|\omega|\,d|\nu|(\omega)}{B},
\]

when both integrals are finite. If f is specified only on the fitting interval, these inputs concern a chosen extension or a chosen finite Fourier approximation. Local analyticity alone does not specify a unique global spectrum or automatically justify an arbitrary extension. When the spectrum is unknown, estimated B and frequency give a heuristic input; no additional accuracy guarantee is claimed.

Retain DC coefficients in B and the average, exactly as the experiment did. Removing an exactly representable bias before forming the inputs is a reasonable alternative, but it would be a new, untested variant.

The exact mean-frequency inputs used here are:

| Target from the study | B | Mean angular frequency |
|---|---:|---:|
| Mixture at pi, 3pi, 5pi; amplitudes 1, 1/2, 1/4 | 7/4 | 15pi/7 |
| Mixture at 2pi, 6pi, 10pi; same amplitudes | 7/4 | 30pi/7 |
| Mixture at 4pi, 12pi, 20pi; same amplitudes | 7/4 | 60pi/7 |
| 1/(1+25x²), using its whole-line extension | 1 | 5 |
| exp(-20x²), using its whole-line extension | 1 | sqrt(80/pi) |
| exp(sin(3pi x)), using its periodic series | e | 3pi[I_0(1)+I_1(1)]/e |

I_n denotes the modified Bessel coefficient. The last calculation uses orders -40 through 40 in the experiment; the omitted coefficients are negligible at the tested precision. Each real wave with amplitude a contributes a conjugate pair of coefficients of magnitude |a|/2, hence total contribution |a| to B.

The four built-in normalized kernel transforms are:

| Activation | r | H(xi) |
|---|---:|---|
| tanh(x) | 1 | a/sinh(a), a=pi xi/2 |
| x Phi(x), exact GELU | 2 | (1+xi²) exp(-xi²/2) |
| x/(1+exp(-x)), Swish with scale 1 | 2 | a² cosh(a)/sinh²(a), a=pi xi |
| exp(-x²) | 0 | exp(-xi²/4) |

The continuous value at xi=0 is one in every row. Rescaling an activation changes H and changes the numerical lambda prediction. Approximate GELU and a scaled Swish must not silently use the exact versions above.

The implementation accepts another `Kernel(name, order, log_transform)`. This extends the formula mechanically. It does not extend the four-activation empirical validation or automatically verify the geometric-tail estimate. A new activation needs its own transform and tail justification; a transform with zeros or oscillatory tails may require summing aliases explicitly.

**Amplitude units remain explicit.** The experiment uses B in the original target's amplitude units, epsilon as the spacing at one, and evaluates relative sampled L2 error. These are preserved for reproduction. They do not make the anchor invariant under multiplying the target by a constant. For an amplitude-invariant extension, normalize the target before forming B or scale epsilon with the target amplitude. That is a separate convention change, not part of these results.

## 4. What the reviewed data establish

The reference implementation reproduces all 576 prediction-table entries: 288 mean and 288 maximum-frequency entries, including four undefined maximum-frequency entries. The maximum relative difference for finite predictions is below 7e-16, due to floating-point evaluation of the logarithmic grid.

The sweeps contain 288 plotted configurations and 264 distinct activation/target/N/p combinations, since N=128, p=53 appears in both sweeps. The stored curves contain 11,520 least-squares solves. The original observed lambda grid has 40 points, approximately 10.55% apart. This grid supports the broad location of the transition more strongly than percent-level claims about an optimum.

One small-width example makes the tradeoff visible. For the mixture at frequencies (2,6,10)pi, N=32, p=53, the original tables give:

| Activation | Maximum-frequency anchor | Error there | Mean-frequency anchor | Error there |
|---|---:|---:|---:|---:|
| tanh | 0.10108 | 8.29e-3 | 0.19678 | 1.49e-9 |
| GELU | 0.44809 | 4.74e-11 | 0.62930 | 1.20e-9 |
| Swish | 0.20199 | 6.68e-2 | 0.39322 | 1.86e-8 |
| Gaussian | 0.31699 | 3.54e-12 | 0.44228 | 1.04e-9 |

Errors in this table are log-interpolated from the original 40-point curves. They are relative sampled L2 errors. The mean prevents severe tanh and Swish failures in this example; the maximum remains more accurate for GELU and Gaussian. A claim that the mean wins every case would be incorrect.

For the precision sweep, median error relative to each curve's smoothed minimum is:

| Activation | Uncapped mean-anchor cases | Raw mean anchor | 0.95 times mean anchor |
|---|---:|---:|---:|
| tanh | 40 | 1.48 | 1.22 |
| GELU | 31 | 1.30 | 1.12 |
| Swish | 30 | 1.06 | 1.02 |
| Gaussian | 42 | 2.01 | 1.18 |

These diagnostics use identical cells before and after applying the margin, and the same five-point median of log error in both numerator and denominator. The margin values are interpolated from the existing curves, so targeted fresh solves were also checked. The complete summaries, including 90th percentiles and width results, are in `validation.json`.

The targeted reruns show why the buffer must not be advertised as universally beneficial. At N=32, the mixed target's Gaussian error decreased from approximately 1.10e-9 to 1.45e-10 after multiplying the anchor by 0.95. For tanh on the same target, the error increased from 1.31e-9 to 8.64e-9. Moving lambda left reduces modeled aliasing in this regime but can worsen numerical fitting. The 28 targeted evaluations are recorded in `direct_anchor_checks.json`; the original full sweep was not rerun.

There is also an exact representational exception: with Gaussian activation, the target exp(-20x²) is a single neuron centered at zero when gamma=sqrt(20). Thus lambda=2sqrt(20)/N gives an exact representation before numerical error. At N=32, this lambda is about 0.27951; a targeted solve gave relative error about 4.53e-15. A generic spectral anchor need not predict such special minima.

The implemented rho correction is small. Across fp64 width predictions, the largest rho was about 1.78e-14. Across the entire mean-frequency precision sweep, the largest was about 0.002794, giving a correction below 1.002803. Keeping rho is inexpensive; dropping rho cannot explain or repair the larger small-width discrepancies in these results.

## 5. Corrections to the experiment's reporting

1. **The non-tone comparison changes two inputs.** For mixtures, max and mean use the same B. For the other three targets, the version called max uses a two-standard-deviation point of spectral energy and truncates B there; the mean version uses full B. The non-tone comparison does not isolate frequency choice, and a two-standard-deviation point is not a true maximum frequency. Preserve the old runs as a comparator with this precise description.

2. **An upper search limit is not a measured threshold.** There are 25 mean and 22 maximum-frequency precision predictions at lambda=1.5. Exclude these cases when claiming the location of a threshold beyond the search interval. The reference implementation marks them explicitly.

3. **Use the precision implemented in the code.** The SVD cutoff is 2^(1-p), matching epsilon, rather than the README's occasional 2^(-p). Below 53 bits, matrix values, target samples, coefficients, products, and partial sums are rounded as specified in the original script, but the SVD solver internals remain fp64. Describe the results as precision emulation, not native low-precision hardware validation.

4. **Keep the actual width definition.** A halo of 32 centers per side gives W=N+65 activation neurons, plus a bias coefficient. N is not the total neuron count. N=32 means 97 activation neurons. The evaluation norm is relative L2 on 4001 points, not a certified continuous supremum norm.

5. **Use comparable error summaries.** The original E(anchor) uses raw log interpolation while E_min uses smoothing; their ratio can therefore be below one. The additional audit uses the same smoothing for both quantities. Neither a 10-times-minimum shoulder nor the supplied global spectral projection is automatically a rigorous lower bound for every finite-interval least-squares fit.

## 6. Frozen use and future changes

The default pipeline is: form full B and the weighted mean angular frequency; compute theta=2 omega/N; evaluate the unchanged first-pair expression with rho; choose the largest feasible lambda on the original grid; convert to gamma=N lambda/2. Record any optional 0.95 margin and any search-limit status.

Keep the raw mean-frequency prediction as the version 1 result. Keep the maximum-frequency predictor as a diagnostic comparison. Do not select between max and mean after inspecting each target's error curve, silently change B, remove the denominator minimum, or add activation-specific frequency caps while retaining the same version label.

Using a weighted sum over all actual frequencies remains the route to a more detailed spectral estimate; a proved construction-specific guarantee must also control the omitted target tail, finite construction, and numerical error. These requirements explain the distinction between the theoretical origin of the rule and its empirically useful one-frequency simplification.

For a new activation, normalization, target class, solver, or halo, regard version 1 as a starting prediction and report that setting separately. The present evidence supports freezing a useful experimental setup, rather than a universal optimal-lambda theorem.

## Files and reproduction

- `anchor_rule_v1.py`: standard-library reference implementation, with four built-ins and a custom-kernel interface.
- `verify_anchor_rule_v1.py`: checks both original prediction tables and recomputes curve diagnostics; requires NumPy.
- `validation.json`: regression results, matched-cell comparisons, and margin diagnostics.
- `direct_anchor_checks.json`: targeted fresh solves using the original experimental fitting function.

Run `python anchor_rule_v1.py` for an example. Run `python verify_anchor_rule_v1.py /path/to/anchor_rule_2026-09-09` against the unpacked original bundle to reproduce the table checks and diagnostics. The original source data remain unchanged.

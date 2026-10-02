# expD41 — Localized variational activation versus tanh and a spectral notch

Status: measurements complete; interpretation draft-pending-Sam. Codex version, 2026-09-29.

**Initialization clarification:** Sam clarified that QI initialization means QI hidden geometry with zero readout. The initial batch documented here used a derivative-constructed readout. Its figures now explicitly say "Constructed QI"; the raw runs are preserved. The frozen-geometry follow-up uses zero readout and zero output bias, with random-readout controls on the same QI geometry. These initialization variants must not be conflated.

## TL;DR

- The localized variational activation does not win uniformly. From QI it trains close to its refitted error, but that error is much higher than tanh's on all three targets. From Xavier it has lower median actual error on sine and slightly lower error on the mixture, but higher error on Runge.
- The spectral-notch control is not uniformly poor. Its final QI sine geometry admits a readout with error about $3\times10^{-14}$, while Adam's own readout is about $5\times10^{-5}$. A notch in one kernel scale does not imply failure for the tested finite geometry or trainable network.
- Construction quality, current-geometry projection quality, and Adam's achieved error remain distinct. Tanh's sine construction starts at $2.6\times10^{-15}$ and ends at $1.5\times10^{-5}$ under the shared schedule, while its final refit is $2.7\times10^{-11}$.
- All 48 runs reached 10,000 steps. Denser-grid endpoint checks preserve the observations; least-squares cutoff sensitivity limits interpretation of exact refit values. These results do not establish a universal activation ranking or contradict the fixed-lattice projection theorem.

## Question / hypothesis

Does an activation designed to provide a stable, accurate projection on a uniform lattice also improve ordinary Adam training? Sam requested a comparison of a standard activation, a deliberately unfavorable spectral-notch activation, and a variationally designed activation, each starting from constructive QI and standard Xavier. Sam subsequently selected a localized regularized design for the main third column; integrated sinc remains a supplementary reference.

## Experiment design

### Network, targets, and measured quantities

Every network has one hidden layer and a free output bias:

$$
\widehat f(x)=d+\sum_{j=1}^{204}v_j\sigma(w_jx+b_j).
$$

All four parameter groups train. Training uses function-value mean squared error on 1,024 midpoint samples in $[-1,1]$. The targets are

$$
f_{\mathrm{sine}}(x)=\sin(2\pi x),\qquad
f_{\mathrm{Runge}}(x)=\frac{1}{1+25x^2},
$$

$$
f_{\mathrm{mixture}}(x)=\sin(2\pi x)+\frac12\sin(6\pi x)+\frac14\sin(14\pi x).
$$

The reported error is $\|\widehat f-f\|_2/\|f\|_2$ on 8,192 independent midpoint samples. A 32,768-point disjoint grid verifies the final models. The bandwidth-selection grid has 2,049 midpoint samples. These grid sizes make all four sets disjoint; they do not create unseen target functions.

At each saved geometry, define $\Phi_{ij}=\sigma(w_jx_i+b_j)$ on the training grid. The ideal readout problem is

$$
(v_{\mathrm{LS}},d_{\mathrm{LS}})=\arg\min_{v,d}\|\Phi v+d\mathbf1-y\|_2.
$$

The measured curve uses a truncated-SVD approximation to this problem, which need not attain the unrestricted minimum. The implementation first subtracts each feature's mean and the target mean, solves the centered system using SciPy's SVD least-squares driver with relative cutoff $10^{-13}$, then recovers the unpenalized bias. Actual Adam and solved readouts are saved and evaluated at the same geometry. The solve never updates the network or optimizer. Thus the dashed curve is an observational numerical refit, conventionally called a readout floor; it is not a certified lower bound on held-out error.

### Activations

Every activation is odd and saturates at $\pm1$. Write $K=\sigma'$ for its derivative kernel.

| Arm | Activation | Derivative kernel |
|---|---|---|
| Standard | $\sigma(z)=\tanh z$ | $K(z)=\operatorname{sech}^2z$ |
| Spectral-notch control | $\sigma(z)=\operatorname{erf}(z/\sqrt2)-\sqrt{2/\pi}\,z e^{-z^2/2}$ | $K(z)=\sqrt{2/\pi}\,z^2e^{-z^2/2}$ |
| Localized variational candidate | $\sigma(z)=\int_0^zK(t)\,dt$ | Frozen even cubic spline, supported on $[-4,4]$, with $\int K=2$ |
| Supplementary sinc | $\sigma(z)=2\operatorname{Si}(\pi z)/\pi$ | $K(z)=2\operatorname{sinc}z$, with $\operatorname{sinc}z=\sin(\pi z)/(\pi z)$ |

Under $\widehat K(\omega)=\int K(z)e^{-i\omega z}\,dz$, the notch kernel has

$$
\widehat K(\omega)=2(1-\omega^2)e^{-\omega^2/2}.
$$

It therefore suppresses frequencies near $\omega=\pm1$ in that fixed scale. This is a mechanism to test, not a proof that every finite network using the activation must train badly. Scaling a neuron moves its notch in physical frequency. Notch also has zero slope at the origin, unlike the other activations; Xavier comparisons include this difference.

### What was optimized for the localized activation

Locality is imposed by compact support of $K$, while regularization penalizes roughness. The integrated activation itself is not compactly supported: it becomes constant outside $[-4,4]$. The spline has knots spaced by $0.25$, even symmetry, mass two, and endpoint conditions $K=K'=0$. There are 16 free symmetric coefficients before the mass constraint. Signed lobes are allowed; monotonicity of $\sigma$ is not imposed.

The objective concerns a unit-spaced, infinite translation lattice. For base frequency $\theta\in[-\pi,\pi]$, define the alias frequencies $\omega_k=\theta+2\pi k$. The function-projection error fraction is

$$
E_K(\theta)=
\frac{\sum_{k\ne0}|\widehat K(\omega_k)/\omega_k|^2}
{\sum_k|\widehat K(\omega_k)/\omega_k|^2}.
$$

This formula has a precise square-integrable interpretation despite $\sigma$ having nonzero tails. Use the adjacent-difference feature $\phi(x)=\sigma(x)-\sigma(x-1)$, which has compact support. Differentiation and the Fourier shift rule give

$$
i\omega\widehat\phi(\omega)=(1-e^{-i\omega})\widehat K(\omega),
\qquad
\widehat\phi(\omega)=\frac{1-e^{-i\omega}}{i\omega}\widehat K(\omega).
$$

At all aliases of $\theta$, the numerator factor is the same because $e^{-i(\theta+2\pi k)}=e^{-i\theta}$. In the frequency vector indexed by aliases, a low-band target has only its central component. Orthogonal projection onto the vector of feature amplitudes retains the fraction $|\widehat\phi(\theta)|^2/\sum_k|\widehat\phi(\omega_k)|^2$. Subtracting that fraction from one and canceling the common factor gives $E_K$. The limit at $\theta=0$ is zero; midpoint integration avoids evaluating the removable limit directly.

We minimize

$$
J[K]=\frac{\int_{-\pi}^{\pi}S(\theta)E_K(\theta)\,d\theta}
{\int_{-\pi}^{\pi}S(\theta)\,d\theta}
+10^{-6}\int |K'(x)|^2\,dx,
\qquad
S(\theta)=\left[1+(\theta/0.5)^2\right]^{-2},
$$

with the prior zero outside $[-\pi,\pi]$. The kernel design uses no values or derivatives of the three test targets. The prior, support, spline family, and penalties are design choices, not conclusions of the projection theorem.

To exclude near-dependent derivative translates, impose

$$
g_K(\theta)=\sum_{m\in\mathbb Z}\langle K,K(\cdot-m)\rangle e^{-im\theta}
\ge0.02\|K\|_2^2
\quad\text{for all }\theta.
$$

Compact support makes $g_K$ a finite cosine polynomial. Its continuous extrema are checked by expressing it as a Chebyshev polynomial in $\cos\theta$ and testing derivative roots and endpoints, with a dense-grid backstop. These are float64 numerical checks, not interval-arithmetic certificates.

SLSQP uses analytic gradients and three deterministic starts: a narrow Gaussian, windowed sinc, and broad smooth profile. The lowest feasible converged initial-grid result is refined on 512 frequency points and aliases $-16,\ldots,16$. Checks use up to 2,048 frequencies, aliases through $\pm64$, and increased quadrature orders. This establishes a numerical local solution in a finite spline family, not a global optimum. The constraint bounds derivative-translate conditioning at the design spacing; it does not bound the finite integrated-feature matrix or Adam's parameter Jacobian.

### Constructive QI, bandwidth selection, and Xavier

QI uses 64 interior centers, including both endpoints, so $h=2/63$. Add 70 uniformly spaced halo centers per side, giving 204 neurons. For a dimensionless bandwidth $\lambda$, set $\gamma=\lambda/h$, $w_j=\gamma$, and $b_j=-\gamma c_j$.

The readout is a derivative/cardinal construction, not a least-squares fit to function values. With stencil indices $r,k=-160,\ldots,160$, solve the target-independent system

$$
T_{rk}=\lambda K(\lambda(r-k)),\qquad Tq=h e_0.
$$

Then set

$$
v_j=\sum_{k=-160}^{160}q_k f'(c_j-kh),
\qquad
d=f(-1)-\sum_jv_j\sigma(\gamma(-1-c_j)).
$$

Thus QI receives analytic derivative information, including outside the fitting interval, and a boundary value. It is intentionally a target-informed constructive start; Xavier receives neither. Step zero always shows this construction before Adam or any observational refit can change its readout.

The bandwidth candidates are $0.10,0.125,0.16,0.20,0.25,0.30,0.40,0.50,0.5302,0.625,0.75,0.875,1,1.125,1.25,1.5,2$. Choose one value per activation minimizing the worst construction error across the three targets on the selection grid. For local only, candidates must also retain $\min g/\|K\|^2\ge0.02$ after scaling. The selected values are tanh $0.25$, notch $0.4$, local $1$, and sinc $1$. These apply to QI only; Xavier ignores $\lambda$.

All screening uses float64 and a $10^{-13}$ cardinal cutoff. At the selected tanh point, the existing repository 30-digit constructor avoids cancellation in the finite cardinal computation; its parameters are then transferred to the float64 network. The other constructors stay float64. This asymmetry is explicit: the test includes the mature repository tanh constructor, not an equal-arithmetic-precision study of all constructors. The notch's selected cardinal system retains a condition number around $2\times10^{11}$, so a poor notch construction cannot by itself establish a capacity failure.

Xavier uses gain-one Glorot uniform draws for both layers and zero biases. With width $M=204$, both weight vectors have bounds $\pm\sqrt{6/(M+1)}$. Seeds 0, 1, and 2 use identical draws across activations. QI is deterministic and run once per target and activation. All initially zero-bias activations are odd, so their initial hidden span cannot fit the even Runge target beyond a constant output; trainable hidden biases can break this restriction.

### Adam protocol and recording

All 48 runs use float64 full-batch Adam for 10,000 steps, with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, and no weight decay. The learning rate warms linearly to $0.002$ over 200 steps, then decays by a cosine schedule to $0.000002$. The common schedule is not tuned separately for activation or initialization.

Step zero, steps 1–20, and 85 approximately geometric checkpoints through step 10,000 save geometry plus both readouts. Plots show the current network rather than best-so-far errors. Main trajectories use Xavier seed zero; endpoint bars show the median and individual values of all three Xavier seeds. All arms have the same nominal parameter count. Initial numerical readout ranks at the declared cutoff are tanh 74, notch 73, local 70, and sinc 88, so nominal width is not independent feature count. In the local QI geometry, 134 halo neurons are exactly saturated on the interval; their geometry gradients are zero and their readouts contribute only constants.

**Code & data**

- Code and configuration: `experiments/expD41_activation_lens/` — activation implementations, spline design, training, figures, and saved-run summary.
- Experiment outputs: `results/checkpoint_D_optimizers/expD41_activation_lens/` — frozen `local_design.json`, `bandwidth.json`, configuration/environment, `summary.json`, and `validation.json`.
- Per-run artifacts: `runs/*.json` contain every measured checkpoint; `runs/*.npz` contain corresponding geometries and actual/solved readouts; `runs/*.pt` contain final network and Adam state.
- Figures, with matching PDF files: [main trajectories](figures/trajectories.png), [expanded early steps](figures/trajectories_early.png), [endpoint bars](figures/endpoint_bars.png), [bandwidth sweep](figures/bandwidth_sweep.png), [sinc reference](figures/sinc_reference.png), [cutoff sensitivity](figures/cutoff_sensitivity.png), and [local design](design_diagnostics.png).
- Corrected zero-readout follow-up: [frozen-geometry report](frozen_geometry/expD41_frozen_results.md).
- Verification: `tests/test_expD41_activations.py`, `tests/test_expD41_local.py`, `tests/test_expD41_protocol.py`.
- Reproduction: `.venv/bin/python -m experiments.expD41_activation_lens.local_design`, then `.venv/bin/python experiments/expD41_activation_lens/run.py all`, then `.venv/bin/python experiments/expD41_activation_lens/summarize.py`. Preserve the frozen design and per-run source signatures when reusing this exact data; a new design or changed training source requires a separate output/version. Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` for the recorded CPU threading setup.

## Results

The final measurements do not produce the intended standard / poor / best ordering. Among the three main activations, tanh has the smallest actual QI endpoint error on every target. Local has the smallest median Xavier error on sine and, by a small margin, the mixture; tanh has the smallest median Xavier error on Runge. No Xavier arm fits the mixture accurately under this budget: all four activation medians remain around $0.46$ relative error.

The table gives actual and refitted errors at step 10,000. QI has one deterministic run; Xavier entries are separate medians of the three corresponding metrics, not a single median-seed model. Sinc is supplementary.

| Target | Activation | QI actual | QI refit | Xavier actual, median | Xavier refit, median |
|---|---|---:|---:|---:|---:|
| Sine | Tanh | $1.54\times10^{-05}$ | $2.72\times10^{-11}$ | $0.0484$ | $3.86\times10^{-05}$ |
| Sine | Notch | $5.08\times10^{-05}$ | $2.88\times10^{-14}$ | $0.0136$ | $2.76\times10^{-10}$ |
| Sine | Localized | $3.97\times10^{-04}$ | $3.96\times10^{-04}$ | $0.0104$ | $3.01\times10^{-04}$ |
| Sine | Sinc (reference) | $1.62\times10^{-06}$ | $2.01\times10^{-14}$ | $6.43\times10^{-03}$ | $4.40\times10^{-07}$ |
| Runge | Tanh | $5.44\times10^{-06}$ | $8.65\times10^{-12}$ | $0.0119$ | $4.82\times10^{-03}$ |
| Runge | Notch | $6.15\times10^{-05}$ | $3.25\times10^{-09}$ | $0.0237$ | $0.0166$ |
| Runge | Localized | $2.14\times10^{-04}$ | $2.14\times10^{-04}$ | $0.0337$ | $9.82\times10^{-06}$ |
| Runge | Sinc (reference) | $4.68\times10^{-05}$ | $1.04\times10^{-10}$ | $0.0511$ | $0.0393$ |
| Sine mixture | Tanh | $4.94\times10^{-05}$ | $9.90\times10^{-11}$ | $0.464$ | $0.227$ |
| Sine mixture | Notch | $2.11\times10^{-04}$ | $3.23\times10^{-13}$ | $0.458$ | $0.206$ |
| Sine mixture | Localized | $5.27\times10^{-04}$ | $5.23\times10^{-04}$ | $0.455$ | $0.144$ |
| Sine mixture | Sinc (reference) | $4.60\times10^{-07}$ | $5.53\times10^{-15}$ | $0.458$ | $0.359$ |

The separation between actual and solved readouts is particularly informative for Runge. Local/Xavier ends at median actual error $0.0337$, while refitting its final geometry gives $9.82\times10^{-6}$ at the declared cutoff. Tanh/Xavier has the better actual error, $0.0119$, but the worse refit, $0.00482$. Thus a comparison of actual training error alone would miss the local activation's much better final solved geometry in this case. This observation does not identify why Adam fails to use that geometry fully.

The local QI runs tell a different story: actual and refitted endpoint errors are close on all three targets. For sine they are $3.97\times10^{-4}$ and $3.96\times10^{-4}$. The readout gap has nearly closed, yet the available fit is less accurate than the tanh and notch QI geometries. The three local QI floors are unchanged when the readout cutoff varies from $10^{-14}$ to $10^{-12}$, so their approximately $10^{-4}$ plateaus are not explained by this particular cutoff range.

The notch geometry supports a very accurate sine approximation both initially and after training, despite a much worse derivative-cardinal construction and Adam readout. It also has the best median Xavier sine refit among these four activations. Calling this activation universally "awful" is contradicted by the observed fits. Conversely, its Xavier Runge results are worse than tanh's. Neither its Fourier notch nor its zero central slope alone predicts the ordering of all these outcomes. This is not a target deliberately placed at the notch: the selected QI scale is $\gamma=12.6$, so its physical notch is at angular frequency $12.6$, whereas the sinusoidal target frequencies are $2\pi$, $6\pi$, and $14\pi$. Bandwidth selection can avoid a bad frequency alignment, and subsequent slope training can change it further.

### The construction at step zero

Because the QI initializer is part of the experiment, its initial error must remain visible rather than being replaced with a function-value solve:

| Target | Tanh | Notch | Localized | Sinc reference |
|---|---:|---:|---:|---:|
| Sine | $2.63\times10^{-15}$ | $0.254$ | $0.028$ | $2.77\times10^{-05}$ |
| Runge | $1.35\times10^{-11}$ | $2.52\times10^{-04}$ | $0.0279$ | $7.15\times10^{-08}$ |
| Sine mixture | $2.37\times10^{-10}$ | $0.222$ | $0.0247$ | $4.55\times10^{-04}$ |

The shared Adam schedule worsens all three tanh constructions substantially. The expanded early-step figure shows the departure from the accurate start. This demonstrates a failure of this training schedule to preserve these constructions, not a proof that every choice of Adam settings must do so. Notch and local constructions improve by the final step; their sizable initial errors also show that extending the tanh derivative-cardinal formula to a new kernel is not automatically a high-accuracy construction.

For supplementary sinc, QI actual final errors are $1.62\times10^{-6}$ on sine, $4.68\times10^{-5}$ on Runge, and $4.60\times10^{-7}$ on the mixture. It is therefore better than the local candidate on all three QI endpoints here. Its Xavier Runge error is nevertheless worse than the other three activations. Infinite-lattice optimality does not imply superiority under both initializations in this finite Adam experiment.

### Design and numerical validation

The selected compact spline has objective $J=3.6704\times10^{-5}$, comprising weighted projection-error fraction $3.3713\times10^{-5}$ and roughness penalty $2.9908\times10^{-6}$. These are objective components, not the empirical relative $L_2$ errors in the tables. Its derivative Gram minimum divided by kernel energy is $0.0200075$, just above the required $0.02$, and its maximum/minimum ratio is about 171. Increasing the frequency grid from 512 to 1,024 changes the objective by $0.042\%$; a 2,048-point check is also saved. Expanding the alias cutoff from 16 to 64 at matched frequency resolution changes the objective by approximately $1.6\times10^{-14}$. The selected design is a validated feasible local solution, with no claim of a globally best activation.

The activation and protocol suite passed 34 focused checks, including analytical versus automatic gradients, the derivative construction, and an exact comparison showing that an observational solve leaves the next Adam update unchanged. Local checks were rerun after final design refinement. Every saved trajectory has the expected step sequence, finite geometry/readout arrays, a final optimizer artifact, and a matching configuration/source fingerprint.

Reevaluating all final saved readouts on 32,768 disjoint points changes actual errors by at most $0.019\%$ and refitted errors by at most $0.44\%$. This shows no substantial discrepancy at the tested resolutions. It does not remove cutoff dependence: for example, the tanh/Xavier seed-zero sine refit varies from $1.25\times10^{-5}$ to $4.98\times10^{-4}$ over the tested cutoffs, and the sinc/QI sine refit varies from approximately $1.8\times10^{-15}$ to $2.6\times10^{-13}$. Exact near-floor rankings should therefore not be interpreted beyond the specified numerical solve.

### Figures

- **Main trajectories:** 3×3, target rows and tanh/notch/local columns; linear Adam-step axis and a shared logarithmic relative-error axis. Red is QI, blue is Xavier seed zero; solid is the current Adam readout, dashed is the observational least-squares readout. Step-zero markers preserve the initial construction.
- **Expanded early trajectories:** the same data and colors with a symmetric-log step axis. This makes departure from the QI start visible without replacing the requested linear-step figure.
- **Endpoint bars:** one panel per target; each activation has four bars with the same color meanings. Hatched bars are refits, plain bars are actual trained models. Xavier bar tops are medians over three seeds; dots retain the individual results. Lower tops mean smaller error; bar area has no interpretation on the log axis.
- **Bandwidth sweep:** four activation panels; target colors, solid construction errors, and dashed observational refits. The selected value is marked, and local candidates that violate its derivative-Gram requirement are marked separately. The plot shows float64 screening; the final tanh construction uses the stated 30-digit refinement.
- **Sinc reference:** one panel per target using the same four curves and error limits as the main figure. Its idealized infinite-grid role is distinct from finite-network performance.
- **Local design diagnostics:** kernel, integrated activation, spectral projection-error fraction, and derivative Gram symbol. These display what was optimized and which stability condition was checked; they are not Adam convergence diagnostics.
- **Readout cutoff sensitivity:** one panel per target; the final dense-grid refit error is reevaluated at three cutoffs. Activation colors are shared across panels; solid lines are QI and dashed lines are medians across Xavier seeds. The main experiment uses the middle cutoff, $10^{-13}$.

## Additional details

The theory concerns the best function projection in a fixed infinite translation space, while the experiment measures a particular constructive initializer and finite-parameter Adam optimization. These are three different questions. In particular, a derivative-cardinal interpolant need not achieve the same error as the best function-value projection for a newly designed kernel.

Sam's interpretation of the local result is useful: this activation nearly attained its available fit in all three constructed-start runs, despite the higher approximation floor. Its actual/floor ratios at the endpoint are 1.0042, 1.0004, and 1.0086 for sine, Runge, and the mixture. This is evidence of successful readout-gap closure in those runs, not simply an activation failure. It does not yet show equally rapid convergence from zero readout or to the same absolute accuracy as the other kernels.

The local objective also makes a specific tradeoff that the label "best" obscured. It minimizes an arithmetic average of squared projection error across frequencies: reducing larger high-frequency errors can outweigh making tiny low-frequency errors worse. The selected kernel improves the declared objective relative to every recorded feasible starting point, but is not uniformly better at every frequency. Its derivative Gram condition ratio is about 171, versus about 2.47 for the feasible windowed-sinc starting profile: the constraint prevents excessive dependence, while the objective does not minimize the condition number. Neither this tradeoff nor the chosen spectral prior proves an Adam ranking.

Finite halos affect sinc differently from compactly supported kernels. Zero-bias Xavier places all initial transition centers at zero, and shared gain one does not equalize nonlinear response scales. A shared optimizer schedule makes the experiment reproducible but cannot establish each activation's best attainable training outcome. The target functions were used in bandwidth selection, so the results do not establish transfer to unseen functions.

Near floating-point precision, an observational refit can have a slightly larger measured error than the current readout because of cutoff and rounding. Re-solving at three cutoffs and checking saved coefficients on a denser grid are included to distinguish this effect from a generalization failure.

## Conclusions

Under this specified constructor, bandwidth selection, and Adam schedule, local variational design does not supply a uniformly superior trained activation, and the spectral-notch activation is not uniformly inferior. The measured differences between constructive, solved-readout, and actual training errors show that those three objectives must remain separate; a causal theory connecting them remains open.

## Open questions

- Which objective should be optimized to improve Adam: fixed-space projection error, conditioning of the integrated readout, or the full parameter Jacobian? This experiment distinguishes them but does not isolate their causal roles.
- Does a construction that explicitly enforces low-frequency reproduction improve the localized QI start without using a function-value least-squares initializer?
- Does any advantage survive bandwidth choices frozen on separate target functions and a matched optimizer-tuning budget?

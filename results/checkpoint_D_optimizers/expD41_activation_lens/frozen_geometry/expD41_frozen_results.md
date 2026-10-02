# expD41 follow-up — Adam readout learning on frozen QI geometry

Status: measurements complete; interpretation pending Sam. Codex version, 2026-09-29.

## TL;DR

- This follow-up uses Sam's clarified QI initialization: the QI centers and widths, with all readout weights and output bias zero. The previous batch used derivative-constructed readouts and is now labeled explicitly as such.
- On frozen geometry, zero-start Adam with the localized activation has lower final error than tanh and notch on Runge and the sine mixture, but higher error on sine. Thus the candidate has a useful advantage under some conditions, even though its approximation floors are higher.
- It does not reproduce the earlier near-floor attainment in every case. Local's zero-start actual/floor ratios are 1.80 on sine, 1.41 on Runge, and 55.6 on the mixture. The constructed-start, moving-geometry runs had ratios below 1.01 on all three.
- All 48 readout-only runs completed 10,000 steps. Geometry and features remain unchanged, and dense independent-grid checks preserve the observations. These results distinguish approximation accuracy from optimizer attainment; they do not identify one universally best activation.

## Question / hypothesis

Sam observed that the variational activation nearly attained its available readout fit in every earlier constructed-start run, despite a higher approximation floor. Does Adam also learn effectively from zero readout when its hidden geometry is held fixed? This removes geometry movement and the constructed readout's head start, while preserving the previously selected kernels and bandwidths.

## Experiment design

For each activation and target, load the exact hidden parameters from the saved QI step-zero network of the parent experiment. The hidden feature matrix on the training samples is

$$
\Phi_{ij}=\sigma(w_jx_i+b_j),\qquad
\widehat y=\Phi v+d\mathbf1.
$$

The vectors $w,b$ are fixed. Only the readout $v$ and output bias $d$ are optimizer parameters. Feature values are cached using the same torch activation implementations as the parent networks; each update evaluates an explicit residual and its gradient. Training does not use normal equations or injected solved coefficients. This cached computation was checked against the full neural network with frozen hidden parameters for all four activations, including agreement of three consecutive Adam updates.

Both starts use identical QI hidden geometry:

- **Zero readout:** $v=0$, $d=0$. Every relative-error trajectory begins at exactly one.
- **Random readout control:** reuse the parent experiment's gain-one Xavier output-weight draws, seeds 0, 1, and 2, with $d=0$. Hidden weights and centers remain QI; this is not a full Xavier network.

No new bandwidth tuning is performed. The selected dimensionless bandwidths remain tanh $\lambda=0.25$, notch $0.4$, local $1$, and sinc $1$, with 64 interior centers, $h=2/63$, and 70 halo centers per side. These are the previously selected QI geometries, not a proof of globally optimal finite geometry.

Targets are $\sin(2\pi x)$, $1/(1+25x^2)$, and $\sin(2\pi x)+\frac12\sin(6\pi x)+\frac14\sin(14\pi x)$ on $[-1,1]$. Adam, float64 arithmetic, 10,000 steps, learning-rate schedule, sample grids, and recording times match the parent experiment: 1,024 training points, 8,192 independent evaluation points, 200-step warmup to $0.002$, then cosine decay to $0.000002$, and default Adam moments/epsilon with no weight decay.

The reported error is $e=\|\widehat f-f\|_2/\|f\|_2$ on the independent grid. A free-bias, centered least-squares refit with relative singular-value cutoff $10^{-13}$ is computed once per fixed geometry. It is shared by both readout starts and remains horizontal throughout the plots. Its coefficients never enter Adam. The computed refit is an observational numerical reference, not a certified lower bound on evaluation error.

The frozen protocol's explicit trainable-parameter metadata governs this follow-up; inherited parent settings describe the original experiment. Code verifies unchanged geometry and unchanged cached training features at every run's completion. Both readout starts and all seeds use the same fixed feature values.

**Code & data**

- Implementation: `experiments/expD41_activation_lens/frozen.py`; figures: `experiments/expD41_activation_lens/plot_frozen.py`.
- Protocol checks: `tests/test_expD41_frozen.py`.
- Data in this folder: `config.json`, `bandwidth.json`, `environment.json`, `runs/*.json`, `runs/*.npz`, `runs/*.pt`, `summary.json`, and `validation.json`.
- Figures, each also available as PDF: [main trajectories](figures/frozen_trajectories.png), [early steps](figures/frozen_early.png), [endpoint bars](figures/frozen_endpoints.png), and [sinc reference](figures/frozen_sinc_reference.png).
- Parent methods and initial constructed-start data: [parent report](../expD41_results.md).
- Reproduce using `.venv/bin/python experiments/expD41_activation_lens/frozen.py all` from the repository root. The existing source and parent-data fingerprints must match when reusing saved results.

## Results

At 10,000 steps, zero-readout local beats tanh and notch on Runge and the mixture, while tanh and notch have smaller sine errors. This is a real difference in attained accuracy under a shared budget, rather than merely a comparison of distances to different floors. Supplementary sinc performs best among the four on Runge and the mixture from zero, but is worse on sine.

| Target | Activation | Zero-start Adam | Fixed-geometry refit | Random-readout Adam, median |
|---|---|---:|---:|---:|
| Sine | Tanh | $3.99\times10^{-4}$ | $8.90\times10^{-15}$ | $2.55\times10^{-3}$ |
| Sine | Notch | $3.56\times10^{-4}$ | $1.83\times10^{-14}$ | $3.77\times10^{-3}$ |
| Sine | Localized | $7.12\times10^{-4}$ | $3.95\times10^{-4}$ | $6.28\times10^{-3}$ |
| Sine | Sinc (reference) | $9.98\times10^{-4}$ | $1.19\times10^{-14}$ | $2.52\times10^{-3}$ |
| Runge | Tanh | $1.42\times10^{-3}$ | $9.41\times10^{-12}$ | $7.90\times10^{-3}$ |
| Runge | Notch | $2.03\times10^{-3}$ | $3.25\times10^{-9}$ | $6.93\times10^{-3}$ |
| Runge | Localized | $3.03\times10^{-4}$ | $2.15\times10^{-4}$ | $0.0103$ |
| Runge | Sinc (reference) | $1.65\times10^{-5}$ | $1.11\times10^{-10}$ | $3.59\times10^{-3}$ |
| Sine mixture | Tanh | $0.214$ | $6.24\times10^{-11}$ | $0.214$ |
| Sine mixture | Notch | $0.203$ | $4.49\times10^{-13}$ | $0.206$ |
| Sine mixture | Localized | $0.0294$ | $5.28\times10^{-4}$ | $0.0417$ |
| Sine mixture | Sinc (reference) | $3.86\times10^{-3}$ | $1.29\times10^{-14}$ | $5.32\times10^{-3}$ |

The localized kernel's approximation space is less accurate, yet Adam reaches it more closely. On Runge, local's actual error is $3.03\times10^{-4}$ against a $2.15\times10^{-4}$ reference; tanh's actual error is $1.42\times10^{-3}$ against a $9.41\times10^{-12}$ reference. Local therefore has both lower actual error and much less remaining readout error in this comparison.

The mixture reveals the limit of that positive interpretation. Local's $0.0294$ actual error is about seven times lower than tanh's $0.214$, but remains about 56 times its own $5.28\times10^{-4}$ refit. In the earlier constructed-start moving-geometry run it ended within 1% of the refit. Starting point and geometry movement both differ between those experiments; this follow-up does not isolate which of those two changes accounts for that gap.

The zero-start endpoint is below the median random-start error in 11 of the 12 activation/target cases. The exception is tanh on the mixture, where the random median is slightly smaller. Since hidden geometry is identical, these differences arise from readout initialization under this finite training schedule. They do not establish a general optimal initialization theorem.

All saved geometries and feature hashes remain unchanged. The two focused frozen-protocol tests pass. Reconstructing final saved readouts on 32,768 disjoint points changes actual errors by at most $0.010\%$ and refit errors by at most $0.315\%$. All geometry, readout, optimizer, and source artifacts are saved; least-squares cutoff checks at $10^{-14}$, $10^{-13}$, and $10^{-12}$ are retained in validation data.

### Figures

- **Frozen trajectories:** 3×3 target/activation panels with linear steps and a shared log-error axis. Red starts with zero readout; blue randomizes only the readout and shows seed zero. The black dashed reference is one fixed-geometry least-squares solve, shared by both starts. This is the direct test of readout learning at fixed QI geometry.
- **Expanded early steps:** the same trajectories with a symmetric-log step axis, making the initial descent visible.
- **Endpoint bars:** three target panels compare each main activation's zero-start endpoint, median random-start endpoint with all three seeds, and shared least-squares reference.
- **Sinc reference:** the same three-line comparison for the supplementary idealized activation.

## Additional details

The useful theoretical distinction is between the best approximation available in a feature span and an optimizer's progress toward that approximation. Let $A=[\Phi,\mathbf1]$ include the output bias, let $p=P_Ay$ be the exact orthogonal projection onto its column space, and let $\theta=(v,d)$ denote the current readout. Then

$$
y-A\theta=(y-p)+(p-A\theta).
$$

The first term is orthogonal to the column space of $A$, while the second belongs to that space. Their inner product is therefore zero. Expanding the squared norm gives

$$
\frac{\|y-A\theta\|^2}{\|y\|^2}
=\underbrace{\frac{\|y-p\|^2}{\|y\|^2}}_{\text{approximation error squared}}
+\underbrace{\frac{\|p-A\theta\|^2}{\|y\|^2}}_{\text{remaining readout error squared}}.
$$

This exact identity concerns an exact projection and the same fitting norm; the truncated numerical refit and separate evaluation grid are approximations to that setting. It explains why proximity to a higher floor and a smaller absolute error are different performance claims.

Readout optimization also has a precise geometric structure. For $L(\theta)=\|A\theta-y\|^2/(2m)$, where $m$ is the sample count, the Hessian is $H=A^TA/m$. If $s_i$ is a singular value of $A$, the corresponding curvature is $s_i^2/m$. In ordinary gradient descent with constant step size $\eta$, the coefficient error along that direction is multiplied by $1-\eta s_i^2/m$ each step. Very small singular values produce slowly corrected directions when the step is limited by larger curvatures. Adam changes this dynamics through its evolving coordinate scaling, so this formula is motivation for examining learning difficulty, not an exact convergence prediction for the plotted runs.

Sam's observation about the previous batch is correct: all three local constructed-start runs end within 1% of their numerical readout references. That is successful readout-gap closure. The present data qualify its scope: the property is not automatic from zero readout, even with the same initial hidden geometry. A more accurate lens can be harder to exploit, but a higher floor by itself also makes a near-floor criterion easier to satisfy.

The design objective was a weighted average of squared projection error under a prescribed spectrum, with compact support, roughness regularization, and a derivative-Gram lower bound. It neither minimized readout learning time nor demanded uniformly best accuracy at every frequency. The spectral-notch control likewise was not guaranteed to fail on these targets: the selected notch frequency does not coincide with the sinusoidal target frequencies. The experiment therefore motivates separating attainable approximation, time to a common absolute tolerance, and remaining readout error rather than assigning a single "best/worst" label.

## Conclusions

The localized candidate provides a useful fixed-geometry, zero-readout learning advantage on two of these targets under the tested Adam schedule, while retaining a higher approximation floor and failing to reach that floor on the mixture. The evidence supports investigating a tradeoff between approximation accuracy and optimizer access; it does not establish a uniform learning guarantee.

## Open questions

- At equal approximation error, does the localized design consistently reduce the readout-learning budget?
- How much of the earlier near-floor attainment came from the constructed readout, and how much from geometry adaptation?
- Which design objective best balances projection accuracy with convergence on the full integrated-feature readout problem?

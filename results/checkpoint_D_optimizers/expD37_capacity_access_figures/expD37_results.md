# Measured versions of the frozen-readout schematics

## 1. Question

Replace the two schematic figures with actual optimizer results and measurements from the revised *Capacity versus accessibility in frozen tanh networks* note. Preserve the two-panel and three-panel layouts, but define the quantities and allow the data to determine the curves.

## 2. Figures

The main target is the note's mixed sine, \(f(x)=[\sin(2\pi x)+0.1\sin(20\pi x)]/\sqrt{0.505}\).

- [Optimizer comparison: mixed sine](optimizer_comparison.png).
- [Polynomial tails, damping, and time bounds: mixed sine](theorem_diagnostics.png).
- Companion optimizer comparisons: [sine](companions/sine_optimizer.png), [quadratic](companions/quadratic_optimizer.png), [Runge](companions/runge_optimizer.png).
- [Theorem diagnostics: Runge](companions/runge_theorem.png).

These are PNGs. Existing D36 figures and trajectories were preserved. The directory `data/` holds numerical results, checks, provenance, and resumable states for the two newly trained coordinate maps.

## 3. Shared setup and what was reused

Use the existing D36 geometry: \(N=256\), spacing \(h=1/128\), 257 centers on the closed interval plus 24 halo centers on each side, hence \(W=305\) tanh neurons and 306 readout parameters including the output bias. There are 1,021 training midpoints and 4,093 independent evaluation midpoints on \([-1,1]\). Every neuron shares the same positive slope within a run. Geometry is frozen and physical readout coefficients start at zero.

The optimizer figure uses all nine saved lambda values, through \(\lambda=1\); \(\gamma=\lambda/h\). Thus the reference \(\lambda=0.25\) is **gamma 32**, not the schematic's 64. The theorem diagnostics use exact gamma values 4, 8, 16, 32, 64, 128, with the same centers and training samples.

D36 supplied raw coefficients, square-root coefficient scaling, and unscaled adjacent differences. Two additional maps were trained **sequentially**, each for 20,000 steps: coefficient scaling by alpha, and adjacent differences scaled by square roots of cumulative alpha. Each new map contributes 72 runs: two optimizers × nine lambdas × four targets. No existing training was repeated.

## 4. Read the optimizer figure

Each point is the **final** evaluation relative L2 error at step 20,000, not the best checkpoint. Both panels share axes. Connecting segments only join measured gamma values; no fitted or schematic curve was used.

Use the note's notation: \(A\) contains the tanh features and constant feature, divided by \(\sqrt m\). Physical coefficients are \(c=M\theta\), so the matrix seen by the optimizer is \(B=AM\). The five legend entries specify \(M\), rather than the labels “collective” and “individual.”

| Figure label | Physical readout map |
|---|---|
| Direct weights | \(c=\theta\) |
| Weights scaled by \(\sqrt\alpha\) | \(c_j=\sqrt{\alpha_j}\theta_j\), including output bias |
| Weights scaled by \(\alpha\) | \(c_j=\alpha_j\theta_j\), including output bias |
| Adjacent differences, unscaled | Hidden features become \(\phi_j-\phi_{j+1}\), with the final \(\phi_W\) anchor and constant retained |
| Adjacent differences scaled by \(\sqrt s\) | Multiply difference feature \(j\) by \(\sqrt{s_j}\), where \(s_j=\sum_{\ell\le j}\alpha_\ell\) over hidden neurons; output bias gets \(\sqrt{\alpha_0}\) |

Here \(\alpha\) is the PR's fixed coefficient-allowance vector, computed at reference \(\lambda=0.25\) and **held fixed across gamma**. It is not a learned quantity. Ordinary interior alpha is approximately 0.01944, the largest halo alpha is 2.09254, and output-bias alpha is 16.77257. The last map scales cumulative coordinates **before** differencing, as specified in the note. All five maps are invertible, so their exact feature spans agree at a given gamma.

GD uses \(\eta=1/\|B\|_2^2\), recalculated for each fixed matrix and then held constant. Adam uses the existing native learning rate 0.001, betas 0.9/0.999, epsilon \(10^{-8}\), and no schedule. Adam was not separately retuned for the new maps. Consequently these compare the stated coordinate choices and rate policies, not the best achievable optimizer under each choice. Coefficient scaling includes bias and halo effects.

The numerical least-squares references remain in the saved data and D36 plots; they are not an additional line in these schematic-shaped optimizer panels. Final Adam values may include oscillations: the last-1,000-step minimum, median, and maximum are saved beside the final values in `data/optimizer_figure_values.npz`.

## 5. Read the three theorem panels

These diagnostics use the **direct readout**, \(M=I\), and the training grid. They do not mix the optimizer maps from the first figure. Fix a requested relative training error \(\epsilon=10^{-3}\).

**Left: required correction versus readout access.** The projection \(P_k\) extracts the best sampled polynomial of degree at most \(k\); \(Q_k=I-P_k\) removes it. Define \(D_k=\|Q_ky\|_2/\|y\|_2\) and the unit residual pattern \(q_k=Q_ky/\|Q_ky\|_2\). Then:

- Black is \([D_k-\epsilon]_+^2\): the squared amount of polynomial-tail correction required by the tolerance.
- Orange and green are \(\mu_k/\|B\|_2^2\), where \(\mu_k=\|B^Tq_k\|_2^2\): the squared readout gradient for that unit correction, normalized by the strongest curvature.
- Dashed red is the note's upper bound \(\mathcal B_k/\|B\|_2^2\), with \(\mathcal B_k=W e_k(4)^2\). The full Eq. (5) prefactor is included.

The shaded region starts where no tail correction is required at this tolerance. The black curve then equals zero and cannot appear on a log axis. Small access values that are unresolved in FP64 are omitted, not replaced by zero or a numerical plateau. Polynomial degree is not Fourier frequency.

**Middle: one damped Gauss–Newton step.** Every curve begins from the **same** unit residual \(q_{16}=Q_{16}y/\|Q_{16}y\|_2\), independent of gamma and damping. For each relative damping \(\rho=\zeta/\|B\|_2^2\), solve one damped readout step spectrally and plot the remaining norm

\[
\mathcal R(q_{16};\zeta)=\|\zeta(BB^T+\zeta I)^{-1}q_{16}\|_2.
\]

Zero means that step removed the probe; one means it left it unchanged. A curve shifted right means the same correction can be removed with larger damping. This is a fresh one-step calculation at each x value, not a training trajectory or repeated GN solve. The full-target versions and cutoff-dependent limiting remainders are also saved in the data, because a polynomial tail need not be entirely in the feature span.

**Right: how much delay the note can certify.** Start from zero coefficients and let \(\dot\theta=-B^T(B\theta-y)\). The blue curve is the SVD-based solution for the first time \(T_\epsilon\) at which \(\|B\theta-y\|_2/\|y\|_2\le\epsilon\), multiplied by \(\|B\|_2^2\). Orange is Eq. (4), using measured access; green replaces that access with the slope-only upper bound from Eq. (6):

\[
T_\epsilon\|B\|_2^2\ \ge\ C_\epsilon\max_k\frac{[D_k-\epsilon]_+^2}{\mu_k/\|B\|_2^2}
\ \ge\ C_\epsilon\max_k\frac{[D_k-\epsilon]_+^2}{\mathcal B_k/\|B\|_2^2},
\qquad C_\epsilon=\frac{\log(1/\epsilon)}{(1-\epsilon)^2}.
\]

The plotted maxima use degrees 0–96; the measured-access maximum excludes unresolved points. These are weaker than maximizing over all allowable degrees. **Normalized flow time is not a measured GD iteration count and is not an Adam guarantee.** No run was trained for the enormous times shown in blue.

## 6. Measured outcomes

For mixed sine, the normalized time comparison is:

| Gamma | Spectral flow prediction | Bound from measured access | Bound from slope cap |
|---:|---:|---:|---:|
| 4 | \(1.363\times10^{23}\) | \(4.900\times10^{19}\) | \(2.655\times10^{17}\) |
| 8 | \(1.740\times10^{13}\) | \(5.209\times10^9\) | \(1.672\times10^7\) |
| 16 | \(1.971\times10^8\) | \(1.560\times10^5\) | \(1.524\times10^2\) |
| 32 | \(3.213\times10^5\) | \(1.756\times10^3\) | 3.329 |
| 64 | \(2.111\times10^5\) | 309.6 | 3.333 |
| 128 | \(1.291\times10^5\) | 180.9 | 3.334 |

The bounded-slope certificate is substantial at small gamma for this target and tolerance, while leaving a large additional delay unexplained. At large gamma it becomes weak. This is a frozen-readout result; it does not identify a mechanism preventing gamma from learning.

The optimizer curves do not follow the schematic's consistent ranking. For example, at gamma 32 on mixed sine, final GD errors are 0.0500 (direct), 0.0916 (square-root alpha), 0.7709 (alpha), 0.00287 (unscaled differences), and 0.00540 (scaled differences). Final Adam errors at the same point are approximately \(5.06\times10^{-4}\), \(5.70\times10^{-4}\), 0.0178, \(1.07\times10^{-5}\), and \(9.79\times10^{-6}\). These are the specified fixed-budget comparisons.

## 7. Validation and numerical scope

Five focused tests cover the note's sharp single-mode equality, a direct damped solve with an out-of-span residual, polynomial orthogonality and the quadratic zero-tail control, arbitrary-bias structural bounds, and an unattainable residual.

The discrete-GD spectral formula was separately compared against actual saved raw-GD training for all nine lambdas and four targets at steps 1, 100, and 20,000. Maximum absolute discrepancy in relative error was \(6.1\times10^{-13}\). This validates the spectral calculation against ordinary GD at the tested budgets, not its attainability under arbitrarily long finite-precision training.

The theorem diagnostics factor \(B\) directly, never a rounded Gram matrix. Two independent LAPACK SVD drivers agree on the predicted hitting times within \(3.7\times10^{-7}\) relative error. Relative singular-value cutoffs \(10^{-12},10^{-13},10^{-14},10^{-15}\) were swept; faint bands show their effect. At gamma 4, the mixed-sine time changes about 1.25% at the coarsest cutoff, with much closer agreement at the finer cutoffs. Removed singular modes are not declared an exact nullspace.

For the strongest mixed-target measured-access certificate, gamma 4 and degree 63, the complete target, polynomial projection, and tanh features were rebuilt at 80 and 120 digits. They give \(\mu_{63}=3.39311831781977587686\ldots\times10^{-21}\), agreeing beyond 44 printed digits. The FP64 value differs by \(3.13\times10^{-7}\) relatively. This high-precision audit covers the directional access, **not** a full high-precision SVD of every matrix. Source-data hashes and detailed checks are under `data/`.

## 8. Reproduce or extend

Run from the repository root:

```bash
.venv/bin/python experiments/expD37_capacity_access_figures/run.py train
.venv/bin/python experiments/expD37_capacity_access_figures/run.py diagnostics
.venv/bin/python experiments/expD37_capacity_access_figures/precision_check.py
.venv/bin/python experiments/expD37_capacity_access_figures/run.py plot
.venv/bin/python -m pytest -q experiments/expD37_capacity_access_figures/test_diagnostics.py
```

Training resumes the two new maps from saved state. Plotting reads saved results without retraining. The original D36 trajectories remain the source for the other three maps. Configuration is in `experiments/expD37_capacity_access_figures/config.yaml`.

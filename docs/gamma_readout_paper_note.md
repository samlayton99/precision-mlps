# Gamma controls readout learning: a paper statement and three-panel figure

Gamma controls access to fine-scale corrections through an explicit smoothing
of the feature kernel. Following that smoothing through the actual finite
geometry predicts how long the readout needs to learn a target, even when the
target is attainable. In the common-slope experiment, the predicted gamma-8
versus gamma-64 delay is 985.70–987.49 times; the executed delay is 986.59 times.
The figure connects an observed slope-scale gap, this quantitative prediction,
and recovery when slopes grow with width. These are retrospective training
results for a prescribed geometry and target, rather than a sharp guarantee
over every dictionary satisfying a slope cap.

**Table 1. Notation and the roles of the three panels.** Residuals are relative
Euclidean norms on training samples; time counts optimizer updates.

| Symbol or term | Meaning |
|---|---|
| $W,N,h$ | Hidden width, number of cells, and spacing $h=2/N$; here $W=N+2\lceil\sqrt N\rceil+1$. |
| $a_j,\gamma$ | Learned slope of neuron $j$ in panel A; common fixed slope in panels B/C. |
| $J_\gamma,K_\gamma$ | Empirically normalized feature matrix and readout kernel $J_\gamma J_\gamma^T$. |
| $L_\gamma$ | Largest kernel eigenvalue $\|K_\gamma\|$. |
| $M_\gamma(\omega)$ | Explicit attenuation of physical frequency $\omega$. |
| $\eta_\gamma,n_\epsilon$ | Prescribed GD step and first update reaching residual $\epsilon$. |
| $E_n,E_\infty$ | Relative training residual after $n$ updates and its limiting floor. |
| Certified interval | Necessary and sufficient times whose endpoint claims have independent interval-arithmetic checks. |
| Panels A / B / C | Joint-training observation / readout timing validation / common-slope intervention across widths. |

## Statement for the main paper

> **Gamma controls both representation and acquisition.** For a fixed geometry,
> a tanh dictionary with common slope $\gamma$ is a smoothed step dictionary,
> with explicit filter $M_\gamma(\omega)=z/\sinh z$, where
> $z=\pi|\omega|/(2\gamma)$. The same filtered kernel determines the
> unlearnable target component and the acquisition rate of each learnable
> component. Retaining the finite geometry gives quantitative acquisition-time
> bounds that closely match executed gradient descent. Thus small gamma can
> delay learning an attainable target, beyond any limitation on approximation.

**Theorem (gamma-dependent readout acquisition).** Fix samples $x_i$, centers
$c_j$, a common slope $\gamma>0$, and a nonzero sampled target
$y_i=f(x_i)/\sqrt m$. Let

$$
(J_\gamma)_{i0}=\frac1{\sqrt m},\qquad
(J_\gamma)_{ij}=\frac{\tanh(\gamma(x_i-c_j))}{\sqrt m},\qquad
K_\gamma=J_\gamma J_\gamma^T.
$$

Train only the raw readout coefficients on
$\frac12\|J_\gamma\theta-y\|^2$, starting at $\theta_0=0$, with
$0<\eta_\gamma\le1/L_\gamma$. Write
$E_n=\|y-J_\gamma\theta_n\|/\|y\|$. Define the fixed step-feature kernel
$k_\infty(s,t)=1+\sum_j\operatorname{sign}(s-c_j)\operatorname{sign}(t-c_j)$.
Then the continuous feature kernel before sampling satisfies

$$
\boxed{
k_\gamma=S_\gamma^{(1)}S_\gamma^{(2)}k_\infty,
\qquad \widehat\rho_\gamma(\omega)=M_\gamma(\omega)
=\frac{\pi|\omega|/(2\gamma)}{\sinh(\pi|\omega|/(2\gamma))},
}
\tag{1}
$$

where $S_\gamma$ is whole-line convolution with
$\rho_\gamma(t)=\frac\gamma2\operatorname{sech}^2(\gamma t)$, and
$M_\gamma(0)=1$. The empirical kernel is
$(K_\gamma)_{ii'}=k_\gamma(x_i,x_{i'})/m$.
For its positive eigenvalues and orthonormal eigenvectors $(\lambda_i,u_i)$, put
$p_i=|u_i^Ty|^2/\|y\|^2$ and
$E_\infty^2=\|P_{\ker K_\gamma}y\|^2/\|y\|^2$. Its exact GD curve is

$$
\boxed{
E_n^2=E_\infty^2+
\sum_{\lambda_i>0}p_i(1-\eta_\gamma\lambda_i)^{2n}.
}
\tag{2}
$$

All the target weights, eigenvalues, and the floor in (2) belong to the
kernel in (1). No monotonicity in gamma is assumed. The floor describes
finite-sample representability; it is not a bound on continuum approximation
or held-out error. Equation (2) separates that floor from acquisition time.

**Proof sketch.** The density $\rho_\gamma$ has unit mass and cumulative
integral $[1+\tanh(\gamma t)]/2$, so convolving a centered step gives the
corresponding tanh feature and preserves the constant feature. Applying this
identity to both arguments of the step kernel proves (1); the Fourier
transform of $\rho_\gamma$ gives the displayed multiplier. The GD residual
satisfies $r_n=(I-\eta_\gamma K_\gamma)^ny$. Expanding $y$ in the orthogonal
positive eigenspaces and the nullspace proves (2). The
[technical note](gamma_factorized_readout.md) supplies the transform proof,
finite construction, and approximation bounds.

For $|\omega|/\gamma$ large, $M_\gamma(\omega)\sim2ze^{-z}$: smaller
gamma suppresses fine corrections. To obtain numerical learning times, this
filter must be applied through the finite geometry. Fourier frequencies
are generally not eigenvectors of the sampled kernel, so multiplying its
individual eigenvalues by $M_\gamma^2$ would not implement the theorem.

## Turning the mechanism into training-time bounds

The finite predictor has the form

$$
\widetilde K_{\gamma,Q}
=F_QD_{\gamma,Q}G_QD_{\gamma,Q}F_Q^T,
\qquad G_Q=C_QC_Q^T.
\tag{3}
$$

The matrices $F_Q,C_Q$ contain only the samples, centers, and step expansion;
$D_{\gamma,Q}$ contains the explicit multipliers in (1), with bias entry one.
The construction retains all geometric couplings. A controlled Fourier
expansion approximates the original nonperiodic features; it does not impose
periodic training data. Diagonalizing (3) and using (2) gives
$\widetilde E_n$ without observing GD iterates or fitting a learning rate.

The explicit feature remainder gives a kernel discrepancy bound
$\|K_\gamma-\widetilde K_{\gamma,Q}\|\le\Delta_Q$. When both kernels
contract at the prescribed step, telescoping their update powers yields
$|E_n-\widetilde E_n|\le n\eta_\gamma\Delta_Q$. The technical note also
gives a tighter target-dependent radius by retaining how the discrepancy
acts on the residual modes. Let $d_n$ be the smaller valid radius. For
$n_\epsilon=\inf\{n:E_n\le\epsilon\}$, checked integer witnesses give

$$
\boxed{
\widetilde E_a-d_a>\epsilon,\quad
\widetilde E_b+d_b\le\epsilon
\quad\Longrightarrow\quad
a+1\le n_\epsilon\le b.
}
\tag{4}
$$

The necessary-time statement uses monotonicity of the true residual. The
upper envelope itself need not be monotone; the sufficient endpoint is
checked directly. The tighter radius can evaluate the original operator's
action, but uses no optimization trajectory. The purely analytic radius
already gives nearly the same timing accuracy with more retained harmonics.

**Table 2. Quantified delay for the sine mixture, $W=559$.** The combined
filter bounds enclose every executed first hit at $\epsilon=0.01$; this
corresponds to relative squared loss $10^{-4}$. Endpoints have independent
192-bit interval-arithmetic certificates for nominal-real tanh on the
archived binary inputs and saved steps.

| Common slope $\gamma$ | Necessary updates | Sufficient updates | Executed first hit |
|---:|---:|---:|---:|
| 8 | 15,784,048 | 15,812,623 | 15,798,313 |
| 12 | 186,057 | 186,058 | 186,057 |
| 16 | 61,792 | 61,792 | 61,792 |
| 64 | 16,013 | 16,013 | 16,013 |

The gamma-8 bracket spans 0.181% of its observed acquisition time. All four
runs reach 1%, establishing that the large delay is compatible with
attainability. The normalized step is approximately $\eta_\gamma L_\gamma=0.5$
throughout. A control that changes only overall kernel magnitude predicts
16,013 updates at every gamma, whereas the explicit filter recovers the
986.59-fold delay. The
[detailed evaluation](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/REPORT.md)
documents the analytic-only bounds, control targets, and floating-point
sensitivity. The endpoint certificates do not certify every plotted curve
or the floating-point allowance formula.

## The main-paper figure

<figure>
  <img src="../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/paper_three_panel.png" alt="A: learned median and maximum slopes stay below the construction reference across widths; B: certified acquisition times overlay executed GD hits; C: scaling the common slope with width lowers the residual after a fixed update budget" style="max-width: 100%;">
  <figcaption><strong>Figure 1. Observed scale mismatch, quantified readout delay, and recovery with width.</strong> All panels use the same sine-mixture target. <strong>A:</strong> After 20,000 joint Adam updates, faint points show each of five seeds' median and maximum absolute hidden slopes; solid curves show their respective medians across seeds. The dashed curve is the constructive slope reference. Every seed's maximum stays below that reference, which is not asserted to be a necessary approximation threshold. <strong>B:</strong> With 559 fixed-center features and zero-initialized raw readout GD, certified necessary–sufficient intervals closely match executed times to 1% training residual. The inset divides each interval by its observed first-hit time; these are deterministic bounds, not confidence intervals. Lines between tested slopes guide the eye. <strong>C:</strong> After 200,000 raw-readout GD updates, fixed gamma 4 leaves residual near 0.42 across widths, while construction-matched slopes give residuals from 0.0034 to 0.0016. The dotted line is 1%. Panels B and C use curvature-normalized steps; panel A studies a different, joint-training process and provides empirical motivation rather than a consequence of the readout theorem.</figcaption>
</figure>

[Vector PDF](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/paper_three_panel.pdf).
The figure gives each panel a distinct role: A identifies the observed scale
gap, B tests the theorem's quantitative prediction, and C tests the proposed
intervention. The existing
[filter and target-weighted spectrum figure](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/three_panel.png)
remains supporting evidence for the mechanism rather than duplicating the
timing panel in the main figure.

The target is $f(x)=\sin(2\pi x)+\frac12\sin(6\pi x)+\frac14\sin(10\pi x)$
on $[-1,1]$. Training uses $m=16N+1$ equally spaced endpoint samples and
half empirical mean squared error. Frozen dictionaries use the existing
equispaced center construction with halo $\lceil\sqrt N\rceil$.
Panels A/C use $N=128,256,512,1024$; panel B uses $N=512$.
Panel A trains all hidden slopes, offsets, and readout parameters with Adam
at rate $10^{-3}$, epsilon $10^{-8}$, and moments $(0.9,0.999)$.
Hidden slopes and readout weights start independently uniform on
$[-\sqrt{6/(W+1)},\sqrt{6/(W+1)}]$; all biases start at zero. All five seeds
are shown. Panels B/C initialize every readout coefficient at zero and keep
the hidden parameters fixed; each case uses its saved step, approximately
$0.5/L_\gamma$.

Panel C varies both width and the specified slope strategy. Within each
width, the target, samples, centers, coefficient metric, and update budget
are identical between strategies. Its points are measured training
residuals, not new certified predictions or evidence of an irreducible floor
at gamma 4. This figure makes no held-out generalization claim.

## What gamma proportional to width recovers

The constructive reference uses $\lambda=\gamma h=1/4$, so
$\gamma=N/8=\Theta(W)$. For a grid-relative frequency $\xi=h\omega$,
substitution into the same filter gives the following consequence.

**Corollary (preserving access at the dictionary resolution).** If
$\gamma_N=\lambda/h$ for fixed $\lambda>0$, then

$$
M_{\gamma_N}(\xi/h)
=\frac{\pi|\xi|/(2\lambda)}{\sinh(\pi|\xi|/(2\lambda))},
\tag{5}
$$

independent of width. In any fixed band $|\xi|\le\Xi$, it is bounded below
by $[\pi\Xi/(2\lambda)]/\sinh[\pi\Xi/(2\lambda)]>0$ for $\Xi>0$.
For fixed gamma and nonzero $\xi$, as $h\to0$,

$$
M_\gamma(\xi/h)^2
\sim\frac{\pi^2\xi^2}{\gamma^2h^2}
\exp\!\left(-\frac{\pi|\xi|}{\gamma h}\right).
\tag{6}
$$

**Proof.** Substitute $\omega=\xi/h$ into (1), use that $z/\sinh z$
decreases for $z>0$, and apply its large-$z$ asymptotic.

Thus gamma proportional to width removes an exponentially growing
attenuation penalty for corrections whose physical frequency grows with
dictionary resolution. The proportionality constant matters: a positive
width-independent filter can still be small. The remaining finite geometry
and target weights in (2) determine the learning time; (5) alone does not
guarantee width-independent raw-GD times.

The scaling must specify a lower scale as well as an upper scale:
$\gamma=O(W)$ alone includes constant gamma. The reference takes
$\gamma=\Theta(W)$. For a fixed physical frequency, fixed gamma already
gives width-independent attenuation, while gamma proportional to width
makes $M_\gamma(\omega)\to1$. Panel C uses a fixed target and illustrates
the benefit of that intervention. It does not test a target family whose
frequencies grow with width, or establish that linear scaling is necessary
for every fixed target and tolerance.

## Reproduction and evidence record

From the repository root, run:

```bash
MPLCONFIGDIR=/tmp/gamma-paper-mpl \
python -m experiments.expD36_frozen_gamma_probe.paper_figure
```

The [plotting entry point](../experiments/expD36_frozen_gamma_probe/paper_figure.py)
reads the tracked full-sweep manifest, joint-training records, raw-GD case
and evaluation records, dictionary metadata, filter summary, and interval
audit. It checks target and seed indexing, zero readout initialization,
completed update budgets, saved step normalization, and agreement of each
timing endpoint with its archived certificate and executed hit. Its
[compact data record](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/paper_figure_data.json)
contains every plotted value, the plotting-source hash, and SHA-256 hashes
for all 23 source files. Seeds are 0–4; no seed is selected or excluded.
All inputs predate this figure. Reproduction generates PNG, PDF, and JSON
only; it uses no new GPU hours or optimizer trajectories.

The figure was inspected at a two-column width of 7.2 inches. All four plotted
intervals agree with their existing certificates and contain the executed
hits. The 20 focused filter/transfer tests pass; the full non-slow suite has
768 passes, 9 skips, 4 deselections, and the same 17 failing test identifiers
as the recorded baseline. The
[figure validation record](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/paper_validation.json)
records the checks and artifact hashes.

# How gamma controls readout learning

**An attainable target can still be expensive to learn.** With identical centers, target, and zero initialization, our frozen tanh readout reaches 1% relative error after **15.8 million GD updates at gamma 8**, versus **16,013 at gamma 64**. Adam also misses 1% at gamma 8 within 200,000 updates. We explain the mechanism explicitly: gamma smooths features, smoothing changes their correlations with needed error patterns, and the resulting target-weighted spectrum determines GD speed.

**Notation.** Eigenvalues use $\mu$; all norms are Euclidean.

| Symbol | Meaning |
|---|---|
| $J_\gamma$, $K_\gamma=J_\gamma J_\gamma^T$ | Normalized feature matrix and readout kernel |
| $\mu_i,u_i$; $\rho_i=\mu_i/\mu_1$ | Eigenpairs; eigenvalue relative to the largest |
| $y,r_n$; $p_i=|u_i^Ty|^2/\|y\|^2$ | Target, residual; target energy in eigenvector $i$ |

## 1. Why feature correlations control learning

Train only the linear readout $w$ in $J_\gamma w$, starting at zero, on $m$ input samples $x_a$. The $W$ tanh features have a common slope and consecutive equally spaced centers $c_j$, spacing $h$:

$$
J_\gamma[a,j]=\frac{\tanh(\gamma(x_a-c_j))}{\sqrt m},\qquad
J_\gamma[a,0]=\frac1{\sqrt m},\qquad y_a=\frac{f(x_a)}{\sqrt m}.
$$

Minimizing $\frac12\|J_\gamma w-y\|^2$ is ordinary half mean-squared error. GD changes residuals by $r_{n+1}=(I-\eta K_\gamma)r_n$. The response to a unit error pattern $v$ is therefore

$$
q_\gamma(v)=v^TK_\gamma v=\|J_\gamma^Tv\|^2
=\sum_i\mu_i(\gamma)|u_i(\gamma)^Tv|^2.
$$

This is the total squared correlation with the features. For an eigenvector it equals its learning eigenvalue; for other patterns it averages eigenvalues. If the residual is $v$, the correction along $v$ is $\eta q_\gamma(v)$. The largest eigenvalue limits the stable shared step, $0<\eta<2/\mu_1$, making the relative eigenvalues $\rho_i$ central to learning speed.

## 2. The explicit gamma mechanism

A tanh is a smoothed step: $\tanh(\gamma\,\cdot)=s_\gamma*\operatorname{sign}$, where $s_\gamma(x)=\frac\gamma2\operatorname{sech}^2(\gamma x)$. Its Fourier multiplier is

$$
M_\gamma(\omega)=\frac{z}{\sinh z},\qquad z=\frac{\pi|\omega|}{2\gamma},\qquad M_\gamma(0)=1.
$$

Small gamma averages over a wider region and suppresses rapid variation. To carry this into the kernel, replace the center sum by an integral. Subtracting the whole-line deficit of two tanh features from their saturated-product baseline gives the explicit contribution $B_\gamma$ below. The correction $R_\gamma$ restores finite endpoints and discrete centers:

$$
B_\gamma[a,b]=\frac{W+1}{m}-\frac{2}{hm}d_{ab}\coth(\gamma d_{ab}),\qquad
d_{ab}=x_a-x_b,\qquad R_\gamma=K_\gamma-B_\gamma.
$$

At $d=0$, use $d\coth(\gamma d)=1/\gamma$.

**Theorem (gamma-dependent kernel response).** For these centers, fixed samples, slopes $\Gamma>\gamma>0$, and any real vector $v$, set $V_v(\omega)=\sum_a v_a e^{-i\omega x_a}$. Then

$$
\boxed{q_\Gamma(v)-q_\gamma(v)=\mathcal A_{\Gamma,\gamma}(v)+v^T(R_\Gamma-R_\gamma)v,}
$$

$$
\mathcal A_{\Gamma,\gamma}(v)=\frac{2}{\pi hm}\int_{\mathbb R}
\frac{M_\Gamma(\omega)^2-M_\gamma(\omega)^2}{\omega^2}|V_v(\omega)|^2\,d\omega\ \ge0.
$$

**In words:** response changes by an explicit gain from reduced smoothing plus a finite-geometry correction. The pattern enters through $|V_v|^2$; gamma enters through $M_\Gamma^2-M_\gamma^2$. The kernel pairs two features, hence the squared multiplier. The identity is exact, including the continuous zero-frequency limit. The correction is signed and need not be small for every direction. Appendix A proves the statement.

**Direct validation.** Use $W=559$, $h=1/256$, $m=8{,}193$ uniform samples on $[-1,1]$, and target $f(x)=\sin(2\pi x)+\frac12\sin(6\pi x)+\frac14\sin(10\pi x)$. At gamma 8, the most target-aligned resolved eigenvector with $0<\rho_i\le2\times10^{-6}$ is mode 22, carrying **4.14% of target energy**. This retrospective choice uses the initial kernel and target, without training outcomes. Freeze that vector and predict its response at larger gamma by $q_8(v)+\mathcal A_{\Gamma,8}(v)$. Figure 1 shows a roughly **1,000-fold response increase**, with discrepancy below **0.061%** at gammas 12, 16, and 64. These are measured corrections for this probe. The gain is evaluated through the equivalent closed-form entries $2[d\coth(8d)-d\coth(\Gamma d)]/(hm)$, requiring no new eigenspace calculation.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_optimizer_access/optimizer_access_mechanism.png" alt="Gamma-dependent smoothing and the response of one fixed target-relevant pattern, measured and predicted." style="max-width: 100%;">
  <figcaption><strong>Figure 1. The mechanism predicts the response change.</strong> Left: analytic squared multipliers; dotted guides mark target frequencies. Right: direct finite-kernel response (circles) and reference response plus explicit Fourier gain (line). The same gamma-8 eigenvector is used throughout; it need not remain an eigenvector at larger gamma.</figcaption>
</figure>

## 3. From the mechanism to measured learning

A larger average response can coexist with remaining slow components. To determine acquisition time, retain each actual eigenvalue and its target weight. From zero initialization, classical GD gives

$$
e(n)^2:=\frac{\|r_n\|^2}{\|y\|^2}
=\sum_i p_i(1-\eta\mu_i)^{2n}
=\sum_i p_i(1-\rho_i/2)^{2n}\quad\text{when }\eta=1/(2\mu_1).
$$

Nullspace terms have factor one. A ratio of $10^{-6}$ requires about two million updates for one e-fold reduction of that component. The measured finite spectrum, target weights, and actual archived step predict **15,798,313; 186,057; 61,792; and 16,013 updates** to 1% at gammas 8, 12, 16, and 64, exactly matching executed crossings. All four therefore attain the studied accuracy. No decay rate is fitted to training.

**Adam tests whether the same difficult directions retain error.** Its adaptive updates do not follow GD's eigenmode decay law. In Figure 2B, the gamma-8 band $0<\rho_i\le2\times10^{-6}$ starts with **4.33%** of target energy and retains **$1.47094\times10^{-4}$** after 200,000 Adam updates, exceeding the $10^{-4}$ squared-error budget for 1% accuracy. Corresponding band energies at larger gammas are below budget. Energies are normalized by $\|y\|^2$; the ratio-defined subspaces may change with gamma.

Figure 2C reports first crossings of an EMA of squared relative error with one shared **100-update half-life**: no crossing at gamma 8, then **7,688; 1,900; and 1,486** updates. Crossings can recur; this measures first acquisition, not sustained accuracy. The EMA's initial-loss memory prevents any crossing before **1,329 updates**, compressing differences between the fastest runs. Appendix B gives actual loss curves, protocol, and sensitivity checks.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_optimizer_access/optimizer_access_three_panel.png" alt="Measured kernel spectra, needed and remaining slow-mode energy, and GD and Adam acquisition times." style="max-width: 100%;">
  <figcaption><strong>Figure 2. Spectrum, target overlap, and acquisition.</strong> A: leading 64 finite-kernel eigenvalue ratios; ranks are ordered separately at each gamma. B: cumulative initial target energy (dashed) and Adam residual energy at 200,000 updates (solid). The blue residual exceeds the 1% error budget in slow modes alone. C: executed GD crossings and measured-spectrum forecasts, alongside first Adam EMA crossings. The arrow marks censoring at 200,000; GD has a longer budget. All panels use the same target and geometry.</figcaption>
</figure>

## Appendix A. Full proof

Let $I=[c_1-h/2,c_W+h/2]$ and $F_{ab,\gamma}(c)=\tanh(\gamma(x_a-c))\tanh(\gamma(x_b-c))$. The identity $1-\tanh u\tanh v=\coth(u-v)(\tanh u-\tanh v)$ and the integral of two translated tanh profiles give

$$
\int_{\mathbb R}(1-F_{ab,\gamma}(c))\,dc=2d_{ab}\coth(\gamma d_{ab}).
$$

The coincident limit is $2/\gamma$. Since $|I|=Wh$, subtract this deficit from the constant baseline and restore the exterior part to obtain $K_\gamma=B_\gamma+R_\gamma$, with

$$
R_\gamma[a,b]=\frac1m\left[\sum_{j=1}^{W}F_{ab,\gamma}(c_j)-\frac1h\int_I F_{ab,\gamma}(c)\,dc\right]
+\frac1{hm}\int_{\mathbb R\setminus I}(1-F_{ab,\gamma}(c))\,dc.
$$

The terms are the center quadrature discrepancy and finite-interval correction. The bias contributes $1/m$ to $B_\gamma$ and cancels across slopes.

Both $s_\gamma*\operatorname{sign}$ and $\tanh(\gamma\,\cdot)$ vanish at zero and have derivative $2s_\gamma$, proving the smoothing identity. With $\widehat g(\omega)=\int g(x)e^{-i\omega x}\,dx$, substitute $u=e^{2\gamma x}$ and $a=\omega/(2\gamma)$:

$$
\widehat s_\gamma(\omega)=\int_0^\infty\frac{u^{-ia}}{(1+u)^2}\,du
=\Gamma(1-ia)\Gamma(1+ia)=\frac{\pi a}{\sinh(\pi a)}=M_\gamma(\omega).
$$

The beta integral and gamma-function reflection identity give the middle equalities; $\Gamma(\cdot)$ here denotes the gamma function.

Put $t_\gamma(u)=\tanh(\gamma u)$. Differentiating the deficit $\int[1-t_\gamma(u)t_\gamma(u-d)]\,du$ twice and integrating by parts gives

$$
\frac{d^2}{dd^2}[2d\coth(\gamma d)]
=-\int t_\gamma(u)t_\gamma''(u-d)\,du
=\int t_\gamma'(u)t_\gamma'(u-d)\,du
=4(s_\gamma*s_\gamma)(d).
$$

The boundary term vanishes and $s_\gamma$ is even. Thus the integrable difference $D(d)=2[d\coth(\gamma d)-d\coth(\Gamma d)]$ has transform

$$
\widehat D(\omega)=\frac{4(M_\Gamma(\omega)^2-M_\gamma(\omega)^2)}{\omega^2},\qquad
\widehat D(0)=\frac{\pi^2}{3}(\gamma^{-2}-\Gamma^{-2}).
$$

Use $\widehat{D''}=-\omega^2\widehat D$ and expand $z/\sinh z$ at zero. This multiplier decreases with $z>0$, so $M_\Gamma\ge M_\gamma$ and $\widehat D\ge0$. Finally, $B_\Gamma[a,b]-B_\gamma[a,b]=D(x_a-x_b)/(hm)$. Fourier inversion contributes $1/(2\pi)$; summing against $v_av_b$ gives $|V_v|^2$ and the theorem's prefactor $2/(\pi hm)$. Adding the exact finite correction completes the proof. $\square$

## Appendix B. Adam protocol and supporting checks

**Metric and protocol.** The common Adam setting uses initial rate $10^{-3}$, epsilon $10^{-12}$, and moments $(0.9,0.999)$. The rate stays fixed through 20,000 updates, decays by cosine to $10^{-6}$ at 50,000, then stays fixed through 200,000. Define

$$
M_0=1,\qquad M_n=\beta M_{n-1}+(1-\beta)e(n)^2,\qquad \beta=2^{-1/100}.
$$

First EMA acquisition is the first $M_n\le10^{-4}$, calculated from every update. Plots show $\sqrt{M_n}$. Later upcrossings occur 73, 97, and 98 times at gammas 12, 16, and 64; sustained raw crossings occur around 41,000. The bound $M_n\ge\beta^n$ gives the 1,329-update memory floor. The shared window is a retrospective visualization choice.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_optimizer_access/optimizer_access_schedule.png" alt="Actual Adam error envelopes and EMA curves with first crossings marked." style="max-width: 100%;">
  <figcaption><strong>Figure B1. Actual training and the acquisition metric.</strong> Bands show raw-error minima and maxima in update bins; lines show the loss EMA's square root. Vertical dashed lines mark first crossings. The gray window marks rate decay. Curves are sampled for display; crossing calculations use every update.</figcaption>
</figure>

**Robustness.** Five targets are included: the sine mixture, $\exp(\sin(3\pi x))$, $1/(1+25x^2)$, $\sqrt5x^2$, and $\sqrt2\sin(2\pi x)$. Alongside the common setting, archived settings were selected from five rates and two epsilons using median validation error at five checkpoints from 40,000 to 50,000 updates; continuation preserved optimizer state. This selection rewards late accuracy, not early acquisition. Figure B2 shows dependence on both target and optimizer setting.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_optimizer_access/optimizer_access_targets.png" alt="First EMA acquisition for all five targets and both Adam setting choices." style="max-width: 100%;">
  <figcaption><strong>Figure B2. Target and optimizer dependence.</strong> All cases use the 100-update half-life. Arrows denote no crossing by 200,000. The results do not support universal monotone improvement with gamma.</figcaption>
</figure>

The primary common-setting ordering persists at EMA half-lives 30, 100, and 300. At 1,000, crossings cluster near 28,000-30,000 and reorder. Validation-selected settings also change the first-crossing comparison (Figure B3).

<figure>
  <img src="../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_optimizer_access/optimizer_access_ema_sensitivity.png" alt="EMA acquisition versus shared half-life and the initial-loss memory floor." style="max-width: 100%;">
  <figcaption><strong>Figure B3. Window sensitivity.</strong> Each horizontal position uses one half-life across gammas. Dotted vertical lines mark 100; the gray dashed line is the initial-loss memory floor. Triangles denote censoring at 200,000. All points use the same saved trajectories.</figcaption>
</figure>

**Numerical evidence.** Archived FP64 calculations retain singular values above $10^{-14}$ of the largest; omitted directions are tracked separately from exact nullspace. Independent SVD and residual projections reproduce slow-band energies within $5.1\times10^{-14}$. Adam projections measure current error and may increase during training. These are training-grid optimization results. The [evidence directory](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_optimizer_access/) contains plotted values, source hashes, and verification records; no new training was performed.

Reproduce from the repository root:

```bash
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m experiments.expD36_frozen_gamma_probe.gamma_mechanism_analysis
python -m experiments.expD36_frozen_gamma_probe.adam_spectral_analysis
python -m experiments.expD36_frozen_gamma_probe.adam_ema_analysis --archive /path/to/adam_raw_scalar_archive.npz
python -m experiments.expD36_frozen_gamma_probe.optimizer_access_figure
latexmk -pdf -outdir=/tmp/gamma-optimizer-latex docs/gamma_optimizer_access_note.tex
```

Extract full scalar traces with `adam_trace_audit.py`; the retained compact EMA artifacts suffice to regenerate figures without that extraction.

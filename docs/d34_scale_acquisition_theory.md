# Signal depletion and finite-time scale acquisition in D34

This companion argument addresses D34's actual random-readout initialization and simultaneous, equal-rate GD in physical coordinates. The [focused evidence report](../results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/README.md) tests its measurable ingredients. The separate [zero-readout note](readout_driven_slope_signal_depletion.md) supplies an early-window asymptotic mechanism under different initial conditions; it does not prove the statements about the later D34 trajectory below.

The proposed paper claim is finite-time and target dependent: coarse fitting initially reduces the slope gradient; subsequent nonlinear recovery need not deliver enough suitably directed, sufficiently distributed motion to acquire a specified population of scales. Readout growth can restore signal. A complete explanation must therefore account for its recovery as well as its depletion. The propositions below turn this claim into quantities that can fail on the measured trajectory. They do not yet provide an initialization-only long-time prediction.

## 1. Use the actual model and distinguish three questions

D34 trains

$$
F(x)=\sum_{j=1}^{W}c_j\tanh(a_jx+b_j)+d,
\qquad L=\tfrac12\langle r,r\rangle_m,
\qquad r=F-y,
\qquad w_{n+1}=w_n-\eta\nabla L(w_n).
$$

Here $\langle\cdot,\cdot\rangle_m$ is the mean on the symmetric training grid, $w=(a,b,c,d)$, and all four blocks use $\eta=0.002$. The focused runs use $W=177$, 2,048 training points, and the original paired seeds 0–4. Both slopes and readouts start random and nonzero. The flow expressions below are identities for the corresponding vector field evaluated at GD states; any time integration requires a separate discretization check.

Define

$$
R=\|r\|_m,\qquad \Xi=\frac{\|g_a\|_2}{R},\qquad
\chi=-\frac{\operatorname{sign}(a)^Tg_a}{\sqrt W\,\|g_a\|_2}.
$$

Away from zero slopes, the exact flow identity is

$$
\frac{d\bar\gamma}{dt}=\frac{R\Xi\chi}{\sqrt W},
\qquad \bar\gamma=W^{-1}\sum_j|a_j|,\qquad -1\le\chi\le1.
\tag{1}
$$

Thus a recovered $\Xi$ answers whether a slope signal exists. Its sign alignment $\chi$ answers whether it increases mean slope magnitude. Neither alone answers whether many neurons acquire large slopes. For GD, use the exact increment $\delta\gamma_{jn}=|a_{jn}-\eta g_{ajn}|-|a_{jn}|$, including crossings through zero. In particular,

$$
\Delta\bar\gamma_n=-\frac{\eta}{W}\operatorname{sign}(a_n)^Tg_{an}+r_{{\rm cross},n}.
\tag{2}
$$

For each neuron retain $P_j=\sum_n(\delta\gamma_{jn})_+$ and $N_j=\sum_n(-\delta\gamma_{jn})_+$. The equality $P_j-N_j=\gamma_{j,\rm end}-\gamma_{j,\rm start}$ checks every-update accounting. The share of $\sum_jP_j$ carried by the largest $\lceil0.1W\rceil$ entries and the participation fraction $(\sum_jP_j)^2/(W\sum_jP_j^2)$ distinguish widespread motion from concentrated motion. These are descriptive statistics, not assumptions of the theorem.

## 2. Attribute depletion without assuming readout dominance

Where $R,\|g_a\|_2>0$, the exact equal-rate flow identity is

$$
-\frac{d}{dt}\log\Xi=D_c+D_d+D_q,
\qquad
D_I=\frac{g_a^T H_{aI}g_I}{\|g_a\|_2^2}
-\frac{\|g_I\|_2^2}{R^2},\qquad q=(a,b).
\tag{3}
$$

Positive $D_I$ depletes normalized signal; negative $D_I$ regenerates it. The output-bias term belongs in D34. This differential attribution includes normalization by the changing residual. It is not the result of retraining with one block frozen, and a block's ratio to the total is unstable when signed contributions cancel.

Writing $s_j(x)=\operatorname{sech}^2(a_jx+b_j)$ makes the readout contribution interpretable:

$$
(H_{ac}g_c)_j
=c_j\left\langle xs_j,\sum_k\tanh(a_kx+b_k)(g_c)_k\right\rangle_m
+(g_c)_j\langle r,xs_j\rangle_m.
\tag{4}
$$

The first term changes the residual through the readout. The second changes the coefficient multiplying the slope tangent. Either term can have either signed projection onto $g_a$. The diagnostic records both and the residual-normalization term separately. Growing $\|c\|_2$ alone does not establish useful signal regeneration.

**Why the zero-readout majority fraction cannot be imported.** Consider the centered scalar affine reference $F=(a^Tc)x$, with target projection $\beta x$, and put $A=\|a\|_2^2$, $C=\|c\|_2^2$, $e=a^Tc-\beta$, and $\mu_2=\langle x,x\rangle$. Its equations are

$$
\dot a=-\mu_2 e c,\qquad \dot c=-\mu_2 e a,
\qquad \dot A=\dot C=-2\mu_2 e\,a^Tc.
$$

Consequently $A-C$ is invariant. Assuming the reference approaches coarse fit with a nonzero unfitted residual and nondegenerate $A_\infty,C_\infty$, equation (3) gives

$$
D_c\longrightarrow\mu_2 A_\infty,\qquad
D_a\longrightarrow\mu_2 C_\infty,\qquad
\frac{D_c}{D_c+D_a}\longrightarrow
\frac{A_\infty}{A_\infty+C_\infty}.
\tag{5}
$$

Equal initial norms imply a one-half limiting attribution. Independent isotropic hidden/readout laws with equal limiting squared norms therefore suggest a one-half limit in this centered affine reduction. Zero initial readout gives a different invariant and permits the larger fraction in the separate note. This calculation is a counterexample to a universal readout-majority claim, not a theorem for finite D34 with random biases and noncentered targets. Actual D34 attribution must be measured or proved in its full setting.

## 3. Preserve regenerated coarse forcing and the sign of the remaining force

Let $P$ be the fixed empirical orthogonal projection onto $\operatorname{span}\{1,x\}$. Define the exact tanh slope forces

$$
(g_a^C)_j=c_j\langle Pr,xs_j\rangle_m,\qquad
g_a^T=g_a-g_a^C,
\qquad v_C=-W^{-1}\operatorname{sign}(a)^Tg_a^C,
\qquad v_T=-W^{-1}\operatorname{sign}(a)^Tg_a^T.
\tag{6}
$$

Then $\Delta\bar\gamma_n=\eta(v_C+v_T)+r_{{\rm cross},n}$. This is a residual projection using the full tanh tangent, not an affine-activation approximation. $Pr$ can regenerate after its first decline; the decomposition continues to include it. A negative $v_T$ means the remaining-residual force opposes mean sharpening, not that it necessarily harms approximation. Its instantaneous contribution to descent of the remaining-residual loss under the actual slope update is $(g_a^T)^Tg_a$, which the diagnostic also records.

A high-order target's small direct coupling to broad features does not bound the full current gradient. Evolving coefficients, changing tangents, and a regenerated coarse residual all remain possible. A long-time depletion theorem would have to control these terms along the joint trajectory.

## 4. Population acquisition has an exact distance and a conditional budget

Fix a slope threshold $\Gamma>0$ and a population fraction $p\in(0,1]$. At a starting state $a_s$, sort the deficits $(\Gamma-|a_{sj}|)_+$ increasingly as $d_{(1)},\ldots,d_{(W)}$ and set

$$
\mathcal D_{p,\Gamma}(a_s)
=\left(\sum_{j=1}^{\lceil pW\rceil}d_{(j)}^2\right)^{1/2}.
\tag{7}
$$

**Proposition 1 (distance and path budget).** $\mathcal D_{p,\Gamma}$ is exactly the Euclidean distance from $a_s$ to the set of slope vectors having at least $\lceil pW\rceil$ magnitudes at least $\Gamma$. For the discrete GD trajectory define

$$
B_{s,N}=\eta\sum_{n=s}^{N-1}\|g_{an}\|_2
=\eta\sum_{n=s}^{N-1}R_n\Xi_n.
\tag{8}
$$

If $B_{s,N}<\mathcal D_{p,\Gamma}(a_s)$, the trajectory cannot acquire that population at any update from $s$ through $N$.

*Proof.* Moving one coordinate to magnitude $\Gamma$ costs at least its positive deficit, attained by moving outward with its original sign (either sign at zero). Choosing the smallest required deficits minimizes the squared Euclidean cost. The triangle inequality gives $\|a_k-a_s\|_2\le B_{s,k}\le B_{s,N}$ for every $k\le N$. Combining these facts proves the statement. No small-slope approximation is needed.

The measured $B$ makes this a retrospective exclusion, subject to numerical error. An initialization-only theorem requires an independently justified upper envelope for $R_n\Xi_n$. A bound that exceeds $\mathcal D$ is inconclusive, even if no neuron actually acquires the threshold.

For comparison, if a descent inequality $L_n-L_{n+1}\ge\alpha\eta\|\nabla L_n\|_2^2$ holds throughout the interval, Cauchy–Schwarz only gives

$$
B_{s,N}\le
\sqrt{\frac{\eta(N-s)}{\alpha}(L_s-L_N)}.
\tag{9}
$$

This generic energy bound should be compared with (8) before attributing a useful exclusion specifically to depleted slope signal. A measured minimum descent ratio is a retrospective check, not a guarantee on future updates.

**Proposition 2 (reference-to-population transfer).** If an independently justified trajectory enclosure gives $\|a_n-\widehat a_n\|_2\le E_n$, then for any $0<\epsilon<\Gamma$,

$$
\frac{\#\{j:|a_{nj}|\ge\Gamma\}}{W}
\le\min\left\{1,
\frac{\#\{j:|\widehat a_{nj}|\ge\Gamma-\epsilon\}}{W}
+\frac{E_n^2}{W\epsilon^2}\right\}.
\tag{10}
$$

*Proof.* Every acquired coordinate outside the reference count requires at least $\epsilon$ coordinate error. Sum squared errors. A numerical reference discrepancy can assess prediction quality but does not itself establish an enclosure. The existing cubic and degree-7 references cannot supply a long-time certificate once their measured errors become large.

## 5. What would validate or defeat the proposed explanation?

The focused comparison fixes the targets, seeds, equal rate, and 600k horizon already present in D34. Its outcomes support different levels of claim:

| Claim | Quantitative test | Outcome that defeats the claim |
|---|---|---|
| Readout accounts for a majority of early depletion | Signed $D_c,D_d,D_q$ over a specified, resolved interval; check quadrature and GD/flow error if integrated | Readout's integrated signed contribution is not a majority, or the interval has no net depletion |
| Total signal remains depleted | Full current $\Xi$, including recovered signal, relative to an independently specified envelope | Recovery above that envelope within its claimed interval |
| Recovery fails to produce broad sharpening | Exact positive/negative per-neuron travel, $\chi$, concentration, and threshold counts | Broad, sustained outward movement that acquires the stated population |
| A signal budget explains finite-time non-acquisition | Compare $B_{s,N}$ and $\mathcal D_{p,\Gamma}$, alongside the generic energy budget | $B\ge\mathcal D$ leaves this bound inconclusive; a measured acquisition contradicts a purported strict exclusion |
| A low-order reference predicts later dynamics | Errors in current gradients and signed motion against actual tanh | Growing error at the claimed horizon; agreement in the early interval cannot repair it |

Use a norm budget where it is informative and signed population diagnostics where it is not. Do not infer a continuous maximum from sparse plotting samples. Do not claim that a construction's $\lambda=\Gamma(2/128)$ is a universal necessary accuracy scale for an unrestricted network. The evidence can establish failure to acquire specified construction-reference populations; necessity requires a separate approximation result with hypotheses that cover the moving trained network.

The remaining proof problem is to control the coupled readout, coarse residual, and useful population motion under D34 initialization up to a target-dependent stopping time. A random-readout extension of the early theorem and a verified nonlinear continuation are plausible routes. Neither is completed by the identities or retrospective budgets here. The zero-readout experiments remain useful for their own mechanism, but matching them would not close the D34 claim.

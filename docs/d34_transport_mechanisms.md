# Failed scale acquisition as deterministic transport

Why does noiseless gradient descent remain far from the gamma regime motivated by the approximation theory, even after the loss decreases and slope motion revives? The working explanation has three parts: fitting accessible residual components removes their driving force; the remaining error can couple only weakly to the current slopes; and revived motion can be too small, misdirected, or concentrated to deliver the required population of scales. These mechanisms concern the observed failure to reach the intended scale and precision regime. Partial error reduction through better readout fitting is not evidence that this regime has been acquired.

This note develops the intuition behind the [conditional barrier walkthrough](d34_barrier_theorem_walkthrough.md). It proves elementary results for the actual tanh vector field and for an illustrative coupled model, then connects them to the measured D34 trajectories. The purpose is to understand mechanisms present in the failure cases. The results do not establish a long-time barrier from D34 initialization.

**Notation. Inner products are empirical means over the fixed training grid.**

| Symbol | Meaning |
|---|---|
| $W,m$ | Physical neuron count and training-sample count. |
| $a,b,c,d$ | Slopes, hidden biases, readout weights, and output bias. |
| $\rho=W^{-1}\sum_j\delta_{(a_j,b_j,c_j)}$ | Joint empirical probability measure of neuron parameters. |
| $r=F-y$, $R=\Vert r\Vert_m$, $L=R^2/2$ | Residual, residual RMS, and half-MSE. |
| $\eta$, $t_n=n\eta$ | Common GD step and physical training time; all rates are equal here. |
| $\gamma=\lvert a\rvert$, $P_\Gamma=\rho\{\lvert a\rvert\ge\Gamma\}$ | Slope magnitude and population fraction above a chosen scale. |
| $p,\Gamma$ | Required fraction and scale in the conditional population event. |
| $e_C,e_H,z_C$ | Coarse residual coefficients, retained non-affine coefficients, and lag behind their induced coarse balance. |
| $T_a,S$ | Effective slope-force map and effective residual kernel after coarse balancing. |
| $\varepsilon,M,C_0,R_0$ | Small parameter and constants in the local persistence result. |
| $f,Y,h=Y-f,\mathcal H$ | Surrogate output coefficient, its target, remaining driving coefficient, and accumulated drive. |
| $Zq_h$ | Orthogonal unresolved target component in the surrogate. |

## 1. What the observed failure includes

The main comparison uses width 177, independent random nonzero readouts, simultaneous GD with $\eta=0.002$, 2,048 training midpoints, and seeds 0–4 through 600,000 updates, or $T=1200$. The degree-3 and degree-9 targets share their constant and linear coefficients. Their remaining target component has degree 3 or 9, respectively. The [recovery report](../results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/README.md) and [transport report](../results/checkpoint_D_optimizers/expD34_readout_race/transport_barrier/README.md) give the individual trajectories and numerical controls.

**Observed mechanisms. The summaries concern the original equal-rate trajectories; shares and norms are different measurements.**

| Target | Observation | Mechanism the observation supports | What it does not establish |
|---|---|---|---|
| Degree 9 | Large remaining error, tiny slope force, and slight mean contraction after the coarse transient. Direct mode-9 force is much smaller than the total effective force. | Weak access to the hard target; most surviving motion corrects generated lower-order errors. | Permanent trapping, or a unique readout-driven cause. |
| Degree 3 | Most neurons grow modestly; late fitting is largely allocated to readout weights. Errors and gammas remain far from the intended precision and scale regime. | Recovery with insufficient scale displacement; partial fitting can continue without acquiring the intended scales. | Growth confined to a few neurons, or successful scale acquisition inferred from lower readout error. |
| Sine | Stronger force recovery, more concentrated upward travel, and at most one neuron above 3.2 per original seed. | Recovered force without enough broadly distributed outward transport. | A proved causal mechanism in which the first escaping neuron suppresses the rest. |

Over updates 20k–600k, 93.2%–94.9% of degree-3 neurons have net growth, but no neuron reaches 3.2. The median mean slope is about 0.22. The subsequent [fixed-geometry readout measurements](../results/checkpoint_D_optimizers/expD34_readout_race/useful_slopes/README.md) reduce degree-3 relative evaluation MSE from 0.75053 on the initial geometry to 0.02480 on the final geometry under a common fresh-readout budget. That is partial fitting improvement, still far from precision; it does not demonstrate acquisition of the geometry sought by the theory. Degree 9 remains near 0.75001 under that diagnostic.

The event $P_\Gamma\ge p$ gives a precise measurement of scale acquisition, for example $p=0.1$ and $\Gamma=3.2$. It is one diagnostic of the broader failure to reach the theoretical regime. A theorem excluding this event is not by itself an approximation lower bound for every heterogeneous network. The present interpretation does not require such a necessity theorem: the observed errors remain large and the intended scales are not acquired.

These are retrospective training-mechanism observations. The additional evaluation grid measures deterministic approximation error, not statistical generalization. No new training or parameter-selection experiment is introduced here.

## 2. The transport field is generated by the remaining residual

For $|x_i|\le1$, define

$$
F_\rho(x)=d+W\int c\tanh(ax+b)\,d\rho,
\qquad
\langle u,v\rangle_m=\frac1m\sum_{i=1}^m u(x_i)v(x_i).
$$

Writing $u=ax+b$ and $s=\operatorname{sech}^2u$, the equal-rate characteristic velocity is

$$
v_a=-c\langle r,xs\rangle_m,\qquad
v_b=-c\langle r,s\rangle_m,\qquad
v_c=-\langle r,\tanh u\rangle_m,\qquad
\dot d=-\langle r,1\rangle_m.
\tag{1}
$$

The continuity equation and its exact GD counterpart are

$$
\partial_t\rho+\nabla\cdot(\rho v[\rho,d])=0,
\qquad
\rho_{n+1}=(\operatorname{Id}+\eta v[\rho_n,d_n])_\#\rho_n,
\quad
d_{n+1}=d_n-\eta\langle r_n,1\rangle_m.
\tag{2}
$$

The pushforward symbol $\#$ means that every particle moves through the indicated map. Atomic $\rho$ gives the finite-network gradient-flow ODE in continuous time and the finite-network simultaneous GD update in discrete time. Replacing those atoms by the continuous initialization law is an additional approximation. There is no diffusion: new mass at large scales must arrive by transporting existing particles.

The slope marginal is generally unclosed. Its conditional outward velocity for $\gamma>0$ involves

$$
\mathbb E_{\rho_t}\!\left[
\operatorname{sign}(a)v_a\mid |a|=\gamma
\right].
$$

Readout, bias, and sign correlations matter even when two gamma histograms look alike. Nor does a large residual ensure a large velocity: the residual must correlate with the specific tangent functions in (1). The field changes as fitting changes those correlations.

This transport formulation has precedents in [Mei, Montanari, and Nguyen](https://arxiv.org/abs/1804.06561) and [Chizat and Bach](https://arxiv.org/abs/1805.09545). Their results motivate distributional and characteristic analysis; their scaling and hypotheses do not supply a barrier for these D34 runs.

## 3. Arrest in the affine model and slow motion near it

### Proposition 1: an accessible residual can remove its own driving field

Replace tanh by its affine part, so $F=d+\sum_jc_j(a_jx+b_j)$. If

$$
\langle r,1\rangle_m=\langle r,x\rangle_m=0,
$$

then every parameter gradient vanishes. This state is fixed under both gradient flow and GD, including when $r\ne0$.

**Proof.** In this model,

$$
v_a=-c\langle r,x\rangle_m,\quad
v_b=-c\langle r,1\rangle_m,\quad
v_c=-a\langle r,x\rangle_m-b\langle r,1\rangle_m.
$$

The two assumed correlations also make $\dot d=0$. The update is therefore zero in every block.

The actual tanh model also has a simple stationary configuration: take $a_j=b_j=0$, $d=0$, arbitrary $c_j$, and a target orthogonal to $1$ and $x$. Substitution into (1) gives zero velocity with loss $\|y\|_m^2/2$. This proves stationarity, not attraction from random initialization.

The affine example isolates the depletion mechanism. Fitting the constant and linear components removes the whole accessible field even though non-affine error remains. Both geometry and readout can contribute to that fitting. It does not imply that equal-rate D34 is dominated by readout alone, or that nonlinear forcing stays absent afterwards.

### Proposition 2: exact bounds on the nonlinear force

Put $\ell_0=\langle r,1\rangle_m$ and $\ell_1=\langle r,x\rangle_m$. At every tanh state,

$$
\begin{aligned}
|v_a|&\le |c|\bigl(|\ell_1|+R(|a|+|b|)^2\bigr),\\
|v_b|&\le |c|\bigl(|\ell_0|+R(|a|+|b|)^2\bigr),\\
|v_c|&\le |a||\ell_1|+|b||\ell_0|
+\frac R3(|a|+|b|)^3.
\end{aligned}
\tag{3}
$$

**Proof.** The identities and bounds

$$
|1-\operatorname{sech}^2u|=\tanh^2u\le u^2,
\qquad
|\tanh u-u|\le |u|^3/3
$$

follow from $|\tanh u|\le|u|$ and integration of $\tanh^2u$. Split the slope and bias tangents into their affine part and remainder. Split $\tanh u=u+(\tanh u-u)$ for the readout. Apply Cauchy–Schwarz in the empirical norm and $|u|\le |a|+|b|$, $|x|\le1$. No Taylor remainder is discarded.

Equation (3) separates coarse signal from nonlinear access to the remaining error. After coarse relaxation, small $a,b,c$ multiply the latter by three small-parameter factors. A large $R$ can consequently coexist with a small characteristic speed.

### Corollary 2.1: conditional persistence in a small-parameter neighborhood

At time $s$, suppose every neuron satisfies

$$
\max(|a_j|,|b_j|,|c_j|)\le M\varepsilon,\qquad M,\varepsilon>0.
$$

Until the first neuron reaches the boundary of the larger box with coordinate magnitudes $2M\varepsilon$, assume on the interval being considered that

$$
R\le R_0,\qquad |\ell_0|,|\ell_1|\le C_0\varepsilon^2.
\tag{4}
$$

Define

$$
K_*=\max\left\{
2MC_0+32M^3R_0,\;
4MC_0+\frac{64}{3}M^3R_0
\right\}.
$$

If $K_*>0$, any such first boundary time satisfies

$$
t_{\rm exit}-s\ge \frac{M}{K_*\varepsilon^2}.
\tag{5}
$$

For GD, assume (4) at every old state before the first boundary-reaching update. If that update occurs at index $N$, then $(N-s_{\rm idx})\eta$ satisfies the same bound, where $s_{\rm idx}$ is the starting update. The claim only concerns exit within the interval where (4) is assumed.

**Proof.** Inside the larger box, (3) bounds all three coordinate speeds by $K_*\varepsilon^3$. A coordinate must move at least $M\varepsilon$ to reach the boundary. Integrate its absolute speed, or sum its exact GD increments including the step reaching the boundary. If $K_*=0$, those coordinate velocities vanish throughout the assumed interval.

For $\varepsilon=W^{-1/2}$ and constants independent of width, the lower bound is proportional to $W$. This concerns a change comparable to the initial parameter scale, not arrival at a fixed large gamma. Its nontrivial hypothesis is continued coarse-residual control. Output bias remains trained, and the corollary does not prove that its evolution or the joint moments preserve (4). It explains what local conditions produce a delay without assuming a small gradient directly.

## 4. Coarse balance allows depletion and regeneration

In the actual network, higher-degree fitting continually induces constant and linear error. Permanent removal of the coarse residual would therefore remove a measured part of the mechanism.

Use fixed empirical orthonormal residual modes, with $e_C=(e_0,e_1)^T$ and $e_H=(e_2,\ldots,e_\ell)^T$. Let $J_a$ be the slope Jacobian of these coefficients, and let $K$ be the total modal tangent kernel, including all trained blocks. When $K_{CC}$ is invertible, define

$$
B=K_{CC}^{-1}K_{CH},\qquad
z_C=e_C+Be_H,\qquad
T_a=J_{a,H}^T-J_{a,C}^TB,\qquad
S=K_{HH}-K_{HC}B.
$$

The [walkthrough](d34_barrier_theorem_walkthrough.md) derives the exact identities

$$
g_a=J_{a,C}^Tz_C+T_ae_H+g_{a,\perp},
\qquad
\dot e_H=-Se_H-K_{HC}z_C+f_H.
\tag{6}
$$

Here $g_{a,\perp}$ and $f_H$ retain the omitted residual. The balanced coarse force is already inside $T_ae_H$; a small tracking term does not mean the coarse residual contributes no force. In the degree-3 data, the raw coarse-residual contribution supplies all the net post-20k mean growth and more, while the remaining-residual contribution opposes it in each original seed.

Equation (6) describes motion near a moving coarse balance. Tracking estimates determine whether fast coarse relaxation keeps up. The surviving map $T_a$ determines whether the remaining residual moves slopes. Both may change and regenerate force. Even fine-residual relaxation can increase $\|T_ae_H\|$ by changing residual direction; no monotone-decay assertion follows from $S\succeq0$ alone.

The formal width predictions in the [technical note](d34_transport_scale_barrier.md) illustrate the target dependence. With bounded rescaled joint moments, low-order non-affine forcing gives population force $\|g_a\|$ of order $W^{-1}$, suggesting rescaled-geometry drift at time $W$. For degree 9, direct target coupling has scale $W^{-4}$, but generated lower-mode errors can produce the larger $W^{-2}$ force. Their candidate drift times are $W^4$ and $W^2$, respectively. The coarse-elimination correction contributes at leading order and cannot be omitted.

These are formal regimes, not established acquisition-time laws. At $W=177$, the horizon has $T/W=6.78$ and $T/W^2=0.0383$. This is consistent with nonlinear recovery on the lower-order targets and weak motion on degree 9. It does not imply that the recovered lower-order trajectories approach the intended precision or gamma regime.

## 5. A coupled model where fitting exhausts the drive before scale acquisition

The following surrogate generates its driving field through its own residual and trains both parameter blocks at the same rate. Depletion will follow from these coupled dynamics.

Choose two fixed empirical orthonormal functions $q,q_h$ and set

$$
F(x)=f q(x),\qquad
f=\sum_{j=1}^W c_ja_j^3,\qquad
y(x)=Yq(x)+Zq_h(x),\qquad Y>0,\quad Z\in\mathbb R.
$$

Then

$$
L=\tfrac12\bigl((f-Y)^2+Z^2\bigr),\qquad h=Y-f.
\tag{7}
$$

The cubic coefficient is motivated by the first nonlinear term of tanh after affine fitting. Here $a$ is a slope surrogate: its large values are not quantitatively calibrated to tanh gamma. The component $Zq_h$ is inaccessible by construction. Taking $Z\ne0$ makes explicit that depletion of the accessible driving signal can leave nonzero total error. Taking $Z=0$ recovers the single-component model.

### Proposition 3: coupled fitting with bounded scale acquisition

Assume $a_{j,0}>0$, $c_{j,0}=a_{j,0}/\sqrt3$, and $f_0<Y$. For gradient flow, and for simultaneous GD with

$$
A_*=(\sqrt3Y)^{1/4},\qquad
0<\eta\le \frac1{32A_*^6},
\tag{8}
$$

the following hold:

1. The balance $c_j=a_j/\sqrt3$ persists, and every $a_j$ increases while $h>0$.
2. The coefficient $f$ increases to $Y$ without overshoot; $L$ tends to $Z^2/2$.
3. Every slope stays at most $A_*$, and for all times or updates,

   $$
   P_\Gamma\le
   \min\left\{1,\frac{\sqrt3Y}{W\Gamma^4}\right\}.
   \tag{9}
   $$

4. If all initial slopes are equal, all limiting slopes equal

   $$
   a_\infty=(\sqrt3Y/W)^{1/4}.
   \tag{10}
   $$

Thus (9) excludes a prescribed population whenever its right-hand side is strictly below $p$, including while every neuron grows and an unresolved target component remains.

**Proof for flow.** The full gradients give

$$
\dot a_j=3c_ja_j^2h,\qquad \dot c_j=a_j^3h.
$$

On the proposed balance, $\dot c_j=\dot a_j/\sqrt3$. Consequently,

$$
\dot a_j=\sqrt3h a_j^3,\qquad
f=\frac1{\sqrt3}\sum_j a_j^4,\qquad
\dot h=-4h\sum_j a_j^6.
\tag{11}
$$

Starting with $h_0>0$, the last equation keeps $h$ positive, so $f\le Y$ and the slopes increase. The formula for $f$ bounds each slope by $A_*$ and prevents finite-time escape. Since $\lambda_0=4\sum_j a_{j,0}^6>0$, $h(t)\le h_0e^{-\lambda_0t}$, giving convergence of $f$ to $Y$. Finally, $W P_\Gamma\Gamma^4\le\sum_j a_j^4\le\sqrt3Y$ proves (9); symmetry and the limit of $f$ give (10).

**Proof for GD.** Both gradients are evaluated at the old state:

$$
a_j'=a_j+3\eta c_ja_j^2h,\qquad
c_j'=c_j+\eta a_j^3h.
$$

Substitution shows that $c_j'=a_j'/\sqrt3$ whenever $c_j=a_j/\sqrt3$. This is exact preservation for this special balance, not a general assertion that GD preserves flow invariants. Thus

$$
a_j'=a_j(1+\eta\sqrt3h a_j^2).
\tag{12}
$$

Suppose $f\le Y$. Then $a_j\le A_*$ and $\eta\sqrt3h a_j^2\le\eta A_*^6\le1/32$, so $a_j\le a_j'\le2a_j$. By the mean value theorem on $\sum_j a_j^4/\sqrt3$,

$$
4\eta h\sum_j a_j^6
\le f'-f
\le32\eta h\sum_j a_j^6
\le32\eta h A_*^6
\le h.
\tag{13}
$$

The penultimate bound uses $\sum_j a_j^6\le A_*^2\sum_j a_j^4\le A_*^6$. Hence $f'\le Y$, closing the induction, and

$$
0\le h_{n+1}\le(1-\eta\lambda_0)h_n.
$$

Under (8), $0<\eta\lambda_0\le1/8$. This proves convergence and the same population and symmetric-limit claims.

The step bound is sufficient and deliberately conservative. The positivity and balance assumptions are essential to the proof: signed cancellations in a general tanh network would invalidate the inference from output coefficient to $\sum_j a_j^4$.

### Corollary 3.1: accumulated drive and unequal growth

For flow, let $\mathcal H(t)=\int_0^t h(u)\,du$. Equation (11) gives

$$
a_j(t)^{-2}=a_{j,0}^{-2}-2\sqrt3\,\mathcal H(t).
\tag{14}
$$

For $a_{j,0}<\Gamma$, threshold crossing requires, and along this flow occurs exactly when,

$$
\mathcal H(t)\ge
\mathcal H_{j,\Gamma}:=
\frac{a_{j,0}^{-2}-\Gamma^{-2}}{2\sqrt3}.
\tag{15}
$$

The drive is finite because $h$ decays exponentially. It is fixed self-consistently: $\mathcal H_\infty$ is the unique value below the first pole of (14) for which

$$
\frac1{\sqrt3}\sum_j
\left(a_{j,0}^{-2}-2\sqrt3\,\mathcal H_\infty\right)^{-2}=Y.
\tag{16}
$$

The left side starts at $f_0<Y$, increases strictly, and diverges at the first pole, establishing uniqueness. If $\mathcal H_{j,\Gamma}>\mathcal H_\infty$, that neuron never acquires the scale, even in the limit. Equality allows arrival only in the limit; a smaller required drive gives finite-time arrival. Already acquired neurons need no positive drive.

For GD, define $\mathcal H_n^\eta=\eta\sum_{k=0}^{n-1}h_k$. The elementary inequality $(1+x)^{-2}\ge1-2x$ for $x\ge0$ and (12) imply

$$
a_{j,n}^{-2}\ge a_{j,0}^{-2}-2\sqrt3\,\mathcal H_n^\eta.
\tag{17}
$$

Thus $\mathcal H_n^\eta\ge\mathcal H_{j,\Gamma}$ is necessary for acquisition. It is not the exact flow identity or a sufficient discrete criterion. The discrete drive is finite by the contraction in Proposition 3.

Since $h\le h_0$, either dynamics also requires physical time at least $\mathcal H_{j,\Gamma}/h_0$ for an initially unacquired particle. For initial slopes of order $W^{-1/2}$, a fixed larger threshold requires a nominal time at least of order $W$, when $h_0$ stays of order one. It may never be reached.

Larger particles have a smaller drive requirement. Their share of $\dot f$ in (11) is proportional to $a_j^6$. Moreover, for $a_i>a_j$, the ratio $a_i/a_j$ increases under both the flow and the map (12). Fitting by the larger particles therefore consumes the common residual while the smaller ones still require more drive.

For equal particles the growth is instead completely distributed. Starting at scale $W^{-1/2}$, their limiting scale is $W^{-1/4}$ for fixed $Y$: substantial relative growth that remains small in absolute size. This illustrates how population growth and incomplete scale acquisition can coexist.

**Interpretation and limits.** The surrogate establishes depletion during joint fitting, deterministic delay, and limited acquisition under a shared residual. With $Z\ne0$, loss remains above zero even after that driving signal is exhausted. It does not claim that the missing tanh directions are exactly inaccessible, that D34 remains on the special balance, or that these formulas predict its gamma values. The proposed feedback in which early large neurons reduce the drive available to others is an exact mechanism here and a hypothesis for the sine trajectories. It has not been isolated by the current D34 evidence. No conclusion that partial fitting constitutes successful geometry acquisition follows from this example.

## 6. Population flux permits revival and isolated escape

The preceding models explain why velocity can weaken. A population statement also needs to measure where that velocity carries mass.

### Proposition 4: conditional exclusion through a transition band

Choose $0<\gamma_0<\Gamma$ and a smooth function $\psi(a)\in[0,1]$, zero for $|a|\le\gamma_0$, one for $|a|\ge\Gamma$, and nondecreasing in $|a|$. Define $M_\psi(t)=\int\psi(a)\,d\rho_t$. Then

$$
P_\Gamma(t)\le M_\psi(t),\qquad
\dot M_\psi(t)=\int\psi'(a)v_a\,d\rho_t.
\tag{18}
$$

Suppose a nonnegative function $E(t)$ bounds the integral on $[s,T]$. If

$$
M_\psi(s)+\int_s^T E(u)\,du<p,
\tag{19}
$$

then $P_\Gamma(t)<p$ throughout that interval.

For actual GD, let

$$
\Delta M_{\psi,n}
=\int[\psi(a+\eta v_{a,n})-\psi(a)]\,d\rho_n.
\tag{20}
$$

If nonnegative $E_n$ satisfy $\Delta M_{\psi,n}\le E_n$ and $M_{\psi,s}+\sum_{n=s}^{N-1}E_n<p$, the same exclusion holds at every update through $N$.

**Proof.** The indicator of $|a|\ge\Gamma$ is bounded above by $\psi$. Apply the weak transport identity to $\psi$, or the pushforward in (2), and integrate or telescope. Nonnegative upper bounds make the terminal allowance valid for every intermediate time.

These statements apply to empirical measures, so no density or continuum initialization is assumed. In flow, only the transition band contributes. In GD, the complete step matters: a particle can cross the band even if $\psi'(a)=0$ at its old position.

The flux can stay small while forces revive elsewhere, while trajectories reverse, or while a small mass escapes. Its initial allowance $M_\psi(s)$ includes some mass below $\Gamma$, so a wider band is not automatically a sharper bound. A retrospective flux measurement is informative but does not independently predict its future envelope.

For finite neurons, the flow integral in (18) is

$$
-\frac1W\sum_j\psi'(a_j)
\left[(J_{a,C}^Tz_C)_j+(T_ae_H)_j+(g_{a,\perp})_j\right].
\tag{21}
$$

Equation (21) connects the conditional barrier to the residual mechanism while retaining signs, cancellations, and location. A norm-only bound loses those distinctions.

The existing path theorem uses the exact distance $\mathcal D_{p,\Gamma}(a_s)$ to the acquired-population set and the total slope path $\eta\sum_n\|g_{a,n}\|$. The reported median paths over 20k–600k are 4.242, 5.809, and 0.009432 for sine, degree 3, and degree 9, compared with distances about 12.8 for the 10% event at 3.2. Every original seed satisfies the exclusion individually. For degree 3, even the measured slope-energy bound is too loose, while the path works. This is why total loss decrease alone does not explain the population barrier.

## 7. What these results contribute to the mechanism account

The affine calculation identifies exact loss of access to a remaining residual. The tanh bounds explain how weak nonlinear coupling can delay departure from the initial regime. The coupled surrogate demonstrates that ongoing fitting can exhaust a common driving coefficient before enough particles reach the specified scales, including with unresolved error. The transition-band result turns signed, localized motion into a conditional population exclusion.

For D34, the common explanation must allow several distinct late behaviors. Degree 9 has little force with which to leave the small-scale regime. Degree 3 regains motion broadly, but both its partial readout correction and its slope growth remain far from the intended precision and scale regime. Sine regains stronger and more uneven motion without broad acquisition. The measured readout effects can both remove and regenerate signal; a universal monotone competition rule would miss the observations.

The analytical examples establish possible mechanisms and identify quantities that distinguish them. They do not calibrate a single reduced equation to D34. Whether nonlinear drift eventually exposes the hard target, whether an early escaping group depletes the remaining drive, and whether a population flux bound can be predicted beyond an observed window remain open. An improved readout fit at the observed small gammas does not resolve those questions or the failure to reach precision.

## Sources and evidence scope

- [Barrier walkthrough](d34_barrier_theorem_walkthrough.md): population distance, effective-force decomposition, coarse tracking, and exact GD accounting.
- [Transport theory](d34_transport_scale_barrier.md): physical-width normalization, nonlinear layer-balance defect, and formal width powers.
- [Signal recovery](../results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/README.md): early attribution, signed coarse and remaining-residual motion, per-neuron travel, and original-seed exclusions.
- [Transport evidence](../results/checkpoint_D_optimizers/expD34_readout_race/transport_barrier/README.md): actual-state effective forces, generated lower-mode contributions, forecasts, and width/rate checks.
- [Fixed-geometry and continuation evidence](../results/checkpoint_D_optimizers/expD34_readout_race/useful_slopes/README.md): fresh-readout errors, crossed slope/bias diagnostics, and matched frozen-block continuations. The numerical comparisons are used here as measurements of partial fitting and force changes; their stronger interpretation as successful acquisition is not adopted.

Propositions 1–4 and their corollaries are derived in this note. Their proofs, rather than numerical trajectories of the surrogate, establish their claims. The empirical figures and tables in the linked reports describe the tested finite training windows and retain their own numerical controls; none is promoted here to a uniform future envelope or an initialization-only theorem.

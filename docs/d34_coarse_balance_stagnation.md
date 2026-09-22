# Conditional scale stagnation near coarse balance

Why can scale acquisition remain weak after the force from coarse disequilibrium has subsided? This question starts from a training state, leaving the timing of entry into that state for later. The aim is to explain persistence of poor scales despite substantial fine error. The [coupled-force exposition](d34_transport_scale_barrier.md#4-coupled-evolution-of-the-residual-and-slope-sensitivity) identifies the relevant feedback; here we develop a surrogate in which that feedback has a definite direction.

The refinement is to retain the coarse output alongside the cubic and ninth-degree components. Near coarse balance, shrinking a slope and increasing its readout can preserve the coarse output while reducing the network's unwanted cubic component. At sufficiently small scales, that loss reduction outweighs the benefit available from increasing the hard component. Here stagnation of acquisition permits continuing readout motion and slope contraction. We prove a conditional slope-contraction result for a symmetric sector of this surrogate, including simultaneous GD, and show that the local inward effective field also occurs for exact tanh in the same symmetric odd sector. An exact surrogate layer-energy identity survives arbitrary signed, heterogeneous particles. Establishing that these mechanisms control the actual D34 trajectories is the purpose of the proposed experiments; partial fitting remains far from the intended scale and precision regime.

**Notation. All output modes are orthonormal in the empirical mean inner product. Surrogate forces below are per particle unless a vector norm is displayed.**

| Symbol | Meaning |
|---|---|
| $W,a_j,c_j$ | Width, slope, and readout weight. |
| $z_C$, $J_{c,C}^Tz_C$ | Coarse tracking error and its contribution to the readout gradient in the full modal decomposition. |
| $C$, $M_c$ | Full coarse kernel and its readout-weight contribution. |
| $\varphi_1,\varphi_3,\varphi_9$ | Surrogate output coefficients. |
| $Y_1,Y_9$ | Positive coarse and hard target coefficients; the cubic target coefficient is zero. |
| $\alpha_3\ne0$, $\alpha_9>0$ | Fixed coefficients in the surrogate features. |
| $a,c$, $\beta=Wac$ | Common positive slope, common positive readout, and coarse output in the symmetric sector. |
| $z$, $F_a,F_c$ | Scalar coarse tracking error and effective slope/readout gradients in that sector. |
| $\mathcal H(a,\beta)$ | Polynomial determining the sign of the effective slope gradient. |
| $\bar a,\beta_*,\delta$ | Slope cap, lower coarse-output bound, and allowed relative tracking contribution in the conditional result. |
| $\eta$ | Common raw-coordinate GD step. |

## 1. What small readout disequilibrium actually controls

For equal rates, the exact readout decomposition is

$$
g_c=T_ce_H+q_c^z+g_{c,\perp},\qquad q_c^z=J_{c,C}^Tz_C.
\tag{S1}
$$

With the omitted contribution controlled, suppressing $q_c^z$ leaves readout evolution driven by the effective fine residual. The coarse balance incorporated into $T_c$ remains present. In particular, the raw coarse residual need not be zero, and the readout need not be stationary.

There is a quantitative issue before transferring this hypothesis to other parameter blocks. Let $J_C$ concatenate the coarse Jacobians of all trained parameters, so $C=J_CJ_C^T$, and let $M_c=J_{c,C}J_{c,C}^T$. If $M_c$ is positive definite, then

$$
\|J_C^Tz_C\|^2
\le \Omega_c^2\|q_c^z\|^2,\qquad
\Omega_c^2=\lambda_{\max}(M_c^{-1/2}CM_c^{-1/2}).
\tag{S2}
$$

**Proof.** The two squared force norms are $z_C^TCz_C$ and $z_C^TM_cz_C$. Apply the Rayleigh-quotient bound after substituting $M_c^{1/2}z_C$. Similarly, the coarse contribution to fine-residual evolution obeys

$$
\|K_{HC}z_C\|
\le\|K_{HC}M_c^{-1/2}\|\,\|q_c^z\|.
$$

Thus small readout disequilibrium controls the other corrections only with the associated amplification factors. If $M_c$ is singular or poorly conditioned, small $q_c^z$ alone supplies little information about some coarse directions. Including the output bias in the readout block gives a different matrix and should be reported separately. The existing evidence for a small tracking contribution to slopes does not itself establish the readout condition.

For the one-coarse-mode symmetric model below, $M_c=Wa^2$, $C=W(a^2+c^2)$, and $\Omega_c^2=1+c^2/a^2$. This already anticipates why a growing readout-to-slope ratio matters.

## 2. Retain the coarse output in the surrogate

Take fixed modes $q_1,q_3,q_9$ and define

$$
F=\varphi_1q_1+\varphi_3q_3+\varphi_9q_9,\qquad
\varphi_k=\alpha_k\sum_j c_ja_j^k,
\qquad \alpha_1=1,
$$

$$
y=Y_1q_1+Y_9q_9,\qquad
e_1=\varphi_1-Y_1,\quad e_3=\varphi_3,\quad e_9=\varphi_9-Y_9,
\qquad L=\tfrac12(e_1^2+e_3^2+e_9^2).
\tag{S3}
$$

This is an odd-output surrogate. A fitted constant mode can be added independently. The model omits hidden biases and the other nonlinear modes; it is not a Taylor truncation of the full tanh model. Keeping $\alpha_3$ explicit allows its sign to differ from $\alpha_9$, as it does for the leading cubic and ninth-degree terms of tanh. No numerical gamma threshold is transferred from this model to D34.

Ordinary equal-rate gradient flow trains both parameter blocks:

$$
\dot a_j=-c_j(e_1+3\alpha_3a_j^2e_3+9\alpha_9a_j^8e_9),
\qquad
\dot c_j=-a_j(e_1+\alpha_3a_j^2e_3+\alpha_9a_j^8e_9).
\tag{S4}
$$

Simultaneous GD uses these same gradients at the old state. No readout fitting solve, parameter projection, or rebalancing step is part of that algorithm.

### Exact coarse balancing in the symmetric sector

Suppose all neurons have the same $a>0$ and $c>0$. Symmetry is preserved by both flow and GD. Write $\beta=Wac$, so

$$
\varphi_3=\alpha_3\beta a^2,\qquad
\varphi_9=\alpha_9\beta a^8.
$$

The coarse kernel is $C=W(c^2+a^2)$. Computing its cross-kernel with each fine mode gives

$$
B_3=\frac{\alpha_3a^2(3c^2+a^2)}{c^2+a^2},\qquad
B_9=\frac{\alpha_9a^8(9c^2+a^2)}{c^2+a^2},\qquad
z=e_1+B_3e_3+B_9e_9.
\tag{S5}
$$

This is the same coarse-balance construction used in the actual-network theory. Substituting $e_1=z-B_3e_3-B_9e_9$ into the gradients gives

$$
g_a=cz+F_a,\qquad g_c=az+F_c,
$$

$$
F_a=\frac{2ca^6}{c^2+a^2}\,\mathcal H(a,\beta),\qquad
F_c=-\frac ca F_a,
\qquad
\mathcal H(a,\beta)=\alpha_3^2\beta-4\alpha_9Y_9a^4
+4\alpha_9^2\beta a^{12}.
\tag{S6}
$$

For example, $3\alpha_3a^2-B_3=2\alpha_3a^4/(c^2+a^2)$ and $9\alpha_9a^8-B_9=8\alpha_9a^{10}/(c^2+a^2)$, which yield the slope formula directly. The corresponding readout differences yield $F_c=-(c/a)F_a$.

The effective directions preserve the coarse output to first order: $cF_a+aF_c=0$. In fact,

$$
\dot\beta=-W(c^2+a^2)z.
\tag{S7}
$$

Consequently, at a state with $z=0$ and $\mathcal H>0$, slopes contract and readouts increase while the coarse output is instantaneously unchanged. This behavior is generated by the effective fine force, including its coarse-balance correction.

Equation (S7) does not make $z=0$ an invariant set. The balance changes as the fine residual and parameters move. For GD, even an update starting at $z=0$ changes the coarse output by a quadratic term:

$$
\beta'-\beta=-\eta W(c^2+a^2)z+\eta^2Wg_ag_c.
$$

The result below therefore treats small tracking force as a condition on the interval, rather than enforcing it through a modified training algorithm.

## 3. A conditional contraction result

The sign in (S6) supplies a state-dependent criterion. In particular,

$$
4\alpha_9Y_9a^4<\alpha_3^2\beta
\quad\Longrightarrow\quad \mathcal H(a,\beta)>0.
\tag{S8}
$$

The cubic term enters through its square: its contraction effect does not depend on the sign convention for $q_3$. The hard target supplies the competing negative term in $\mathcal H$, but it is multiplied by $a^4$. This identifies a region in parameter space where the effective force points toward smaller scales, without predicting when training enters that region.

**Proposition.** Fix $\bar a>0$, $\beta_*>0$, and $0\le\delta<1$ such that

$$
4\alpha_9Y_9\bar a^4<\alpha_3^2\beta_*.
$$

Start the symmetric surrogate at $0<a_s\le\bar a$ and $c_s\ge a_s$. On an interval where $\beta\ge\beta_*$ and the readout tracking force satisfies

$$
|az|\le\delta\frac ac F_a,
\tag{S9}
$$

the slope is nonincreasing and the readout is increasing under flow. For simultaneous GD the same conclusion holds at every update for which these conditions and

$$
\eta(1+\delta)F_a<a
\tag{S10}
$$

hold at the old state. No particle acquires a threshold above $\bar a$ while the conditions persist.

**Proof.** As long as $a\le\bar a$ and $\beta\ge\beta_*$, (S8) ensures $F_a>0$. Equation (S9) implies $|cz|\le\delta F_a$, hence

$$
(1-\delta)F_a\le g_a\le(1+\delta)F_a.
$$

Thus $\dot a=-g_a<0$. Also $g_c=-(c/a)F_a+az<0$ whenever $c\ge a$, because $|az|\le\delta(a/c)F_a<(c/a)F_a$. It follows that $c$ increases, so $c\ge a$ is preserved. A first-exit argument prevents $a$ from increasing through $\bar a$. Moreover, $F_a\le2W(\alpha_3^2+4\alpha_9^2\bar a^{12})a^7$ in this region, so the slope cannot reach zero in finite flow time. In GD, (S10) ensures $0<a'=a-\eta g_a<a$, and the same bound on $g_c$ gives $c'>c$. Induction proves the claim until a stated regime condition fails. The slope cap is a conclusion, not an assumption imposed throughout the trajectory.

**Remaining-error corollary.** If the interval also satisfies $\beta\le\beta^*$ and $\alpha_9\beta^*\bar a^8\le Y_9/2$, then $|e_9|=Y_9-\alpha_9\beta a^8\ge Y_9/2$. Thus the contraction result can coexist with a uniformly substantial hard residual. This additional coarse-output upper bound is a condition to control or measure.

The conditions on $\beta$ and tracking are substantive. This proposition does not prove that coarse tracking remains accurate indefinitely, nor that arbitrary D34 particles become positive and identical. It establishes a contraction mechanism inside an explicit sector of an ordinary-GD surrogate.

### Why the readout condition has this particular scale

In this sector $q_c^z=az$ and $q_a^z=cz$, so $q_a^z=(c/a)q_c^z$. To keep tracking smaller than a fraction of the slope force, the readout tracking force must therefore be smaller by the compensating factor $a/c$, as in (S9). A small ratio $|q_c^z|/|F_c|$ alone is insufficient when $c/a$ is large: the corresponding slope ratio is

$$
\frac{|q_a^z|}{F_a}
=\frac{c^2}{a^2}\frac{|q_c^z|}{|F_c|}.
\tag{S11}
$$

This is an experimentally testable qualification of the proposed regime. Report absolute forces and the transfer factor along with readout-relative smallness. Near cancellation, a ratio with a nearly zero denominator should be marked unresolved rather than stabilized into an apparent finding.

### The loss calculation behind the contraction

At fixed coarse output $\beta$, the fine loss as a function of scale is

$$
V_\beta(a)=\tfrac12\left[\alpha_3^2\beta^2a^4
+(\alpha_9\beta a^8-Y_9)^2\right],\qquad
V_\beta'(a)=2\beta a^3\mathcal H(a,\beta).
\tag{S12}
$$

Reducing the unwanted cubic output saves loss at order $a^4$. The leading benefit of fitting the ninth-degree target appears at order $a^8$. At sufficiently small scales, flattening therefore decreases loss even as it reduces the already weak representation of the hard target. Large remaining hard error is compatible with an inward scale force.

The readout must vary as $c=\beta/(Wa)$ along this coarse-output level set. Its induced Euclidean metric is $W(1+c^2/a^2)$, and

$$
F_a=\frac{V_\beta'(a)}{W(1+c^2/a^2)}.
$$

This reproduces (S6) at $z=0$ and explains the direction using the loss itself. The level-set calculation is an interpretation of the instantaneous effective field. Actual GD uses (S4) and the evolving $z$; it does not solve a constrained optimization problem at each update.

Keeping $\beta$ fixed in an idealized projected flow would give $\dot a\sim-2\alpha_3^2Wa^7$ as $a\to0$, and hence $a\sim(12\alpha_3^2Wt)^{-1/6}$. This illustrative slow contraction requires fixed $\beta$ and $c=\beta/(Wa)$, with readouts growing without bound. It is not an asymptotic claim for ordinary GD or tanh. The conditional sign result does not depend on this idealization.

## 4. The local contraction mechanism survives exact tanh

The ordering of the two loss contributions can be checked without a polynomial activation surrogate. Retain identical positive slopes and readouts, set hidden biases to zero, and use a symmetric training grid on which the degree-9 polynomial mode is resolved. Let $q_1=x/\sigma$, where $\sigma=\|x\|_m$, and let $q_9$ be normalized and orthogonal to all lower-degree polynomials. An already fitted constant target can be represented by the output bias. The remaining target is $Y_1q_1+Y_9q_9$.

Define

$$
\psi_1(a)=\langle q_1,\tanh(ax)\rangle_m,\qquad
w(a)=\frac{P_H\tanh(ax)}{\psi_1(a)},
$$

where $P_H$ removes the constant and linear components and retains their full empirical orthogonal complement. On a level set of coarse output $\beta=Wc\psi_1(a)>0$, the fine loss is exactly

$$
V_\beta^{\tanh}(a)=\tfrac12\|\beta w(a)-Y_9q_9\|_m^2.
$$

**Local direction result.** For fixed $\beta>0$ and finite $Y_9$, the derivative of this fine loss is positive for all sufficiently small $a>0$. The small-scale cap can be chosen uniformly when $\beta$ ranges over a fixed compact positive interval. Hence the exact effective slope force is inward in this symmetric sector. This statement concerns the force after coarse balancing; the actual slope direction still requires control of the tracking contribution.

**Proof.** The Taylor expansion and empirical orthogonality give

$$
\psi_1(a)=\sigma a+O(a^3),\qquad
w(a)=-\frac{a^2}{3\sigma}P_Hx^3+O(a^4),
\qquad \langle q_9,w(a)\rangle_m=O(a^8).
$$

The vector $P_Hx^3$ is nonzero on the stated grid. With $\kappa_3^2=\|P_Hx^3\|_m^2/(9\sigma^2)>0$, analyticity of the finite-grid expressions therefore yields

$$
\frac{dV_\beta^{\tanh}}{da}
=2\beta^2\kappa_3^2a^3
+O(\beta^2a^5)+O(\beta|Y_9|a^7)>0
$$

for sufficiently small positive $a$. Along the coarse-output level set,

$$
\frac{dc}{da}=-\frac{c\psi_1'(a)}{\psi_1(a)},\qquad
F_a^{\tanh}
=\frac{dV_\beta^{\tanh}/da}
{W\left[1+(c\psi_1'/\psi_1)^2\right]},\qquad
F_c^{\tanh}=-\frac{c\psi_1'}{\psi_1}F_a^{\tanh}.
$$

These are the Euclidean projected-gradient components tangent to the coarse-output level set, equivalently the effective forces from coarse elimination. Since $\psi_1,\psi_1'>0$ at positive $a$, they give slope contraction and readout growth. The hidden biases remain zero in this odd sector by symmetry.

The corresponding tracking-force ratio is $q_a^z/q_c^z=c\psi_1'/\psi_1\sim c/a$. Thus the need to scale the readout tracking tolerance also survives exact tanh. Once that contribution is smaller than the inward effective slope force, the actual flow points inward; GD additionally requires a step small enough to preserve the slope's positive sign.

This result establishes a local direction for exact tanh, with all its higher modes retained. It supplies neither an explicit D34 gamma boundary nor a persistence theorem for random heterogeneous neurons, and does not establish stability against perturbations that split identical neurons. In transport language, the symmetric population moves inward when its tracking correction is controlled. Extending that inward field to a region containing the observed joint parameter distribution is the next substantive step.

## 5. What survives heterogeneous particles

The symmetric sector makes the direction explicit, but real D34 states have heterogeneous signed parameters. One exact identity in the same three-mode surrogate requires neither positivity nor symmetry. Define

$$
I=\tfrac12\left(\sum_j a_j^2-\sum_j c_j^2\right).
$$

Using (S4), the coarse contributions cancel between the two layers, giving

$$
\dot I=-2\varphi_3^2+8(Y_9-\varphi_9)\varphi_9.
\tag{S13}
$$

Indeed, the slope derivative of mode $k$ contributes $-k e_k\varphi_k$ to the slope energy, while its readout derivative contributes $-e_k\varphi_k$ to the readout energy. Their difference is $-(k-1)e_k\varphi_k$. The factor is zero for the coarse mode, two for the cubic mode, and eight for the ninth-degree mode.

For simultaneous GD, the exact counterpart is

$$
I'-I=\eta\left[-2\varphi_3^2+8(Y_9-\varphi_9)\varphi_9\right]
+\frac{\eta^2}{2}(\|g_a\|^2-\|g_c\|^2).
\tag{S14}
$$

When the hard-mode contribution is small, correction of the generated cubic output decreases hidden-layer energy relative to readout energy. This supports investigating flattening accompanied by readout growth. It does not alone prove decreasing slope energy: both energies could increase with the readout increasing faster. Population stagnation in heterogeneous tanh networks still requires signed slope motion, its distribution, and the terms omitted by this surrogate.

## 6. Experiments implied by the theory

The immediate objective is to test the direction mechanism and the conditions supporting it. The starting states are observed checkpoints; no experiment in this first round needs to explain entry from initialization. Use degree 9 first, with all five existing seeds. The current theory does not justify selecting only seeds with contraction or declaring a fixed gamma threshold necessary for every representation.

**Proposed sequence. These experiments have not been run as part of this note.**

| Stage | Measurement or comparison | Scientific question |
|---|---|---|
| Existing-state regime audit | At joint-GD updates 20k, 100k, and 600k, reconstruct readout, slope, and bias tracking forces; $M_c$, $\Omega_c$; retained and omitted fine forces. | Is readout disequilibrium actually negligible at the scale relevant to slope motion? |
| Direction and mode audit | Decompose the effective force into generated lower-mode contributions and the hard mode. Measure $-a^TF_a$, mean signed gamma velocity, and each neuron's outward force. | Does correcting generated error drive contraction, or do heterogeneous contributions change the sign? |
| Autonomous short forecasts | From the unmodified 20k and 100k states, compare full tanh GD with exact-tanh residual truncations retaining modes $\{0,1,3,9\}$ and $\{0,\ldots,9\}$. | Does a small modal system preserve the observed coupled evolution over the next interval? |
| State-conditioned force probes | At the same parameters, vary only the hard-target coefficient in the effective-force diagnostic. | How large must hard-target forcing be to reverse the current radial direction near coarse balance? |

The first audit uses [retained checkpoint states](../results/checkpoint_D_optimizers/expD34_readout_race/useful_slopes/curated/states) and the existing modal Jacobians. Deduplicate unchanged-GD states across fork archives. Record $q_c^z$ separately from the output-bias contribution, and measure the other blocks directly even if (S2) is well conditioned. The integral of a small force can still matter over a long interval, so short continuations must accumulate its contribution as well as sample its magnitude.

The direction audit should retain the original mode labels and signed projections. Under the degree-9 target, the retained lower-mode coefficients are generated by the network itself. Their individual force norms do not add: test whether their combined effective force contracts the slope energy, and whether mean gamma and the population distribution agree with that diagnosis. Compare the three-mode balance in (S13) with the full tanh layer-balance identity in the [technical note](d34_transport_scale_barrier.md#2-layer-balance-exposes-the-source-of-readout-amplification). Biases and higher-order terms are measured discrepancies, not presumed negligible corrections.

For the short forecasts, use the existing width 177, 2,048 training midpoints, random-readout checkpoint arrays, FP64, and raw-coordinate step $\eta=0.002$. Forecast 20,000 updates, or physical time 40, from each of the two starting checkpoints and five seeds. The three arms give 30 continuations. Use a matched half-step control for seed 0 at both starts in all three arms, giving six additional numerical controls. These are proposed bounded runs, not launched jobs.

Each modal surrogate evaluates exact tanh features and its own retained residual at its own current parameters. All four parameter blocks train simultaneously. This retains the exact activation and observed starting geometry while testing the residual simplification suggested by the analytical model. It also makes the missing even modes and modes 5 and 7 explicit: the four-mode forecast can fail, and the ten-mode forecast tests whether those omissions explain the failure. No arm imports later true parameters, freezes the kernel, refits readouts, or imposes $z_C=0$.

Record force and motion diagnostics every 200 updates, together with every-step accumulated slope path and positive/negative gamma travel. Evaluate initial and terminal full parameters, coarse output, hard residual, and slope distributions. The initial same-state comparison and the autonomous forecast answer different questions; report both. Compare surrogate discrepancies with half-step discrepancies and with the signed force being explained. Near a direction reversal, report absolute discrepancies and an unresolved sign if numerical or modal errors are comparable to that force. No empirical accuracy cutoff is promoted here to a theorem hypothesis.

The hard-target probe is a diagnostic at fixed parameters. The effective force is affine in the varied target coefficient, so its slope-energy projection has a directly computable zero whenever the hard-mode projection is nonzero. In the symmetric analytical model this zero is

$$
Y_9^{\rm crit}(a,\beta)
=\frac{\alpha_3^2\beta}{4\alpha_9a^4}
+\alpha_9\beta a^8.
\tag{S15}
$$

The leading $a^{-4}$ dependence is a conditional prediction of that sector. In actual tanh states, compute the corresponding zero from the measured effective modal forces and test whether the predicted competition is present. Changing the target coefficient also changes the coarse tracking error of the raw gradient; the probe must report this change. It does not predict an actual-GD direction after a target change unless the required tracking condition remains satisfied.

All of these are optimization-mechanism measurements on the training grid. There is no hyperparameter selection or statistical generalization claim. Full approximation errors can accompany the motion diagnostics, but partial error reduction is not the success criterion for scale acquisition. A result in which the readout condition fails, generated modes do not contract scales, or the modal forecast misses the dynamics narrows or rejects this mechanism for the tested regime.

## Derivation checks and evidence scope

The formulas and propositions in this note are derived above. Bounded numerical checks verified 500 symmetric force decompositions and constrained-loss derivatives, 300 heterogeneous layer-energy identities with their exact GD increments, 400 readout-force transfer inequalities, and 300 phase boundaries with 600 perturbed GD contraction steps. The exact-tanh effective-force and metric identities were also checked on 257 symmetric midpoints at $a=0.2,0.1,0.05,0.025$, with $W=7$, $\beta=0.7$, and $Y_9=0.866$; the normalized loss derivative approached its positive leading coefficient as predicted. These checks support transcription and algebra. Applicability to heterogeneous D34 states remains untested.

A further ordinary-GD check used $W=7$, $a_0=0.3$, $c_0=0.35$, $\alpha_3=\alpha_9=1$, $Y_9=0.866$, and $\eta=0.002$. The fixed coarse target was chosen once as $Y_1=\beta_0+B_{3,0}e_{3,0}+B_{9,0}e_{9,0}$, making $z_0=0$. No later rebalancing was applied. Over 10,000 updates, $a$ decreased to 0.274081, $c$ increased to 0.384617, and $\beta$ changed from 0.735 to 0.737913. The largest $|cz|/F_a$ was 0.02812, and every step satisfied the contraction and positivity checks. This is a verification example inside the surrogate, not a D34 experiment or evidence for a universal stagnation mechanism.

The advance over the earlier two-mode proposal is a coarse-retaining model with an explicit inward-force region and a measurable tracking tolerance, supported by a local exact-tanh calculation in the symmetric sector. The remaining empirical question is whether actual stalled states occupy a comparable region once heterogeneity and biases are retained.

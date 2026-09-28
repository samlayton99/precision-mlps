# Transport, residual modes, and a finite-time gamma barrier

Why can noiseless gradient descent reduce the error and revive slope motion while remaining far from the intended gamma and precision regime? This extension of the [D34 scale-acquisition argument](d34_scale_acquisition_theory.md) studies that question for D34's random nonzero readouts and equal learning rates. The central mechanism is a feedback: fitting changes the residual that drives slope motion, while parameter motion changes sensitivity to that residual. Acquisition requires this feedback to sustain enough outward motion across the population. Partial error correction through better readout fitting does not establish that the required geometry has been acquired.

The evidence motivates centering the post-transient theory on the effective fine force. Section 4 develops its coupled evolution and a cubic surrogate. The [conditional stagnation note](d34_coarse_balance_stagnation.md) gives an example-led account of the mechanisms and the new frozen-readout scale check. Its [detailed persistence argument](d34_coarse_balance_stagnation_details.md#9-from-a-force-decomposition-to-a-persistence-prediction) resolves the degree-9 regime into slow correction of generated quadratic/cubic errors and a weak hard-mode drive. It also derives a full-GD confinement bound from a local hard-mode loss floor. The results are conditional on specified starting states; they do not establish entry into the regime from initialization. Population statements always concern a specified threshold and horizon; the strongest local bound excludes even an isolated neuron crossing scale 1 on its stated interval.

**Notation. Physical parameter coordinates and the empirical mean inner product are used throughout.**

| Symbol | Meaning |
|---|---|
| $W,m$ | Physical neuron count and training-sample count. |
| $a,b,c,d$, $\gamma=\lvert a\rvert$ | Slopes, hidden biases, readout weights, output bias, and slope magnitude. |
| $\rho$, $r$, $R=\Vert r\Vert_m$ | Joint parameter probability measure, prediction residual, and full residual RMS. |
| $\eta$, $t=n\eta$, $\kappa$ | GD step, physical training time, and readout/geometry learning-rate ratio; the coupled exposition uses $\kappa=1$. |
| $e_C,e_H,z_C$ | Constant/linear residual coefficients, retained fine coefficients, and coarse tracking error. |
| $T_a,G_a=T_a^TT_a,S$ | Effective slope-force map, its Gram matrix, and the effective kernel for fine-residual fitting. |
| $F_a=T_ae_H$, $A_a=\Vert F_a\Vert$ | Effective slope gradient and its norm; its contribution to velocity is $-F_a$. |
| $R_H,u_H$ | Fine-residual norm and unit direction, $e_H=R_Hu_H$. |
| $\mu_a^{\rm eff},\lambda^{\rm eff}$ | Squared slope coupling and fitting rate per unit squared fine residual. |
| $q_a,\varepsilon_H$ | Corrections to the effective slope gradient and fine-residual evolution. |
| $p,\Gamma$ | Required population fraction and slope threshold. |

The [educational walkthrough and metrics guide](d34_barrier_theorem_walkthrough.md) derives the population-barrier theorem step by step, gives an explicit constant-bound corollary for GD, and maps the mathematical quantities to the exported diagnostics.

The distinction between an identity, a numerical prediction, and a proved enclosure is essential. The exact transport equation includes every finite-network trajectory. A quadrature solution initialized from the probability law instead of the realized particles is a separate approximation. A finite residual basis is another approximation. Agreement between these predictions and training is evidence for a mechanism, but is not a uniform theorem about all seeds or arbitrarily late training.

## Quantities and normalization

Let $W$ be the physical network width, $m$ the number of training samples, $z=(a,b,c)$ a neuron's physical parameters, $d$ the output bias, and $\rho$ a probability measure on neuron parameters. Write $\langle\cdot,\cdot\rangle_m$ for the empirical mean inner product. The readout/geometry learning-rate ratio is $\kappa$; the main claim concerns $\kappa=1$. Define

$$
F_\rho(x)=d+W\int c\tanh(ax+b)\,d\rho(z),\qquad
r=F_\rho-y,\qquad L=\tfrac12\|r\|_m^2.
$$

The slope is $\gamma=|a|$. A population threshold is specified by $(p,\Gamma)$, meaning that at least fraction $p$ has $\gamma\ge\Gamma$. The empirical residual basis $q_0,\ldots,q_K$ is orthonormal on the training grid; $q_0,q_1$ span the constant and linear functions. Denote its orthogonal projector by $P_K$, modal residual by $e$, and omitted residual by $r_\perp=(I-P_K)r$. Subscripts $C$ and $H$ denote coarse modes $0,1$ and the remaining retained modes, respectively.

## 1. A transport equation with the correct D34 velocity

Put $u=ax+b$, $h=\tanh u$, and $s=\operatorname{sech}^2u$. The characteristic velocity is

$$
v_a=-c\langle r,xs\rangle_m,\qquad
v_b=-c\langle r,s\rangle_m,\qquad
v_c=-\kappa\langle r,h\rangle_m,
\qquad \dot d=-\kappa\langle r,1\rangle_m.
\tag{1}
$$

The continuity equation is

$$
\partial_t\rho+\nabla_z\cdot(\rho v)=0.
\tag{2}
$$

For $\rho=W^{-1}\sum_j\delta_{z_j}$, (1) is exactly the D34 gradient-flow ODE. Euler discretization with step $\eta$ is simultaneous D34 GD. There is no infinite-width approximation in this assertion. For any differentiable test function $\psi$, the weak equation is $d\int\psi\,d\rho/dt=\int\nabla\psi\cdot v\,d\rho$; applying the chain rule to each particle proves it.

For weighted quadrature $\rho=\sum_i w_i\delta_{z_i}$, $\sum_iw_i=1$, each node still follows (1). Its Euclidean loss gradient has an extra factor $Ww_i$. Dividing by this factor is necessary to approximate transport: replacing $W$ by the number of quadrature nodes would change both the model and its dynamics. The dissipation identity is

$$
-\dot L=W\int\left(g_a^2+g_b^2+\kappa g_c^2\right)d\rho+\kappa g_d^2,
\qquad v=-(g_a,g_b,\kappa g_c).
\tag{3}
$$

Equivalently, with $D=\operatorname{diag}(1,1,\kappa)$, the transport mobility is $v=-W^{-1}D\nabla_z(\delta L/\delta\rho)$. Importing a conventional Wasserstein gradient-flow time scale without this factor would change D34's dynamics. For compact initial support and $|x|\le1$, dissipation also gives $R(t)\le R(0)$, $|\dot c|\le\kappa R(0)$, and $|\dot a|,|\dot b|\le |c|R(0)$. These bounds keep support finite on every finite interval and give the usual characteristic well-posedness route. They are too loose for the desired barrier: the direct support estimate contains $\kappa R(0)^2t^2/2$, already of order $10^6$ at the experimental horizon when $R(0)$ is order one. Existence of the PDE is therefore much easier than a useful gamma bound.

Transport descriptions of two-layer learning have established precedents in [Mei, Montanari, and Nguyen](https://arxiv.org/abs/1804.06561) and [Chizat and Bach](https://arxiv.org/abs/1805.09545). Their general framework does not by itself supply a gamma barrier for this initialization or scaling. Equations (1)–(3) are derived here in D34's coordinates.

The law experiment starts from independent uniforms on $[-\sqrt{6/(W+1)},\sqrt{6/(W+1)}]^3$. It averages away realized finite-seed fluctuations. Tensor Gauss–Legendre quadrature at orders 8, 12, and 16 tests numerical convergence of this law experiment, not convergence to a particular random seed.

### Why the literal infinite-width limit can lose the relevant phenomenon

Set $(A,B,C)=\sqrt W(a,b,c)$. Then

$$
F=d+\sqrt W\int C\tanh((Ax+B)/\sqrt W)\,d\rho
=d+\int C(Ax+B)\,d\rho+O(W^{-1}),
\tag{4}
$$

provided the corresponding higher moments stay controlled on the time interval. The leading rescaled characteristic equations are

$$
\dot A=-C\langle r,x\rangle_m,\quad
\dot B=-C\langle r,1\rangle_m,\quad
\dot C=-\kappa\langle r,Ax+B\rangle_m.
$$

Thus the formal fixed-time, infinite-width limit is affine. Its nonlinear correction need not remain negligible over a width-dependent escape time. A theorem proved only after taking this limit could suppress the very finite-width recovery we want to explain. The implemented law PDE keeps physical $W$ finite and the activation exact. Establishing a uniform-in-time moment bound is part of a theorem, not an assumption justified by the expansion itself.

## 2. Layer balance exposes the source of readout amplification

Along each characteristic, with $I=a^2+b^2-c^2/\kappa$,

$$
\dot I=2c\langle r,h-us\rangle_m.
\tag{5}
$$

For affine activation the defect vanishes. For tanh it does not have a fixed sign. Integrating (5) gives an exact explanation of how readout norm can outgrow hidden norm; it avoids simply assuming the readout remains small. In one GD step,

$$
I_{n+1}-I_n=2\eta c_n\langle r_n,h_n-u_ns_n\rangle_m
+\eta^2(g_{a,n}^2+g_{b,n}^2-\kappa g_{c,n}^2).
\tag{6}
$$

Both terms are accumulated at every update. For projected-residual training the same identities hold with $r$ replaced by $P_Kr$. A useful moment proof must bound the signed defect, or suitable moments that determine it, without inserting the observed small-slope trajectory as a premise. Bounding it only by absolute readout and residual norms is generally too loose at long horizons. Nonlinear tanh moments generate an unclosed hierarchy; the PDE organizes that hierarchy but does not remove it.

## 3. Which kernel quantity could explain a barrier?

For finite particles define modal Jacobians $J_a,J_b,J_c$ with entries, for example, $(J_a)_{kj}=\langle q_k,c_jxs_j\rangle_m$. For quadrature multiply the characteristic tangent columns by $\sqrt{Ww_j}$. Define $J_d$ similarly and

$$
K=K_a+K_b+K_c+K_d,
\quad K_a=J_aJ_a^T,\quad K_b=J_bJ_b^T,
\quad K_c=\kappa J_cJ_c^T,\quad K_d=\kappa J_dJ_d^T.
\tag{7}
$$

In the modal vector identities, $g_a$ means the finite-network slope gradient, or the weighted node vector $(\sqrt{Ww_i}\,g_a(z_i))_i$ for quadrature. Its squared norm is therefore $W\int g_a(z)^2d\rho$, consistent with (3). The raw characteristic force in (1) remains unweighted.

For the complete residual representation, $\dot e=-Ke$. With a finite basis the exact full-network equation has the omitted-mode forcing $f$:

$$
\dot e=-Ke+f,\qquad
f_k=-\langle q_k,\mathcal K r_\perp\rangle_m,
\tag{8}
$$

where $\mathcal K$ is the full empirical tangent operator. The implemented modal forecast evolves parameters using $P_Kr$ at its own current state. Consequently it evolves its kernel as well, and never reads a later true-network state. This is a finite residual representation, not a closed ODE in $e$ alone: the feature distribution remains part of the state.

The total kernel predicts residual relaxation, whereas $K_a$ identifies slope motion. Even in a centered affine model, swapping slope and readout vectors can preserve the output and total tangent kernel while changing the slope kernel. Residual contraction alone therefore does not imply geometry acquisition. Moreover, positive gradient norm does not specify outward direction or how motion is distributed across neurons.

### Eliminate the coarse residual without discarding its regeneration

Assume $K_{CC}$ is invertible on the interval in question. This is measured, without an arbitrary inverse regularizer. Define

$$
B=K_{CC}^{-1}K_{CH},\quad z_C=e_C+Be_H,\quad
S=K_{HH}-K_{HC}B,\quad
T_a=J_{a,H}^T-J_{a,C}^TB.
\tag{9}
$$

Then the following are exact algebraic/differential identities:

$$
g_a=J_{a,C}^Tz_C+T_ae_H+g_{a,\perp},
\qquad
\dot e_H=-Se_H-K_{HC}z_C+f_H,
\tag{10}
$$

$$
\dot z_C=-(K_{CC}+BK_{HC})z_C
+(\dot B-BS)e_H+f_C+Bf_H.
\tag{11}
$$

The coarse balance is $e_C\approx-Be_H$, rather than zero. $T_ae_H$ includes its regenerated contribution and its interference with the direct fine-mode slope force. Dropping $e_C$ instead produces the wrong reduced slope force. Equation (11) also shows why a small coarse residual or small $\dot K_{CC}$ alone is insufficient: a tracking bound must control the full forcing, including $\dot B$, $BS$, and omitted modes. The tracking matrix need not be symmetric, but the kernel structure supplies a useful metric estimate below.

There is also an exact blockwise interpretation of $S$. Define $T_I=J_{I,H}^T-J_{I,C}^TB$ for each parameter block, with the same transpose convention for the single bias column. Then

$$
S=T_a^TT_a+T_b^TT_b+\kappa T_c^TT_c+\kappa T_d^TT_d.
\tag{11a}
$$

To prove this, stack the rate-weighted Jacobian blocks into $J$. Expanding $(J_H^T-J_C^TB)^T(J_H^T-J_C^TB)$ and using $K_{CC}B=K_{CH}$ gives $S$. Thus the effective slope kernel $G_a=T_a^TT_a$ satisfies $0\preceq G_a\preceq S$. The ratio $e_H^TG_ae_H/(e_H^TSe_H)$ measures the slope share after coarse relaxation. This identifies a precise candidate for a readout-depletion argument: readout fitting can dominate the effective fine-mode dissipation while $G_a$ remains small in the current residual direction. Proving $S$ small alone does not establish this allocation, and a bound along the target direction is weaker than a uniform operator inequality. The identity also retains interference inside each $T_I$ instead of assigning independent positive forces to coarse and fine residuals.

If $\|U_C(t,s)\|\le M e^{-\alpha(t-s)}$ bounds that evolution operator and $b(t)=\| (\dot B-BS)e_H+f_C+Bf_H\|$, variation of constants gives

$$
\|z_C(t)\|\le M e^{-\alpha(t-s)}\|z_C(s)\|
+M\int_s^t e^{-\alpha(t-u)}b(u)\,du.
\tag{12}
$$

The evolution-operator hypothesis can itself be reduced to kernel quantities. Set $C=K_{CC}$, $Q=K_{CH}$, $A=C+C^{-1}QQ^T$, and $V=z_C^TCz_C/2$. Since $CA=C^2+QQ^T$, equation (11), with forcing $f_z$, gives

$$
\dot V=-z_C^T(C^2+QQ^T)z_C+\tfrac12z_C^T\dot C z_C+z_C^TCf_z.
\tag{12a}
$$

Define

$$
\alpha_C(t)=\lambda_{\min}(C)
-\tfrac12\lambda_{\max}(C^{-1/2}\dot C C^{-1/2}).
$$

Then $d\|z_C\|_C/dt\le-\alpha_C\|z_C\|_C+\|f_z\|_C$, interpreted by continuity at zero. In particular, if $c_-I\preceq C\preceq c_+I$ and $\alpha_C\ge\alpha>0$ on the interval, (12) holds with prefactor $\sqrt{c_+/c_-}$. A convenient sufficient drift check is $\|\dot C\|/\lambda_{\min}(C)^2<2$, though the signed eigenvalue expression is sharper. The implementation uses the conservative Frobenius norm for this simpler check. This proves coarse tracking from explicit kernel conditioning, drift, and forcing bounds, without assuming the tracking residual small.

It remains a conditional estimate: those kernel and forcing bounds must be obtained from an independently controlled trajectory or invariant region for an initialization theorem. Sampling them on the completed true run is a diagnostic, not a uniform-in-time proof. This is nevertheless a more concrete proof obligation than assuming an arbitrary stable evolution operator or already-small slope gradients.

For actual GD, there is an exact discrete counterpart; (12) must not be applied unchanged. Use a prime for the next GD state and define the modal step defect $R^\Delta=e'-e+\eta Ke$. This includes both omitted-mode forcing and the nonlinear finite-step remainder. Direct substitution gives

$$
z_C'=M^\Delta z_C+h^\Delta,\qquad
M^\Delta=I-\eta(C+B'Q^T),
$$

$$
h^\Delta=(B'-B-\eta B'S)e_H+R_C^\Delta+B'R_H^\Delta.
\tag{12b}
$$

Consequently $\|z_C'\|_{C'}\le\beta\|z_C\|_C+\|h^\Delta\|_{C'}$, with the directly defined factor $\beta=\|C'^{1/2}M^\Delta C^{-1/2}\|_2$. Iterating this inequality gives products of $\beta$ and accumulated forcing; a uniform factor below one gives geometric tracking. There is no discarded time-step error in (12b). A predictive GD theorem must bound the step defect and kernel changes, rather than measure them only after training. The half-step experiments check the practical flow/GD difference, while the population path inequalities below apply to GD exactly.

## 4. Coupled evolution of the residual and slope sensitivity

The force decomposition tells us which residual moves the slopes at a given state. A mechanism for failed acquisition must also explain how that force changes as training proceeds. A small force can persist, grow into a short episode of motion, or develop into sustained transport. These possibilities depend on the same feedback: parameters determine which residual directions exert force, and their movement changes the residual itself.

### What the observations let us concentrate on

The dense equal-rate trajectories cover five targets and seeds 0–4 at width 177, with $\eta=0.002$ and 2,048 training midpoints. Over updates 20,000–600,000, the median ratio of integrated coarse-tracking slope-force norm to integrated full slope-force norm is 0.186% for sine, 0.0517% for Runge, 0.436% for degree 3, 0.00142% for degree 5, and 0.00111% for degree 9. The largest sampled pointwise ratio is about 1.60%. Integrated omitted-force ratios are below $10^{-11}$ in these trajectories, indicating numerical negligibility in the retained degree-65 representation, rather than an exact vanishing theorem.

This supports studying $T_ae_H$ as the principal slope force on that observed interval. It does not remove the coarse balance already incorporated in $T_a$, nor the retained fine residual $e_H$. The distinction matters: a generated lower-degree residual belongs to $e_H$ and can dominate the force even when the difficult target coefficient has much higher degree.

The [curated measurements](../results/checkpoint_D_optimizers/expD34_readout_race/useful_slopes/curated/artifact_audit.json) supply the numbers above. They are retrospective observations of ordinary GD. The reported partial fitting and slope recoveries remain far from the intended scale and precision regime; those improvements do not establish successful geometry acquisition.

### Separate remaining error from sensitivity to that error

Work with the realized finite network and equal rates. Collect the exact corrections in (10) as

$$
q_a=J_{a,C}^Tz_C+g_{a,\perp},\qquad
\varepsilon_H=-K_{HC}z_C+f_H.
$$

Then the gradient-flow equations read

$$
\dot a=-F_a-q_a,\qquad F_a=T_ae_H,\qquad
\dot e_H=-Se_H+\varepsilon_H.
\tag{C1}
$$

The two corrections have different roles. Small $q_a$ says that the effective force approximates the instantaneous slope velocity. The term $\varepsilon_H$ affects the residual that will supply future force. Neither assertion by itself controls how accurately an independently evolved reduced model predicts the entire parameter trajectory.

To expose the feedback, write

$$
R_H=\|e_H\|,\qquad u_H=e_H/R_H,\qquad
\lambda^{\rm eff}=u_H^TSu_H,\qquad
\mu_a^{\rm eff}=u_H^TG_au_H.
$$

For $R_H>0$, differentiation of the norm and then of $e_H/R_H$ gives

$$
\dot R_H=-\lambda^{\rm eff}R_H+u_H^T\varepsilon_H,
\qquad
\dot u_H=-(S-\lambda^{\rm eff}I)u_H
+\frac{(I-u_Hu_H^T)\varepsilon_H}{R_H}.
\tag{C2}
$$

Meanwhile,

$$
A_a=R_H\sqrt{\mu_a^{\rm eff}}.
\tag{C3}
$$

Equation (C3) separates the amount of error from the sensitivity of slopes to its current direction. Large $R_H$ supplies little motion when $\mu_a^{\rm eff}$ is small. Fitting can change this coupling even at fixed $T_a$, because different components of the residual relax at different rates. Parameter evolution supplies a second source of change by moving $T_a$ itself.

The transport PDE retains that second source. Its joint distribution records the correlations among $a,b,c$ that determine the modal Jacobians and hence $T_a$ and $S$. Along the actual flow, $\dot T_a$ is their derivative under all parameter updates. Equations (C1)–(C2) therefore organize a coupled system whose state still includes the joint parameters; they do not close an ODE for $R_H$ and $u_H$ alone. In particular, allowing $T_a$ to evolve while prescribing its future values from a completed run would provide an attribution, not an autonomous prediction.

### When does the effective force amplify?

Differentiate the force before normalizing it:

$$
\dot F_a=\dot T_ae_H-T_aSe_H+T_a\varepsilon_H,
\qquad
\frac12\frac{dA_a^2}{dt}
=\frac12e_H^T(\dot G_a-G_aS-SG_a)e_H
+F_a^TT_a\varepsilon_H.
\tag{C4}
$$

The first term in $\dot F_a$ changes the map from residual to force; the second moves the residual through that map. Positive semidefiniteness of $S$ ensures dissipation of fine-residual energy when $\varepsilon_H=0$. It does not determine the sign of the second term's projection on $F_a$, because $G_a$ and $S$ need not commute.

For a concrete check, take $T_a=(1,0)$, $S=\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)$, and $e_H=(1,-3)^T$, with fixed matrices and no corrections. Here $0\preceq G_a\preceq S$, as required by (11a). Yet $\dot e_H=(1,5)^T$, so $d\|e_H\|^2/(2dt)=-14$ while $dA_a^2/(2dt)=1$. Fitting decreases error and increases slope force at this state. This is a local algebraic example, not an identified D34 trajectory.

For $A_a>0$, a more interpretable form of (C4) is

$$
\frac{d\log A_a}{dt}
=\underbrace{\frac{u_H^T\dot G_au_H}{2\mu_a^{\rm eff}}}_{\alpha_{\rm map}}
+\underbrace{\frac{-u_H^TG_a(S-\lambda^{\rm eff}I)u_H}{\mu_a^{\rm eff}}}_{\alpha_{\rm rotation}}
-\lambda^{\rm eff}
+\underbrace{\frac{u_H^TG_a\varepsilon_H}{R_H\mu_a^{\rm eff}}}_{\alpha_{\rm correction}}.
\tag{C5}
$$

To obtain it, use $\log A_a=\log R_H+\tfrac12\log\mu_a^{\rm eff}$ and differentiate $\mu_a^{\rm eff}=u_H^TG_au_H$. The radial parts of the correction cancel, leaving the final term shown. The equations in (C1) and (C4) remain valid at zero residual or zero force; the normalized quantities are undefined there and should not be interpreted by dividing by a small regularizer. In particular, zero force at an instant does not imply permanent arrest: $\dot F_a$ can be nonzero.

Equation (C5) identifies the competition we need to understand. Changing parameters can strengthen or weaken sensitivity. Residual rotation can move error toward stronger or weaker slope directions. Residual-magnitude decay removes drive at the nonnegative rate $\lambda^{\rm eff}$. Amplification occurs when their sum is positive. The individual terms need not keep their signs throughout training.

Readout evolution enters both sides of this competition. It contributes to fitting through $S$, while its changing coefficients also alter $T_a$. A growing readout norm therefore does not establish a growing slope force, and a large readout dissipation share does not establish permanent suppression. Changing a block's learning rate changes the coupled trajectory and its coarse balance; it is not an isolated intervention on one term of (C5).

The [force-driver code](../experiments/expD34_readout_race/mechanism.py) exports $F_a^T\dot T_ae_H$ and $-F_a^TT_aSe_H$. Dividing by $A_a^2$ gives $\alpha_{\rm map}$ and $\alpha_{\rm rotation}-\lambda^{\rm eff}$, respectively. Thus its term called fine relaxation includes rotation as well as residual-magnitude decay.

**Observed integrated log-force contributions. Unchanged GD, updates 20,000–600,000 ($t=40$–1200), degree-65 diagnostics, arithmetic means of per-seed integrals across seeds 0–4. Entries are dimensionless contributions to $\log[A_a(1200)/A_a(40)]$.**

| Target | Changing map | Fine relaxation, including rotation | Tracking correction |
|---|---:|---:|---:|
| Sine | $+4.261$ | $-1.476$ | $+0.00217$ |
| Runge | $+4.743$ | $-4.734$ | $+0.00377$ |
| Degree 3 | $-3.092$ | $+1.855$ | $-0.00545$ |
| Degree 5 | $+0.0794$ | $-0.0522$ | $+3.47\times10^{-7}$ |
| Degree 9 | $-0.0119$ | $-0.1565$ | $+1.62\times10^{-6}$ |

These are trapezoidal integrals of flow derivatives evaluated at actual GD states, with at most 2,000 updates between samples. Including the omitted contribution, the summed integrals agree with the endpoint log-force change to within $9\times10^{-4}$ in all 25 trajectories. This checks the diagnostic accounting at the available resolution; it is not an exact discrete identity or a bound on future derivatives. The source tables are the three `seed{0,1,2}_fork20000_metrics` and two `dense_seed{3,4}_metrics` directories in the linked curated package, restricted to `arm=joint`. Each trajectory is counted once.

The opposing terms explain why one final force norm cannot identify the history. Runge has large map amplification almost offset by residual evolution. Degree 3 has a positive integrated fine-relaxation contribution despite its falling force over this interval; its earlier and later episodes need not share the same balance. Degree 9's residual norm changes very little, yet residual evolution weakens its already tiny force by changing the small components to which slopes respond. Degrees 5 and 9 begin this interval with weak coupling, so modest changes in log force provide little absolute motion. None of these observations alone supplies a future amplification bound.

### A coupled example where amplification ends before large scales are acquired

An analytical model can make the competition explicit. The [mechanism companion](d34_transport_mechanisms.md#5-a-coupled-model-where-fitting-exhausts-the-drive-before-scale-acquisition) studies one accessible coefficient

$$
\varphi=\sum_j c_ja_j^3,\qquad h=Y-\varphi,\qquad
L_{\rm cub}=\tfrac12(h^2+Z^2),\qquad Y>0.
$$

Here $Z$ is an optional orthogonal target coefficient that the model cannot represent. It makes remaining error explicit, but supplies no force. The scalar $\varphi$ is the surrogate output coefficient; it is distinct from the omitted-mode forcing $f$ in (8). Both slopes and readouts train at the same rate:

$$
\dot a_j=3c_ja_j^2h,\qquad \dot c_j=a_j^3h.
$$

Assume $a_{j,0}>0$, $c_{j,0}=a_{j,0}/\sqrt3$, and $\varphi_0<Y$. This balance is preserved. Substitution yields

$$
\dot a_j=\sqrt3\,h a_j^3,\qquad
\varphi=\frac1{\sqrt3}\sum_j a_j^4,\qquad
\dot h=-4h\sum_i a_i^6.
\tag{C6}
$$

Every particle grows while the common driving coefficient $h$ is fitted away. The question is whether growth strengthens its own force quickly enough to compensate.

**Proposition: local amplification competes with collective depletion.** Under these assumptions, the positive slope speed $V_j=\dot a_j$ satisfies

$$
\frac{d\log V_j}{dt}
=3\sqrt3\,h a_j^2-4\sum_i a_i^6.
\tag{C7}
$$

**Proof.** Differentiate $\log V_j=\log\sqrt3+\log h+3\log a_j$ and substitute (C6). The first term on the right of (C7) is the increase of that particle's speed through its own growth. The second is the decrease of the shared residual through all particles' fitting. Both terms arise from the same simultaneous learning dynamics.

If all particles are equal, (C7) becomes

$$
\frac{d\log V}{dt}=\sqrt3\,a^2(3Y-7\varphi).
\tag{C8}
$$

Thus, when $\varphi_0<3Y/7$, the speed increases until $\varphi=3Y/7$ and decreases afterwards. If training starts beyond that point, the speed decreases from the outset. Slopes continue growing throughout either history. As proved in the companion, $h$ tends to zero and

$$
a_\infty=(\sqrt3Y/W)^{1/4},\qquad
\frac1W\#\{j:a_j\ge\Gamma\}
\le\min\left\{1,\frac{\sqrt3Y}{W\Gamma^4}\right\}
\tag{C9}
$$

for equal-particle convergence and the general positive balanced population bound, respectively. Starting near $W^{-1/2}$, the equal particles can therefore undergo substantial relative growth and a force-amplification episode while ending at a scale that decreases with width. Unequal particles have different amplification terms in (C7), but share the same depletion term: larger particles can keep accelerating after smaller ones have begun to slow.

This is a proved mechanism in the surrogate. The positivity and balance hypotheses enable its population bound; signed cancellations in tanh need not obey it. The companion also proves balance preservation and bounded acquisition for simultaneous GD under an explicit sufficient step bound. The derivative identity and the precise $3/7$ peak above concern flow; they are not asserted as exact finite-step GD formulas. Likewise, the surrogate's $a$ is not calibrated to D34 gamma, and its inaccessible $Z$ component cannot explain how a weak but nonzero hard-target force evolves.

### The next model must let the hard residual participate

The natural extension gives both target components a force and lets each residual change under both parameter blocks. Take fixed empirical orthonormal functions $q_3,q_9$ and define

$$
F_{\rm two}=\varphi_3q_3+\varphi_9q_9,\qquad
\varphi_k=\sum_j c_ja_j^k,\qquad h_k=Y_k-\varphi_k,
\qquad L_{\rm two}=\tfrac12(h_3^2+h_9^2).
$$

Exact equal-rate gradient flow in this surrogate is

$$
\dot a_j=c_j(3h_3a_j^2+9h_9a_j^8),\qquad
\dot c_j=h_3a_j^3+h_9a_j^9,
\tag{C10}
$$

and the residuals evolve jointly:

$$
\frac{d}{dt}\begin{pmatrix}h_3\\h_9\end{pmatrix}
=-K^{\rm two}\begin{pmatrix}h_3\\h_9\end{pmatrix},\qquad
K^{\rm two}_{k\ell}
=\sum_j\left(a_j^{k+\ell}+k\ell c_j^2a_j^{k+\ell-2}\right),
\quad k,\ell\in\{3,9\}.
\tag{C11}
$$

These formulas follow by differentiating each $\varphi_k$ using (C10). The two terms in $K^{\rm two}$ are the readout and slope contributions; each is a Gram matrix. Thus the loss cannot increase, while the same moving parameters change the force and the rates at which the two residuals relax. The ninth-degree residual now contributes to the slope force through $9c_ja_j^8h_9$ and to readout motion through $a_j^9h_9$.

For comparable residual coefficients at small $|a_j|$, the direct ninth-degree slope contribution is smaller than the cubic contribution by a factor proportional to $|a_j|^6$. Growing a particle changes that imbalance. But even with $Y_3=0$, the network generally generates $\varphi_3\ne0$, hence $h_3=-\varphi_3$: correction of its own lower-order output can compete with the hard target's force. This is the interaction a two-mode model can investigate without prescribing a residual-decay law from outside.

The focused theoretical question is whether the coupled system can build substantial hard-mode sensitivity before its initially accessible drive is exhausted, and how much outward motion remains if it cannot. A useful result would identify an interval of slow acquisition with a weak, nonzero hard force, allow subsequent amplification, and account for whether the movement is broad or concentrated. These are proposed results to establish, not consequences already proved by (C10)–(C11). The cubic balance is generally broken by the additional mode, so (C9) cannot simply be carried over.

The model is an analytical surrogate rather than a derived truncation of the D34 equations. Actual tanh modal coefficients include biases, additional powers and modes, and coarse-balance corrections. Establishing the correspondence requires checking those terms and their signs. Freezing its kernel retains the residual coupling while removing changes in its coefficients. Whether that local approximation predicts the relevant interval is an empirical question; externally prescribing the residual decay would bypass that test.

For persistence after coarse relaxation, the [conditional stagnation note](d34_coarse_balance_stagnation.md) derives an inward-force region and a quantitative tracking tolerance in a symmetric three-mode surrogate, together with a local exact-tanh result. Its degree-9 D34 experiments identify generated quadratic and cubic errors as the dominant drivers of mean contraction. An autonomous exact-tanh model retaining modes 0, 1, 2, 3, and 9 closely reproduces the coupled motion over the tested interval. Removing the lower-mode penalties leaves an outward hard-mode force too weak to acquire appreciable scale. Heterogeneous particles still move in both directions, so the population statement uses an outward-travel budget alongside the signed mean-force condition.

The persistence analysis narrows the mechanism further. Two force-carrying
directions, almost purely quadratic and cubic in output space, explain the slow
relaxation with a fixed local tangent. Their coupled residual equations yield
a bounded transient movement budget plus a weak persistent hard force. A
separate argument subtracts the hard-mode loss that cannot be removed inside
a parameter ball before applying GD descent. A step-containment induction
then closes that ball without assuming the future trajectory stays inside.
Ordinary FP64 evaluation excludes scale 1 for at least ten million additional
updates from each of ten retained degree-9 starting states; this is an
analytical conditional result with numerical constants, not a directed-rounding
certificate or a theorem from random initialization. The same bound keeps
relative MSE above 0.749998. The [derivation and evidence](d34_coarse_balance_stagnation_details.md#10-testing-what-keeps-the-force-small)
separate this confinement claim from the accuracy of the local predictor.

### From force feedback to a finite-time acquisition statement

Force strength is only one ingredient of transport. Away from zero slopes, (C1) gives

$$
\frac{d\bar\gamma}{dt}
=\frac{\chi_a A_a}{\sqrt W}
-\frac{\operatorname{sign}(a)^Tq_a}{W},\qquad
\chi_a=-\frac{\operatorname{sign}(a)^TF_a}{\sqrt W A_a},
\qquad \bar\gamma=\frac1W\sum_j|a_j|.
\tag{C12}
$$

Here $|\chi_a|\le1$ when $A_a>0$. Amplification can strengthen inward motion, and positive mean growth can be concentrated in a few particles. The transition-band flux in (16) asks the sharper population question: how much mass actually approaches and crosses the requested threshold?

The coupled analysis can feed the path bound in (14) without assuming monotone force decay. For example, if an independently justified integrable envelope $\omega_a(t)$ satisfies $d\log A_a/dt\le\omega_a(t)$ on an interval where $A_a>0$, then

$$
\int_s^T A_a(t)\,dt
\le A_a(s)\int_s^T
\exp\!\left(\int_s^t\omega_a(v)\,dv\right)dt.
\tag{C13}
$$

Positive episodes of $\omega_a$ are allowed. Adding a bound on $\int_s^T\|q_a\|dt$ and comparing with the distance to the required population event yields a conditional exclusion. The hard part is obtaining $\omega_a$ from controlled residual/parameter dynamics. Integrating the measured terms in (C5) only reconstructs a completed force history.

For the actual discrete training algorithm, use the exact update instead:

$$
a_N-a_{n_0}=-\eta\sum_{n=n_0}^{N-1}(F_{a,n}+q_{a,n}),\qquad
\|a_N-a_{n_0}\|
\le\eta\sum_{n=n_0}^{N-1}(A_{a,n}+\|q_{a,n}\|).
\tag{C14}
$$

Let $D_{p,\Gamma}(a_{n_0})$ be the Euclidean distance from the initial slope vector to the set with at least $\lceil pW\rceil$ slopes satisfying $|a_j|\ge\Gamma$. A terminal sum below that distance excludes acquisition at every intermediate update, since all prefix sums are no larger. No diffusion or stochastic escape term enters this argument. Flow derivatives evaluated at GD states help explain the mechanism; a predictive GD bound must control the discrete sums or justify the corresponding step remainders. The research target is therefore a coupled explanation of the available outward travel before sensitivity weakens or fails to develop, with the existing barrier supplying the final population implication.

## 5. Three routes from these equations to a population barrier

**Moment control.** For $M_{2q}(t)=\int|a|^{2q}d\rho$, the weak transport equation gives $\dot M_{2q}=2q\int a|a|^{2q-2}v_a\,d\rho$ and

$$
\rho_t\{|a|\ge\Gamma\}\le M_{2q}(t)/\Gamma^{2q}.
\tag{13}
$$

A bound $M_{2q}(t)<p\Gamma^{2q}$ uniformly over the horizon proves the desired population barrier. This does not require every neuron to remain below $\Gamma$. Equation (5), cross moments, and residual-mode equations suggest candidate differential inequalities, but high moments and the signed balance defect still require closure. Substituting measured moments into (13) is only a retrospective bound.

**Effective slope force.** Let $D_{p,\Gamma}(a_s)$ be the Euclidean distance to the set having at least $\lceil pW\rceil$ acquired slopes, as defined in the companion note. If

$$
\int_s^T\left(\|J_{a,C}^T\|\|z_C\|+\|T_ae_H\|+\|g_{a,\perp}\|\right)dt
<D_{p,\Gamma}(a_s),
\tag{14}
$$

population acquisition is impossible throughout that interval. The research challenge is to predict the integrand through (9)–(12). Equation (14) itself follows from the length of the slope path and is not a new suppression theorem.

An alternative uses the target-dependent slope allocation

$$
\theta(t)=\frac{\|g_a\|^2}{-\dot L},\qquad
B_{s,T}^2\le(T-s)\int_s^T\|g_a\|^2dt
=(T-s)\int_s^T\theta(t)(-\dot L)dt.
\tag{15}
$$

For full residual coordinates, $\theta=e^TK_ae/(e^TKe)$. A uniform bound $\theta\le\bar\theta$ yields $B_{s,T}^2\le(T-s)\bar\theta[L(s)-L(T)]$. The target-weighted ratio is the relevant quantity; the operator norm of $K$ alone is insufficient. A small late value of $\theta$ need not make the integrated early-to-late budget small. For GD, the implemented exact discrete Cauchy–Schwarz bound uses $\eta\sum\|g_{a,n}\|^2$ directly. Replacing this sum by loss decay requires a separate discrete descent estimate.

**Signed population transport.** For a smooth bounded increasing approximation $H_\epsilon$ to the threshold indicator, define $P_\epsilon=\int H_\epsilon(|a|-\Gamma)d\rho$. Then

$$
\dot P_\epsilon=\int H_\epsilon'(|a|-\Gamma)\operatorname{sign}(a)v_a\,d\rho.
\tag{16}
$$

This is a flux near the threshold, not mean slope motion. For a density, the sharp version integrates $v_a\rho$ on $a=\Gamma$ minus its value on $a=-\Gamma$. Positive and negative travel are recorded separately because cancellation and concentration can prevent population acquisition despite recovered gradient norms. For atomic measures use the weak/smoothed equation or exact threshold counts; a classical density flux is not assumed.

## 6. What turns a successful forecast into a theorem?

### A testable width prediction before a long-time theorem

The rescaling in (4) offers a more specific prediction than merely saying that a slope gradient is small. In a post-coarse-fit regime with bounded rescaled moments and an order-one coarse inverse, tanh's first nonlinear term gives retained slope tangents for modes 2 and 3 of individual size $O(W^{-3/2})$. Their population norm is $O(W^{-1})$. The coarse-elimination correction in $T_a$ has the same order, so it must be retained even at leading order. Consequently the effective slope kernel on these modes, $T_a^TT_a$, is $O(W^{-2})$.

For sine and the degree-3 target, an order-one non-affine target coefficient can therefore produce $\|g_a\|=O(W^{-1})$ after the coarse transient. For the degree-9 target, direct coupling to mode 9 first occurs through the eighth-order term of $xs$ and has population norm $O(W^{-4})$. But that is not the full force: the model's own order-$W^{-1}$ nonlinear residual in lower modes acts through their order-$W^{-1}$ tangent, giving an order-$W^{-2}$ contribution. Thus the proposed leading scale for the degree-9 *full* slope force is $W^{-2}$, not $W^{-4}$, absent cancellations or a larger remaining coarse transient.

These are formal power-counting predictions in a stated regime, not matching upper and lower bounds. They predict approximate collapse of $W\|g_a\|$ for sine/degree 3 and $W^2\|g_a\|$ for degree 9 across widths, at a fixed post-transient time (the experiment uses $t=40$). They also identify a potential nonlinear drift time: since rescaled characteristic velocity is order $W^{-1}$ for low-order non-affine forcing, order-one rescaled geometry changes can begin on times of order $W$. A frozen-kernel extrapolation can therefore fail before the much longer initial fine-mode residual relaxation time suggested by a kernel of size $W^{-2}$.

The corresponding degree-9 self-curvature time is of order $W^2$; direct mode-9 forcing alone has the still longer formal time $W^4$. Indeed, the $L^2(\rho)$ norm of the rescaled slope velocity $\dot A=-\sqrt W g_a$ equals the population slope-force norm used above. At $W=177$, the experimental horizon has $T/W=6.78$ but $T/W^2=0.0383$. This predicts that low-order non-affine targets can leave the affine regime while the high-order target still undergoes weak correction of its own generated lower modes. It is a regime prediction with moment-control and cancellation caveats, not a proof that the degree-9 trajectory remains trapped until either nominal time scale.

Moment growth, cancellation, unresolved coarse tracking, or finite-seed fluctuations can invalidate this proposed scaling regime. Width experiments must measure those alternatives rather than interpreting every departure as a different fitted exponent. In particular, this expansion does not control the moments up to time 1200 and cannot alone certify a gamma barrier. It supplies a concrete kernel prediction to test while pursuing that control.

For a predicted distribution $\widehat\rho_t$ and a justified Wasserstein error bound $W_2(\rho_t,\widehat\rho_t)\le\varepsilon(t)$, any $0<\delta<\Gamma$ gives

$$
\rho_t\{|a|\ge\Gamma\}\le
\widehat\rho_t\{|a|\ge\Gamma-\delta\}+\varepsilon(t)^2/\delta^2.
\tag{17}
$$

Proof: in a coupling, a true particle beyond $\Gamma$ either has a predicted partner beyond $\Gamma-\delta$ or their slope coordinates differ by at least $\delta$; apply Markov's inequality to squared displacement. The proof only requires a bound on the gamma-marginal Wasserstein distance. Thus for paired finite networks one may use $\varepsilon=\|a-\widehat a\|_2/\sqrt W$ even without a bound on the other parameter coordinates. The observed forecast error is not an independent proof of that radius.

Equation (17) separates three obligations: certify the prediction's population tail; bound time-step and modal/quadrature errors; and, for a law forecast, control finite-seed fluctuations. Large global Lipschitz/Grönwall estimates may be valid yet useless at time 1200. A useful proof would exploit coarse relaxation, target-weighted effective forcing, or a localized moment enclosure. Failure of one route is a mathematical result to report, rather than a reason to replace its missing bound with an observed small gradient.

## Experimental discrimination

The locked protocol in the [experiment README](../experiments/expD34_readout_race/README.md#transport-and-residual-basis-extension) tests degree 33 against degrees 9, 17, and 65; law quadrature orders 8, 12, and 16; full residual evolution; and a half time step at matched physical time. Adaptive controls extend quadrature to orders 24 and 32 and check the full residual at order 32. Fresh seeds assess independent predictions. Width and readout-rate changes assess transfer beyond the primary equal-rate setting. Frozen-initial-kernel predictions test whether kernel evolution is needed; the existing affine reference tests whether nonlinear features are needed.

There are three separate scientific questions. Does the exact-tanh modal forecast track actual trajectories and population tails? Does the law PDE predict the distribution of finite-seed outcomes, especially rare escapes? Do (5), (9)–(12), and (15) identify a quantity that can be bounded without the completed true trajectory? Positive answers to the first two do not automatically answer the third. The intended paper claim should be limited accordingly.

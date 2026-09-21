# Transport, residual modes, and a finite-time gamma barrier

This extension of the [D34 scale-acquisition argument](d34_scale_acquisition_theory.md) studies the actual random-readout, equal-rate initialization. Its purpose is to predict when signal depletion persists, when signal regenerates without population scale acquisition, and when a substantial population acquires the specified scale. These outcomes need not have the same explanation. All barrier statements below concern a specified finite horizon; they allow isolated escaping neurons.

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

For the complete residual representation, $\dot e=-Ke$. With a finite basis the exact full-network equation has the omitted-mode forcing $f$:

$$
\dot e=-Ke+f,\qquad
f=-P_K\mathcal K r_\perp,
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

Then $d\|z_C\|_C/dt\le-\alpha_C\|z_C\|_C+\|f_z\|_C$, interpreted by continuity at zero. In particular, if $mI\preceq C\preceq MI$ and $\alpha_C\ge\alpha>0$ on the interval, (12) holds with prefactor $\sqrt{M/m}$. A convenient sufficient drift check is $\|\dot C\|/\lambda_{\min}(C)^2<2$, though the signed eigenvalue expression is sharper. This proves coarse tracking from explicit kernel conditioning, drift, and forcing bounds, without assuming the tracking residual small.

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

## 4. Three routes from these equations to a population barrier

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

## 5. What turns a successful forecast into a theorem?

### A testable width prediction before a long-time theorem

The rescaling in (4) offers a more specific prediction than merely saying that a slope gradient is small. In a post-coarse-fit regime with bounded rescaled moments and an order-one coarse inverse, tanh's first nonlinear term gives retained slope tangents for modes 2 and 3 of individual size $O(W^{-3/2})$. Their population norm is $O(W^{-1})$. The coarse-elimination correction in $T_a$ has the same order, so it must be retained even at leading order. Consequently the effective slope kernel on these modes, $T_a^TT_a$, is $O(W^{-2})$.

For sine and the degree-3 target, an order-one non-affine target coefficient can therefore produce $\|g_a\|=O(W^{-1})$ after the coarse transient. For the degree-9 target, direct coupling to mode 9 first occurs through the eighth-order term of $xs$ and has population norm $O(W^{-4})$. But that is not the full force: the model's own order-$W^{-1}$ nonlinear residual in lower modes acts through their order-$W^{-1}$ tangent, giving an order-$W^{-2}$ contribution. Thus the proposed leading scale for the degree-9 *full* slope force is $W^{-2}$, not $W^{-4}$, absent cancellations or a larger remaining coarse transient.

These are formal power-counting predictions in a stated regime, not matching upper and lower bounds. They predict approximate collapse of $W\|g_a\|$ for sine/degree 3 and $W^2\|g_a\|$ for degree 9 across widths, at a fixed post-transient time (the experiment uses $t=40$). They also identify a potential nonlinear drift time: since rescaled characteristic velocity is order $W^{-1}$ for low-order non-affine forcing, order-one rescaled geometry changes can begin on times of order $W$. A frozen-kernel extrapolation can therefore fail before the much longer initial fine-mode residual relaxation time suggested by a kernel of size $W^{-2}$.

Moment growth, cancellation, unresolved coarse tracking, or finite-seed fluctuations can invalidate this proposed scaling regime. Width experiments must measure those alternatives rather than interpreting every departure as a different fitted exponent. In particular, this expansion does not control the moments up to time 1200 and cannot alone certify a gamma barrier. It supplies a concrete kernel prediction to test while pursuing that control.

For a predicted distribution $\widehat\rho_t$ and a justified Wasserstein error bound $W_2(\rho_t,\widehat\rho_t)\le\varepsilon(t)$, any $0<\delta<\Gamma$ gives

$$
\rho_t\{|a|\ge\Gamma\}\le
\widehat\rho_t\{|a|\ge\Gamma-\delta\}+\varepsilon(t)^2/\delta^2.
\tag{17}
$$

Proof: in a coupling, a true particle beyond $\Gamma$ either has a predicted partner beyond $\Gamma-\delta$ or their slope coordinates differ by at least $\delta$; apply Markov's inequality to squared displacement. For paired finite networks one may use $\varepsilon=\|a-\widehat a\|_2/\sqrt W$. The observed forecast error is not an independent proof of that radius.

Equation (17) separates three obligations: certify the prediction's population tail; bound time-step and modal/quadrature errors; and, for a law forecast, control finite-seed fluctuations. Large global Lipschitz/Grönwall estimates may be valid yet useless at time 1200. A useful proof would exploit coarse relaxation, target-weighted effective forcing, or a localized moment enclosure. Failure of one route is a mathematical result to report, rather than a reason to replace its missing bound with an observed small gradient.

## Experimental discrimination

The locked protocol in the [experiment README](../experiments/expD34_readout_race/README.md#transport-and-residual-basis-extension) tests degree 33 against degrees 9, 17, and 65; law quadrature orders 8, 12, and 16; full residual evolution; and a half time step at matched physical time. Fresh seeds assess independent predictions. Width and readout-rate changes assess transfer beyond the primary equal-rate setting. Frozen-initial-kernel predictions test whether kernel evolution is needed; the existing affine reference tests whether nonlinear features are needed.

There are three separate scientific questions. Does the exact-tanh modal forecast track actual trajectories and population tails? Does the law PDE predict the distribution of finite-seed outcomes, especially rare escapes? Do (5), (9)–(12), and (15) identify a quantity that can be bounded without the completed true trajectory? Positive answers to the first two do not automatically answer the third. The intended paper claim should be limited accordingly.

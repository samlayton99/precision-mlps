# Understanding the gamma barrier: a theorem and a measurement guide

Why can training fit the output while a population of slopes fails to acquire a specified scale? The useful object is the slope force that remains after the easily fitted residual components have relaxed. Its magnitude, direction, and distribution across neurons answer different questions. This note derives a conditional finite-time population-barrier theorem and explains how to measure its ingredients even when a complete proof from initialization is unavailable.

The theorem applies to actual simultaneous GD, including random nonzero readouts and nonlinear signal regeneration. It does not establish that its hypotheses hold for every D34 initialization. The [technical theory note](d34_transport_scale_barrier.md) gives the broader transport formulation; the [evidence report](../results/checkpoint_D_optimizers/expD34_readout_race/transport_barrier/README.md) records the numerical tests. Here the goal is to make the mathematical steps and diagnostic interpretation accessible without prior knowledge of Schur complements or transport PDEs.

**Notation. Physical parameters and empirical norms are used throughout.**

| Symbol | Meaning |
|---|---|
| $W,m$ | Physical network width and number of training samples. |
| $a,b,c,d$ | Slopes, hidden biases, readout weights, and output bias. |
| $\gamma_j=\lvert a_j\rvert$ | Magnitude of slope $j$; its sign is retained in the dynamics. |
| $\eta,\kappa,t_n=\eta n$ | Geometry step, readout/geometry rate ratio, and physical training time. D34's main setting has $\kappa=1$. |
| $r,R,L$ | Training residual, its empirical RMS, and half-MSE: $R=\Vert r\Vert_m$, $L=R^2/2$. |
| $g_I$ | Loss gradient in parameter block $I$. The actual slope velocity is $-g_a$. |
| $e_C,e_H$ | Residual coefficients in coarse modes $\{0,1\}$ and retained higher modes. |
| $K,C,Q$ | Rate-weighted modal tangent kernel, its coarse block $C=K_{CC}$, and coupling $Q=K_{CH}$. |
| $B,z_C$ | Coarse equilibrium map $B=C^{-1}Q$ and tracking error $z_C=e_C+Be_H$. |
| $T_a,G_a,S$ | Effective slope map, its Gram matrix $G_a=T_a^TT_a$, and effective total kernel. |
| $p,\Gamma,\mathcal D$ | Required population fraction, slope threshold, and distance to that acquired-population set. |
| $\tau=\Vert z_C\Vert_C$ | Tracking error in the kernel metric: $\tau^2=z_C^TCz_C$. |

## 1. Start with the outcome we want to explain

On D34's one-dimensional training midpoints $x_i\in[-1,1]$, the network and objective are

$$
F(x)=d+\sum_{j=1}^W c_j\tanh(a_jx+b_j),\qquad
r=F-y,\qquad
\langle u,v\rangle_m=\frac1m\sum_{i=1}^m u(x_i)v(x_i),\qquad
L=\tfrac12\langle r,r\rangle_m.
$$

D34 initializes $a,b,c$ with independent uniform draws on $[-\sqrt{6/(W+1)},\sqrt{6/(W+1)}]$ and sets $d=0$. The population theorem below does not require this distribution, but the width predictions use its $W^{-1/2}$ parameter scale. In particular, the readout is not initialized to zero.

Writing $u_j=a_jx+b_j$ and $s_j=\operatorname{sech}^2u_j$ gives

$$
(g_a)_j=c_j\langle r,xs_j\rangle_m,\quad
(g_b)_j=c_j\langle r,s_j\rangle_m,\quad
(g_c)_j=\langle r,\tanh u_j\rangle_m,\quad
g_d=\langle r,1\rangle_m.
$$

For $\eta>0$ and $\kappa>0$, simultaneous GD uses these gradients at the old state:

$$
(a,b)_{n+1}=(a,b)_n-\eta(g_a,g_b)_n,\qquad
(c,d)_{n+1}=(c,d)_n-\eta\kappa(g_c,g_d)_n.
$$

Unsubscripted vector norms are Euclidean; matrix norms are spectral unless marked with $F$ for the Frobenius norm. The sample-space norm $\|\cdot\|_m$ uses an empirical mean, not a sum. There are three separate outcomes to measure: whether a slope gradient exists; whether it points toward increasing slope magnitudes; and whether enough neurons move far enough. A large maximum gamma answers none of the population questions by itself.

For fixed $p\in(0,1]$ and $\Gamma>0$, define acquisition by

$$
P_\Gamma(a)=\frac1W\#\{j:|a_j|\ge\Gamma\}\ge p.
$$

For example, $p=0.1$, $\Gamma=3.2$, and $W=177$ require 18 acquired slopes. One escaping neuron is compatible with the barrier. These thresholds specify an outcome, not a necessary representation condition: proving that accurate approximation requires these scales is a separate problem.

At starting update $s$, sort the deficits $d_j=(\Gamma-|a_{sj}|)_+$ increasingly. The exact Euclidean distance to acquisition is

$$
\mathcal D_{p,\Gamma}(a_s)=
\left(\sum_{j=1}^{\lceil pW\rceil}d_{(j)}^2\right)^{1/2}.
\tag{1}
$$

To see this, making one unacquired coordinate reach magnitude $\Gamma$ costs at least its deficit, achieved by moving outward. Choose the cheapest required coordinates, including already acquired coordinates with zero deficit. This explains both the sorting and the square root. If $\mathcal D=0$, acquisition has already occurred; a strict exclusion starting at that state is impossible.

## 2. Express the residual in coordinates that reveal competition

Choose fixed functions $q_0,\ldots,q_\ell$ orthonormal under the empirical mean inner product. In D34 they are empirical Legendre modes; $q_0,q_1$ span constants and linear functions. Write

$$
e_k=\langle q_k,r\rangle_m,\qquad
r=\sum_{k=0}^{\ell}e_kq_k+r_\perp.
$$

This changes residual coordinates without approximating tanh. Approximation enters only if we neglect $r_\perp$ or train a separate forecast with the projected residual. In the following identities, the omitted contribution remains explicit.

For each parameter block, define its modal Jacobian. For slopes,

$$
(J_a)_{kj}=\langle q_k,c_jxs_j\rangle_m,
\qquad g_a=J_a^Te+g_{a,\perp},
\qquad (g_{a,\perp})_j=c_j\langle r_\perp,xs_j\rangle_m.
$$

Jacobian rows index residual modes and columns index parameters; $J_d$ is a single column. Stack the rate-weighted blocks as

$$
J=[J_a\;J_b\;\sqrt\kappa J_c\;\sqrt\kappa J_d],\qquad
K=JJ^T=K_a+K_b+K_c+K_d.
$$

Thus $K_a=J_aJ_a^T$, while $K_c=\kappa J_cJ_c^T$ and similarly for $d$. In gradient flow, the retained residual obeys

$$
\dot e=-Ke+f,\qquad
f_k=-\langle q_k,\mathcal K r_\perp\rangle_m,
\tag{2}
$$

where $\mathcal K$ is the full empirical tangent operator. With a complete residual basis, $f=0$. The total kernel describes residual relaxation. Its slope block describes how that relaxation moves slopes. These are different measurements.

### How the transport PDE fits into this description

Let $\rho=W^{-1}\sum_j\delta_{(a_j,b_j,c_j)}$. Moving these atoms with velocity $v=-(g_a,g_b,\kappa g_c)$ is equivalently

$$
\partial_t\rho+\nabla\cdot(\rho v)=0,
\qquad \dot d=-\kappa g_d,
\qquad F_\rho=d+W\int c\tanh(ax+b)\,d\rho.
$$

This is the same gradient-flow system; Euler stepping gives the GD update above. Replacing the realized atoms by the continuous initialization law is an additional approximation, not an identity for a particular seed.

For law quadrature with masses $w_i$, the nodes follow the same characteristic velocity. Their Euclidean loss gradients have an extra factor $Ww_i$, which must be divided out. Kernel columns use $\sqrt{Ww_i}$ weighting, and the slope-force norm is $\sqrt{W\sum_iw_i g_a(z_i)^2}$. Physical width stays $W$ when quadrature resolution increases. The finite-network acquisition distance (1) uses integer neuron counts; continuum mass may split and instead uses the relaxed weighted distance implemented in `weighted_distance`.

## 3. Remove the coarse transient without removing regeneration

Partition modes into $C=\{0,1\}$ and the remaining retained modes $H$. The coarse equation is

$$
\dot e_C=-Ce_C-Qe_H+f_C,\qquad C=K_{CC},\quad Q=K_{CH}.
$$

If $C$ is positive definite, define $B=C^{-1}Q$. With slowly changing higher modes and kernel, and small omitted forcing, the instantaneous coarse balance is $e_C\approx-Be_H$. It is generally not zero. The higher modes continually force the coarse modes while coarse fitting continually removes that forcing.

Define the error in tracking this moving balance,

$$
z_C=e_C+Be_H,
$$

and substitute $e_C=z_C-Be_H$ into the slope gradient:

$$
g_a=
\underbrace{J_{a,C}^Tz_C}_{\text{tracking contribution}}
+\underbrace{(J_{a,H}^T-J_{a,C}^TB)e_H}_{\text{effective fine-mode contribution}}
+\underbrace{g_{a,\perp}}_{\text{omitted modes}}.
\tag{3}
$$

The effective map is $T_a=J_{a,H}^T-J_{a,C}^TB$. It has $W$ rows and one column per retained fine mode, while $G_a=T_a^TT_a$ acts in fine-residual coordinates. It combines direct higher-mode forcing with the coarse response it induces. This subtraction is where interference enters. Simply discarding the coarse residual would retain $J_{a,H}^Te_H$ and miss that interference.

The fine-mode equation becomes

$$
\dot e_H=-Se_H-Q^Tz_C+f_H,\qquad
S=K_{HH}-Q^TC^{-1}Q.
\tag{4}
$$

The matrix $S$ is the Schur complement: the effective residual kernel after coarse relaxation. Its positivity has a useful geometric explanation. In the rate-weighted parameter space, $P_C=J_C^TC^{-1}J_C$ is the orthogonal projection onto the span of the coarse tangent vectors. Hence

$$
E=(I-P_C)J_H^T=J_H^T-J_C^TB,\qquad S=E^TE.
$$

We have removed the part of each fine tangent that can be accounted for by coarse tangents. Separating $E$ into parameter blocks gives the exact identity

$$
S=G_a+G_b+\kappa G_c+\kappa G_d,
\qquad G_I=T_I^TT_I,\qquad 0\preceq G_a\preceq S.
\tag{5}
$$

Consequently $e_H^TG_ae_H$ is the squared effective slope-force norm. The quotient $e_H^TG_ae_H/(e_H^TSe_H)$ is its share of effective fine-mode dissipation. A small share can indicate that other blocks do most of the fitting, but does not by itself imply a small force: the denominator also matters.

### A two-mode example: the total kernel cannot determine slope learning

Consider local linear models with one slope-like parameter and one readout-like parameter, at equal rates. Rows of each Jacobian are coarse and fine modes; columns are the two parameter blocks:

$$
J^{(1)}=\begin{pmatrix}1&0\\1&1\end{pmatrix},\qquad
J^{(2)}=\begin{pmatrix}0&1\\1&1\end{pmatrix}.
$$

Both have

$$
K=\begin{pmatrix}1&1\\1&2\end{pmatrix},\qquad B=1,\qquad S=1.
$$

At coarse balance, $e_C=-e_H$. In model 1, the slope force is $e_C+e_H=0$: direct fine forcing cancels the regenerated coarse contribution. In model 2, the slope force is $e_H$. Thus $G_a=0$ versus $G_a=1$, despite identical total residual kernels. This algebraic example illustrates the measurement problem; it is not a claim that these matrices describe a particular D34 trajectory.

## 4. Establish tracking, then convert force into a population barrier

Small $z_C$ must be established rather than assumed from a small $e_C$. Differentiating $z_C$ in flow gives

$$
\dot z_C=-(C+C^{-1}QQ^T)z_C+f_z,
\qquad
f_z=(\dot B-BS)e_H+f_C+Bf_H.
\tag{6}
$$

The forcing includes changes in the equilibrium map, evolution of the fine residual, and omitted modes. A bound on $\dot C$ alone cannot control it.

The matrix multiplying $z_C$ is not necessarily symmetric. Use the metric $\tau^2=z_C^TCz_C$ instead of guessing Euclidean contraction. For $V=\tau^2/2$,

$$
\dot V=-z_C^T(C^2+QQ^T)z_C
+\tfrac12z_C^T\dot C z_C+z_C^TCf_z.
$$

Since $C^2\succeq\lambda_{\min}(C)C$, it follows that

$$
\dot\tau\le-\alpha_C\tau+\|f_z\|_C,
\qquad
\alpha_C=\lambda_{\min}(C)
-\tfrac12\lambda_{\max}(C^{-1/2}\dot C C^{-1/2}).
\tag{7}
$$

Interpret the norm inequality by a limiting argument at $\tau=0$. A uniformly positive $\alpha_C$ means that relaxation beats metric drift; the forcing then sets a tracking floor. This is a conditional estimate derived from explicit kernel quantities, not a statement that the tracking error always decays to zero.

### Actual GD has an exact discrete tracking equation

Use a prime for the next GD state, not the next saved checkpoint. Define the exact modal step defect $R^\Delta=e'-e+\eta Ke$. It includes the nonlinear finite-step remainder and omitted modes. Substitution into $z_C'=e_C'+B'e_H'$ gives

$$
z_C'=M^\Delta z_C+h^\Delta,
\quad M^\Delta=I-\eta(C+B'Q^T),
$$

$$
h^\Delta=(B'-B-\eta B'S)e_H+R_C^\Delta+B'R_H^\Delta.
\tag{8}
$$

Therefore

$$
\tau_{n+1}\le\beta_n\tau_n+\|h_n^\Delta\|_{C_{n+1}},\qquad
\beta_n=\|C_{n+1}^{1/2}M_n^\Delta C_n^{-1/2}\|_2.
\tag{9}
$$

There is no discarded step-size term. Products of the $\beta_n$ and accumulated forcing bound tracking; a constant bound below one gives the simpler corollary below. Sampled factors below one do not prove that every intervening factor is below one.

### Theorem: a sufficient finite-time population barrier for GD

Consider updates $s,\ldots,N$ with $N>s$, positive definite coarse blocks, and the exact decomposition (3). Suppose nonnegative envelopes satisfy, at every update $n=s,\ldots,N-1$,

$$
\|J_{a,C,n}^Tz_{C,n}\|\le u_n,\qquad
\sqrt{e_{H,n}^TG_{a,n}e_{H,n}}\le v_n,\qquad
\|g_{a,\perp,n}\|\le w_n.
$$

If

$$
\mathcal B_{s,N}:=\eta\sum_{n=s}^{N-1}(u_n+v_n+w_n)
<\mathcal D_{p,\Gamma}(a_s),
\tag{10}
$$

then $P_\Gamma(a_k)<p$ at every update $k=s,\ldots,N$.

**Proof.** Equation (3) and the triangle inequality give $\|g_{a,n}\|\le u_n+v_n+w_n$. The GD update then gives, for every $k\le N$,

$$
\|a_k-a_s\|\le\sum_{n=s}^{k-1}\|a_{n+1}-a_n\|
=\eta\sum_{n=s}^{k-1}\|g_{a,n}\|
\le\mathcal B_{s,N}.
$$

Any acquired-population state is at least the distance (1) away. The strict inequality excludes every such state, including intermediate updates. This proves the theorem. It permits any number of individual escapes short of the specified fraction. No small-slope expansion or monotonic-loss assumption enters this proof.

The bound deliberately ignores beneficial cancellation between the three force contributions. It can consequently be inconclusive even when the exact path is short. Failure of (10), or of a tracking hypothesis, does not prove acquisition.

### Corollary: explicit constants turn tracking into a persistence horizon

An especially useful inequality removes an otherwise unknown tangent prefactor:

$$
\|J_{a,C}^Tz_C\|^2
=z_C^TJ_{a,C}J_{a,C}^Tz_C
\le z_C^TCz_C=\tau^2.
\tag{11}
$$

Suppose on all required updates that

$$
\beta_n\le\bar\beta<1,\quad
\|h_n^\Delta\|_{C_{n+1}}\le\eta F_0,\quad
\|T_{a,n}e_{H,n}\|\le\epsilon,\quad
\|g_{a,\perp,n}\|\le\delta,
\qquad 0\le\bar\beta<1.
$$

For $M=N-s$ and $A_M=\sum_{j=0}^{M-1}\bar\beta^j=(1-\bar\beta^M)/(1-\bar\beta)$, iteration of (9) gives

$$
\tau_{s+j}\le\bar\beta^j\tau_s+
\frac{\eta F_0}{1-\bar\beta}(1-\bar\beta^j).
$$

Using (11) in (10), the resulting upper bound on path is

$$
\overline{\mathcal B}_{s,N}
=\eta A_M\tau_s
+\frac{\eta^2F_0}{1-\bar\beta}(M-A_M)
+\eta M(\epsilon+\delta).
\tag{12}
$$

The first term is the decaying coarse transient, the second is sustained regeneration, and the last is effective fine forcing plus truncation. With $\alpha=(1-\bar\beta)/\eta$ and $\Delta t=\eta M$, a simpler sufficient bound is

$$
\overline{\mathcal B}_{s,N}
\le\frac{\tau_s}{\alpha}
+\Delta t\left(\frac{F_0}{\alpha}+\epsilon+\delta\right).
\tag{13}
$$

If this is below $\mathcal D$, acquisition is excluded. When the denominator is positive and $\mathcal D>\tau_s/\alpha$, this yields the sufficient horizon

$$
\Delta t<
\frac{\mathcal D-\tau_s/\alpha}{F_0/\alpha+\epsilon+\delta},
\tag{14}
$$

within the interval where the assumed bounds hold. If the denominator is zero, the transient budget alone suffices on that interval. A nonpositive numerator makes this simplified bound inconclusive; the sharper finite sum (12) can still help. The flow inequality (7), with $\alpha_C\ge\alpha$ and $\|f_z\|_C\le F_0$, gives the analogous integral bound.

The unresolved analytical task is to obtain these constants from initialization, target structure, or a verified enclosing region. Choosing them as maxima along the completed true run only produces a retrospective bound. Predicting them from an independently evolved approximation is useful, but becomes a proof only with a justified approximation-error bound.

## 5. Metrics that distinguish the mechanisms

The quantities below are useful without satisfying the theorem. Tracking and omitted-mode metrics identify which approximation is failing. A large $Qe_H$ motivates tracking a nonzero coarse balance; it does not alone imply inaccurate tracking. A small effective slope share identifies where fitting effort goes. Poor outward alignment or concentrated movement helps explain why a recovered norm does not produce broad scale acquisition.

For source references in the tables, **T** denotes the [transport evidence directory](../results/checkpoint_D_optimizers/expD34_readout_race/transport_barrier/), and **R** the [signal-recovery directory](../results/checkpoint_D_optimizers/expD34_readout_race/signal_recovery/). “Derived” means the formula is given here but there is no identically named exported scalar. The formulas are authoritative; code field names are their implementation crosswalk.

### Signal strength, residual coupling, and allocation

Define, where denominators are positive,

$$
\Xi=\frac{\|g_a\|}{R},\qquad
\mu_a^{\rm eff}=\frac{e_H^TG_ae_H}{\|e_H\|^2},\qquad
\lambda^{\rm eff}=\frac{e_H^TSe_H}{\|e_H\|^2},\qquad
\theta_a^{\rm eff}=\frac{\mu_a^{\rm eff}}{\lambda^{\rm eff}}.
\tag{15}
$$

The factorization separates residual size, the effective fitting rate $\lambda^{\rm eff}$, and its allocation to slopes $\theta_a^{\rm eff}$. Their product $\mu_a^{\rm eff}$ summarizes coupling to slopes. In particular,

$$
\|T_ae_H\|=\|e_H\|\sqrt{\mu_a^{\rm eff}}
=\|e_H\|\sqrt{\theta_a^{\rm eff}\lambda^{\rm eff}}.
$$

Here $\lambda^{\rm eff}$ describes decay of the fine residual norm when the tracking and omitted-mode terms in (4) are negligible. It is a residual-direction Rayleigh quotient, not the smallest eigenvalue of $S$. If $e_H$ changes direction, the quotient can change even with a fixed kernel.

**Signal metrics. Read absolute magnitudes together with normalized quantities.**

| Metric | What it tells us; what it does not | Existing source |
|---|---|---|
| $R=\sqrt{2L}$; $\Vert g_a\Vert$ | Remaining error and actual instantaneous slope speed. Large error alone does not imply usable slope signal. | R `state_diagnostics.csv`: `residual_rms`, `grad_a_norm`; T `actual_kernels.csv`: `full_slope_norm`. |
| $\Xi=\Vert g_a\Vert/R$ | Coupling per unit total residual. Its recovery need not increase absolute speed if $R$ falls. | R `xi`; T `runs/*/curves.npz`, trace `xi`. |
| $\Vert T_ae_H\Vert$; $\mu_a^{\rm eff}$ | Remaining effective force and coupling per unit fine residual. Neither determines outward direction. | T kernel tables: `effective_slope_norm`; $\mu_a^{\rm eff}$ is derived using `residual_modes`. |
| $\theta_a=\Vert g_a\Vert^2/\mathcal E$ | Fraction of instantaneous full-field dissipation, where $\mathcal E=\Vert g_a\Vert^2+\Vert g_b\Vert^2+\kappa\Vert g_c\Vert^2+\kappa g_d^2$. A large share can multiply a tiny total. | T traces: `slope_share`; kernel-table `modal_a_share` uses retained modes and is a distinct approximation. |
| $\theta_a^{\rm eff}$; corresponding readout share | Allocation after coarse relaxation. Readout weight and output bias are separate blocks. | T `effective_a_share`, `effective_c_share`, `effective_d_share`; $\lambda^{\rm eff}$ is derived. |
| $v_k=(T_a)_{:,k}e_k$ for $k\in H$ | Which residual modes supply force. Components interfere; their norms do not add. | T `effective_mode3_norm`, `effective_mode9_norm`; `effective_mode*_along_full` equals $g_a^Tv_k/\Vert g_a\Vert^2$. |

The along-full projections can be negative or exceed one; they are signed contributions, not probabilities. Comparing $\|v_9\|$ with $\|T_ae_H\|$ tests direct target coupling versus force generated through other residual modes. Individual modal contributions depend on the chosen basis. Norms and quadratic forms above are invariant under orthonormal rotations within the fixed coarse and fine subspaces, but changing those subspaces changes the decomposition.

### Coarse balance, stability, and approximation error

**Tracking metrics. Small coarse error and accurate tracking of its moving balance are distinct.**

| Metric | Interpretation | Existing source |
|---|---|---|
| $\Vert e_C\Vert$ | How much constant/linear error remains; not a bound on its regeneration. | R `coarse_norm`; T trace `coarse_norm`. |
| $\Vert Ce_C\Vert$, $\Vert Qe_H\Vert$ | Coarse removal and regeneration. Similar large norms suggest a balance but do not establish opposite directions. | T `coarse_fitting_norm`, `coarse_regeneration_norm`. |
| $\Vert Ce_C+Qe_H\Vert$ | Unbalanced retained coarse forcing, before omitted forcing. Tests cancellation directly. | Derived from kernel blocks and residual modes. |
| $\Vert z_C\Vert$, $\tau=\sqrt{z_C^TCz_C}$, $\Vert J_{a,C}^Tz_C\Vert$ | Euclidean tracking, theorem's metric tracking, and its actual slope contribution. They have different normalizations. | T `tracking_error` is Euclidean; $\tau$ is derived; `transient_slope_norm` is the contribution. |
| $\lambda_{\min}(C)$, $\lambda_{\max}(C)$ | Resolution and conditioning of the coarse inverse. | T `coarse_min_eigenvalue`, `coarse_max_eigenvalue`, `coarse_inverse_resolved`. |
| $\alpha_C$; $\Vert\dot C\Vert_F/\lambda_{\min}(C)^2$ | Signed flow-contraction lower bound; conservative sufficient drift check. A ratio below 2 suffices for positive $\alpha_C$. | T `tracking_metric_decay_lower`, `coarse_drift_ratio`. |
| $\Vert f_z\Vert_C$; $\beta_n$, $\Vert h_n^\Delta\Vert_{C_{n+1}}/\eta$ | Regeneration floor and exact GD tracking ingredients. | Flow metric norm is derived: `tracking_forcing_norm` is Euclidean. Discrete fields are `discrete_tracking_factor`, `discrete_tracking_forcing_per_time`. |
| $\Vert g_{a,\perp}\Vert$; projection error | Whether the retained modes explain the slope force. Report absolute error when the force is tiny. | T `omitted_slope_norm`, `projection{9,17,33,65}_relative_error`; trace `slope_defect_norm` uses the run's training projection. |
| Cosine between effective and tracking slope forces | Whether these two vector contributions reinforce or cancel; unrelated to outward alignment with slope signs. | T `slow_transient_alignment`. |

Kernel diagnostics use degree 65 even for a degree-33 forecast; the forecast's training degree and diagnostic degree are different. `omitted_slope_norm` therefore refers to the degree-65 diagnostic tail, whereas the trace's `slope_defect_norm` compares the actual training projection with the full residual at the forecast's own state. A small error at one state does not bound divergence between independently evolved trajectories.

The exported `discrete_tracking_factor` is computed by taking one virtual full-GD step from a selected state. It does not describe the entire gap between saved checkpoints. The five original-seed diagnostic times cannot certify a uniform bound over 600k updates. Also, kernel derivatives in these tables follow the full vector field at the supplied state; on a projected-forecast state they are not derivatives along its projected training field.

### Depletion attribution is different from dissipation allocation

For equal-rate flow, let $\mathcal H_{aI}$ be the loss Hessian block and group hidden parameters as $I=q=(a,b)$. Where $R,\|g_a\|>0$,

$$
-\frac{d}{dt}\log\Xi= D_c+D_d+D_q,
\qquad
D_I=\frac{g_a^T\mathcal H_{aI}g_I}{\|g_a\|^2}
-\frac{\|g_I\|^2}{R^2}.
\tag{16}
$$

This follows by differentiating both $\log\|g_a\|$ and $\log R$. Positive $D_I$ depletes normalized signal; negative $D_I$ regenerates it. The second term accounts for the shrinking residual. It is why raw Hessian action alone does not give normalized-signal depletion.

R `state_diagnostics.csv` exports `D_c`, `D_d`, `D_q`, and splits the readout term into `D_c_filter`, `D_c_amplitude`, and `D_c_normalization`. The first changes the residual through readout fitting, the second changes the readout coefficient multiplying the slope tangent, and the last accounts for residual normalization. They sum to `D_c`. The formula here is for $\kappa=1$; other rates require the corresponding rate factors.

R `early_attribution.csv` integrates these flow diagnostics at GD states, with quadrature and GD-identity discrepancies recorded. It is not an exact discrete causal attribution or a retraining experiment with a parameter block frozen. Ratios of signed contributions become unstable when the total nearly cancels. Neither a large readout dissipation share nor a growing readout norm alone establishes positive $D_c$.

### Direction, concentration, and the population budget

Define outward alignment and mean magnitude by

$$
\chi=-\frac{\operatorname{sign}(a)^Tg_a}{\sqrt W\|g_a\|},\qquad
\bar\gamma=\frac1W\sum_j|a_j|.
$$

Away from zero slopes, flow satisfies $\dot{\bar\gamma}=R\Xi\chi/\sqrt W$. The factors separately identify error, coupling, and outward direction. GD instead requires the exact increment

$$
\Delta\gamma_{jn}=|a_{jn}-\eta g_{a,jn}|-|a_{jn}|.
$$

This handles crossings through zero. Summing its positive and negative parts gives $P_j,N_j$ with the exact identity $P_j-N_j=\gamma_{j,N}-\gamma_{j,s}$.

**Population metrics. These distinguish available force from acquired scales.**

| Metric | Interpretation | Existing source |
|---|---|---|
| $\chi$; $\chi_{\rm eff}=-\operatorname{sign}(a)^TT_ae_H/(\sqrt W\Vert T_ae_H\Vert)$ | Outward alignment of the full and effective forces. Positive values concern the mean, not every neuron. | R `signed_alignment`; T `effective_outward_alignment`. |
| Exact $\Delta\bar\gamma$, positive/negative travel, crossing correction | Actual GD motion, including sign crossings and reversal. Large positive travel can be undone later. | R `delta_mean_gamma`, `crossing_remainder`; `movement_windows.csv`: `positive_travel`, `negative_travel`; T states retain per-node `positive`, `negative` and cumulative mean `crossing`. |
| $\big(\sum_jP_j\big)^2/(W\sum_jP_j^2)$; top-10% share | Participation fraction and concentration of positive travel. Uniform positive travel gives participation 1; one moving neuron gives $1/W$. | R movement windows: `positive_participation_fraction`, `positive_top10_share`; separate `net_positive_*` fields use final net growth. |
| $P_\Gamma$; mean, median, upper quantile, maximum gamma | Actual outcome and distribution shape. Mean/maximum alone cannot establish population acquisition. | R state diagnostics; T endpoints: `fraction_1`, `fraction_3.2`, `fraction_16`, `mean_gamma`, `max_gamma`, `gamma_second_moment`. |
| $B_{s,N}=\eta\sum_n\Vert g_{a,n}\Vert$; $\mathcal D$; margin $\mathcal D-B$ | Exact slope path and distance to acquisition. Positive margin gives retrospective exclusion on a verified computed trajectory. | R `gradient_path_budget`; T `budgets.csv`: `path`, `distance`, `path_excludes`; margin is derived. |
| $E_{s,N}=\sqrt{\Delta t\,\eta\sum_n\Vert g_{a,n}\Vert^2}$ | Cauchy–Schwarz upper bound on path. It can be much looser than the path itself. | T `energy_bound`, `energy_excludes`; `slope_energy_share` is an integrated share, not an endpoint share. |

The count and concentration formulas above are for finite networks. For law quadrature, use probability masses rather than treating nodes as equally weighted neurons. For example, with weighted force $\widetilde g_{a,i}=\sqrt{Ww_i}g_a(z_i)$, outward alignment is $-\sum_i\sqrt{w_i}\operatorname{sign}(a_i)\widetilde g_{a,i}/\|\widetilde g_a\|$.

The older R tables also use construction-scale aliases $\lambda=(2/N_{\rm ref})\gamma$. With $N_{\rm ref}=128$, `fraction_lambda_005` and `fraction_lambda_025` mean fractions above raw gamma 3.2 and 16. The construction budget $N_{\rm ref}$ is different from physical width $W=177$ and update index $N$.

All budget windows must use matching endpoints and physical time. Do not integrate sparse plotted gradient norms to replace the every-update path accumulator. Likewise, a flow loss-dissipation identity cannot replace $\eta\sum\|g_a\|^2$ in GD without a discrete descent estimate.

For a density, threshold acquisition is governed by outward flux at $a=\pm\Gamma$, not the global mean velocity. More generally, for a smooth threshold approximation $H_\varepsilon$,

$$
\frac{d}{dt}\int H_\varepsilon(|a|-\Gamma)\,d\rho
=\int H_\varepsilon'(|a|-\Gamma)\operatorname{sign}(a)v_a\,d\rho.
$$

This is a useful additional diagnostic, but no dedicated threshold-flux column is currently exported. For atomic GD, exact counts and step increments avoid assuming a classical density. Smoothing should also remove the cusp at zero, or place its transition away from zero and interpret the flow formula almost everywhere.

### Moment growth and numerical trust

The per-neuron layer-balance quantity $I=a^2+b^2-c^2/\kappa$ obeys

$$
I_{n+1}-I_n=2\eta c_n\langle r_n,\tanh u_n-u_n\operatorname{sech}^2u_n\rangle_m
+\eta^2(g_{a,n}^2+g_{b,n}^2-\kappa g_{c,n}^2).
\tag{17}
$$

The flow part vanishes for affine activation but has no fixed sign for tanh. It measures a route by which readout and hidden norms can separate. T states store width-weighted sums as `balance_flow` and `balance_discrete`; endpoints store `balance_error`. Projected training uses its projected residual in this identity. This is an accounting check and a source term to study, not an assumed invariant.

Track moments of all rescaled parameters $\sqrt W(a,b,c)$ when testing the small-feature expansion. The exported gamma second moment is only one of those moments; higher joint moments must be reconstructed from states. For any $q\ge1$, the population bound $P_\Gamma\le\mathbb E|a|^{2q}/\Gamma^{2q}$ is valid, but predicting the moment remains a separate closure problem.

T `paired_errors.csv`, `refinement.csv`, and `law_errors.csv` distinguish realized-network forecast error, numerical refinement, and law-to-network discrepancy. An observed Wasserstein distance is not a certified future error radius. If such a radius $\varepsilon$ were independently justified, the gamma-marginal coupling bound would give, for $0<d_0<\Gamma$,

$$
P_\Gamma\le\widehat P_{\Gamma-d_0}+\varepsilon^2/d_0^2.
$$

At zero residual, zero force, or zero dissipation, the corresponding normalized quantities are undefined; inspect the unnormalized numerators. Some diagnostic code uses a tiny denominator safeguard, which must not be interpreted as resolving a ratio below numerical accuracy. Honor `coarse_inverse_resolved` and `attribution_resolved`; do not insert an arbitrary inverse regularizer and call it the same theorem. Small negative computed eigenvalues of theoretically positive Gram matrices require numerical-error checks before interpretation.

## 6. What the width prediction adds

The matched polynomial targets are $y_k=0.3q_0+0.4q_1+\sqrt{0.75}\,q_k$, with $k=3$ or $9$. They have identical affine coefficients and unit empirical RMS; their non-affine target content differs. D34's sine target is $y(x)=\sin(2\pi x)$, which also has low-order non-affine coefficients. This target structure is essential to the following comparison.

With $(A,\widetilde B,\widetilde C)=\sqrt W(a,b,c)$, bounded sufficiently high joint moments give the formal expansion

$$
F=d+\mathbb E[\widetilde C(Ax+\widetilde B)]
-\frac1{3W}\mathbb E[\widetilde C(Ax+\widetilde B)^3]+\cdots.
$$

The leading model is affine. The network's first non-affine output term is order $W^{-1}$; it can create lower-mode residuals even when those modes are absent from the target. This does not make the full unfitted target residual small. For modes 2 and 3, the slope tangent of one neuron first appears at order $W^{-3/2}$; summing squared contributions over $W$ neurons gives population norm $W^{-1}$. Coarse elimination contributes at the same order and must be retained.

An order-one degree-3 target coefficient therefore gives an effective force of proposed scale $W^{-1}$. For the degree-9 target, direct coupling first appears in the eighth-order term of $x\operatorname{sech}^2(ax+b)$, giving population scale $W^{-4}$. The network's generated lower-mode residual, however, is order $W^{-1}$ and acts through an order-$W^{-1}$ effective tangent. Its contribution is therefore order $W^{-2}$ and can dominate the direct target force.

**Formal regime predictions. These require controlled moments, resolved coarse relaxation, and no cancellation eliminating the proposed leading term.**

| Forcing source | Population slope-force scale | Candidate time for order-one rescaled slope change |
|---|---:|---:|
| Low-order non-affine target, including sine/degree 3 | $W^{-1}$ | $W$ |
| Generated lower-mode error for degree 9 | $W^{-2}$ | $W^2$ |
| Direct degree-9 coupling alone | $W^{-4}$ | $W^4$ |

Indeed the RMS rescaled-slope velocity is $\sqrt{W^{-1}\sum_j\dot A_j^2}=\|g_a\|$. This translates the force powers into relative geometry-change times. It does not give the time to reach a fixed large $\Gamma$: that would leave the bounded-rescaled-parameter regime. The powers are formal leading-order predictions, not proved lower bounds or a uniform trapping theorem.

At $W=177$, $T=1200$ gives $T/W=6.78$ but $T/W^2=0.0383$. This predicts a separation between possible nonlinear recovery for low-order targets and weak motion for degree 9. Test it with $W\|g_a\|$ and $W^2\|g_a\|$ at matched post-transient times, alongside tracking and moment diagnostics. Do not infer the scaling from a fitted exponent alone.

## 7. Read the existing D34 results through these metrics

The primary examples use actual equal-rate GD, random readouts, $W=177$, $\eta=0.002$, and 2,048 training samples. The kernel table evaluates the original five seeds at five selected states. These are training-dynamics diagnostics, not held-out generalization measurements. Fresh-seed validation tests predictions on new initializations of the same tasks.

**Endpoint medians at update 600k. Effective shares use the degree-65 modal decomposition at actual GD states; force norms use the full residual. Each column is aggregated separately.**

| Target | Full slope-force norm | Effective slope share | Effective readout-weight share | Effective outward alignment |
|---|---:|---:|---:|---:|
| Sine | 0.01194 | 87.22% | 12.77% | 0.118 |
| Degree 3 | 0.0005921 | 4.16% | 92.94% | 0.345 |
| Degree 9 | $7.504\times10^{-6}$ | 67.36% | 12.31% | −0.135 |

For sine, substantial recovered force coexists with modest outward alignment and only isolated large slopes. For degree 3, the late effective fitting effort goes mostly to readout weights, although slopes underwent earlier recovery. For degree 9, slopes receive most of a very small total; their large share is compatible with negligible motion. Negative effective alignment indicates decreasing mean magnitude from that force, not a failure to decrease loss. Hidden and output biases account for additional shares.

<figure>
  <img src="../results/checkpoint_D_optimizers/expD34_readout_race/transport_barrier/actual_effective_force.png" alt="Full, effective, and coarse-tracking slope-force norms and slope dissipation shares for five actual D34 seeds and three targets" style="max-width: 100%;">
  <figcaption>Actual D34 states at five selected times, with lines connecting observations. The effective force follows the full force after coarse relaxation. The lower panels show modal slope dissipation shares, whose magnitude alone does not determine the remaining slope speed. Sparse connecting lines do not resolve every regeneration episode or prove uniform contraction.</figcaption>
</figure>

**Population budget example. Medians for original-seed degree-33 forecasts over updates 20k–600k, with $p=0.1$ and $\Gamma=3.2$. These are measured budgets, not independently established envelopes.**

| Target | Acquisition distance $\mathcal D$ | Exact accumulated path $B$ | Slope-energy bound $E$ |
|---|---:|---:|---:|
| Sine | 12.825 | 4.242 | 9.035 |
| Degree 3 | 12.848 | 5.809 | 14.108 |
| Degree 9 | 12.811 | 0.009432 | 0.009442 |

For degree 3, the energy bound is too loose while the path excludes acquisition. This is why replacing the force history by total energy can lose the desired conclusion. All original-seed actual replay paths also exclude acquisition; their budgets differ from the paired forecasts by at most $9.30\times10^{-7}$, as recorded in T `verification/actual_path_comparison.csv`. A theorem applies case by case; the median table is illustrative.

The 15 fresh forecasts correctly predict all measured threshold counts, including one sine neuron above 3.2 per fresh seed. These numerical forecasts retain neuron parameters; they do not close the dynamics in a few residual coordinates. No tested rate acquires the primary 10% population at 3.2, so the campaign does not demonstrate all three proposed outcome classes at that threshold.

Rate interventions also constrain the explanation: slowing readout increases degree-3 mean gamma but decreases sine mean gamma at the same horizon. Readout can regenerate useful signal as well as remove it. The initial normalized-signal attribution is approximately one-half to readout weights in the prior audit, not the larger zero-readout fraction from a different initialization.

The width-operator check approaches powers 1, 1, and 2 at an affine-law reference. Actual finite-seed exponents over widths 89–353 are approximately 0.81, 0.72, and 1.68 and vary across seeds. Together these support a regime prediction while leaving uniform moment control and finite-width transfer unresolved.

## 8. A practical reading order

1. **Specify the event and interval.** Fix $(p,\Gamma)$, starting state, horizon, physical rates, and whether the data are actual GD, a projected forecast, or law quadrature. Inspect counts and distributions before naming a barrier.
2. **Separate error from signal.** Read $R$, $\|g_a\|$, and $\Xi$ together. For the effective force, separate residual size, coupling $\mu_a^{\rm eff}$, and allocation $\theta_a^{\rm eff}$.
3. **Check the coarse explanation.** Compare removal and regeneration vectors, tracking contribution, and omitted force. A small coarse residual or positive sampled contraction rate alone is insufficient.
4. **Identify what changes the signal.** Use signed $D_I$ for normalized depletion/regeneration, and effective modal contributions for surviving force. Use rate interventions to test dynamical consequences; do not equate instantaneous attribution with an intervention.
5. **Follow motion into the population.** Inspect alignment, positive/negative travel, concentration, and threshold counts. Recovery without acquisition can involve insufficient total motion, reversal, concentration, or a combination.
6. **Choose the strength of the conclusion.** An exact measured path below $\mathcal D$ gives retrospective exclusion, subject to numerical verification. Independent forecasts give predictive evidence. Independently justified envelopes in (10)–(14) give a theorem. Each is useful, but they answer different questions.

The mathematical assumptions identify quantities that need control, and failures of those assumptions identify mechanisms worth studying. The metrics remain useful even when the proof cannot yet close: they show whether the obstruction is weak coupling, readout allocation, renewed coarse forcing, poor direction, concentrated motion, or an inadequate approximation.

## Sources and reconstruction

Definitions and implementations are in [transport.py](../experiments/expD34_readout_race/transport.py), [transport_analyze.py](../experiments/expD34_readout_race/transport_analyze.py), and [recovery.py](../experiments/expD34_readout_race/recovery.py). The [earlier scale-acquisition note](d34_scale_acquisition_theory.md) derives signed motion, attribution, and the exact distance formula. The [transport theory](d34_transport_scale_barrier.md) supplies the flow and discrete tracking identities used here. This walkthrough adds their explicit constant-bound population corollary and connects them to the exported measurements.

T `actual_kernels.csv` contains full-field diagnostics of the verified original actual-GD replay. T `kernels.csv` identifies each additional run, training projection, width, and rate. R tables use the original actual replay. Full-field diagnostics on forecast states describe the original GD vector field at those parameters, while the forecast's updates and cumulative budgets use its projected training field. Do not combine the two as an exact trajectory identity when projection error is appreciable.

The selected states can be evaluated with `transport.modal_diagnostics(..., degree=65, kappa=...)` to reconstruct modal Jacobians, residual coefficients, kernel blocks, and $\dot K$; use `transport_analyze.diagnostics(..., eta=...)` for the added discrete-step scalars. The latter routine returns the same underlying arrays for deriving $\tau$, $\mu_a^{\rm eff}$, and $\lambda^{\rm eff}$.

For example, reconstruct $C=K_{CC}$, $Q=K_{CH}$, solve $CB=Q$, and form $z_C=e_C+Be_H$, $T_a=J_{a,H}^T-J_{a,C}^TB$, and $S=K_{HH}-Q^TB$. This yields the derived metrics without fitting coefficients to future data. Do not silently mix values from different states, bases, or rate conventions. The evidence report includes reproduction commands, validation logs, prediction-freeze records, and numerical-error controls.

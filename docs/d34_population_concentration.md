# Persistence from population concentration and target moments

The question is why a wide population keeps learning its slopes slowly after
the coarse tracking transient. A bound on future force reinforcement leaves
part of that question inside its premise. Here the premise concerns how
parameter energy is distributed across neurons. We derive the fine sensitivity
and population-growth bounds from that structure, using exact tanh inequalities
and fixed target moments. Quantitative usefulness is a separate empirical test.

| Symbol | Meaning |
|---|---|
| $a_j,b_j,c_j$ | Physical slope, hidden bias, and readout of neuron $j$; $a_j=\gamma_j$. |
| $r_j^2=a_j^2+b_j^2+c_j^2$, $M=\sum_jr_j^2$ | Individual and total hidden parameter energy. |
| $S_k=\sum_jr_j^k$, $\chi_k=W^{k/2-1}S_k/M^{k/2}$ | Population moment and its dimensionless concentration. |
| $P_C,P_H$ | Orthogonal projections onto affine outputs and their complement. |
| $g=P_Hy$, $f_H=P_Hf$, $e_H=f_H-g$ | Fine target, output, and residual. |
| $F,R$ | Effective fine gradient, including compensation, and tracking gradient. |
| $E_0$ | Initial fine-residual norm for effective flow; initial full-residual norm for GD. |
| $\tau_k=\|P_{\mathcal P_k}g\|$ | Fixed target loading through polynomial degree $k$. |
| $h=2/N_{\rm ref}$, $\lambda_{\rm RMS}=h\|a\|/\sqrt W$ | Reference spacing and normalized population slope scale. |

## 1. An assumption about organization, rather than motion

Suppose equal parameter energy is spread over all $W$ neurons. Then
$\chi_6=1$. If the same total energy is equally spread over $k$ active
neurons, $\chi_6=(W/k)^2$. This motivates the effective count
$W/\sqrt{\chi_6}$: it measures how many neurons share the energy in this
third-moment sense. It is not a literal count of neurons with nonzero weights.

Multiplying every hidden parameter by any common factor changes $M$ and the
slopes, but leaves $\chi_6$ unchanged. A bound on concentration therefore
permits arbitrarily large common scales. It also permits individual outliers;
their energy contributes to the aggregate moment. We assume no upper bound on
any individual neuron and no maximum-over-time concentration bound.

The structural hypothesis is that concentration does not accumulate too
quickly. The ODE will then bound growth of total energy. This is still a future
structural condition. Proving that the ODE itself preserves that condition is
a further question; it is not supplied by observing concentration at the start.

## 2. Derive the sensitivity before making a persistence assumption

Use a probability measure on $[-1,1]$ and the network
$f_\theta=d+\sum_jc_j\tanh(a_jx+b_j)$. Let $J_C=D(P_Cf)$,
$J_H=Df_H$, and assume $J_CJ_C^*$ is nonsingular. The exact decomposition is

$$
\Pi=I-J_C^*(J_CJ_C^*)^{-1}J_C,\qquad
F=\Pi J_H^*e_H,\qquad R=\nabla L-F,
\quad L=\tfrac12\|f-y\|^2.
$$

$\Pi$ is an orthogonal projection. Coarse compensation is included in $F$;
we do not assume it is small. Effective flow is $\dot\theta=-F$, and
ordinary gradient flow is $\dot\theta=-F-R$.

**Lemma 1 — concentration controls nonlinear sensitivity.** For exact tanh,

$$
\|J_H\|\le C_3\sqrt{S_6},\qquad
\|f_H\|\le A_3 S_4,
\qquad C_3=\frac{\sqrt6}{2},\quad A_3=\frac{\sqrt6}{8}.
\tag{1}
$$

**Proof.** Subtract the affine output $\sum_jc_j(a_jx+b_j)$ before projecting.
The global inequalities $|\tanh u-u|\le|u|^3/3$ and
$|\operatorname{sech}^2u-1|\le u^2$ bound its output and derivative.
At fixed $r^2=a^2+b^2+c^2$, use $|ax+b|\le\sqrt2\sqrt{a^2+b^2}$
and maximize the resulting products. The output bound is $A_3r^4$;
the squared sum of its derivative-column bounds is $C_3^2r^6$.
Summing outputs and squared columns proves (1). Orthogonal projection cannot
increase these norms. These are global inequalities, not a small-argument
assumption or a polynomial substitute for the dynamics. $\square$

Along effective flow, $(\|e_H\|^2)'=-2\|F\|^2$, so
$\|e_H(t)\|\le E_0$. Full gradient flow has decreasing full residual energy,
which supplies the same bound using the full residual at the start. Therefore

$$
\boxed{\|F\|\le \frac{C_3E_0}{W}\sqrt{\chi_6}\,M^{3/2}.}
\tag{2}
$$

This is the mechanism supplied by the assumptions: distributed parameter
energy produces small nonlinear sensitivity, and dissipation prevents the
residual from supplying an increasing norm multiplier. No measured future
force or reinforcement is used in (2).

## 3. Close the population-growth inequality

**Theorem 1 — a bound from accumulated concentration.** Start with $M_0>0$.
For effective flow, define

$$
B_6(t)=\int_0^t\sqrt{\chi_6(s)}\,ds,\qquad
D(t)=1-\frac{2C_3E_0M_0}{W}B_6(t).
$$

On every interval where $D(t)>0$,

$$
M(t)\le\frac{M_0}{D(t)},\qquad
\lambda_{\rm RMS}(t)\le h\sqrt{\frac{M_0}{W D(t)}}.
\tag{3}
$$

**Proof.** Since $M$ excludes only the output bias,
$\dot M\le2\sqrt M\|F\|$. Substitution of (2) gives
$\dot M\le2C_3E_0\sqrt{\chi_6}M^2/W$. Divide by $M^2$ and integrate
the derivative of $-1/M$ to obtain (3). A first-exit argument gives the same
conclusion without assuming in advance that the mass stays bounded.
The slope bound follows from $\sum_j a_j^2\le M$. $\square$

For example, if $B_6(t)\le Kt$, order-one population growth requires flow time
of order $W/(E_0M_0K)$. This is a conditional width scaling, not a universal
update count. The denominator can eventually vanish. The theorem asserts
slow growth while its budget lasts; it does not assert equilibrium.

**Tracking and native GD.** Let $S_R(t)$ bound accumulated tracking travel
$\int_0^t\|R\|$. Set $b_0(t)=\sqrt{M_0}+S_R(t)$. Then (3) generalizes to

$$
\sqrt{M(t)}\le
\frac{b_0(t)}{\sqrt{1-2C_3E_0b_0(t)^2B_6(t)/W}}.
\tag{4}
$$

For GD, replace both integrals by left-point sums at the actual iterates and
assume the full-residual norm is at most $E_0$. This is implied by loss descent;
it is a distinct optimizer condition, not a consequence of concentration.

**Proof of transfer.** For a fixed endpoint, put all allowed tracking travel
into the initial radius. Subtracting the tracking already spent leaves a
nonnegative reserve. The radius plus its reserve satisfies
$\dot b\le a(t)b^3$, with $a=C_3E_0\sqrt{\chi_6}/W$.
For GD, the exact triangle inequality gives
$b_{n+1}\le b_n+\eta_na_nb_n^3$. The exact positive solution of
$\dot b=a_nb^3$ over a step dominates its Euler increment. Composition of
these exact steps gives (4) with $B_6=\sum_n\eta_n\sqrt{\chi_{6,n}}$.
This proof needs no trajectory-closeness approximation. $\square$

## 4. Use the affine projection to avoid wasting the bound

The generic constant in (1) bounds terms that the fine projection removes.
For the symmetric input measures in these experiments, only two cubic output
shapes remain. Define

$$
U_2=x^2-\mu_2,\qquad U_3=x^3-\frac{\mu_4}{\mu_2}x,
\quad s_2=\|U_2\|^2,\quad s_3=\|U_3\|^2,
\quad \mu_k=\langle x^k,1\rangle.
$$

These shapes are orthogonal. For $\Psi_3=-\sum_jc_j(a_jx+b_j)^3/3$,

$$
P_H\Psi_3=-\Big(\sum_jc_ja_j^2b_j\Big)U_2
 -\frac13\Big(\sum_jc_ja_j^3\Big)U_3.
\tag{5}
$$

**Lemma 2 — sharper population constants.** Put

$$
D_3^2=\frac8{27}s_2+\frac3{16}s_3,\qquad
B_3^2=\frac1{64}s_2+\frac3{256}s_3.
$$

Then, with $J_3=D(P_H\Psi_3)$,

$$
\|J_3\|\le D_3\sqrt{S_6},\qquad
\|P_H\Psi_3\|\le B_3S_4.
\tag{6}
$$

**Proof.** Differentiating (5) gives the exact Hilbert–Schmidt identity

$$
\|J_3\|_{\rm HS}^2=
s_2\sum_j(4c_j^2a_j^2b_j^2+c_j^2a_j^4+a_j^4b_j^2)
s_3\sum_j(c_j^2a_j^4+a_j^6/9).
$$

Writing $A=a^2,B=b^2,C=c^2$, with $A+B+C=r^2$, the first bracket is at
most $8r^6/27$ and the second at most $3r^6/16$. For the output of one
neuron, the squared norm is $s_2CA^2B+s_3CA^3/9$;
$CA^2B\le r^8/64$ and $CA^3/9\le3r^8/256$.
Sum squared derivative norms and use the triangle inequality for outputs.
$\square$

Under the uniform continuous measure, $s_2=4/45$ and $s_3=4/175$.
The empirical midpoint measure supplies its own exact moments. Thus the
smaller constants come from the input geometry, not from fitting a force curve.

The exact tanh remainders give

$$
\|J_H-J_3\|\le C_5\sqrt{S_{10}},\qquad
\|f_H-P_H\Psi_3\|\le A_5S_6,
\tag{7}
$$

where $A_5=\frac2{15}\frac{2^{5/2}5^{5/2}}{6^3}$ and $C_5=6A_5$.
Consequently both the generic bound and
$E_0(D_3\sqrt{S_6}+C_5\sqrt{S_{10}})$ bound the force. Their minimum is
also a valid bound. This refinement introduces an accumulated higher moment;
it does not impose a maximum-neuron cutoff.

## 5. Target moments identify the available drive

Let $\tau_k=\|P_{\mathcal P_k}g\|$. Every column of $J_3$ belongs to
the quadratic–cubic output space. Decomposing the exact force before applying
norm inequalities gives

$$
\|F\|\le D_3\sqrt{S_6}(\tau_3+Q)
 +C_5E_0\sqrt{S_{10}},\qquad
Q=\min\{A_3S_4,\ B_3S_4+A_5S_6\}.
\tag{8}
$$

**Proof.** Write $J_H^*e_H=J_3^*f_H-J_3^*g+(J_H-J_3)^*e_H$.
Bound the three terms with (6), the polynomial target projection, and (7).
Apply the contractivity of $\Pi$. $\square$

The first contribution in (8) separates target loading from generated output.
The fixed target quantity $\tau_3$ vanishes for a gap through degree three;
it is retained for sine and localized targets. For bounded dimensionless
moments, bounded $M$, and $\tau_3=0$, this bound is order $W^{-2}$ rather
than $W^{-1}$. The gap alone does not guarantee that those moments persist.

An additional comparison retains $J_5=D(P_H\Psi_5)$, where
$\Psi_5=2\sum_jc_j(a_jx+b_j)^5/15$. Its force allowance is

$$
D_3\sqrt{S_6}\tau_3+C_5\sqrt{S_{10}}\tau_5
 +(D_3\sqrt{S_6}+C_5\sqrt{S_{10}})Q
 +C_7E_0\sqrt{S_{14}},
\tag{9}
$$

with $A_7=\frac{17}{315}\frac{2^{7/2}7^{7/2}}{8^4}$ and $C_7=8A_7$.
This follows from the same argument with the global seventh-degree derivative
remainder. No target-specific constant is fitted.

**Growing comparison.** Substitute $S_k=\chi_k b^k/W^{k/2-1}$ into any of
these force bounds to obtain $V(t,b)$. Solve
$\dot b=V(t,b)+\|R\|$, starting at $\sqrt{M_0}$. Each bound, and their
minimum, is nondecreasing in $b$. Scalar comparison proves $\sqrt M\le b$;
native GD uses $b_{n+1}=b_n+\eta_n[V(n,b_n)+\|R_n\|]$.
The coefficients describe the evolving population. None is a frozen Jacobian
or a measured future effective force.

**Output consequence.** The same radius gives, at each time,

$$
\frac{\|f-y\|}{\|y\|}\ge
\frac{[\|g\|-Q(t,b(t))]_+}{\|y\|}.
\tag{10}
$$

For effective flow there is also the energy floor
$\|e_H(t)\|^2\ge[\|e_H(0)\|^2-2\int_0^tV(s,b(s))^2ds]_+$.
The capacity inequality (10) applies directly to GD without a separate
energy-defect allowance. The empirical norm is the training-grid norm;
continuous-measure conclusions require their own quadrature control.
The reference $\lambda=0.25$ is a construction benchmark, not a necessary
accuracy threshold. The output conclusion does not depend on choosing it.

## 6. What these statements explain, and what they still assume

The conclusion follows from three independently described ingredients:
distributed parameter energy, weak nonlinear sensitivity after removing the
affine output, and limited target loading of the surviving low-degree shapes.
Residual dissipation bounds the available error multiplier. Coarse compensation
is retained in the exact projected dynamics.

This is conditional on the accumulated population structure. It does not prove
that concentration remains moderate. It also does not establish universal
dominance of generated-error correction: earlier signed-balance measurements
show target-driven growth can be larger. Equations (8)–(9) safely allow that
growth. A further signed refinement must respect those observations.

The numerical test compares the generic concentration bound, the analytic
projection refinement, and the target-moment refinement. The earlier bound
using measured polynomial population coherences is a separately labeled
reference. Its better performance, if observed, identifies structure that the
concentration-only hypotheses discard; it is not evidence that the simpler
theorem already explains the same interval.

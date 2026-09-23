# A transport model for changing geometry under an almost fixed error load

The polynomial surrogate gives a concrete answer to what should move in the
transport PDE. At small parameters, the leading particles carry rescaled
slopes, biases and readouts. Their velocity depends on the distribution's
second and fourth moments, even when the driving target-error moments remain
constant. This makes geometry feedback explicit without assuming that the
effective sensitivity map stays fixed.

The leading model below has exact conservation and dissipation identities.
Its connection to tanh GD is a **conditional finite-time approximation** on
a bounded, coarsely conditioned domain. It is not an unconditional
infinite-width limit or a proof that training enters that domain.

**Notation.**

| Symbol | Meaning |
|---|---|
| $W$, $h=2/N_{\rm ref}$ | Actual neuron count and construction-scale normalization; they are distinct |
| $X=(\alpha,\beta,\zeta)=\sqrt W(a,b,c)$ | Rescaled coordinates of one neuron |
| $\tau=\eta n/W$ | Time for the leading small-parameter dynamics |
| $\rho_W=W^{-1}\sum_j\delta_{X_j}$ | Empirical probability measure of neurons |
| $v=\langle x^2\rangle_m$ | Empirical input variance on the symmetric grid |
| $m_2,m_3$ | Fixed quadratic and cubic moments of the negative fine target |

## 1. Example: geometry can change while its driving errors hardly do

Suppose $a,b,c$ remain of order $W^{-1/2}$, with bounded rescaled coordinates.
Then the network's fine output is $O(W^{-1})$. Its fine residual is therefore
$-y_H+O(W^{-1})$, although the coefficients converting that residual into
slope motion can evolve. This is the regime described by
[the small-parameter theorem](d34_mechanism_rate_theorems.md#8-a-width-dependent-rate-without-freezing-the-effective-map),
not a claim inferred merely from a small frozen-model relaxation clock.

Use the full empirical orthogonal complement of $\{1,x\}$ as the fine space,
write $y_H=P_Hy$, and assume a symmetric grid with $v>0$ and $|x_i|\le1$.
Define the fixed numbers

$$
m_2=-\langle y_H,x^2\rangle_m,\qquad
m_3=-\langle y_H,x^3\rangle_m.
$$

These are raw polynomial moments, not necessarily normalized orthogonal-mode
coefficients. The cubic Taylor term supplies the leading raw fine gradient,
after its common factor $W^{-3/2}$ is removed:

$$
G(X)=\begin{pmatrix}
-2\zeta\alpha\beta m_2-\zeta\alpha^2m_3\\
-\zeta\alpha^2m_2\\
-\alpha^2\beta m_2-\alpha^3m_3/3
\end{pmatrix}.
\tag{1}
$$

The components correspond to slope, bias and readout. They are the gradient
of the quartic function
$U(X)=-m_2\zeta\alpha^2\beta-m_3\zeta\alpha^3/3$.
Equation (1) is not the final force: the coarse compensation must still be
included.

## 2. Theory: the distribution determines its own compensating velocity

For expectation with respect to $\rho$, form

$$
K[\rho]=\begin{pmatrix}
1+\mathbb E(\zeta^2+\beta^2)&\sqrt v\,\mathbb E(\alpha\beta)\\
\sqrt v\,\mathbb E(\alpha\beta)&v\mathbb E(\zeta^2+\alpha^2)
\end{pmatrix},\qquad
s[\rho]=\begin{pmatrix}
\mathbb E(\zeta G_\beta+\beta G_\zeta)\\
\sqrt v\,\mathbb E(\zeta G_\alpha+\alpha G_\zeta)
\end{pmatrix}.
$$

Assume $K\succeq\kappa I$ and put $\ell=(\ell_0,\ell_1)^T=K^{-1}s$.
The balanced force and characteristic equations are

$$
A(X;\rho)=\begin{pmatrix}
G_\alpha-\sqrt v\,\zeta\ell_1\\
G_\beta-\zeta\ell_0\\
G_\zeta-\beta\ell_0-\sqrt v\,\alpha\ell_1
\end{pmatrix},\qquad
\frac{dX}{d\tau}=-A(X;\rho),\quad\frac{dd}{d\tau}=\ell_0.
\tag{2}
$$

The leading coarse outputs are $d+\mathbb E(\zeta\beta)$ and
$\sqrt v\,\mathbb E(\zeta\alpha)$. Their parameter Jacobian has Gram matrix
$K$. Multiplying the raw fine force by that Jacobian produces $s$; solving
$K\ell=s$ therefore removes its coarse motion. The entry 1 in $K_{00}$
comes from the output bias. Dropping that coordinate would change the model.

The empirical measure of these characteristics satisfies

$$
\partial_\tau\rho+\nabla_X\!\cdot(\rho V[\rho])=0,
\qquad V[\rho](X)=-A(X;\rho).
\tag{3}
$$

For finitely many neurons this statement is exact in the weak sense:
$d\int\phi\,d\rho/d\tau=\int\nabla\phi\cdot V\,d\rho$.
No smooth density, added noise or diffusion is required. The PDE is nonlinear
through the second moments in $K$ and fourth moments in $s$. Keeping $m_2,m_3$
fixed consequently does not freeze the velocity. In general, the evolution
of fourth moments involves sixth moments: $K$ and $s$ alone do not close the
dynamics. The evolving particle distribution retains these correlations.

### Two exact identities explain what the compensation does

Along (2), direct differentiation and $K\ell=s$ give

$$
\frac{d}{d\tau}\{d+\mathbb E(\zeta\beta)\}=0,
\qquad
\frac{d}{d\tau}\mathbb E(\zeta\alpha)=0.
\tag{4}
$$

These are leading affine-output constraints, not conservation of every tanh
coarse coefficient. Readouts and geometry may change substantially while
preserving their aggregate products. They cannot generally be replaced by a
single readout-size variable.

The same calculation gives

$$
\frac{d}{d\tau}\mathbb E U
=-\mathbb E|G|^2+\ell^Ts
=-\mathbb E|A|^2-\ell_0^2\le0.
\tag{5}
$$

Thus this model is a constrained gradient motion of the leading quartic
potential. Its dissipation does not imply outward slopes: $-A_\alpha$ has
no fixed sign relative to $\alpha$. The quartic potential is not generally
convex or coercive, so (5) alone proves neither trapping nor bounded support.

## 3. Conditional consistency with tanh GD

On a fixed compact domain in $X$, with bounded target norm and both the
affine matrix $K$ above and the tanh coarse Gram uniformly conditioned, the Taylor bounds in the
[polynomial theorem](d34_polynomial_surrogate_theorem.md) prove uniformly

$$
F_{a,b,c,j}^{\rm tanh}=W^{-3/2}A(X_j;\rho_W)+O(W^{-5/2}),
\qquad F_d^{\rm tanh}=-W^{-1}\ell_0+O(W^{-2}).
\tag{6}
$$

Here $F^{\rm tanh}$ is the full-complement effective gradient, without coarse
tracking. The raw cubic truncation has the stated remainder; replacing the
fine residual by $-y_H$ costs the same order because $P_Hf=O(W^{-1})$.
The coarse Jacobian differs from its affine expression by $O(W^{-1})$ in
operator norm. Inverse-Gram stability then gives (6), including the
compensating term. This is a remainder estimate, not just a formal expansion.

With $\Delta\tau=\eta/W$, actual GD consequently satisfies

$$
X_{j,n+1}=X_{j,n}+\Delta\tau\{
V[\rho_{W,n}](X_{j,n})+\varepsilon_{j,n}-\mathcal R_{j,n}\},
\quad |\varepsilon_{j,n}|\le C/W,\quad
\mathcal R_{j,n}=W^{3/2}(r_C)_{a,b,c,j,n}.
$$

Thus the errors to control are the Taylor remainder, Euler discretization
and rescaled tracking. For the output-bias equation the corresponding
tracking term is $W(r_C)_d$; small slope tracking alone does not control it.

On the compact conditioned domain, bounded polynomial derivatives and the
identity for differences of $K^{-1}$ make the velocity Lipschitz in both
particle position and distribution. Couple equal-weight particles with the
same labels. If $E_n$ is their maximum position discrepancy from (2), then

$$
E_{n+1}\le(1+L\Delta\tau)E_n+
\Delta\tau\{C/W+\max_j|\mathcal R_{j,n}|\}
+C_E(\Delta\tau)^2.
\tag{7}
$$

For $n\Delta\tau\le T$, iteration bounds the error by
$e^{LT}[E_0+CT/W+C_ET\Delta\tau+
\sum_{k<n}\Delta\tau\max_j|\mathcal R_{j,k}|]$.
This is a finite-particle consistency statement; it does not require taking
a mean-field limit. An analogous coupling gives stability in the first
Wasserstein distance for bounded-support solutions of (3).
To close the hypotheses, place the reference inside a larger conditioned
domain and verify that this error bound fits its remaining margin, then use
first exit. Observing the future path inside that domain is not such a proof.

## 4. Prediction: a nonlinear transport mechanism, with a separate higher-order clock

If characteristic support remains bounded by $|\alpha|\le R$ through time
$T$, its normalized slopes satisfy $\lambda=h|\alpha|/\sqrt W\le hR/\sqrt W$.
Adding the controlled discrepancy from (7) gives an acquisition exclusion for
the actual particles whenever $h(R+E_n)/\sqrt W<\lambda_*$ throughout the
window. Bounded support is essential; it cannot be inferred from (5).

The mechanistic gain is explicit: even an almost unchanged driving error can
produce changing slope directions because readout–geometry correlations
change $s$, $K$ and the local cubic force. Loss improvement and preservation
of the coarse outputs therefore do not guarantee sustained outward transport.

When $m_2=m_3=0$, however, $G=A=0$ in this leading model. For targets
orthogonal through degree three, generated lower-mode error and higher Taylor
terms first appear at the next order. Their conditional parameter clock is
$\eta n/W^2$, as proved in
[the high-mode refinement](d34_polynomial_surrogate_theorem.md#4-a-slower-conditional-clock-when-the-target-has-no-quadratic-or-cubic-load).
A stationary leading PDE then means that its order omitted the driving force;
it does not mean that exact GD is stationary. No numerical accuracy claim or
universal acquisition time follows from this derivation alone.

## 5. The next clock: generated error competes with fourth- and fifth-degree target load

**Example.** A target orthogonal through degree three is invisible to the
leading force (1), but the network can still generate quadratic and cubic
error. Correcting that error competes with the target's fourth- and
fifth-degree components, accessed by the quintic activation term.
For a target orthogonal through degree five, such as a pure ninth-degree
empirical polynomial, even that opposing target term disappears at this
order. The following model makes this distinction without asserting that
every slope contracts.

**Theory.** Let $P_H$ project onto the full empirical complement of $1,x$,
put $u=\alpha x+\beta$, and assume $y_H$ is orthogonal to all polynomials
through degree three. Define two output functions and a potential:

$$
q_3=-\frac13P_H\mathbb E_\rho[\zeta u^3],\qquad
q_5=\frac{2}{15}P_H\mathbb E_\rho[\zeta u^5],\qquad
\mathcal E(\rho)=\frac12\|q_3\|_m^2-\langle y_H,q_5\rangle_m.
\tag{8}
$$

Indeed $P_Hf=W^{-1}q_3+W^{-2}q_5+O(W^{-3})$ on a bounded rescaled
parameter domain. The first nonconstant term in the fine loss is
$W^{-2}\mathcal E$: the $W^{-1}$ target pairing vanishes by orthogonality.
The raw particle force is the spatial gradient of its functional derivative,

$$
G_2(X;\rho)=\nabla_X\left[
-\frac13\langle q_3,\zeta u^3\rangle_m
-\frac{2}{15}\langle y_H,\zeta u^5\rangle_m\right].
\tag{9}
$$

Here $q_3$ is held fixed in the displayed particle derivative; its dependence
on the distribution has already been accounted for by differentiating
$\frac12\|q_3\|^2$. Repeating the coarse projection from (2), replace
$G$ by $G_2$ in $s$, set $\ell^{(2)}=K^{-1}s_2$, and define

$$
A_2=G_2-(\sqrt v\zeta\ell^{(2)}_1,\,
\zeta\ell^{(2)}_0,\,
\beta\ell^{(2)}_0+\sqrt v\alpha\ell^{(2)}_1).
$$

On the slower clock $\tau_2=\eta n/W^2$, the model is
$X'=-A_2$, $d'=\ell^{(2)}_0$, with transport equation
$\partial_{\tau_2}\rho+\nabla_X\cdot(-\rho A_2)=0$.
The same affine invariants and dissipation calculation give

$$
(d+\mathbb E\zeta\beta)'=0,\qquad
(\mathbb E\zeta\alpha)'=0,\qquad
\mathcal E'=-\mathbb E|A_2|^2-(\ell^{(2)}_0)^2\le0.
\tag{10}
$$

This is an exact statement about the next-order model. Under the compact
domain and uniform coarse-conditioning hypotheses of Section 3, Taylor
remainders and inverse-Gram stability also give the conditional estimates

$$
F^{\tanh}_{abc,j}=W^{-5/2}(A_2)_j+O(W^{-7/2}),\qquad
F^{\tanh}_d=-W^{-2}\ell^{(2)}_0+O(W^{-3}).
\tag{11}
$$

The combined full-parameter error in (11) is $O(W^{-3})$. These bounds
concern the exact effective fine field, not ordinary GD's tracking force.
On the $\tau_2$ clock the particle forcing remainder is $O(W^{-1})$;
ordinary GD additionally contributes $W^{5/2}(r_C)_{abc,j}$, and its
output-bias equation contributes $W^2(r_C)_d$. The characteristic comparison
(7) therefore applies with these replacements and step $\eta/W^2$.
Its bounded-domain, tracking, and first-exit premises still require proof.

**Geometry relative to readout: no zero-slope assumption.** Define
$\mathcal Q=\frac12\mathbb E(\alpha^2+\beta^2-\zeta^2)$.
The coarse-projection contribution cancels exactly from its derivative:
each affine output is bilinear in geometry and readout. The generated
cubic term has geometry degree six and readout degree two in its squared
norm; the target pairing has geometry degree five and readout degree one.
Euler's homogeneous-function identity therefore gives

$$
\mathcal Q'=-2\|q_3\|_m^2+4\langle y_H,q_5\rangle_m.
\tag{12a}
$$

For targets orthogonal through degree five, geometry energy grows no faster
than readout energy, irrespective of the conserved coarse outputs. This is
a relative-energy statement, not absolute geometry contraction: both could
grow, with readout energy growing faster.

**An illustrative absolute contraction statement.** Write
$B=\mathbb E\zeta\beta$, $C_1=\mathbb E\zeta\alpha$, and
$C_0=d+B$, where $C_0,C_1$ are conserved. Homogeneity of the degree-four
cubic output and degree-six fifth-order output gives

$$
\mathbb E[X\cdot G_2]=4\|q_3\|_m^2-6\langle y_H,q_5\rangle_m.
$$

Consequently the adjusted parameter energy
$\mathcal R=\frac12\mathbb E|X|^2+(d-C_0)^2
=\frac12\mathbb E|X|^2+B^2$ satisfies the exact identity

$$
\mathcal R'=-4\|q_3\|_m^2+6\langle y_H,q_5\rangle_m
+2\sqrt v\,C_1\ell^{(2)}_1.
\tag{12}
$$

Thus if the target is orthogonal through degree five and the conserved
affine slope is zero, $\mathcal R$ is nonincreasing. The constant coarse
output may be nonzero. More generally the sufficient condition is
$6\langle y_H,q_5\rangle_m+2\sqrt v C_1\ell^{(2)}_1
\le4\|q_3\|_m^2$. A changing output bias is why contraction of the
unadjusted energy $\frac12\mathbb E|X|^2$ does not follow automatically.

The affine-slope condition matters too. For a single-atom distribution with
$\beta=0$, $\alpha>0$, $\zeta>\alpha$, and a target orthogonal through
degree five, let $t=\|P_Hx^3\|_m^2>0$. Then $B=0$ and direct substitution
gives

$$
\mathcal R'=-\frac{2t\zeta^2\alpha^6(\alpha^2-\zeta^2)}
{9(\alpha^2+\zeta^2)}>0.
\tag{13}
$$

In this example the individual equations are

$$
\alpha'=-\frac{2t\zeta^2\alpha^7}{9(\alpha^2+\zeta^2)}<0,
\qquad
\zeta'=\frac{2t\zeta^3\alpha^6}{9(\alpha^2+\zeta^2)}>0.
\tag{14}
$$

They preserve $\alpha\zeta$: the slope contracts while the readout grows
to maintain the affine fit. The projected dynamics still decrease
$\mathcal E$, although the readout growth can increase total parameter
energy. This supplies a concrete contraction mechanism at nonzero coarse
slope, while showing why dissipation is not a universal radial barrier.
The zero-$C_1$ absolute-radius result above is illustrative and should not
be imposed on a fitted target with nonzero coarse slope.

**Prediction.** This model distinguishes fourth- or fifth-degree forcing from a
ninth-degree regime dominated, at this order, by correcting generated
lower modes. Under the contraction conditions, it also gives the
simultaneous population bound
$\rho_{\tau_2}(|\alpha|\ge r)\le2\mathcal R(0)/r^2$ at every time.
That bound does not count distinct particles that ever cross: different
particles could cross at different times. Nor does it establish a sign for
each slope. Transferring its confinement or population conclusions to
ordinary GD requires the conditional errors in (11) and the tracking
allowances to fit the desired margins. No archived numerical accuracy or
unconditional persistence claim is made here.

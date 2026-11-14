# A signed-energy escape-time bound for native tanh GD

Research derivation, November 13, 2026. The numerical example below evaluates explicitly stated hypothetical allowances, not quantities asserted to hold along a saved trajectory. The result is conditional and has not received independent mathematical review. A subsequent read-only evaluation on saved GD runs found that this closed comparison is unhelpful for PyTorch-default initialization; see the evaluation note below. No new training was performed for that evaluation.

## Result and scope

Retaining the sign of the network-output contribution removes the positive self-output term that appears when the entire gradient is bounded by its norm. The resulting native-GD energy inequality needs moments only through order eight, plus low-order target loading and the affine residual. Under declared small-energy and weak-target-loading conditions, it gives an explicit multi-million-update lower bound on leaving a small-energy region.

This is not a time-uniform confinement theorem. It does not yet explain the larger PyTorch initialization, and the persistence of its moment and affine-residual allowances is a remaining premise. It strengthens a particular regime of the existing explanation rather than proving that arbitrary training cannot acquire scale.

## 1. Model and definitions

Use the same raw network and half-MSE as the paper:

\[
q_\theta(x)=b_{\rm out}+\sum_{j=1}^W w_j\tanh(\gamma_jx+\beta_j),
\qquad L=\tfrac12\|q_\theta-f\|_Z^2,
\qquad |x_i|\le1.
\]

The empirical inner product is the sample average. All parameters undergo simultaneous Euclidean GD with the same nonnegative step size eta. Put

\[
p=(\gamma,\beta,w),\quad M=\|p\|^2,\quad
r_j^2=\gamma_j^2+\beta_j^2+w_j^2,\quad S_k=\sum_j r_j^k.
\]

The output bias is excluded from M but remains trainable. Let P_C project orthogonally onto span{1,x}; let P_H=I-P_C. Define q_H=P_H q, f_perp=P_H f, e=q-f, e_C=P_C e, and e_H=P_H e. Write E_C=||e_C||_Z and E_H=||e_H||_Z.

For k=3,5, let P_k denote the orthogonal projection onto P_H span{1,x,...,x^k}, and let tau_k=||P_k f_perp||_Z. These are target quantities, independent of training.

Use the paper's constants

\[
\alpha_3=\tfrac13,\quad\alpha_5=\tfrac2{15},\quad\alpha_7=\tfrac{17}{315},
\qquad A_k=\alpha_k\frac{2^{k/2}k^{k/2}}{(k+1)^{(k+1)/2}},
\qquad C_3=\tfrac{\sqrt6}{2}.
\]

No symmetry of the sample measure is needed for the generic constants used here.

## 2. Exact signed identity

For u_j=gamma_j x+beta_j, define the output's response to radial scaling of p:

\[
z=\sum_j w_j\bigl(\tanh u_j+u_j\operatorname{sech}^2u_j\bigr).
\]

The identity <p,gradient_p L>=<e,z>_Z follows directly by differentiating q. Consequently one native GD step satisfies exactly

\[
M_{n+1}-M_n=-2\eta_n\langle e_n,z_n\rangle_Z
+\eta_n^2\|\nabla_pL_n\|^2.
\tag{1}
\]

Define

\[
d_5=P_H\sum_jw_j\bigl[u_j\operatorname{sech}^2u_j-3\tanh u_j+2u_j\bigr].
\]

Because P_H annihilates the affine function sum_j w_j u_j and the output bias,

\[
P_H z=4q_H+d_5.
\]

Substitution in (1) gives

\[
\begin{aligned}
M_{n+1}-M_n
={}&\eta_n\bigl[-8\|q_H\|_Z^2-2\langle q_H,d_5\rangle_Z
+2\langle f_\perp,P_Hz\rangle_Z
-2\langle e_C,P_Cz\rangle_Z\bigr]
+\eta_n^2\|\nabla_pL_n\|^2.
\end{aligned}
\tag{2}
\]

The self-output terms have the exact completion of the square

\[
-8\|q_H\|_Z^2-2\langle q_H,d_5\rangle_Z
=-8\|q_H+d_5/8\|_Z^2+\tfrac18\|d_5\|_Z^2
\le\tfrac18\|d_5\|_Z^2.
\tag{3}
\]

This step preserves the stabilizing contribution before taking norms. It replaces a positive allowance proportional to the network's own nonlinear output by a squared higher-order defect.

## 3. Global remainder bounds

The following estimates are global in u, not restricted to a Taylor convergence neighborhood. Let

\[
g_5(u)=u\operatorname{sech}^2u-3\tanh u+2u.
\]

For u>=0, h(u)=tanh u-u sech²u obeys h(0)=0 and

\[
0\le h'(u)=2u\tanh u\operatorname{sech}^2u\le2u^2.
\]

Therefore 0<=h(u)<=2u³/3, and

\[
0\le g_5'(u)=2\tanh u\,h(u)\le\tfrac43u^4.
\]

Integration and odd symmetry give |g_5(u)|<=4|u|⁵/15. The same per-neuron product maximization used in the paper yields

\[
\|d_5\|_Z\le2A_5S_6.
\tag{4}
\]

Let

\[
q_3=-\tfrac13P_H\sum_jw_ju_j^3,\qquad
q_5=\tfrac2{15}P_H\sum_jw_ju_j^5.
\]

Then ||q_3||_Z<=A_3 S_4 and ||q_5||_Z<=A_5 S_6. Expanding z through order five gives

\[
P_Hz=4q_3+6q_5+r_7,\qquad \|r_7\|_Z\le8A_7S_8.
\tag{5}
\]

For (5), use the paper's seventh-order remainder h_7=tanh u-u+u³/3-2u⁵/15 and its derivative bounds. The radial remainder is w(h_7+u h_7'), whose absolute value is at most 8 alpha_7 |w||u|⁷. Since q_3 and q_5 lie in the corresponding low-degree polynomial spaces,

\[
2\langle f_\perp,P_Hz\rangle_Z
\le8A_3\tau_3S_4+12A_5\tau_5S_6+16A_7\|f_\perp\|_ZS_8.
\tag{6}
\]

Finally, |u sech²u|<=1/2 and |tanh u|<=1 imply

\[
-2\langle e_C,P_Cz\rangle_Z\le3E_C\|w\|_1.
\tag{7}
\]

The first scalar inequality follows from cosh²u>=1+u²>=2|u|.

## 4. Signed-energy comparison for native GD

Combining (2)–(7) gives

\[
\boxed{
\begin{aligned}
M_{n+1}-M_n\le\eta_n\bigl[
&\tfrac12 A_5^2S_{6,n}^2
+8A_3\tau_3S_{4,n}
+12A_5\tau_5S_{6,n}\\
&+16A_7\|f_\perp\|_ZS_{8,n}
+3E_{C,n}\|w_n\|_1\bigr]
+\eta_n^2\|\nabla_pL_n\|^2.
\end{aligned}}
\tag{8}
\]

An explicit bound on the discrete correction is

\[
\|\nabla_pL\|
\le C_3 E_H\sqrt{S_6}+E_C\sqrt{1+2M}.
\tag{9}
\]

The nonlinear Jacobian bound is the paper's cubic-remainder estimate, before applying the parameter-space projection. For the coarse term, the full output Jacobian, including output bias, has squared Hilbert–Schmidt norm at most 1+2M: |tanh u|<=|u|, |u|²<=2(gamma²+beta²), and |tanh' u|<=1. Orthogonal projection and restriction to p cannot increase that bound.

Unlike a gradient-norm bound, (8) does not turn the network's own nonlinear output into an outward driving force. It does retain positive higher-order remainders and target-loading terms. Thus it is an escape-time tool, not a proof of permanent confinement.

## 5. A closed first-exit statement

Suppose M_0<=m_0<m_star. Up to the first exit from M<m_star, assume

\[
\chi_6=\frac{W^2S_6}{M^3}\le K_6,
\qquad
\chi_{10}=\frac{W^4S_{10}}{M^5}\le K_{10},
\quad E_H\le H_0,\quad E_C\le\varepsilon_C.
\]

Cauchy–Schwarz gives

\[
S_4\le\sqrt{K_6}\frac{M^2}{W},\qquad
S_6\le K_6\frac{M^3}{W^2},\qquad
S_8\le\sqrt{K_6K_{10}}\frac{M^4}{W^3},\qquad
\|w\|_1\le\sqrt{WM}.
\]

Insert these expressions into the brackets in (8) to define a nonnegative increasing function U(M), and define

\[
G(M)=C_3H_0\sqrt{K_6}\frac{M^{3/2}}{W}
+\varepsilon_C\sqrt{1+2M}.
\]

For constant step size eta, let

\[
d_\star=\eta U(m_\star)+\eta^2G(m_\star)^2.
\]

The first exit time T_star satisfies

\[
\boxed{T_\star\ge\frac{m_\star-m_0}{d_\star}.}
\tag{10}
\]

To prove (10), every step before first exit begins with M<m_star and has increment at most d_star. Summing these increments, including the crossing step, gives m_star<=M_0+T_star d_star. The use of M<m_star in this first-exit argument does not assume the desired conclusion. The separate concentration and coarse-residual allowances still need justification over the candidate interval.

While the barrier holds, gamma_RMS<=sqrt(m_star/W). The paper's output inequality also remains available: ||q_H||_Z<=A_3 S_4, so the same barrier can support a target-dependent error floor.

## 6. Explicit illustrative constants

Consider the declared hypotheses

\[
W=705,\quad\eta=0.002,\quad m_0=6,\quad m_\star=12,
\]
\[
K_6=3,\quad K_{10}=12,\quad H_0=1.01,\quad
\varepsilon_C=10^{-7},\quad
\tau_3,\tau_5\le10^{-4},\quad \|f_\perp\|_Z\le1.
\]

Use the conservative constants A_3<=0.307, A_5<=0.196, A_7<=0.136, C_3<=1.225. At M=12,

\[
S_4<0.3541,\quad S_6<0.01044,\quad S_8<0.0003551,
\quad\sqrt{WM}<92.
\]

These give U(12)<0.000892 and G(12)<0.1265. Hence

\[
d_\star<0.002(0.000892)+0.002^2(0.1265)^2
<1.85\times10^{-6}.
\]

Consequently the energy cannot rise from at most 6 to 12 in fewer than **3.2 million additional GD updates**, and throughout that interval

\[
\gamma_{\rm RMS}<\sqrt{12/705}<0.131.
\]

This is illustrative theorem arithmetic, not validation of these allowances on any experiment for 3.2 million updates. In particular, a target with substantial quintic loading, such as a normalized degree-five polynomial, does not satisfy the displayed tau_5 hypothesis.

The residual allowance in this numerical example can be derived from a starting full residual RMS at most 1.01, instead of assumed separately. Before first exit, the current full Jacobian norm is at most sqrt(1+2M)<=5, so the proposed step has length at most 0.0101. Its segment stays within hidden energy 13. There the Jacobian norm is at most sqrt(27), the residual is below 1.063, and the loss Hessian norm is at most 27+1.063(4 sqrt(13)+sqrt(2))<44. Since eta*44<1, the usual descent estimate closes the induction. This argument controls full residual up to first exit; it does not establish the stronger affine-residual or concentration allowances.

## 7. What this improves, and what it does not

- The positive self-output contribution in the norm-based comparison is replaced by a squared higher-order defect. With bounded energy and concentration, that defect scales as W^-4; weak low-degree target loading exposes the target remainder at W^-3.
- Native GD's squared-step correction is retained. No gradient-flow approximation is used in (8)–(10).
- The result is a quantitative first-escape bound. It does not require growth to be logarithmic or forbid eventual escape.
- It still needs persistent concentration and affine-residual control. Small tracking R is not automatically the same as small affine residual e_C; they must not be substituted for one another.
- The displayed long-time example requires small total parameter energy. It does not apply to the larger-energy PyTorch runs and should not be advertised as their explanation.
- For broad applicability, the unresolved theory is how to replace or justify these allowances—especially in the larger-initialization regime—without assuming that scale acquisition is already limited.

The signed identity and global remainder algebra were checked symbolically. The derivation above, including the first-exit argument, needs independent review before use as a paper theorem.


## 8. Subsequent evaluation on saved GD runs

The signed identities and the one-step moment inequality were checked on 1,066 saved states. With matched constant checkpoint-derived allowances over 30k–50k, the proposed energy ceiling reaches twice its starting energy sooner than the original ceiling in all 24 default-initialization D62 cases. The width-705 default P5 case also worsens (9 versus 101 updates). The squared absolute-moment remainder loses large cancellations visible in the exact aggregate remainder. This does not refute the small-energy conditional example, but it does rule out presenting the current closed comparison as a successful fix for the default runs.

Details: [saved-run evaluation](../../../results/checkpoint_D_optimizers/expD62_joint_training_diagnostics/signed_energy_audit/results.md). The earlier illustrative allowances have not been established over millions of steps.

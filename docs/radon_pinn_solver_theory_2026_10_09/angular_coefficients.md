# Nonlinear PDE linearizations have bounded angular coefficient rank

The differential structure supplies an exact result for general local nonlinear PDEs: their linearized coefficients depend on direction through a finite-dimensional polynomial space whose size depends on input dimension and differential order, not on the number of Radon directions. This does not assume that the unknown solution or the complete Jacobian is low rank. It explains why a fast factorization of the fixed derivative banks can be reused throughout nonlinear and inverse iterations, without expanding pairs of directions.

These are direct algebraic derivations for a standard one-hidden-layer network. No new numerical experiment or Lean theorem is claimed.

## 1. Fixed network and a general local equation

Let the input z lie in R^D (including time or declared physical parameter inputs when appropriate). A vector-output network with H fields is

\[
q_h(z)=b_h+\sum_{m=1}^M\sum_{j=1}^N
w_{mjh}\psi\bigl(\gamma(\theta_m\cdot z-c_j)\bigr),
\qquad h=1,\ldots,H.
\]

Geometry is fixed. Assume ψ has the derivatives required by the declared equation. For a multi-index α of order r,

\[
\partial^\alpha q_h(z)=\gamma^r\sum_{m,j}
w_{mjh}\theta_m^\alpha\psi^{(r)}(s_{mj}(z)),
\quad s_{mj}(z)=\gamma(\theta_m\cdot z-c_j).
\]

At one point let the residual component e be an arbitrary differentiable function

\[
R_e(z)=F_e\bigl(z,\{\partial^\alpha q_h(z)\}_{h,\alpha},p\bigr),
\]

with p the finite-dimensional physical unknowns. No polynomial form for F is assumed. Unknown coefficient fields enter among the q_h with their own derivative channels. Boundary and data equations use the same formalism whenever they are local pointwise operators; nonlocal measurements retain their separately declared linear/nonlinear action.

## 2. Angular coefficient rank theorem

At a current candidate define the scalar sensitivity

\[
C_{eh\alpha}(z)=\frac{\partial F_e}{\partial u_{h\alpha}},
\qquad u_{h\alpha}=\partial^\alpha q_h(z).
\]

The derivative with respect to a neuron readout is

\[
\frac{\partial R_e(z)}{\partial w_{mjh}}
=\sum_{r=0}^{r_*}\gamma^r
\psi^{(r)}(s_{mj}(z))P_{ehr}(z,\theta_m),
\]

where

\[
\boxed{P_{ehr}(z,\theta)=
\sum_{|\alpha|=r}C_{eh\alpha}(z)\theta^\alpha.}
\tag{1}
\]

Thus P_ehr is a homogeneous degree-r polynomial in θ, regardless of the nonlinearity, frequency content or spatial complexity of the current solution.

For K arbitrary target points, the K-by-M coefficient matrix P_ehr factors as

\[
[P_{ehr}(z_k,\theta_m)]_{km}
= [C_{eh\alpha}(z_k)]_{k\alpha}
  [\theta_m^\alpha]_{\alpha m}.
\]

Consequently its rank is at most

\[
\min\!\left(K,M,\binom{D+r-1}{r}\right).
\tag{2}
\]

If only certain derivatives are declared, their count replaces the binomial bound. Known contractions such as a Laplacian can lower it further. On the sphere, polynomial restrictions may satisfy additional relations. Those are optional reductions; (2) already holds without them.

**Proof.** Differentiate the local residual with respect to its finite derivative list, then apply the network chain rule. Group multi-indices by derivative order. Counting degree-r monomials gives (2). Nothing about this proof restricts F to weak nonlinearity or q to a simple target. The required differentiability is at the candidate being linearized; a nonsmooth constitutive law needs its appropriate generalized derivative rather than an unannounced smooth surrogate.

The full Jacobian also contains ψ^(r)(s_mj(z)), which depends jointly on point, direction and center. Therefore (2) is a low-rank statement only about the equation's coefficient modulation; it is not a low-rank approximation to the whole network or Jacobian.

## 3. What changes in a nonlinear iteration

The geometric derivative operators A_α, the direction monomials and their approximation factorizations are fixed. The coefficient functions C_ehα(z_k) change with the current fields and physical parameters. Accordingly,

\[
J_q=\sum_\alpha C_\alpha A_\alpha
\]

in block notation, with the C_α acting in small physical-point/field blocks. A fast geometric forward/transpose plan can be reused; nonlinearity does not require building new direction-pair interaction tables. A preconditioner for the current Jacobian can still change substantially and its update cost is a separate question.

For orders through r_*, all monomials together number binom(D+r_*,r_*). That dimension dependence is explicit. For fixed differential order it is polynomial in D, though the sampling, representation and aggregation costs are not thereby made dimension-free.

## 4. Navier–Stokes specialization

Let z=(x,t) with d spatial coordinates, and split each direction as θ_m=(a_m,τ_m). Unit spacetime norm gives |a_m|²+τ_m²=1. Let U_m^(r) denote the vector velocity readout sum using ψ^(r), without the factor γ^r, and P_m^(1) the corresponding pressure sum.

For constant kinematic viscosity ν, the usual advective-form momentum residual is

\[
\boxed{
R_u=\gamma\sum_m(\tau_m+u\cdot a_m)U_m^{(1)}
-\nu\gamma^2\sum_m|a_m|^2U_m^{(2)}
+\gamma\sum_m a_m P_m^{(1)}-f.
}
\tag{3}
\]

The continuity residual is γΣ_m a_m·U_m^(1). Thus the residual uses value, first-derivative and second-derivative radial banks. It need not construct every entry of the velocity gradient or Hessian.

The advection/time coefficient matrix has the exact factorization

\[
C_{km}=\tau_m+u(z_k)\cdot a_m
=[\mathbf1\ \ U]\begin{bmatrix}\tau^T\\A^T\end{bmatrix},
\qquad\operatorname{rank}C\le d+1.
\tag{4}
\]

This is true for an arbitrarily complex resolved velocity field. The velocity matrix U is not assumed low rank beyond having d physical components. It is the angular dependence of the differential operator that is simple.

The Newton correction includes both (u·∇)δu and (δu·∇)u. The latter is another directional contraction, γΣ_m(δu·a_m)U_m^(1), and preserves the same geometry. Dropping it would change the linearization; it has not been dropped. Pressure, divergence and unknown viscosity columns must remain present. Variable viscosity in the symmetric-stress equation adds terms beyond νΔu, detailed in [the compatibility derivation](navier_stokes_compatibility.md).

The advection form and divergence of the symmetric stress agree with standard incompressible physical equations at divergence-free solutions under the usual constant-viscosity assumptions, but off-shell residuals differ by gradient/divergence terms. The declared residual form must be retained during inference; incompressibility cannot be assumed before it is enforced.

## 5. Architectural and complexity consequences

All U_m and P_m are evaluations of the same fixed hidden activations with different output readouts. Products such as u·a_m occur in PDE residual evaluation, not in the network architecture. The output remains an ordinary affine–activation–affine MLP. Invertible coordinate changes and fixed linear encoders compose with the final readout matrix and introduce no hidden layer or product gate.

The theorem is activation-independent wherever ψ has the required derivatives. Tanh-specific localized kernels, Fourier multipliers and numerical bounds belong to the implementation of A_α; the local nonlinear coefficient factorization survives a future supported activation change.

For K scattered points the direct localized Navier–Stokes actions still have a term O(K M d s), with stencil size s and appropriate constant-pass counts. Equation (4) does not eliminate it because each direction has its own radial profile value. The unresolved global aggregation theorem remains necessary when K grows with M N.

The concrete progress is that general nonlinearity does not force a new M² construction after a fast derivative-bank factorization has been found. The difficult fixed geometry can be separated from changing nonlinear coefficients without choosing an easier PDE family. This does not prove a fast preconditioner, global nonlinear convergence, or machine-epsilon accuracy at every physical parameter regime.

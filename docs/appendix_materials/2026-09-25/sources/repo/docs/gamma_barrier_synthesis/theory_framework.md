# A theory connecting approximation, readout optimization, and geometry motion

Internal mathematical framework, 14 September 2026. This is a proposed organization of established identities and remaining proof obligations, not a claim that the full joint-GD barrier has been proved. No new training was run. Independent reviewers checked the fiber calculation, the schedule bound, and the interpretation of the geometry gradients.

## 1. State the intended conclusion at the right level

The first tractable claim is about a fixed uniform-grid tanh representation, a specified accuracy, and Euclidean readout GD. Changing lambda can improve readout learning rates while raising the irreducible aliasing error. That produces a finite-time accuracy tradeoff.

The larger claim concerns whether joint GD can discover an alternative geometry that avoids this tradeoff. That requires a separate trajectory argument. The fixed-grid theorem cannot establish it by itself.

Both claims must specify the target's spectral content. A very small singular value is irrelevant to fitting a target that has negligible projection on its left singular vector. Conditioning is a worst-direction property; a particular target's convergence is not determined by condition number alone.

## 2. One exact expression contains aliasing and unfinished readout fitting

Use a finite periodic uniform grid with a continuous population squared loss, fixed lambda, and zero initial readout. Treat the constant output separately. At a nonzero base frequency theta below the grid Nyquist frequency, assume the target has amplitude B and no target content at its aliases theta+2pi m, m nonzero. This is the single-tone case of the C07 fiber theory. Real tones use conjugate pairs with the same contraction law.

The readout's Fourier coefficient a controls all those output frequencies together. Let w_m denote its actual transfer to frequency theta+2pi m. Choose orthonormal coefficient Fourier coordinates so ordinary Euclidean GD is preserved; include output-loss normalization in w_m. Then the contribution to half squared error is

\[
\ell(a)=\frac12\left(|a w_0-B|^2+\sum_{m\ne0}|a w_m|^2\right).
\]

Define the alias-to-main energy ratio and the coefficient curvature by

\[
S=\frac{\sum_{m\ne0}|w_m|^2}{|w_0|^2},
\qquad q=\sum_m|w_m|^2=|w_0|^2(1+S).
\]

Completing the square gives

\[
a_* = \frac{\overline{w_0}B}{q},\qquad
\ell(a)=\frac{|B|^2}{2}\frac{S}{1+S}
+\frac q2|a-a_*|^2.
\]

The first term is the best possible approximation error. The second is the loss from unfinished readout fitting. Under readout GD with real step sizes eta_k,

\[
a_{k+1}-a_*=(1-\eta_kq)(a_k-a_*).
\]

Starting at a_0=0 therefore gives the exact relative squared-error identity

\[
\boxed{
E_K^2(\theta,\lambda)
=\underbrace{\frac{S}{1+S}}_{\text{aliasing floor}}
+\underbrace{\frac1{1+S}\prod_{k<K}(1-\eta_kq)^2}_{\text{unfinished readout fitting}}.
}
\]

For targets with multiple base frequencies and no target aliases, sum these expressions with weights equal to the target's relative squared Fourier amplitudes. The identity is an elementary combination of the existing fiber projection formula and linear GD, rather than a novelty claim.

This connects two observations without equating them. The ghost ratio S determines the limiting accuracy; q determines the rate at which that frequency approaches its limit. Both come from the same generator spectrum. Conditioning compares q across coefficient modes; a single q is a curvature, not a condition number.

For tanh, the generator spectrum away from zero is proportional to

\[
w_\lambda(\xi)=\frac{\widehat{\operatorname{sech}^2}(\xi/\lambda)}{i\xi}.
\]

Smaller lambda suppresses distant ghosts, but also suppresses the absolute response at larger base frequencies. Increasing lambda can improve those learning rates while increasing their ghost ratios. A complete quantitative tradeoff must include the strongest curvature that limits the shared learning rate and the frequencies the target actually needs.

There is an exact monotonicity statement for the raw tanh transfer:

\[
|w_\lambda(\xi)|=\frac{\pi}{\lambda\,\sinh(\pi|\xi|/(2\lambda))}.
\]

For a fixed nonzero base frequency with |theta|<pi, every absolute transfer grows with lambda, and every ghost/main ratio grows with lambda. The latter ratio is sinh(pi|theta|/(2lambda))/sinh(pi|theta+2pi m|/(2lambda)); its monotonicity follows from the fact that t coth(t) increases for positive t. Hence both q and S increase. Larger q improves contraction at a fixed rate while eta q<=1; once overshoot occurs, that monotonic improvement is no longer automatic. This establishes a tradeoff under this lambda change without claiming that every form of better conditioning raises aliasing. An invertible coefficient preconditioner changes the learning metric while preserving the approximation space and its floor.

Normalization matters: using the normalized kernel transform with value one at zero instead of the actual sech-squared transform with value two changes q by a factor of four. With a period of P grid units, a mean population loss and a unitary coefficient DFT give transfer amplitudes w_lambda/sqrt(P), hence curvature sum|w_lambda|^2/P. These factors cancel from S but not from GD times.

This is a population/continuous torus calculation. Sampling only at exact centers collapses distinct aliases onto the same sampled mode, so it cannot be used as the sampled-loss formula for D30. Finite intervals, halo columns, and nonuniform centers require their actual feature matrix and a separate approximation comparison. An infinite-lattice tanh model also has an unbounded transfer near zero frequency; a finite periodic model with its constant mode treated separately avoids silently assuming a bounded global curvature there.

## 3. A fixed-geometry finite-time theorem has clear premises

For any fixed normalized readout matrix A, let M=sigma_max(A)^2 be its largest curvature. Consider scalar schedules satisfying 0<=eta_k<=2/M at every step. This ensures every represented mode is nonexpansive at every step; it is stronger than observing decreasing loss on one particular trajectory.

For a smaller curvature m=sigma_j(A)^2 with rho=m/M<=1/4, the corresponding residual component obeys

\[
|r_j(K)|=|r_j(0)|\prod_{k<K}(1-\eta_km)
\ge |r_j(0)|(1-2\rho)^K
\ge |r_j(0)|e^{-4\rho K}.
\]

Thus a loaded slow mode gives a quantitative training-time lower bound. For relative target amplitude b_j=|r_j(0)|/||y|| greater than the requested tolerance epsilon,

\[
K\ge\frac{1}{4\rho}\log\frac{b_j}{\varepsilon}.
\]

A precise theorem combining aliasing and optimization would establish two premises over the chosen fixed-grid family:

1. Outside an accuracy-admissible lambda range, the aliasing floor exceeds epsilon.
2. Everywhere inside that range, the target has a represented component of relative amplitude at least b_0>epsilon with curvature ratio at most rho_0<=1/4.

Then every lambda either fails the approximation requirement even after a solve, or requires at least log(b_0/epsilon)/(4rho_0) GD steps in the stated schedule class. This is the clean accuracy-versus-training-time result to aim for. C07/C08 support the first premise in their settings. D26/D30 establish actual spectral convergence predictions in the measured fixed matrices. A uniform target-weighted bound over the intended admissible range still needs to be established; a bad global condition number cannot replace it.

## 4. Geometry dynamics are an additional question

Let theta denote all geometry parameters and define the exact finite-sample profile and readout gap,

\[
F(\theta)=\min_v L(\theta,v),\qquad G=L-F.
\]

In a smooth region, set h=grad_theta F and k=grad_theta G. Joint gradient flow with geometry rate eta_theta gives

\[
\dot F=-\eta_\theta\bigl(||h||^2+h^Tk\bigr).
\]

This is why norm dominance alone is insufficient. A larger k helps F if aligned, leaves its intrinsic instantaneous descent unchanged if orthogonal, and opposes it if the dot product is sufficiently negative. D28 contains examples of both positive and negative alignment. D29 establishes that changing the relative influence can improve refitted geometry, not that the two contributions universally cancel.

F does not intrinsically seek higher gamma. Its descent seeks better approximation; that can mean increasing some scales, reducing others, or moving centers. On an aliasing wall it can favor smaller scales. The derivative of the exact F uses solved coefficients, whereas the note's current-readout outside-span term uses current coefficients. They are different vector fields.

The actual measured F_tau uses a singular-value cutoff. It is a numerical reference, with extra derivative terms from the retained subspace and potential rank crossings. For a theory about practical high precision, the admissible coefficient norm or numerical stability criterion must be explicit. Unrestricted exact F does not penalize enormous cancelling coefficients.

## 5. Drift bounds constrain the available time but do not make schedules irrelevant

For raw Euclidean GD satisfying sufficient descent

\[
L_k-L_{k+1}\ge\alpha\eta_k||\nabla L_k||^2,
\]

the displacement of the slope vector satisfies

\[
||a_K-a_0||^2\le\frac{L_0}{\alpha}\sum_{k<K}\eta_k.
\]

Ordinary learning-rate decay reduces the accumulated rate at a fixed iteration budget. Increasing an admissible rate or allowing more iterations increases the budget. Neither automatically supplies a gradient pointing toward useful geometry. Tied scales, block rates, and reparameterizations change the metric and hence the bound; our restricted successful interventions are compatible with this fact.

The scalar schedule restriction in Section 3 is also important. Without it, known eigenvalues allow exact-arithmetic GD polynomial schedules that annihilate modes in finitely many steps. Ordered from largest curvature downward, these can even reduce the actual loss at every step while later steps would amplify perturbations in previously eliminated modes. Roundoff can revive those modes. This prevents identifying observed monotone loss with uniform per-mode stability, and prevents claiming impossibility for every schedule. No practical robust schedule of that kind has been established by the current synthesis.

The missing joint-GD theorem would show that, from the specified initialization, the trajectory remains within a region whose best admissible approximation exceeds epsilon for the stated budget. This needs both an attainable-error obstruction throughout that region and a movement or sensitivity bound that keeps the trajectory there. Distance to the QI construction alone is insufficient because a different accurate representation may be closer.

## 6. Where the note stands

| Part of the note | What is supported | What remains |
|---|---|---|
| Section 2: Fourier scale response | Transform identity and controlled frequency-dependent pairing | Persistent assumptions that turn it into a dominant joint-training escape-time barrier |
| Section 3: readout projection | Exact decomposition; geometry/readout interactions; unsuccessful tested freezes | Causal proof that readout adaptation removes useful geometry force before enough improvement occurs |
| Section 4: squared sensitivity | Exact fixed-readout recurrence verified against runs; local geometry result | Residual-loaded projected geometry sensitivity bounds along relevant joint trajectories |
| Section 5: limited drift | Valid inequalities; restrictive saved-trajectory certificate | Prospective constants, optimizer assumptions, and necessity of leaving the bounded region for accuracy |
| Section 6: separate tangent weakness and compensation | Clear diagnostic framework | Quantitative attribution of the observed geometry weakness to those separate factors |

There is enough structure for a principled fixed-geometry accuracy-versus-GD-time theory. There is not yet enough for a general theorem that standard joint GD cannot discover accurate geometry. Fourier filtering can explain part of the relevant sensitivities; it need not carry the entire argument.

Sources: [C07 hardened conclusions](../../results/checkpoint_C_geometry/expC07_lambda_energy_rule/lambda_rule/hardened_rule.md), [C08 anchor results](../../results/checkpoint_C_geometry/expC08_anchor_rule/expC08_results.md), and the experiment links and independent reviews in the [full synthesis](synthesis.md).

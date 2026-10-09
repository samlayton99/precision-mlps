# Final proof polish — 2026-10-09

This pass keeps the approved theorem assumptions and construction scope. One independent agent audited the principal source-to-network chain, then reviewed and improved the angular argument. No experiments, coefficient-acquisition work, stronger tanh boundary correction, commits, or publication were added.

## A common argument for angular error

Let I be normalized spherical integration, Q the actual positive Gauss–Jacobi rule, F the angular target, and P its canonical harmonic truncation through degree 2q−1. The new shared lemmas prove

I(P)=I(F), and Q(P)=I(P).

The first identity follows because every positive-degree harmonic has zero mean; the constant component contains the entire mean. The second is the already verified polynomial exactness of the actual rule. Consequently,

I(F)−Q(F)=Q(P−F),

so |I(F)−Q(F)| ≤ ||P−F||∞. The previous generic comparison bound paid twice for that tail. `Gaussian.canonical_partial_integral` and `Gaussian.canonical_foldedSphere_error_of_partial` now supply this step once to both the analytic and Sobolev endpoints.

The analytic angular certificate is now

E_angular ≤ A sqrt((1+exp(−a))/(1−exp(−a))^(d−1)) exp(−a q),

with its former factor 2 removed. A bounds the canonical exponentially weighted harmonic norm, a is its exponential regularity parameter, and q is the quadrature order. The canonical Sobolev angular certificate also loses its former factor 2. Their hypotheses and exponents are unchanged. The actual construction's angular constants and order selectors use the smaller constants, including the general, GELU and tanh endpoints. Existing representation-witness series bounds remain valid as originally stated.

## One assembly step

`weighted_approximation_error` combines a proved angular error with the positive weighted sum of already proved spoke errors. `normalized_approximation_error` gives angular error plus the common spoke error when weights sum to one. Five construction proofs now use this shared step, as do the general positive-quadrature and anchored-transport arguments.

Exact origin anchoring still costs at most twice the uncentered scalar error. Thus the centered common-error certificate remains

E_total ≤ E_angular + 2 E_spoke.

The scalar factor 2, scalar epsilon/4 budget, and complete halo counts were preserved. Removing the angular factor does not remove the anchoring factor.

## Tighter integer-order bound

For the actual order ord(C,b,eta)=ceil(max(1,max(1,log(C/eta))/b)), with b>0,

ord(C,b,eta) ≤ (1+1/b) max(1,log(C/eta)).

The previous bound used 2+1/b. The same order definition is retained; the improved estimate propagates to the explicit width constants. It also covers C=0 under the existing convention. This does not change the asymptotic exponent.

## Audit scope and clarification

The independent review found no semantic correctness defect or hidden target-approximation premise in the decisive source-to-network chain. It checked the original target assumptions, source extension, scalar/angular certificates, actual network architecture, anchoring obligations, and halo-inclusive counts. This was a targeted semantic review, not a second line-by-line proof of the entire library.

The local algebraic endpoint is explicitly a uniform polynomial-approximation profile class with angular Sobolev regularity. It does not independently derive that class from arbitrary raw C^s target derivatives. Allocation optimality is for the proved continuous certificate, not all integer configurations or all networks. Neither qualification was weakened or concealed by this pass.

## Verification

The final full `lake build` passed (8850 jobs). `lake env lean Audit.lean` passed for 69 principal and supporting declarations, using only `propext`, `Classical.choice`, and `Quot.sound`. A proof-library scan found no placeholders, custom axiom declarations, unsafe definitions, or native-decide shortcuts. Existing unrelated style warnings remain. Audit output is in `final_axiom_audit.txt`; the full build log is in ignored `scratch/polish-final-build.log`.

The work completed within the 07:45:19–08:15:19 UTC time window. The independent reviewer also checked propagation of the smaller constants and the shared assembly/rounding changes. No assumptions were strengthened, no convergence exponent improvement is claimed, and no compilation problem was addressed by weakening a theorem.

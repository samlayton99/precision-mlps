# Local construction width and exact GELU architecture

Completed 2026-10-08; rounding constant sharpened 2026-10-09. Owned files: `QIRadon/LocalScaling.lean` and `QIRadon/LocalGELU.lean`. Existing modules were not changed.

## What is now proved

`LocalScaling.localAccuracyOrder_le` bounds the exact positive integer order

\[
\operatorname{ord}(C,b,\eta)=\left\lceil\max\left(1,\frac{\max(1,\log(C/\eta))}{b}\right)\right\rceil
\]

by \((1+1/b)\max(1,\log(C/\eta))\), for \(b>0\). The nonnegative error constant may be zero; no artificial positive target norm is required. `localAccuracyOrder_log_bound` specializes the budget to \(\eta=\epsilon/2\), proving

\[
\operatorname{ord}(C,b,\epsilon/2)
\le D(C,b)|\log\epsilon|,
\quad
D(C,b)=(1+1/b)\{1+|\log\max(1,2C)|\},
\]

for \(0<\epsilon\le e^{-1}\).

For exactly the angular order \(q=\operatorname{ord}(U,a,\epsilon/2)\) and preparation order \(N=\operatorname{ord}(V,b,\epsilon/2)\) used by `LocalMultidimensional.local_analytic_accuracy_and_width`, `local_total_log_width` bounds the cardinality of the actual finite direction × center index type:

\[
W_{\rm grid}\le D(U,a)^n
\left[(2H/\pi+2HZ+1)(\operatorname{preparationGrowth}(T_0)+3)^\kappa\right]
D(V,b)^\kappa |\log\epsilon|^{n+\kappa},
\]

where \(n=d-1\), \(\kappa=\max(1,1/\mu+1/\nu)\). The finite grid definition includes every halo center, and its cardinality is used directly. `local_total_grid_epsilon_bigO` proves the corresponding formal Big-O statement. `local_grid_with_anchors_epsilon_bigO` includes any fixed, explicitly charged number of additional anchor neurons; it does not assume that an arbitrary activation can realize its anchors.

`LocalGELU.raw_spokes_network` expands the actual finite GELU ridge sums into ordinary hidden weights, hidden biases, readout weights, and one output bias. `anchored_spokes_is_network` combines all order-two directional anchors into one affine function, then represents that function by **two GELU neurons total** using GELU(z) − GELU(−z) = z. No fitting is used in these identities.

`local_gelu_is_network` applies that identity to the actual local Chebyshev preparation and Fourier-defined densities. The same network parameters reproduce `localFiniteNetwork` at every input. `local_gelu_accuracy_and_width` then combines this exact architecture with the existing actual finite accuracy theorem, using the proved standard-GELU kernel and envelope. It proves error at most epsilon and the actual grid-count bound plus two.

Finally, `local_gelu_total_epsilon_bigO` proves

\[
W_{\rm GELU}=O(|\log\epsilon|^d)
\]

for the actual rounded orders, halo grid, and two global affine-anchor neurons.

## Scope and assumptions

These results preserve the local analytic profile and canonical angular norm assumptions of the preceding construction theorem. They do not manufacture a source theorem from a bare target function: the separate source-to-canonical-angular-norm bridge supplies those hypotheses. That bridge is now proved in `KnownFunctionConstruction.lean`. The constants can depend on fixed dimension; no dimension-independent complexity constant is claimed. The width is the hidden-neuron count, not arithmetic cost, numerical precision, or sample cost.

For a general activation whose r-th derivative is the admissible kernel, r ≥ 2 still needs an explicit implementation of its degree-(r−1) anchor and the resulting cost. Standard GELU at r=2 is now closed with exactly two units. Gaussian at r=0 needs none, and an r=1 activation needs only an output bias. No theorem here assumes that every activation can realize arbitrary polynomial anchors exactly.

## Verification

Both source modules checked with the pinned Lean 4.33.0 toolchain. `lake build QIRadon.LocalScaling QIRadon.LocalGELU` builds their artifacts. The root aggregator was deliberately not edited. Principal theorem axioms are checked separately in `scratch/AuditLocalScaling.lean`; all ten checked declarations depend only on `propext`, `Classical.choice`, and `Quot.sound`. No `sorry`, `admit`, or project axiom was introduced.

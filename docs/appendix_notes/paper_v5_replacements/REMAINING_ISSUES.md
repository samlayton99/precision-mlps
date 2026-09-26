# Remaining issues in paper version 5

These findings refer to the supplied version-5 PDF. They are not automatic edits to the main paper. The theorem numbering below is the main paper's numbering.

## 1. Equation (4) and the algorithm covered by Theorem 3.1

**Location:** Section 3.1, page 4; Equation (4).

Equation (4) defines the returned readout as an unregularized least-squares minimizer. The appendix's coefficient and recovery guarantee instead uses envelope-based column scaling, a truncated-SVD cutoff, and the stated numerical run conditions. These are different specifications: a truncated solution need not be an exact unregularized minimizer, and arbitrary minimizers need not have the claimed coefficient bound.

The smallest fix is to describe Equation (4) as the least-squares objective and explicitly identify scaled, truncated-SVD recovery as the algorithm covered by Theorem 3.1, with a reference to `sec:recovery`. The true-p-bit implementation is separately documented in `app:pbit-solver`; it should not be presented as automatically satisfying the certified scaled-solver assumptions.

## 2. The main width corollary is stronger than the appendix corollary

**Location:** Corollary 3.1.1, page 5.

The appendix proves a fixed-design statement: if the numerical/aliasing allowance is uniformly bounded by tau over a certified range, widths satisfying W >= log(A/(epsilon-tau))/c attain epsilon in that range. Its constants A and c can depend on lambda. The main corollary permits choosing lambda as epsilon changes while stating W = O(log(1/epsilon)). The fixed-design proof alone does not establish uniform constants for that changing design.

Either retain the fixed-design, above-allowance/range qualification in the main corollary, or supply the additional uniform-in-design argument. The replacements preserve the existing qualified result; they do not silently strengthen it.

## 3. State the precise role of the bandwidth rule

**Location:** First sentence of Section 3.3, page 6.

The displayed compact score controls the tanh pole/aliasing contribution. The appendix also distinguishes the single representative-frequency heuristic from the full frequency-weighted bound. It does not derive an optimizer that jointly minimizes the readout-amplification and boundary-correction terms.

A minimal replacement for the opening sentence is:

> We predict the relative bandwidth lambda from the sampled target by setting a frequency-dependent aliasing score equal to an effective tolerance epsilon_eff.

Keep the following equation. The extra activation rules are now specified in `app:bw-general-rule`, with the scope of their empirical use stated explicitly.

## 4. Theorem 3.2: normalization, scope, and numerical examples

**Location:** Section 3.4 and Figure 3, pages 6–7.

The new appendix fully specifies the normalized feature matrix, objective, step size, continuous-center matrix, Fourier multiplier, lattice and tail corrections, interlacing, target weights, and error-bound direction. It supplies the missing proof.

For readability, introduce A_gamma in the main theorem or the preceding paragraph: its bias column is M^{-1/2}, its tanh columns have entries M^{-1/2} tanh(gamma(z_i-x_j)), and the normalized targets are f(z_i)/sqrt(M). The stated eta = 1/(2 mu_1) uses the objective ||A_gamma c-y||^2/2.

The result is a pointwise finite-network rate/error enclosure. It proves monotonicity of the continuous-center matrix with gamma and makes the finite corrections explicit. It does not assert universal monotonicity of every normalized finite-kernel eigenvalue. A particular increase is certified when the corresponding intervals separate; the text should keep claims about observed ratio increases tied to the evaluated geometries.

The width-153 example now has its correct configuration and target in `app:note-step-example`. The displayed 9.20e11 count is a refinement-checked floating-point evaluation of an analytic lower bound, not an executed trillion-step run or a directed-rounding certificate. This qualification is now explicit.

Figure 3 is a different, width-512 experiment with a different mixed-sine target. Its exact geometry, sample grid, slope choices, cutoff/target-weight handling, and gradient-descent protocol still need a matching appendix paragraph. Do not reuse the width-153 protocol for it.

## 5. Remaining production fixes

- Figure 4 still has the placeholder caption “Needs to be updated to a 3-panel figure.”
- The PDF still contains unresolved appendix, theorem, figure, and citation references. `REPLACE_INSTRUCTIONS.md` gives the labels fixed by this bundle; it cannot supply missing bibliography entries or the unprovided experiment sections.
- Theorem 3.3 and its trajectory/protocol appendix remain outside this pass, as requested.

No further experiment is needed to implement the ordering and compression changes. The priority is matching the main statements and references to the proofs and protocols already available.

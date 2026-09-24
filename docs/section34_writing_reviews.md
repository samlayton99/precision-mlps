# Section 3.4 writing reviews

The review criterion is whether each sentence advances the argument and gives the reader enough information to assess it. There is no word-count target. The surrounding Section 3 motivates accurate arithmetic primitives, constructs their geometry, and explains the relative-bandwidth regime. This subsection must explain the additional cost of learning those weights and lead naturally into constructing arithmetic circuits.

## Review 1: argument and scope

Reviewed the first complete draft against Section 3 of submission draft (19), the proof, and the completed two-million-update data. A separate reader checked the progression and claims.

The opening now says that **holding the construction's relative bandwidth fixed** requires slopes proportional to width. This avoids turning a property of this construction into a universal lower bound for every accurate MLP. The scale-dependent spectral-bias antecedents are cited when the mechanism is first introduced.

The relative-rate paragraph now explains the essential constraint once: the largest eigenvalue limits a shared stable step, so target components with small eigenvalue ratios decay slowly. The multiplier paragraph then connects slope to the continuous-center quadratic form, the finite-grid correction, and finally the output-error bound. The target-energy consequence follows the theorem immediately, before the proof sketch. The proof explicitly substitutes into **squared relative output error**, rather than calling that scalar a residual.

The validation paragraph distinguishes the dense-grid direct-fit error from GD training error and reports trajectory-wide tightness as well as endpoint tightness. The joint-training paragraph leads with the accuracy gap between learned and supplied features. Slope summaries are identified as measurements of acquisition, not a universal sufficiency test. The actual Adam-scaled Jacobian diagnostic is quantified and distinguished from a convergence theorem.

The concluding transition states what the construction supplies—geometry and readout weights—and which search costs it avoids. The text does not claim a proved joint-training equilibrium mechanism or a general impossibility result for language models. The longer-horizon and intervention results remain pending for the second review; they may change the empirical interpretation.

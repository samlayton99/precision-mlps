# Section 3.4 writing reviews

The review criterion is whether each sentence advances the argument and gives the reader enough information to assess it. There is no word-count target. The surrounding Section 3 motivates accurate arithmetic primitives, constructs their geometry, and explains the relative-bandwidth regime. This subsection must explain the additional cost of learning those weights and lead naturally into constructing arithmetic circuits.

## Review 1: argument and scope

Reviewed the first complete draft against Section 3 of submission draft (19), the proof, and the completed two-million-update data. A separate reader checked the progression and claims.

The opening now says that **holding the construction's relative bandwidth fixed** requires slopes proportional to width. This avoids turning a property of this construction into a universal lower bound for every accurate MLP. The scale-dependent spectral-bias antecedents are cited when the mechanism is first introduced.

The relative-rate paragraph now explains the essential constraint once: the largest eigenvalue limits a shared stable step, so target components with small eigenvalue ratios decay slowly. The multiplier paragraph then connects slope to the continuous-center quadratic form, the finite-grid correction, and finally the output-error bound. The target-energy consequence follows the theorem immediately, before the proof sketch. The proof explicitly substitutes into **squared relative output error**, rather than calling that scalar a residual.

The validation paragraph distinguishes the dense-grid direct-fit error from GD training error and reports trajectory-wide tightness as well as endpoint tightness. The joint-training paragraph leads with the accuracy gap between learned and supplied features. Slope summaries are identified as measurements of acquisition, not a universal sufficiency test. The actual Adam-scaled Jacobian diagnostic is quantified and distinguished from a convergence theorem.

The concluding transition states what the construction supplies—geometry and readout weights—and which search costs it avoids. The text does not claim a proved joint-training equilibrium mechanism or a general impossibility result for language models. The longer-horizon and intervention results remain pending for the second review; they may change the empirical interpretation.

## Review 2: definitions and reader progression

A separate reader checked the revised section against Section 3.3, the proof, and the empirical protocol. The main defect was the order of explanation: the theorem introduced the corrected kernel before the reader knew why it existed, and the final empirical paragraph introduced a full Jacobian metric just as the argument should resolve.

The opening now starts from the useful bandwidth identified in Section 3.3 and states the acquisition question before discussing spectral bias. The explicit continuous-center kernel and the purpose of its two corrections now precede the theorem. The reader sees how gamma enters, why a finite-grid correction is needed, and which definitions the appendix supplies before encountering the rate endpoints. The proof sketch can consequently focus on interlacing and the GD error identity, without explaining the kernel a second time.

The joint-training paragraph now defines $a_j$ through the actual learned feature $\tanh(a_jx+b_j)$. Its slope statistics are identified as endpoints, since the trajectories need not increase monotonically. The main result is the output-accuracy gap accompanying substantial but heterogeneous slope acquisition. Detailed adaptive-metric percentages move to the empirical appendix, where the constant-rate control and its seed variability can be explained properly. The main text retains the local-direction interpretation and its distinction from the frozen-GD theorem.

This review changed the sequence of the argument and the placement of evidence; it did not optimize a length metric. Final numerical values and the causal interpretation of slope interventions remain for the third review after the longer runs complete.

## Review 3: quantitative argument and empirical interpretation

An independent reader checked the revised section, proof, five-million-update evidence, and rendered figure against the argument progression in Section 3.3. The theorem's direction, normalization, and all main numerical claims were independently verified. The numerical lower bound is now stated as $0.2098$, rounding down rather than converting an approximate $0.210$ into an overstated strict inequality.

The opening names the contribution directly: a slope-dependent output-error lower bound for the finite tanh construction. The validation paragraph keeps the concrete connection between slope, the highest target harmonic's multiplier, and the measured residual. This gives the reader a physical instance of the mechanism before asking them to interpret the aggregate tightness numbers.

The longer experiments changed the joint-training argument. Adam improves with the larger budget, and its upper slope percentile approaches the uniform reference. The revised text reports both facts, explains that only 4–7 of 512 features reach that reference, and uses checked direct fits and geometry interventions to identify the unresolved task as acquiring a useful combination of slopes and centers. It does not interpret a numerical direct-fit residual as an exact capacity floor or a finite budget as permanent failure. The actual Adam-scaled full Jacobian and constant-rate control now appear together in the appendix, where their endpoint-only interpretation can be stated alongside the evidence. Introducing another metric in the main conclusion interrupted its transition from measured acquisition cost to direct construction.

The main proof outline now gives the two necessary steps—interlacing and substitution into the exact GD error—and explicitly identifies the actual target projections. The full appendix retains the proof of each kernel identity and correction. This removes repeated theorem notation while keeping the mechanism and assumptions available to the reader.

The figure review retained A's theorem-versus-execution comparison and C's slope acquisition from exactly B's runs. Panel B's legend moved above the data. Its fixed-Adam display line now aggregates compact bin medians into 200 equal-width intervals, while preserving both endpoints and every raw within-bin extremum in the shading. Lighter shading and a thinner reference line separate typical progress from short spikes without hiding either.

The resulting argument is: the construction identifies useful geometry; slope quantitatively controls access to accurate readouts at that geometry; joint training improves but remains substantially less accurate within the measured budget; supplying geometry and readout directly avoids both search and iterative acquisition. This review evaluated what each statement contributes and whether the evidence supports it, without a word-count target.

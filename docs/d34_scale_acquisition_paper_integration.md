# Integrating the scale-acquisition argument into the paper

The [manuscript draft](d34_scale_acquisition_paper_draft.md) now follows the
reader's questions: what fails, what drives the remaining motion, why that
force strengthens slowly, and what can be proved and checked. It is
self-contained; operational provenance stays in this memo and the figure
records. Section, theorem, equation, and figure numbers remain provisional.

## Placement and chronology

Submission version 19 establishes the construction, compresses bandwidth
selection in Section 3.3, and leaves a joint-training placeholder after the
frozen-feature result in Section 4. The newer six-page spectrum note supplies
executed frozen-feature evidence and a clear joint-training observation.
Use that newer evidence in place of version 19's schematic Figure 3. Its
Section 3.5 observation becomes the opening of the dynamics argument, not
a second observation repeated elsewhere.

The revised argument has five steps:

1. **Observe the precision gap and population scale together.** Figure 4
   places joint Adam/GD and fixed-geometry Adam/GD output errors beside the
   RMS relative slopes of the same joint runs. Keep the common five-million
   budget and selection protocol visible. A supplied geometry is a useful
   comparison, not a universally necessary geometry.
2. **Identify the surviving force.** Explain $\nabla L=R+F$ before using
   either symbol in a theorem. Figure 5a shows the GD tracking transition.
   Coarse compensation remains in $F$ after tracking $R$ becomes small.
3. **Explain persistence under evolving geometry.** Introduce residual
   relaxation and geometry/compensation feedback through their effects on
   force. Figure 5b shows positive but limited reinforcement on the smooth
   step. This motivates the accumulated-feedback condition; it does not
   assume force decay or a stationary slope equilibrium.
4. **State the conditional theorem.** Its visible conclusions bound raw
   output improvement and RMS slope growth. The variables match the opening
   figure. Explain the force-amplification and collective-travel proof in
   three sentences; move the derivative formulas and disturbances to the
   appendix.
5. **Check the condition and its usefulness.** State the cross-target
   coverage and six-target bounds after the theorem. Figure S1 gives the
   detailed premise, force, displacement, and error comparisons in the
   appendix. Distinguish these 20k-to-120k GD audits from the much longer
   observation and from Adam.

This order puts the phenomenon before the explanation and the explanation
before the assumption. It avoids asking the reader to interpret a feedback
ratio or an endpoint-displacement bound before knowing why either matters.

## Figure roles and reading instructions

**Figure 4: the observation.** Reuse the spectrum note's paired layout with
RMS only. The rebuilt figure keeps all four error curves, both joint RMS
curves, seed variation, and the supplied $1/4$ reference. It removes the
99th-percentile curves. The ordinate is the actual scale
$\lambda_{\rm RMS}(t)=h\|\gamma(t)\|_2/\sqrt W$, not displacement since a
restart. The error ordinate is raw relative training error, with the saved
extrema preserved in display bands. Do not claim failure at 1% for the
Adam endpoint: its $1.81\times10^{-3}$ error already passes that threshold.
The relevant observation is its precision gap to $6.62\times10^{-7}$ on
supplied features.

**Figure 5: mechanisms.** Panel (a) shows full, effective fine, and coarse
tracking slope-gradient norms for mixed sine at width 177 over 600k GD
updates. Panel (b) uses the smooth-step effective continuation at width 705,
from age 20k through age 120k. It integrates the signed terms in the exact
log-force identity, separating positive feedback from negative relaxation.
The sum matches the actual log-force change. Caption and titles identify
the different studies; neither the force components nor the times are
spliced together. In particular, panel (a)'s crossing does not establish
panel (b)'s post-transient premise. That premise is separately audited.

The smooth-step example is useful because it prevents a misleading
contraction explanation: reinforcement dominates relaxation, force grows
2.187-fold, yet error remains near 49%. Limited amplification of initially
weak force is the point. Panel (a)'s late force increase likewise rules out
describing all trajectories as permanently flat.

**Figure S1: theorem evaluation.** The previous four-panel figure is retained
in Appendix D. It checks accumulated feedback against its allowance,
compares GD and effective force, bounds RMS displacement from the restart,
and compares output floors with errors. The displacement is explicitly
distinguished from Figure 4's RMS level. The 1% line is useful for this
six-target audit; it is not carried into the long Adam comparison.

The new figures and source/input hashes are documented in the
[narrative figure record](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/paper_narrative_final/README.md).
The retained conditional evaluations have their own
[bound figure record](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/paper_draft_final/README.md).
No new training was required. Function-preserving cloning and geometry
interventions remain in Appendix E as tests of competing explanations.

## The theorem presented to the reader

Theorem 4.2 remains a conditional result after tracking becomes unimportant.
Its assumption bounds accumulated reinforcement through evolving output
derivatives. Its conclusion allows evolving slopes, biases, and readouts,
positive force growth, and isolated neuron escapes. Neither a frozen
Jacobian nor an initial-data-only guarantee is required.

The main scale conclusion is now RMS growth, obtained directly from the
existing collective parameter-travel bound by triangle inequality. This
changes the presentation, not the assumptions or dynamics. The appendix
retains the stronger temporal statement about the fraction of neurons that
ever acquire a specified increment. No new maximum-neuron or concentration
premise is introduced.

Keep the exact force-growth identity and its labeled terms in the main
text. Defer the projector, compensation multiplier, Hessian contractions,
GD interpolant, and explicit tracking/discretization allowances. The RMS
allowance is $\Delta_{\rm scale}=hD_A/\sqrt W$ in the appendix's notation.
Output error supplies an independent success criterion because RMS alone
does not specify center coverage or useful output geometry.

The 23-target, two-seed audit establishes breadth; six dense continuations
check longer intervals and the reduction more closely. These counts must
not be added as independent target families. The observed factor-four
allowance is informative over the checked GD interval, not a prediction
covering all five million updates in Figure 4. Adam remains an empirical
motivation here, without a quantitative transfer of the GD theorem.

## Page budget and surrounding text

The draft has two main figures, each with two panels. They replace the old
main four-panel bound figure and absorb the spectrum note's joint-training
observation. Do not include a duplicate version of that observation or the
99th-percentile discussion. Keep the full six-target comparison, detailed
protocols, and numerical constants in the appendix. Final page fit requires
the full manuscript's LaTeX source and a compiled layout.

Version 19's main content reaches page 12, so the nine-page target still
requires compression elsewhere. Move the long primitive inventory and
Boolean/algorithm examples to the appendix, retain the compilation theorem
and representative examples, fold the repeated Corollary 5.1.1 specialization
into its preceding paragraph, and remove the full-page pipeline placeholder.
These cuts make room for the mechanism without reducing mathematical font
size or hiding the empirical setup.

Replace the contribution claim that the readout simply "rapidly absorbs
the residual" with:

> **Why gradient training can miss high precision.** Frozen-feature bounds
> quantify the learning cost of insufficient slopes. Joint-training
> experiments show a precision gap alongside limited population scale
> acquisition. For GD after the tracking transient, bounded aggregate
> reinforcement of the effective fine gradient limits both RMS scale growth
> and output improvement. Cross-target audits show that this condition can
> persist while the network continues to evolve.

Use $f,q_\theta,\gamma,\beta,w,b$ consistently for target, network, slopes,
hidden biases, readouts, and output bias. Preserve $F$ for effective fine
gradient and $R$ for tracking. The archived width-705 normalization uses
$h=2/512$; Figure 4 uses $h=2/467$ for width 512. Both are reference spacings,
not measured spacings between learned centers. Let the final manuscript
renumber the imported sections and figures automatically.

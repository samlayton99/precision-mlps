# Integrating the scale-acquisition argument into the paper

The [compiled paper section](../output/pdf/d34_scale_acquisition_paper.pdf)
is the current typeset deliverable: one page of prose, equations, and theorem;
one separate page holding the two figure pairs; and nine appendix pages.
The main-material footprint is still two pages with figures included.
The [main LaTeX inclusion](d34_scale_acquisition_paper_main.tex)
and [appendix inclusion](d34_scale_acquisition_paper_appendix.tex) can be
imported into the manuscript. They follow the reader's questions: what
fails, what drives the remaining motion, why that force strengthens slowly,
and what can be proved and checked. Reader-facing text is self-contained;
operational provenance stays in this memo and the figure records.

The main file includes [the two figure floats](d34_scale_acquisition_paper_figures.tex)
at its end. In the review PDF they occupy a separate float page; in the full
paper they can be placed near the relevant discussion. This exposes the
writing budget directly without shrinking plots or counting their space as
free. The observation pair should replace the existing joint-training
figure, so it is counted only once in the complete manuscript.

The [review wrapper](d34_scale_acquisition_paper_review.tex) matches the
spectrum note's 10-point Times, 5.5-by-9-inch text area. It sets the proposed
insertion to Section 3.5, Theorem 3.3, and Figures 4–5; the inclusion files
use automatic labels and contain no hard-coded counters. This is a review
layout, not verification of the complete paper's nine-page fit. The
[Markdown companion](d34_scale_acquisition_paper_draft.md) carries the same
argument and evidence, with a more segmented appendix and local equation
numbering. The older `output/pdf/scale_acquisition.pdf` is an archived
technical note, not this draft.

Build from the repository root:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tmp/pdfs/scale_acquisition_review docs/d34_scale_acquisition_paper_review.tex
cp tmp/pdfs/scale_acquisition_review/d34_scale_acquisition_paper_review.pdf output/pdf/d34_scale_acquisition_paper.pdf
```

The final PDF was rendered page by page and inspected for clipping, figure
legibility, mathematical layout, and reading order. The build has no
unresolved references or overfull/underfull boxes. Its two-page main
footprint comprises one writing page and one figure page. The writing page
includes the force identity, theorem, feedback-loop interpretation, and
empirical check at normal 10-point size.

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
   either symbol in a theorem. Figure 5a shows small tracking during the
   actual post-transient continuation used in panel (b).
   Coarse compensation remains in $F$ after tracking $R$ becomes small.
3. **Explain persistence under evolving geometry.** Introduce residual
   relaxation and geometry/compensation feedback through their effects on
   force. Figure 5b shows positive but limited reinforcement on the smooth
   step. This motivates the accumulated-feedback condition; it does not
   assume force decay or a stationary slope equilibrium. Briefly state the
   complementary loop mechanism: weak force permits little travel, limited
   travel produces limited extra reinforcement, and weak force persists.
4. **State the conditional theorem.** Its visible conclusions bound raw
   output improvement and RMS slope growth. The variables match the opening
   figure. Explain the force-amplification and collective-travel proof in
   a short paragraph; move the derivative formulas and disturbances to the
   appendix.
5. **Check the condition and its usefulness.** State the cross-target
   coverage and six-target bounds after the theorem. Figure S2 pairs the
   premise check with force evolution; Figure S3 pairs RMS displacement
   with output error. Distinguish these 20k-to-120k GD audits from the much
   longer observation and from Adam.

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

**Figure 5: mechanisms.** Both panels use smooth step, width 705, seed 30,
and the same unmodified checkpoint at 20k GD updates, followed to 120k.
Panel (a) shows full-parameter effective fine and tracking norms on the
GD path. Panel (b) integrates the exact signed force-growth identity along
the paired effective flow, separating positive feedback from negative
relaxation. Both axes show total training age. This replaces the earlier
pairing of different targets and widths, and uses the same full-gradient
norm as the force identity. Tracking is below 0.1963% of effective force
at sampled times; the GD/effective-flow force discrepancy is below 0.0795%.

The smooth-step example is useful because it prevents a misleading
contraction explanation: reinforcement dominates relaxation, force grows
2.187-fold, yet error remains near 49%. Limited amplification of initially
weak force is the point. No panel supports a universal contraction claim.

**Figure S1: the earlier tracking transition.** The width-177, five-seed
mixed-sine history is retained as a separate appendix figure. It uses
slope-block norms over 600k updates. Its crossing does not determine the
restart time for the matched mechanism figure. Keeping it separate preserves
the informative early history without mixing cohorts or norm definitions.

**Figures S2–S3: theorem evaluation in two pairs.** The former four-panel
composite is absent from the new note. Figure S2 checks accumulated feedback
against its allowance beside GD/effective-force agreement. Figure S3 bounds
RMS displacement from the restart beside output-error floors. Displacement
is explicitly distinguished from Figure 4's RMS level. The 1% line belongs
to this six-target audit; it is not carried into the long Adam comparison.

The new figures and source/input hashes are documented in the
[narrative figure record](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/paper_latex_figures/README.md).
The retained conditional evaluations have their own
[bound figure record](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/paper_latex_bounds_final/README.md).
No new training was required. Post-processing ran on Modal with a 4-GiB
limit; the final narrative and bound jobs peaked at 339.1 and 152.7 MiB,
respectively. The plotter exports vector PDFs plus SVG/PNG previews.
Function-preserving cloning and geometry interventions remain in the
appendix as tests of competing explanations.

## The theorem presented to the reader

Theorem 3.3 remains a conditional result after tracking becomes unimportant.
Its assumption bounds accumulated reinforcement through evolving output
derivatives. Its conclusion allows evolving slopes, biases, and readouts,
positive force growth, and isolated neuron escapes. Neither a frozen
Jacobian nor an initial-data-only guarantee is required.

Theorem A.2 now contains the population feedback-loop theorem and its full
first-exit proof, corresponding to Theorem 17 in the longer population note.
It also states a disturbance extension for GD. Its stronger premises control
additional reinforcement per accumulated population travel and accumulated
force concentration. The main text gives the mechanism in a short paragraph;
the constants, timescale, and proof remain in Appendix A.4.

The stronger sufficient criterion has narrower coverage: all 46 broad paths
and six dense paths pass over 20k further updates; three of the six dense
paths pass at 100k. The broader feedback-budget theorem remains useful on all
six longer continuations. This is why the loop theorem supplements the main
result instead of replacing it. The appendix reports the retrospective
fitting of the response coefficient and its longer-interval failures.

The main scale conclusion is now RMS growth, obtained directly from the
existing collective parameter-travel bound by triangle inequality. This
changes the presentation, not the assumptions or dynamics. The appendix
retains the stronger temporal statement about the fraction of neurons that
ever acquire a specified increment. No new maximum-neuron or concentration
premise is introduced.

Keep the exact force-growth identity and its labeled terms in the main
text. Defer the projector, compensation multiplier, Hessian contractions,
GD interpolant, and explicit tracking/discretization allowances. The RMS
allowance is $\Delta_{\rm scale}=h[t e^{B_t}U(t)+V(t)]/\sqrt W$.
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
99th-percentile discussion. The main argument now occupies one page in
the review layout, with no reduced font sizes; the two figure pairs add one
separate page. Its nine-page appendix holds both proofs, the six-target
comparison, and protocols. Final page fit
still requires the complete manuscript's LaTeX source and its actual style.

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

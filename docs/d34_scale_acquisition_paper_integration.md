# Integrating the population argument into manuscript version 19

This is an author-facing placement memo. The accompanying
[manuscript draft](d34_scale_acquisition_paper_draft.md) is self-contained:
its main-text section uses the paper's notation, and its appendix contains
the precise theorem, proof, GD disturbance terms, and empirical methods.
The section and figure numbers are provisional.

## What the latest manuscript already establishes

The source is version 19, created September 24, 2026, containing 15 PDF pages.
Theorem 3.1 on page 5 gives a clean conclusion with labeled contributions,
then sends admissibility and implementation details to the appendix.
Section 3.3 compresses bandwidth selection into one equation and its
interpretation. The new dynamics section follows those presentation choices.

Section 4 starts on page 6. It first explains why QUILL uses width-scaled
slopes, then gives the frozen-feature learning-time theorem on page 7.
Its final paragraph is a placeholder for why joint training fails to acquire
those scales. The new Section 4.2 replaces that paragraph. Add a Section 4.1
heading to the existing frozen-feature argument; retain its role and avoid
re-proving it in the new subsection.

The PDF's main content reaches the Discussion heading on page 12. The
nine-page budget therefore requires net compression elsewhere. The new
subsection has roughly 700 words before its caption, two displayed equations,
and one compact four-panel figure. Reserve approximately 1.5–2 pages for it
and about 2–2.5 pages for Section 4 as a whole, subject to the actual LaTeX
layout. This is a drafting allocation, not a verified final page count.

## The argument, in order

1. **A useful geometry is known.** Section 3 provides a constructive reference
   with relative bandwidth $\lambda=\gamma h$. This is a benchmark, not a
   theorem that every accurate network requires that geometry.
2. **Small slopes can make remaining error hard to learn.** The existing
   frozen-feature theorem identifies restricted readout access to necessary
   target corrections. Keep this conditional on the target tail and slope
   regime appearing in that theorem.
3. **Joint training need not repair this limitation quickly.** Open the new
   subsection with a counterexample to the idea that any slope or force growth
   solves the problem: the smooth-step force roughly doubles while raw error
   remains about 49%.
4. **Identify the force that matters after the transient.** Explain
   $\nabla L=R+F$ in words. Tracking becomes small; balanced coarse compensation
   remains inside the effective fine gradient. This paragraph establishes
   the regime without introducing a new catalogue of force terms.
5. **Explain persistence through the force identity.** Fitting away current
   driving error depletes force. Evolving geometry and compensation can
   reinforce it. The theorem assumes an aggregate bound on that reinforcement,
   whose persistence is checked empirically.
6. **State the conditional population and output theorem.** Present the two
   interpretable consequences, a short proof sketch, and their meaning for a
   target scale and accuracy.
7. **Test the condition, then its consequences.** Use the main figure and
   broad coverage statement to show persistent feedback slack, limited force
   amplification, small population movement, and large raw error.

The main theorem should follow one motivating observation, with the
systematic empirical evaluation after it. Placing all evidence after the
theorem would make the reader first confront an unexplained assumption;
placing the full experimental catalogue first would delay the mechanism.

## Why this is the main-text theorem

Theorem 4.2 is a direct corollary of the existing accumulated-feedback
identity and disturbance comparison. It uses a second-power population
counting bound. This is sufficient here and removes the need to introduce
fourth moments and force concentration into the main statement.

The two displayed conclusions have distinct roles. The acquisition fraction
states what happens to the population, allowing isolated escapes. The
output-error floor establishes failure under a common raw-error criterion,
without assuming that missing a QUILL slope benchmark proves inaccuracy.
The sole reinforcement allowance $B_t$ has a physical interpretation;
the two GD allowances collect only tracking and discretization effects.
Their exact definitions are in the appendix, just as the numerical allowances
of Theorem 3.1 are deferred.

Keep the force-growth identity in the main text. Its two labeled terms
explain what the hypothesis measures and why it is mechanistic. Move the
projector, compensation multiplier, Hessian contractions, integrating-factor
calculation, and GD interpolant into the appendix. No theorem about entry
from initialization or a checkpoint-only persistence certificate belongs in
this main argument.

The newer response-to-travel bootstrap is supplementary. Its shorter
coverage on some 100k continuations should not displace the more broadly
informative accumulated-feedback theorem. The paper need not claim that a
single stronger sufficient condition explains every observed duration.

## What each empirical panel establishes

The new Figure 4 uses all six densely audited targets, one seed, width 705,
and the same restart and continuation budget. It has been generated from
the existing scalar traces and small endpoint arrays.

- **Panel (a): the hypothesis persists.** Plot accumulated directional
  feedback divided by the allowance $4d_{\rm dir}(0)t$. A horizontal line
  at one makes slack and failure interpretable. The factor four is from the
  previously evaluated sensitivity family, not a universal constant.
- **Panel (b): force need not decay.** Plot normalized effective force for
  GD and effective flow. Increasing, decreasing, and nearly constant examples
  all appear. This tests the evolving reduction and makes clear why a
  contraction or stationary-point story is too restrictive.
- **Panel (c): population movement is insufficient.** Compare endpoint RMS
  changes in normalized slopes with the conditional upper allowance. This
  replaces a maximum-neuron plot in the main figure. The population count in
  the theorem follows from collective travel; endpoints alone do not measure
  the fraction that ever escaped.
- **Panel (d): the failure is in raw output accuracy.** Plot the actual
  errors and conditional effective-flow floors using attached readouts,
  with the same checkpoints and a 1% reference line. No least-squares refit
  is substituted for the trained network.

The main figure establishes sampled persistence and useful consequences.
It is not, by itself, a causal intervention. Function-preserving cloning,
feedback interventions, and large geometry injections belong in the appendix
as tests of competing explanations and of the condition's domain. The familiar
tracking-force and slope-scale plots should also remain there as evidence
for the post-transient reduction. They use different widths, seeds, and
horizons; do not splice them into Figure 4 as if they were the same runs.

The 23-target, two-seed result supplies breadth; the six-target study supplies
denser temporal verification. State both counts without adding them together.
The main text reports the broadly interpretable facts: at most 55% usage of
the longer allowance, RMS slope-movement bounds below $8\times10^{-4}$, and
error floors above 39%. Detailed rates, target formulas, sample spacing,
tracking effects, and refinement checks remain in the appendix.

## Changes needed elsewhere in the paper

Replace the second contribution bullet's statement that the readout
"rapidly absorbs the residual" and its unqualified claim of geometry failure
with the following:

> **Why gradient training can miss the high-precision regime.** QUILL exposes
> a width-dependent slope regime, while a frozen-feature theorem quantifies
> the learning cost of insufficient slopes. For joint GD after the initial
> transient, we prove that bounded aggregate reinforcement of the effective
> fine gradient limits population scale acquisition and preserves output
> error. Cross-target experiments show that the condition remains satisfied
> over long measured intervals, explaining failure despite continued
> parameter movement.

In the introduction's preceding prose, qualify "gradient descent fails to
realize" by the studied targets and training budgets. Avoid turning the
construction's sufficient geometry into a universal necessary condition.
The related-work discussion can continue to distinguish frozen readout
access from evolving-geometry persistence.

To make room, move the long primitive inventory in Table 2 and the Boolean
and algorithm examples to the appendix, retaining representative examples
and the compilation theorem. Corollary 5.1.1 repeats the specialization already
given immediately after Theorem 5.1 and can be folded into that paragraph.
The full-page pipeline placeholder on page 10 can be replaced with a compact
diagram or removed. These are concrete sources of space for the mechanism
section; shrinking the mathematical font is not the proposed compression.

Figure 3 in version 19 is explicitly labeled schematic with numerical
results pending. Its frozen-feature evidence must be populated from its own
audit before submission. Figure 4 here does not verify the old Figure 3's
claims. The new figure displaces the existing pipeline placeholder's figure
number; let the manuscript's references renumber automatically.

The archived pair $(W,N)=(705,512)$ is not the literal
$W=N+2\lceil\sqrt N\rceil+1$ geometry in the current construction theorem.
Keep $h=2/512$ labeled as the reference normalization used in these
experiments. The output-error conclusion is independent of that convention.
Frozen-feature and joint-training panels answer different questions; any
direct GD/Adam ranking still needs the same raw-error criterion, training
budget, and checkpoint-selection rule.

## Notation and verification

The draft follows the manuscript's $f$ for the target and $q_\theta$ for the
network, with $\gamma,\beta,w,b$ for slope, hidden bias, readout, and output
bias. These correspond to $a,b,c,d$ in the working technical notes. The
canonical force names $F$ (effective fine gradient) and $R$ (tracking) are
unchanged. Here $B_t$ is accumulated feedback, not a population sixth norm.

The manuscript draft is self-contained and contains no repository citations.
The generated figure is available in PNG and SVG formats. Its numerical
provenance and plotting source are documented in the
[figure evidence record](../results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/paper_draft_final/README.md).
The final nine-page fit still requires the full manuscript's LaTeX source;
the source corresponding to PDF version 19 was not found in this checkout.

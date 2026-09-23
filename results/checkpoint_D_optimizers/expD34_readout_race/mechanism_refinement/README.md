# Testing what limits post-transient scale acquisition

This study asks whether small slopes persist because coupled feedback restores
their previous geometry, because readouts remove the error driving geometry
before slopes move far, or because Adam's moment history suppresses net motion.
The experiments below distinguish these explanations through predictions issued
before each continuation. Their outcomes are not yet established.

**Table 1. Notation shared with the acquisition theorem.**

| Symbol | Meaning |
|---|---|
| $a,b,c,d$ | Signed slopes, hidden biases, readouts, and output bias. |
| $\gamma_j=|a_j|$ | Physical slope magnitude. |
| $\lambda_j=h|a_j|$, $h=2/N_{\mathrm{ref}}$ | Normalized scale; $N_{\mathrm{ref}}$ is construction resolution. |
| $W$ | Actual neuron count, including halo neurons. |
| $F=Te$ | Effective fine gradient, including the balanced coarse contribution. |
| $r_C$ | Coarse disequilibrium gradient. |
| $R=r_C+g_\perp$ | Tracking and omitted-mode correction. |

## A prospective test of coupled feedback

The original force audit covers 23 function instances. The present development
cohort uses the 13 original functions at seed 0 and the 10 additional functions
at seed 22; confirmation uses seeds 20 and 23 respectively. All GD forks are at
600,000 updates. These functions have already been studied: confirmation refers
to new intervention outcomes, not previously unseen target functions.

The primary observations are per-neuron normalized-scale change, its rate,
positive and negative travel, and the fraction ever reaching $\lambda=0.25$.
Loss and readout magnitude provide context; neither substitutes for acquisition.
Training remains noiseless full-batch optimization in raw physical coordinates.

**Matched pulses.** Find an outward direction in the nullspace of the coarse
output Jacobian, coarse-disequilibrium derivative, and actual slope-gradient
derivative. The latter includes residual curvature. Use symmetric positive and
negative pulses at three halved amplitudes, with matching errors checked against
their predicted orders. A numerically unresolved outward direction is a design
limitation, not evidence for stagnation. Compare retained displacement, return,
and amplification with fork-issued frozen-map and full-Hessian local-response
predictions. Both derivative matrices are fixed at the fork; the second retains
the local derivative of the changing effective map.
Anchor both predictions at each pulsed state's actual initial gradient. Small
$R$ does not imply small $DR$: retain its derivative consistently when testing
the subsequent feedback response.

**Exact splitting.** Replace each neuron by two or four identical copies with
readout $c_j/k$. Compare ordinary rates, geometry-only compensation, readout-only
compensation, and both compensations. Geometry compensation multiplies its rate
by $k$; readout compensation divides its rate by $k$. Both together reproduce
the original trajectory exactly. Main runs may use the exact replica-symmetry
quotient after verification against explicitly expanded networks. The quotient
has hidden mobility $1/k$ and aggregate-readout mobility $k$ for ordinary split
GD. Compute coarse balance in each branch's actual parameter metric. Keep the
original $h$ for clones; genuine width changes require independent networks.

**Adam histories.** At the original 13 functions' 600k Adam checkpoints, seed 0
is development and seed 1 is confirmation. Independently attenuate future slope
tracking inputs to first and second moments by factors 1 or 0.9. Preserve all
incoming moments and the update count. Square the complete modified input to
the second moment, including cross terms. Estimate both channels' two phases
from the fork and one virtual ordinary Adam update; issue periodic-driver and
constant-driver forecasts before observing intervention continuations. These
are conditional local forecasts, not a claim that only tracking oscillates.

Initial common horizons end at 20k additional updates. No endpoint is a
theoretical barrier. Any later protocol revision must identify which results
informed it; confirmation outcomes are not used silently to retune predictions.

## What would refine the theorem?

Predicted return motivates a conditional response-stability bound; retained
offset with decaying drive motivates a finite-displacement bound. Splitting
tests whether relative readout and geometry timescales explain these responses.
Adam tests whether a cancelling tracking signal still changes effective
mobility through its second-moment history. A failed forecast identifies an
unresolved coupling; it is not repaired by fitting the observed future curve.

The GD premise is interval-wide small tracking, separately in direct slope force
and fine-residual forcing. The experiment does not prove entry into that regime
or persistence of this premise. Any empirical error envelope is distinguished
from a uniform, numerically certified theorem enclosure.

## Execution and evidence preservation

The eight-hour research window begins at 2026-09-23 08:18:25 UTC. New GPU use is
capped at four GPU-hours and must also fit the reconciled prior authorization.
The effective-feedback ledger records 5.204 GPU-hours; charging the older Adam
audit's 0.789 hours as well still leaves approximately four hours below ten.
Previously queued plateau GPU jobs 1025, 1036, 1038, 1040, and 1041 were cancelled
before allocation, as checked in Slurm accounting at the start of this study.
All numerical work uses Runpod Slurm; at most two GPUs may be allocated at once.

Keep input hashes, prediction files, sparse states, online motion accumulators,
verification results, and final analysis. Dense windows are short. Remove only
redundant campaign scratch after checking that unique evidence is retained.
The last part of the research window is reserved for analysis, theorem revision,
and a directly authored explanation of the results.

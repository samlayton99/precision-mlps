# Evaluate the proved persistence conditions before claiming useful delay

This retrospective audit asks whether the new exact-tanh moment theorem
gives useful intervals on the populations excluded by the earlier alignment
theorem. The mathematical proof is reviewed before numerical evaluation.
Its correctness and its practical usefulness are separate questions.

| Quantity | Meaning |
|---|---|
| $W$, $h$ | Physical neuron count and verified construction spacing. |
| $r_0$ | Maximum Euclidean norm of $X_j=\sqrt W(a_j,b_j,c_j)$. |
| $M_0$, $E_{s,0}$ | Initial total particle moment and slope–readout moment. |
| $Y_0$ | Initial full fine-residual norm; not the full training residual. |
| $L_0$, $z_0$ | Initial total loss and coarse disequilibrium for ordinary GD. |
| $T=t/W$ | Reduced effective-flow time; $t=\eta N$ for an update-count conversion. |
| $\lambda=h|a|$ | Normalized slope scale. |

## Fixed evidence and roles

Use the 223 distinct states in the existing population-coverage inventory,
with the same source hashes and construction metadata. The 40 natural GD
continuations supply 320 saved states and per-neuron counters accumulated
over every update. The regular continuation panel contains six targets,
two seeds, and three physical widths; the broad static panel contains
23 target instances. All are retrospective evidence. No new training,
target selection, or held-out generalization claim is involved.

Use exact tanh features and the full empirical complement of constant and
linear functions. Read learning rates from explicit case or run metadata.
Missing learning rates prevent an update-count statement, but do not prevent
evaluation in continuous physical time. Never replace $h$ by a formula in
physical width.

## Initial-data evaluation

For every state evaluate the fixed multipliers
$1.01,1.025,1.05,1.1,1.25,1.5,2$. In the effective-flow theorem the multiplier
sets only the outer support radius; its moment bounds depend on the proposed
horizon through the proved formulas. In the GD corollary it sets
$R=\mu r_0$, $\overline M=\mu^2M_0$, and
$\underline E_s=E_{s,0}/\mu^2$. This convention fixes the regional choices
before reading new outcomes.

Compute the effective-flow horizon by monotone search with a strict positive
Gram perturbation gap and a strict support margin. For GD, iterate the proved
scalar bounds until the first failed support, moment, or step-size condition.
Retain every multiplier and choose the best horizon from initial quantities
only. Distinguish a valid fractional-time bound from one that supports a
positive integer number of GD updates. Recheck every reported integer endpoint.

Record invalid initial hypotheses, failed conditioning, failed step-size
tests, exhausted moment/support margins, missing metadata, numerical
uncertainty, and valid but uninformative acquisition bounds separately.
Do not replace negative conditioning gaps by positive squares or numerical
floors. Do not replace a failed theorem condition by the observed trajectory.

## Compare only the applicable conclusions

Effective-flow conditions are an applicability calculation. They do not
bound the ordinary-GD trajectories without the discrete corollary.
At saved steps within a valid GD interval, compare support, moments,
conditioning, tracking, and fine-force norms with the scalar envelopes.
Use cumulative counters to compare travel and the fraction of labels ever
reaching $\lambda=0.25$. Those counters cover that threshold only; other
thresholds receive analytic bounds or explicitly sampled diagnostics.

Compare new horizons with the previous generic closure, whose best reported
interval was 18 updates, and with the available 20,000-update continuations.
These are descriptive comparisons, not requirements that the new theorem
must pass. Report which inequality limits each case. A short valid horizon
is evidence that the constants need improvement, not a proof of rapid
acquisition or a reason to discard the case.

## Verification and retained artifacts

Verify the raw particle force, metric rescaling including output bias,
projection contraction, Gram perturbation, moment directional derivatives,
and force derivative against independent computations. Exercise mixed signs,
nonzero biases, small slope–readout energy, zero residual, and deliberately
invalid margins. Distinguish these code checks from tests of the mechanism.

All numerical execution uses CPU Slurm allocations with GPUs disabled.
Keep source/input hashes, compact CSV and JSON tables, a small set of
figures, job logs, and failure counts. Do not create long training traces.
FP64 outputs are numerical evaluations, not rounding-controlled certificates.
Promising positive intervals receive a separate outward-rounded check; if
none is informative, report that outcome and identify the limiting estimate.

The report is authored directly in Markdown after inspecting the evidence.
The central note will link the full proof and the empirical assessment.
Adam and the PI packet remain outside this round.

## Follow-up fixed after inspecting the primary outcome

The primary calculation admitted 22 of 223 initial regions and supported
at most five ordinary-GD updates. A separately proved refinement in
Section 6 of the GD note feeds the initial effective-force norm into all
subsequent movement bounds. It replaces the tracking recurrence's uniform
regional speed allowance by the recursively bounded actual speed. This is
a follow-up motivated by the primary failure, not a preregistered result.

Evaluate the refinement on exactly the same panels and seven multipliers,
in a separate output directory using `--force-coupled`. Retain the primary
outcome. Do not change initial conditioning tests or insert future observed
forces. Compare valid horizons and the same archived conclusions. A longer
but still short interval remains an uninformative long-horizon result.

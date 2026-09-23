# Testing why a weak effective fine force persists

This study starts from archived GD states after the coarse fit has settled.
The question is whether the effective fine force stays weak because it is
actively damped, because its reinforcing feedback is slow, or because its
motion points away from outward scale acquisition. We test the feedback
inside the dominant force; we retain the coarse correction in every released
GD trajectory.

The [theory note](../../../../docs/d34_state_dependent_persistence.md) gives
the identities, conditional discrete theorem and proofs. This protocol fixes
the tests before continuation. It does not claim that checkpoint diagnostics
are uniform neighborhood bounds.

## What changes, and what stays matched

For full-complement effective gradient $F$, force energy satisfies

$$
\frac{d}{dt}\frac{\|F\|^2}{2}
=-\mathcal D-\mathcal C-F^TDF[R].
$$

Residual relaxation contributes $\mathcal D\ge0$. Loaded geometry curvature
$\mathcal C$ can damp or amplify. The last term transfers the statement to
ordinary GD and is measured explicitly. All three terms are evaluated before
seeing a continuation.

Six feedback arms independently multiply geometry feedback and residual
relaxation by $(\kappa,\nu)=(1,1),(0,0),(0,1),(2,1),(1,0),(1,2)$.
They have the same full gradient at the fork. Their first step therefore
agrees. The natural arm is ordinary GD at every later state; other arms are
controlled vector fields. Their exact local contrasts verify the code.
Whether those contrasts remain predictable over a finite window tests the
scientific closure.

The primary closure is deliberately small: freeze the fork's logarithmic
force-growth rate $k_0$, predict $F_n=\exp(\eta k_0n)F_0$, and freeze the
direct correction at $R_0$. Each arm uses its own analytically computed
ordinary-flow rate. The baseline freezes the entire initial gradient.
Predictions and source/input hashes are saved before continuation. No rate
is fitted to future states. A successful force-norm prediction accompanied
by a poor slope-vector prediction rejects scalar growth as a sufficient
closure and identifies direction as a missing state variable.

A second test changes actual networks and then releases ordinary GD.
Physical kicks keep slopes exactly fixed and match the coarse output,
coarse disequilibrium, initial slope gradient and force norm to first order.
They maximize or minimize intrinsic amplification in the feasible tangent
space. Amplitudes are $0,\pm0.01,\pm0.005,\pm0.0025$ in the declared block
RMS metric. Halving checks linear rate changes and quadratic matching errors.
The full force direction is not matched. An unresolved feasible direction
is recorded as unavailable, not as a negative scientific result.

## Fixed breadth and horizons

Feedback uses six target instances: degree-five moment, mixed sine, left
Gaussian, right smooth bump, right tanh step and absolute-value kink.
Each has independently initialized $N_{\rm ref}=128,512,1024$ states,
with physical widths $W=177,705,1409$. Development seed 30 and confirmation
seed 32 each supply 18 forks. Late pure sine and degree-nine states add two
forks per cohort, using seeds 0 and 20 respectively. This gives 40 forks
and 240 feedback branches. The full 23-function development and confirmation
archives also receive checkpoint diagnostics.

Physical kicks use the six $N_{\rm ref}=128$ forks and the two late forks
per cohort: 16 source states and at most 112 branches. No target is removed
because its prediction is poor. The fixed learning rate is $\eta=0.002$;
saved horizons are 0, 1, 2, 10, 100, 1,000, 10,000 and 20,000 additional
updates. These horizons test local persistence and introduce no universal
training-step barrier.

## Evidence that can discriminate the hypotheses

The main quantities are force norm, the full slope-motion vector, signed
change in $\lambda_j=h|a_j|$, positive slope travel, sign-crossing correction,
and distinct first hits of $\lambda=0.25$. Here $h=2/N_{\rm ref}$, not
$1/W$. Initial occupants and new hits are reported separately.

For feedback, compare each arm with natural GD at the same fork. Score the
preissued contrast against the zero-contrast prediction and report its sign,
size and vector error. Removing damping should increase initial force growth
by the computed relaxation contribution; removing reinforcement should
decrease it when the computed loaded geometry contribution is amplifying.
If their effects remain small because both clocks are slow, that supports
weak reinforcement over the tested window, rather than a strong restoring
mechanism. A resolved opposite contrast rejects the declared forecast.

For physical kicks, compare displacement from each branch's own initial
state, plus antisymmetric positive/negative responses and amplitude scaling.
Retain the ordinary-GD loaded correction when forecasting. Small or
unresolved responses are inconclusive; neither sign can be counted as
confirmation after the fact.

The theorem requires regional bounds on amplification, discretization and
tracking response. Passing these trajectory tests motivates those bounds;
it does not certify them. Checkpoint rates, sampled future rates, analytic
conditional proofs and verified numerical enclosures remain distinct claims.

## Execution limits and verification

The round starts at 19:19:09 UTC on 2026-09-23 and stops by 21:19:09 UTC.
At most 3,600 allocated GPU-seconds may be used, including compilation and
failures. All numerical preparation, tests, continuations and analysis run
under Runpod Slurm. The budget ledger records allocations before launch.
The source upload uses the existing authorized campaign directory.

Independent derivatives verify the energy identity and generated/target
curvature split. Tests also check exact natural-GD recovery, initial matching,
second-step scaling, zero-force handling, physical constraint scaling,
signed travel accounting and failed-update handling. Sparse states and
cumulative budgets replace long optimization traces. Feedback has priority
over physical continuations if the wall or GPU cap becomes binding.

## Follow-up diagnostics after observing the initial results

The user requested both a looser acquisition envelope and continued study of
the coupled mechanism. The following analyses were added after the primary
outcomes were known. They are retrospective comparisons, with no model
coefficients fitted to future states; they are not additional prospective confirmations.

The envelope audit compares saved speed ratios and all-step accumulated
effective travel with the issued reference. A factor two is a diagnostic
candidate selected after inspecting the misses. Its acquisition-margin
calculation uses observed tracking travel and is explicitly not a certificate.

For physical responses, a frozen full loss Hessian predicts the derivative
of displacement by $[(I-\eta H_0)^n-I]v$. A parallel prediction uses the
frozen derivative of $F$. These keep vector coupling omitted by the scalar
model but freeze subsequent operator evolution. Existing coupled quintic
models instead evolve all parameters and sensitivities from each physical
fork. One retains the projected fine field; the other retains the full
polynomial gradient to account for the transient introduced by finite kicks.
Each uses a constant initial exact-minus-polynomial correction computed at
its own fork. The fine model matches initial $F$; the full model matches
initial $g$. The correction need not remain tangent away from the fork.

The primary response comparison uses amplitude 0.0025 at 20,000 updates;
all other amplitudes, horizons, failures and unsupported predictions remain
available. Raw slope-vector response, readout response and baseline motion
are different scores. Finite polynomial trajectories do not establish
Taylor validity, particularly at the late sine forks. These diagnostics
locate missing coupling and motivate a subsequent prospective test.

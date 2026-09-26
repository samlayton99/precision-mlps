# Frozen-geometry readout sweep: experiment map

Status: phase one selected and completed on 20 September 2026 as expD36, through 20,000 steps. See [the result record](../../results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/expD36_results.md). The broader ablations below remain proposals. Direct theorem tests are deferred.

## Phase-one specification from Sam

This section supersedes the broader candidate design below where they differ. Sam authorized the first run after the design discussion; its four figures and resumable data are now saved. The proposed browser was dropped.

- Figure/browser layout: 3 rows by 4 function columns.
- Top row: relative L2 error versus lambda, with GD, Adam, and dotted numerical least squares. Four static figures select step 2,000, 10,000, 20,000, and best recorded error.
- Middle row: GD relative L2 error versus gradient step, one curve per lambda.
- Bottom row: Adam relative L2 error versus gradient step, one curve per lambda. Use the same lambda colors in both trajectory rows and across targets. A vertical marker can identify the top row's selected step.
- Initial horizon: 20,000 steps, with full optimizer checkpoints for continuation.
- N=256 under the existing h=2/N grid convention; identical positive gammas and uniform centers. Hidden geometry frozen throughout. Physical readout and output bias initialized to zero.
- Eight approximately logarithmically spaced scales, from the PR's Xavier slope RMS to lambda=.5. Replace the nearest interior log-grid point by lambda=.25.
- Approximately 1,024 training samples and 4,096 independent evaluation samples, adjusted to avoid systematic grid alignment. Proposed: 1,021 training midpoints and 4,093 evaluation midpoints on [-1,1]. These counts are coprime to one another and N; the two odd midpoint grids share the central sample but are otherwise distinct. Endpoint grids require reasoning about interval counts rather than point counts.
- Steps only; elapsed-time comparisons and the browser are dropped for this phase.
- Selected: note sine/quadratic/mixed sine plus repository Runge; raw coordinates; GD rate 1/||B||²; Adam rate .001, betas (.9,.999), epsilon 1e-8; constant schedules. Retain 24 halo centers per side.

With 24 halo centers per side, this grid has W=305 neurons. The PR's tanh-gain Xavier slope RMS is (5/3)*sqrt(2/(W+1)), giving gamma approximately .134742 and lambda approximately .00105267. The selected lambda values are approximately .00105267, .00253907, .00612432, .0147720, .0356306, .0859419, .25, .5. The .25 point replaces .207294 in the strict geometric grid.

Zero initialization removes physical-start scaling as a phase-one factor: sqrt(h)*0 is still zero. A coordinate map c=M theta remains consequential because it changes updates. A common scalar map for every readout parameter is equivalent to a learning-rate rescaling for GD; different bias/halo scales and neighboring maps are not a common scalar map.

## Question and sources

**At a fixed training budget, how much does changing lambda improve readout fitting, and how much of that improvement survives better readout conditioning?**

Use the empirical part of Section 5 of [the revised note](../../papers/optimization_notes/frozen_geometry_capacity_access_note.pdf), the merged PR #2 components reviewed in [pr_conditioning_review.md](pr_conditioning_review.md), and Junmiao's message. The message prioritizes trained relative L2 error across scales, with least squares as a capacity reference. It explicitly does not ask us to explain the entire gap to machine precision with gamma.

The PR supplies useful coordinate transforms and starts, not this complete frozen-lambda training sweep. Its existing entry points need adaptation: one fixes lambda and uses line search, while another updates slopes. Reuse its components, not those entry points unchanged.

## 1. Core experiment

For each selected target and lambda:

1. Construct the geometry and freeze every hidden slope and bias/center.
2. Train only the readout coefficients and output bias, separately with ordinary full-batch GD and Adam.
3. Save evaluations at shared training steps and resumable optimizer checkpoints.
4. Compute a separate numerical least-squares readout reference for this exact geometry.

Recommended first geometry: uniform centers and a common positive gamma. With center spacing h, lambda = gamma h. This gives the horizontal axis a single unambiguous meaning. Small lambda cases can match the typical scale of Xavier without also changing centers or slope heterogeneity.

Recommended first physical readout: zero, including output bias. Recommended first coordinates: raw readout coefficients. These are proposed baselines, not yet selected run settings. Hidden geometry is frozen, so zero readout does not prevent the readout from learning.

Sweep lambda logarithmically from the measured Xavier-like range for the selected width through .25, with additional points beyond .25. Candidate larger values are .5, 1, and 2; final endpoints and density remain open. Retain the same centers across this sweep. Refine interesting intervals by adding independent lambda cases.

Use FP64 and a fixed finite training domain [-1,1], with one shared training grid and a separate dense evaluation grid. Report

$$E=\frac{\|f_{\mathrm{eval}}-y_{\mathrm{eval}}\|_2}{\|y_{\mathrm{eval}}\|_2},$$

where f_eval is the prediction vector and y_eval is the target vector on the evaluation grid. This is relative L2 error, not squared error or MSE. Store training error separately.

The existing grid convention is h=2/N, with W=N+1+2H neurons when there are H halo centers on each side. Show N, W, h, gamma, and lambda in metadata; do not call both N and W the width without clarification. Preserve the existing 24-per-side halo setting unless explicitly choosing the note's different convention. At N128, lambda=.25 means gamma16; at N512 it means gamma64.

Target library: sine, mixed sine, Runge, Gaussian envelope from the recent campaign, plus the note's quadratic and explicitly named note versions of sine/mixed sine. The note's mixed sine is not the earlier three-carrier target. Record equations and amplitudes; a viewer checkbox must identify a particular target definition. All targets use the same training interval in this campaign; including the Gaussian envelope does not introduce a whole-line objective.

## 2. Main browser view

| Control or display | Meaning |
|---|---|
| Horizontal axis | Lambda, logarithmic by default; show .25 as a reference marker |
| Vertical axis | Independent-grid relative L2 error, logarithmic by default |
| Function checklist | One subplot column per selected target |
| Main curves | GD and Adam under the currently selected conditions |
| Shared budget slider | Every point is evaluated at the same stored update count |
| Best-so-far button | Minimum recorded evaluation error up to the selected budget, separately for each run |
| Dotted reference | Numerical least-squares readout fit for that geometry, evaluated on the same evaluation grid |
| Condition selectors | Geometry family, width, readout coordinates, physical initialization, optimizer settings, seed/aggregation |
| Comparison selector | Overlay one additional factor at a time, or facet it into rows, without automatically mixing every ablation |
| Export | Download the displayed figure as PNG, with the selected budget and conditions identified; optionally export the plotted values |

Best-so-far and best-hyperparameters are different views. The first searches time within one trajectory; a later hyperparameter-envelope view searches a declared set of configurations at an equal budget. Neither should silently optimize over geometry families, widths, or seeds. Show the winning step/configuration when displaying an envelope. Best-so-far means best **recorded** evaluation, not an unobserved minimum between snapshots.

Steps and elapsed training time are separate budgets. Save elapsed training time and evaluation/checkpoint overhead separately so a time view can be added. A shared step count does not imply equal compute across widths or hardware.

Display running, completed, failed, and missing cases distinctly. At a requested step, a run that has not reached that step contributes no point; do not reuse its earlier value as if it had. Initially plot only common recorded steps. Keep axis limits consistent across function columns by default, with an explicit option to inspect a narrower range.

The least-squares reference is computed using the same physical feature matrix and solver policy across coordinate maps. Changing an invertible map must not create a different reference by changing numerical truncation conventions. Label it numerical: finite precision and rank cutoffs matter. Also, the reference minimizes training error; its independent-grid error is a reference curve, not a mathematical lower bound on every evaluated model.

## 3. Ablations, separated by what they change

### A. Readout coordinates and neighboring features

Write the physical readout as c=M theta: c contains actual output coefficients, theta contains the optimized coordinates, and M is a fixed invertible map. Raw coordinates use M=I. Changing M preserves the exact representable functions but changes optimizer dynamics.

The PR defines positive reference scales alpha_j for neurons, with separate halo and output-bias treatment. Keep these scales fixed at the reference lambda .25 while sweeping actual lambda.

| Map | What it tests |
|---|---|
| Raw coefficients | Baseline cumulative tanh features |
| Collective scaling: coefficient multiplier sqrt(alpha_j) | First characteristic-scale coordinate choice from the PR |
| Individual scaling: coefficient multiplier alpha_j | Stronger relative coordinate scaling |
| Collective neighboring | Neighbor differences with cumulative square-root scaling |
| Individual neighboring | Neighbor differences with cumulative full scaling |
| Unscaled neighboring, additional control | Isolate the neighbor basis change without the PR's extra coefficient scaling |

Neighboring uses adjacent tanh differences and retains the final tanh anchor and constant feature. The full transform is invertible. Removing the anchor would change capacity. Use the PR's actual cumulative-scale transform, not independent scaling of difference columns guessed from the name.

For every coordinate comparison, re-encode **the same physical initial readout** into each map. This holds the initial prediction fixed. Localization is most directly interpretable for ordered uniform centers and identical slopes; heterogeneous-slope versions are a later comparison.

### B. Physical readout initialization

Let sigma_X be the Xavier standard deviation and xi_j a shared standard Gaussian draw. Candidate physical starts are:

- Zero.
- Ordinary Xavier: sigma_X xi_j.
- Uniformly reduced Xavier: sqrt(h) sigma_X xi_j.
- Reference-scaled Xavier: sqrt(alpha_j) sigma_X xi_j.
- Alpha-Xavier: alpha_j sigma_X xi_j.
- Optional signed envelope: alpha_j sign(xi_j), a distinct non-Gaussian start.

Do not confuse the physical start with the coordinate map. The PR's individual-coordinate comparison reused the collective physical start. Independent alpha-Xavier starts are a separate ablation. For interior neurons, sqrt(alpha)-Xavier already has typical physical size O(h), because the Xavier draw also shrinks with width. An additional O(h) multiplier on Xavier is a different, smaller start.

Reuse random draws across lambda, optimizers, and maps. Multiple identical zero-start deterministic runs are not independent seed trials.

### C. Geometry family

1. Uniform centers, identical slopes: core.
2. Same centers, heterogeneous Xavier-shaped slope magnitudes, scaled to a declared mean lambda: isolates slope heterogeneity.
3. Fixed random centers, identical slopes: isolates center placement.
4. Full Xavier hidden weights and biases, scaled together: earlier-session comparison.

For full Xavier, multiplying both hidden weights and biases by the same positive factor preserves centers. The PR's fixed-center Xavier-slope initialization is not full Xavier geometry. Heterogeneous models have a distribution of lambda_j; use a declared horizontal-axis statistic and retain its distribution and maximum in metadata. Do not describe that statistic as the common slope or the theorem's maximum slope.

### D. Optimizer rate, epsilon, and schedule

- Recommended GD policy from the note: eta = kappa/||B||_2^2, where B is the training-normalized feature matrix in the chosen readout coordinates and kappa is a dimensionless rate multiplier. Candidate kappa=1 and .5. This gives each geometry a comparable stable rate relative to its largest curvature.
- A shared fixed numerical GD rate is a separate useful control. It answers a different question and must remain separately labeled.
- Adam: ordinary Adam, no weight decay; sweep a small declared set of learning rates and epsilons with equal tuning budgets across lambda. Include constant versus prescribed decaying schedules.
- Coordinate changes affect Adam too. A fixed native epsilon does not mean the same physical-gradient threshold under diagonal scaling. Include a matched-threshold diagonal control only if needed, rather than silently treating all epsilon settings as equivalent.

Learning-rate multipliers, physical-initialization multipliers, and readout coordinate scales are three distinct ablations. A global coordinate multiplier with curvature-normalized GD is largely redundant in exact arithmetic; relative neuron/halo/bias scaling is the useful coordinate question.

### E. Size and robustness

- Grid resolution: candidate N128, 256, 512, 1024. The note starts at 512; the recent session mostly used 128. Choose one first.
- Halo and bias: hold fixed initially; later compare selected halo/bias scaling rules while preserving the feature span. A change in halo count changes geometry and is a separate experiment.
- Width comparisons at fixed lambda scale gamma with 1/h. Add fixed-gamma width comparisons separately if desired.
- Paired seeds for random geometry/readouts; store each seed before choosing any aggregate display.
- Training/evaluation density: fixed initially, selected resolution checks later, especially for narrow features at large lambda.

## 4. Additive execution and storage

Use one campaign with explicit independent run configurations, not separate ad hoc scripts for every plot. Store the target definition, geometry, initialization, optimizer policy, precision, and dataset version with each run. Reuse each geometry's reference solve across optimizer runs.

Save inexpensive metrics frequently, dense early evaluations, and shared later milestones. The note's 10k, 50k, and 200k budgets are natural proposed milestones; exact horizon and snapshot cadence remain open. Save physical readouts for selected snapshots and full optimizer state for resumable checkpoints. Adam moments, scheduler state, and the update count are required for genuine continuation.

An extended run keeps its original schedule definition. A cosine schedule designed for 10k steps cannot be relabeled as a 200k cosine run after finishing; changed schedules become explicitly labeled continuations or new runs.

Keep storage simple: one campaign directory with compact data/checkpoints, a figures/export folder, and a running writeup. The viewer reads an index of available runs and updates as results arrive. Adding lambda points, targets, maps, seeds, or longer continuations must preserve earlier data. Do not generate all possible static plots automatically.

## 5. Suggested order and open choices

1. **Coverage:** one width, uniform identical slopes, raw coordinates, zero readout, broad lambda grid, GD and Adam, selected targets.
2. **Conditioning:** add PR maps with matched physical starts over the same lambda range, retaining the baseline.
3. **Initialization:** compare starts under selected coordinate maps; do not immediately run the full product of every factor.
4. **Generality:** add geometry families, widths, seeds, and selected halo/sample controls.

Use longer horizons for representative weak, intermediate, and strong cases, not only early winners; late curve crossings are part of the question. Any envelope comparing methods uses equal budgets and explicitly lists its candidate set.

Sam still selects the first width, exact targets, lambda grid, initial readout/maps, rate policies, seeds, horizon, and snapshot density. This document organizes those choices without launching them.

Deferred: polynomial projections and access bounds, target-tail certificates, training-time lower bounds, coefficient-budget capacity tests, and the one-step damped-GN theorem probe. These can later consume compatible saved geometry and residual snapshots; they are not part of this first campaign.

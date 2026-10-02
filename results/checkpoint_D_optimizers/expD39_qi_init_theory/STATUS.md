# D39 coordinator record

2026-09-29. Owner: current Codex task. User authorized careful theory audit and
iterative initializer experiments. D38 remains the historical control.

## Frozen first-stage design (before new training)

Two hidden layers of 512 tanh neurons, same D38 Adam recipes, paired readout
and batch RNG, same train/validation/test partitions. No test evaluation in
screening. First stage: Airfoil, Kin8nm, corrected SARCOS, Superconductivity;
seed 0; 10k steps. Added Superconductivity before training to test input
dimension above the 24-direction starting rank.

Eleven arms: legacy QI; exact-spacing-only correction; balanced 24 banks;
balanced plus centered projection bands; centered plus 25% collar; centered
plus one common spacing chosen to cover every band; centered lambda .5;
centered lambda 1; centered 64 directions with 8 centers each; centered banks
in only layer 1 or only layer 2. The first three
are a sequential correction ladder. All later arms change one factor from
the centered arm. Balanced banks have sizes 21/22 rather than 22 plus a tail
of 6. The common-gamma arm also uses common actual spacing; keeping variable
spacing while forcing gamma constant would contradict lambda=gamma*h.

Selection: geometric mean across the four tasks of minimum validation MSE
relative to legacy QI; identical checkpoint schedules. Lock a candidate
before its held-out evaluation. Confirm across all six tasks, 20k steps,
seeds 0/1/2. D38's completed 20k controls can be reused because the recipe
and data do not change. Also run the spacing-only correction on the four
screen tasks to isolate that bug from the selected candidate.

The 64-direction arm changes direction allocation and the second-layer RNG
offset; this is a family comparison, not a claim about one particular shared
direction set. The other corrected arms share the same unit directions.

Validation evidence is exploratory and adaptive. Three seeds on one fixed
split do not establish across-split superiority. Noise, direction coverage,
and learned coefficients preclude importing the 1-D machine-precision
theorem into these tabular runs.

## Progress

- Read original paper, Section 3 rewrite, implementation and current ridge/depth notes.
- Independent read-only theory and implementation reviews complete; no blocking training defect found. Qualified manuscript claims are recorded in the theory audit.
- Seventeen focused tests passed before the first stage; 18 pass with the second-iteration controls. All 44 first-stage and eight follow-up validation-only runs are complete.
- Canonical fp64 derivative-convolution construction reaches maximum error 4.8e-12 on sin(pi*x), consistent with the repository's documented fp64 regime. Separate 54-cell analytic geometry check complete.
- Run identities inherit D38 recipe fields. Actual initialization overrides are identified by the scheme and detailed layer diagnostics, with a resolved settings manifest alongside; do not interpret inherited qi_lambda/qi_layers fields as the override definition.
- Adaptive second iteration, specified before its eight validation-only runs: the approximate direction-count/normalized-slope factorial in `followup_plan.md`. It separates softer slopes from allocating more directions. Total screening: 52 runs, 13 arms across four tasks.
- Before any held-out confirmation: retain the global best new candidate even if it does not beat original QI, report that flag explicitly, and additionally confirm each screened task's validation winner (original QI requires no new run). This secondary task-specific analysis follows Sam's permission for per-task tuning. Reuse the same 20k recipes, run three seeds, and also confirm the isolated spacing fix on all four screened tasks. The selection and complete job list will be locked together.

## Confirmation selection record

All 52 validation-only screening runs completed. The globally selected new arm is sharp64 (64 directions, eight centers, lambda=5/7), with screening ratio0.9974 versus originalQI: effectively parity at one-seed resolution. Per-task winners: Airfoil originalQI; Kin8nm soft24; SARCOS last_only; Superconductivity spacing. Selection, score inputs and source hashes are frozen in selection.json. Thirty-six 20k-step runs cover the globalcandidate, task-specific winners and isolatedspacing controls, three seeds each. No further initializer selection will use their test scores.

## Completed analysis

All 88 new training runs are complete, including all 36 confirmation runs. The selection file and training sources remain unchanged. Confirmation matching and all 720 LS fitting-loss checks pass across the 36 new runs and 36 reused D38 controls. The initial/final train/validation readout audit covers every one of the 52 pilots; its checks reproduce saved trace errors.

At validation-selected checkpoints, soft24 reduces Kin8nm mean test MSE by 11.8% versus original QI and last_only reduces SARCOS by 6.7%; both win all three paired seeds. Their gains over standard are 14.7% and 2.5%, respectively; the SARCOS gain over standard holds in two of three pairs. The global candidate's geometric-mean ratio is .9959 versus original QI, so there is no clear universal improvement. The isolated spacing fix has small, mixed effects.

The completed writeup is expD39_results.md, with machine-readable confirmation_summary.json and task_policy_summary.json. The task-recipe overview retains original QI on unscreened Bike Sharing and Pol, explicitly as defaults rather than further discoveries. Plots mark values outside the common MSE limits. Theory/implementation and final reporting reviews are complete; the rank-cutoff, sampled-error and task-policy qualifications are included.

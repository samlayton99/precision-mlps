# Cross-function effective-gradient audit: evidence index

The [tutorial synthesis](../../../../docs/d34_cross_function_audit.md) explains
the two gain terms, residual forcing, forecast defects, and the missing
correction in transferring a frozen-model movement budget to ordinary GD.
This directory preserves evidence from existing trajectories: no training,
replay, or revised forecast was performed.

**Evidence glossary. All force ratios use Euclidean norms unless marked signed.**

| Artifact term | Meaning |
|---|---|
| `existing` | 13 original targets, seeds 0–4, three forks: 195 starts. |
| `fresh` | The same 13 targets, seeds 20–21: 78 starts. |
| `heldout` | Ten new function instances, seeds 22–23: 60 starts. This audit of them is retrospective. |
| `start`, `horizon` | Fork update (100k/400k/600k) and additional updates (0/1k/10k/50k/200k). |
| `balanced` | The contribution includes its minus sign: $-J_C^TBe_H$. |
| `force_energy_rate` | A contribution to $d(\|F_a\|^2/2)/dt$, not outward scale velocity. |
| `new_ever` | Distinct neurons first reaching a threshold after the fork; initial occupants excluded. |

## Curated records

- [Report metrics](summary_final/report_metrics.json): cohort/horizon summaries,
  per-function endpoints, exact exceedance counts and state identifiers.
- [Verification and summary manifest](summary_final/summary.json): input hashes,
  maximum identity errors, and aggregation definitions.
- Compressed [per-state force measurements](summary_final/forces_combined.csv.gz),
  [force summaries](summary_final/forces_summary.csv.gz), and
  [acquisition measurements](summary_final/budgets_combined.csv.gz). Compression
  preserves the original CSV bytes. Cohort, function, and family summaries are
  separately labeled; “original” is a bookkeeping group rather than a scientific
  function family. Repeated starts are not independent function draws.
- Completion/provenance records for [original](existing/complete.json),
  [fresh-seed](fresh/complete.json), and [new-function](heldout/complete.json)
  audits: source archive hashes, runtime, device, and verification.
- [Transfer manifest](provenance/evidence/transfer_manifest.json): original and
  transferred hashes, selected NPZ members, and their byte-level hashes. The
  forecast manifests preserve original issuance times.
- [Slurm accounting](provenance/accounting.psv), including failed attempts;
  [numerical test log](provenance/pilot2-1136.out) and
  [archive/forecast test log](provenance/tests2-1139.out): 23 focused tests pass.

The two figures, [acquisition allowances](summary_final/01_surrogate_and_actual_acquisition.png)
and [force defects](summary_final/02_endpoint_force_defects.png), are explained
alongside their derivations in the tutorial. The former compares surrogate
upper allowances with actual neuron counts, not certified bounds on GD. The
latter shows sampled endpoint force norms, not cumulative displacement errors.

## Reproduction and scope

Source: [`cross_function_audit.py`](../../../../experiments/expD34_readout_race/cross_function_audit.py),
[`cross_function_summary.py`](../../../../experiments/expD34_readout_race/cross_function_summary.py),
and the [launcher instructions](../../../../experiments/expD34_readout_race/README.md#cross-function-audit-of-movement-budgets-and-force-defects).
The audit implementation is commit `2143173`, based on completed campaign
revision `f69e1bb`. Per-cohort completion manifests hash the code present during
the numerical audit; final summary formatting was completed afterward in
`2143173` without changing numerical measurements.

All numerical work ran in Runpod Slurm CPU allocations, using JAX 0.11.1,
NumPy 2.5.1, FP64, and the CPU backend. Plotting used Matplotlib 3.11.2 from
an isolated package directory. Original learning rates, targets, states,
retained basis, thresholds, and forecasts were unchanged. Jobs 1135–1142 used
no GPU allocation. Job 1137 completed all 333 starts and 1665 states; jobs
1136 and 1139 contain the passing checks; job 1142 produced the final summaries
and figures. The legacy test attempted in job 1138 could not collect because
PyTorch was absent. Initial loader and plotting failures were corrected and
their logs retained.

The full tables and vectors were downloaded alongside these curated records:
each cohort has `forces.csv`, `budgets.csv`, `modes.csv`, and `vectors.npz`;
`summary_final` contains combined and summarized CSVs. These larger artifacts
remain outside Git. The remote copy is under
`/workspace/junmiaoh/experiments/precision-mlps/cross-function-audit-f69e1bb/full`.
Inputs are in its sibling `evidence` directory. The transferred input capsule
has SHA256 `5591bafde7ec9905917b66d116806e1456ed7a5316dafd7ebd0f0b6b2c26a9af`.

The source forecasts were issued before their continuations. The new spectral
allowances use only those starting checkpoints, but their additional
comparison and the force attribution were designed after campaign results
existed. Five sampled states do not establish a bound at every intervening
update. The analytical surrogate bounds are evaluated in FP64, not directed
rounding. No general function-distribution or Adam conclusion follows from
this GD audit.

# Readout-race evidence

The scientific interpretation is in [REPORT.md](REPORT.md). The [experiment README](../../../experiments/expD34_readout_race/README.md) defines the model and executable protocol. All hidden affine parameters and all readouts train independently with ordinary simultaneous GD.

**Evidence packages. Each package compares a single update horizon; continuations reuse the original trajectories.**

| Directory | Comparison |
|---|---|
| `core20k` | Three widths, seeds 0–4, five targets, seven ratios, 20k updates. |
| `replicate20k` | Width 177, seeds 5–9, otherwise the core protocol. |
| `refine40k` | Width 177, seed 0, both steps halved and updates doubled; same geometry time as core. |
| `continue100k` | Width 177, original seeds 0–4, continued unchanged to 100k. |
| `continue600k` | The same original width-177 trajectories at 600k, with sparse full-state analysis and complete scalar histories. |
| `refine200k` | Seed-0 half-step runs at the same geometry time as the original 100k comparison. |
| `provenance` | Original bundle manifests, initialization/data hashes, and Slurm allocation checks. |
| `verification` | Full-horizon PyTorch comparison, saved-state Hessian spectra, lossless-compression ledgers, and repository validation logs. |

`summary.json` and `summary.csv` contain all model endpoints, failures, window errors, scale counts, and event times. `rate_contrasts.csv` contains paired actual and reference contrasts against ratio one. The three audit tables summarize every analyzed reference, matched-target, and sample-probe row; they do not average away failing cases. The `last_` and `max_abs_` prefixes distinguish the final observed value from a descriptive maximum over analyzed states. Detailed sample-probe curves are retained for width 177, seed 0.

The compact NPZ files contain the exact floating-point columns needed by the plots, identified by `plot_trace_columns`; unretained columns are not zero-valued observations. The full raw trajectory archive was retired at the user's request on 2026-09-20 after verifying all curated numerical hashes. Each package's `evidence_manifest.json` records its original source analysis directories, analysis schedules, and hashes of the curated numerical files. Figures are generated separately from those files:

```bash
MPLCONFIGDIR=/tmp/race-mpl python -m experiments.expD34_readout_race.plot \
  --root results/checkpoint_D_optimizers/expD34_readout_race/core20k
```

Use the same command with `continue100k` or `continue600k` for the longer original-seed comparisons. Full source analysis tables remain under `/workspace/junmiaoh/experiments/precision-mlps/analysis/readout_race`. The former raw directory `runs/readout_race` has been deleted; its every-update traces, dense parameter/gradient snapshots, and resumable checkpoints are no longer available there. Bundle names identify reference budget and seed; `p0` denotes tanh and `p1/p3/p5/p7` denote independently trained activation degrees. No raw high-volume training archive is required to redraw the committed figures. The [retired metadata archive](provenance/retired_raw_metadata.tar.gz) preserves the original raw manifests and environment records.

The full width-89 reference table remains stored remotely as `reference_metrics.csv.gz`; curation reads this lossless form automatically. Historical byte-for-byte checks for the now-retired affine-reference NPZ archives remain in `verification/race-compression*.json`.

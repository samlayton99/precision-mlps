# Bimodal gamma experiment

Run from the repository root with its existing Python environment. D38's training engine and D39's centered reference remain unchanged. The experiment design and current status are in results/checkpoint_D_optimizers/expD40_bimodal_gamma/STATUS.md; the second mode range is recorded in followup_plan.md there.

Use four independent processes with shard indices 0 through 3. Each command below shows shard 0:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD40_bimodal_gamma.run screen --shards 4 --shard 0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD40_bimodal_gamma.followup screen --shards 4 --shard 0
```

After all 56 pilots finish, lock selection and plot the screen:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD40_bimodal_gamma.analyze select
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD40_bimodal_gamma.analyze screen
```

Use the **followup confirmation driver** for the combined campaign; it dispatches both original and mild arms with their exact saved configurations. The first-stage run.py driver predates mild arms.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD40_bimodal_gamma.followup confirm --shards 4 --shard 0
```

After all selected confirmations finish:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD40_bimodal_gamma.analyze confirm
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.expD40_bimodal_gamma.diagnostics final --ablate
```

Existing completed runs are reused only when identities match. Source changes are guarded by stored hashes. Screening evaluates train and validation only; confirmation records trained and observational LS readouts on all splits. The diagnostic ablation uses train/validation only and does not retrain or refit the readout.

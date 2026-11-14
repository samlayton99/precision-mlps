# Reconstruction of the appendix's native-GD bound

This experiment reconstructs the numerical regime reported in Appendix F.3, page 47, of the supplied submission PDF (referred to as Appendix G in the discussion). It is not an exact replication of the published 23-target, two-seed panel: the PDF does not specify its initialization, complete target list, or seed values.

## Settings and declared choices

- Width 705, including the full neuron budget; h=2/512 defines the construction reference. The trained centers are free rather than prescribed halo centers.
- Full-batch float64 native GD on half MSE, constant learning rate 0.002.
- 2,048 midpoint training points on [-1,1]; 8,192 separate midpoint evaluation points.
- Targets normalized by training-grid RMS, without mean subtraction.
- Analyze updates 25,000–125,000; the preceding 20,000–25,000 window supplies a concentration reference.
- Three declared targets: degree-five Legendre polynomial, exp(-40x²)sin(16πx), and x².
- A smaller initialization draws slopes, hidden biases, and readouts uniformly within ±sqrt(6/(705+1)); output bias is zero. Slopes and hidden biases use NumPy seed [30,512]; readouts use [30,512,24]. This is the historical C05/D24 draw convention used by the initial D60 reconstruction.
- A PyTorch-default control uses degree-five Legendre, the same width, target normalization, sampling, GD rate, and duration. PyTorch seed is 30. This controls the target and training configuration while changing the initialization family and draw.

The smaller initialization is a reconstruction choice, not something identified in the PDF. All three smaller-initialization targets share the same initial parameters. There is one seed, not a recreated two-seed panel.

## What counts as success

Compare the generic, affine-projected, and target-refined allowances in Appendix F.4–F.7. A continuation is useful only while both hold:

1. The calculated RMS scale ceiling remains below lambda=0.25.
2. The calculated relative training-error floor remains above 1%.

The summary records the first update at which either condition fails. The reported duration is consecutive useful updates from 25k; a zero duration means the criterion already fails at the starting checkpoint. Native training continues even after a bound loses usefulness.

`bound.py` transcribes the paper in physical B coordinates. `summarize.py` independently evolves the normalized variable q=B/sqrt(W), checks agreement of the force choices and crossing steps, and evaluates the combined scale/error criterion at every recurrence step. E0 is fixed at the maximum recorded nonaffine residual over the analyzed interval. Moment and tracking allowances are interpolated from saved checkpoints, matching the empirical scope of the paper's audit rather than providing directed-rounding enclosures.

The concentration diagnostic reports the accumulated mean sqrt(chi6) after 25k divided by its mean over 20k–25k. This is a declared interpretation of the paper's preceding-window diagnostic; the PDF does not provide enough implementation detail to identify it as an exact reproduction.

## Execution and provenance

The original run was stopped at 25k for the smaller-initialization cases and 20k for the default control when the user redirected work. `resume.py` continues those saved states without changing the original training or bound source files. It normalizes the original manifest's tuple/list representation during validation; the original entry point otherwise rejects its own JSON manifest on resume. Source hashes and configurations are verified before launching.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/expD60_feature_bound_reconstruction/resume.py
.venv/bin/python experiments/expD60_feature_bound_reconstruction/summarize.py
```

`summarize.py --partial` produces a labeled preview without replacing the final report. Results are under `results/checkpoint_D_optimizers/expD60_feature_bound_reconstruction/`. The final `reconstruction/` directory contains the scale/error comparison figure, PDF, detailed JSON, and per-case CSV. Historical stop metadata is retained, and `resumed.json`/`resume_manifest.json` record the continuation.

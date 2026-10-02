# Adam coefficient interpretation

Coordinator: current Codex thread, September 30, 2026.

User criterion: explain individual learned readouts or geometry-defined branches as measurements of target derivatives. Reconstructing a derivative by summing the original neurons does not satisfy the task.

Evidence sources: unchanged 2.3M-step expD06 collective-normalization Adam states, seeds 0 and 1. Those are distinct from the unavailable 320k image states. No new training.

Completed a bounded round of tests. Width-bin derivative families fail (best descriptive/oracle error about 90%); one compact latent scale-space field also fails held-out coefficient prediction. A geometry-only seven-neuron cofactor rule conditionally predicts one readout from six others, with errors 12.39% and 8.42% on 21 and 35 selected queries, versus 31.52% and 25.66% for center-only quadratic interpolation. Coverage is only 5.5% and 8.8% at that threshold; wider-coverage failures are saved.

An analytic local feature contrast permits large coefficient changes with tiny function changes. Actual Adam coefficients have very little of the most invisible contrasts, so this does not dismiss their observed structure as arbitrary. This is a partial coefficient regularity result, not a target-to-readout derivative interpretation.

Deliverables: findings.md with derivations and limitations, metrics.json with full model/threshold sweeps and provenance, three PNG/PDF figures, and reproducible scripts. The broader research question remains unresolved. No source state was changed and no new training was run.

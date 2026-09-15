# expD04 archive -- uncorrected figures (Catherine, 2026-07-22 to 07-30)

**Do not cite these.** They were superseded on 2026-09-08 by `../varpro_corrected/`. Three defects:

1. Every floor-level line starts from the exact QI geometry with a least-squares readout, i.e. the construction, which is at 1e-14 before any optimizer step. The plots show the initialization, not the optimizer.
2. The plotted number was the minimum eval error along the trajectory, seeded with the eval error at initialization, so it could never be worse than the start and hid divergences.
3. The LBFGS/SSBroyden arms solved the readout through a normal-equations fallback (capped near 1e-8, the fake "sqrt(eps) wall"), and the Kaufman projector used the full unpivoted-QR column space while the head was solved on a truncated subspace.

The CSVs behind these figures were never committed and are not recoverable.

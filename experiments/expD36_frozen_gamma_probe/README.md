# Frozen-gamma theorem probe

This experiment measures whether the necessary-polynomial-tail theorem explains
readout learning times. The first matrix contains two targets, four frozen
slopes, and raw/collective coordinates at core resolution $N=512$ and hidden
width $W=559$. The configuration is frozen before execution. Zero readout
initialization makes the target the initial correction in every cell.

The sine mixture is $\sin(2\pi x)+0.5\sin(6\pi x)+0.25\sin(10\pi x)$; the
control is $\sqrt{5}x^2$. These are not D06's differently defined `mixed` target.
D06 supplies the corrected-halo geometry and reference allowances, held fixed
as gamma varies. All training uses half empirical MSE in FP64.

## Evidence roles

The endpoint grid supplies training and theorem measurements. The offset
validation grid selects between the two declared Adam learning rates. The
independent midpoint evaluation grid measures fixed-budget precision and never
selects a recipe. All features are frozen; only readouts are trainable.

Dense QR, SVD, refits, and high-precision calculations are offline diagnostics.
They do not enter optimizer updates or claim scalable-optimizer performance.
The older optimizer-program promotion gates do not govern this theorem audit.

## Diagnostic conventions

Unpivoted Householder transforms retain every sample-space complement row.
An independent discrete orthogonal-polynomial recurrence checks the projection.
The code saves directional access, sampled subspace access, Frobenius norms,
and analytic bounds separately. SVD cutoff floors describe retained numerical
models, not mathematical rank. Predicted large horizons remain extrapolations.

FP64 access estimates carry a heuristic roundoff monitor, not a rigorous error
enclosure. High-precision agreement is reported separately and does not create
an interval certificate. The nominal real tanh dictionary and exact arithmetic
on a stored FP64 matrix are distinct numerical objects.

The quadratic has exactly zero polynomial tail above degree two. Its measured
roundoff tail is retained in artifacts but cannot generate a time certificate.
This control need not have gamma-independent optimization dynamics.

## Execution contract

The authorized limit is two allocated H200 GPU-hours, at most two GPUs at once,
and a CPU diagnostic job capped at 60 minutes. All remote computation uses
Slurm. Initial GPU allocations request at most 55 minutes each, with the
remaining ten GPU-minutes reserved for necessary checks/retries. Compilation,
idle allocated time, and failures count. No automatic scope or budget expansion.

Implementation checks cover projection/complement identities, target definitions,
access bounds, the single-mode discrete certificate, mixed-mode slack, gradients,
coordinate maps, and actual GD versus spectral evolution. Scientific success
does not require tight bounds or a favorable gamma intervention.

The accompanying results report will distinguish trained precision, attainable
tolerances, theorem slack, arithmetic limitations, and unexecuted horizons.

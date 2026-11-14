# NPU fairness audit — November 12, 2026

The original comparison is a valid measurement of the stated 50k-step Adam
protocol, but it does not establish the best accuracy of the NPU architecture.
The user subsequently requested an updated complex-NPU Adam-to-L-BFGS comparison.
Those publication inputs are separate in `../npu_adam_lbfgs/`; the readout-fitting
and RealNPU diagnostics here are not included in those curves.

## Implementation and protocol

- The Torch gated complex forward formula matches the authors' released
  `submission/src/npu.jl`: real/imaginary powers, gated radius and phase,
  machine epsilon, and forward-clipped input gates. Glorot real weights,
  zero imaginary weights and gates initialized at 0.5 match the reference.
- Gates outside [0,1] have zero clamp gradient. This is also true of the
  reference implementation; it is not an accidental extra restriction here.
- The reference NAU is an unconstrained linear map without bias. Our added
  head bias increases flexibility; composed and positive-input candidates
  are disclosed extensions, not exact reproductions of the paper's architecture.
- The original NPU pilots covered three variants, two rates (.001 and .01),
  six widths and seed 0: 36 candidates per primary target. A 10k validation
  winner was continued at each width. This may miss slow-starting configurations.
- All compared methods use the same function-value training grid at each width.
  No derivative or exterior target labels are supplied. Float64 and physical-dx
  relative L2 are shared. NPU quadrature explicitly resolves input zero crossings.
- Width matching is not parameter-count matching. The composed model has more
  parameters at a given width. The selected plotted models may use different widths.
- The original baseline has no L-BFGS or fitted readout; QUILL uses least squares
  with numerical solver/cutoff tuning. Therefore the table measures two fitting
  procedures, rather than isolating representational capacity.

## Bounded sensitivity checks

1. All eight selected trained models received up to 200 full-batch L-BFGS
   iterations and independent readout-solver diagnostics. Both MLP and NPU
   received the same optional solver opportunity. Candidate selection uses
   validation; independent higher-order integration measures the selected model.
2. All 48 original 50k models received pivoted-QR readout fits at cutoffs 1e-14
   and machine epsilon, retaining the original model as a candidate.
   Validation-selected NPU errors after this diagnostic are approximately
   0.675 (Airy), 1.89e-5 (Kepler), 6.03e-5 (elliptic K), and 0.103 (quotient).
   Some fitted readouts have enormous coefficient norms and fail the very strict
   integration-agreement tolerance. Their raw variation is retained in JSON;
   these are approximate diagnostic results, not certified high precision.
3. Sixteen fresh runs compare complex NPU and RealNPU at width 105, learning
   rate .005, and seeds 0/1 across all four targets, with 50k Adam steps and
   the original data/schedule. The .005 rate is from the released simple-task
   script. Best complex-NPU Kepler error is 4.00e-4, versus 1.37e-3 in the old
   selected comparison. RealNPU did not improve on complex NPU in this bounded
   check. This is not an exhaustive RealNPU study.

The paper's simple arithmetic experiment uses unregularized Adam; omitting L1
here is not in itself a reference-implementation error. Its equation-discovery
experiment does use Adam followed by L-BFGS, motivating that additional optimizer
check. Neither experiment is the same task as fitting our special functions.

## Verification and interpretation

All 18 existing implementation/regression tests passed. Original checkpoints and
figures were not changed by the audit. Source hashes, candidates, saved refined
models and integration comparisons accompany the diagnostics.

The important improvement is procedural: report the actual favorable optimizer
protocol, preserve reference-compatible options, select using validation, and
avoid claims that NPU can never achieve high precision. A final approximation
study could also report readout-polished methods explicitly, rather than silently
calling those models Adam-to-L-BFGS.

Primary references:
- https://proceedings.neurips.cc/paper_files/paper/2020/file/48e59000d7dfcf6c1d96ce4a603ed738-Paper.pdf
- https://github.com/nmheim/NeuralPowerUnits/blob/submission/src/npu.jl
- https://github.com/nmheim/NeuralPowerUnits/blob/submission/src/nau.jl
- https://github.com/nmheim/NeuralPowerUnits/blob/submission/scripts/simple/run.jl

The latter two source files were read from the previously saved reference copy
under expK03, because their web fetches failed. No inverse-square-root results
were used in this audit.

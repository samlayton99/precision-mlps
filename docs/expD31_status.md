# expD31 — Separate Adam streams for approximation and readout-gap gradients

Status: complete; coordinator main task. All 48 runs finish 500 updates. Seven new Adam checks and five existing profile-gradient checks pass. Four figures, independent-grid/cutoff checks, and the experiment writeup are ready for Sam. No training was repeated to repair plot formatting.

Sam requested Adam on both DF and DG, with mu outside normalization, at mu=1000,10000,100000. Geometry is the existing vector of raw slopes and biases. Maintain independent Adam first/second moments for DF_tau and DG_tau, then take theta <- theta - eta(mu AdamDirection(DF_tau) + AdamDirection(DG_tau)). Use ordinary Adam for the trained readout. Add ordinary Adam as a control and reuse the matched weighted-GD trajectories. Same four functions, three initializations, seed 0, 500 updates, eta=0.002, samples and SVD cutoff as expD29. No clipping, rate retuning, gradient gating, or coefficient replacement. Adam beta=(0.9,0.999), epsilon=1e-8.

Requirements section-8 checklist, before implementation:

1. One ordinary forward/backward and one dense feature SVD per split update, plus sparse offline sensitivity checks. SVD costs O(n m^2 + m^3); this fails the gradient-class production budget.
2. Two geometry Adam streams and one readout Adam stream use O(m) persistent memory. The diagnostic SVD needs O(n m + m^2) transient arrays; saved scientific states are not optimizer state.
3. No Krylov memory; dense SVD is the bottleneck.
4. No iterative Krylov/collective mechanism or constant-reduction scalability claim.
5. Tiny DF can be unresolved numerically. Epsilon attenuates very small components, but mu outside Adam can amplify numerical sensitivity. Record alternative-driver/backend gradient and proposed-step changes; do not silently alter the update.
6. Fixed fp64 and the inherited cutoff 1e-13 times the leading singular value. This is not precision-agnostic.
7. No feedback controller or loss-based step acceptance. Stop/report nonfinite states or failed required SVDs.
8. This is an explicitly requested diagnostic, not promotion of the previously rejected normalized-tiny-gradient production strategy.
9. Compare ordinary Adam and the previous weighted GD at matched mu. Split Adam at mu=1 would not equal ordinary Adam, so label the control accurately.
10. Already fails architecture-blind/scaling gates; no production-readiness claim.
11. Falsification: motion alone is not success. Plot actual L, refitted F_tau, and mean absolute slope; audit independent-grid refits, coefficient norms, rank, and numerical sensitivity. Outside multipliers at this eta permit initial coordinate steps near 2,20,200 when gradients exceed epsilon.

Implementation validation: compare each Adam stream to independent PyTorch Adam over signed, zero, and tiny gradients; verify outside multiplication and independent moment histories; compare the combined step to independent Adam optimizers acting on the same pre-step state; verify ordinary-Adam trajectory and readout update without installing solved coefficients. Reuse the existing independently tested retained-projector derivative.

Completed findings: Xavier/mu=1000 improves actual fitting across all four targets and final refitted approximation on mixed sine, Runge, and Gaussian. Runge refitted independent relative L2 is 2.35e-6 (survives three cutoffs), with mean gamma 10.33. All four Xavier/mu=1000 trajectories visit excellent early refitted geometries; several then lose them while the moments retain substantial motion. Larger weights often damage geometry, and Xavier/Runge at mu=100000 has independent refit error about 413 despite sampled relative error about 0.1. The latter persists on 65,536 evaluation samples. Initial Xavier directions have material sensitivity to SVD/backend choices; no stable-recipe claim. Final outputs are organized in figures/ and data/ under expD31_split_adam, with one expD31_results.md writeup; the session catalogue and global result index have been updated.

Smaller-Xavier follow-up complete: 16 runs at mu=100,500,1000,2000 with the unchanged trainer/rate. Saved all 501 states and reconstructed every refitted model on 8,192 independent points; best states also checked at three cutoffs and on 65,536 points. Four repeated mu=1000 loss/F/gamma histories match bitwise. Added 500-step and first-30-step figures with relative-L2 middle rows and linear gamma axes. No machine-precision geometry recovered; the writeup records best and final errors.

10,000-step VarPro extension complete. Xavier, four targets, mu=50,100,250,500, two versions only: constant eta=.002 versus cosine decay from the start to eta=.000002 over all 10,000 steps. The same scalar rate applies to both geometry streams and the ordinary-Adam readout; mu stays outside normalization and moments never reset. No first-500-step warm phase. All 32 split runs plus four new scheduled ordinary-Adam controls finish; four completed constant controls are reused. Outputs record explicit refitted relative error every split step, independent-grid errors at saved states, and dense-grid/cutoff checks of best saved and final states. Two new schedule checks and thirteen inherited implementation checks pass. The original dense-SVD diagnostic practicality assessment remains unchanged. Latest Sam steering is to deliver clear runs and plots without interpreting them; this extension adds the two requested figures and protocol documentation only.

Saved-data balance figures added for both long-run schedules: scalar F/G above and recorded post-Adam, post-multiplier F/G geometry-update norm ratio below. Shared logarithmic vertical limits and linear step axes. All 32 operands were checked for positivity/finiteness and the loss identity; no training repeated.

Dynamic post-Adam ratio extension authorized: r=0.01,0.1,1,10,100; same four Xavier targets, 10,000 updates, constant and cosine common rates. At each pre-update state choose mu_t=r||u_G||/||u_F|| after independent Adam normalization. No readout replacement or moment reset. Record effective mu, achieved ratio, both stream norms, existing refit metrics, and numerical sensitivity.

Pre-build checklist for this diagnostic:
1. Same forward/backward and dense feature SVD per step; the new balance calculation adds vector norms/scaling, O(P). Sparse alternative-backend audits remain offline diagnostics.
2. Two geometry Adam streams and one readout stream: O(P) persistent optimizer state, with the inherited transient dense SVD arrays. No new dataset-sized optimizer state.
3. No Krylov/history rank parameter; inherited SVD cost still prevents production scalability.
4. Two extra vector-norm reductions per balanced update, with no iterative collective loop.
5. Exact-zero/nonfinite direction norms make the requested positive ratio undefined and stop the run with a recorded reason; no denominator floor silently changes r. Small nonzero directions remain eligible under the requested rule.
6. Float64, inherited Adam epsilon and SVD cutoff unchanged; no precision-agnostic claim.
7. The controller uses direction norms, not loss differences. It can amplify numerically uncertain directions: alternative-SVD/backend audits measure this, without silently imposing a confidence gate that would change the requested ratio.
8. This directly probes the known risk of normalizing signals below their numerical resolution. It is an explicitly requested diagnostic, not promotion of that mechanism as a stable recipe.
9. Existing fixed-mu and ordinary-Adam trajectories are the controls; no additional controller, smoothing, clipping, or retuning is introduced.
10. No production/litmus-test claim; those gates remain unmet by per-step dense SVD.
11. Verify achieved post-Adam ratios at every applied step and compare a multi-step trainer run against independently differentiated objectives and independent Adam histories. Plot actual loss, refitted relative L2, and readable mean gamma. Record failed runs rather than hiding them.

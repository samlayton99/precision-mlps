# A conditional scale bound that remains informative for default initialization

**Later assessment:** The radial-gain premise does not justify a claim of essentially permanent scale trapping. It can fail as gradients vanish even when slopes remain bounded. The empirical checkpoint results remain correct, but this is not a successful long-time repair. See the accumulated-motion analysis in docs/review_process/iclr_2027.

The common slope-radial threshold is K=1000.272842, twice the largest observed value on the default-initialization 20k–30k prefixes. It is frozen before evaluating the later window and the 2x/4x cases. This is exploratory temporal validation; the candidate condition was developed after inspecting some later states.

| Initialization | Radial condition passes | Largest observed radial gain | Largest conditional RMS ceiling among passing cases |
|---|---:|---:|---:|
| 1x | 48/48 | 924.76 | 4.554 |
| 2x | 47/48 | 1389.41 | 4.661 |
| 4x | 48/48 | 848.55 | 5.065 |

All 144 cases pass the independent sufficient loss-descent check over their analyzed finite horizon (48 default cases use the previously evaluated initialization-to-200k check; the 96 added cases use a fresh 30k-to-200k starting-state check). These are floating-point evaluations of analytic inequalities, not interval certificates. The new radial premise is supported only at saved checkpoints every 500 updates.

The single added-initialization radial failure is Airy, 2x initialization, LR 0.001: observed radial gain reaches 1389.41, above the unchanged threshold 1000.27. Its measured slopes still lie below the candidate ceiling, but that does not make its failed premise pass. No bound is claimed for that run under the fixed threshold.

The full-parameter version using a separately calibrated threshold passed only 44/48 primary runs. The successful primary statement specifically controls the slope block and RMS slope; it is not a bound on every parameter or the maximum individual slope.

For the primary 48 cases, the worst constant RMS ceiling is about 4.56, below the QUILL reference 18.375. No step-count term appears in the ceiling. It remains a bound for arbitrarily long continuations only if its premises persist. The saved data stop at 200k and do not prove that persistence; some radial gains are still rising near the endpoint.

The smallest primary threshold that would make the conservative ceiling touch the construction reference is approximately 16528. This indicates headroom in the implication, not validation that any chosen threshold holds forever. The QUILL reference is sufficient for the construction, not necessary for every accurate fit.

No new training was run. All 50,064 evaluated states come from saved trajectories. Autograd independently checks the diagnostic on four varied states; 10,000 randomized exact-descent cases check the discrete algebraic coefficients. The proof is elementary but has not received independent mathematical review.

The remaining mathematical task is to derive or weaken the radial-gain premise from structural properties of the tanh dynamics. The present contribution is a useful conditional implication plus a directly falsifiable diagnostic, not an unconditional explanation of small scales.

[Proof](../../../docs/review_process/iclr_2027/radial_loss_budget_bound.md) · [Figure](radial_loss_budget.png) · [Primary results](results.json) · [Extra initialization results](extra_initializations.json)

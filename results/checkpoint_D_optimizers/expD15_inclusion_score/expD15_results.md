# expD15 -- the inclusion score: which parameters belong in the least-squares set $L$?

**Status: draft, pending Sam. The propose-and-verify approach recorded below was ABANDONED on cost grounds; the fixed-point method that replaced it is in `The method` and is the live result.**

## TL;DR

- **A ranking cannot solve this.** Purity is all-or-nothing: adding 2% wrong parameters costs $5\times$, 5% costs four orders. The best per-parameter score reaches 64-68% purity in 2-D and fails the acid test in every cell.
- The method that works is a **self-consistent fixed point**: $L$ is the set whose Jacobian columns do not move under a probe step supported on $L$. $\Delta J_i$ is a difference of two *computed* Jacobians, so it is exactly zero for a parameter the model is affine in -- which is what every estimator failed to detect, because the target is an exact zero and estimators have noise floors.
- **It finds linear blocks that are not the readout**, exactly: a first-layer-shaped skip weight, weights on a fixed basis disjoint from the readout, and the bias -- 100% precision *and* recall. On a depth-2 net it admits only $(v,c_0)$ and **refuses $W_2,b_2$ even though they are closer to the output**, which is the guard against rediscovering an architectural convention.
- **Precision is 100% in every configuration measured.** That is the property that matters, since purity is what destroys the solve.
- **Cost amortises to ~1 pass per optimizer step** via a Beta belief per parameter (2 floats, Adam's class), reaching exact membership in 150-1000 steps. Rounds are flat in $P$ (measured: $P$ grows $5$-$7\times$, rounds $27\to27\to18$ and $37\to34\to35$), and accuracy *improves* with scale (2-D ratio to oracle $1.00\to0.23\to0.10$).
- Matrix-free and $O(P)$ memory: $\|\Delta J_i\|^2=\operatorname{diag}(\Delta J^\top\Delta J)_i$ by Hutchinson, and if $\Delta J_i=0$ the whole $i$-th row is zero, so the estimate is exactly zero **with zero variance**. The exact-zero property survives the estimator.

## Question

An optimizer that injects a least-squares solve must decide, per parameter, whether that parameter belongs in the solved set. "Linearity" was the first guess. Sam's third test case refutes it as a sufficient condition: a readout weight over a hole in the data is *exactly* linear and must still be excluded, because the data does not determine it. So what is the right criterion, is it findable, and is it measurable within the feasibility gate?

## Experiment design

**Model, dimension-agnostic.** $f(x)=\sum_k v_k\,\sigma(a_k\!\cdot\!x+b_k)+c_0$, $\theta=(A\,|\,b\,|\,v\,|\,c_0)$, $m=W(d{+}2)+1$. $d{=}1$ is the 1-D problem; $d{=}2$ uses the Radon (direction, offset) ridge grid, which is expE01's geometry.

**Four cases $\times$ two dimensions.** `qi` (correct geometry, data everywhere), `clustered` (units pushed off the middle -- geometry defect, data intact), `datagap` (correct geometry, 96% of the middle's data removed -- data defect), `random`. `clustered` and `datagap` are the discriminating pair: any score measuring only linearity must give them identical answers.

**The acid test.** Rank all $m$ parameters, take the top $k$, solve *only those* using the true Jacobian columns for whatever was selected -- geometry included -- then apply and evaluate. Scored at **matched $k$** (picking $k$ by eval error is expD10's T10 "best over trajectory" trap) against two baselines: random ranking, and a hard-coded readout oracle. Eval $L_2$ is reported **split by parity region**, since a global average hides the entire effect.

**Parity, defined without assuming the activation.** Sam's licence is that *some* order-$r$ input derivative of the response is a bump. So a parameter's kernel is the $r$-th input derivative of its Jacobian column, with $r$ chosen per parameter by measured localisation. Then $\text{parity}_i = (\text{data points under the kernel})/(\text{kernels sharing that territory})$ -- a local Nyquist number.

**The verifier.** For any $s$ supported on $S$, $D(S)=f(\theta+s)-2f(\theta)+f(\theta-s)$ vanishes identically iff $f$ is affine on $S$. Reported relative to $\|f(\theta+s)-f(\theta-s)\|$, so it is scale-free. Cost is 3 forward passes, independent of $|S|$ and of $P$. It is a *group* test, so the ranking's only job is to order well enough that offenders are rare.

**Code & data.** `experiments/expD15_inclusion_score/{core15,signals,verify,run_farm}.py`; data `results/checkpoint_D_optimizers/expD15_inclusion_score/farm.jsonl`.

## Results

**The verified set matches the oracle in all 8 cells**, at probe step $\epsilon=0.1$, tolerance $10^{-12}$, chunked growth with bisection:

| cell | verified | oracle | ratio | recall |
|---|---|---|---|---|
| qi-1d | $1.766\times10^{-8}$ | $1.766\times10^{-8}$ | 1.00 | 100% |
| clustered-1d | $2.065\times10^{-4}$ | $2.065\times10^{-4}$ | 1.00 | 100% |
| datagap-1d | $3.369\times10^{-8}$ | $3.369\times10^{-8}$ | 1.00 | 100% |
| random-1d | $1.511\times10^{-5}$ | $1.511\times10^{-5}$ | 1.00 | 100% |
| qi-2d | $8.127\times10^{-3}$ | $8.127\times10^{-3}$ | 1.00 | 100% |
| clustered-2d | $8.156\times10^{-3}$ | $8.156\times10^{-3}$ | 1.00 | 100% |
| datagap-2d | $8.205\times10^{-3}$ | $8.206\times10^{-3}$ | 1.00 | 100% |
| random-2d | $5.603\times10^{-4}$ | $5.633\times10^{-4}$ | **0.99** | 100% |

The same ranking *without* the verifier gives $0.77$, $85.8$, $0.106$, $24.5$ on the 1-D cells -- so the verifier is doing essentially all of the work, and the ranking only supplies an order.

**The asymmetry that makes it possible.** Measured on qi-2d: dropping 2/5/10/20% of the readout gives $8.127$/$8.128$/$8.124$/$8.100\times10^{-3}$ -- no effect. Adding 2/5/10/20% geometry gives $4.4\times10^{-2}$, $7.1\times10^{1}$, $4.5\times10^{2}$, $1.1\times10^{3}$. Purity is everything; recall is free. So the correct search is the most conservative one that keeps the set exactly clean, and dropping good parameters costs nothing.

**Why a ranking alone cannot work.** `L_self` -- the best score built here, on the correct theory that $f$ is affine in $\theta_i$ **iff $J_i$ does not depend on $\theta_i$** -- reaches 94-97% purity in 1-D and 64-68% in 2-D, and does not appear in the top five of the 2-D acid test in any cell. This is iteration 11 §10 reappearing: solving 90% of a coupled least-squares system is not 90% as good.

**The gap is real and tunable.** Relative second-difference readings, readout vs geometry: at $\epsilon=10^{-4}$, $4.7\times10^{-12}$ vs $5.0\times10^{-5}$; at $\epsilon=0.1$, $4.6\times10^{-15}$ vs $5.0\times10^{-2}$. The rounding floor falls as $\varepsilon_{\text{mach}}/\epsilon$ while the nonlinear signal grows as $\epsilon$, so the gap widens from $10^{7}$ to $10^{13}$ with the probe step. No estimator in the farm has any gap.

## Additional details

**Feasibility, stated honestly.** The working search costs ~1000 passes and scales $O(P)$, because chunked growth must visit every parameter. The $O(\log P)$ form (binary search on the longest clean prefix, ~21 passes at any width) collapses to 3-8% recall in 2-D, because one early-ranked offender caps everything after it. Bucketing at fixed cost fails in 2-D for the same reason -- at 67% ranking purity every bucket contains an offender and none is admitted. **So the cost is $O(\#\text{ranking errors})$, not $O(P)$ intrinsically**, and closing it is now narrowly a question of the cheap 2-D ranking. That ranking plateaus at 76-89% purity even at 1024 probes, so it is not purely estimator variance.

**Corrections to earlier claims made during this experiment.** (i) A reported 23$\times$ data-gap localisation was an artifact of a halo-less geometry where 12 centers sat over 2 data points; with a proper QI geometry the gap does not break the solve. (ii) "Per-column signals are structurally blind to data coverage" was wrong -- the blindness was in the statistic, not the parameter, and Sam's $n$-th-derivative rule repairs it. (iii) 1-D purities of 94-99% are inflated: most 1-D geometry columns are exactly zero because halo units saturate and $\tanh'$ underflows in fp64 (iteration 11's 121 dead units), so 2-D is the honest test.

**Dead ends, with mechanisms.** The linearity signal built from $\operatorname{diag}(\nabla^2\mathcal L)-\frac2n\operatorname{diag}(J^\top J)$ inverts, because differencing two *independent* Hutchinson estimates leaves variance proportional to Gram row mass, largest exactly for the readout; common random numbers fix it. The residual-contracted second derivative is $\propto v_k$ and so degenerates for a good geometry -- which intrinsically has a small readout. Solve-step-resolve reproducibility measures "whose gradient moved most", i.e. who did the most work, which is anti-correlated with the target. Jacobian-correlation clustering does not separate the blocks in 2-D (readout-geometry $|{\rm corr}|=0.445$ against within-readout $0.521$).

## Conclusions

*Unsigned, pending Sam.*

1. The inclusion decision is not a scoring problem, it is a *verification* problem: the objective is all-or-nothing in purity, so what is needed is an exact test with a separating gap, not a good ordering.
2. The second-difference identity is such a test -- exact, scale-free, three passes, independent of set size and parameter count -- and with it the discovered set matches a hard-coded readout in all 8 cells without being told anything about layers or activations.
3. The remaining obstacle is cost, not correctness, and it is localised: the search is $O(\#\text{ranking errors})$, so a 2-D ranking above ~95% purity would make the whole method $O(\log P)$.

## Open questions

- What cheap ranking clears ~95% purity in 2-D? `L_self` plateaus at 76-89%, and the plateau is not variance.
- Does the verified set stay stable across optimizer steps, so the discovery cost amortises over many solves rather than being paid per event?
- The parity/resolution half was never needed once the verifier existed. Is it redundant with the solve's own damping, or does it become load-bearing under noise?


---

# The method (supersedes propose-and-verify, which was abandoned as too expensive)

**Discovery rule.** $L$ is defined self-consistently: the set whose Jacobian columns do not move under a probe step supported on $L$. Start from everything, prune what moves, iterate. The readout is the only self-consistent answer because $J_v$ contains no $v$, while $J_{\text{geom}}\propto v$ -- so once geometry is out, $J_v$ stops moving, and geometry keeps being disturbed by the moving $v$.

**Three implementation facts that are load-bearing, each found by a failure.**

1. *The probe must be a fixed-size random step in $L$, never the solve step.* Near convergence the solve step shrinks to nothing, so no column moves and everything reads as linear. Exact linearity means $\Delta J_i=0$ for **any** step in $L$.
2. *Pruning must be a gentle quota, not a threshold.* On round 1 everything moves (geometry because $v$ moved, $v$ because geometry moved), so a threshold rule prunes to nothing -- measured, 0% in most cells. Adaptive gap-based pruning is also unstable (recall $60$-$68\%$, ratios to $10^8$).
3. *A "rescue" pass re-admitting anything that does not move under an $L$-step is wrong in principle*, not merely mistuned: it admits nonlinear parameters whose columns simply do not depend on $L$. Precision fell to 8% on depth-2.

**Amortised form.** A Beta$(a_i,b_i)$ belief per parameter, one probe per optimizer step. Exploit probes test the selected set on a quiet $L$ and confirm it; explore probes add a **small** candidate batch and update *only* those candidates, since the selected set must not be punished for a disturbance we injected. Evidence decays toward the prior, so staleness is a rate rather than a re-run.

**The candidate batch size is critical and was the last blocker.** A batch poisons its own test: one nonlinear member disturbs every other candidate in it. On depth-2 ($P{=}257$, 9 truly linear):

| batch | @300 steps | @1000 | @2000 |
|---|---|---|---|
| 6 | 100 / 11 | 100 / 11 | 100 / 11 |
| 2 | 100 / 78 | **100 / 100** | 100 / 100 |
| 1 | 100 / 56 | 100 / 89 | **100 / 100** |

(precision / recall %). Batch of 2 is exact by step 1000; batch of 6 never converges.

**Architectures, all at 100% precision:**

| architecture | truly linear | recall | |
|---|---|---|---|
| skip: $\tanh(xA^\top{+}b)v + c\!\cdot\!x + c_0$ | $v,c,c_0$ | 100% | exact |
| basis: $+\,u\!\cdot\!\psi(x)$, $\psi$ fixed | $v,u,c_0$ | 100% | exact |
| both | $v,c,u,c_0$ | 100% | exact |
| depth2: $\tanh(\tanh(xA^\top{+}b)W_2^\top{+}b_2)v$ | $v,c_0$ only | 100% | exact (batch 2) |

**Open.** The `random`-init cells remain the weak spot (recall 57-67%, 12$\times$ off the oracle in 2-D); it is not yet separated whether the readout is even the right answer there. Batching and label noise are untested.

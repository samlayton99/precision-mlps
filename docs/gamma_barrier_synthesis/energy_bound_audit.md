# Energy-based audit of a raw-parameter drift bound

This is a mathematical deduction for the current synthesis, not a new experiment or a novelty claim. The bound concerns reaching a specified slope regime. It does not establish that the regime is necessary for accurate approximation.

## Gradient flow

Let `p` contain all trainable raw parameters, including the slope vector `a`, and let the differentiable training objective satisfy `L >= 0`. Under ordinary unit-rate Euclidean gradient flow,

\[
\dot p=-\nabla L(p),\qquad
L(p(0))-L(p(T))=\int_0^T\|\dot p(t)\|_2^2\,dt.
\]

Cauchy–Schwarz therefore gives

\[
\|a(T)-a(0)\|_2^2
\le T\int_0^T\|\dot a(t)\|_2^2\,dt
\le T\,[L(0)-L(T)]\le TL(0).
\]

These inequalities also hold for the path length of the slope vector in place of its endpoint distance: `integral ||adot|| <= sqrt(T L(0))`. Frozen parameter blocks can be omitted from `p`. They do not invalidate the argument for the remaining Euclidean flow.

If a target set of slopes is at Euclidean distance at least `D` from the initialization, entering that set requires

\[
T\ge D^2/L(0).
\]

All statements concern finite intervals on which the solution and energy identity are defined.

## Discrete simultaneous GD

Suppose

\[
p_{k+1}=p_k-\eta_k\nabla L(p_k),\qquad \eta_k>0,
\]

and every step has sufficient descent

\[
L_k-L_{k+1}\ge\alpha\eta_k\|\nabla L(p_k)\|_2^2
\quad\text{for a common }\alpha>0.
\]

For example, a Lipschitz-gradient bound `beta` along every update segment and `eta_k <= 1/beta` imply `alpha = 1/2`. A globally bounded Hessian is not automatic for the full tanh/readout model. Merely observing monotonically decreasing loss is also not sufficient to assert a fixed positive `alpha`.

Writing `S_K = sum_{k<K} eta_k`, weighted Cauchy–Schwarz gives

\[
\begin{aligned}
\|a_K-a_0\|_2^2
&=\Big\|\sum_{k<K}\eta_k\nabla_aL_k\Big\|_2^2\\
&\le S_K\sum_{k<K}\eta_k\|\nabla_aL_k\|_2^2\\
&\le \frac{S_K}{\alpha}(L_0-L_K)
\le \frac{S_KL_0}{\alpha}.
\end{aligned}
\]

Thus reaching distance `D` requires `S_K >= alpha D^2/L_0`. At constant learning rate,

\[
K\ge\frac{\alpha D^2}{\eta L_0}.
\]

A diminishing-rate schedule enters through its accumulated learning rate; the iteration count alone does not determine the drift budget. If `L_0=0`, the initial gradient is zero for a differentiable nonnegative objective and no motion occurs.

## Reaching many large slopes

Suppose at least `M` slopes must reach `|a_j| >= Gamma`, and all initial slopes satisfy `|a_j(0)| <= g_0 < Gamma`. The reverse triangle inequality implies, regardless of signs or sign changes,

\[
D^2\ge M(\Gamma-g_0)^2.
\]

This also holds when the subset of `M` neurons is chosen at the endpoint: the initial uniform bound applies to every possible subset. A bound on the **mean** initial gamma is insufficient for this particular uniform hypothesis; one must use actual initial magnitudes or a verified bound. More generally use the sum of the appropriate squared positive gaps to the target set.

If `Gamma = lambda_*/h = Theta(W)`, `g_0=O(1)`, `L_0=O(1)`, and a fixed positive fraction `M=Theta(W)` must attain that threshold, the necessary effective time is `Omega(W^3)`. Requiring only a single slope gives `Omega(W^2)`. Iterations add the reciprocal learning-rate factor. The width conclusion requires the actual center/halo convention to satisfy `h^{-1}=Theta(W)`.

Illustrative arithmetic only: take `M=177`, `g_0=1`, `Gamma=16`, `L_0=0.3`, `eta=0.002`, and `alpha=1/2`. Then `D^2 >= 39,825`, so the discrete lower bound is **33,187,500 steps** (unit-rate flow: 132,750 time units). For one slope the analogous count is 187,500 steps. These are hypothetical constants, **not measured bounds on a particular run**; in particular the example presupposes that all 177 slopes must grow to 16, which has not been proved necessary. A mean-gamma observation cannot substitute for the per-slope premises.

## Remaining loss gives a sharper late-time bound

Restart at any time `t_f` or step `k_f`. Future displacement `D` requires

\[
H\ge D^2/L(t_f)
\]

for unit-rate flow, and

\[
\sum_{k=k_f}^{K-1}\eta_k\ge\alpha D^2/L_{k_f}
\]

for sufficiently descending discrete GD. This provides a direct sense in which a small **actual training loss** leaves little raw-parameter travel over a fixed future budget, without a readout-coefficient bound. For example, a relative RMS error `epsilon` has `L = epsilon^2 mean(y^2)/2` under the experiment's half-MSE convention.

An evaluation-only least-squares floor is **not** the actual GD objective and cannot replace `L_{k_f}` in this inequality. For genuine smooth VarPro flow one may instead apply the energy argument to its actual objective `Phi`, but the raw full-GD dynamics are different.

## Learning rates and parameterization matter

For a constant positive diagonal preconditioner `D_p`, flow `pdot = -D_p grad L` obeys

\[
\int_0^T\|\dot p\|_{D_p^{-1}}^2\,dt=L(0)-L(T),\qquad
\|p(T)-p(0)\|_{D_p^{-1}}^2\le T L(0).
\]

The same weighted metric gives the discrete result for fixed block learning-rate ratios, assuming sufficient descent in that metric. A larger geometry rate reduces the cost of physical slope motion. For varying diagonal rates, a safe Euclidean slope bound replaces `sum eta_k` by the sum of the largest slope-coordinate step coefficient at each iteration, under the corresponding sufficient-descent hypothesis. Momentum, Adam, arbitrary large non-descending GD steps, or optimizer resets do not inherit the ordinary-GD bound automatically.

Euclidean GD in log-gamma pays distance in log-gamma, not raw gamma. A common tied gamma pays the scalar parameter distance, not the sum of distances of every displayed copy, unless the update is scaled to reproduce the independent-coordinate metric. In particular, a summed-gradient update for one shared gamma and a mean-gradient update for that gamma have different effective clocks. These distinctions explain why a raw-coordinate lower bound can coexist with successful reparameterized or block-rate interventions.

## Comparison with Section 5 and limits of the conclusion

The note's raw drift estimate uses `|partial_a L| <= X |c| R` and then bounds its time integral. The present energy estimate is complementary: it uses no Fourier gap, readout bound, sample-gap condition, or presumed residual decay. It applies collectively to many slopes and can give a stronger width power under a specified large-slope target set. The note's centered inverse-square/cube bounds concern a different coordinate metric and stronger spatial assumptions.

Neither estimate is a proof that ordinary GD cannot fit the target. Necessary remaining work includes:

1. Establishing that the desired accuracy with the desired coefficient/numerical constraints requires enough large slopes, rather than merely showing that the QI construction has them.
2. Establishing the discrete descent constants and learning-rate/width normalization for the actual optimizer under discussion.
3. Comparing the resulting bound with practical horizons using measured initial/current losses and distances, rather than asymptotic powers alone.
4. If the desired statement is about an accuracy plateau, connecting remaining distance or projected sensitivity to a lower bound on attainable error.

Finite total squared speed is not finite total path length on an infinite interval. The `sqrt(T)` drift bound grows without limit, so this energy argument gives finite-horizon impracticality, **not nonreachability**. It also does not isolate readout compensation: any route that lowers actual loss reduces the available energy. Identifying readout adaptation as the specific cause still needs separate directional evidence.


## Violating this sufficient-descent condition does not prove instability

A finite-horizon lower bound conditional on `alpha=1/2` cannot be described as a proof that faster motion requires unstable training. Consider the scalar quadratic `L(p)=beta p^2/2`: GD is asymptotically stable whenever `0 < eta beta < 2`, whereas its exact descent coefficient is `alpha = 1 - eta beta/2`. The rate `eta=1.9/beta` is stable and loss-decreasing but has `alpha=0.05`, so it violates the `alpha=1/2` condition. Nonmonotonic methods can also be stable overall.

To reach the specified distance faster than a valid bound, at least one premise must fail: the target-distance assumption, loss budget, metric, accumulated learning rate, or sufficient-descent constant. That does not identify instability as the violated premise. Our larger geometry-rate or reparameterized successes therefore can coexist with this argument. We do **not** currently have a theorem that acquiring useful gamma necessarily requires instability.

# expF20 — Local ridge PDE feasibility — preliminary measured results

## TL;DR

- The repository already contains analytic ridge derivatives, streamed forward/transpose operations, nonlinear linearizations, and a historical million-neuron PDE result with 1,287 independent coordinates. Its existing streamed path still evaluates all centers.
- The new prototype uses local center windows, prefix-sum value tails, and matching transposes. At 128 directions and 1,024 points, second-derivative evaluation takes approximately 24 ms for both 1.05 million and 4.19 million neurons.
- All 1,048,704 readouts can remain independent unknowns. Raw coordinates fit the sampled equations but produce 24% held-out field error. An invertible smoothness-based coordinate map gives $1.3\times10^{-5}$ field error after 200 iterations; increasing interior samples improves it to $3.7\times10^{-6}$. These runs hit iteration budgets, with appreciable held-out PDE residuals.
- Explicitly restricted function coordinates give much higher accuracy: 120 unknowns in a 1,081,600-neuron nonlinear network reach $1.4\times10^{-14}$ field error on the smooth target. A more oscillatory target needs 276 unknowns to reach $6.1\times10^{-13}$. A much smaller neuron bank matches low-degree results; neuron count alone does not explain accuracy.

## Question / hypothesis

Can the chain-rule ridge structure support practical PDE computation at millions of neurons, and does fast evaluation suffice to solve for millions of independent readouts? We separate representation evaluation, independent-readout solution selection, and reduced-coordinate approximation accuracy.

## Experiment design

The flat network is $q(x)=b+\sum_{m,j}w_{mj}\tanh(\gamma(\theta_m\cdot x-c_j))$, with uniformly spaced centers and equally spaced unit directions over $[0,\pi)$. Positive-order derivative evaluation retains centers with $|\gamma(\theta_m\cdot x-c_j)|\le T$. Value evaluation splits tanh into its step limit and a localized correction. Prefix sums evaluate the step contribution; difference-array accumulation followed by a prefix sum applies its transpose. The native float64 CPU kernel supports orders zero, one and two, with direction factors supplied separately. Unit directions make the sum of second profile derivatives equal to the Laplacian. Uniform center spacing permits an exponential recurrence along each local window. Compilation uses optimization without fast-math.

With $\gamma h=0.25$ and $T=18$, each projected point visits approximately 145 centers. Work is approximately $O(MN+KMs)$, including coefficient scans and transpose output initialization, where $s$ is the local stencil width. Storage includes $MN$ coefficients and $MK$ projections, but no $K\times MN$ feature matrix. The current wrapper retains all projections; integration with the existing point/direction streaming engine remains future work.

Operator benchmarks use 1,024 points, deterministic random weights, three-repeat median local timings, and an independent all-center dense control at 16 points. The main sweep holds 128 directions fixed and varies centers from 257 to 32,769. A second sweep keeps roughly 1.05 million neurons and varies directions from 16 to 512. Peak RSS includes Python/SciPy and dense-control temporaries. Timings are preliminary on the local Apple Silicon Mac mini, with one numerical-library thread per process. Some solve runs overlapped; their timings are not a controlled speed competition.

Free-readout tests use centers on $[-1.25,1.25]$, an output bias, and every $MN$ readout as an independent unknown. The unit-disk reference is $u=e^{0.3x}\cos(1.3y)+0.1xy$. Forcing is manufactured for either $-\Delta u=f$ or $-\Delta u+u^3=f$. Only forcing and Dirichlet boundary values enter optimization; all unknowns start at zero. Base million-width tests use 1,024 scrambled Sobol interior points and 256 boundary points; sample refinement uses 8,192 interior points. Validation uses 2,048 independently sampled interior points and 401 shifted boundary points. Training seeds are 813 and, for one sensitivity check, 814; validation seed is 18973.

The objective is squared interior residual RMS plus $4^2$ times squared boundary residual RMS. Raw coordinates use $w=z/\gamma^2$. Smooth coordinates use the symmetric invertible center-wise map $w=\operatorname{IDCT}(d_k\operatorname{DCT}(z))$, where $d_k=h/[1+(\pi k/(Nh))^2]^p$. Every multiplier is positive: no modes are removed. Main tests use $p=1$, with $p=2$ as a sensitivity check. This changes the implicit solution-selection preference as well as conditioning and relative bias scaling; it is not a controlled conditioning-only comparison.

Main LSMR runs use 50-iteration chunks and a 200-iteration budget; the raw million case uses 100. A separate $p=2$ run uses one uninterrupted 400-iteration solve. Nonlinear chunks use $J=-\Delta+3q^2$ and right-hand side $f+2q^3$, with residual-decreasing backtracking. All four recorded nonlinear steps were full steps. Timings include validation between chunks. Finite-budget iterates are not claimed to attain a global minimum-norm solution.

Reduced controls reuse the existing QUILL encoding and polynomial angular map $w=Ea$. Collocation values and Laplacians come from actual tanh neurons, not substituted polynomial derivatives. The small reduced matrices are materialized and solved with dense QR/Gauss–Newton. The million geometry has 256 directions and 4,225 centers including halos, with $\gamma h=0.2$. Degrees 6, 10, 14, 18 and 22 give 28, 66, 120, 190 and 276 independent function coefficients. The harder target replaces $\cos(1.3y)$ with $\cos(5y)$. These tests use 1,024 interior and 128 boundary training points, and 2,048 independent interior plus 401 boundary validation points. A 9,248-neuron control uses matched degrees 6, 10 and 14. The large angular overresolution is representation stress, not evidence that these targets need a million neurons.

Field relative error is $\|q-u\|_2/\|u\|_2$ on held-out points. PDE RMS is $\sqrt{\operatorname{mean}(|\mathcal Nq-f|^2)}$ there. These are independent manufactured-solution checks, not a universal PDE error certificate.

**Code & data.** New code: `experiments/expF20_local_ridge_pde/{local_kernel.c,local_operator.py,run.py,run_reduced.py,verify_results.py,plot_results.py,config.yaml,README.md}`; tests: `tests/test_local_ridge_operator.py`. Per-run JSONs, saved coefficients and `metrics.jsonl` are beside this report. Figures: `feasibility.png`, `operator_scaling.png`. Existing code: `experiments/expF01_linear_de_zoo/pinn_poc.py:190–223`, `experiments/expF19_radon_direct_pde/solver/ridge_pinn_features.py:9`, `experiments/expF19_radon_direct_pde/route2_directional_operator.py:60–75,197–242,299–382`, and `experiments/expF19_radon_direct_pde/solver/streamed_residual.py:212–251`. Historical million result: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/five_dimensional_refinement_control/results.json`. Python environment: `/Users/sam/venv/quill-radon/bin/python`.

## Results

### Existing implementation

The early PDE proof of concept uses dense least squares for independent ridge readouts at widths up to 2,304. The later directional operator implements analytic tanh jets, derivative-bank reuse, forward and transpose operations, and generic nonlinear residual linearizations. Its unknowns are function/profile coordinates. Its panels still visit every center: batching bounds memory but retains $O(KMN)$ activation work. Boundary, initial, observation and inverse-parameter constraints already exist. The generic default can use product gates, so those neuron counts must be distinguished from the flat ridge path.

A verified historical five-input nonlinear result has 1,003,833 neurons, 6,561 directions and 1,287 function coordinates. It records field error $2.73\times10^{-9}$, PDE RMS $1.73\times10^{-8}$, approximately 917 seconds cold solve and 476 MB peak RSS, with a budget exit. This is prior saved evidence, not a rerun. It differs from our new two-dimensional tests, so their times are not a controlled before/after comparison. No existing combination of local center windows, prefix-sum values and million-independent-readout solving was found in the inspected ridge paths.

### New operator scaling

| Neurons | Directions | Centers/direction | Value forward | Laplacian forward | Laplacian transpose | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 32,896 | 128 | 257 | 15.7 ms | 22.8 ms | 22.8 ms | 104 MB |
| 131,200 | 128 | 1,025 | 16.3 ms | 24.2 ms | 24.5 ms | 107 MB |
| 1,048,704 | 128 | 8,193 | 16.7 ms | 23.7 ms | 23.9 ms | 144 MB |
| 4,194,432 | 128 | 32,769 | 18.6 ms | 24.1 ms | 24.9 ms | 267 MB |

All rows evaluate 1,024 points. A dense float64 feature matrix would occupy 8.59 GB at the million geometry and 34.36 GB at four million; neither is formed. At roughly fixed million width, increasing directions from 16 to 512 raises Laplacian forward time from 3.1 to 94.6 ms. Center refinement is accelerated; angular resolution remains expensive.

The existing dense multi-profile evaluator is faster at the small 289-center geometry. At 4,225 centers, the new path is approximately 1.8–2.9 times faster for multi-profile values and 3.3–4.4 times for Laplacians on the bounded audit batch. Larger ratios against direct per-direction dense evaluation also include implementation differences.

### Independent-readout PDE results

Every row below retains 1,048,704 independent neuron readouts plus bias and exits at its iteration budget.

| Coordinates / equation | Interior points | Iterations | Field relative error | PDE RMS | Solve + monitor |
|---|---:|---:|---:|---:|---:|
| Raw / linear | 1,024 | 100 | $2.42\times10^{-1}$ | 1.95 | 6.8 s |
| Smooth $p=1$ / linear | 1,024 | 200 | $1.31\times10^{-5}$ | $4.13\times10^{-3}$ | 29.0 s |
| Smooth $p=1$, second seed / linear | 1,024 | 200 | $1.23\times10^{-5}$ | $4.13\times10^{-3}$ | 31.2 s |
| Smooth $p=1$ / linear | 8,192 | 200 | $3.67\times10^{-6}$ | $8.37\times10^{-4}$ | 104.0 s |
| Smooth $p=1$ / nonlinear | 1,024 | 200 | $4.49\times10^{-5}$ | $4.19\times10^{-3}$ | 39.7 s |
| Smooth $p=2$ / linear | 1,024 | 200, 50-step chunks | $6.67\times10^{-5}$ | $5.54\times10^{-3}$ | 28.6 s |
| Smooth $p=2$ / linear | 1,024 | 400, uninterrupted | $6.43\times10^{-6}$ | $1.28\times10^{-3}$ | 57.5 s |

The raw million case has weighted training residual $2.09\times10^{-6}$ despite 24% field error. Smooth coordinates have a larger training residual and a far better field. Minimizing the finite sample objective in arbitrary readout coordinates is insufficient here. The $p=2$ runs change both budget and restart policy, so they do not isolate restarting. A smaller overdetermined control with 4,112 neurons, 4,096 interior points and 256 boundary points gives raw field error $4.08\times10^{-5}$ and smooth error $4.66\times10^{-6}$ after 200 iterations. Million width is not uniformly better under these budgets and sample policies.

### Reduced-coordinate PDE results

| Nonlinear target | Neurons | Independent coordinates | Field relative error | PDE RMS |
|---|---:|---:|---:|---:|
| $e^{0.3x}\cos(1.3y)+0.1xy$ | 1,081,600 | 120 | $1.37\times10^{-14}$ | $4.03\times10^{-12}$ |
| $e^{0.3x}\cos(5y)+0.1xy$ | 1,081,600 | 120 | $1.22\times10^{-6}$ | $2.62\times10^{-4}$ |
| Same harder target | 1,081,600 | 190 | $1.02\times10^{-9}$ | $2.86\times10^{-7}$ |
| Same harder target | 1,081,600 | 276 | $6.12\times10^{-13}$ | $2.08\times10^{-10}$ |

The smooth linear degree-14 solve reaches $1.52\times10^{-14}$ field error. Geometry/features through degree 14 take 4.84 seconds; the entire bundle of 12 solves and audits takes 5.04 seconds. The degree-22 refinement bundle takes 7.83 seconds, including 7.62 seconds of geometry/features. Its final nonlinear solve takes 0.026 seconds after features exist. Feature setup includes validation features. The 9,248-neuron control has essentially identical accuracy at matched degree: richer independent function coordinates drive the harder-target accuracy improvement.

### Figures

- `feasibility.png`: fixed-direction evaluation times versus neuron count; free-readout field and PDE convergence; and nonlinear reduced-coordinate accuracy versus independent coefficient count. The two kinds of unknowns are separated explicitly.
- `operator_scaling.png`: direction-count cost at nearly fixed width, and measured local/direct-dense speedups on the same 16-point batch. No full-batch dense timing is extrapolated and presented as measured.

## Additional details

Seventeen targeted tests pass, including independent dense derivatives, mixed physical derivatives, transpose identities, finite-difference and transpose checks of the quadratic $q\Delta q$ residual, cutoff sensitivity and existing directional-operator tests. Every saved free solution was checked using the full ordinary tanh bank at 16 fresh points: value disagreement below $5\times10^{-15}$ and Laplacian disagreement below $10^{-12}$. Reduced compiled-network and wider-cutoff checks also stay around $10^{-12}$ or below for the solved Laplacians. These sampled audits are not domain-wide bounds.

Conservative exact-arithmetic tail bounds for derivative orders zero, one and two are $2e^{-2T}\|w\|_1$, $4\gamma e^{-2T}\|w\|_1$ and $8\gamma^2e^{-2T}\|w\|_1$ for unit direction factors. Roundoff and prefix-sum cancellation add error. Cutoff 18 is not a universal accuracy guarantee as slope grows. Integration should preserve the existing anchored value arithmetic for large cancelling halo weights.

The prototype is CPU-only, with two-dimensional test PDEs and kernel derivatives through order two. It has not been tested at billions of neurons. It does not establish generic high-dimensional, stiff, turbulent, nonsmooth or noisy-data performance. No production solver was changed. Review found and fixed a concurrent compilation race and a failed-line-search acceptance edge case; the recorded nonlinear run used full decreasing steps and was unaffected.

## Conclusions

Local windows and prefix sums make actual forward and transpose operations practical at one to four million tanh neurons in these tests. Reliable solution accuracy still requires control of function coordinates, smoothness and sampling; efficient evaluation alone does not provide that control.

## Open questions

Can declared function-space regularization and adaptive residual sampling make free-readout accuracy predictable without fixing a low-degree polynomial space? Can the kernel join the existing streamed forward/adjoint path while retaining anchored accuracy? How should cutoff, angular resolution and solver tolerance be coupled to physical PDE error?

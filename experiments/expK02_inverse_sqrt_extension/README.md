# Reciprocal square root with a smooth halo extension

The multiscale cardinal-QI construction works on **f(x) = 1/sqrt(x), x in [1e-4, 1]**, without changing the target on that interval. The selected ordinary, single-hidden-layer tanh network has **4,815 neurons**, measured physical-domain relative L2 of approximately **5e-14** with NumPy and **3.9e-14** with PyTorch. The maximum relative error over 45,056 test points is **3.53e-13**. These are empirical numerical measurements, not certified uniform bounds.

The important change is the geometry: 15 geometrically scaled QI blocks replace one global uniform lattice. Their outputs combine into one shallow MLP. This is a multiscale extension of the existing cardinal-QI recipe, not a claim that the original uniform-grid recipe reaches this precision at this width. There is no training, least-squares readout fit, log-input feature, or target approximation surrogate.

## Domain and the halo problem

For epsilon = 1e-4, use

\[
 x=m+rz,\qquad m=0.50005,\quad r=0.49995,\quad z\in[-1,1].
\]

For stable evaluation near the left endpoint, the implementation uses the equivalent forms `z=(x-epsilon)/r-1` and `x=epsilon+r*(z+1)`. The singularity x=0 maps to z=-1.0002000200020003, only about 2e-4 beyond the left endpoint. Naively extending the formula into an ordinary halo therefore samples through the singularity.

A smooth extension resolves the undefined halo samples. Local resolution is also needed: the original target itself changes rapidly on the scale of epsilon. An extension cannot remove that variation while preserving the target.

## An exact bounded extension

Define a cutoff H(u) that is zero for u<=1, one for u>=2, and transitions on [1,2] by

\[
 H(u)=I_{u-1}(p+1,p+1),\qquad p=16,
\]

where I is the regularized incomplete beta function ([NIST DLMF, §8.17](https://dlmf.nist.gov/8.17)). Its derivative on the transition interval is proportional to `(u-1)^p(2-u)^p`, so its first p derivatives match the constant pieces at both ends. Thus H is C^16.

Set s_0=epsilon/2, s_j=2^j s_0, and choose J so s_J>=1. Here J=15 and s_J=1.6384. For x>0 define

\[
 d_{\rm ext}(x)=-\frac{1}{2x^{3/2}}
       \left[H(x/s_0)-H(x/s_J)\right],
 \qquad
 F(x)=\varepsilon^{-1/2}+\int_\varepsilon^x d_{\rm ext}(t)\,dt.
\]

Set d_ext=0 for x<=0. In fact it is already zero for x<=s_0 and x>=2s_J, so the formula never needs to evaluate a singular power there. F is bounded and C^17 on the real line, and constant outside that finite interval. On [epsilon,1], the bracket is exactly one, hence F'=f' and F(epsilon)=f(epsilon); therefore **F=f throughout the requested interval**. The extension is smooth, but is not claimed to be globally analytic.

The actual constructor needs derivative samples and one primitive value. Quadrature in `Extension.value` provides the primitive outside the original domain; it does not generate training labels or change the original function.

## One prototype at 15 scales

The cutoff difference telescopes:

\[
 H(x/s_0)-H(x/s_J)=
 \sum_{j=0}^{J-1}\left[H(x/s_j)-H(x/(2s_j))\right].
\]

Each summand is supported on [s_j,4s_j]. With u=x/s_j, every derivative component has the same shape,

\[
 D(u)=-\tfrac12u^{-3/2}[H(u)-H(u/2)],
\]

up to its known scale factor. This permits a single QI construction for a primitive of D on u in [0,4]. Map that prototype by u=2+2t, t in [-1,1]. Its derivative in t is `-u^(-3/2)[H(u)-H(u/2)]`; it is zero outside u in [1,4]. A zero derivative outside the support is therefore a smooth, valid halo extension for the prototype.

If Q(t) is the constructed prototype primitive, use

\[
 q(x)=\varepsilon^{-1/2}+\sum_{j=0}^{J-1}s_j^{-1/2}
 \left[Q\!\left(\frac{x}{2s_j}-1\right)
       -Q\!\left(\frac{\varepsilon}{2s_j}-1\right)\right].
\]

Only affine input maps, constant factors and sums appear. Expanding the sums gives one ordinary tanh MLP; there is no multiplication of network outputs or nonlinear input transform. The decomposition is exact before the numerical QI approximation.

The selected prototype uses N=256 intervals, halo R=32 on each side, cardinal stencil half-width Kc=160, and lambda=0.25. Total hidden width is `15*(256+2*32+1)=4815`. The stencil solve and convolution use 30-decimal mpmath arithmetic; derivative samples and exported coefficients are float64. The same dimensionless stencil is reused across widths with the exact h-scaling, using an experiment-local cache.

## Centering and export

Write Q(t)=c+sum_n a_n tanh(gamma(t-c_n)). In centered z coordinates, each block has

\[
 w_{jn}=\frac{\gamma r}{2s_j},\quad
 \theta_{jn}=\gamma\left(\frac{\varepsilon}{2s_j}-1-c_n\right),\quad
 b_{jn}=w_{jn}+\theta_{jn},\quad v_{jn}=s_j^{-1/2}a_n.
\]

The output bias is `100 - sum(v*tanh(theta))`, accumulated with `math.fsum`. The implementation evaluates theta as `gamma*(epsilon/(2*s_j)) - lambda*n` using the integer grid index n, and factors the slope through `r/epsilon` and the dyadic scale ratio. This keeps `b-w=theta` exact in float64 for the widths in the sweep; the constructor checks it. This avoids losing precision by expanding large, nearly cancelling expressions for the bias. Ordinary forward evaluation still experiences rounding near z=-1, which matters at the observed error floor. The largest centered hidden weight is 159,984; the readout L1 norm is about 57.52.

## Measurements and limits

The reported relative L2 uses physical dx on [epsilon,1]. Since f(x)^2 dx = d(log x), it can be evaluated as

\[
 \frac{\|q-f\|_{L^2(dx)}}{\|f\|_{L^2(dx)}}
 =\left[\frac{1}{\log(1/\varepsilon)}
       \int_{\log\varepsilon}^{0}
       \left(\frac{q(e^t)-f(e^t)}{f(e^t)}\right)^2dt\right]^{1/2}.
\]

Log coordinates are used only for integration and sampling, never as network features. Composite Gauss-Legendre rules with 64, 128, 256 and 512 points per dyadic interval give roughly 4.9e-14 to 5.3e-14. Differences at this level include floating-point evaluation noise. An ordinary Torch Linear/Tanh/Linear network gives 3.88e-14. A separate 60-decimal evaluation of the saved float64 network at 25 log-spaced points has maximum relative error 1.00e-13.

The selected model is checked on 45,056 endpoint-inclusive log, uniform and random points. Maximum sampled absolute error is 3.50e-11, occurring near the large-valued left edge; maximum sampled relative error is 3.53e-13. The derivative partition matches f' to about 5.2e-16 relatively in a separate check.

A smaller model with N=128 and halo 32 uses 2,895 neurons and reaches about 1.8e-13 relative L2. Halo and smoothness sweeps are saved, including an accidentally favorable halo-16 result; the selected halo is 32, where increasing it further does not materially change the precision regime. Neighboring epsilon values 1e-3 and 1e-5 also work, at about 1.1e-14 and 2.8e-13 relative L2 respectively, using the same prototype parameters and an adjusted number of scales.

The global uniform cardinal-QI baseline was tested through N=16,384, with 160 halo centers on each side. Its error remains about 1.07, with very large cancelling readout weights. At that spacing the left extension transition is not sampled internally at all; using cutoff orders 4, 8 or 16 gives identical sampled derivatives and identical results there. This is a result for this direct uniform recipe and tested resolution, not an optimality claim about all uniform dictionaries or fitted readouts.

## Reproduction and saved artifacts

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK02_inverse_sqrt_extension/run.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK02_inverse_sqrt_extension/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expK02_inverse_sqrt_extension/scaling_window.py
~/venv/quill-radon/bin/python experiments/expK02_inverse_sqrt_extension/plot.py
~/venv/quill-radon/bin/python -m pytest -q tests/test_inverse_sqrt_extension.py
```

Outputs: `results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension/`.

- `inverse_sqrt_qi.png` / `.pdf`: extension, width scaling, and pointwise error.
- `models/multiscale_p16_N256_R32.npz`: selected centered shallow MLP.
- `selected_torch.pt`: standard Torch model state plus input-domain metadata.
- `selected_predictions.npz`: dense physical inputs, targets, predictions and relative errors.
- `verification.json`, `compact_sweep.json`, `halo_sweep.json`, `results.json`: measured checks and all sweeps.
- `config.json`, `selected_config.json`, `provenance.json`: settings and source hashes.

This was exploratory, with configuration choices informed by pilot runs. No Adam experiment was run for this target.

The scaling panel now uses a linear total-width axis from 2k to 6k. `scaling_2k_6k.json` contains 18 measured widths (1,935 through 6,015, bracketing the displayed window), all with 15 blocks, cutoff order 16, and halo 32. The selected 4,815-neuron model is unchanged.

The standalone scaling figure is `inverse_sqrt_scaling.png` / `.pdf`: only the 15-block construction, with a linear 1k–5k width axis. Data are in `scaling_1k_5k.json`, including newly measured coarse models down to 990 neurons to bracket the window. Reproduce with `scaling_window.py --window 1k-5k`, then `plot_scaling.py` from this experiment directory (using the repository root as the working directory).

# Gamma controls the scales available to readout gradient descent

**Gamma changes how quickly a readout learns an attainable target.** For a
fixed set of neuron centers, a common slope gamma applies an explicit
frequency filter to the features and hence to the kernel governing readout
updates. Small gamma strongly attenuates fine spatial variation. Carrying
this filter through the finite training geometry predicts the measured
learning delay: in our controlled comparison, reducing gamma from 64 to 8
requires 986.59 times as many updates to reach the same attainable accuracy.
Across five tested targets, the measured slowdown ranges from 3.34 to
986.59, and all 19 available executed acquisition times lie inside their
predicted intervals; one further run is censored.
The mechanism is exact; the size of the delay depends on the target,
geometry, and optimizer. The theorem below states the mechanism, the main
figure tests its quantitative consequences, and the appendix supplies the
standard conversion from kernels to learning times.

**Table 1. Main-text notation.** Gamma is a physical slope; frequency is
measured in radians per unit input distance.

| Symbol | Meaning |
|---|---|
| $\gamma,c_j,W$ | Common hidden slope, fixed center of neuron $j$, and hidden width. |
| $x_i,m$ | Training inputs and their count. |
| $k_\gamma,K_\gamma$ | Feature inner-product kernel and its sampled, mean-normalized update matrix. |
| $\rho_\gamma$ | Unit-mass averaging density that turns a sharp step into a tanh feature. |
| $M_\gamma(\omega)$ | Amplitude retained by this averaging at frequency $\omega$. |
| $r_n,\eta_\gamma$ | Vector of training residuals after $n$ updates and the readout GD step. |

## The readout kernel gives gamma a direct role in optimization

Consider

$$
f_{\gamma,\theta}(x)=b+\sum_{j=1}^W w_j\tanh\!\bigl(\gamma(x-c_j)\bigr).
$$

Freeze the centers and common slope, and train only the readout
$\theta=(b,w_1,\ldots,w_W)$ from zero on half empirical mean squared error.
Comparisons vary gamma while holding the samples, target, centers, and raw
coefficient coordinates fixed. We specify the step-size rule as part of
the comparison.

The feature vector is
$\phi_\gamma(x)=(1,\tanh(\gamma(x-c_1)),\ldots,\tanh(\gamma(x-c_W)))^T$.
Define $k_\gamma(x,x')=\phi_\gamma(x)^T\phi_\gamma(x')$ and
$(K_\gamma)_{i\ell}=k_\gamma(x_i,x_\ell)/m$. Writing $r_n$ for target
values minus current predictions at the training inputs, ordinary readout
GD gives

$$
\boxed{r_{n+1}=(I-\eta_\gamma K_\gamma)r_n.}
$$

Thus the kernel describes which prediction errors a coefficient update
corrects together. Similar feature responses at two inputs make their
predictions move together; learning a difference between them can be slow.
Explaining gamma's optimization effect therefore requires identifying how
gamma changes this kernel.

## Theorem: gamma explicitly filters the learning kernel

For any fixed centers and common slope $\gamma>0$, define the reference
step kernel and averaging density

$$
k_{\mathrm{step}}(x,x')
=1+\sum_{j=1}^W\operatorname{sign}(x-c_j)\operatorname{sign}(x'-c_j),
\qquad
\rho_\gamma(t)=\frac{\gamma}{2}\operatorname{sech}^2(\gamma t).
$$

**Theorem (gamma-controlled smoothing).** The tanh readout kernel is
exactly the fixed step kernel averaged over each input:

$$
k_\gamma(x,x')=
\iint_{\mathbb R^2}\rho_\gamma(x-t)\rho_\gamma(x'-s)
k_{\mathrm{step}}(t,s)\,dt\,ds.
$$

In either input, this averaging multiplies a frequency-$\omega$ component
by

$$
\boxed{
M_\gamma(\omega)=\frac{z}{\sinh z},
\qquad z=\frac{\pi|\omega|}{2\gamma},
\qquad M_\gamma(0)=1.
}
$$

Consequently, a common-slope cap $0<\gamma\le\Gamma$ implies
$M_\gamma(\omega)\le M_\Gamma(\omega)$. Frequencies well below gamma
are nearly preserved, while frequencies well above gamma have exponentially
small multipliers: $M_\gamma(\omega)\sim2ze^{-z}$ as $z\to\infty$.

The reference step changes from $-1$ to $+1$ at each center. Averaging it
with $\rho_\gamma$ produces precisely the original tanh feature, with a
transition width proportional to $1/\gamma$. Applying the same averaging
to both factors in a feature inner product proves the kernel identity.
The Fourier transform of the density gives the displayed multiplier;
Appendix A supplies the proof. No periodic assumption on the samples or
centers is needed.

**The quantitative mechanism is selective attenuation of spatial scales.**
For example, reducing gamma from 64 to 8 changes the multiplier at
$\omega=2\pi$ from 0.9960 to 0.7851, but at $\omega=10\pi$ it changes
from 0.9074 to 0.02584. The finer component's multiplier becomes about
35.1 times smaller. Gamma therefore changes the relative strength of
different corrections, even when a curvature-based step compensates for
overall kernel magnitude.

The finite sampling and center geometry determine how these filtered
components combine into learning directions. Fourier waves need not be
eigenvectors of the sampled kernel, so a multiplier squared is generally
not a finite-kernel eigenvalue or an inverse learning time. We retain those
couplings when computing the prediction. The cap guarantees feature
attenuation; the target's interaction with the resulting kernel determines
the learning delay.

A minimal example separates this effect from capacity. With one centered
neuron, samples $-s,+s$, and labels $-1,+1$, every positive gamma fits
exactly using $w=1/\tanh(\gamma s)$ and $b=0$. Nevertheless, at step
$\eta=0.5$, reaching 1% residual takes 925 updates when $\gamma s=0.1$
and 14 when $\gamma s=1$. The kernel's largest eigenvalue is one in both
cases. The target's contrast direction has eigenvalue
$\tanh^2(\gamma s)$, which explains the different rates. Appendix B gives
the general learning law and this example's exact count.

## The explicit filter retains the observed timing accuracy

We apply the gamma filter to the fixed geometry, compute the resulting
kernel's target-dependent GD decay, and compare its predicted crossings to
executed readout GD. The calculation uses the target and prescribed step,
without fitting a rate to a training trajectory. A controlled finite
expansion makes the calculation practical; its approximation bounds give
the intervals below. This is a retrospective check against archived runs.

The primary comparison uses 559 hidden neurons, 8,193 equally spaced
training samples on $[-1,1]$, fixed equispaced centers with a boundary halo,
zero readout initialization, and target

$$
f^\star(x)=\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(10\pi x).
$$

Each saved step is approximately $0.5/L_\gamma$, where $L_\gamma$ is the
largest eigenvalue of $K_\gamma$. Thus the comparison uses the same
curvature-normalized step rule. All four models actually attain 1% relative
residual, equivalent to relative squared loss $10^{-4}$.

**Table 2. Predicted and executed updates to 1% training residual.**
Intervals are deterministic necessary–sufficient counts for the fixed
problem above, with independently certified endpoints; they are not
statistical confidence intervals.

| Common slope | Filter-derived interval | Executed first hit |
|---:|---:|---:|
| 8 | 15,784,048–15,812,623 | 15,798,313 |
| 12 | 186,057–186,058 | 186,057 |
| 16 | 61,792–61,792 | 61,792 |
| 64 | 16,013–16,013 | 16,013 |

The predicted gamma-8/gamma-64 delay ratio is **985.70–987.49**; the
executed ratio is **986.59**. At gamma 8, the interval's total width is
0.181% of the observed time. A control that only rescales the gamma-64
kernel to match each case's largest eigenvalue predicts 16,013 updates
at every gamma. The full frequency-dependent filter accounts for the large
delay that this scalar control misses.

This establishes a quantitative optimization effect at a tolerance all
four models can reach. The exact mechanism holds for arbitrary fixed
centers with a common slope; the tight numerical intervals evaluate the
specified geometry and target. They do not assert that every target slows
by the same factor or that learning time is monotone in gamma.

The archived study also evaluates four other targets on the same geometry
and at the same four slopes. They cover a single oscillation frequency, a
localized rational profile, a quadratic, and an exponential of a sine.
Every target reaches 1% at both gamma 8 and gamma 64, allowing a measured
comparison of acquisition times at those endpoints.

**Table 3. The gamma-induced delay depends strongly on the target.**
Each entry is the executed first-hit count at gamma 8 divided by that at
gamma 64, rounded to two decimals. The samples, centers, zero readout
initialization, 1% relative-residual tolerance, and curvature-normalized
step rule are shared with Table 2. Appendix E gives the predicted
intervals, executed counts, and verification status for all four slopes.

| Target | Exact function $f^\star(x)$ | Measured slowdown |
|---|---|---:|
| Single sine | $\sqrt2\sin(2\pi x)$ | 3.34× |
| Runge | $1/(1+25x^2)$ | 45.16× |
| Quadratic | $\sqrt5x^2$ | 58.17× |
| Exponential of sine | $\exp(\sin(3\pi x))$ | 178.05× |
| Sine mixture | $\sin(2\pi x)+\frac12\sin(6\pi x)+\frac14\sin(10\pi x)$ | 986.59× |

The same gamma filter produces very different acquisition delays because
each target places different energy in the finite kernel's learning
directions. These observations support a quantitative, target-dependent
explanation rather than a universal slowdown factor. In particular, the
quadratic still needs 58.17 times as many updates at gamma 8: low polynomial
degree alone does not imply fast acquisition in this tanh dictionary.
The finite geometry and target alignment remain part of the calculation.

## Three panels: observed scale gap, measured delay, and a slope intervention

<figure>
  <img src="../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/paper_three_panel.png" alt="A: learned slopes remain below a width-dependent construction reference; B: filter-derived acquisition intervals match executed readout GD hits; C: increasing gamma with width reduces fixed-budget training residual" style="max-width: 100%;">
  <figcaption><strong>Figure 1. Gamma controls access to an attainable target during readout training.</strong> All panels use the sine-mixture target above. <strong>A:</strong> After 20,000 joint Adam updates, faint points show the median and maximum absolute hidden slope for each of five seeds; solid curves show their medians across seeds. The plotted $a_j$ denotes a learned hidden slope. Every seed's maximum remains below the dashed construction reference $\gamma=N/8$; this reference is sufficient in the construction, not an asserted necessary threshold. <strong>B:</strong> With 559 fixed-center features and zero-initialized raw readout GD, filter-derived intervals closely match executed updates to 1% relative residual. The inset divides each interval by its executed hit; connecting lines guide the eye. <strong>C:</strong> After 200,000 raw readout updates, fixed gamma 4 leaves residual near 0.42, while $\gamma=N/8$ gives 0.0034–0.0016 across widths 153–1089. Panels B/C use steps approximately equal to half the inverse largest kernel eigenvalue. Panel A motivates the scale question from joint training; the theorem concerns common-slope frozen features.</figcaption>
</figure>

[Vector PDF](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/paper_three_panel.pdf).
Panel C tests an intervention suggested by the filter: as center spacing
shrinks, increase gamma to preserve access to finer spatial scales.
Specifically, gamma proportional to inverse center spacing keeps the filter
unchanged at frequencies measured relative to that spacing. For this
construction, that means gamma proportional to width. Appendix C states
the scaling precisely.

Panel C holds the physical target fixed. It demonstrates the intervention's
finite-budget benefit; the residual near 0.42 is not an established
irreducible error floor, and constant training time across widths is not a
consequence of the filter identity. All panels measure training behavior.
The supporting [filter and target-weighted spectrum figure](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/three_panel.png)
shows the intermediate mechanism; Appendix D records the protocols and
evidence provenance.

## Appendix A. Proof of the gamma mechanism

The density has unit mass and cumulative integral
$\int_{-\infty}^a\rho_\gamma(t)\,dt=[1+\tanh(\gamma a)]/2$.
Splitting at the step's center gives

$$
\int_{\mathbb R}\rho_\gamma(x-t)\operatorname{sign}(t-c)\,dt
=\tanh(\gamma(x-c)).
$$

Insert the finite sum defining $k_{\mathrm{step}}$ into the double
integral. Each pair of steps becomes a pair of tanh features; the constant
term stays one. Bounded features and an integrable density justify the
exchanges. The steps themselves need not be integrable on the whole line.

To derive the multiplier, use Fourier convention
$\widehat\rho(\omega)=\int_{\mathbb R}\rho(t)e^{-i\omega t}\,dt$.
For a real number $a>0$, set

$$
I(a)=\int_{\mathbb R}\frac{e^{-iau}}{\cosh^2u}\,du,
\qquad F(\zeta)=\frac{e^{-ia\zeta}}{\cosh^2\zeta}.
$$

Integrate $F$ counterclockwise around the rectangle with vertices
$-R,R,R+i\pi,-R+i\pi$, where $R>0$, and let $R\to\infty$.
On either vertical side, $|e^{-ia\zeta}|\le e^{a\pi}$ and
$|\cosh(\pm R+iv)|^2=\sinh^2R+\cos^2v\ge\sinh^2R$ for
$0\le v\le\pi$. The vertical integrals therefore tend to zero.
Since $F(u+i\pi)=e^{a\pi}F(u)$ and the upper side is traversed
from right to left, the two horizontal integrals tend to
$(1-e^{a\pi})I(a)$.

The only pole inside the rectangle is the double pole at
$\zeta_0=i\pi/2$. Writing $w=\zeta-\zeta_0$ and using
$\cosh(\zeta_0+w)=i\sinh w$ gives

$$
F(\zeta_0+w)=e^{a\pi/2}
\left(-\frac1{w^2}+\frac{ia}{w}+O(1)\right),
\qquad
\operatorname{Res}_{\zeta=\zeta_0}F=ia e^{a\pi/2}.
$$

The residue theorem now yields

$$
(1-e^{a\pi})I(a)=-2\pi a e^{a\pi/2},
\qquad
I(a)=\frac{\pi a}{\sinh(\pi a/2)}.
$$

The odd sine part of the real-line integrand integrates to zero, so $I$
is real and even; the formula extends to negative $a$. At $a=0$,
$I(0)=\int_{\mathbb R}\operatorname{sech}^2u\,du=2$, also the continuous
limit of this expression. Substituting $u=\gamma t$ proves the transform
with its normalization:

$$
\widehat\rho_\gamma(\omega)
=\frac12 I(\omega/\gamma)
=\frac{\pi|\omega|/(2\gamma)}
{\sinh(\pi|\omega|/(2\gamma))}
=M_\gamma(\omega),
\qquad M_\gamma(0)=1.
$$

In particular, averaging a wave $e^{i\omega t}$ gives

$$
\int_{\mathbb R}\rho_\gamma(x-t)e^{i\omega t}\,dt
=M_\gamma(\omega)e^{i\omega x}.
$$

This makes the frequency interpretation precise in either kernel input,
without taking an ordinary integrable Fourier transform of the step
features themselves. For $z=\pi|\omega|/(2\gamma)$, expansion of
$\sinh z$ gives $M_\gamma=1-z^2/6+O(z^4)$ as $z\to0$, while
$\sinh z\sim e^z/2$ gives $M_\gamma\sim2ze^{-z}$ as $z\to\infty$.

Finally, $z/\sinh z$ decreases for $z>0$, since its derivative has
numerator $\sinh z-z\cosh z<0$. The latter expression vanishes at zero
and has derivative $-z\sinh z<0$. Reducing gamma increases $z$, proving
the cap inequality.

## Appendix B. Standard GD corollary and certified acquisition times

Let $y_i=f^\star(x_i)$ be the nonzero training target vector, and let
$\Phi_\gamma$ have rows $\phi_\gamma(x_i)^T$. Then
$K_\gamma=\Phi_\gamma\Phi_\gamma^T/m$. With zero initialization and
$0<\eta_\gamma\le1/L_\gamma$, define the relative residual and
acquisition time by

$$
E_\gamma(n)=\frac{\|(I-\eta_\gamma K_\gamma)^ny\|}{\|y\|},
\qquad
n_\epsilon(\gamma)=\min\{n\ge0:E_\gamma(n)\le\epsilon\}.
$$

Here $n$ is integer, $0<\epsilon<1$, and an empty crossing set has time
$+\infty$. This is an exact calculation from the kernel and target; it
requires no executed GD trajectory.

**Corollary (target-dependent acquisition).** Let $u_i(\gamma)$ be
orthonormal eigenvectors for the positive eigenvalues $\lambda_i(\gamma)$ of
$K_\gamma$, and set $p_i(\gamma)=|u_i(\gamma)^Ty|^2/\|y\|^2$.
Let $E_{\mathrm{floor}}(\gamma)$ be the minimum relative residual over
all readouts in this same tanh model. Then

$$
E_\gamma(n)^2=E_{\mathrm{floor}}(\gamma)^2+
\sum_i p_i(\gamma)(1-\eta_\gamma\lambda_i(\gamma))^{2n}.
$$

This follows by expanding $y$ in an orthonormal eigenbasis of the symmetric
kernel. Nullspace components remain unchanged and give the floor; positive
modes decay by the displayed factors. The weights are target energy
fractions, equivalently squared cosine alignments with unit eigenvectors.
Gamma changes both the rates and these weights through the explicit filter
in the main theorem. Eigenvalues are denoted by $\lambda_i$, independently
of the dimensionless bandwidth used in Appendix C.

For the two-point example, the target lies in the contrast mode, whose
eigenvalue is $\tanh^2(\gamma s)$, and the floor is zero. For
$0<\eta<1$ this gives

$$
n_\epsilon(\gamma)=
\left\lceil\frac{\log\epsilon}
{\log(1-\eta\tanh^2(\gamma s))}\right\rceil
\sim\frac{\log(1/\epsilon)}{\eta\gamma^2s^2}
\quad(\gamma s\to0).
$$

For the full experiment, we evaluate the filtered features using a finite
harmonic expansion. The [technical construction](gamma_factorized_readout.md)
retains the original sample/center couplings and bounds the omitted terms,
including the effect of distant transitions in its auxiliary periodic
representation. The finite expansion approximates the original nonperiodic
tanh model; it does not introduce independently trainable Fourier weights.

Write $\widehat E_\gamma(n)$ for the computed residual prediction and
$d_\gamma(n)$ for a justified margin satisfying
$|E_\gamma(n)-\widehat E_\gamma(n)|\le d_\gamma(n)$.
If integer witnesses $a,b$ satisfy

$$
\widehat E_\gamma(a)-d_\gamma(a)>\epsilon,
\qquad
\widehat E_\gamma(b)+d_\gamma(b)\le\epsilon,
\quad\text{then}\quad
a+1\le n_\epsilon(\gamma)\le b.
$$

The true residual is nonincreasing under the stated step condition, which
proves the necessary count. The second inequality directly supplies a
sufficient count. The simplest margin is $n\eta_\gamma\Delta$, where
$\Delta$ bounds the kernel approximation error and both update operators
are contractions; the implemented refinement also retains the error's
action on the target. Dividing necessary and sufficient counts across
gammas gives the reported ratio interval. All approximation matrices and
full margin formulas belong to the linked technical proof.
The margin formulas are exact-arithmetic statements. The verification
status of their numerical evaluations is recorded separately in Appendix E.

## Appendix C. Gamma proportional to width preserves relative bandwidth

For center spacing $h$, define dimensionless bandwidth $\beta=\gamma h$
and grid-relative frequency $\xi=h\omega$. Keeping $\beta>0$ fixed gives

$$
M_{\beta/h}(\xi/h)=
\frac{\pi|\xi|/(2\beta)}{\sinh(\pi|\xi|/(2\beta))},
$$

independent of $h$, with value one at $\xi=0$. Conversely, for fixed
gamma and nonzero $\xi$, as $h\to0$,

$$
M_\gamma(\xi/h)^2\sim
\frac{\pi^2\xi^2}{\gamma^2h^2}
\exp\!\left(-\frac{\pi|\xi|}{\gamma h}\right).
$$

Our construction uses $h=2/N$, $W=N+2\lceil\sqrt N\rceil+1$, and
$\gamma=N/8$, so $\beta=1/4$ and $\gamma=\Theta(W)$.
This scaling avoids a growing exponential attenuation at the dictionary's
resolution. An upper bound $\gamma=O(W)$ alone also allows fixed gamma
and does not imply this conclusion. At fixed physical frequency, fixed
gamma already gives width-independent attenuation; linear scaling instead
makes that multiplier approach one. The remaining geometry and target
weights still determine training time.

## Appendix D. Protocol, provenance, and verification scope

Figure 1 is reused from the archived study. Panel A trains all hidden slopes,
offsets, and readout parameters for 20,000 Adam updates at learning rate
$10^{-3}$, epsilon $10^{-8}$, and moments $(0.9,0.999)$.
Hidden slopes and readout weights are initialized independently uniformly
on $[-\sqrt{6/(W+1)},\sqrt{6/(W+1)}]$; biases start at zero.
Seeds 0–4 are all shown. Panels B/C use deterministic frozen dictionaries
and zero-initialized raw readout GD. Panels A/C use $N=128,256,512,1024$;
B uses $N=512$. Training grids have $m=16N+1$ equally spaced endpoint
samples on $[-1,1]$. Frozen centers have spacing $2/N$ with a halo of
$\lceil\sqrt N\rceil$ additional centers at each end.

Within each width in C, gamma is the intervention; samples, target, centers,
readout coordinates, and the 200,000-update budget are shared. The step
follows each kernel's curvature. These plotted residuals are measured;
certified times have not been computed for every width. The joint-training
observations in A do not establish a theorem about joint Adam dynamics.

The primary intervals have independent 192-bit interval-arithmetic endpoint
certificates for nominal-real tanh evaluated on the archived binary inputs
and saved steps. They do not certify each floating-point iterate or every
plotted curve. The underlying filter calculation exposes its arithmetic
sensitivity allowance separately. Across five targets and four slopes,
all 19 available executed hits lie inside the selected intervals; the
remaining run was censored. Appendix E distinguishes the four certified
primary intervals from the sixteen control-target intervals, which have
floating-point sensitivity checks without separate interval certificates.
This is training evidence, with no held-out
generalization claim and no newly executed training for this note.

The [evaluation report](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/REPORT.md)
contains control targets, the scalar-rescaling control, analytic versus
combined margins, and arithmetic sensitivity checks. The
[figure data](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/paper_figure_data.json)
record every plotted value and 23 source hashes; the
[validation record](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/paper_validation.json)
distinguishes prior implementation tests from documentation checks.
The [earlier expanded exposition](gamma_readout_paper_note.md) preserves
the worked matrix construction. Reproduce the existing figure from the
repository root with:

```bash
MPLCONFIGDIR=/tmp/gamma-paper-mpl \
python -m experiments.expD36_frozen_gamma_probe.paper_figure
```

## Appendix E. Complete five-target acquisition results

The five target functions are defined exactly in Table 3. Every row below
uses the same 559-neuron geometry, 8,193 training samples, zero raw readout
initialization, half empirical mean squared error, and saved step for its
gamma, approximately $0.5/L_\gamma$. The tolerance is 1% relative residual.
Each interval is the selected combined filter prediction: the analytic
approximation margin and the target-action refinement are combined as
described in the technical proof. Harmonic resolutions are selected from
the calculated bounds, without fitting endpoints to executed GD hits.

The verification column describes the numerical evidence for each interval:

- **Interval-certified:** an independent 192-bit interval-arithmetic audit
  certifies the excluded and sufficient iterates for nominal-real tanh on
  the archived binary inputs and step. This applies to the sine mixture.
- **FP64-checked:** the prediction has double-precision reference and
  arithmetic-sensitivity checks, including a disclosed heuristic allowance
  for transcendental evaluations. No independent interval-arithmetic
  certificate was computed for these targets. A zero-width numerical
  interval alone does not establish a certified exact first hit.

**Table 4. All twenty target–gamma predictions and their executed checks.**
Times count ordinary readout GD updates to 1% training residual. All 19
available first hits lie inside the selected intervals. The quadratic at
gamma 12 is censored and is not counted as an executed crossing.

| Target | Gamma | Predicted interval | Executed first hit | Verification |
|---|---:|---:|---:|---|
| Sine mixture | 8 | 15,784,048–15,812,623 | 15,798,313 | Interval-certified |
| Sine mixture | 12 | 186,057–186,058 | 186,057 | Interval-certified |
| Sine mixture | 16 | 61,792–61,792 | 61,792 | Interval-certified |
| Sine mixture | 64 | 16,013–16,013 | 16,013 | Interval-certified |
| Exponential of sine | 8 | 426,231–426,249 | 426,240 | FP64-checked |
| Exponential of sine | 12 | 34,752–34,753 | 34,753 | FP64-checked |
| Exponential of sine | 16 | 11,961–11,961 | 11,961 | FP64-checked |
| Exponential of sine | 64 | 2,394–2,394 | 2,394 | FP64-checked |
| Runge | 8 | 26,104–26,104 | 26,104 | FP64-checked |
| Runge | 12 | 3,566–3,566 | 3,566 | FP64-checked |
| Runge | 16 | 1,606–1,606 | 1,606 | FP64-checked |
| Runge | 64 | 578–578 | 578 | FP64-checked |
| Quadratic | 8 | 879,431–879,555 | 879,493 | FP64-checked |
| Quadratic | 12 | 269,286–269,299 | Censored at 200,000 | FP64-checked |
| Quadratic | 16 | 119,631–119,633 | 119,632 | FP64-checked |
| Quadratic | 64 | 15,119–15,119 | 15,119 | FP64-checked |
| Single sine | 8 | 7,466–7,466 | 7,466 | FP64-checked |
| Single sine | 12 | 5,263–5,264 | 5,263 | FP64-checked |
| Single sine | 16 | 4,289–4,289 | 4,289 | FP64-checked |
| Single sine | 64 | 2,233–2,233 | 2,233 | FP64-checked |

The censored run stopped before acquisition. Its separate original-kernel
spectral forecast is 269,292 updates, inside the filter-derived interval,
but neither forecast is an executed hit. The other nineteen rows compare
predictions against actual GD crossings. These checks assess the
target-dependent timing calculation on the archived training problems;
they do not extend the theorem to a universal learning-time ordering over
targets or dictionaries.

The [numerical summary](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/summary.json)
contains all twenty selected intervals, their harmonic resolutions, and
the executed-hit records. The
[interval audit](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/interval_audit.json)
supports the four certification labels. The
[evaluation report](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/REPORT.md)
documents sensitivity checks and unresolved individual approximation
resolutions as well as the selected results.

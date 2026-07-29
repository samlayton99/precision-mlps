# Iteration 11, explained from scratch

**Status: mechanisms and numbers final; conclusions unsigned pending Sam.**

This document assumes you have not been following the campaign. It builds the method up from the model equations, with every step of the math written out, and then reports what happened. Nothing is referenced without being defined here first.

The short version, so you know where we are going: the network is *exactly linear in some of its parameters and curved in the others*. Linear parameters can be solved in closed form to machine precision; curved ones cannot. So the optimizer measures which parameters are which, runs ordinary Adam on everything, and additionally solves the linear ones exactly every 200 steps. On our test problems this reaches machine precision on all 30 cells. It also fails in two specific ways that turned out to be more informative than the success, and those are explained too.

---

# 1. The problem

## 1.1 The network and the loss

We approximate a function $f^\star:[-1,1]\to\mathbb{R}$ with a one-hidden-layer $\tanh$ network of width $W$:

$$f(x;\theta) \;=\; \sum_{k=1}^{W} v_k \,\tanh(w_k x + b_k) \;+\; c_0 .$$

The parameters are

$$\theta \;=\; (\underbrace{w_1,\dots,w_W}_{\text{slopes}},\ \underbrace{b_1,\dots,b_W}_{\text{offsets}},\ \underbrace{v_1,\dots,v_W}_{\text{output weights}},\ c_0) \;\in\; \mathbb{R}^{m}, \qquad m = 3W+1 .$$

Each hidden unit $\tanh(w_k x + b_k)$ is a smooth step. Its *centre* is where the step happens, at $x = -b_k/w_k$, and its *sharpness* is $w_k$. So $(w,b)$ says where the steps are and how sharp they are -- call this the **geometry** -- and $(v, c_0)$ says how much of each step to add up -- call this the **readout**.

We are given training inputs $x_1,\dots,x_n$ (here $n = 2003$ equally spaced points) with targets $y_i = f^\star(x_i)$. Define the **residual vector** and the loss:

$$r(\theta) \in \mathbb{R}^n, \qquad r_i(\theta) = f(x_i;\theta) - y_i, \qquad \mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^n r_i(\theta)^2 = \frac{1}{n}\|r(\theta)\|_2^2 .$$

We report error as **relative $L_2$** on a finer grid of 4001 points not used for training:

$$\text{rel } L_2 \;=\; \frac{\bigl(\sum_j (f(\tilde x_j;\theta) - f^\star(\tilde x_j))^2\bigr)^{1/2}}{\bigl(\sum_j f^\star(\tilde x_j)^2\bigr)^{1/2}} .$$

## 1.2 What "machine precision" means and why this is hard

Double-precision floating point stores about 16 significant decimal digits; the spacing between representable numbers near 1 is $\epsilon_{\text{mach}} = 2.22\times10^{-16}$. "Machine precision" here means relative $L_2$ around $10^{-14}$ to $10^{-15}$ -- essentially every digit the arithmetic can hold.

The motivating fact for this whole project: if you *construct* this network by hand using the quasi-interpolant theory, you get $\sim10^{-15}$. If you *train* it with a standard optimizer, you stall at $10^{-3}$ to $10^{-10}$. Training is losing five to twelve digits that provably exist in the architecture. Iteration 11 is an attempt to close that.

---

# 2. The structural fact everything is built on

## 2.1 The derivatives, written out

Differentiate $f$ with respect to each kind of parameter. Write $t_k = \tanh(w_k x + b_k)$ and recall $\tanh' = 1 - \tanh^2$.

$$\frac{\partial f}{\partial v_k} = t_k, \qquad\qquad \frac{\partial f}{\partial c_0} = 1,$$

$$\frac{\partial f}{\partial b_k} = v_k (1 - t_k^2), \qquad \frac{\partial f}{\partial w_k} = v_k\, x\, (1 - t_k^2) .$$

Now the second derivatives, which are the interesting ones:

$$\boxed{\frac{\partial^2 f}{\partial v_k^2} = 0}, \qquad \frac{\partial^2 f}{\partial c_0^2} = 0,$$

$$\frac{\partial^2 f}{\partial b_k^2} = -2 v_k\, t_k (1 - t_k^2), \qquad \frac{\partial^2 f}{\partial w_k^2} = -2 v_k\, x^2\, t_k (1 - t_k^2) .$$

The readout second derivatives are **exactly zero**, not small -- $f$ is a linear function of $(v, c_0)$, full stop. The geometry second derivatives are not zero. That asymmetry is the entire basis of the method.

Note also, for later: the geometry second derivatives are **proportional to $v_k$**. If a unit's output weight is zero, the network is momentarily flat in that unit's geometry too. This will matter enormously in Section 4.3.

## 2.2 The readout is a least-squares problem

Freeze the geometry $(w,b)$. Then each hidden unit gives a fixed function of $x$, so define the **feature matrix**

$$\Phi \in \mathbb{R}^{n\times W}, \qquad \Phi_{ik} = \tanh(w_k x_i + b_k),$$

append a column of ones for the constant $c_0$, and collect the readout into one vector:

$$A = [\,\Phi \;\; \mathbf{1}\,] \in \mathbb{R}^{n\times(W+1)}, \qquad a = \begin{bmatrix} v \\ c_0\end{bmatrix} \in \mathbb{R}^{W+1} .$$

Then the network's predictions at the training points are exactly $A a$, and

$$\mathcal{L} = \tfrac{1}{n}\|A a - y\|_2^2 .$$

This is an ordinary linear least-squares problem. Its solution is available in closed form, and a good numerical routine solves it to machine precision in one shot. **Given the right geometry, the readout is not an optimization problem at all -- it is a linear solve.**

At $N=256$ this matrix is $2003 \times 462$.

## 2.3 Then why is this hard?

Two reasons, and the method addresses both.

First, we do not get to assume which parameters are the readout. Hard-coding "solve the last layer" would work here and fail on any architecture where the useful linear block is somewhere else, or where there isn't one. The optimizer has to *discover* the split from measurements. Sections 4 and 5 are that discovery.

Second, the geometry still has to be learned by ordinary optimization, and moving the geometry invalidates the readout solve. That coupling is the hard part, and Section 10 shows it is the binding constraint.

---

# 3. Why gradient descent cannot solve even the easy part

It is worth being precise about this, because "just train the readout longer" is the obvious objection.

## 3.1 The spectrum of $A$

Any matrix has a singular value decomposition $A = U\Sigma V^\top$, where $\Sigma$ is diagonal with entries $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. Think of $\sigma_j$ as "how strongly the data sees direction $j$ of the readout". Measured for our $2003\times462$ matrix at $N=256$:

$$\sigma_1 = 7.7\times10^{2}, \qquad \frac{\sigma_{100}}{\sigma_1} = 5\times10^{-5}, \qquad \frac{\sigma_{400}}{\sigma_1} = 3\times10^{-19}, \qquad \sigma_{\min} = 0 \text{ (numerically)} .$$

The singular values fall off a cliff. Some directions of the readout are essentially invisible to the data.

## 3.2 The rate argument

Run gradient descent on $\tfrac1n\|Aa - y\|^2$ with the largest stable step size. Decompose the error into singular directions. The component along direction $j$ shrinks by a factor $\bigl(1 - \sigma_j^2/\sigma_1^2\bigr)$ per step. So the number of steps needed to fix direction $j$ is roughly

$$t_j \;\sim\; \left(\frac{\sigma_1}{\sigma_j}\right)^{2} .$$

Putting numbers in:

| $\sigma_j/\sigma_1$ | steps needed |
|---|---|
| $10^{-3}$ | $10^{6}$ |
| $10^{-6}$ | $10^{12}$ |
| $10^{-10}$ | $10^{20}$ |

To reach $10^{-13}$ relative error you need the directions down around $\sigma_j/\sigma_1 \sim 10^{-11}$, which is $10^{22}$ steps. This is not a tuning problem or a patience problem. It is arithmetic. Momentum improves the exponent from $\kappa$ to $\sqrt{\kappa}$ and still leaves you at $10^{11}$ steps.

Meanwhile a direct SVD-based solve gets the same answer to machine precision immediately, because it never iterates -- it inverts each direction once. **This is the whole reason for having an exact solve inside the optimizer.**

## 3.3 One numerical trap worth naming

The tempting way to solve least squares is to form the *normal equations* $A^\top A\, a = A^\top y$, which is a small square system. Do not. Squaring a matrix squares its singular values, so it squares the condition number:

$$\kappa(A^\top A) = \kappa(A)^2 .$$

With $\kappa(A) \sim 10^{8}$ you get $\kappa(A^\top A) \sim 10^{16}$, which exceeds what double precision can represent, so the answer has no correct digits. This repo measured a seven-decade loss from exactly this. We work with $A$ directly via its SVD throughout.

---

# 4. Discovering which parameters are linear: the probe

We need a test that, given only the ability to evaluate the network, says "the model is linear in this parameter" or "it is not". No knowledge of the architecture.

## 4.1 The idea: second differences

For a smooth scalar function $g(t)$, Taylor expansion gives

$$g(\varepsilon) - 2g(0) + g(-\varepsilon) \;=\; \varepsilon^2 g''(0) + \frac{\varepsilon^4}{12} g''''(0) + \dots$$

so

$$\frac{g(\varepsilon) - 2g(0) + g(-\varepsilon)}{\varepsilon^2} \;\approx\; g''(0) .$$

Apply this along coordinate $i$ of the parameter vector, using the whole vector of predictions rather than a scalar. Let $e_i$ be the $i$-th unit vector and let $f(\theta) \in \mathbb{R}^n$ denote the predictions at all training points. Define

$$q_i \;=\; \frac{\bigl\| f(\theta + \varepsilon_i e_i) - 2 f(\theta) + f(\theta - \varepsilon_i e_i)\bigr\|_2}{\varepsilon_i^{2}}\; s_i^2, \qquad s_i = \max(|\theta_i|, 1), \quad \varepsilon_i = 10^{-3} s_i .$$

Three forwards per coordinate, no derivatives, no knowledge of the model.

Because $\varepsilon_i = 10^{-3}s_i$, the $s_i^2$ factor cancels the $\varepsilon_i^2$ in the denominator except for a constant, and the whole thing reduces to

$$q_i \;\approx\; s_i^2 \left\|\frac{\partial^2 f}{\partial \theta_i^2}\right\|_2 .$$

Read that as: *how much curvature does the model show when this parameter moves by a fixed fraction of its own size*. The $s_i^2$ makes the number comparable across parameters of wildly different magnitude, which matters because in our networks $w_k \approx N/8$ can be in the hundreds while $v_k$ is order one.

## 4.2 What it returns for our model

Substituting the derivatives from 2.1:

- For readout coordinates, $\partial^2 f/\partial v_k^2 = 0$ exactly, so $q_k$ is zero up to floating-point rounding.
- For geometry coordinates, $\partial^2 f/\partial b_k^2 = -2v_k t_k(1-t_k^2) \ne 0$ in general, so $q$ is genuinely nonzero.

Measured at $N=256$ on a trained-ish state: geometry $q \approx 6\times10^{2}$, readout $q \approx 4\times10^{-8}$. Ten orders of separation. The test works, and it never needed to know what a "layer" is.

## 4.3 Why the probe is run at two points -- with the measurement that forced it

Look again at

$$\frac{\partial^2 f}{\partial b_k^2} = -2 v_k\, t_k (1-t_k^2) .$$

This is proportional to $v_k$. Our standard initialization sets the entire readout to zero, $v = 0$. At that point *every geometry second derivative is exactly zero too*, and the probe cannot tell geometry from readout at all.

This is not hypothetical. Probing the actual initialization at a single point:

| probe | max $q$ on geometry | max $q$ on readout |
|---|---|---|
| single point, at $\theta_0$ | $0.000$ | $0.000$ |
| two points (as shipped) | $6.17\times10^{2}$ | $4.04\times10^{-8}$ |

At a single point everything reads as perfectly linear, so everything would be admitted for exact solving, which is nonsense. The fix is to also probe at one randomly perturbed point $\theta_0 + 0.1\,s\odot(\pm1)$ and take the elementwise maximum. A genuinely linear parameter has zero second derivative at *every* point; a curved one only looks flat at special points. Ten orders of separation reappear immediately.

## 4.4 The rounding floor

$q_i$ is built from a difference of three nearly-equal numbers, so it is exposed to cancellation. Each evaluation of $f$ carries absolute rounding error about $\epsilon_{\text{mach}}\|f\|$, and the combination $f_+ - 2f_0 + f_-$ accumulates roughly four such errors. Dividing by $\varepsilon^2 = (10^{-3}s)^2$ amplifies them:

$$q_{\text{noise}} \;\approx\; \frac{4\,\epsilon_{\text{mach}}\,\|f\|}{(10^{-3})^{2}} \;=\; 4\times10^{6}\,\epsilon_{\text{mach}}\|f\| .$$

Any $q_i$ below $100\,q_{\text{noise}}$ is set to exactly zero and treated as "measured linear". Without this, the next step (normalizing by the maximum) would take pure rounding noise and promote it to a meaningful-looking value on a problem that is genuinely linear.

## 4.5 Normalizing

Finally $\hat q_i = q_i / \max_j q_j$, so the score is dimensionless and lives in $[0,1]$. A parameter is *linear enough* when $\hat q_i < 10^{-8}$. Given the measured ten-order separation, anything in the range $10^{-6}$ to $10^{-10}$ would give the same answer; this is not a delicate constant.

---

# 5. The other half of the test: is the parameter informed?

## 5.1 A small gradient means two opposite things

Suppose $\partial \mathcal{L}/\partial\theta_i \approx 0$. That can mean:

1. this parameter is already at its best value, or
2. this parameter has no effect on the loss at all, so the data says nothing about it.

These are opposites, and the probe cannot distinguish them either -- a parameter with no effect is trivially linear.

## 5.2 The concrete case: saturated units

Our networks place some hidden units deliberately outside the data range (a "halo"). For such a unit, $|w_k x + b_k| > 20$ for every training point, so $\tanh$ has saturated to $\pm 1$ and its derivative $1 - t_k^2$ underflows to **exactly zero** in double precision. The consequences:

$$\frac{\partial f}{\partial w_k} = v_k x (1-t_k^2) = 0, \qquad \frac{\partial^2 f}{\partial w_k^2} = 0 .$$

So these parameters are perfectly linear *and* perfectly invisible. Measured at $N=512$: all 642 such coordinates had gradient exactly $0.0$, against a readout whose typical gradient magnitude was $1.7\times10^{-2}$.

That separation is absolute, so the test needs no threshold at all:

$$\text{informed}_i \;\iff\; \text{info}_i > 0, \qquad \text{info}_i = \hat v_i \;\text{(Adam's second moment, already stored)} .$$

Adam maintains $\hat v_i$ as a running average of squared gradients, so this costs nothing. At the very first probe, before any moments exist, $|g_i|$ is used instead.

## 5.3 The membership rule

A parameter is solved exactly only if it is **both linear and informed**, with a deadband so membership cannot chatter:

$$i \in E \iff \hat q_i < 10^{-8} \;\text{ and }\; \text{info}_i > 0, \qquad\text{removed when } \hat q_i > 10^{-7} \;\text{ or }\; \text{info}_i = 0 .$$

Removal is not freezing. A removed parameter simply goes back to being handled by Adam like everything else. Nothing is ever locked, and a parameter that saturates later leaves on its own.

Call $E$ the **exploit set** and $B = |E|$.

---

# 6. Solving the certified parameters exactly

## 6.1 The subproblem

Since the model is linear in the parameters of $E$, the change in predictions from changing them is exactly linear:

$$f(\theta + \delta) - f(\theta) = J_E\, \delta \qquad\text{for any } \delta \text{ supported on } E,$$

where $J_E \in \mathbb{R}^{n\times B}$ has columns

$$(J_E)_{\cdot j} = \frac{\partial f}{\partial \theta_{i_j}} \in \mathbb{R}^n .$$

We want the new residual to be as small as possible, so we solve

$$\delta^\star = \arg\min_{\delta}\ \|J_E \delta + r\|_2^2, \qquad \theta_E \leftarrow \theta_E + \delta^\star .$$

Worth pausing on what $J_E$ is in our case. From 2.1, $\partial f/\partial v_k = \tanh(w_k x + b_k)$, which is precisely column $k$ of the feature matrix $\Phi$, and $\partial f/\partial c_0 = 1$ is the appended ones column. **So $J_E$ is exactly the matrix $A = [\Phi\ \ \mathbf{1}]$ of Section 2.2**, and the exploit solve is exactly the classical readout least-squares solve. The difference from simply hard-coding that solve is that here the columns were *discovered* by measurement, and the same code finds the corresponding block in a two-layer network without being told.

## 6.2 Getting the columns

Column $j$ is obtained by one forward-mode directional derivative (a "JVP"), which evaluates $\tfrac{d}{dt}f(\theta + t e_{i_j})|_{t=0}$ at the cost of about one forward pass. So building $J_E$ costs $B$ forward-equivalents. That is the dominant cost of a solve, and it is why solves happen every 200 steps rather than every step.

## 6.3 Solving it stably

Compute the SVD $J_E = U\Sigma V^\top$ and form

$$\delta^\star = V\,\Sigma^{+}\,U^\top(-r), \qquad (\Sigma^+)_{jj} = \begin{cases} 1/\sigma_j & \sigma_j > 10^{-15}\sigma_1\\ 0 & \text{otherwise.}\end{cases}$$

Two things are happening here.

**Truncation.** Directions with $\sigma_j$ below the cutoff are directions the data cannot resolve. Dividing by them would amplify rounding noise by $1/\sigma_j$, producing enormous meaningless parameter changes. Setting those to zero instead is the standard fix.

**Minimum norm.** Because $A$ has a null space, many different readouts give the same predictions. The formula above returns the smallest such $\delta$. This is the canonical choice, and it turns out to matter for a reason we could not have anticipated: Section 10 shows the damage done by subsequent geometry motion is proportional to $\|v\|$, so keeping $\|v\|$ small is directly protective. Measured after a solve at $N=256$: $\|v\| = 0.371$, and the resulting error is $1.14\times10^{-14}$.

## 6.4 Refusing to solve what cannot be seen

Even after the "informed" test, some columns can be very nearly null. Before solving, drop any column with

$$\|(J_E)_{\cdot j}\| < 10^{-8}\,\max_l \|(J_E)_{\cdot l}\| .$$

The measured gap here is again enormous -- live columns are order 1, blind columns are $10^{-14}$ or less. Skipping this filter was a real failure: the solve produced a geometry correction of norm 29 and drove one test case to $2\times10^{2}$. Dropped parameters simply receive no correction that round.

---

# 7. The complete algorithm

```
initialize theta; Adam moments m = v = 0
r_ref  <- residual(theta)
info   <- |gradient(theta)|                  # one extra backward, once
E      <- certify(theta, info)               # Sections 4 and 5

for t = 1, 2, 3, ...:

    # ---- base regime: ordinary Adam on EVERY parameter ----
    r, g  <- residual and gradient at theta          # one fused forward+backward
    m     <- 0.9 m + 0.1 g
    v     <- 0.999 v + 0.001 g^2
    theta <- theta - lr * m_hat / (sqrt(v_hat) + 1e-8)
    info  <- v_hat                                   # free, reused next probe

    # ---- every 200 steps: re-certify if the model actually moved ----
    if t mod 200 == 0:
        drift <- || residual(theta) - r_ref ||
        r_ref <- residual(theta)
        if drift > 10 * eps_mach * ||y||  and  backoff elapsed:
            E <- certify(theta, info)
            backoff <- 1 if E changed else min(2*backoff, 8)

    # ---- exploit regime: exact solve on the certified set ----
    if t == 1 or t mod 200 == 0:
        r     <- residual(theta)                     # fresh
        J_E   <- [ JVP(theta, e_i) for i in E ]      # B forward-equivalents
        drop columns of J_E below 1e-8 of the largest
        delta <- truncated-SVD solve of  min || J_E delta + r ||
        theta_E <- theta_E + delta
```

That is the whole method. There is no tether, no trust region, no line search, no learning-rate schedule, no mode switch, and no control decision that ever compares two loss values.

Two design points worth stating explicitly, because both were wrong in earlier drafts and had to be fixed by measurement:

- **Adam runs on every parameter, including the certified ones.** Membership decides only where the *extra* solve is applied. An earlier version excluded certified parameters from Adam, which left the readout frozen at a stale optimum for 200 steps while the features moved underneath it; that cost about $4\times$ on real data.
- **The residual is recomputed fresh, never carried as state.** Earlier iterations in this campaign carried the residual to avoid re-rolling rounding noise, which is correct for an iterative method. It is wrong here: it feeds Adam a gradient derived from a 200-step-old linearization. A one-shot direct solve is unbothered by a slightly noisy right-hand side.

## Cost

Per ordinary step: one forward and one backward. Same as Adam. Per refresh: one forward for the drift check; when the certificate fires, $2(2m+1)$ forwards for the dense probe; and $B$ JVPs plus a $B\times B$ SVD for the solve. Amortized over 200 steps this is roughly one to two extra forward-equivalents per step.

The dense per-parameter probe is a diagnostic that is affordable on small problems and is what the dial movies visualize. The deployable version samples a handful of coordinates per named tensor, costing a number of forwards proportional to the number of tensors rather than parameters.

## Memory, and the budget in every setup

The solver's working state is $B \times B$, so $B$ is capped at $B \le \sqrt{C m}$ with $C = 1024$, keeping the state proportional to the parameter count. Here is what that cap actually was in each experiment, against what was actually used:

| setup | $m$ (params) | linear block | $b_{\text{cap}}$ | $B$ used | cap binding? |
|---|---|---|---|---|---|
| 1D, $N{=}32$ | 520 | 174 | 729 | 293-444 | no ($b_{\text{cap}} > m$) |
| 1D, $N{=}64$ | 616 | 206 | 794 | 323-522 | no |
| 1D, $N{=}128$ | 808 | 270 | 909 | 380-728 | no |
| 1D, $N{=}256$ | 1384 | 462 | 1190 | 580-594 | no |
| 1D, $N{=}512$ | 2764 | 922 | 1682 | 1032-1043 | no |
| batching, $N{=}256$ | 1384 | 462 | 1190 | ~583 | no |
| depth-2, $32{\times}32$ | 1153 | 33 | 1086 | 33 | no |
| depth-2, $64{\times}64$ | 4353 | 65 | 2111 | 65 | no |
| *(historical)* 1D $N{=}512$, $C{=}256$ | 2764 | 922 | **841** | 841 | **YES -- truncated** |

Reading the table:

- "Linear block" is the readout $(v, c_0)$, of size $W+1$ in the one-layer runs and $h+1$ in the two-layer ones.
- $B$ used exceeds the linear block in the one-layer runs because of the late saturated-geometry admissions described in 8.1; those columns are all dropped at solve time, so the number that does arithmetic is the linear block.
- The ranges span the six target functions -- how many units saturate during training is target-dependent.
- At $N{=}32$ the cap ($729$) is larger than the entire parameter vector ($520$), so it is not merely slack, it is inert.
- The final row is the configuration that produced the twelve-order failure of Section 9.5, kept because it is the only case where the cap did any work at all.

So on every shipped run the memory budget never binds. That is worth stating plainly rather than presenting the $\sqrt{m}$ rule as if it were being stress-tested: on these problems it is not. Section 11 explains why this architecture cannot stress-test it sensibly.

---

# 8. What actually worked

All runs start from the constructed geometry with the readout set to zero, with every parameter free to move. "Floor" means a direct truncated-SVD solve on that frozen initial geometry -- the best any method could do without moving the geometry at all.

## 8.1 The main result

**All 30 cells reach or beat the floor**, across six target functions and five widths. The table gives the ratio (achieved error) / (floor); values at or below 1 mean the optimizer matched or beat a direct solve, which it can do because it also improves the geometry slightly.

| target | $N{=}32$ | $64$ | $128$ | $256$ | $512$ |
|---|---|---|---|---|---|
| `sine` | 0.42 | 0.70 | 0.65 | 0.69 | 0.75 |
| `sine_8pi` | 1.01 | 0.74 | 0.72 | 0.88 | 0.96 |
| `runge` | 1.03 | 0.75 | 0.79 | 0.76 | 0.84 |
| `sine_mixture` | 1.03 | 1.00 | 0.76 | 0.96 | 0.86 |
| `exp` | 0.43 | 0.24 | 1.37 | 0.62 | 0.13 |
| `abs_cubed` | 1.01 | 1.00 | 1.00 | 0.99 | 0.98 |

In absolute terms the deep cells are at machine precision: `exp` at $N{=}512$ reaches $2.1\times10^{-15}$; `sine` at $N{=}512$, $1.3\times10^{-14}$. The cells that are not at $10^{-14}$ are limited by the target, not the optimizer -- `abs_cubed` is only twice differentiable and its own floor is $10^{-6}$ to $10^{-10}$, and the optimizer sits exactly on it.

One cell deserves a callout. `sine_8pi` at small width stalled far above its floor in *every* previous iteration of this campaign, and that had been tentatively blamed on the geometry being inadequate. It reaches $1.4\times10^{-13}$ here, below its floor. It was the optimizer, not the geometry.

### What the certificate actually selects, and an honest word about how impressive that is

At $N=256$ ($W=461$, $m=1384$, readout $=462$):

| step | $B$ | readout in $E$ | geometry in $E$ |
|---|---|---|---|
| 0 | 462 | 462 | 0 |
| 160 | 462 | 462 | 0 |
| 3000 | 583 | 462 | 121 |

Two things to take from this, one of them deflationary.

**The readout is always admitted, and that is not a discovery.** For this model class the readout is *exactly* linear at every point in parameter space -- $\partial^2 f/\partial v_k^2 = 0$ identically, as Section 2.1 shows -- so it certifies at step 0 and never stops certifying. All 462 coordinates are in $E$ from the first probe to the last, and all 462 survive the observability filter. On an architecture whose last layer is linear and whose loss is squared error, the probe is *confirming a fact that follows from the architecture*, not finding something unknown. It would be overselling this to call the one-layer result a discovery.

What the certificate does earn on these problems is narrower and still worth something: it reaches that answer without being told anything about layers, it correctly **refuses** everything else (including, on the two-layer network, the second hidden layer $W_2, b_2$, which is closer to the output than the block it does admit), and it refuses the invisible units. The genuine test of discovery is a problem whose linear block is *not* the last layer, and that has not been run -- it is the sharpest open item in Section 11.

**The 121 late geometry admissions are a real wrinkle.** By step 3000, 121 geometry coordinates have entered $E$. These are units that saturated *during* training: their probe reads zero, and the "informed" test passes only because Adam's second moment is an exponential moving average that decays by $0.999$ per step and so never quite reaches zero -- it remembers gradients from before the unit went blind. So the informed test, as implemented, has a memory and is slow to notice a parameter going dark.

That would be a bug except that the observability filter of Section 6.4 catches all of them: measured at the final step, **all 121 of those geometry columns are dropped as unobservable, and all 462 readout columns are kept**. The two guards turn out to cover different failure modes -- the informed test catches units that are blind from the start, the column filter catches units that go blind later -- and the overlap is what keeps the solve clean. That redundancy was not designed; it is worth recording as luck that held.

## 8.2 It also works on minibatches

Pool of 8012 points, random batches every step, **no accumulation across batches of any kind**:

| target | full pool | 3/4 | 1/2 | 1/4 | 1/16 | floor |
|---|---|---|---|---|---|---|
| `sine` | $3.4\times10^{-14}$ | $1.4\times10^{-14}$ | $1.1\times10^{-14}$ | $2.1\times10^{-14}$ | $1.0\times10^{-11}$ | $3.0\times10^{-14}$ |
| `sine_8pi` | $1.6\times10^{-13}$ | $3.9\times10^{-14}$ | $3.2\times10^{-14}$ | $4.9\times10^{-14}$ | $9.2\times10^{-11}$ | $1.5\times10^{-13}$ |
| `runge` | $2.2\times10^{-14}$ | $1.1\times10^{-14}$ | $1.2\times10^{-14}$ | $8.2\times10^{-15}$ | $9.5\times10^{-13}$ | $3.3\times10^{-14}$ |
| `exp` | $7.0\times10^{-15}$ | $7.6\times10^{-15}$ | $8.2\times10^{-15}$ | $8.8\times10^{-15}$ | $2.1\times10^{-13}$ | $2.8\times10^{-14}$ |

At or below the floor from full batch down to quarter batches, then a gentle rise -- a smooth decay, not a cliff. A single random 2003-point batch per step also matches a full-pool step on all four targets.

This is worth flagging because earlier work in this campaign concluded that minibatching imposes a $1/\sqrt{b}$ accuracy wall that no batch-local method can cross. That conclusion was about *carrying deep-spectrum iterative state across resampled batches*, where the fine directions get scrambled from one batch to the next. This method carries no such state: each solve is a self-contained least-squares problem on the current batch's rows, and the certificate reads a fixed set of gauge rows so sampling noise never enters the membership decision. There is nothing for resampling to scramble.

The gentle degradation at $1/16$ has a natural reading: the exploit set has $B = 462$ unknowns, so a 2003-row batch is comfortably overdetermined ($b/B \approx 4.3$) while a 501-row batch is nearly square ($b/B \approx 1.1$) and the solve starts fitting the batch rather than the function. Consistent across all four targets, though not tested by varying $B$ independently, so treat it as the natural explanation rather than a law.

## 8.3 The certificate generalizes to a second architecture

On a two-hidden-layer network from random initialization, the certificate admits exactly $(v, c_0)$ -- 65 of 4353 parameters -- and leaves $W_1, b_1, W_2, b_2$ entirely to Adam. That is the right answer, and it was found by measurement rather than assumed. This was the guard against the whole approach being a dressed-up "solve the last layer" rule.

## 8.4 The iterative alternative loses

Sam asked for a head-to-head against an iterative solver, so a second arm was built: identical certificate, but the exploit set is optimized by dense BFGS with exact step lengths instead of a direct solve. It never reaches a floor on any of the 30 cells, typically falling 3 to 9 orders short. The direct solve wins outright, for the reason in Section 3: iterating cannot recover directions whose singular values are $10^{-11}$ of the largest, no matter how good the iteration is, whereas a factorization inverts them once.

---

# 9. What broke, and the single law that explains it

## 9.1 Adam's step does not get smaller as you approach the answer

Adam's update is

$$\Delta\theta_i = -\eta\,\frac{\hat m_i}{\sqrt{\hat v_i} + \epsilon}, \qquad \epsilon = 10^{-8}.$$

If the gradient is consistent, $\hat m_i \approx g_i$ and $\sqrt{\hat v_i} \approx |g_i|$, so

$$\Delta\theta_i \approx -\eta\,\frac{g_i}{|g_i|} = -\eta\,\mathrm{sign}(g_i).$$

The step has size $\eta$ **regardless of how small the gradient is**. Halving the error does not halve the step. This is fine for ordinary training and fatal next to an exact solve.

## 9.2 The coupling law

After a solve, the readout is optimal *for the current features*. Adam then moves the geometry by about $\eta$ per coordinate. Write the perturbed features as $\phi_k + \Delta\phi_k$. The prediction changes by

$$\Delta f = \sum_k v_k\,\Delta\phi_k \quad\Longrightarrow\quad \|\Delta f\| \;\lesssim\; \|v\|\cdot\|\Delta\phi\| \;\sim\; \|v\|\,\eta .$$

So **one base step re-injects error proportional to $\|v\|\,\eta$**. Measured over six decades of $\eta$ and five decades of $\|v\|$, the data lie on that line:

| | $\|v\|$ after solve | error at the solved point | after one step at $\eta=10^{-3}$ |
|---|---|---|---|
| one-layer, constructed geometry | $0.56$ | $1.5\times10^{-14}$ | $4.6\times10^{-4}$ |
| two-layer, random features | $1.0\times10^{5}$ | $6.2\times10^{-11}$ | $1.5\times10^{2}$ |

One law explains three separate observations:

- the **sawtooth** in the one-layer loss curves -- each solve reaches the floor, the following 200 Adam steps walk it back up, the next solve fixes it again;
- the **two-layer catastrophe** -- random features are ill-conditioned, so the minimum-norm readout is huge ($\|v\| = 10^5$), and the same $\eta$ produces $10^5$ times more damage; the solve at iteration 1 reaches $6\times10^{-11}$ and iteration 2 is at $10^{4}$;
- why the **"tether"** used in earlier iterations of this campaign was load-bearing: it slowed exactly the nonlinear coordinates, which is precisely what keeps $\|v\|\eta$ small. Removing it entirely was too aggressive.

It also explains why minimum-norm truncation in Section 6.3 is not merely a numerical nicety -- it is what keeps $\|v\|$ small, and therefore what keeps the coupling damage small.

## 9.3 There is a clean escape, and it already happens by accident

Look again at Adam's denominator. Once $\sqrt{\hat v_i} \ll \epsilon$, the update becomes

$$\Delta\theta_i \approx -\frac{\eta}{\epsilon}\,\hat m_i,$$

which *is* proportional to the gradient and therefore does vanish as the problem is solved. So Adam has a settling regime; it engages when gradients fall below about $\epsilon = 10^{-8}$.

This is visible in the data. In the `exp` panel of the $N{=}256$ loss curves the error steps down one decade per solve and then goes flat *on the floor* from about iteration 1200 onward -- it converged and stayed. The same happens for `sine` on the two-layer bench at width 64, flat at $5.6\times10^{-11}$ from iteration 1100. Other cells never get there and sawtooth forever.

The fix therefore is not exotic: make the base step vanish as the certified block's residual vanishes, either by tying $\eta$ to the measured residual or by choosing $\epsilon$ so this regime engages at the right scale. One knob, with a measured target rather than a guessed constant. That is the main open item.

## 9.4 When solving early actively hurts

On the two-layer bench the split is sharp:

| target | width | Adam alone | this method | Adam's final features, refit | this method's features, refit |
|---|---|---|---|---|---|
| `sine` | 64 | $4.5\times10^{-3}$ | $\mathbf{5.6\times10^{-11}}$ | $2.4\times10^{-10}$ | $5.8\times10^{-11}$ |
| `runge` | 64 | $1.3\times10^{-3}$ | $5.5\times10^{-3}$ | $\mathbf{1.6\times10^{-6}}$ | $5.7\times10^{-3}$ |

On `sine` the method beats Adam by seven to eight orders. On `runge` it *loses*, and the last two columns say why: Adam's final features are worth $1.6\times10^{-6}$ under a refit, this method's only $5.7\times10^{-3}$. **Pinning the readout every 200 steps damaged feature learning by three orders.**

The reason is intuitive once stated. If you hold the readout at the exact optimum for the current features, the residual becomes orthogonal to everything those features can express. The gradient that reaches the features is then the part of the error they *cannot* currently explain, which is a much weaker learning signal than the ordinary one. `runge` is a peaked function whose features genuinely need to move; `sine` is smooth and random features nearly suffice, so pinning costs nothing and the exact solve collects the reward.

The same mechanism explains the real-data result: on three standard regression tasks this method is about $4\times$ worse than plain Adam. **The exploit regime pays when the remaining work is in the linear block, and costs when it is in the features.**

## 9.5 Two selection bugs, both instances of the same confusion

**The budget rule was backwards.** When more parameters qualify than the memory budget allows, the first version kept the *most linear* ones. At $N{=}512$ the budget binds, and this handed **628 of 841 slots to invisible halo units, starving the readout to 213 of its 922 coordinates**. All six targets diverged to about $10^{10}$. The rule is backwards because, per Section 5, linearity is exactly what invisibility looks like. Ranking by data energy instead fixes it.

**Admission needed the same correction.** Even with budget to spare, blind coordinates were being admitted and contributing near-null columns, costing about one order. Requiring $\text{info} > 0$ to enter -- not merely to win a tiebreak -- removes them. Because the separation is absolute (exact zeros), this needs no threshold.

Both are the same mistake: treating "the gradient is tiny" as though it meant one thing. It means two opposite things, and the whole certificate exists to tell them apart.

## 9.6 A carried residual cannot be minibatched

The batching bench crashed on the BFGS arm with a length mismatch: an 8012-long carried residual used against a 6009-row batch. This is not a plumbing slip, it is the design. A carried residual is a vector *indexed by specific training rows*; under resampling those rows change every step. The direct-solve arm has no such state, which is exactly why it batches without modification. The condition is now an explicit guard rather than a crash.

---

# 10. Honest limits

**These results start from a good geometry.** Every one of the 30 floor results begins from the constructed geometry with the readout zeroed. What has been shown is that the optimizer *preserves and exploits* a good geometry all the way to machine precision at every width. It has **not** been shown to find such a geometry from scratch. The project's actual success criterion requires reaching this precision without construction-based initialization, and that has not been run. The nearest thing here is the two-layer random-init bench, which does not reach the floor.

**On real data it is worse than Adam**, by about $4\times$, for the reasons in 9.4.

**The memory rule fits this architecture badly, and the constant is doing work.** The budget is $B \le \sqrt{Cm}$ so the $B\times B$ state stays proportional to $m$. The strict form of the rule ($C = 1$, i.e. $B \approx \sqrt m$) is *unsatisfiable on this network family at any width*: a one-hidden-layer network's exactly-linear block is its readout, which is $\Theta(m)$ -- here $m/3$. At $N{=}512$ a strict budget would allow $B = \sqrt{2764} \approx 52$ against a 922-coordinate readout. Realistic architectures are the opposite case (a $10^9$-parameter model's linear head is $10^3$ to $10^4$, well under $\sqrt{k} \approx 3\times10^4$), so the rule binds only on this degenerate toy. Both sides are measured: with the cap truncating the readout, three of six targets lose twelve orders; without truncation, all six reach the floor. **The lesson is that truncation is only safe among *independent* candidates; a coupled block is all-or-nothing, because solving 90% of a least-squares system is not 90% as good.**

**The implementation is not the memory-efficient form.** It materializes $J_E$ in full, an $n\times B$ matrix (about 59 MB in the largest batching cell). Reducing this to $O(B^2)$ means streaming row-blocks through an incremental QR and keeping only the triangular factor -- standard and arithmetically identical, but not implemented here. Memory claims above describe the solver state, not the current code's peak usage.

**Batching caveats.** The batching runs use epoch reshuffling, not independent draws with replacement, and the targets are noiseless, so nothing here speaks to the statistical floor that label noise imposes.

---

# 11. Conclusions

Unsigned, pending Sam's review. What the data supports:

1. A certificate combining a second-difference nonlinearity probe with gradient energy discovers a network's exactly-linear block blind, on two different architectures, and an exact truncated-SVD solve on that block holds the direct-solve floor on all 30 toy cells and under minibatching down to quarter batches.
2. This is a statement about preserving and exploiting a good geometry, not about finding one.
3. The binding obstacle is a measured law rather than a tuning problem: a scale-free base step re-injects $\|v\|\eta$ of error immediately after each solve, and the escape (Adam's settling regime) is already visible where the method converges cleanly.
4. Solving early is not free. It pins the readout and weakens the feature learning signal, which is why the method loses on `runge` and on real data.

## Open questions

- **When should the solve engage?** Running pure Adam while the drift gauge reads real motion, and engaging solves as it quiets, is the natural rule -- ideally without introducing a hand-set constant.
- **Make the base step vanish** as the certified residual vanishes, so the coupling term dies. The `exp` and two-layer `sine` panels show this happening by accident; make it deliberate.
- **Reset Adam's moments** on coordinates the solve just moved. Cheap, principled, unmeasured.
- **Test on a problem where the linear block is not the last layer**, to check the certificate is really finding structure rather than rediscovering an architectural convention.

## Where the figures are

- `1D_test/scaling_2x3.png` -- error against width, one panel per target, both arms, floor dashed.
- `1D_test/loss_curves_2x2_N256.png` -- error against iteration; the sawtooth of 9.2, and `exp` going flat on the floor (9.3).
- `1D_test/gifs_varpro/dial_N256.gif` -- the certificate itself: red is being solved, grey is Adam only, with the thresholds and the measured-zero band drawn in.
- `1D_test/gifs_varpro/params_N{128,256,512}.gif` -- the parameters themselves against their centres, with the theoretical construction values overlaid.
- `depth2/depth2_loss_2x2.png` and `depth2/gifs_varpro/dial.gif` -- the two-layer bench and its certificate over five parameter groups.
- `batching/softwin_batchsize_vs_error.png`, `batching/strongwin_interpolation_bars.png` -- Section 8.2.
- `coupling_law.png` -- Section 9.2, the $\|v\|\eta$ line against measurement.

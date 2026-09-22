# Gamma controls optimization access

[Two-page PDF](gamma_optimization_pi_brief.pdf) · [LaTeX source](gamma_optimization_pi_brief.tex)

**Large slopes can be needed to acquire an attainable target within a practical training budget.** In our controlled readout experiment, changing the common tanh slope from 8 to 64 reduces the updates needed for 1% relative residual from **15,798,313 to 16,013**. Both models reach that accuracy. We explain this optimization gap by deriving gamma's explicit frequency filter, carrying it through the finite learning kernel, and predicting a **985.70–987.49-fold delay**, enclosing the measured **986.59-fold delay**. The finite-budget benefit also appears with Adam across five targets.

## Mechanism and complete proof

Fix centers $c_j$, training inputs $x_i$, and nonzero target vector $y=(y_i)$, with $j=1,\ldots,W$ and $i=1,\ldots,m$. Train only the raw readout coefficients $\theta=(b,w_1,\ldots,w_W)$ from zero on half mean squared error:

$$
f_{\gamma,\theta}(x)=b+\sum_{j=1}^W w_j\tanh(\gamma(x-c_j)),\qquad \gamma>0.
$$

**Proposition.** Gamma smooths a fixed dictionary of steps, and hence both arguments of its kernel, with a frequency multiplier

$$
M_\gamma(\omega)=\frac{z}{\sinh z},\qquad z=\frac{\pi|\omega|}{2\gamma},\qquad M_\gamma(0)=1.
$$

A cap $\gamma\le\bar\gamma$ implies $M_\gamma(\omega)\le M_{\bar\gamma}(\omega)$. Fine scales, $|\omega|\gg\gamma$, are exponentially attenuated: $M_\gamma(\omega)\sim2ze^{-z}$.

*Proof.* Let $\rho_\gamma(t)=\gamma\operatorname{sech}^2(\gamma t)/2$. Its integral is one, so splitting at a step's center gives

$$
\int_{\mathbb R}\rho_\gamma(x-t)\operatorname{sign}(t-c)\,dt
=2\int_{-\infty}^{x-c}\rho_\gamma(u)\,du-1
=\tanh(\gamma(x-c)).
$$

Define the fixed reference $k_{\rm step}(t,s)=1+\sum_j\operatorname{sign}(t-c_j)\operatorname{sign}(s-c_j)$. Taking feature inner products yields the exact tanh kernel

$$
\begin{aligned}
k_\gamma(x,x')&=1+\sum_j\tanh(\gamma(x-c_j))\tanh(\gamma(x'-c_j))\\
&=\iint\rho_\gamma(x-t)\rho_\gamma(x'-s)k_{\rm step}(t,s)\,dt\,ds.
\end{aligned}
$$

The bias is preserved by unit mass; bounded features justify exchanging the finite sum and integrals. To compute the Fourier multiplier, set $a=\omega/(2\gamma)$ and $v=e^{2\gamma t}$. The beta integral and Euler's gamma reflection identity give

$$
\widehat\rho_\gamma(\omega)
=\int_0^\infty\frac{v^{-ia}}{(1+v)^2}\,dv
=\Gamma(1-ia)\Gamma(1+ia)
=\frac{\pi a}{\sinh(\pi a)}=M_\gamma(\omega),
$$

with the value at zero given by continuity; $\Gamma$ here is Euler's gamma function. Indeed, $\Gamma(1+ia)=ia\Gamma(ia)$ and $\Gamma(ia)\Gamma(1-ia)=\pi/\sin(\pi ia)$ give the last equality. Convolving a wave $e^{i\omega x}$ multiplies it by this transform. For $z>0$, the derivative of $z/\sinh z$ has numerator $\sinh z-z\cosh z<0$, since that numerator starts at zero and has derivative $-z\sinh z<0$. This proves the cap inequality; $\sinh z\sim e^z/2$ proves the asymptotic. No periodic geometry is assumed. $\square$

**From the mechanism to training.** Sample the kernel: $(K_\gamma)_{i\ell}=k_\gamma(x_i,x_\ell)/m$. With residual $r_n=y-(f_{\gamma,\theta_n}(x_i))_{i=1}^m$ and step $0<\eta_\gamma\le1/\|K_\gamma\|_2$, ordinary least-squares GD gives

$$
r_{n+1}=(I-\eta_\gamma K_\gamma)r_n,\qquad
E_\gamma(n)=\frac{\|(I-\eta_\gamma K_\gamma)^ny\|_2}{\|y\|_2}.
$$

This is standard kernel-gradient dynamics ([Yao, Rosasco, and Caponnetto, 2007, §3.3](https://yao-lab.github.io/publications/YaoCapRos07_EarlyStop.pdf)). An eigenmode of eigenvalue $\lambda$ decays by $(1-\eta_\gamma\lambda)^n$; the target determines how much residual lies in it. The contribution is the explicit gamma-to-kernel mechanism. We retain the actual center and sample couplings when calculating $E_\gamma$, without fitting a training curve. The first $n$ with $E_\gamma(n)\le\epsilon$ is the acquisition time; bounding the finite expansion's error brackets this crossing. Fourier multipliers are not generally finite-kernel eigenvalues.

## Measured consequences

<figure>
  <img src="../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/pi_brief_three_panel.png" alt="A: gamma 8 suppresses fine frequencies much more strongly than gamma 64. B: certified acquisition intervals match executed readout GD. C: increasing gamma improves the 200,000-update training residual for all five targets under GD and both common and selected Adam settings." style="max-width: 100%;">
  <figcaption><strong>Gamma changes the corrections available to training.</strong> <strong>A:</strong> Exact filter; marked frequencies are the three sine-mixture components. <strong>B:</strong> Executed first hits of 1% relative training residual versus certified necessary–sufficient intervals; intervals are narrower than the markers, not statistical confidence intervals. <strong>C:</strong> Relative training residual at gamma 8 divided by that at gamma 64 after 200,000 updates. Values above one favor gamma 64; these are error ratios, not timing ratios. All empirical panels use the same fixed equispaced centers with a boundary halo, $W=559$, $m=8193$ equally spaced inputs on $[-1,1]$, raw coordinates, and zero initialization. GD uses $\eta_\gamma\simeq0.5/\|K_\gamma\|_2$. Adam's common initial rate/epsilon are $10^{-3}/10^{-12}$; selected settings use the predeclared validation protocol below. All are archived deterministic training results.</figcaption>
</figure>

**Why the delay is mechanistic.** The primary target is $\sin(2\pi x)+\tfrac12\sin(6\pi x)+\tfrac14\sin(10\pi x)$. At its finest component, changing gamma from 64 to 8 reduces $M_\gamma(10\pi)$ from 0.9074 to 0.02584, about 35-fold. Carrying the whole filtered kernel through the residual formula predicts the observed 986.59-fold GD delay. A control that changes only the kernel's overall magnitude predicts 16,013 updates at every gamma under the normalized step rule. The gamma-8 interval has total width 0.181% of its executed hit; the gamma-64 count is exact. Thus the strong effect survives normalization for largest curvature and occurs at an accuracy both models attain.

**Across targets and Adam.** Alongside the mixture, panel C includes $\exp(\sin(3\pi x))$, $1/(1+25x^2)$, $\sqrt5x^2$, and $\sqrt2\sin(2\pi x)$. Raising gamma from 8 to 64 reduces the 200k residual for all five: by **2.31–83.06-fold for GD**, **1.35–61.97-fold for common-settings Adam**, and **12.89–770.22-fold for selected Adam**. For the mixture, common Adam leaves 1.213% residual at gamma 8 versus 0.0196% at 64. The benefit therefore extends empirically beyond GD. It is target- and protocol-dependent; the GD timing formula is not an Adam theorem, and neither monotone improvement nor a universal necessary gamma threshold is claimed.

**Adam protocol.** Both Adam variants hold their initial rate through 20k updates, cosine-decay to $10^{-3}$ of it at 50k, then hold that terminal rate through 200k. Selection crosses five initial rates $10^{-5},\ldots,10^{-1}$ with epsilons $10^{-8},10^{-12}$ and minimizes median validation residual over five checkpoints at 40k–50k on 4096 offset inputs. Moments and counters persist into continuation. Selection never uses the plotted 200k endpoint. These comparisons assess optimization on the training samples, not held-out generalization.

**Supporting material.** [Full review and numerical certification](gamma_optimization_paper_note.pdf), [finite-kernel construction](gamma_factorized_readout.md), and [plotted values and source hashes](../results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep/refinements/gamma_factorized_kernel/pi_brief_figure_data.json). The figure is reproduced with `python -m experiments.expD36_frozen_gamma_probe.pi_brief_figure`; compile the PDF with `latexmk -pdf -outdir=/tmp/gamma-pi-latex docs/gamma_optimization_pi_brief.tex` from the repository root. No additional training was run for this brief.

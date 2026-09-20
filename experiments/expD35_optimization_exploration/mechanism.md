# What the EMA and neighboring coordinates can change

Two local calculations help interpret this exploration. The first is exact for a fixed least-squares system. The second describes neighboring tanh features with uniform slopes on the real line. Neither calculation proves convergence of the jointly trained finite network.

For half-MSE with a fixed readout matrix, a singular direction with singular value $\sigma$ has curvature $k=\sigma^2$. If $x_t$ is its coefficient error, ordinary GD gives

$$
x_{t+1}=(1-\eta k)x_t.
$$

Stability of the strongest direction limits $\eta$, while the weakest target-relevant direction controls the eventual rate. Thus the iteration count scales with the condition number of the normal matrix, not exponentially with that condition number. Exponential slowing can arise when the relevant singular values themselves decay exponentially with the geometry scale.

For the active gradient filter used here,

$$
e_t=\alpha e_{t-1}+(1-\alpha)kx_t,
\qquad
x_{t+1}=x_t-\eta(kx_t+\rho e_t),
$$

the two-state recurrence is

$$
\begin{bmatrix}x_{t+1}\\e_t\end{bmatrix}
=
\begin{bmatrix}
1-\eta k[1+\rho(1-\alpha)]&-\eta\rho\alpha\\
(1-\alpha)k&\alpha
\end{bmatrix}
\begin{bmatrix}x_t\\e_{t-1}\end{bmatrix}.
$$

For $0<\alpha<1$, $\rho\geq0$, and $k>0$, the scalar mode is asymptotically stable precisely when

$$
0<\eta k<\frac{2(1+\alpha)}{1+\alpha+\rho(1-\alpha)}.
$$

Its slow eigenvalue, for sufficiently small $\eta k$, is

$$
1-\eta(1+\rho)k+O((\eta k)^2).
$$

Consequently the EMA can amplify slow, persistent gradients without amplifying rapidly alternating gradients by the same amount. For example, $\alpha=0.98$ and $\rho=8$ give a weak-mode gain of nine, while the stability ceiling is $\eta k<1.85047\ldots$. Multiplying ordinary GD's rate by nine instead requires $\eta k<2/9$. This distinction motivates both the exact-rate control and the larger-rate control. The filter still leaves a sufficiently weak mode slow; it does not remove the small singular value.

The recurrence describes temporal frequencies of gradients. The Fourier plots in the experiment describe spatial frequencies of functions. They are connected only through optimization dynamics: a weak spatial singular mode tends to evolve slowly in time. Those two frequency notions should not be conflated. The scalar calculation also does not directly apply to Adam's changing, coordinate-dependent normalization or to moving geometry. Initializing the filter with the first gradient can create a large transient even inside the scalar asymptotic stability region.

Now consider adjacent centers $t$ and $t+h$, with one common slope $\gamma>0$:

$$
B_t(x)=\tanh(\gamma(x-t))-\tanh(\gamma(x-t-h)).
$$

With Fourier convention $\widehat B(\omega)=\int_{\mathbb R}B(x)e^{-i\omega x}\,dx$,

$$
\widehat B_t(\omega)
=\frac{2\pi}{\gamma}
e^{-i\omega(t+h/2)}
\frac{\sin(\omega h/2)}{\sinh(\pi\omega/(2\gamma))},
\qquad \widehat B_t(0)=2h.
$$

One derivation differentiates $B_t$, uses the transform of $\operatorname{sech}^2$, and divides by $i\omega$. The needed transform follows directly from $u=e^{2x}$:

$$
\int_{\mathbb R}\operatorname{sech}^2(x)e^{-ikx}\,dx
=2\int_0^\infty\frac{u^{-ik/2}}{(1+u)^2}\,du
=2\Gamma(1-ik/2)\Gamma(1+ik/2)
=\frac{\pi k}{\sinh(\pi k/2)}.
$$

Differencing removes the broad antiderivative component and produces localized features. It retains the exponential high-frequency factor $\exp[-\pi|\omega|/(2\gamma)]$. With $\lambda=\gamma h$ and $\theta=\omega h$, this factor is $\exp[-\pi|\theta|/(2\lambda)]$. Differencing therefore has a principled conditioning benefit, but cannot make arbitrarily weak high-frequency directions accessible at a fixed small bandwidth. Differentiating this factor with respect to the slope introduces polynomial factors while retaining the exponential suppression.

Finite intervals, the bias and final anchor, halos, unequal learned slopes, and nonuniform affine centers all change the actual singular spectrum. The experiment measures that spectrum and the residual's projection onto it. A large global condition number alone is insufficient evidence of an optimization barrier: the remaining target error must also occupy weak directions. Conversely, a favorable median bandwidth alone is insufficient evidence of useful geometry. The detached fits, coefficient norms, and function-space update measurements address these distinctions.

For the fixed-center model, the exact physical slope gradient is

$$
\frac{\partial L}{\partial\gamma_j}
=w_j\left\langle r,(x-t_j)\operatorname{sech}^2(\gamma_j(x-t_j))\right\rangle,
\qquad L=\tfrac12\langle r,r\rangle.
$$

Training $\lambda_j=h\gamma_j$ with native GD gives $\Delta\gamma_j=-(\eta/h^2)\partial L/\partial\gamma_j$. This removes the explicit coordinate scale; it cannot restore a residual projection that readout training has already removed, or an exponentially attenuated projection onto the remaining residual. Readout magnitude is another multiplicative factor. This is the local signal mechanism behind the proposed race. A proof of a lasting barrier additionally needs control of the trajectory, the readout magnitudes, and the residual projections; choosing the coordinate scales alone does not provide those bounds.

There is also a distinct explanation for large opposing parameter-block motions late in training. Let $J_r$ and $J_g$ be the readout and geometry Jacobians in the trained coordinates, with the training quadrature included. For an infinitesimal plain GD step,

$$
\Delta f_r=-\eta J_rJ_r^T r,\qquad
\Delta f_g=-\eta J_gJ_g^T r,
$$

and therefore

$$
\langle r,\Delta f_r\rangle=-\eta\|J_r^Tr\|^2\leq0,
\qquad
\langle r,\Delta f_g\rangle=-\eta\|J_g^Tr\|^2\leq0.
$$

The two function increments can nevertheless have a strongly negative mutual inner product: their components perpendicular to the residual can cancel. Such cancellation indicates redundant or nearly redundant directions in the joint parameterization. It does not by itself show that geometry training harms descent or that freezing geometry would improve the result. The measured increments use exact finite changes with readouts changed first, so nonlinear cross effects and a different validation quadrature are disclosed separately. The infinitesimal identities above apply to the training quadrature and plain GD, not automatically to filtered GD or Adam.

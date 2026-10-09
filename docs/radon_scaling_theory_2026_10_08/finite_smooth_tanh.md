# A stable finite-smooth tanh construction with an explicit finite halo

This result complements the local analytic corrected-halo theorem. It constructs a one-hidden-layer tanh network from a supplied Fourier description of the target derivative. It uses a finite frequency cutoff, uniformly spaced centers, an ordinary finite halo, and a constant anchor. No rational halo correction, least-squares fit, auxiliary output polynomial, or additional layer is used.

Its main advantage is a bounded total absolute readout size. For a finite-smooth Fourier class, it gives algebraic approximation with algebraic width and logarithmic working-precision growth. This global Fourier assumption is distinct from a local derivative bound. The polynomial-to-tanh alternative remains a valid way to use local smoothness, but its conservative precision certificate is substantially less favorable.

The proof is a tanh specialization of the master derivative-kernel certificate in `activation_certificate.md`, section 2. Every cutoff, alias, and finite-halo bound is calculated explicitly below. No experiments or Lean development are part of this note.

## 1. Target and access assumptions

Fix a target interval \([-T,T]\), where \(T>0\). The real target \(g\) has a derivative \(p=g'\), with a specified full-line Fourier representation satisfying

\[
p(s)=\frac1{2\pi}\int_{\mathbb R}\widehat p(\omega)e^{i\omega s}\,d\omega.
\]

The transform is absolutely integrable and conjugate symmetric. For a specified \(s_*>0\), suppose the known finite number

\[
\boxed{
A_{s_*}:=\frac1{2\pi}\int_{\mathbb R}
(1+|\omega|)^{s_*}|\widehat p(\omega)|\,d\omega
}
\]

is available as a certified bound. The value \(g(0)\) is also supplied. An ordinary integrable Fourier transform of \(g\) itself is not required.

In addition to this function-class assumption, evaluating the transform or its finite integrals with certified accuracy is part of the access model. Weighted Fourier integrability alone does not turn arbitrary point samples into a certified integration oracle.

If \(A_{s_*}=0\), then \(p=0\) and the target is the constant \(g(0)\); no neurons are needed. The following assumes \(A_{s_*}>0\).

The exponent \(s_*\) concerns a Fourier moment of the **derivative**. This is not the statement that every locally \(C^{s_*}\) target has the same rate. Additional extension and regularity arguments would be required to relate a local derivative assumption to this class.

## 2. All parameters and coefficients

Choose a frequency cutoff \(B\ge1\) such that

\[
t=s_*\log(1+B)\ge1.
\]

Set

\[
\boxed{
\tau=\frac{t}{2B},\quad
\gamma=\frac{\pi B}{t},\quad
h=\frac{\pi}{4B},\quad
\Omega=\frac{2\pi}{h}=8B,\quad
\lambda=\gamma h=\frac{\pi^2}{4t}.
}
\]

The physical halo length is

\[
\ell_h=\frac{t}{\gamma}=\frac{t^2}{\pi B}.
\]

Keep precisely the grid centers

\[
c_j=jh,\quad -J\le j\le J,
\qquad J=\left\lfloor\frac{T+\ell_h}{h}\right\rfloor.
\]

This grid is symmetric and contains zero. It need not contain \(\pm T\) exactly. The first omitted right center satisfies \((J+1)h>T+\ell_h\), and the first omitted left center is its negative. Thus all omitted centers are at least \(\ell_h\) from the target interval.

Define the bounded, bandlimited density by the finite integral

\[
\boxed{
\rho_B(c)=\frac1{2\pi}\int_{-B}^{B}
\widehat p(\omega)
\frac{\sinh(\tau\omega)}{\tau\omega}
e^{i\omega c}\,d\omega.
}
\]

The quotient at zero is defined by its limit, one. Conjugate symmetry makes \(\rho_B(c)\) real for real \(c\). The raw tanh readouts, bias, and final network are

\[
u_j=\frac h2\rho_B(c_j),\qquad
b_g=g(0)-\sum_{j=-J}^J u_j\tanh(-\gamma c_j),
\]

\[
\boxed{
q(s)=b_g+\sum_{j=-J}^J u_j\tanh(\gamma(s-c_j)).
}
\]

These are the complete network parameters. Only the coefficient construction uses Fourier integration; evaluation is an ordinary finite tanh sum and a constant.

## 3. The inverse filter does not create large density amplitudes

For \(v\ge0\),

\[
\frac{\sinh v}{v}\le e^v.
\]

For example, this follows from \(1-e^{-2v}\le2v\), with the continuous limit at zero. Concavity of \(\log(1+u)\) gives, for \(0\le u\le B\),

\[
\frac uB\log(1+B)\le\log(1+u).
\]

Consequently,

\[
e^{\tau u}
=\exp\left(\frac{s_*u}{2B}\log(1+B)\right)
\le(1+u)^{s_*/2}.
\]

The density integral therefore satisfies

\[
\boxed{
\|\rho_B\|_\infty
\le\frac1{2\pi}\int_{-B}^B
|\widehat p(\omega)|\frac{\sinh(\tau|\omega|)}{\tau|\omega|}\,d\omega
\le A_{s_*}.
}
\tag{1}
\]

The largest pointwise inverse-filter multiplier in the used frequency band is at most \((1+B)^{s_*/2}\). It grows polynomially in \(B\), not exponentially in \(B\).

## 4. Differentiate the actual network to separate its errors

The normalized tanh derivative kernel and its Fourier transform are

\[
K_\gamma(u)=\frac\gamma2\operatorname{sech}^2(\gamma u),
\qquad
\widehat K_\gamma(\omega)=\frac{\tau\omega}{\sinh(\tau\omega)}.
\]

The network derivative is exactly

\[
q'(s)=h\sum_{j=-J}^J\rho_B(jh)K_\gamma(s-jh).
\]

Temporarily retain the infinite kernel sum

\[
D_h(s)=h\sum_{j\in\mathbb Z}\rho_B(jh)K_\gamma(s-jh).
\]

This sum converges absolutely because the density is bounded and the kernel decays exponentially. No convergence of an unanchored infinite tanh sum is assumed.

Sampling a single frequency and applying the Fourier-series identity gives

\[
h\sum_j e^{i\omega jh}K_\gamma(s-jh)
=\sum_{k\in\mathbb Z}
\widehat K_\gamma(\omega+k\Omega)e^{i(\omega+k\Omega)s}.
\]

Substituting the density integral into \(D_h\) therefore leaves three errors: frequencies omitted by the cutoff, nonzero aliases \(k\ne0\), and spatial centers omitted by the finite halo. All three are bounded next.

## 5. Cutoff error

The cutoff target derivative is

\[
p_B(s)=\frac1{2\pi}\int_{-B}^B\widehat p(\omega)e^{i\omega s}\,d\omega.
\]

Since \((1+|\omega|)^{s_*}\ge(1+B)^{s_*}\) outside the band,

\[
\boxed{
\|p-p_B\|_\infty
\le A_{s_*}(1+B)^{-s_*}=A_{s_*}e^{-t}.
}
\tag{2}
\]

## 6. The complete alias sum

For \(|\omega|\le B\), write \(a=\tau\omega\), so \(|a|\le t/2\). If the alias index is a nonzero integer \(k\), then

\[
\tau(\omega+k\Omega)=a+4kt.
\]

Let \(n_k=|k|\ge1\) and \(y=|a+4kt|\). Then

\[
(4n_k-\tfrac12)t\le y\le(4n_k+\tfrac12)t,
\qquad y-|a|\ge(4n_k-1)t.
\]

The kernel-ratio magnitude is

\[
\left|\frac{\widehat K_\gamma(\omega+k\Omega)}
{\widehat K_\gamma(\omega)}\right|
=\frac{y}{\sinh y}\frac{\sinh|a|}{|a|}.
\]

Use \(\sinh|a|/|a|\le e^{|a|}\) and \(y\ge7/2\):

\[
\begin{aligned}
\left|\frac{\widehat K_\gamma(\omega+k\Omega)}
{\widehat K_\gamma(\omega)}\right|
&\le\frac{2y}{1-e^{-2y}}e^{-(y-|a|)}\\
&\le\frac{(8n_k+1)t}{1-e^{-7}}e^{-(4n_k-1)t}.
\end{aligned}
\]

Sum both signs of \(k\), and put \(z=e^{-4t}\):

\[
\begin{aligned}
\sum_{k\ne0}\left|\frac{\widehat K_\gamma(\omega+k\Omega)}
{\widehat K_\gamma(\omega)}\right|
&\le\frac{2t e^t}{1-e^{-7}}\sum_{n_k\ge1}(8n_k+1)z^{n_k}\\
&=\frac{2t e^{-3t}}{1-e^{-7}}\frac{9-z}{(1-z)^2}\\
&\le20t e^{-3t}\le3e^{-t}.
\end{aligned}
\]

For the penultimate step, \(0<z\le e^{-4}\), and the remaining multiplier is less than 20. For the last step, \(20t e^{-2t}\) decreases for \(t\ge1\), and its value at one is less than three.

Integrating against \(|\widehat p|/(2\pi)\), whose integral is bounded by \(A_{s_*}\), proves

\[
\boxed{\|D_h-p_B\|_\infty\le3A_{s_*}e^{-t}.}
\tag{3}
\]

This bounds every nonzero alias, rather than only the two nearest aliases.

## 7. The finite halo error

For real \(u\),

\[
K_\gamma(u)\le2\gamma e^{-2\gamma|u|}.
\]

For \(|s|\le T\), the distances to omitted centers on either side are at least \(\ell_h+lh\), \(l=0,1,\ldots\). Using (1),

\[
\begin{aligned}
|D_h(s)-q'(s)|
&\le4A_{s_*}\gamma h
\sum_{l\ge0}e^{-2\gamma(\ell_h+lh)}\\
&=\frac{4A_{s_*}\lambda}{1-e^{-2\lambda}}e^{-2t}.
\end{aligned}
\]

The elementary inequality \((1-e^{-u})^{-1}\le1+1/u\) for \(u>0\) gives

\[
\frac{4\lambda}{1-e^{-2\lambda}}\le4\lambda+2
\le\pi^2+2<12,
\]

because \(\lambda=\pi^2/(4t)\) and \(t\ge1\). Consequently,

\[
\boxed{
\|D_h-q'\|_{L^\infty([-T,T])}
\le12A_{s_*}e^{-2t}\le5A_{s_*}e^{-t}.
}
\tag{4}
\]

These are ordinary truncated halos. The result does not invoke the rational correction used in the separate local analytic theorem.

## 8. Function error, width, and tolerance choice

Adding (2), (3), and (4) gives

\[
\|q'-g'\|_{L^\infty([-T,T])}\le9A_{s_*}e^{-t}.
\]

The chosen constant guarantees \(q(0)=g(0)\). Integrate the derivative discrepancy between zero and any \(s\in[-T,T]\):

\[
\boxed{
\|q-g\|_{L^\infty([-T,T])}
\le9T A_{s_*}(1+B)^{-s_*}.
}
\tag{5}
\]

For a requested exact-arithmetic approximation tolerance \(\varepsilon_s>0\), take

\[
\boxed{
B\ge\max\left\{1,e^{1/s_*}-1,
\left(\frac{9TA_{s_*}}{\varepsilon_s}\right)^{1/s_*}-1\right\}.
}
\tag{6}
\]

The exact number of neurons is \(W_s=2J+1\). The floor definition gives

\[
\boxed{
W_s\le\frac{2(T+\ell_h)}h+1
=\frac{8TB}{\pi}+\frac{8t^2}{\pi^2}+1.
}
\tag{7}
\]

The total halo count is at most

\[
2\ell_h/h+2=8t^2/\pi^2+2.
\]

For fixed \(T,A_{s_*},s_*\), this gives

\[
W_s=O\bigl(\varepsilon_s^{-1/s_*}+\log^2(1/\varepsilon_s)\bigr)
=O(\varepsilon_s^{-1/s_*}).
\]

The logarithm-squared halo count is included. Its physical extent \(t^2/(\pi B)\) tends to zero as the cutoff increases.

## 9. Coefficient size and numerical precision

By (1), each readout satisfies \(|u_j|\le hA_{s_*}/2\). Thus

\[
\boxed{
\sum_{j=-J}^J|u_j|
\le\frac{W_sh}{2}A_{s_*}
\le A_{s_*}(T+\ell_h+h/2),
\qquad
|b_g|\le|g(0)|+\sum_j|u_j|.
}
\tag{8}
\]

For fixed target constants, these bounds stay bounded as the tolerance tends to zero. The construction therefore does not require exponentially large cancelling output weights.

Suppose every computed density value has absolute error at most \(\eta_\rho\), and the anchor has error at most \(\eta_0\). If the remaining arithmetic and geometry are exact and the bias is recomputed from the supplied anchor,

\[
\|\widetilde q-q\|_\infty
\le\eta_0+2\sum_j|\Delta u_j|
\le\boxed{\eta_0+[2(T+\ell_h)+h]\eta_\rho}.
\tag{9}
\]

To reserve \(\varepsilon_{\rm data}\) for these errors, one concrete choice is

\[
\eta_0\le\varepsilon_{\rm data}/2,
\qquad
\eta_\rho\le
\frac{\varepsilon_{\rm data}}{2[2(T+\ell_h)+h]}.
\]

Final arithmetic, hidden-parameter storage, and tanh evaluation have separate tolerances. The bounds \(|\tanh|\le1\), \(|\tanh'|\le1\), and (8) propagate these tolerances. Since \(W_s\) and \(\gamma\) grow at most algebraically in \(1/\varepsilon_s\), and the density inverse multiplier grows at most polynomially in \(B\), the corresponding arithmetic precision requirements grow logarithmically in \(1/\varepsilon_s\), for fixed target constants and certified transform access. This statement is about conditioning and precision; it is not a claim that arbitrary transform quadrature has logarithmic computational cost.

The density coefficient integral is finite. The explicit midpoint quadrature certificate in `activation_certificate.md`, section 8, applies if its frequency integrand has the stated second-derivative bound; split at known nonsmooth points first. Analytic or interval quadrature provides other certified realizations. Without adequate transform-access information, finite-sample automatic certification is not proved by the Fourier moment assumption alone.

## 10. Relation to the other certificates

- This is a stable algebraic construction under a global Fourier moment assumption on \(g'\). It cannot be relabeled as a theorem for arbitrary local \(C^s\) data with the same exponent.
- The local finite-smooth polynomial-to-tanh route in `activation_certificate.md` remains valid under a suitable real approximation bound, but its complex polynomial norm grows exponentially in its degree. Its guaranteed working bits grow like the polynomial degree, typically \(O(\varepsilon^{-1/s})\), rather than logarithmically. That difference is a practical distinction, not just a proof detail.
- The present uncorrected-halo scheme is not the source of the analytic logarithmic-width result. If one uses a logarithmic frequency cutoff for analytic data, the ordinary halo can contribute logarithm-squared width. The separate rational halo correction is what proves logarithmic scalar width for the local analytic tanh class.
- When used in a Radon sum with positive normalized angular weights, the derivative-profile assumptions and constants must be supplied per direction or uniformly. Scalar errors then add with those weights; angular error remains a separate certificate.

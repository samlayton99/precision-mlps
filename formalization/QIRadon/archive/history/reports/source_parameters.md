# Explicit source margins and scalar ellipse transfer

`QIRadon/Radon/SourceParameterChoice.lean` constructs
`standardSourceParameters R σ hR hσ` from only R>1 and σ>0.

Its numerical choices are:

- g=(R-1)/4, T=1+g, S=1+2g, R₀=1+3g, Q=(S+R₀)/2.
- L*=1+Q/sqrt(R₀²-Q²).
- η=log(1+σ/(4R))/2 and σ₀=σ/(4 exp(2η)).
- h=min((R₀-S)/8, (S-T)/4, σ₀/(4L*)).
- b=min(η, log(1+h)/2).

Every field of `SourceParameters` is proved. All quantities requiring positive
values are positive. S+4h≤Q<R₀, T+2h<S, and L*(4h)≤σ₀ hold. The actual rotation
margin σ₀+(exp(2η)-1)(R+σ₀) equals σ/2, so the target tube retains reserve.
The actual orbit displacement exp(2b)-1≤h is also verified. The factor two in
these exponentials comes from the established complex rotation bound.

`QIRadon/Radon/EllipseTube.lean` proves

  ||cos(z)-cos(Re z)|| ≤ exp(|Im z|)-1.

Consequently the closed Bernstein ellipse with angular strip half-width
log(1+δ/2) lies strictly inside scalarTube(L,δ) whenever L≥1 and δ>0. The
nearest real point is explicitly cos(Re z), and the ellipse uses at most δ/2
of the source tube width. `ellipse_transfer` transfers actual complex
differentiability and magnitude bounds. `scalar_slice_ellipse_transfer` derives
the needed one-variable analyticity from the actual joint complex C-infinity
source profile. No several-complex-variable analyticity theorem is assumed.

A scaled variant handles [-A,A], 0<A≤L, with strip half-width
log(1+δ/(2A)). It transfers the same bound to z↦G(Az), without replacing a tube
hypothesis by a larger enclosing disk.

# Independent-validation selection: verified Bernstein alternative

`QIRadon/StatisticalSelection.lean` is built. The pinned Mathlib revision did
not contain a Bernstein or Bennett theorem. It contained Chernoff bounds,
independent moment-generating-function factorization, exponential series, and
Popoviciu's variance inequality. This module derives Bernstein from those
results; it does not substitute Hoeffding or assume a concentration conclusion.

## Probability foundation

The actual exponential-series remainder satisfies

  |exp(u)-1-u| ≤ u²/[2(1-|u|/3)],  |u|<3.

The proof uses (k+2)!≥2*3^k and a geometric series. Integrating this bound for a
centered bounded variable gives the variance-sensitive MGF bound. Independence
factors the actual MGF of the sum, and Mathlib's Chernoff theorem gives

  P(|Σ_i X_i|≥t) ≤ 2 exp(-t²/[2(V+Lt/3)]),

where |X_i|≤L, E[X_i]=0, and V bounds the sum of their second moments. The
standard 1/3 Bernstein constant is retained.

## What the data theorem assumes

- A finite nonempty family of fixed candidate functions Q_j and their widths.
- n>0 independent validation input points with a common actual law ν.
- Measurable target f and candidates, with known |f|,|Q_j|≤B and B>0.
- The validation values are the actual target values f(x).
- Confidence parameter 0<δ<1.

The candidate functions are deterministic in the theorem: they are frozen
before the validation draw. The theorem does not permit fitting or generating
new candidates using this validation sample. Dependence between different
candidates' losses on the same validation point is allowed; the finite union
bound accounts for searching the family.

The risk is the squared L² error under the input probability law:

  R_j=∫|f(x)-Q_j(x)|² dν(x).

The actual input law identifies this population integral with each validation
loss expectation. This equality is proved using HasLaw, not supplied as a
statistical accuracy hypothesis.

## Computed certificate and selector

Let K be the number of candidates, L=(2B)², s=log(2K/δ), and

  r = L sqrt(s/(2n)) + 2Ls/(3n).

The module proves this readable formula equals the formal confidence radius.
Popoviciu supplies the automatic variance bound L²/4. This is a known bounded-
loss variance bound, not an empirical-variance estimate. The more general
Bernstein core retains arbitrary proved second-moment upper bounds.

Compute empirical mean squared error E_j on the validation points. With
probability at least 1-δ, simultaneously for every tested candidate,

  |E_j-R_j|<r.

Form the upper certificate U_j=E_j+r. Among candidates with U_j≤ε, select the
smallest width. `chooseSmallest` supplies the actual finite minimum selector.
On the simultaneous good event:

- The selected candidate has true squared risk at most ε.
- It is no wider than any tested candidate satisfying R_j+2r≤ε.
- If at least one candidate has that 2r margin, a passing candidate exists.

`validation_selection_certificate` combines the probability event and these
selection conclusions. The guarantee is among the tested frozen family, with
an explicit statistical margin. It does not claim the globally smallest neural
network or exact minimum width among models arbitrarily close to the risk
threshold.

## Main interfaces and limits

- `bernstein_two_sided`: genuine bounded variance-sensitive concentration.
- `simultaneous_validation_good_probability`: finite-family loss certificate.
- `frozen_squaredLoss_confidence`: actual f,Q and independent input samples.
- `chooseSmallest_spec`: minimum width among empirical certificates passing.
- `validation_selection_certificate`: soundness and tested-family width guarantee.

This is squared-L² certification under ν, not a uniform/sup-norm theorem, a
relative-error theorem, or a noise-removal theorem. Noisy observed targets would
require an explicit observation/noise model to relate observed loss to clean
risk. The statistical radius remains; no finite-sample machine-precision claim
is made. This complements the separate deterministic coverage certificate and
finite-sample obstruction theorem.

All headline theorem axiom audits contain only propext, Classical.choice, and
Quot.sound. There are no sorry, admit, or additional axioms.

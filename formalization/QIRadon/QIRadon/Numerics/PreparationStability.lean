import QIRadon.Scalar.LocalProfileRate

/-! Perturbed real samples propagated through the actual preparation and
finite activation construction. No regularity is assumed of the sample errors. -/

noncomputable section
open MeasureTheory FourierTransform Polynomial
open scoped Real SchwartzMap
namespace QIRadon.PreparationStability
open QIRadon.ProfilePreparation QIRadon.LocalProfileRate QIRadon.KernelRates QIRadon.ScalarRate

theorem samplePolynomial_difference {n : ℕ} (hn : 0 < n) {f g : ℝ → ℝ} {η x : ℝ}
    (hη : 0 ≤ η) (hdata : ∀ j < n, |f (sampleNode n j)-g (sampleNode n j)| ≤ η)
    (hx : |x| ≤ 1) :
    |(samplePolynomial n f).eval x-(samplePolynomial n g).eval x| ≤ 2*n*η := by
  have hh := samplePolynomial_bound (f := f-g) hn hη hdata hx
  have he : samplePolynomial n (f-g) = samplePolynomial n f-samplePolynomial n g :=
    (sampleOperator n).map_sub f g
  simpa only [he,Polynomial.eval_sub] using hh

theorem realPrepared_difference (N : ℕ) {f g : ℝ → ℝ} {η T₀ x : ℝ}
    (hη : 0 ≤ η)
    (hdata : ∀ j < N+1, |f (sampleNode (N+1) j)-g (sampleNode (N+1) j)| ≤ η)
    (hx : |x| ≤ 1) :
    |realPrepared N f T₀ x-realPrepared N g T₀ x| ≤ 2*(N+1)*η := by
  have hp := samplePolynomial_difference (Nat.succ_pos N) hη hdata hx
  unfold realPrepared
  rw [preparedProfile_ofReal,preparedProfile_ofReal,Complex.ofReal_re,Complex.ofReal_re,
    ← sub_mul,abs_mul,abs_of_nonneg (flatWindow_nonneg N (sq_nonneg _))]
  exact ((mul_le_mul_of_nonneg_left (flatWindow_le_one N (sq_nonneg _)) (abs_nonneg _)).trans
    (by simpa using hp))

/-- Explicit function error from noisy samples through every exact preparation
and kernel-construction step. The numerical sample term grows at most linearly
in the number of samples, rather than being hidden inside a deconvolution oracle. -/
theorem sampled_network_error
    {g : ℂ → ℂ} {data φ : ℝ → ℝ} {δ M η : ℝ}
    (hδ : 0 < δ) (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ))
    (hbound : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M) (hM : 0 ≤ M) (hη : 0 ≤ η)
    {r : ℕ} (hφ : ContDiff ℝ r φ) (K : 𝓢(ℝ, ℝ)) {m ν μ T₀ H Z : ℝ} (hm : m ≠ 0)
    (hkernel : ∀ u, iteratedDeriv r φ u = m*K u)
    (E : SpectralEnvelope (𝓕 (fun u : ℝ => (K u:ℂ))) ν) (S : SpatialEnvelope K μ)
    (hT : 1 < T₀) {N : ℕ} (hN : 0 < N)
    (hdata : ∀ j < N+1,
      |data (sampleNode (N+1) j)-(g (sampleNode (N+1) j:ℂ)).re| ≤ η)
    (hH : 2 ≤ H) (hsep : E.A+4 ≤ E.a*(H/2)^ν)
    (hZ : 0 ≤ Z) (hhalo : E.A+3 ≤ (S.b/2)*Z^μ)
    {x : ℝ} (hx : |x| ≤ 1) :
    |localActivationNetwork φ data K r N m T₀ H Z ν μ hN (lt_trans zero_lt_one hT) x-
      (g (x:ℂ)).re| ≤
      (localMomentConstant (M+η) T₀ r * envelopeConstant E S/(r.factorial:ℝ))*Real.exp (-2*N) +
      2*(N+1)*η + interpolationConstant M δ*Real.exp (-(δ/2)*N) +
      (2*(N+1)*M)*Real.exp (-(N:ℝ)*windowRate T₀) := by
  let f : ℝ → ℝ := fun t => (g (t:ℂ)).re
  have hf (j : ℕ) : |f (sampleNode (N+1) j)| ≤ M :=
    ellipse_real_bound hδ.le hbound (sampleNode_abs_le_one (N+1) j)
  have hsample : ∀ j < N+1, |data (sampleNode (N+1) j)| ≤ M+η := by
    intro j hj
    have hh := (abs_sub_le (data (sampleNode (N+1) j)) (f (sampleNode (N+1) j)) 0).trans
      (add_le_add (hdata j hj) (by simpa using hf j))
    simpa only [sub_zero,add_comm η] using hh
  have hnet := prepared_activation_geometric_error hφ K hm hkernel E S (add_nonneg hM hη)
    (lt_trans zero_lt_one hT) hN hsample hH hsep hZ hhalo x (abs_le.mp hx)
  have hnoise := realPrepared_difference (f := data) (g := f) (T₀ := T₀) N hη hdata hx
  have hprep := preparedProfile_real_error hδ hg hbound hT N hx
  have htri₁ := abs_sub_le (localActivationNetwork φ data K r N m T₀ H Z ν μ hN
    (lt_trans zero_lt_one hT) x) (realPrepared N data T₀ x) (g (x:ℂ)).re
  have htri₂ := abs_sub_le (realPrepared N data T₀ x) (realPrepared N f T₀ x) (g (x:ℂ)).re
  have hprep' : |realPrepared N f T₀ x-(g (x:ℂ)).re| ≤
      interpolationConstant M δ*Real.exp (-(δ/2)*N)+
      (2*(N+1)*M)*Real.exp (-(N:ℝ)*windowRate T₀) := by
    simpa only [realPrepared,f,interpolationConstant,windowRate,abs_sub_comm] using hprep
  have hnet' : |localActivationNetwork φ data K r N m T₀ H Z ν μ hN
      (lt_trans zero_lt_one hT) x-realPrepared N data T₀ x| ≤
      (localMomentConstant (M+η) T₀ r*envelopeConstant E S/(r.factorial:ℝ))*Real.exp (-2*N) := hnet
  exact (htri₁.trans (add_le_add hnet' (htri₂.trans (add_le_add hnoise hprep')))).trans_eq (by ring)

end QIRadon.PreparationStability

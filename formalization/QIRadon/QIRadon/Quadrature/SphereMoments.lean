import QIRadon.Radon.Polar
import QIRadon.Quadrature.GaussianMoments
import Mathlib.MeasureTheory.Integral.Pi
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

/-!
# Spherical moments from polar Gaussian integration

The sphere measure here is Mathlib's actual `volume.toSphere`, not an abstract
functional asserted to have the desired moments.
-/

open MeasureTheory Set Metric
open scoped BigOperators

namespace QIRadon.Gaussian

/-- Remove the explicit radial density carried by `volumeIoiPow`. -/
theorem integral_volumeIoiPow (m : ℕ) (f : ℝ → ℝ) :
    (∫ r : Ioi (0 : ℝ), f r.val ∂Measure.volumeIoiPow m) =
      ∫ r in Ioi (0 : ℝ), r ^ m * f r := by
  simp only [Measure.volumeIoiPow, ENNReal.ofReal]
  rw [integral_withDensity_eq_integral_smul,
    integral_subtype_comap measurableSet_Ioi
      (fun r : ℝ ↦ Real.toNNReal (r ^ m) • f r)]
  · apply setIntegral_congr_fun measurableSet_Ioi
    intro r hr
    dsimp only
    rw [NNReal.smul_def, Real.coe_toNNReal _ (pow_nonneg hr.le _)]
    rfl
  · exact (measurable_subtype_coe.pow_const _).real_toNNReal

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V] [Nontrivial V]

/-- Actual Gaussian polar separation for a degree-k homogeneous function. -/
theorem homogeneous_gaussian_polar (P : V → ℝ) (k : ℕ)
    (hhom : ∀ (r : ℝ) (x : V), P (r • x) = r ^ k * P x)
    (hint : Integrable (fun x : V ↦ P x * Real.exp (-(‖x‖ ^ 2)))) :
    (∫ x : V, P x * Real.exp (-(‖x‖ ^ 2))) =
      (Real.Gamma (((k + (Module.finrank ℝ V - 1) : ℕ) : ℝ) / 2 + 1 / 2) / 2) *
      ∫ θ : sphere (0 : V) 1, P θ.val ∂((volume : Measure V).toSphere) := by
  rw [QIRadon.integral_polar_iterated hint]
  have hinner : ∀ θ : sphere (0 : V) 1,
      (∫ r : Ioi (0 : ℝ),
        P (r.val • θ.val) * Real.exp (-(‖r.val • θ.val‖ ^ 2))
        ∂Measure.volumeIoiPow (Module.finrank ℝ V - 1)) =
      P θ.val *
        (Real.Gamma (((k + (Module.finrank ℝ V - 1) : ℕ) : ℝ) / 2 + 1 / 2) / 2) := by
    intro θ
    have hθ : ‖θ.val‖ = 1 := by simp
    calc
      _ = ∫ r : Ioi (0 : ℝ),
          P θ.val * (r.val ^ k * Real.exp (-(r.val ^ 2)))
          ∂Measure.volumeIoiPow (Module.finrank ℝ V - 1) := by
        apply integral_congr_ae
        filter_upwards [] with r
        rw [hhom, norm_smul, Real.norm_eq_abs, abs_of_pos r.property, hθ, mul_one]
        ring
      _ = P θ.val * (∫ r : Ioi (0 : ℝ),
          r.val ^ k * Real.exp (-(r.val ^ 2))
          ∂Measure.volumeIoiPow (Module.finrank ℝ V - 1)) := integral_const_mul _ _
      _ = P θ.val * (∫ r in Ioi (0 : ℝ),
          r ^ (k + (Module.finrank ℝ V - 1)) * Real.exp (-(r ^ 2))) := by
        rw [integral_volumeIoiPow (Module.finrank ℝ V - 1)
          (fun r : ℝ ↦ r ^ k * Real.exp (-(r ^ 2)))]
        congr 1
        apply setIntegral_congr_fun measurableSet_Ioi
        intro r _
        dsimp only
        rw [pow_add]
        ring
      _ = _ := by
        rw [integral_pow_gaussian_Ioi]
        congr 3
        ring
  simp_rw [hinner]
  rw [integral_mul_const, mul_comm]

section Coordinates

variable {ι : Type*} [Fintype ι]

/-- Coordinate monomial on the actual Euclidean space. -/
def coordinateMonomial (k : ι → ℕ) (x : EuclideanSpace ℝ ι) : ℝ := ∏ i, x i ^ k i

theorem coordinateMonomial_homogeneous (k : ι → ℕ) (r : ℝ) (x : EuclideanSpace ℝ ι) :
    coordinateMonomial k (r • x) = r ^ (∑ i, k i) * coordinateMonomial k x := by
  simp only [coordinateMonomial, PiLp.smul_apply, smul_eq_mul, mul_pow,
    Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum]

theorem coordinate_gaussian_product (k : ι → ℕ) (x : EuclideanSpace ℝ ι) :
    coordinateMonomial k x * Real.exp (-(‖x‖ ^ 2)) =
      ∏ i, (x i ^ k i * Real.exp (-(x i ^ 2))) := by
  rw [EuclideanSpace.real_norm_sq_eq, ← Finset.sum_neg_distrib, Real.exp_sum,
    Finset.prod_mul_distrib]
  rfl

theorem integrable_coordinate_gaussian (k : ι → ℕ) :
    Integrable (fun x : EuclideanSpace ℝ ι ↦
      coordinateMonomial k x * Real.exp (-(‖x‖ ^ 2))) := by
  have hi : Integrable (fun x : ι → ℝ ↦
      ∏ i, (x i ^ k i * Real.exp (-(x i ^ 2)))) :=
    Integrable.fintype_prod (fun i ↦ integrable_pow_gaussian (k i))
  have h := ((PiLp.volume_preserving_ofLp ι).integrable_comp_emb
    (MeasurableEquiv.toLp 2 (ι → ℝ)).symm.measurableEmbedding).mpr hi
  simpa only [Function.comp_def, coordinate_gaussian_product] using h

theorem integral_coordinate_gaussian (k : ι → ℕ) :
    (∫ x : EuclideanSpace ℝ ι,
      coordinateMonomial k x * Real.exp (-(‖x‖ ^ 2))) =
      ∏ i, ∫ t : ℝ, t ^ k i * Real.exp (-(t ^ 2)) := by
  simp_rw [coordinate_gaussian_product]
  have h := (PiLp.volume_preserving_ofLp ι).integral_comp
    (MeasurableEquiv.toLp 2 (ι → ℝ)).symm.measurableEmbedding
    (fun x : ι → ℝ ↦ ∏ i, (x i ^ k i * Real.exp (-(x i ^ 2))))
  calc
    _ = ∫ x : ι → ℝ, ∏ i, (x i ^ k i * Real.exp (-(x i ^ 2))) := h
    _ = _ := integral_fintype_prod_volume_eq_prod
      (fun (i : ι) (t : ℝ) ↦ t ^ k i * Real.exp (-(t ^ 2)))

/-- Actual surface-monomial moments characterized by products of scalar Gaussian
moments and one radial Gamma factor. This is a sphere integration theorem, not
an assumed quadrature exactness property. -/
theorem sphere_monomial_moment_scaled [Nonempty ι] (k : ι → ℕ) :
    (Real.Gamma (((∑ i, k i) + (Fintype.card ι - 1) : ℕ) / 2 + 1 / 2) / 2) *
      (∫ θ : sphere (0 : EuclideanSpace ℝ ι) 1,
        coordinateMonomial k θ.val ∂((volume : Measure (EuclideanSpace ℝ ι)).toSphere)) =
      ∏ i, ∫ t : ℝ, t ^ k i * Real.exp (-(t ^ 2)) := by
  have h := homogeneous_gaussian_polar (coordinateMonomial k) (∑ i, k i)
    (coordinateMonomial_homogeneous k) (integrable_coordinate_gaussian k)
  rw [integral_coordinate_gaussian] at h
  simpa only [finrank_euclideanSpace] using h.symm

/-- Closed formula for even coordinate moments of actual surface measure. -/
theorem sphere_even_monomial_moment [Nonempty ι] (k : ι → ℕ) :
    (∫ θ : sphere (0 : EuclideanSpace ℝ ι) 1,
        coordinateMonomial (fun i ↦ 2 * k i) θ.val
        ∂((volume : Measure (EuclideanSpace ℝ ι)).toSphere)) =
      2 * (∏ i, Real.Gamma ((k i : ℝ) + 1 / 2)) /
        Real.Gamma ((∑ i, (k i : ℝ)) + (Fintype.card ι : ℝ) / 2) := by
  have h := sphere_monomial_moment_scaled (fun i ↦ 2 * k i)
  simp_rw [integral_even_pow_gaussian] at h
  have hcard : 0 < Fintype.card ι := Fintype.card_pos
  have hexponent :
      (((∑ i, 2 * k i) + (Fintype.card ι - 1) : ℕ) : ℝ) / 2 + 1 / 2 =
      (∑ i, (k i : ℝ)) + (Fintype.card ι : ℝ) / 2 := by
    push_cast
    rw [Nat.cast_sub (by omega : 1 ≤ Fintype.card ι)]
    push_cast
    rw [← Finset.mul_sum]
    ring
  rw [hexponent] at h
  have hg : Real.Gamma ((∑ i, (k i : ℝ)) + (Fintype.card ι : ℝ) / 2) ≠ 0 :=
    (Real.Gamma_pos_of_pos (by
      have hc : (0 : ℝ) < Fintype.card ι := by exact_mod_cast hcard
      have hs : (0 : ℝ) ≤ ∑ i, (k i : ℝ) := Finset.sum_nonneg (fun i _ ↦ Nat.cast_nonneg _)
      linarith)).ne'
  apply (eq_div_iff hg).mpr
  linarith

/-- If any coordinate exponent is odd, the actual spherical moment is zero. -/
theorem sphere_monomial_moment_zero_of_odd [Nonempty ι]
    (k : ι → ℕ) (j : ι) (hj : Odd (k j)) :
    (∫ θ : sphere (0 : EuclideanSpace ℝ ι) 1,
        coordinateMonomial k θ.val
        ∂((volume : Measure (EuclideanSpace ℝ ι)).toSphere)) = 0 := by
  have h := sphere_monomial_moment_scaled k
  obtain ⟨l, hl⟩ := hj
  have hjint : (∫ t : ℝ, t ^ k j * Real.exp (-(t ^ 2))) = 0 := by
    rw [hl]
    exact integral_odd_pow_gaussian l
  rw [Finset.prod_eq_zero (Finset.mem_univ j) hjint] at h
  have hg : Real.Gamma (((∑ i, k i) + (Fintype.card ι - 1) : ℕ) / 2 + 1 / 2) / 2 ≠ 0 := by
    apply div_ne_zero _ (by norm_num)
    apply (Real.Gamma_pos_of_pos _).ne'
    have hn : (0 : ℝ) ≤ ((∑ i, k i) + (Fintype.card ι - 1) : ℕ) := Nat.cast_nonneg _
    linarith
  exact (mul_eq_zero.mp h).resolve_left hg

/-- Surface area obtained from the zero-degree Gaussian moment. -/
theorem surface_mass [Nonempty ι] :
    (volume : Measure (EuclideanSpace ℝ ι)).toSphere.real Set.univ =
      2 * Real.Gamma (1 / 2) ^ Fintype.card ι /
        Real.Gamma ((Fintype.card ι : ℝ) / 2) := by
  have h := sphere_even_monomial_moment (ι := ι) (fun _ ↦ 0)
  simpa [coordinateMonomial] using h

/-- Normalized integral over the actual Euclidean unit sphere. -/
noncomputable def sphereAverage (f : sphere (0 : EuclideanSpace ℝ ι) 1 → ℝ) : ℝ :=
  (∫ θ, f θ ∂((volume : Measure (EuclideanSpace ℝ ι)).toSphere)) /
    (volume : Measure (EuclideanSpace ℝ ι)).toSphere.real Set.univ

/-- Exact normalized even moments; these are the target identities to which the
recursive Jacobi rule must be matched. -/
theorem sphereAverage_even_monomial [Nonempty ι] (k : ι → ℕ) :
    sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ ι) 1 ↦
      coordinateMonomial (fun i ↦ 2 * k i) θ.val) =
      Real.Gamma ((Fintype.card ι : ℝ) / 2) *
        (∏ i, Real.Gamma ((k i : ℝ) + 1 / 2)) /
      (Real.Gamma ((∑ i, (k i : ℝ)) + (Fintype.card ι : ℝ) / 2) *
        Real.Gamma (1 / 2) ^ Fintype.card ι) := by
  rw [sphereAverage, sphere_even_monomial_moment, surface_mass]
  have hc : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos (α := ι)
  have hs : (0 : ℝ) ≤ ∑ i, (k i : ℝ) := Finset.sum_nonneg (fun i _ ↦ Nat.cast_nonneg _)
  have hg₀ : Real.Gamma (1 / 2) ≠ 0 := (Real.Gamma_pos_of_pos (by norm_num)).ne'
  have hgd : Real.Gamma ((Fintype.card ι : ℝ) / 2) ≠ 0 :=
    (Real.Gamma_pos_of_pos (by linarith)).ne'
  have hgk : Real.Gamma ((∑ i, (k i : ℝ)) + (Fintype.card ι : ℝ) / 2) ≠ 0 :=
    (Real.Gamma_pos_of_pos (by linarith)).ne'
  field_simp

theorem sphereAverage_monomial_zero_of_odd [Nonempty ι]
    (k : ι → ℕ) (j : ι) (hj : Odd (k j)) :
    sphereAverage (fun θ : sphere (0 : EuclideanSpace ℝ ι) 1 ↦
      coordinateMonomial k θ.val) = 0 := by
  rw [sphereAverage, sphere_monomial_moment_zero_of_odd k j hj, zero_div]

end Coordinates

end QIRadon.Gaussian

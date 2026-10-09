import QIRadon.Radon.RotationLift

/-! Entire scalar evaluation along a complex rotation word. The continuation
uses the complex bilinear coordinate dot product, not a Hermitian product of two
variable arguments. Its real restriction is the actual rotated real projection.
-/
noncomputable section
open Set Metric
open scoped BigOperators
namespace QIRadon.Radon
open QIRadon.ComplexRotation
variable {ι : Type*} [Fintype ι]

/-- The complex bilinear extension of the real Euclidean dot product. -/
def complexBilinear (u v : EuclideanSpace ℂ ι) : ℂ := ∑ i, u i * v i

@[simp] theorem complexBilinear_sub_left (u v w : EuclideanSpace ℂ ι) :
    complexBilinear (u-v) w = complexBilinear u w-complexBilinear v w := by
  simp [complexBilinear, sub_mul, Finset.sum_sub_distrib]

/-- With a fixed real second vector, the bilinear dot product equals the
Hermitian product with that real vector in the conjugate slot. This identity
justifies the norm estimate; the varying argument remains complex-linear. -/
theorem complexBilinear_real_right (u : EuclideanSpace ℂ ι) (x : EuclideanSpace ℝ ι) :
    complexBilinear u (realEmbedding x) = inner ℂ (realEmbedding x) u := by
  simp [complexBilinear, PiLp.inner_apply, RCLike.inner_apply]

@[simp] theorem complexBilinear_reals (θ x : EuclideanSpace ℝ ι) :
    complexBilinear (realEmbedding θ) (realEmbedding x) = (inner ℝ θ x : ℂ) := by
  simp [complexBilinear, PiLp.inner_apply, RCLike.inner_apply, mul_comm]

theorem complexBilinear_real_norm_le (u : EuclideanSpace ℂ ι) (x : EuclideanSpace ℝ ι) :
    ‖complexBilinear u (realEmbedding x)‖ ≤ ‖u‖*‖x‖ := by
  rw [complexBilinear_real_right]
  have hh := norm_inner_le_norm (𝕜 := ℂ) (realEmbedding x) u
  simpa only [realEmbedding.norm_map, mul_comm] using hh

variable [DecidableEq ι]

/-- Actual scalar coordinate of the complex-rotated direction at input x. -/
def rotationEvaluation {n : ℕ} (pairs : Fin n → ι × ι)
    (θ x : EuclideanSpace ℝ ι) (angles : Fin n → ℂ) : ℂ :=
  complexBilinear (rotationWord pairs angles (realEmbedding θ)) (realEmbedding x)

/-- Entire dependence on every complex angle, jointly. -/
theorem rotationEvaluation_entire {n : ℕ} (pairs : Fin n → ι × ι)
    (θ x : EuclideanSpace ℝ ι) :
    AnalyticOnNhd ℂ (rotationEvaluation pairs θ x) univ := by
  intro a _
  have hrot := rotationWord_entire pairs (realEmbedding θ) a (mem_univ _)
  have hh := ((innerSL ℂ (realEmbedding x)).analyticAt
    (rotationWord pairs a (realEmbedding θ))).comp
      (f := fun z : Fin n → ℂ => rotationWord pairs z (realEmbedding θ)) hrot
  unfold rotationEvaluation
  simp_rw [complexBilinear_real_right]
  exact hh

@[simp] theorem rotationEvaluation_zero {n : ℕ} (pairs : Fin n → ι × ι)
    (θ x : EuclideanSpace ℝ ι) : rotationEvaluation pairs θ x 0 = (inner ℝ θ x : ℂ) := by
  simp [rotationEvaluation]

/-- Exact dimension-independent displacement from the original real coordinate.
The same ℓ¹ angle budget as the actual vector rotation is retained. -/
theorem rotationEvaluation_displacement {n : ℕ} (pairs : Fin n → ι × ι)
    (θ x : EuclideanSpace ℝ ι) (angles : Fin n → ℂ) :
    ‖rotationEvaluation pairs θ x angles - (inner ℝ θ x : ℂ)‖ ≤
      (Real.exp (2*∑ i, ‖angles i‖)-1)*‖θ‖*‖x‖ := by
  have hh := complexBilinear_real_norm_le
    (rotationWord pairs angles (realEmbedding θ)-realEmbedding θ) x
  rw [complexBilinear_sub_left, complexBilinear_reals] at hh
  change ‖rotationEvaluation pairs θ x angles-(inner ℝ θ x : ℂ)‖ ≤ _ at hh
  apply hh.trans
  have hd := rotationWord_displacement pairs angles (realEmbedding θ)
  rw [realEmbedding.norm_map] at hd
  exact mul_le_mul_of_nonneg_right hd (norm_nonneg x)

/-- Unit-direction specialization used by the Radon angular continuation. -/
theorem rotationEvaluation_unit_displacement {n : ℕ} (pairs : Fin n → ι × ι)
    (θ : sphere (0:EuclideanSpace ℝ ι) 1) (x : EuclideanSpace ℝ ι) (angles : Fin n → ℂ) :
    ‖rotationEvaluation pairs θ.val x angles - (inner ℝ θ.val x : ℂ)‖ ≤
      (Real.exp (2*∑ i, ‖angles i‖)-1)*‖x‖ := by
  have hθ : ‖θ.val‖ = 1 := mem_sphere_zero_iff_norm.mp θ.property
  simpa only [hθ, mul_one] using rotationEvaluation_displacement pairs θ.val x angles

/-- Explicit domain safety for the actual scalar coordinate. Strict angle budget
keeps the point in the open scalar tube even when the margin bound is an equality. -/
theorem rotationEvaluation_mem_scalarTube {n : ℕ} (pairs : Fin n → ι × ι)
    (θ : sphere (0:EuclideanSpace ℝ ι) 1) (x : EuclideanSpace ℝ ι)
    {angles : Fin n → ℂ} {L δ η : ℝ} (hx : ‖x‖ ≤ 1) (hL : 1 ≤ L)
    (hangles : (∑ i, ‖angles i‖) < η) (hmargin : Real.exp (2*η)-1 ≤ δ) :
    rotationEvaluation pairs θ.val x angles ∈ scalarTube L δ := by
  have hθ : ‖θ.val‖ = 1 := mem_sphere_zero_iff_norm.mp θ.property
  have hbase : |inner ℝ θ.val x| ≤ L := by
    have hi := norm_inner_le_norm (𝕜 := ℝ) θ.val x
    rw [Real.norm_eq_abs, hθ, one_mul] at hi
    exact (hi.trans hx).trans hL
  refine mem_iUnion.mpr ⟨inner ℝ θ.val x, mem_iUnion.mpr ⟨abs_le.mp hbase, ?_⟩⟩
  rw [mem_ball_iff_norm]
  have hfactor : 0 ≤ Real.exp (2*∑ i, ‖angles i‖)-1 := by
    apply sub_nonneg.mpr
    apply Real.one_le_exp_iff.mpr
    positivity
  have hstrict : Real.exp (2*∑ i, ‖angles i‖)-1 < Real.exp (2*η)-1 := by
    have hh := Real.exp_lt_exp.mpr (show 2*∑ i, ‖angles i‖ < 2*η by linarith)
    linarith
  have hd := rotationEvaluation_unit_displacement pairs θ x angles
  have hnorm := mul_le_mul_of_nonneg_left hx hfactor
  rw [mul_one] at hnorm
  exact (hd.trans hnorm).trans_lt (hstrict.trans_le hmargin)

/-- One explicit admissible scalar rotation budget. It is independent of word
length and input dimension; other construction margins can reduce it further. -/
def scalarRotationBudget (δ : ℝ) : ℝ := Real.log (1+δ)/2

theorem scalarRotationBudget_pos {δ : ℝ} (hδ : 0 < δ) : 0 < scalarRotationBudget δ := by
  unfold scalarRotationBudget
  apply half_pos
  exact Real.log_pos (by linarith)

theorem scalarRotationBudget_margin {δ : ℝ} (hδ : 0 < δ) :
    Real.exp (2*scalarRotationBudget δ)-1 = δ := by
  unfold scalarRotationBudget
  rw [show 2*(Real.log (1+δ)/2)=Real.log (1+δ) by ring,
    Real.exp_log (by linarith : 0 < 1+δ)]
  ring

/-- At real angles this is exactly the real rotated projection, including all
diagonal (identity) pairs. No coordinate-order or sign convention is changed. -/
theorem rotationEvaluation_ofReal {n : ℕ} (pairs : Fin n → ι × ι)
    (θ x : EuclideanSpace ℝ ι) (angles : Fin n → ℝ) :
    rotationEvaluation pairs θ x (fun i => (angles i : ℂ)) =
      (inner ℝ (totalRealRotationWord pairs angles θ) x : ℂ) := by
  unfold rotationEvaluation
  rw [rotationWord_ofReal_total, complexBilinear_reals]

end QIRadon.Radon

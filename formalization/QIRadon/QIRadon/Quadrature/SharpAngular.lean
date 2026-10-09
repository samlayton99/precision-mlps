import QIRadon.Quadrature.HarmonicGenerating
import QIRadon.Quadrature.CanonicalConvergence

noncomputable section
open MeasureTheory
namespace QIRadon.SharpAngular
open QIRadon.Gaussian QIRadon.HarmonicGenerating QIRadon.SobolevTail

/-- Exact positive dimension trace at the analytic weight. -/
def traceConstant (n : ℕ) (a : ℝ) : ℝ :=
  (1 + Real.exp (-a)) / (1 - Real.exp (-a))^n

theorem traceConstant_nonneg (n : ℕ) {a : ℝ} (ha : 0 < a) :
    0 ≤ traceConstant n a := by
  have hr : Real.exp (-a) < 1 := Real.exp_lt_one_iff.mpr (neg_lt_zero.mpr ha)
  unfold traceConstant
  positivity

def shape (n : ℕ) (a : ℝ) (k : ℕ) : ℝ :=
  Real.sqrt (Module.finrank ℝ (HarmonicSpace n k)) * Real.exp (-a*k)

theorem shape_sq (n k : ℕ) (a : ℝ) :
    (shape n a k)^2 = (Module.finrank ℝ (HarmonicSpace n k) : ℝ) * Real.exp (-2*a*k) := by
  unfold shape
  rw [mul_pow, Real.sq_sqrt (Nat.cast_nonneg _), ← Real.exp_nat_mul]
  congr 2
  norm_num
  ring

theorem trace_series (n : ℕ) {a : ℝ} (ha : 0 < a) :
    HasSum (fun k : ℕ => (Module.finrank ℝ (HarmonicSpace n k) : ℝ) * Real.exp (-a*k))
      (traceConstant n a) := by
  have hr : Real.exp (-a) < 1 := Real.exp_lt_one_iff.mpr (neg_lt_zero.mpr ha)
  have hs := harmonic_dimension_hasSum n (Real.exp_pos (-a)).le hr
  have heq : (fun k : ℕ => (Module.finrank ℝ (HarmonicSpace n k) : ℝ) * Real.exp (-a*k)) =
      (fun k : ℕ => (Module.finrank ℝ (HarmonicSpace n k) : ℝ) * (Real.exp (-a))^k) := by
    funext k
    rw [← Real.exp_nat_mul]
    congr 2
    ring
  rw [heq]
  exact hs

theorem shape_summable (n : ℕ) {a : ℝ} (ha : 0 < a) :
    Summable (fun k => (shape n a k)^2) := by
  have hs := (trace_series n (a := 2*a) (by linarith)).summable
  simpa only [shape_sq, neg_mul] using hs

/-- Trace tail retains the full dimension generating function, rather than
replacing it by a factorial estimate for separate components. -/
theorem shape_tail (n N : ℕ) {a : ℝ} (ha : 0 < a) :
    (∑' j, (shape n a (j+N))^2) ≤ Real.exp (-a*N) * traceConstant n a := by
  have htr := trace_series n ha
  have htrtail := (summable_nat_add_iff N).mpr htr.summable
  have hs := (summable_nat_add_iff N).mpr (shape_summable n ha)
  have hp : ∀ j : ℕ, (shape n a (j+N))^2 ≤
      Real.exp (-a*N) * ((Module.finrank ℝ (HarmonicSpace n (j+N)) : ℝ) * Real.exp (-a*((j+N:ℕ):ℝ))) := by
    intro j
    rw [shape_sq]
    have he : Real.exp (-a*((j+N : ℕ) : ℝ)) ≤ Real.exp (-a*N) := by
      apply Real.exp_le_exp.mpr
      push_cast
      nlinarith [Nat.cast_nonneg (α := ℝ) j]
    have hid : Real.exp (-2*a*((j+N : ℕ) : ℝ)) =
        Real.exp (-a*((j+N : ℕ) : ℝ)) * Real.exp (-a*((j+N : ℕ) : ℝ)) := by
      rw [← Real.exp_add]
      congr 1
      ring
    rw [hid]
    have hh := mul_le_mul_of_nonneg_left he
      (show 0 ≤ (Module.finrank ℝ (HarmonicSpace n (j+N)) : ℝ) * Real.exp (-a*((j+N : ℕ) : ℝ)) by positivity)
    convert! hh using 1 <;> ring
  have hb := hs.tsum_le_tsum hp (htrtail.mul_left (Real.exp (-a*N)))
  rw [tsum_mul_left] at hb
  have ht := shifted_energy_le
    (fun k => show 0 ≤ (Module.finrank ℝ (HarmonicSpace n k) : ℝ) * Real.exp (-a*k) by positivity)
    htr.summable N
  rw [htr.tsum_eq] at ht
  exact hb.trans (mul_le_mul_of_nonneg_left ht (Real.exp_pos _).le)

/-- Uniform harmonic-series tail with the exact trace constant. -/
theorem harmonic_partial_error (n N : ℕ) (v : (k : ℕ) → HarmonicSpace n k)
    {a A : ℝ} (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ => Real.exp (2*a*k) * ‖v k‖^2))
    (henergy : (∑' k : ℕ, Real.exp (2*a*k) * ‖v k‖^2) ≤ A^2)
    (θ : UnitSphere n) :
    |(∑' k, harmonicEvaluation n k θ (v k)) -
      ∑ k ∈ Finset.range N, harmonicEvaluation n k θ (v k)| ≤
      A * Real.exp (-(a/2)*N) * Real.sqrt (traceConstant n a) := by
  let u := fun k : ℕ => Real.exp (a*k) * ‖v k‖
  have hu : ∀ k, 0 ≤ u k := by intro k; dsimp [u]; positivity
  have hv : ∀ k, 0 ≤ shape n a k := by intro k; unfold shape; positivity
  have hu_eq : (fun k => (u k)^2) = (fun k : ℕ => Real.exp (2*a*k) * ‖v k‖^2) := by
    funext k
    dsimp [u]
    rw [mul_pow, ← Real.exp_nat_mul]
    congr 2
    norm_num
    ring
  have hu_sum : Summable (fun k => (u k)^2) := by rw [hu_eq]; exact hsum
  have hu_energy : (∑' k, (u k)^2) ≤ A^2 := by rw [hu_eq]; exact henergy
  have hh := energy_times_shape_tail hu hv hA hu_sum (shape_summable n ha) hu_energy N
  have hid : ∀ k, u k * shape n a k =
      Real.sqrt (Module.finrank ℝ (HarmonicSpace n k)) * ‖v k‖ := by
    intro k
    change Real.exp (a*k) * ‖v k‖ *
      (Real.sqrt (Module.finrank ℝ (HarmonicSpace n k)) * Real.exp (-a*k)) =
      Real.sqrt (Module.finrank ℝ (HarmonicSpace n k)) * ‖v k‖
    have he : Real.exp (a*k) * Real.exp (-a*k) = 1 := by
      rw [← Real.exp_add]
      simp [neg_mul]
    calc
      _ = Real.sqrt (Module.finrank ℝ (HarmonicSpace n k)) * ‖v k‖ *
          (Real.exp (a*k) * Real.exp (-a*k)) := by ring
      _ = _ := by rw [he, mul_one]
  have hp : ∀ k, ‖harmonicEvaluation n k θ (v k)‖ ≤ u k * shape n a k := by
    intro k
    rw [Real.norm_eq_abs, hid]
    exact harmonicEvaluation_bound_dimension n k θ (v k)
  have hP := hh.1.of_norm_bounded hp
  have hPt := (summable_nat_add_iff N).mpr hP
  have hi := hP.sum_add_tsum_nat_add N
  have heq : (∑' k, harmonicEvaluation n k θ (v k)) -
      ∑ k ∈ Finset.range N, harmonicEvaluation n k θ (v k) =
      ∑' j, harmonicEvaluation n (j+N) θ (v (j+N)) := by linarith
  rw [heq]
  have hb := hPt.norm.tsum_le_tsum (fun j => hp (j+N)) ((summable_nat_add_iff N).mpr hh.1)
  have hshape := shape_tail n N ha
  calc
    _ ≤ ∑' j, ‖harmonicEvaluation n (j+N) θ (v (j+N))‖ := norm_tsum_le_tsum_norm hPt.norm
    _ ≤ ∑' j, u (j+N) * shape n a (j+N) := hb
    _ ≤ A * Real.sqrt (∑' j, (shape n a (j+N))^2) := hh.2
    _ ≤ A * Real.sqrt (Real.exp (-a*N) * traceConstant n a) :=
      mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hshape) hA
    _ = _ := by
      rw [Real.sqrt_mul (Real.exp_pos _).le, Real.sqrt_eq_rpow,
        Real.rpow_def_of_pos (Real.exp_pos _), Real.log_exp]
      have he : (-a * (N : ℝ)) * (1/2) = -(a/2)*N := by ring
      rw [he]
      ring

/-- Sharp-constant analytic convergence of the actual folded sphere quadrature
from the target's canonical harmonic energy. Exact preservation of the mean
leaves only the positive weighted tail error, with no factor two. -/
theorem canonical_quadrature_error (q : ℕ) (hq : 0 < q) (n : ℕ)
    (F : C(UnitSphere n, ℝ)) (heven : ∀ θ, F (sphereNeg n θ) = F θ)
    {a A : ℝ} (ha : 0 < a) (hA : 0 ≤ A)
    (hsum : Summable (fun k : ℕ => Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n F k‖^2))
    (henergy : (∑' k : ℕ, Real.exp (2*a*k) * ‖continuousHarmonicCoordinates n F k‖^2) ≤ A^2) :
    |(∫ θ, F θ ∂normalizedSphere n) -
      ∑ i : SphereHalfIndex q n, foldedSphereWeight q hq n i * F (sphereNodeOnSphere q hq n i.val)| ≤
      A * Real.sqrt (traceConstant n a) * Real.exp (-a*q) := by
  let v := continuousHarmonicCoordinates n F
  let P : UnitSphere n → ℝ := fun θ =>
    ∑ k ∈ Finset.range (2*q), harmonicEvaluation n k θ (v k)
  let err := A * Real.exp (-a*q) * Real.sqrt (traceConstant n a)
  have hp : ∀ θ, |F θ - P θ| ≤ err := by
    intro θ
    have hh := harmonic_partial_error n (2*q) v ha hA hsum henergy θ
    have he := canonical_harmonic_pointwise_analytic n F a A ha hA hsum henergy θ
    rw [he]
    have hexp : -(a/2)*((2*q : ℕ):ℝ) = -a*q := by push_cast; ring
    simpa only [P, v, err, hexp] using hh
  have herr := canonical_foldedSphere_error_of_partial q hq n F heven err hp
  apply herr.trans_eq
  dsimp [err]
  ring

end QIRadon.SharpAngular

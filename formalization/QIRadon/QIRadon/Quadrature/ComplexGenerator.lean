import QIRadon.Quadrature.ComplexPlaneRotation
import QIRadon.Quadrature.CasimirEstimate
import QIRadon.Quadrature.RotationFlow
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Analysis.Complex.Liouville
import Mathlib.Analysis.Calculus.ContDiff.WithLp

/-! Actual complex rotation generators and direct one-variable Cauchy bounds.
The domain assumptions concern a complex extension and its values on actual
rotation products. No spectral norm or derivative bound is an input. -/

open MeasureTheory Set Metric
open scoped ContDiff BigOperators

namespace QIRadon.Casimir

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 200000

open Gaussian ComplexRotation

variable {n : ℕ}

def complexGeneratorLinear (i j : Fin (n+1)) :
    EuclideanSpace ℂ (Fin (n+1)) →ₗ[ℂ] EuclideanSpace ℂ (Fin (n+1)) where
  toFun x := if i=j then 0 else planeGenerator i j x
  map_add' x y := by
    by_cases hij : i=j
    · simp [hij]
    · ext k
      simp only [hij, if_false, planeGenerator, PiLp.add_apply]
      split_ifs <;> simp [add_comm]
  map_smul' c x := by
    by_cases hij : i=j
    · simp [hij]
    · ext k
      simp only [hij, if_false, planeGenerator, PiLp.smul_apply, RingHom.id_apply, smul_eq_mul]
      split_ifs <;> simp

def complexGenerator (i j : Fin (n+1)) :
    EuclideanSpace ℂ (Fin (n+1)) →L[ℂ] EuclideanSpace ℂ (Fin (n+1)) :=
  (complexGeneratorLinear i j).toContinuousLinearMap

def complexRotationDerivative (i j : Fin (n+1))
    (G : EuclideanSpace ℂ (Fin (n+1)) → ℂ) (x : EuclideanSpace ℂ (Fin (n+1))) : ℂ :=
  fderiv ℂ G x (complexGenerator i j x)

theorem complexRotationDerivative_contDiffOn {U : Set (EuclideanSpace ℂ (Fin (n+1)))}
    (hU : IsOpen U) {G : EuclideanSpace ℂ (Fin (n+1)) → ℂ}
    (hG : ContDiffOn ℂ ∞ G U) (i j : Fin (n+1)) :
    ContDiffOn ℂ ∞ (complexRotationDerivative i j G) U :=
  (hG.fderiv_of_isOpen hU (by simp)).clm_apply (complexGenerator i j).contDiff.contDiffOn

def complexRotationDerivativeWord (G : EuclideanSpace ℂ (Fin (n+1)) → ℂ) :
    List (RotationPair n) → EuclideanSpace ℂ (Fin (n+1)) → ℂ
  | [] => G
  | p::w => complexRotationDerivative p.1 p.2 (complexRotationDerivativeWord G w)

theorem complexRotationDerivativeWord_contDiffOn {U : Set (EuclideanSpace ℂ (Fin (n+1)))}
    (hU : IsOpen U) {G : EuclideanSpace ℂ (Fin (n+1)) → ℂ}
    (hG : ContDiffOn ℂ ∞ G U) (w : List (RotationPair n)) :
    ContDiffOn ℂ ∞ (complexRotationDerivativeWord G w) U := by
  induction w with
  | nil => exact hG
  | cons p w ih => exact complexRotationDerivative_contDiffOn hU ih p.1 p.2

theorem complexRotationDerivative_curve {U : Set (EuclideanSpace ℂ (Fin (n+1)))}
    (hU : IsOpen U) {G : EuclideanSpace ℂ (Fin (n+1)) → ℂ}
    (hG : ContDiffOn ℂ ∞ G U) (i j : Fin (n+1))
    (x : EuclideanSpace ℂ (Fin (n+1))) (hx : x ∈ U) :
    HasDerivAt (fun z : ℂ ↦ G (complexPlaneRotate i j z x))
      (complexRotationDerivative i j G x) 0 := by
  have hf : DifferentiableAt ℂ G x :=
    (hG.differentiableOn (by simp) x hx).differentiableAt (hU.mem_nhds hx)
  have hf' : HasFDerivAt G (fderiv ℂ G x) (complexPlaneRotate i j 0 x) := by
    simpa using hf.hasFDerivAt
  exact hf'.comp_hasDerivAt 0 (complexPlaneRotate_hasDerivAt_zero i j x)

abbrev ComplexRotationStep (n : ℕ) := RotationPair n × ℂ

def complexOrbit : List (ComplexRotationStep n) → EuclideanSpace ℂ (Fin (n+1)) →
    EuclideanSpace ℂ (Fin (n+1))
  | [], x => x
  | p::w, x => complexPlaneRotate p.1.1 p.1.2 p.2 (complexOrbit w x)

def complexOrbitBudget (w : List (ComplexRotationStep n)) : ℝ :=
  (w.map (fun p ↦ ‖p.2‖)).sum

theorem complexOrbitBudget_nonneg (w : List (ComplexRotationStep n)) :
    0 ≤ complexOrbitBudget w := by
  induction w with
  | nil => simp [complexOrbitBudget]
  | cons p w ih => simpa [complexOrbitBudget] using add_nonneg (norm_nonneg p.2) ih

theorem complex_rotation_word_bound {U : Set (EuclideanSpace ℂ (Fin (n+1)))}
    (hU : IsOpen U) {G : EuclideanSpace ℂ (Fin (n+1)) → ℂ}
    (hG : ContDiffOn ℂ ∞ G U) {δ B r : ℝ} (hr : 0 < r)
    (hdom : ∀ (v : List (ComplexRotationStep n)) (θ : UnitSphere n),
      complexOrbitBudget v < δ → complexOrbit v (realEmbedding θ.val) ∈ U)
    (hval : ∀ (v : List (ComplexRotationStep n)) (θ : UnitSphere n),
      complexOrbitBudget v < δ → ‖G (complexOrbit v (realEmbedding θ.val))‖ ≤ B)
    (w : List (RotationPair n)) :
    ∀ (v : List (ComplexRotationStep n)) (θ : UnitSphere n),
      complexOrbitBudget v + (w.length:ℝ)*r < δ →
      ‖complexRotationDerivativeWord G w (complexOrbit v (realEmbedding θ.val))‖ ≤ B / r^w.length := by
  induction w with
  | nil =>
    intro v θ hbudget
    simpa [complexRotationDerivativeWord] using hval v θ (by simpa using hbudget)
  | cons p w ih =>
    intro v θ hbudget
    let x := complexOrbit v (realEmbedding θ.val)
    let P := complexRotationDerivativeWord G w
    have hbudget' : complexOrbitBudget v + ((w.length:ℝ)+1)*r < δ := by
      simpa only [List.length_cons, Nat.cast_add, Nat.cast_one] using hbudget
    have hinside (z : ℂ) (hz : z ∈ closedBall (0:ℂ) r) :
        complexOrbitBudget ((p,z)::v) + (w.length:ℝ)*r < δ := by
      have hz' : ‖z‖ ≤ r := by simpa only [mem_closedBall, dist_zero_right] using hz
      simp only [complexOrbitBudget, List.map_cons, List.sum_cons]
      change ‖z‖ + complexOrbitBudget v + (w.length:ℝ)*r < δ
      linarith
    have hdomain (z : ℂ) (hz : z ∈ closedBall (0:ℂ) r) :
        complexPlaneRotate p.1 p.2 z x ∈ U := by
      apply hdom ((p,z)::v) θ
      have hh := hinside z hz
      have hlen : 0 ≤ (w.length:ℝ)*r := by positivity
      linarith
    have hPd := complexRotationDerivativeWord_contDiffOn hU hG w
    have hdiff : DifferentiableOn ℂ (fun z : ℂ ↦ P (complexPlaneRotate p.1 p.2 z x))
        (closedBall (0:ℂ) r) := by
      intro z hz
      have hP := (hPd.differentiableOn (by simp) _ (hdomain z hz)).differentiableAt
        (hU.mem_nhds (hdomain z hz))
      apply (hP.comp z ?_).differentiableWithinAt
      exact ((complexPlaneRotate_entire p.1 p.2 x) z trivial).differentiableAt
    have hdc : DiffContOnCl ℂ (fun z : ℂ ↦ P (complexPlaneRotate p.1 p.2 z x))
        (ball (0:ℂ) r) := by
      apply DifferentiableOn.diffContOnCl
      rwa [closure_ball _ hr.ne']
    have hb : ∀ z ∈ sphere (0:ℂ) r, ‖P (complexPlaneRotate p.1 p.2 z x)‖ ≤ B/r^w.length := by
      intro z hz
      exact ih ((p,z)::v) θ (hinside z (sphere_subset_closedBall hz))
    have hc := Complex.norm_deriv_le_of_forall_mem_sphere_norm_le hr hdc hb
    have hx : x ∈ U := by simpa using hdomain 0 (by simpa using hr.le)
    have hder := (complexRotationDerivative_curve hU hPd p.1 p.2 x hx).deriv
    rw [hder] at hc
    apply hc.trans_eq
    simp only [List.length_cons, pow_succ, div_div]


theorem complex_rotation_word_zero_bound {U : Set (EuclideanSpace ℂ (Fin (n+1)))}
    (hU : IsOpen U) {G : EuclideanSpace ℂ (Fin (n+1)) → ℂ}
    (hG : ContDiffOn ℂ ∞ G U) {δ B : ℝ} (hδ : 0 < δ)
    (hdom : ∀ (v : List (ComplexRotationStep n)) (θ : UnitSphere n),
      complexOrbitBudget v < δ → complexOrbit v (realEmbedding θ.val) ∈ U)
    (hval : ∀ (v : List (ComplexRotationStep n)) (θ : UnitSphere n),
      complexOrbitBudget v < δ → ‖G (complexOrbit v (realEmbedding θ.val))‖ ≤ B)
    (w : List (RotationPair n)) (θ : UnitSphere n) :
    ‖complexRotationDerivativeWord G w (realEmbedding θ.val)‖ ≤
      B * (2*(w.length:ℝ)/δ)^w.length := by
  by_cases hw : w.length = 0
  · have hw' : w = [] := List.length_eq_zero_iff.mp hw
    subst w
    simpa [complexRotationDerivativeWord, complexOrbit] using hval [] θ (by simpa [complexOrbitBudget] using hδ)
  · have hl : 0 < (w.length:ℝ) := Nat.cast_pos.mpr (Nat.pos_of_ne_zero hw)
    have hr : 0 < δ/(2*(w.length:ℝ)) := by positivity
    have hbudget : complexOrbitBudget ([] : List (ComplexRotationStep n)) +
        (w.length:ℝ)*(δ/(2*(w.length:ℝ))) < δ := by
      simp only [complexOrbitBudget, List.map_nil, List.sum_nil, zero_add]
      have he : (w.length:ℝ)*(δ/(2*(w.length:ℝ))) = δ/2 := by field_simp
      rw [he]
      linarith
    have h := complex_rotation_word_bound hU hG hr hdom hval w [] θ hbudget
    apply h.trans_eq
    rw [div_eq_mul_inv, ← inv_pow, inv_div]

def complexSphereJet {U : Set (EuclideanSpace ℂ (Fin (n+1)))}
    (hU : IsOpen U) (hcontains : ∀ θ : UnitSphere n, realEmbedding θ.val ∈ U)
    (G : EuclideanSpace ℂ (Fin (n+1)) → ℂ) (hG : ContDiffOn ℂ ∞ G U) : RotationJet n :=
  fun w ↦ ⟨fun θ ↦ (complexRotationDerivativeWord G w (realEmbedding θ.val)).re, by
    apply Complex.continuous_re.comp
    exact (complexRotationDerivativeWord_contDiffOn hU hG w).continuousOn.comp_continuous
      (realEmbedding.continuous.comp continuous_subtype_val) hcontains⟩

theorem complexPlaneRotate_real_flow (i j : Fin (n+1)) (t : ℝ) (θ : UnitSphere n) :
    complexPlaneRotate i j (t:ℂ) (realEmbedding θ.val) =
      realEmbedding (rotationFlow i j t θ).val := by
  by_cases hij : i=j
  · simp [complexPlaneRotate, rotationFlow, hij]
  · simp only [rotationFlow, dif_neg hij]
    convert complexPlaneRotate_real i j hij t θ.val using 1
    rfl

theorem complexSphereJet_isRotationJet {U : Set (EuclideanSpace ℂ (Fin (n+1)))}
    (hU : IsOpen U) (hcontains : ∀ θ : UnitSphere n, realEmbedding θ.val ∈ U)
    (G : EuclideanSpace ℂ (Fin (n+1)) → ℂ) (hG : ContDiffOn ℂ ∞ G U) :
    IsRotationJet (complexSphereJet hU hcontains G hG) := by
  intro w i j
  apply hasRotationDerivative_of_zero
  intro θ
  have h := complexRotationDerivative_curve hU
    (complexRotationDerivativeWord_contDiffOn hU hG w) i j (realEmbedding θ.val) (hcontains θ)
  have hr := h.real_of_complex
  simp_rw [complexPlaneRotate_real_flow] at hr
  exact hr

theorem complexSphereJet_word_bound {U : Set (EuclideanSpace ℂ (Fin (n+1)))}
    (hU : IsOpen U) (hcontains : ∀ θ : UnitSphere n, realEmbedding θ.val ∈ U)
    (G : EuclideanSpace ℂ (Fin (n+1)) → ℂ) (hG : ContDiffOn ℂ ∞ G U)
    {δ B : ℝ} (hδ : 0 < δ) (hB : 0 ≤ B)
    (hdom : ∀ (v : List (ComplexRotationStep n)) (θ : UnitSphere n),
      complexOrbitBudget v < δ → complexOrbit v (realEmbedding θ.val) ∈ U)
    (hval : ∀ (v : List (ComplexRotationStep n)) (θ : UnitSphere n),
      complexOrbitBudget v < δ → ‖G (complexOrbit v (realEmbedding θ.val))‖ ≤ B)
    (w : List (RotationPair n)) :
    ‖complexSphereJet hU hcontains G hG w‖ ≤ B*(2*(w.length:ℝ)/δ)^w.length := by
  apply (ContinuousMap.norm_le _ (by positivity)).mpr
  intro θ
  change |(complexRotationDerivativeWord G w (realEmbedding θ.val)).re| ≤ _
  exact (Complex.abs_re_le_norm _).trans
    (complex_rotation_word_zero_bound hU hG hδ hdom hval w θ)

end
end QIRadon.Casimir

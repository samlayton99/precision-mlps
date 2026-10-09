import QIRadon.Quadrature.MixedCauchy
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.Calculus.ContDiff.RestrictScalars

/-! Real restrictions and affine coordinate changes for actual mixed
 derivatives. These lemmas identify analytic orbit derivatives with real Lie
 derivatives; they contain no growth or spectral assumptions. -/

open Set
open scoped ContDiff Topology

namespace QIRadon.MixedRestriction

noncomputable section
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 200000

def realCoordinates (n : ℕ) : (Fin n → ℝ) →L[ℝ] (Fin n → ℂ) :=
  ContinuousLinearMap.pi (fun i ↦ Complex.ofRealCLM.comp (ContinuousLinearMap.proj i))

@[simp] theorem realCoordinates_apply (n : ℕ) (x : Fin n → ℝ) (i : Fin n) :
    realCoordinates n x i = (x i : ℂ) := rfl

@[simp] theorem realCoordinates_single (n : ℕ) (i : Fin n) :
    realCoordinates n (Pi.single i 1) = Pi.single i 1 := by
  ext j
  by_cases hij : j=i <;> simp [hij]

def realMixedPartial {n : ℕ} : List (Fin n) → ((Fin n → ℝ) → ℝ) → (Fin n → ℝ) → ℝ
  | [], F => F
  | i::w, F => fun x ↦ fderiv ℝ (realMixedPartial w F) x (Pi.single i 1)

theorem real_restriction_derivative {n : ℕ} {P : (Fin n → ℂ) → ℂ}
    {x : Fin n → ℝ} (hP : DifferentiableAt ℂ P (realCoordinates n x)) (i : Fin n) :
    fderiv ℝ (fun y : Fin n → ℝ ↦ (P (realCoordinates n y)).re) x (Pi.single i 1) =
      (fderiv ℂ P (realCoordinates n x) (Pi.single i 1)).re := by
  have h1 := hP.hasFDerivAt.restrictScalars ℝ
  have h2 := h1.comp x (realCoordinates n).hasFDerivAt
  have h3 := Complex.reCLM.hasFDerivAt.comp x h2
  have h := congrArg (fun L : (Fin n → ℝ) →L[ℝ] ℝ ↦ L (Pi.single i 1)) h3.fderiv
  simp only [ContinuousLinearMap.comp_apply, realCoordinates_single, Complex.reCLM_apply] at h
  convert h using 1 <;> rfl

theorem mixedPartial_real_restriction {n : ℕ} {U : Set (Fin n → ℂ)}
    (hU : IsOpen U) {P : (Fin n → ℂ) → ℂ} (hP : ContDiffOn ℂ ∞ P U)
    (w : List (Fin n)) :
    ∀ x : Fin n → ℝ, realCoordinates n x ∈ U →
      realMixedPartial w (fun y ↦ (P (realCoordinates n y)).re) x =
        (MixedCauchy.mixedPartial w P (realCoordinates n x)).re := by
  induction w with
  | nil => intro x hx; rfl
  | cons i w ih =>
    intro x hx
    have heq : realMixedPartial w (fun y ↦ (P (realCoordinates n y)).re) =ᶠ[𝓝 x]
        (fun y ↦ (MixedCauchy.mixedPartial w P (realCoordinates n y)).re) := by
      have hmem := (hU.preimage (realCoordinates n).continuous).mem_nhds hx
      filter_upwards [hmem] with y hy
      exact ih y hy
    change fderiv ℝ (realMixedPartial w (fun y ↦ (P (realCoordinates n y)).re)) x
      (Pi.single i 1) = _
    rw [heq.fderiv_eq]
    apply real_restriction_derivative
    exact ((MixedCauchy.mixedPartial_smooth hU hP w).differentiableOn (by simp) _ hx).differentiableAt
      (hU.mem_nhds hx)


theorem realMixedPartial_smooth {n : ℕ} {P : (Fin n → ℝ) → ℝ} {U : Set (Fin n → ℝ)}
    (hU : IsOpen U) (hP : ContDiffOn ℝ ∞ P U) (w : List (Fin n)) :
    ContDiffOn ℝ ∞ (realMixedPartial w P) U := by
  induction w with
  | nil => exact hP
  | cons i w ih =>
    exact (ih.fderiv_of_isOpen hU (by simp)).clm_apply contDiffOn_const

theorem real_affine_derivative {m n : ℕ} {P : (Fin n → ℝ) → ℝ}
    (L : (Fin m → ℝ) →L[ℝ] (Fin n → ℝ)) (b : Fin n → ℝ) {x : Fin m → ℝ}
    (hP : DifferentiableAt ℝ P (L x+b)) (i : Fin m) (j : Fin n)
    (hL : L (Pi.single i 1) = Pi.single j 1) :
    fderiv ℝ (fun y : Fin m → ℝ ↦ P (L y+b)) x (Pi.single i 1) =
      fderiv ℝ P (L x+b) (Pi.single j 1) := by
  have h := hP.hasFDerivAt.comp x (L.hasFDerivAt.add_const b)
  have he := congrArg (fun A : (Fin m → ℝ) →L[ℝ] ℝ ↦ A (Pi.single i 1)) h.fderiv
  simp only [ContinuousLinearMap.comp_apply, hL] at he
  convert he using 1
  rfl

theorem realMixedPartial_affine {m n : ℕ} {U : Set (Fin n → ℝ)}
    (hU : IsOpen U) {P : (Fin n → ℝ) → ℝ} (hP : ContDiffOn ℝ ∞ P U)
    (L : (Fin m → ℝ) →L[ℝ] (Fin n → ℝ)) (b : Fin n → ℝ)
    (e : Fin m → Fin n) (hL : ∀ i, L (Pi.single i 1) = Pi.single (e i) 1)
    (w : List (Fin m)) :
    ∀ x : Fin m → ℝ, L x+b ∈ U →
      realMixedPartial w (fun y ↦ P (L y+b)) x =
        realMixedPartial (w.map e) P (L x+b) := by
  induction w with
  | nil => intro x hx; rfl
  | cons i w ih =>
    intro x hx
    have heq : realMixedPartial w (fun y ↦ P (L y+b)) =ᶠ[𝓝 x]
        (fun y ↦ realMixedPartial (w.map e) P (L y+b)) := by
      have hmem := (hU.preimage (L.continuous.add continuous_const)).mem_nhds hx
      filter_upwards [hmem] with y hy
      exact ih y hy
    change fderiv ℝ (realMixedPartial w (fun y ↦ P (L y+b))) x (Pi.single i 1) = _
    rw [heq.fderiv_eq]
    apply real_affine_derivative L b ?_ i (e i) (hL i)
    exact ((realMixedPartial_smooth hU hP (w.map e)).differentiableOn (by simp) _ hx).differentiableAt
      (hU.mem_nhds hx)


theorem realMixedPartial_eventuallyEq {n : ℕ} {F G : (Fin n → ℝ) → ℝ}
    {x : Fin n → ℝ} (h : F =ᶠ[𝓝 x] G) (w : List (Fin n)) :
    realMixedPartial w F =ᶠ[𝓝 x] realMixedPartial w G := by
  induction w with
  | nil => exact h
  | cons i w ih =>
    have hd := ih.fderiv (𝕜 := ℝ)
    filter_upwards [hd] with y hy
    exact congrArg (fun L : (Fin n → ℝ) →L[ℝ] ℝ ↦ L (Pi.single i 1)) hy

def tailInclusion (n : ℕ) : (Fin n → ℝ) →L[ℝ] (Fin (n+1) → ℝ) :=
  ContinuousLinearMap.pi (Fin.cases 0 (fun i ↦ ContinuousLinearMap.proj i))

@[simp] theorem tailInclusion_apply (n : ℕ) (x : Fin n → ℝ) :
    tailInclusion n x = Fin.cons 0 x := by
  ext i
  refine Fin.cases ?_ (fun j ↦ ?_) i <;> rfl

@[simp] theorem tailInclusion_single (n : ℕ) (i : Fin n) :
    tailInclusion n (Pi.single i 1) = Pi.single i.succ 1 := by
  ext j
  refine Fin.cases ?_ (fun k ↦ ?_) j
  · simp
  · simp [Pi.single_apply]

theorem cons_eq_affine (n : ℕ) (t : ℝ) (x : Fin n → ℝ) :
    Fin.cons t x = tailInclusion n x + Pi.single 0 t := by
  ext j
  refine Fin.cases ?_ (fun k ↦ ?_) j <;> simp

theorem realMixedPartial_cons_head_hasDerivAt {n : ℕ} {U : Set (Fin (n+1) → ℝ)}
    (hU : IsOpen U) (h0 : (0 : Fin (n+1) → ℝ) ∈ U)
    {P : (Fin (n+1) → ℝ) → ℝ} (hP : ContDiffOn ℝ ∞ P U)
    (w : List (Fin n)) :
    HasDerivAt (fun t : ℝ ↦ realMixedPartial w (fun y ↦ P (Fin.cons t y)) 0)
      (realMixedPartial (0 :: w.map Fin.succ) P 0) 0 := by
  have hhead : HasDerivAt (fun t : ℝ ↦ (Fin.cons t (0 : Fin n → ℝ) : Fin (n+1) → ℝ)) (Pi.single 0 1) 0 := by
    apply hasDerivAt_pi.mpr
    intro i
    refine Fin.cases ?_ (fun j ↦ ?_) i
    · convert hasDerivAt_id (0 : ℝ) using 1 <;> rfl
    · simpa using hasDerivAt_const (0 : ℝ) (0 : ℝ)
  have hhead0 : (Fin.cons (0 : ℝ) (0 : Fin n → ℝ) : Fin (n+1) → ℝ) = 0 := by ext i; refine Fin.cases ?_ (fun j ↦ ?_) i <;> rfl
  have heq : (fun t : ℝ ↦ realMixedPartial w (fun y ↦ P (Fin.cons t y)) 0) =ᶠ[𝓝 0]
      (fun t ↦ realMixedPartial (w.map Fin.succ) P (Fin.cons t (0 : Fin n → ℝ))) := by
    have hmem : ∀ᶠ t : ℝ in 𝓝 0, Fin.cons t (0 : Fin n → ℝ) ∈ U :=
      hhead.continuousAt.preimage_mem_nhds (by simpa only [hhead0] using hU.mem_nhds h0)
    filter_upwards [hmem] with t ht
    have h := realMixedPartial_affine hU hP (tailInclusion n) (Pi.single 0 t)
      Fin.succ (tailInclusion_single n) w 0 (by simpa only [← cons_eq_affine] using ht)
    simpa only [← cons_eq_affine] using h
  have hd : DifferentiableAt ℝ (realMixedPartial (w.map Fin.succ) P) 0 :=
    ((realMixedPartial_smooth hU hP (w.map Fin.succ)).differentiableOn (by simp) _ h0).differentiableAt
      (hU.mem_nhds h0)
  have hd' : HasFDerivAt (realMixedPartial (w.map Fin.succ) P)
      (fderiv ℝ (realMixedPartial (w.map Fin.succ) P) 0) (Fin.cons 0 (0 : Fin n → ℝ)) := by
    simpa only [hhead0] using hd.hasFDerivAt
  exact (hd'.comp_hasDerivAt 0 hhead).congr_of_eventuallyEq heq


theorem realMixedPartial_cons_head {n : ℕ} {U : Set (Fin (n+1) → ℝ)}
    (hU : IsOpen U) (h0 : (0 : Fin (n+1) → ℝ) ∈ U)
    {P : (Fin (n+1) → ℝ) → ℝ} (hP : ContDiffOn ℝ ∞ P U)
    (w : List (Fin n)) :
    realMixedPartial (0 :: w.map Fin.succ) P 0 =
      deriv (fun t : ℝ ↦ realMixedPartial w (fun y ↦ P (Fin.cons t y)) 0) 0 :=
  (realMixedPartial_cons_head_hasDerivAt hU h0 hP w).deriv.symm

end
end QIRadon.MixedRestriction

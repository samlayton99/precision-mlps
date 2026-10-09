import QIRadon.Activations.ReLU
import Mathlib.Analysis.Convex.Deriv

/-! Constructive shallow-ReLU interpolation on a uniform mesh, with a
 curvature-controlled algebraic approximation bound. -/

open Set
open scoped BigOperators

namespace QIRadon.ReLUSpline

noncomputable section
set_option backward.isDefEq.respectTransparency false

def node (a h : ℝ) (j : ℕ) : ℝ := a+(j:ℝ)*h

def spline (f : ℝ → ℝ) (a h : ℝ) (N : ℕ) (x : ℝ) : ℝ :=
  f a + ∑ j ∈ Finset.range N, ((f (node a h (j+1))-f (node a h j))/h) *
    (relu (x-node a h j)-relu (x-node a h (j+1)))

theorem node_succ (a h : ℝ) (j : ℕ) : node a h (j+1) = node a h j+h := by
  simp only [node, Nat.cast_add, Nat.cast_one]
  ring

theorem node_mono (a : ℝ) {h : ℝ} (hh : 0 ≤ h) {i j : ℕ} (hij : i ≤ j) :
    node a h i ≤ node a h j := by
  unfold node
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hij) hh)

theorem ramp_zero (a h x : ℝ) (hh : 0 ≤ h) (j : ℕ) (hx : x ≤ node a h j) :
    relu (x-node a h j)-relu (x-node a h (j+1)) = 0 := by
  have hx' : x ≤ node a h (j+1) := hx.trans (node_mono a hh (by omega))
  simp [relu, max_eq_left (sub_nonpos.mpr hx), max_eq_left (sub_nonpos.mpr hx')]

theorem ramp_full (a h x : ℝ) (hh : 0 ≤ h) (j : ℕ) (hx : node a h (j+1) ≤ x) :
    relu (x-node a h j)-relu (x-node a h (j+1)) = h := by
  have hx' : node a h j ≤ x := (node_mono a hh (by omega)).trans hx
  rw [relu, relu, max_eq_right (sub_nonneg.mpr hx'), max_eq_right (sub_nonneg.mpr hx), node_succ]
  ring

theorem ramp_local (a h x : ℝ) (j : ℕ) (hx : x ∈ Icc (node a h j) (node a h (j+1))) :
    relu (x-node a h j)-relu (x-node a h (j+1)) = x-node a h j := by
  simp [relu, max_eq_right (sub_nonneg.mpr hx.1), max_eq_left (sub_nonpos.mpr hx.2)]

theorem spline_succ (f : ℝ → ℝ) (a h : ℝ) (N : ℕ) (x : ℝ) :
    spline f a h (N+1) x = spline f a h N x +
      ((f (node a h (N+1))-f (node a h N))/h) *
        (relu (x-node a h N)-relu (x-node a h (N+1))) := by
  simp only [spline, Finset.sum_range_succ]
  ring

theorem spline_saturated (f : ℝ → ℝ) (a : ℝ) {h : ℝ} (hh : 0 < h)
    (N : ℕ) (x : ℝ) (hx : node a h N ≤ x) : spline f a h N x = f (node a h N) := by
  induction N with
  | zero => simp [spline, node]
  | succ N ih =>
    rw [spline_succ, ih ((node_mono a hh.le (by omega)).trans hx), ramp_full a h x hh.le N hx]
    field_simp
    ring

/-- The explicit sum of 2N ReLU terms is exactly linear interpolation on each
 mesh cell. The piecewise-linear behavior is derived from the network formula. -/
theorem spline_on_cell (f : ℝ → ℝ) (a : ℝ) {h : ℝ} (hh : 0 < h)
    (N k : ℕ) (hk : k < N) (x : ℝ) (hx : x ∈ Icc (node a h k) (node a h (k+1))) :
    spline f a h N x = f (node a h k) +
      ((f (node a h (k+1))-f (node a h k))/h)*(x-node a h k) := by
  induction N generalizing k with
  | zero => omega
  | succ N ih =>
    rw [spline_succ]
    by_cases hkN : k = N
    · subst k
      rw [spline_saturated f a hh N x hx.1, ramp_local a h x N hx]
    · have hk' : k < N := by omega
      have hxN : x ≤ node a h N := hx.2.trans (node_mono a hh.le (by omega))
      rw [ih k hk' hx, ramp_zero a h x hh.le N hxN, mul_zero, add_zero]


/-- The sharp one-cell interpolation estimate, derived from upper and lower
 curvature barriers. First and second derivatives are actual derivatives. -/
theorem interpolation_error (f f₁ f₂ : ℝ → ℝ) (a b M : ℝ) (hM : 0 ≤ M)
    (hf : ∀ y ∈ Icc a b, HasDerivAt f (f₁ y) y)
    (hf₁ : ∀ y ∈ Icc a b, HasDerivAt f₁ (f₂ y) y)
    (hf₂ : ∀ y ∈ Icc a b, |f₂ y| ≤ M)
    (l u x : ℝ) (hl : l ∈ Icc a b) (hu : u ∈ Icc a b) (hlu : l < u)
    (hx : x ∈ Icc l u) :
    |f x - (f l + ((f u-f l)/(u-l))*(x-l))| ≤ M*(u-l)^2/8 := by
  have hq (y : ℝ) : HasDerivAt (fun z : ℝ ↦ (M/2)*z^2) (M*y) y := by
    convert (hasDerivAt_pow 2 y).const_mul (M/2) using 1 <;> first | rfl | (simp; ring)
  have hlin (y : ℝ) : HasDerivAt (fun z : ℝ ↦ M*z) M y := by
    convert (hasDerivAt_id y).const_mul M using 1 <;> first | rfl | simp
  have hplus : ConvexOn ℝ (Icc a b) (fun y ↦ f y+(M/2)*y^2) := by
    apply convexOn_of_hasDerivWithinAt2_nonneg (f' := fun y ↦ f₁ y+M*y)
      (f'' := fun y ↦ f₂ y+M) (convex_Icc a b)
    · intro y hy
      exact ((hf y hy).add (hq y)).continuousAt.continuousWithinAt
    · intro y hy
      exact ((hf y (interior_subset hy)).add (hq y)).hasDerivWithinAt
    · intro y hy
      exact ((hf₁ y (interior_subset hy)).add (hlin y)).hasDerivWithinAt
    · intro y hy
      have h := (abs_le.mp (hf₂ y (interior_subset hy))).1
      linarith
  have hminus : ConcaveOn ℝ (Icc a b) (fun y ↦ f y-(M/2)*y^2) := by
    apply concaveOn_of_hasDerivWithinAt2_nonpos (f' := fun y ↦ f₁ y-M*y)
      (f'' := fun y ↦ f₂ y-M) (convex_Icc a b)
    · intro y hy
      exact ((hf y hy).sub (hq y)).continuousAt.continuousWithinAt
    · intro y hy
      exact ((hf y (interior_subset hy)).sub (hq y)).hasDerivWithinAt
    · intro y hy
      exact ((hf₁ y (interior_subset hy)).sub (hlin y)).hasDerivWithinAt
    · intro y hy
      have h := (abs_le.mp (hf₂ y (interior_subset hy))).2
      linarith
  let α := (u-x)/(u-l)
  let β := (x-l)/(u-l)
  have hden : 0 < u-l := sub_pos.mpr hlu
  have hα : 0 ≤ α := div_nonneg (sub_nonneg.mpr hx.2) hden.le
  have hβ : 0 ≤ β := div_nonneg (sub_nonneg.mpr hx.1) hden.le
  have hsum : α+β=1 := by dsimp [α,β]; field_simp; ring
  have hpoint : α*l+β*u=x := by dsimp [α,β]; field_simp; ring
  have hchord : α*f l+β*f u = f l+((f u-f l)/(u-l))*(x-l) := by
    dsimp [α,β]; field_simp; ring
  have hvariance : α*l^2+β*u^2-x^2 = (x-l)*(u-x) := by
    dsimp [α,β]; field_simp; ring
  have hup := hplus.2 hl hu hα hβ hsum
  have hlo := hminus.2 hl hu hα hβ hsum
  simp only [smul_eq_mul, hpoint] at hup hlo
  have hproduct : (x-l)*(u-x) ≤ (u-l)^2/4 := by nlinarith [sq_nonneg (2*x-l-u)]
  have hscale := mul_le_mul_of_nonneg_left hproduct hM
  apply abs_le.mpr
  constructor <;> nlinarith [hvariance, hchord]


theorem exists_cell (a : ℝ) {h : ℝ} (_hh : 0 < h) (N : ℕ) (hN : 0 < N)
    (x : ℝ) (hx : x ∈ Icc a (node a h N)) :
    ∃ k < N, x ∈ Icc (node a h k) (node a h (k+1)) := by
  induction N with
  | zero => omega
  | succ N ih =>
    by_cases hN0 : N = 0
    · subst N
      exact ⟨0, by omega, by simpa [node] using hx⟩
    · by_cases hxN : x ≤ node a h N
      · obtain ⟨k, hk, hcell⟩ := ih (Nat.pos_of_ne_zero hN0) ⟨hx.1, hxN⟩
        exact ⟨k, by omega, hcell⟩
      · exact ⟨N, by omega, ⟨(lt_of_not_ge hxN).le, hx.2⟩⟩

/-- Uniform error of the explicit finite ReLU sum. The constant is the sharp
 one-cell curvature constant; the representation uses at most 2N neurons. -/
theorem spline_error (f f₁ f₂ : ℝ → ℝ) (a : ℝ) {h : ℝ} (hh : 0 < h)
    (N : ℕ) (hN : 0 < N) (M : ℝ) (hM : 0 ≤ M)
    (hf : ∀ y ∈ Icc a (node a h N), HasDerivAt f (f₁ y) y)
    (hf₁ : ∀ y ∈ Icc a (node a h N), HasDerivAt f₁ (f₂ y) y)
    (hf₂ : ∀ y ∈ Icc a (node a h N), |f₂ y| ≤ M)
    (x : ℝ) (hx : x ∈ Icc a (node a h N)) :
    |f x-spline f a h N x| ≤ M*h^2/8 := by
  obtain ⟨k, hk, hcell⟩ := exists_cell a hh N hN x hx
  have hl : node a h k ∈ Icc a (node a h N) := by
    constructor
    · simpa only [node, Nat.cast_zero, zero_mul, add_zero] using node_mono a hh.le (Nat.zero_le k)
    · exact node_mono a hh.le hk.le
  have hu : node a h (k+1) ∈ Icc a (node a h N) := by
    constructor
    · exact hl.1.trans (node_mono a hh.le (by omega))
    · exact node_mono a hh.le (by omega)
  have hgap : node a h (k+1)-node a h k = h := by rw [node_succ]; ring
  have hlu : node a h k < node a h (k+1) := by rw [node_succ]; linarith
  have he := interpolation_error f f₁ f₂ a (node a h N) M hM hf hf₁ hf₂
    (node a h k) (node a h (k+1)) x hl hu hlu hcell
  rw [hgap] at he
  rw [spline_on_cell f a hh N k hk x hcell]
  exact he


def readout (f : ℝ → ℝ) (a h : ℝ) (N : ℕ) (k : Fin (N*2)) : ℝ :=
  let j := (finProdFinEquiv.symm k : Fin N × Fin 2)
  let s := (f (node a h (j.1.val+1))-f (node a h j.1.val))/h
  if j.2=0 then s else -s

def inputBias (a h : ℝ) (N : ℕ) (k : Fin (N*2)) : ℝ :=
  let j := (finProdFinEquiv.symm k : Fin N × Fin 2)
  if j.2=0 then -node a h j.1.val else -node a h (j.1.val+1)

/-- Explicit ordinary shallow network parameters: every input slope is one,
 output bias is f(a), and two opposite readouts encode each mesh ramp. -/
theorem spline_eq_reluNetwork (f : ℝ → ℝ) (a h : ℝ) (N : ℕ) (x : ℝ) :
    spline f a h N x =
      reluNetwork (readout f a h N) (fun _ ↦ 1) (inputBias a h N) (f a) x := by
  unfold reluNetwork spline
  congr 1
  rw [← (finProdFinEquiv : Fin N × Fin 2 ≃ Fin (N*2)).sum_comp
    (fun k ↦ readout f a h N k * relu (1*x+inputBias a h N k)), Fintype.sum_prod_type]
  simp only [readout, inputBias, Equiv.symm_apply_apply, Fin.sum_univ_two,
    Fin.one_eq_zero_iff, one_mul]
  norm_num only [ite_true, show (2:ℕ) ≠ 1 by decide, ite_false]
  have hterm (j : Fin N) :
      ((f (node a h (j.val+1))-f (node a h j.val))/h)*relu (x+-node a h j.val) +
      (-((f (node a h (j.val+1))-f (node a h j.val))/h))*relu (x+-node a h (j.val+1)) =
      ((f (node a h (j.val+1))-f (node a h j.val))/h)*
        (relu (x-node a h j.val)-relu (x-node a h (j.val+1))) := by
        simp only [sub_eq_add_neg]
        ring
  simp_rw [hterm]
  exact (Fin.sum_univ_eq_sum_range _ _).symm

/-- A width-2N finite ReLU network, given by explicit formulas, approximates
 any twice-differentiable target with bounded curvature at rate N^-2. -/
theorem network_uniform_error (f f₁ f₂ : ℝ → ℝ) (a b M : ℝ) (hab : a < b)
    (N : ℕ) (hN : 0 < N) (hM : 0 ≤ M)
    (hf : ∀ y ∈ Icc a b, HasDerivAt f (f₁ y) y)
    (hf₁ : ∀ y ∈ Icc a b, HasDerivAt f₁ (f₂ y) y)
    (hf₂ : ∀ y ∈ Icc a b, |f₂ y| ≤ M)
    (x : ℝ) (hx : x ∈ Icc a b) :
    |f x-reluNetwork (readout f a ((b-a)/N) N) (fun _ ↦ 1)
      (inputBias a ((b-a)/N) N) (f a) x| ≤ M*(b-a)^2/(8*(N:ℝ)^2) := by
  have hNR : 0 < (N:ℝ) := Nat.cast_pos.mpr hN
  have hh : 0 < (b-a)/(N:ℝ) := div_pos (sub_pos.mpr hab) hNR
  have hlast : node a ((b-a)/N) N = b := by
    unfold node
    field_simp
    ring
  have h := spline_error f f₁ f₂ a hh N hN M hM
    (by simpa only [hlast] using hf) (by simpa only [hlast] using hf₁)
    (by simpa only [hlast] using hf₂) x (by simpa only [hlast] using hx)
  rw [spline_eq_reluNetwork] at h
  apply h.trans_eq
  rw [div_pow]
  ring


/-- The usual C² specialization with the actual first and second derivatives. -/
theorem network_uniform_error_of_contDiff (f : ℝ → ℝ) (hf : ContDiff ℝ 2 f)
    (a b M : ℝ) (hab : a < b) (N : ℕ) (hN : 0 < N) (hM : 0 ≤ M)
    (hcurvature : ∀ y ∈ Icc a b, |deriv (deriv f) y| ≤ M)
    (x : ℝ) (hx : x ∈ Icc a b) :
    |f x-reluNetwork (readout f a ((b-a)/N) N) (fun _ ↦ 1)
      (inputBias a ((b-a)/N) N) (f a) x| ≤ M*(b-a)^2/(8*(N:ℝ)^2) := by
  apply network_uniform_error f (deriv f) (deriv (deriv f)) a b M hab N hN hM
    (fun y _ ↦ (hf.differentiable (by norm_num) y).hasDerivAt) ?_ hcurvature x hx
  intro y _
  have hd := hf.differentiable_iteratedDeriv 1 (by norm_num)
  simpa only [iteratedDeriv_one] using (hd y).hasDerivAt

end
end QIRadon.ReLUSpline

import QIRadon.Scalar.Preparation.ChebyshevApproximation
import QIRadon.Scalar.Preparation.GaussianWindow
import QIRadon.Scalar.Preparation.DerivativeSynthesis
import QIRadon.Scalar.Window
import Mathlib.Algebra.Polynomial.Eval.SMul
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.ChebyshevGauss
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Chebyshev.RootsExtrema
import Mathlib.LinearAlgebra.Lagrange
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!
# Concrete sample-based Chebyshev preparation

This file constructs the actual finite n-sample Chebyshev interpolation
operator. The analytic ellipse-to-geometric-approximation step must be proved
separately; the near-best inequality alone is not that step.
-/

namespace QIRadon.ProfilePreparation

open Polynomial Finset
open Polynomial.Chebyshev

/-- The n Chebyshev-root sampling nodes. -/
noncomputable def sampleNode (n j : ℕ) : ℝ :=
  Real.cos (((2 * (j : ℝ) + 1) / (2 * n)) * Real.pi)

theorem sampleNode_abs_le_one (n j : ℕ) : |sampleNode n j| ≤ 1 :=
  Real.abs_cos_le_one _

/-- Discrete inner product of Chebyshev basis elements on the sampling grid. -/
noncomputable def discreteInner (n k l : ℕ) : ℝ :=
  (1 / n : ℝ) * ∑ j ∈ range n,
    (T ℝ k).eval (sampleNode n j) * (T ℝ l).eval (sampleNode n j)

/-- Discrete orthogonality comes from the proved Gaussian quadrature rule,
not from assuming an interpolation matrix is well conditioned. -/
theorem discreteInner_eq {n k l : ℕ} (hn : n ≠ 0) (hk : k < n) (hl : l < n) :
    discreteInner n k l = if k = l then (if k = 0 then 1 else 1 / 2) else 0 := by
  have hdeg : ((T ℝ k) * (T ℝ l)).degree < 2 * n := by
    apply lt_of_le_of_lt (Polynomial.degree_mul_le _ _)
    simp only [degree_T, Int.natAbs_natCast]
    exact_mod_cast (show k + l < 2 * n by omega)
  have hg := integral_eq_sumZeroes hn hdeg
  simp only [sumZeroes, Polynomial.eval_mul] at hg
  change (∫ x, (T ℝ k).eval x * (T ℝ l).eval x ∂measureT) =
    (Real.pi / n) * ∑ j ∈ range n,
      (T ℝ k).eval (sampleNode n j) * (T ℝ l).eval (sampleNode n j) at hg
  have hid : discreteInner n k l =
      (∫ x, (T ℝ k).eval x * (T ℝ l).eval x ∂measureT) / Real.pi := by
    rw [hg]
    unfold discreteInner
    field_simp
  rw [hid]
  by_cases hkl : k = l
  · subst l
    rw [if_pos rfl]
    by_cases hk0 : k = 0
    · subst k
      rw [if_pos rfl]
      simp only [Nat.cast_zero]
      rw [integral_eval_T_real_mul_self_measureT_zero]
      exact div_self Real.pi_ne_zero
    · rw [if_neg hk0, integral_T_real_mul_self_measureT_of_ne_zero hk0]
      field_simp
  · rw [if_neg hkl, integral_eval_T_real_mul_eval_T_real_measureT_of_ne hkl, zero_div]

/-- Coefficient calculated directly from the n real function samples. -/
noncomputable def sampleCoefficient (n k : ℕ) (f : ℝ → ℝ) : ℝ :=
  ((if k = 0 then 1 else 2) / n : ℝ) *
    ∑ j ∈ range n, f (sampleNode n j) * (T ℝ k).eval (sampleNode n j)

/-- The concrete finite Chebyshev polynomial; no polynomial approximation
oracle or least-squares solve occurs in this definition. -/
noncomputable def samplePolynomial (n : ℕ) (f : ℝ → ℝ) : ℝ[X] :=
  ∑ k ∈ range n, sampleCoefficient n k f • T ℝ k

theorem sampleCoefficient_add (n k : ℕ) (f g : ℝ → ℝ) :
    sampleCoefficient n k (f + g) = sampleCoefficient n k f + sampleCoefficient n k g := by
  simp only [sampleCoefficient, Pi.add_apply, add_mul, Finset.sum_add_distrib, mul_add]

theorem sampleCoefficient_smul (n k : ℕ) (c : ℝ) (f : ℝ → ℝ) :
    sampleCoefficient n k (c • f) = c * sampleCoefficient n k f := by
  simp only [sampleCoefficient, Pi.smul_apply, smul_eq_mul, mul_assoc,
    ← Finset.mul_sum]
  ring

/-- The finite sampled operator is linear. -/
noncomputable def sampleOperator (n : ℕ) : (ℝ → ℝ) →ₗ[ℝ] ℝ[X] where
  toFun := samplePolynomial n
  map_add' f g := by
    simp only [samplePolynomial, sampleCoefficient_add, add_smul, Finset.sum_add_distrib]
  map_smul' c f := by
    simp only [samplePolynomial, sampleCoefficient_smul, RingHom.id_apply,
      ← smul_smul, Finset.smul_sum]

@[simp]
theorem sampleOperator_apply (n : ℕ) (f : ℝ → ℝ) : sampleOperator n f = samplePolynomial n f := rfl

theorem sampleCoefficient_T {n k l : ℕ} (hn : n ≠ 0) (hk : k < n) (hl : l < n) :
    sampleCoefficient n k (T ℝ l).eval = if k = l then 1 else 0 := by
  have hid : sampleCoefficient n k (T ℝ l).eval =
      (if k = 0 then 1 else 2) * discreteInner n l k := by
    unfold sampleCoefficient discreteInner
    ring
  rw [hid, discreteInner_eq hn hl hk]
  by_cases hkl : k = l
  · subst l
    by_cases hk0 : k = 0 <;> simp [hk0]
  · have hlk : l ≠ k := Ne.symm hkl
    simp [hkl, hlk]

/-- Exact reproduction of every basis polynomial below the sample count. -/
theorem samplePolynomial_T {n l : ℕ} (hn : n ≠ 0) (hl : l < n) :
    samplePolynomial n (T ℝ l).eval = T ℝ l := by
  unfold samplePolynomial
  have heq : ∀ k ∈ range n,
      sampleCoefficient n k (T ℝ l).eval • T ℝ k =
        (if k = l then 1 else 0 : ℝ) • T ℝ k := by
    intro k hk
    rw [sampleCoefficient_T hn (mem_range.mp hk) hl]
  rw [Finset.sum_congr rfl heq]
  simp [hl]

/-- Exact reproduction of every degree < n polynomial. -/
theorem samplePolynomial_reproduces {n : ℕ} (hn : n ≠ 0) {P : ℝ[X]}
    (hP : P.degree < n) : samplePolynomial n P.eval = P := by
  have hmem : P ∈ Polynomial.degreeLT ℝ n := by rwa [Polynomial.mem_degreeLT]
  rw [← Polynomial.Sequence.span_degreeLT (chebyshevTsequence ℝ) (by simp),
    show Set.Iio n = Finset.range n by ext; simp,
    Submodule.mem_span_image_finset_iff_exists_fun'] at hmem
  obtain ⟨c, rfl⟩ := hmem
  simp only [chebyshevTsequence]
  have heval : (∑ i ∈ range n, c i • T ℝ i).eval =
      ∑ i ∈ range n, c i • (T ℝ i).eval := by
    ext x
    simp only [Polynomial.eval_finsetSum, Polynomial.eval_smul, Finset.sum_apply,
      Pi.smul_apply]
  rw [heval]
  change sampleOperator n (∑ i ∈ range n, c i • (T ℝ i).eval) = _
  rw [map_sum]
  apply Finset.sum_congr rfl
  intro i hi
  rw [map_smul]
  simp only [sampleOperator_apply, samplePolynomial_T hn (mem_range.mp hi)]

/-- A concrete coefficient norm estimate from actual supplied samples. -/
theorem sampleCoefficient_bound {n k : ℕ} {f : ℝ → ℝ} {M : ℝ}
    (hn : 0 < n) (hM : 0 ≤ M)
    (hf : ∀ j < n, |f (sampleNode n j)| ≤ M) :
    |sampleCoefficient n k f| ≤ 2 * M := by
  have hnR : 0 < (n : ℝ) := Nat.cast_pos.mpr hn
  have hsum : |∑ j ∈ range n, f (sampleNode n j) * (T ℝ k).eval (sampleNode n j)| ≤ n * M := by
    calc
      _ ≤ ∑ j ∈ range n, |f (sampleNode n j) * (T ℝ k).eval (sampleNode n j)| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ range n, M := by
        apply Finset.sum_le_sum
        intro j hj
        rw [abs_mul]
        have ht := abs_eval_T_real_le_one (k : ℤ) (sampleNode_abs_le_one n j)
        exact (mul_le_mul (hf j (mem_range.mp hj)) ht (abs_nonneg _) hM).trans_eq (mul_one M)
      _ = n * M := by simp
  unfold sampleCoefficient
  rw [abs_mul]
  have hfactor : |((if k = 0 then 1 else 2) / n : ℝ)| ≤ 2 / n := by
    by_cases hk : k = 0
    · simp only [if_pos hk]
      rw [abs_of_pos (div_pos zero_lt_one hnR)]
      exact (div_le_div_iff_of_pos_right hnR).mpr (by norm_num)
    · simp only [if_neg hk]
      rw [abs_of_pos (div_pos (by norm_num) hnR)]
  calc
    _ ≤ (2 / n) * (n * M) := mul_le_mul hfactor hsum (abs_nonneg _) (div_pos (by norm_num) hnR).le
    _ = 2 * M := by field_simp

/-- A polynomial Lebesgue bound for the actual n-sample operator. It is not
optimal, but grows only linearly and therefore preserves geometric rates. -/
theorem samplePolynomial_bound {n : ℕ} {f : ℝ → ℝ} {M x : ℝ}
    (hn : 0 < n) (hM : 0 ≤ M)
    (hf : ∀ j < n, |f (sampleNode n j)| ≤ M) (hx : |x| ≤ 1) :
    |(samplePolynomial n f).eval x| ≤ (2 * n : ℝ) * M := by
  unfold samplePolynomial
  rw [Polynomial.eval_finsetSum]
  calc
    _ ≤ ∑ k ∈ range n, |(sampleCoefficient n k f • T ℝ k).eval x| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k ∈ range n, 2 * M := by
      apply Finset.sum_le_sum
      intro k hk
      rw [Polynomial.eval_smul, smul_eq_mul, abs_mul]
      exact (mul_le_mul (sampleCoefficient_bound hn hM hf)
        (abs_eval_T_real_le_one (k : ℤ) hx) (abs_nonneg _) (by positivity)).trans_eq (mul_one _)
    _ = (2 * n : ℝ) * M := by simp; ring


/-- Distinctness of the actual n sample locations. -/
theorem sampleNode_injOn {n : ℕ} (hn : 0 < n) :
    Set.InjOn (sampleNode n) (range n) := by
  have hnR : 0 < (n : ℝ) := Nat.cast_pos.mpr hn
  have hangle : ∀ j ∈ range n,
      ((2 * (j : ℝ) + 1) / (2 * n)) * Real.pi ∈ Set.Icc (0 : ℝ) Real.pi := by
    intro j hj
    have hjR : (j : ℝ) < n := by exact_mod_cast mem_range.mp hj
    constructor
    · positivity
    · have hfrac : (2 * (j : ℝ) + 1) / (2 * n) ≤ 1 := by
        apply (div_le_one (by positivity)).mpr
        have hjN : j + 1 ≤ n := mem_range.mp hj
        have hjR' : (j : ℝ) + 1 ≤ n := by exact_mod_cast hjN
        linarith
      have hm := mul_le_mul_of_nonneg_right hfrac Real.pi_pos.le
      simpa using hm
  intro i hi j hj hij
  have he := Real.injOn_cos (hangle i hi) (hangle j hj) hij
  have he' : (i : ℝ) = j := by
    field_simp at he
    nlinarith [Real.pi_pos]
  exact_mod_cast he'

/-- The operator depends only on its specified n samples. -/
theorem samplePolynomial_congr {n : ℕ} {f g : ℝ → ℝ}
    (hfg : ∀ j < n, f (sampleNode n j) = g (sampleNode n j)) :
    samplePolynomial n f = samplePolynomial n g := by
  unfold samplePolynomial
  apply Finset.sum_congr rfl
  intro k hk
  congr 1
  unfold sampleCoefficient
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  rw [hfg j (mem_range.mp hj)]

/-- The explicit sample formula is exactly the Lagrange interpolant of those
samples. In particular the preparation really uses n values, with no solve. -/
theorem samplePolynomial_eq_interpolate {n : ℕ} (hn : 0 < n) (f : ℝ → ℝ) :
    samplePolynomial n f =
      Lagrange.interpolate (range n) (sampleNode n) (fun j => f (sampleNode n j)) := by
  let P := Lagrange.interpolate (range n) (sampleNode n) (fun j => f (sampleNode n j))
  have hi := sampleNode_injOn hn
  have hdegree : P.degree < n := by
    simpa [P] using Lagrange.degree_interpolate_lt (fun j => f (sampleNode n j)) hi
  have heq : ∀ j < n, f (sampleNode n j) = P.eval (sampleNode n j) := by
    intro j hj
    exact (Lagrange.eval_interpolate_at_node (fun j => f (sampleNode n j)) hi (mem_range.mpr hj)).symm
  rw [samplePolynomial_congr (g := P.eval) heq, samplePolynomial_reproduces (Nat.ne_of_gt hn) hdegree]

theorem samplePolynomial_interpolates {n j : ℕ} (hn : 0 < n) (hj : j < n) (f : ℝ → ℝ) :
    (samplePolynomial n f).eval (sampleNode n j) = f (sampleNode n j) := by
  rw [samplePolynomial_eq_interpolate hn]
  exact Lagrange.eval_interpolate_at_node (fun j => f (sampleNode n j))
    (sampleNode_injOn hn) (mem_range.mpr hj)

theorem samplePolynomial_degree_lt {n : ℕ} (hn : 0 < n) (f : ℝ → ℝ) :
    (samplePolynomial n f).degree < n := by
  rw [samplePolynomial_eq_interpolate hn]
  simpa using Lagrange.degree_interpolate_lt (fun j => f (sampleNode n j)) (sampleNode_injOn hn)

/-- Near-best bound for the concrete interpolation operator. This is an
operator lemma, not an assumption replacing the later analytic approximation
proof. The comparator P is used only in this intermediate inequality. -/
theorem samplePolynomial_error_of_comparator {n : ℕ} {f : ℝ → ℝ} {P : ℝ[X]} {E x : ℝ}
    (hn : 0 < n) (hdegree : P.degree < n) (hE : 0 ≤ E)
    (happrox : ∀ y : ℝ, |y| ≤ 1 → |f y - P.eval y| ≤ E)
    (hx : |x| ≤ 1) :
    |f x - (samplePolynomial n f).eval x| ≤ (1 + 2 * n : ℝ) * E := by
  have hsamples : ∀ j < n, |(f - P.eval) (sampleNode n j)| ≤ E := by
    intro j hj
    exact happrox _ (sampleNode_abs_le_one n j)
  have hbound := samplePolynomial_bound hn hE hsamples hx
  have hlin : samplePolynomial n (f - P.eval) = samplePolynomial n f - P := by
    change sampleOperator n (f - P.eval) = sampleOperator n f - P
    rw [map_sub]
    change samplePolynomial n f - samplePolynomial n P.eval = _
    rw [samplePolynomial_reproduces (Nat.ne_of_gt hn) hdegree]
    rfl
  rw [hlin, Polynomial.eval_sub] at hbound
  have heq : f x - (samplePolynomial n f).eval x =
      (f x - P.eval x) - ((samplePolynomial n f).eval x - P.eval x) := by ring
  rw [heq]
  have ht : |(f x - P.eval x) - ((samplePolynomial n f).eval x - P.eval x)| ≤
      |f x - P.eval x| + |(samplePolynomial n f).eval x - P.eval x| := by
    simpa only [sub_zero, zero_sub, abs_neg] using
      abs_sub_le (f x - P.eval x) 0 ((samplePolynomial n f).eval x - P.eval x)
  calc
    _ ≤ _ := ht
    _ ≤ E + (2 * n : ℝ) * E := add_le_add (happrox x hx) hbound
    _ = (1 + 2 * n : ℝ) * E := by ring


/-- Main sampled interpolation result: local ellipse analyticity implies a
geometric approximation bound for the explicitly defined N+1-sample formula.
No supplied polynomial approximant appears among the assumptions. -/
theorem samplePolynomial_analytic_error {g : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 < δ) (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ))
    (hbound : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M)
    (N : ℕ) {x : ℝ} (hx : |x| ≤ 1) :
    |(g (x : ℂ)).re -
      (samplePolynomial (N + 1) (fun t : ℝ => (g (t : ℂ)).re)).eval x| ≤
      (1 + 2 * (N + 1) : ℝ) *
        ((2 * M) * Real.exp (-δ * N) / (1 - Real.exp (-δ))) := by
  have hM : 0 ≤ M := by
    have hz : (0 : ℂ) ∈ closedStrip δ := by simpa [closedStrip] using hδ.le
    exact (norm_nonneg _).trans (hbound (Complex.cos 0) ⟨0, hz, rfl⟩)
  have hden : 0 < 1 - Real.exp (-δ) :=
    sub_pos.mpr (Real.exp_lt_one_iff.mpr (neg_neg_of_pos hδ))
  let P := cosinePolynomial (stripCoefficient (fun z => g (Complex.cos z))) N
  have hdeg : P.degree < N + 1 := by
    apply lt_of_le_of_lt (cosinePolynomial_degree_le _ N)
    exact_mod_cast Nat.lt_succ_self N
  simpa only [Nat.cast_succ] using samplePolynomial_error_of_comparator (Nat.succ_pos N) hdeg
    (by positivity) (fun y hy => analytic_cosinePolynomial_error hδ hg hbound N hy) hx

/-- Absorbing a linear prefactor into half the geometric rate, explicitly. -/
theorem linear_prefactor_geometric_bound {A B δ x : ℝ}
    (hA : 0 ≤ A) (hB : 0 ≤ B) (hδ : 0 < δ) (hx : 0 ≤ x) :
    (A + B * x) * Real.exp (-δ * x) ≤
      (A + 2 * B / δ) * Real.exp (-(δ / 2) * x) := by
  have hexp : 1 ≤ Real.exp ((δ / 2) * x) :=
    Real.one_le_exp_iff.mpr (mul_nonneg (half_pos hδ).le hx)
  have hxexp : (δ / 2) * x ≤ Real.exp ((δ / 2) * x) := by
    have hh := Real.add_one_le_exp ((δ / 2) * x)
    linarith
  have h1 := mul_le_mul_of_nonneg_left hexp hA
  have h2 := mul_le_mul_of_nonneg_left hxexp (show 0 ≤ 2 * B / δ by positivity)
  have hbase : A + B * x ≤ (A + 2 * B / δ) * Real.exp ((δ / 2) * x) := by
    have hid : (2 * B / δ) * ((δ / 2) * x) = B * x := by field_simp
    rw [hid] at h2
    nlinarith
  have hm := mul_le_mul_of_nonneg_right hbase (Real.exp_pos (-δ * x)).le
  have hid : Real.exp ((δ / 2) * x) * Real.exp (-δ * x) =
      Real.exp (-(δ / 2) * x) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc
    _ ≤ ((A + 2 * B / δ) * Real.exp ((δ / 2) * x)) * Real.exp (-δ * x) := hm
    _ = _ := by rw [mul_assoc, hid]

/-- Clean geometric rate of the actual sampled interpolant, with an explicit
constant depending only on the given ellipse width and target bound. -/
theorem samplePolynomial_geometric_error {g : ℂ → ℂ} {δ M : ℝ}
    (hδ : 0 < δ) (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ))
    (hbound : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M)
    (N : ℕ) {x : ℝ} (hx : |x| ≤ 1) :
    |(g (x : ℂ)).re -
      (samplePolynomial (N + 1) (fun t : ℝ => (g (t : ℂ)).re)).eval x| ≤
      (M * (6 + 8 / δ) / (1 - Real.exp (-δ))) * Real.exp (-(δ / 2) * N) := by
  apply (samplePolynomial_analytic_error hδ hg hbound N hx).trans
  have hM : 0 ≤ M := by
    have hz : (0 : ℂ) ∈ closedStrip δ := by simpa [closedStrip] using hδ.le
    exact (norm_nonneg _).trans (hbound (Complex.cos 0) ⟨0, hz, rfl⟩)
  have hden : 0 < 1 - Real.exp (-δ) :=
    sub_pos.mpr (Real.exp_lt_one_iff.mpr (neg_neg_of_pos hδ))
  have hh := linear_prefactor_geometric_bound (A := 6) (B := 4)
    (by norm_num) (by norm_num) hδ (Nat.cast_nonneg N)
  have hm := mul_le_mul_of_nonneg_left hh (div_nonneg hM hden.le)
  convert! hm using 1 <;> ring


/-- The actual globally decaying preparation, calculated from N+1 real samples.
The polynomial/window appear only in coefficient preparation, not in the final
activation-network evaluator. -/
noncomputable def preparedProfile (N : ℕ) (f : ℝ → ℝ) (T₀ : ℝ) (z : ℂ) : ℂ :=
  ((samplePolynomial (N + 1) f).map Complex.ofRealHom).eval z *
    complexFlatWindow N (z / (T₀ : ℂ))

theorem complexFlatWindow_ofReal (N : ℕ) (x : ℝ) :
    complexFlatWindow N (x : ℂ) = (QIRadon.flatWindow N (x ^ 2) : ℂ) := by
  unfold complexFlatWindow QIRadon.flatWindow
  push_cast
  rfl

theorem preparedProfile_ofReal (N : ℕ) (f : ℝ → ℝ) (T₀ x : ℝ) :
    preparedProfile N f T₀ (x : ℂ) =
      (((samplePolynomial (N + 1) f).eval x * QIRadon.flatWindow N ((x / T₀) ^ 2) : ℝ) : ℂ) := by
  unfold preparedProfile
  have hp : ((samplePolynomial (N + 1) f).map Complex.ofRealHom).eval (x : ℂ) =
      Complex.ofReal ((samplePolynomial (N + 1) f).eval x) := by
    convert! Polynomial.eval_map_apply (p := samplePolynomial (N + 1) f) Complex.ofRealHom x using 1
  rw [hp, show (x : ℂ) / (T₀ : ℂ) = ((x / T₀ : ℝ) : ℂ) by push_cast; rfl,
    complexFlatWindow_ofReal]
  push_cast
  rfl

/-- Actual preparation error before the later coefficient construction. The
window margin is explicit and no global analytic target extension is assumed. -/
theorem preparedProfile_real_error {g : ℂ → ℂ} {δ M T₀ : ℝ}
    (hδ : 0 < δ) (hg : AnalyticOnNhd ℂ g (bernsteinEllipse δ))
    (hbound : ∀ z ∈ bernsteinEllipse δ, ‖g z‖ ≤ M)
    (hT : 1 < T₀) (N : ℕ) {x : ℝ} (hx : |x| ≤ 1) :
    |(g (x : ℂ)).re - (preparedProfile N (fun t : ℝ => (g (t : ℂ)).re) T₀ (x : ℂ)).re| ≤
      (M * (6 + 8 / δ) / (1 - Real.exp (-δ))) * Real.exp (-(δ / 2) * N) +
      (2 * (N + 1) * M) *
        Real.exp (-(N : ℝ) * ((1 / T₀) ^ 2 - 1 - Real.log ((1 / T₀) ^ 2))) := by
  have hT0 : 0 < T₀ := lt_trans zero_lt_one hT
  have hM : 0 ≤ M := by
    have hz : (0 : ℂ) ∈ closedStrip δ := by simpa [closedStrip] using hδ.le
    exact (norm_nonneg _).trans (hbound (Complex.cos 0) ⟨0, hz, rfl⟩)
  have hreal : ∀ y : ℝ, |y| ≤ 1 → |(g (y : ℂ)).re| ≤ M := by
    intro y hy
    have hc : Real.cos (Real.arccos y) = y := Real.cos_arccos (neg_le_of_abs_le hy) (le_of_abs_le hy)
    apply (Complex.abs_re_le_norm _).trans
    apply hbound
    refine ⟨(Real.arccos y : ℂ), ?_, ?_⟩
    · simpa [closedStrip] using hδ.le
    · rw [← Complex.ofReal_cos, hc]
  have hpoly := samplePolynomial_bound (n := N + 1)
    (f := fun t : ℝ => (g (t : ℂ)).re) (Nat.succ_pos N) hM
    (fun j hj => hreal _ (sampleNode_abs_le_one (N + 1) j)) hx
  have hu : 0 ≤ (x / T₀) ^ 2 := sq_nonneg _
  have huv : (x / T₀) ^ 2 ≤ (1 / T₀) ^ 2 := by
    rw [div_pow, div_pow, one_pow]
    apply (div_le_div_iff_of_pos_right (sq_pos_of_pos hT0)).mpr
    have hm := mul_self_le_mul_self (abs_nonneg x) hx
    nlinarith [sq_abs x, hm]
  have hv : 0 < (1 / T₀) ^ 2 := sq_pos_of_pos (div_pos zero_lt_one hT0)
  have hv1 : (1 / T₀) ^ 2 < 1 := by
    have hdiv : 0 < 1 / T₀ := div_pos zero_lt_one hT0
    have hdiv1 : 1 / T₀ < 1 := (div_lt_one hT0).mpr hT
    nlinarith
  rw [preparedProfile_ofReal, Complex.ofReal_re]
  simpa only [Nat.cast_succ] using QIRadon.flatWindow_preparation_error N
    (samplePolynomial_geometric_error hδ hg hbound N hx) hpoly hu huv hv hv1

/-- Complex growth of the explicitly sampled polynomial, with coefficient
bounds obtained from its actual data values. -/
theorem samplePolynomial_complex_bound {f : ℝ → ℝ} {M : ℝ}
    (hM : 0 ≤ M) (N : ℕ)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) (z : ℂ) :
    ‖((samplePolynomial (N + 1) f).map Complex.ofRealHom).eval z‖ ≤
      (2 * M * (N + 1)) * (1 + 2 * ‖z‖) ^ N := by
  unfold samplePolynomial
  rw [Polynomial.map_sum, Polynomial.eval_finsetSum]
  have hbase : 1 ≤ 1 + 2 * ‖z‖ := by linarith [norm_nonneg z]
  calc
    _ ≤ ∑ k ∈ range (N + 1),
        ‖((sampleCoefficient (N + 1) k f • T ℝ k).map Complex.ofRealHom).eval z‖ :=
      norm_sum_le _ _
    _ ≤ ∑ k ∈ range (N + 1), (2 * M) * (1 + 2 * ‖z‖) ^ N := by
      apply sum_le_sum
      intro k hk
      rw [Polynomial.map_smul, map_T, Polynomial.eval_smul, norm_smul]
      have hcoeff : ‖Complex.ofRealHom (sampleCoefficient (N + 1) k f)‖ ≤ 2 * M := by
        change ‖(sampleCoefficient (N + 1) k f : ℂ)‖ ≤ 2 * M
        rw [Complex.norm_real, Real.norm_eq_abs]
        exact sampleCoefficient_bound (Nat.succ_pos N) hM hf
      have hTk := complex_chebyshev_bound k z
      have hpow : (1 + 2 * ‖z‖) ^ k ≤ (1 + 2 * ‖z‖) ^ N :=
        pow_le_pow_right₀ hbase (mem_range_succ_iff.mp hk)
      exact mul_le_mul hcoeff (hTk.trans hpow) (norm_nonneg _) (by positivity)
    _ = _ := by simp; ring


/-- Explicit horizontal-line envelope constant. Its logarithm grows at most
linearly in N apart from the displayed polynomial factor N+1. -/
noncomputable def preparationEnvelope (N : ℕ) (M σ T₀ : ℝ) : ℝ :=
  (2 * M * (N + 1)) * (2 * (1 + 2 * σ + 4 * T₀)) ^ N *
    Real.exp (3 * N * σ ^ 2 / (2 * T₀ ^ 2))

theorem preparationEnvelope_nonneg {N : ℕ} {M σ T₀ : ℝ}
    (hM : 0 ≤ M) (hσ : 0 ≤ σ) (hT : 0 < T₀) :
    0 ≤ preparationEnvelope N M σ T₀ := by
  unfold preparationEnvelope
  positivity

/-- The actual sampled-and-windowed profile is entire. -/
theorem preparedProfile_differentiable (N : ℕ) (f : ℝ → ℝ) (T₀ : ℝ) :
    Differentiable ℂ (preparedProfile N f T₀) := by
  unfold preparedProfile complexFlatWindow
  fun_prop

/-- Explicit Gaussian decay of the actual prepared profile on a horizontal
strip, including all N-dependence of the constant. -/
theorem preparedProfile_horizontal_bound {f : ℝ → ℝ} {M σ T₀ y x : ℝ}
    (hM : 0 ≤ M) (hσ : 0 ≤ σ) (hT : 0 < T₀) (hy : |y| ≤ σ)
    (N : ℕ) (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) :
    ‖preparedProfile N f T₀ ((x : ℂ) + y * Complex.I)‖ ≤
      preparationEnvelope N M σ T₀ * Real.exp (-(N : ℝ) * x ^ 2 / (4 * T₀ ^ 2)) := by
  let z : ℂ := (x : ℂ) + y * Complex.I
  let A : ℝ := 2 * M * (N + 1)
  let B : ℝ := 1 + 2 * σ + 4 * T₀
  let r : ℝ := x ^ 2 / (4 * T₀ ^ 2)
  let d : ℝ := 3 * σ ^ 2 / (2 * T₀ ^ 2)
  have hA : 0 ≤ A := by dsimp [A]; positivity
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hpoly : ‖((samplePolynomial (N + 1) f).map Complex.ofRealHom).eval z‖ ≤
      A * B ^ N * Real.exp ((N : ℝ) * r) := by
    apply (samplePolynomial_complex_bound hM N hf z).trans
    have hp := pow_le_pow_left₀ (by positivity : 0 ≤ 1 + 2 * ‖z‖)
      (chebyshev_growth_gaussian hT hσ hy) N
    have hm := mul_le_mul_of_nonneg_left hp hA
    rw [mul_pow, ← Real.exp_nat_mul] at hm
    simpa only [A, B, r, z, mul_assoc] using hm
  have hy2 : y ^ 2 ≤ σ ^ 2 := by
    have hh := mul_self_le_mul_self (abs_nonneg y) hy
    nlinarith [sq_abs y]
  have hwindow : ‖complexFlatWindow N (z / (T₀ : ℂ))‖ ≤
      (2 : ℝ) ^ N * Real.exp ((N : ℝ) * d - 2 * N * r) := by
    apply (complexFlatWindow_bound N (z / (T₀ : ℂ))).trans
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    apply Real.exp_le_exp.mpr
    have hre : (z / (T₀ : ℂ)).re = x / T₀ := by
      simp [z, Complex.div_ofReal_re]
    have him : (z / (T₀ : ℂ)).im = y / T₀ := by
      simp [z, Complex.div_ofReal_im]
    rw [hre, him]
    dsimp [d, r]
    have hsq : 0 < T₀ ^ 2 := sq_pos_of_pos hT
    apply (mul_le_mul_iff_right₀ hsq).mp
    field_simp
    have hh := mul_le_mul_of_nonneg_left hy2 (Nat.cast_nonneg N)
    nlinarith
  unfold preparedProfile
  change ‖((samplePolynomial (N + 1) f).map Complex.ofRealHom).eval z * complexFlatWindow N (z / (T₀ : ℂ))‖ ≤ _
  rw [norm_mul]
  have hm := mul_le_mul hpoly hwindow (norm_nonneg _) (by positivity)
  apply hm.trans_eq
  have he : Real.exp ((N : ℝ) * r) * Real.exp ((N : ℝ) * d - 2 * N * r) =
      Real.exp ((N : ℝ) * d) * Real.exp (-(N : ℝ) * r) := by
    rw [← Real.exp_add, ← Real.exp_add]
    congr 1
    ring
  calc
    _ = A * (2 * B) ^ N *
        (Real.exp ((N : ℝ) * r) * Real.exp ((N : ℝ) * d - 2 * N * r)) := by rw [mul_pow]; ring
    _ = A * (2 * B) ^ N * (Real.exp ((N : ℝ) * d) * Real.exp (-(N : ℝ) * r)) := by rw [he]
    _ = _ := by
      unfold preparationEnvelope
      dsimp [A, B, d, r]
      have hd : (N : ℝ) * (3 * σ ^ 2 / (2 * T₀ ^ 2)) = 3 * N * σ ^ 2 / (2 * T₀ ^ 2) := by ring
      have hr : -(N : ℝ) * (x ^ 2 / (4 * T₀ ^ 2)) = -(N : ℝ) * x ^ 2 / (4 * T₀ ^ 2) := by ring
      rw [hd, hr]
      ring

/-- Ordinary absolute integrability on every horizontal line, proved for the
actual profile rather than assumed as an analytic-continuation oracle. -/
theorem preparedProfile_horizontal_integrable {f : ℝ → ℝ} {M σ T₀ y : ℝ}
    (hM : 0 ≤ M) (hσ : 0 ≤ σ) (hT : 0 < T₀) (hy : |y| ≤ σ)
    {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) :
    MeasureTheory.Integrable (fun x : ℝ => preparedProfile N f T₀ ((x : ℂ) + y * Complex.I)) := by
  have hr : 0 < (N : ℝ) / (4 * T₀ ^ 2) := by positivity
  have hi := (integrable_exp_neg_mul_sq hr).const_mul (preparationEnvelope N M σ T₀)
  apply hi.mono'
  · exact ((preparedProfile_differentiable N f T₀).continuous.comp
      (Complex.continuous_ofReal.add continuous_const)).aestronglyMeasurable
  · apply Filter.Eventually.of_forall
    intro x
    have hb := preparedProfile_horizontal_bound hM hσ hT hy N hf (x := x)
    simpa only [show -(N : ℝ) * x ^ 2 / (4 * T₀ ^ 2) = -((N : ℝ) / (4 * T₀ ^ 2)) * x ^ 2 by ring] using hb

/-- Quantified shifted-line L1 bound, a polynomial factor times exp(O(N)). -/
theorem preparedProfile_horizontal_L1 {f : ℝ → ℝ} {M σ T₀ y : ℝ}
    (hM : 0 ≤ M) (hσ : 0 ≤ σ) (hT : 0 < T₀) (hy : |y| ≤ σ)
    {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) :
    (∫ x : ℝ, ‖preparedProfile N f T₀ ((x : ℂ) + y * Complex.I)‖) ≤
      preparationEnvelope N M σ T₀ *
        Real.sqrt (Real.pi / ((N : ℝ) / (4 * T₀ ^ 2))) := by
  have hr : 0 < (N : ℝ) / (4 * T₀ ^ 2) := by positivity
  have hi := (integrable_exp_neg_mul_sq hr).const_mul (preparationEnvelope N M σ T₀)
  calc
    _ ≤ ∫ x : ℝ, preparationEnvelope N M σ T₀ * Real.exp (-((N : ℝ) / (4 * T₀ ^ 2)) * x ^ 2) := by
      apply MeasureTheory.integral_mono
        (preparedProfile_horizontal_integrable hM hσ hT hy hN hf).norm hi
      intro x
      have hb := preparedProfile_horizontal_bound hM hσ hT hy N hf (x := x)
      simpa only [show -(N : ℝ) * x ^ 2 / (4 * T₀ ^ 2) = -((N : ℝ) / (4 * T₀ ^ 2)) * x ^ 2 by ring] using hb
    _ = _ := by rw [MeasureTheory.integral_const_mul, integral_gaussian]


open MeasureTheory
open scoped FourierTransform

/-- The growth constant is exactly a polynomial times an exponential in N. -/
theorem preparationEnvelope_exp_form {M σ T₀ : ℝ} (N : ℕ)
    (hσ : 0 ≤ σ) (hT : 0 < T₀) :
    preparationEnvelope N M σ T₀ =
      (2 * M * (N + 1)) *
        Real.exp ((N : ℝ) * (Real.log (2 * (1 + 2 * σ + 4 * T₀)) +
          3 * σ ^ 2 / (2 * T₀ ^ 2))) := by
  have hb : 0 < 2 * (1 + 2 * σ + 4 * T₀) := by positivity
  have hp : (2 * (1 + 2 * σ + 4 * T₀)) ^ N =
      Real.exp ((N : ℝ) * Real.log (2 * (1 + 2 * σ + 4 * T₀))) := by
    rw [Real.exp_nat_mul, Real.exp_log hb]
  unfold preparationEnvelope
  rw [hp, mul_assoc, ← Real.exp_add]
  congr 2
  ring

/-- Actual weighted Fourier moments of the sampled-and-windowed profile.
The complete accuracy-dependent norm bound remains visible. -/
theorem preparedProfile_fourier_moment {f : ℝ → ℝ} {M δ σ T₀ : ℝ}
    (hM : 0 ≤ M) (hδ : 0 ≤ δ) (hσ : σ < δ) (hT : 0 < T₀)
    {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) (r : ℕ) :
    Integrable (fun ω : ℝ => Real.exp (σ * |ω|) * |ω| ^ r *
      ‖angularFourier (preparedProfile N f T₀) ω‖) ∧
    (∫ ω : ℝ, Real.exp (σ * |ω|) * |ω| ^ r * ‖angularFourier (preparedProfile N f T₀) ω‖) ≤
      (preparationEnvelope N M δ T₀ * Real.sqrt (Real.pi / ((N : ℝ) / (4 * T₀ ^ 2)))) *
        (2 * (r.factorial : ℝ) / (δ - σ) ^ (r + 1)) := by
  apply angularFourier_weighted_moment (preparedProfile_differentiable N f T₀)
    (preparationEnvelope_nonneg hM hδ hT) (by positivity) hδ hσ
  intro x y hy
  have hb := preparedProfile_horizontal_bound hM hδ hT hy N hf (x := x)
  convert! hb using 1
  congr 2
  ring

/-- The same proved exponential spectral decay, explicitly in mathlib's cycles
normalization used by the finite-kernel construction. -/
theorem preparedProfile_cycles_decay {f : ℝ → ℝ} {M δ T₀ : ℝ}
    (hM : 0 ≤ M) (hδ : 0 ≤ δ) (hT : 0 < T₀)
    {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) (ξ : ℝ) :
    ‖𝓕 (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)) ξ‖ ≤
      (preparationEnvelope N M δ T₀ * Real.sqrt (Real.pi / ((N : ℝ) / (4 * T₀ ^ 2)))) *
        Real.exp (-(2 * Real.pi * δ) * |ξ|) := by
  have hb := angularFourier_gaussian_strip_bound (preparedProfile_differentiable N f T₀)
    (preparationEnvelope_nonneg hM hδ hT) (show 0 < (N : ℝ) / (4 * T₀ ^ 2) by positivity)
    hδ (fun x y hy => by
      have hh := preparedProfile_horizontal_bound hM hδ hT hy N hf (x := x)
      convert! hh using 1
      congr 2
      ring) (2 * Real.pi * ξ)
  rw [angularFourier_eq_cycles] at hb
  have hξ : (2 * Real.pi * ξ) / (2 * Real.pi) = ξ := by field_simp
  rw [hξ, abs_mul, abs_of_pos (show 0 < 2 * Real.pi by positivity)] at hb
  convert! hb using 1
  congr 2
  ring

/-- All derivative moments required by inverse Fourier differentiation are
proved for the actual prepared profile, using a fixed horizontal strip. -/
theorem preparedProfile_monomial_moments {f : ℝ → ℝ} {M T₀ : ℝ}
    (hM : 0 ≤ M) (hT : 0 < T₀) {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) (r : ℕ) :
    Integrable (fun ξ : ℝ => ξ ^ r •
      𝓕 (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)) ξ) := by
  have hi : Integrable (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)) := by
    simpa only [Complex.ofReal_zero, zero_mul, add_zero, Function.comp_def] using
      preparedProfile_horizontal_integrable hM (show (0 : ℝ) ≤ 1 by norm_num) hT
        (show |(0 : ℝ)| ≤ 1 by norm_num) hN hf
  have hc : Continuous (𝓕 (fun x : ℝ => preparedProfile N f T₀ (x : ℂ))) := by
    change Continuous (VectorFourier.fourierIntegral Real.fourierChar volume (innerₗ ℝ)
      (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)))
    exact VectorFourier.fourierIntegral_continuous Real.continuous_fourierChar (by fun_prop) hi
  apply monomial_spectrum_integrable hc
    (mul_nonneg (preparationEnvelope_nonneg hM (show (0 : ℝ) ≤ 1 by norm_num) hT) (Real.sqrt_nonneg _))
    (show 0 < 2 * Real.pi by positivity)
  intro ξ
  simpa only [mul_one] using preparedProfile_cycles_decay hM (show (0 : ℝ) ≤ 1 by norm_num) hT hN hf ξ

/-- L1-to-kernel connector: the ACTUAL r-th derivative of the prepared profile
is inverse Fourier synthesis of its explicit derivative spectrum. -/
theorem preparedProfile_derivative_synthesis {f : ℝ → ℝ} {M T₀ : ℝ}
    (hM : 0 ≤ M) (hT : 0 < T₀) {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) (r : ℕ) :
    iteratedDeriv r (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)) =
      𝓕⁻ (fun ξ : ℝ => (2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r *
        𝓕 (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)) ξ) := by
  apply profile_derivative_synthesis
    ((preparedProfile_differentiable N f T₀).continuous.comp Complex.continuous_ofReal)
  · simpa only [Complex.ofReal_zero, zero_mul, add_zero, Function.comp_def] using
      preparedProfile_horizontal_integrable hM (show (0 : ℝ) ≤ 1 by norm_num) hT
        (show |(0 : ℝ)| ≤ 1 by norm_num) hN hf
  · exact preparedProfile_monomial_moments hM hT hN hf

/-- The actual derivative spectrum is L1, as required by the finite-kernel
certificate. -/
theorem preparedProfile_derivativeSpectrum_integrable {f : ℝ → ℝ} {M T₀ : ℝ}
    (hM : 0 ≤ M) (hT : 0 < T₀) {N : ℕ} (hN : 0 < N)
    (hf : ∀ j < N + 1, |f (sampleNode (N + 1) j)| ≤ M) (r : ℕ) :
    Integrable (fun ξ : ℝ => (2 * Real.pi * Complex.I * (ξ : ℂ)) ^ r *
      𝓕 (fun x : ℝ => preparedProfile N f T₀ (x : ℂ)) ξ) :=
  derivativeSpectrum_integrable (preparedProfile_monomial_moments hM hT hN hf) r

end QIRadon.ProfilePreparation

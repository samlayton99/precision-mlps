import QIRadon.Scalar.ProfilePreparation

/-! Stable finite real-sample polynomial preparation. The Fejér kernel is
constructed by finite Chebyshev sums; positivity is a sum-of-squares identity.
No analytic continuation of the supplied target is used. -/
noncomputable section
open Polynomial Finset
open Polynomial.Chebyshev
namespace QIRadon.ProfilePreparation.FilteredSamples

/-- Positive Chebyshev Fejér kernel as a polynomial in its first argument. -/
def fejerPolynomial (m : ℕ) (y : ℝ) : ℝ[X] :=
  (1/(m:ℝ)) • ∑ a ∈ range m, ∑ b ∈ range m,
    (T ℝ ((a:ℤ)-(b:ℤ))).eval y • T ℝ ((a:ℤ)-(b:ℤ))

def fejerKernel (m : ℕ) (x y : ℝ) : ℝ := (fejerPolynomial m y).eval x

theorem sum_pair_eq_sq (m : ℕ) (u : ℕ → ℝ) :
    (∑ a ∈ range m, ∑ b ∈ range m, u a*u b) = (∑ a ∈ range m, u a)^2 := by
  rw [pow_two, Finset.sum_mul]
  simp_rw [Finset.mul_sum]

/-- Positivity follows from four real squares after x=cos(t), y=cos(s).
The integer-index Chebyshev formula automatically treats negative a-b correctly. -/
theorem fejerKernel_nonneg (m : ℕ) {x y : ℝ} (hx : |x| ≤ 1) (hy : |y| ≤ 1) :
    0 ≤ fejerKernel m x y := by
  let t := Real.arccos x
  let s := Real.arccos y
  have hxt : Real.cos t = x := Real.cos_arccos (neg_le_of_abs_le hx) (le_of_abs_le hx)
  have hys : Real.cos s = y := Real.cos_arccos (neg_le_of_abs_le hy) (le_of_abs_le hy)
  rw [← hxt, ← hys]
  unfold fejerKernel fejerPolynomial
  simp only [Polynomial.eval_smul, Polynomial.eval_finsetSum, smul_eq_mul, T_real_cos,
    Int.cast_sub, Int.cast_natCast]
  apply mul_nonneg (by positivity)
  let cc := fun a : ℕ => Real.cos ((a:ℝ)*s)*Real.cos ((a:ℝ)*t)
  let cs := fun a : ℕ => Real.cos ((a:ℝ)*s)*Real.sin ((a:ℝ)*t)
  let sc := fun a : ℕ => Real.sin ((a:ℝ)*s)*Real.cos ((a:ℝ)*t)
  let ss := fun a : ℕ => Real.sin ((a:ℝ)*s)*Real.sin ((a:ℝ)*t)
  have hid : ∀ a b : ℕ,
      Real.cos (((a:ℝ)-(b:ℝ))*s)*Real.cos (((a:ℝ)-(b:ℝ))*t) =
      cc a*cc b+cs a*cs b+sc a*sc b+ss a*ss b := by
    intro a b
    simp only [sub_mul, Real.cos_sub]
    dsimp [cc,cs,sc,ss]
    ring
  simp_rw [hid, Finset.sum_add_distrib, sum_pair_eq_sq]
  positivity

/-- The actual finite-sample Fejér operator uses only the L specified real
Chebyshev nodes and has degree less than m. -/
def fejerSamples (m L : ℕ) (f : ℝ → ℝ) : ℝ[X] :=
  (1/(L:ℝ)) • ∑ j ∈ range L, f (sampleNode L j) • fejerPolynomial m (sampleNode L j)

/-- The eventual degree-N-reproducing filter, using 2N real Chebyshev nodes. -/
def filteredSamples (N : ℕ) (f : ℝ → ℝ) : ℝ[X] :=
  (2:ℝ) • fejerSamples (2*N) (2*N) f - fejerSamples N (2*N) f

theorem fejerSamples_eval (m L : ℕ) (f : ℝ → ℝ) (x : ℝ) :
    (fejerSamples m L f).eval x =
      (1/(L:ℝ))*∑ j ∈ range L, f (sampleNode L j)*fejerKernel m x (sampleNode L j) := by
  simp only [fejerSamples, fejerKernel, Polynomial.eval_smul, Polynomial.eval_finsetSum, smul_eq_mul]

/-- Swap only finite sums to expose the discrete Chebyshev moments. -/
theorem fejerSamples_eval_pairs (m L : ℕ) (f : ℝ → ℝ) (x : ℝ) :
    (fejerSamples m L f).eval x = (1/(m:ℝ))*
      ∑ a ∈ range m, ∑ b ∈ range m,
        ((1/(L:ℝ))*∑ j ∈ range L,
          f (sampleNode L j)*(T ℝ ((a:ℤ)-(b:ℤ))).eval (sampleNode L j)) *
            (T ℝ ((a:ℤ)-(b:ℤ))).eval x := by
  simp only [fejerSamples_eval, fejerKernel, fejerPolynomial, Polynomial.eval_smul,
    Polynomial.eval_finsetSum, smul_eq_mul]
  simp_rw [Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro a _
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro b _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Exact sample average of each used Chebyshev mode. -/
theorem sample_average_T {L : ℕ} (hL : 0 < L) {k : ℤ} (hk : k.natAbs < L) :
    (1/(L:ℝ))*∑ j ∈ range L, (T ℝ k).eval (sampleNode L j) =
      if k=0 then 1 else 0 := by
  have hh := discreteInner_eq (Nat.ne_of_gt hL) hk hL
  unfold discreteInner at hh
  rw [T_natAbs ℝ k] at hh
  by_cases hk0 : k=0
  · simpa [hk0] using hh
  · have hka : k.natAbs ≠ 0 := by omega
    simpa [hk0,hka] using hh

theorem difference_natAbs_lt {a b m : ℕ} (ha : a < m) (hb : b < m) :
    ((a:ℤ)-(b:ℤ)).natAbs < m := by omega

/-- Exact preservation of constants, derived from discrete orthogonality. -/
theorem fejerSamples_one {m L : ℕ} (hm : 0 < m) (hmL : m ≤ L) (x : ℝ) :
    (fejerSamples m L (fun _ => 1)).eval x = 1 := by
  rw [fejerSamples_eval_pairs]
  have hL : 0 < L := hm.trans_le hmL
  have hmR : (m:ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  have havg : ∀ a ∈ range m, ∀ b ∈ range m,
      ((1/(L:ℝ))*∑ j ∈ range L,
        1*(T ℝ ((a:ℤ)-(b:ℤ))).eval (sampleNode L j)) =
      if a=b then 1 else 0 := by
    intro a ha b hb
    have hh := sample_average_T hL ((difference_natAbs_lt (mem_range.mp ha) (mem_range.mp hb)).trans_le hmL)
    simpa only [one_mul, sub_eq_zero, Nat.cast_inj] using hh
  have hdiag : ∀ a ∈ range m,
      (∑ b ∈ range m,
        ((1/(L:ℝ))*∑ j ∈ range L,
          1*(T ℝ ((a:ℤ)-(b:ℤ))).eval (sampleNode L j)) *
            (T ℝ ((a:ℤ)-(b:ℤ))).eval x) = 1 := by
    intro a ha
    calc
      _ = ∑ b ∈ range m, if b=a then (1:ℝ) else 0 := by
        apply Finset.sum_congr rfl
        intro b hb
        rw [havg a ha b hb]
        by_cases hab : a=b
        · subst b; simp
        · simp [hab, Ne.symm hab]
      _ = 1 := by simp [ha]
  rw [Finset.sum_congr rfl hdiag]
  simp [hmR]

/-- The exact discrete kernel mass is one, at every evaluation point. -/
theorem fejerKernel_sample_mass {m L : ℕ} (hm : 0 < m) (hmL : m ≤ L) (x : ℝ) :
    (1/(L:ℝ))*∑ j ∈ range L, fejerKernel m x (sampleNode L j) = 1 := by
  have hh := fejerSamples_one hm hmL x
  simpa only [fejerSamples_eval, one_mul] using hh

/-- Positive sample kernel and mass one imply operator norm at most one,
independent of polynomial degree and sample count. -/
theorem fejerSamples_bound {m L : ℕ} (hm : 0 < m) (hmL : m ≤ L)
    {f : ℝ → ℝ} {M x : ℝ} (_hM : 0 ≤ M)
    (hf : ∀ j < L, |f (sampleNode L j)| ≤ M) (hx : |x| ≤ 1) :
    |(fejerSamples m L f).eval x| ≤ M := by
  have hL : 0 < L := hm.trans_le hmL
  rw [fejerSamples_eval, abs_mul, abs_of_nonneg (by positivity : 0 ≤ 1/(L:ℝ))]
  calc
    _ ≤ (1/(L:ℝ))*∑ j ∈ range L, |f (sampleNode L j)*fejerKernel m x (sampleNode L j)| :=
      mul_le_mul_of_nonneg_left (Finset.abs_sum_le_sum_abs _ _) (by positivity)
    _ ≤ (1/(L:ℝ))*∑ j ∈ range L, M*fejerKernel m x (sampleNode L j) := by
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      apply Finset.sum_le_sum
      intro j hj
      have hk := fejerKernel_nonneg m hx (sampleNode_abs_le_one L j)
      rw [abs_mul, abs_of_nonneg hk]
      exact mul_le_mul_of_nonneg_right (hf j (mem_range.mp hj)) hk
    _ = M := by rw [← Finset.mul_sum, mul_left_comm, fejerKernel_sample_mass hm hmL, mul_one]

/-- The actual de la Vallée Poussin sample filter is uniformly bounded by three. -/
theorem filteredSamples_bound {N : ℕ} (hN : 0 < N) {f : ℝ → ℝ} {M x : ℝ}
    (hM : 0 ≤ M) (hf : ∀ j < 2*N, |f (sampleNode (2*N) j)| ≤ M) (hx : |x| ≤ 1) :
    |(filteredSamples N f).eval x| ≤ 3*M := by
  have htwo := fejerSamples_bound (m := 2*N) (by omega) le_rfl hM hf hx
  have hone := fejerSamples_bound hN (by omega : N ≤ 2*N) hM hf hx
  simp only [filteredSamples, Polynomial.eval_sub, Polynomial.eval_smul, smul_eq_mul]
  have hh := abs_sub_le (2*(fejerSamples (2*N) (2*N) f).eval x) 0
    ((fejerSamples N (2*N) f).eval x)
  norm_num only [sub_zero, zero_sub, abs_neg, abs_mul, abs_of_nonneg (by norm_num : (0:ℝ)≤2)] at hh
  linarith

/-- Count one diagonal in the finite m-by-m difference table. -/
theorem sum_oriented_difference (m l : ℕ) (z : ℝ) :
    (∑ a ∈ range m, ∑ b ∈ range m, if a=b+l then z else 0) = ((m-l:ℕ):ℝ)*z := by
  rw [Finset.sum_comm]
  simp only [Finset.sum_ite_eq', mem_range]
  have hfilter : (range m).filter (fun b => b + l < m) = range (m-l) := by
    ext b
    simp only [mem_filter, mem_range]
    omega
  rw [← Finset.sum_filter, hfilter]
  simp

/-- Count the two off-diagonals, or the single central diagonal, exactly. -/
theorem sum_absolute_difference (m l : ℕ) (z : ℝ) :
    (∑ a ∈ range m, ∑ b ∈ range m,
      if ((a:ℤ)-(b:ℤ)).natAbs=l then z else 0) =
      (if l=0 then (m:ℝ) else 2*((m-l:ℕ):ℝ))*z := by
  by_cases hl : l=0
  · subst l
    have heq : ∀ a b : ℕ, (((a:ℤ)-(b:ℤ)).natAbs=0) ↔ a=b := by intro a b; omega
    simp only [heq, Finset.sum_ite_eq, Finset.mem_range, ite_true]
    have hh : (∑ a ∈ range m, if a < m then z else 0) = (m:ℝ)*z := by
      calc
        _ = ∑ a ∈ range m, z := by apply Finset.sum_congr rfl; intro a ha; simp [mem_range.mp ha]
        _ = _ := by simp
    exact hh
  · have hsplit : ∀ a b : ℕ,
        (if ((a:ℤ)-(b:ℤ)).natAbs=l then z else 0) =
          (if a=b+l then z else 0)+(if b=a+l then z else 0) := by
      intro a b
      split_ifs <;> first | omega | ring
    simp_rw [hsplit, Finset.sum_add_distrib]
    rw [sum_oriented_difference, Finset.sum_comm (s := range m) (t := range m)]
    rw [sum_oriented_difference, if_neg hl]
    ring

/-- Exact Fejér multiplier on every reproduced Chebyshev mode. -/
theorem fejerSamples_T {m L l : ℕ} (hl : l < m) (hmL : m ≤ L) (x : ℝ) :
    (fejerSamples m L (T ℝ l).eval).eval x =
      (1-(l:ℝ)/(m:ℝ))*(T ℝ l).eval x := by
  have hm : 0 < m := by omega
  have hL : 0 < L := hm.trans_le hmL
  have hmR : (m:ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  rw [fejerSamples_eval_pairs]
  have hterm : ∀ a ∈ range m, ∀ b ∈ range m,
      ((1/(L:ℝ))*∑ j ∈ range L,
          (T ℝ l).eval (sampleNode L j)*(T ℝ ((a:ℤ)-(b:ℤ))).eval (sampleNode L j)) *
          (T ℝ ((a:ℤ)-(b:ℤ))).eval x =
      if ((a:ℤ)-(b:ℤ)).natAbs=l then
        (if l=0 then 1 else 1/2)*(T ℝ l).eval x else 0 := by
    intro a ha b hb
    have hd := difference_natAbs_lt (mem_range.mp ha) (mem_range.mp hb)
    have hmom := discreteInner_eq (Nat.ne_of_gt hL) (hl.trans_le hmL) (hd.trans_le hmL)
    unfold discreteInner at hmom
    rw [T_natAbs ℝ ((a:ℤ)-(b:ℤ))] at hmom
    rw [hmom]
    by_cases heq : ((a:ℤ)-(b:ℤ)).natAbs=l
    · rw [if_pos heq.symm, if_pos heq, ← T_natAbs ℝ ((a:ℤ)-(b:ℤ)), heq]
    · simp [heq, Ne.symm heq]
  have hsum : (∑ a ∈ range m, ∑ b ∈ range m,
      ((1/(L:ℝ))*∑ j ∈ range L,
          (T ℝ l).eval (sampleNode L j)*(T ℝ ((a:ℤ)-(b:ℤ))).eval (sampleNode L j)) *
          (T ℝ ((a:ℤ)-(b:ℤ))).eval x) =
      ∑ a ∈ range m, ∑ b ∈ range m, if ((a:ℤ)-(b:ℤ)).natAbs=l then
        (if l=0 then 1 else 1/2)*(T ℝ l).eval x else 0 := by
    apply Finset.sum_congr rfl
    intro a ha
    exact Finset.sum_congr rfl (fun b hb => hterm a ha b hb)
  rw [hsum, sum_absolute_difference]
  by_cases hl0 : l=0
  · simp [hl0,hmR]
  · rw [if_neg hl0, if_neg hl0, Nat.cast_sub hl.le]
    field_simp

/-- The filtered operator reproduces every Chebyshev basis polynomial below N. -/
theorem filteredSamples_T {N l : ℕ} (hN : 0 < N) (hl : l < N) :
    filteredSamples N (T ℝ l).eval = T ℝ l := by
  apply Polynomial.funext
  intro x
  simp only [filteredSamples, Polynomial.eval_sub, Polynomial.eval_smul, smul_eq_mul]
  rw [fejerSamples_T (by omega : l < 2*N) le_rfl,
    fejerSamples_T hl (by omega : N ≤ 2*N)]
  have hNR : (N:ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  push_cast
  field_simp
  ring

/-- The sample filter is a concrete linear map in the supplied sample values. -/
def filteredOperator (N : ℕ) : (ℝ → ℝ) →ₗ[ℝ] ℝ[X] where
  toFun := filteredSamples N
  map_add' f g := by
    simp only [filteredSamples, fejerSamples, Pi.add_apply, add_smul,
      Finset.sum_add_distrib, smul_add]
    abel
  map_smul' c f := by
    simp only [filteredSamples, fejerSamples, Pi.smul_apply, smul_eq_mul,
      Finset.smul_sum, smul_sub, smul_smul, RingHom.id_apply]
    simp only [mul_comm, mul_left_comm]

@[simp] theorem filteredOperator_apply (N : ℕ) (f : ℝ → ℝ) :
    filteredOperator N f = filteredSamples N f := rfl

/-- Exact low-degree reproduction, so the stable operator loses no algebraic
approximation exponent when applied to real samples. -/
theorem filteredSamples_reproduces {N : ℕ} (hN : 0 < N) {P : ℝ[X]}
    (hP : P.degree < N) : filteredSamples N P.eval = P := by
  have hmem : P ∈ Polynomial.degreeLT ℝ N := by rwa [Polynomial.mem_degreeLT]
  rw [← Polynomial.Sequence.span_degreeLT (chebyshevTsequence ℝ) (by simp),
    show Set.Iio N = Finset.range N by ext; simp,
    Submodule.mem_span_image_finset_iff_exists_fun'] at hmem
  obtain ⟨c,rfl⟩ := hmem
  simp only [chebyshevTsequence]
  have heval : (∑ i ∈ range N, c i • T ℝ i).eval =
      ∑ i ∈ range N, c i • (T ℝ i).eval := by
    ext x
    simp only [Polynomial.eval_finsetSum, Polynomial.eval_smul, Finset.sum_apply, Pi.smul_apply]
  rw [heval]
  change filteredOperator N (∑ i ∈ range N, c i • (T ℝ i).eval) = _
  rw [map_sum]
  apply Finset.sum_congr rfl
  intro i hi
  rw [map_smul]
  simp only [filteredOperator_apply, filteredSamples_T hN (mem_range.mp hi)]

/-- Four-times-near-best uniform approximation by the actual real-sample
filter. The polynomial comparator is a regularity-class witness, not an input to
the construction: only f's 2N Chebyshev values enter filteredSamples. -/
theorem filteredSamples_error_of_comparator {N : ℕ} (hN : 0 < N)
    {f : ℝ → ℝ} {P : ℝ[X]} {E x : ℝ} (hP : P.degree < N) (hE : 0 ≤ E)
    (happrox : ∀ y : ℝ, |y| ≤ 1 → |f y-P.eval y| ≤ E) (hx : |x| ≤ 1) :
    |f x-(filteredSamples N f).eval x| ≤ 4*E := by
  have hsamples : ∀ j < 2*N, |(f-P.eval) (sampleNode (2*N) j)| ≤ E := by
    intro j _
    exact happrox _ (sampleNode_abs_le_one _ _)
  have hb := filteredSamples_bound hN hE hsamples hx
  have hlin : filteredSamples N (f-P.eval) = filteredSamples N f-P := by
    change filteredOperator N (f-P.eval) = filteredOperator N f-P
    rw [map_sub]
    simp only [filteredOperator_apply, filteredSamples_reproduces hN hP]
  rw [hlin, Polynomial.eval_sub] at hb
  have hid : f x-(filteredSamples N f).eval x =
      (f x-P.eval x)-((filteredSamples N f).eval x-P.eval x) := by ring
  rw [hid]
  have hh := abs_sub_le (f x-P.eval x) 0 ((filteredSamples N f).eval x-P.eval x)
  simp only [sub_zero, zero_sub, abs_neg] at hh
  linarith [happrox x hx]

theorem fejerPolynomial_degree_le (m : ℕ) (y : ℝ) :
    (fejerPolynomial m y).degree ≤ m := by
  apply (Polynomial.degree_smul_le _ _).trans
  apply (Polynomial.degree_sum_le _ _).trans
  apply Finset.sup_le
  intro a ha
  apply (Polynomial.degree_sum_le _ _).trans
  apply Finset.sup_le
  intro b hb
  apply (Polynomial.degree_smul_le _ _).trans
  rw [degree_T]
  exact_mod_cast (difference_natAbs_lt (mem_range.mp ha) (mem_range.mp hb)).le

theorem fejerSamples_degree_le (m L : ℕ) (f : ℝ → ℝ) :
    (fejerSamples m L f).degree ≤ m := by
  apply (Polynomial.degree_smul_le _ _).trans
  apply (Polynomial.degree_sum_le _ _).trans
  apply Finset.sup_le
  intro j _
  exact (Polynomial.degree_smul_le _ _).trans (fejerPolynomial_degree_le m _)

/-- The explicitly sampled polynomial has degree at most 2N. -/
theorem filteredSamples_degree_le (N : ℕ) (f : ℝ → ℝ) :
    (filteredSamples N f).degree ≤ 2*N := by
  unfold filteredSamples
  apply (Polynomial.degree_sub_le _ _).trans
  apply max_le
  · exact (Polynomial.degree_smul_le _ _).trans (fejerSamples_degree_le (2*N) (2*N) f)
  · exact (fejerSamples_degree_le N (2*N) f).trans (by exact_mod_cast (show N ≤ 2*N by omega))

/-- The filtered construction depends only on its 2N prescribed real samples. -/
theorem filteredSamples_congr {N : ℕ} {f g : ℝ → ℝ}
    (hfg : ∀ j < 2*N, f (sampleNode (2*N) j)=g (sampleNode (2*N) j)) :
    filteredSamples N f = filteredSamples N g := by
  have hf : ∀ m, fejerSamples m (2*N) f=fejerSamples m (2*N) g := by
    intro m
    unfold fejerSamples
    congr 1
    exact Finset.sum_congr rfl (fun j hj => by rw [hfg j (mem_range.mp hj)])
  simp only [filteredSamples, hf]

/-- Local polynomial approximability gives the same algebraic rate for an
actual real-sample construction, without a factor N or any complex target data. -/
theorem filteredSamples_algebraic_error {N : ℕ} (hN : 0 < N)
    {f : ℝ → ℝ} {A s : ℝ} (hA : 0 ≤ A)
    (happrox : ∃ P : ℝ[X], P.degree < N ∧
      ∀ x : ℝ, |x| ≤ 1 → |f x-P.eval x| ≤ A*(N:ℝ)^(-s)) :
    ∀ x : ℝ, |x| ≤ 1 → |f x-(filteredSamples N f).eval x| ≤ 4*A*(N:ℝ)^(-s) := by
  obtain ⟨P,hP,hPe⟩ := happrox
  intro x hx
  have hh := filteredSamples_error_of_comparator hN hP (by positivity : 0 ≤ A*(N:ℝ)^(-s)) hPe hx
  simpa only [mul_assoc] using hh

end QIRadon.ProfilePreparation.FilteredSamples

import QIRadon.Quadrature.SphereRule

open scoped BigOperators

namespace QIRadon.Gaussian

noncomputable def sphereJacobiReflection (q : ℕ) (hq : 0 < q) (n : ℕ) : Fin q ≃ Fin q :=
  (gaussJacobi_reflection_of_exact (sphereJacobiExponent n) (sphereJacobiExponent_gt n)
    hq (sphereJacobiNode q hq n) (sphereJacobi_injective q hq n) (sphereJacobi_exact q hq n)).choose

theorem sphereJacobiReflection_spec (q : ℕ) (hq : 0 < q) (n : ℕ) :
    Function.Involutive (sphereJacobiReflection q hq n) ∧
    (∀ i, sphereJacobiNode q hq n (sphereJacobiReflection q hq n i) = -sphereJacobiNode q hq n i) ∧
    (∀ i, sphereJacobiWeight q hq n (sphereJacobiReflection q hq n i) = sphereJacobiWeight q hq n i) :=
  (gaussJacobi_reflection_of_exact (sphereJacobiExponent n) (sphereJacobiExponent_gt n)
    hq (sphereJacobiNode q hq n) (sphereJacobi_injective q hq n) (sphereJacobi_exact q hq n)).choose_spec

noncomputable def sphereReflection (q : ℕ) (hq : 0 < q) :
    (n : ℕ) → SphereIndex q n ≃ SphereIndex q n
  | 0 => Equiv.boolNot
  | n + 1 => Equiv.prodCongr (sphereJacobiReflection q hq n) (sphereReflection q hq n)

def terminalSign (q : ℕ) : (n : ℕ) → SphereIndex q n → Bool
  | 0, i => i
  | n + 1, ij => terminalSign q n ij.2

theorem sphereReflection_involutive (q : ℕ) (hq : 0 < q) (n : ℕ) :
    Function.Involutive (sphereReflection q hq n) := by
  induction n with
  | zero => intro i; exact Bool.not_not i
  | succ n ih =>
    intro i
    apply Prod.ext
    · exact (sphereJacobiReflection_spec q hq n).1 i.1
    · exact ih i.2

theorem terminalSign_reflection (q : ℕ) (hq : 0 < q) (n : ℕ) (i : SphereIndex q n) :
    terminalSign q n (sphereReflection q hq n i) = !terminalSign q n i := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih i.2

theorem liftPoint_neg {n : ℕ} (t : ℝ) (v : EuclideanSpace ℝ (Fin n)) :
    liftPoint (-t) (-v) = -liftPoint t v := by
  ext i
  refine Fin.cases ?_ (fun j ↦ ?_) i
  · simp
  · simp

theorem sphereNode_reflection (q : ℕ) (hq : 0 < q) (n : ℕ) (i : SphereIndex q n) :
    sphereNode q hq n (sphereReflection q hq n i) = -sphereNode q hq n i := by
  induction n with
  | zero =>
    cases i with
    | false => ext j; change (1 : ℝ) = -(-1); norm_num
    | true => ext j; rfl
  | succ n ih =>
    change liftPoint (sphereJacobiNode q hq n (sphereJacobiReflection q hq n i.1))
      (sphereNode q hq n (sphereReflection q hq n i.2)) = _
    rw [(sphereJacobiReflection_spec q hq n).2.1, ih, liftPoint_neg]
    rfl

theorem sphereWeight_reflection (q : ℕ) (hq : 0 < q) (n : ℕ) (i : SphereIndex q n) :
    sphereWeight q hq n (sphereReflection q hq n i) = sphereWeight q hq n i := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change sphereJacobiWeight q hq n (sphereJacobiReflection q hq n i.1) *
      sphereWeight q hq n (sphereReflection q hq n i.2) = _
    rw [(sphereJacobiReflection_spec q hq n).2.2, ih]
    rfl

def SphereHalfIndex (q n : ℕ) := {i : SphereIndex q n // terminalSign q n i = false}

instance sphereHalfIndexFintype (q n : ℕ) : Fintype (SphereHalfIndex q n) :=
  inferInstanceAs (Fintype {i : SphereIndex q n // terminalSign q n i = false})

noncomputable def halfComplementEquiv (q : ℕ) (hq : 0 < q) (n : ℕ) :
    SphereHalfIndex q n ≃ {i : SphereIndex q n // ¬terminalSign q n i = false} where
  toFun i := ⟨sphereReflection q hq n i.val, by
    rw [terminalSign_reflection, i.property]
    decide⟩
  invFun i := ⟨sphereReflection q hq n i.val, by
    rw [terminalSign_reflection]
    cases h : terminalSign q n i.val with
    | false => exact False.elim (i.property h)
    | true => rfl⟩
  left_inv i := Subtype.ext (sphereReflection_involutive q hq n i.val)
  right_inv i := Subtype.ext (sphereReflection_involutive q hq n i.val)

/-- One representative per genuine antipodal pair gives exactly q^(d-1) nodes. -/
theorem sphereHalfIndex_card (q : ℕ) (hq : 0 < q) (n : ℕ) :
    Fintype.card (SphereHalfIndex q n) = q ^ n := by
  classical
  have hc := Fintype.card_congr (halfComplementEquiv q hq n)
  have hs := Fintype.sum_subtype_add_sum_subtype
    (fun i : SphereIndex q n ↦ terminalSign q n i = false) (fun _ ↦ (1 : ℕ))
  simp only [Finset.sum_const, Finset.card_univ, smul_eq_mul, mul_one] at hs
  change Fintype.card (SphereHalfIndex q n) +
    Fintype.card {i : SphereIndex q n // ¬terminalSign q n i = false} =
    Fintype.card (SphereIndex q n) at hs
  rw [← hc, sphereIndex_card] at hs
  omega

/-- Sum over an invariant function is twice the sum over chosen representatives. -/
theorem sum_invariant_eq_twice_half (q : ℕ) (hq : 0 < q) (n : ℕ)
    (F : SphereIndex q n → ℝ)
    (hF : ∀ i, F (sphereReflection q hq n i) = F i) :
    (∑ i, F i) = 2 * ∑ i : SphereHalfIndex q n, F i.val := by
  classical
  have hp := (halfComplementEquiv q hq n).sum_comp
    (fun i : {i : SphereIndex q n // ¬terminalSign q n i = false} ↦ F i.val)
  change (∑ i : SphereHalfIndex q n, F (sphereReflection q hq n i.val)) = _ at hp
  simp_rw [hF] at hp
  have hs := Fintype.sum_subtype_add_sum_subtype
    (fun i : SphereIndex q n ↦ terminalSign q n i = false) F
  change (∑ i : SphereHalfIndex q n, F i.val) +
    (∑ i : {i : SphereIndex q n // ¬terminalSign q n i = false}, F i.val) = _ at hs
  rw [← hp] at hs
  linarith

noncomputable def foldedSphereWeight (q : ℕ) (hq : 0 < q) (n : ℕ)
    (i : SphereHalfIndex q n) : ℝ := 2 * sphereWeight q hq n i.val

theorem foldedSphereWeight_positive (q : ℕ) (hq : 0 < q) (n : ℕ)
    (i : SphereHalfIndex q n) : 0 < foldedSphereWeight q hq n i :=
  mul_pos (by norm_num) (sphereWeight_positive q hq n i.val)

theorem foldedSphereWeight_normalized (q : ℕ) (hq : 0 < q) (n : ℕ) :
    ∑ i, foldedSphereWeight q hq n i = 1 := by
  have h := sum_invariant_eq_twice_half q hq n (sphereWeight q hq n)
    (sphereWeight_reflection q hq n)
  rw [sphereWeight_normalized] at h
  simpa only [foldedSphereWeight, Finset.mul_sum] using h.symm

/-- Antipodal folding preserves integration of every even function. It does not
claim preservation for odd functions or arbitrary noneven polynomials. -/
theorem foldedSphere_sum_even (q : ℕ) (hq : 0 < q) (n : ℕ)
    (F : EuclideanSpace ℝ (Fin (n + 1)) → ℝ) (heven : ∀ x, F (-x) = F x) :
    (∑ i, sphereWeight q hq n i * F (sphereNode q hq n i)) =
      ∑ i : SphereHalfIndex q n,
        foldedSphereWeight q hq n i * F (sphereNode q hq n i.val) := by
  have h := sum_invariant_eq_twice_half q hq n
    (fun i ↦ sphereWeight q hq n i * F (sphereNode q hq n i)) (by
      intro i
      rw [sphereWeight_reflection, sphereNode_reflection, heven])
  rw [h, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  unfold foldedSphereWeight
  ring

end QIRadon.Gaussian

import QIRadon.Quadrature.JacobiGeneral
import QIRadon.Quadrature.OrthogonalPolynomial
import Mathlib.Algebra.Polynomial.RingDivision

/-! Reflection symmetry of the actual symmetric Gauss--Jacobi construction. -/

noncomputable section
open Polynomial MeasureTheory
open scoped BigOperators

namespace QIRadon.Gaussian

/-- The actual unnormalized Jacobi integral is invariant under reflection. -/
theorem rawJacobiMomentGeneral_reflect (a : ℝ) (ha : -1 < a) (p : ℝ[X]) :
    rawJacobiMomentGeneral a ha (p.comp (-X)) = rawJacobiMomentGeneral a ha p := by
  have h := intervalIntegral.integral_comp_neg
    (fun x : ℝ => jacobiDensity a x * p.eval x) (a := (-1 : ℝ)) (b := (1 : ℝ))
  simpa [rawJacobiMomentGeneral, jacobiDensity, Polynomial.eval_comp] using h

/-- Normalizing the Jacobi integral preserves reflection symmetry. -/
theorem jacobiMomentGeneral_reflect (a : ℝ) (ha : -1 < a) (p : ℝ[X]) :
    jacobiMomentGeneral a ha (p.comp (-X)) = jacobiMomentGeneral a ha p := by
  simp only [jacobiMomentGeneral, LinearMap.smul_apply, smul_eq_mul,
    rawJacobiMomentGeneral_reflect]

/-- For a reflection-invariant positive functional, the monic orthogonal
polynomial has parity fixed by its degree. Reflection parity is proved from
uniqueness, not assumed of the nodes. -/
theorem monic_orthogonal_reflection (L : ℝ[X] →ₗ[ℝ] ℝ) (n : ℕ)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → 0 < L (u^2))
    (hsym : ∀ u : ℝ[X], L (u.comp (-X)) = L u)
    (p : ℝ[X]) (hp : p.Monic) (hpn : p.natDegree = n)
    (horth : ∀ u : ℝ[X], u.degree < n → L (p*u) = 0) :
    C ((-1 : ℝ)^n) * p.comp (-X) = p := by
  let q := C ((-1 : ℝ)^n) * p.comp (-X)
  have hq : q.Monic := by
    simpa [q, hpn] using hp.neg_one_pow_natDegree_mul_comp_neg_X
  have hqn : q.natDegree = n := by
    dsimp [q]
    rw [Polynomial.natDegree_C_mul (pow_ne_zero _ (by norm_num : (-1 : ℝ) ≠ 0)), Polynomial.natDegree_comp]
    simp [hpn]
  have hqo : ∀ u : ℝ[X], u.degree < n → L (q*u) = 0 := by
    intro u hu
    have hru : (u.comp (-X)).degree < n := by simpa using hu
    have ho := horth (u.comp (-X)) hru
    have heq : p.comp (-X) * u = (p * u.comp (-X)).comp (-X) := by
      simp [Polynomial.mul_comp, Polynomial.comp_neg_X_comp_neg_X]
    calc
      L (q*u) = (-1 : ℝ)^n * L (p.comp (-X) * u) := by
        dsimp [q]
        rw [mul_assoc, ← Polynomial.smul_eq_C_mul, map_smul]
        rfl
      _ = (-1 : ℝ)^n * L (p * u.comp (-X)) := by rw [heq, hsym]
      _ = 0 := by rw [ho, mul_zero]
  exact monic_orthogonal_unique L n hpos q p hq hp hqn hpn hqo horth

/-- Reflection sends every root of the monic orthogonal polynomial to a root. -/
theorem monic_orthogonal_root_neg (L : ℝ[X] →ₗ[ℝ] ℝ) (n : ℕ)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → 0 < L (u^2))
    (hsym : ∀ u : ℝ[X], L (u.comp (-X)) = L u)
    (p : ℝ[X]) (hp : p.Monic) (hpn : p.natDegree = n)
    (horth : ∀ u : ℝ[X], u.degree < n → L (p*u) = 0)
    {x : ℝ} (hx : p.eval x = 0) : p.eval (-x) = 0 := by
  have he := congrArg (fun u : ℝ[X] => u.eval x)
    (monic_orthogonal_reflection L n hpos hsym p hp hpn horth)
  simp only [Polynomial.eval_mul, Polynomial.eval_C, Polynomial.eval_comp,
    Polynomial.eval_neg, Polynomial.eval_X, hx] at he
  exact (mul_eq_zero.mp he).resolve_left (pow_ne_zero _ (by norm_num : (-1 : ℝ) ≠ 0))

/-- Orthogonal nodal grids of a symmetric functional are closed under negation. -/
theorem orthogonal_nodal_neg_closed (L : ℝ[X] →ₗ[ℝ] ℝ) {n : ℕ}
    (hpos : ∀ u : ℝ[X], u ≠ 0 → 0 < L (u^2))
    (hsym : ∀ u : ℝ[X], L (u.comp (-X)) = L u)
    (node : Fin n → ℝ)
    (horth : ∀ u : ℝ[X], u.degree < n →
      L (Lagrange.nodal Finset.univ node * u) = 0) :
    ∀ i, ∃ j, node j = -node i := by
  intro i
  have hroot : (Lagrange.nodal Finset.univ node).eval (node i) = 0 :=
    Lagrange.eval_nodal_at_node (Finset.mem_univ i)
  have hneg := monic_orthogonal_root_neg L n hpos hsym
    (Lagrange.nodal Finset.univ node) Lagrange.nodal_monic (by simp)
    horth hroot
  rw [Lagrange.eval_nodal] at hneg
  obtain ⟨j, hj, heq⟩ := Finset.prod_eq_zero_iff.mp hneg
  exact ⟨j, (sub_eq_zero.mp heq).symm⟩

/-- Reflecting a symmetric node grid permutes its interpolation basis. -/
theorem lagrange_basis_reflection {n : ℕ} (hn : 0 < n)
    (node : Fin n → ℝ) (hinj : Function.Injective node)
    (σ : Fin n ≃ Fin n) (hinvol : Function.Involutive σ)
    (hnode : ∀ i, node (σ i) = -node i) (i : Fin n) :
    (Lagrange.basis Finset.univ node i).comp (-X) =
      Lagrange.basis Finset.univ node (σ i) := by
  have hdeg : ∀ j : Fin n, (Lagrange.basis Finset.univ node j).degree <
      (Finset.univ : Finset (Fin n)).card := by
    intro j
    rw [Lagrange.degree_basis hinj.injOn (Finset.mem_univ j)]
    simp only [Finset.card_univ, Fintype.card_fin]
    exact_mod_cast Nat.sub_lt hn (by norm_num : 0 < (1 : ℕ))
  apply Polynomial.eq_of_degrees_lt_of_eval_index_eq Finset.univ hinj.injOn
    (by simpa using hdeg i) (hdeg (σ i))
  intro j hj
  rw [Polynomial.eval_comp, Polynomial.eval_neg, Polynomial.eval_X, ← hnode j]
  rw [basis_eval node hinj, basis_eval node hinj]
  have hiff : i = σ j ↔ σ i = j := by
    constructor
    · intro h
      rw [h, hinvol j]
    · intro h
      calc
        i = σ (σ i) := (hinvol i).symm
        _ = σ j := congrArg σ h
  simp only [hiff]

/-- The actual integrated-Lagrange weights of reflected nodes are equal. -/
theorem weight_reflection (L : ℝ[X] →ₗ[ℝ] ℝ)
    (hsym : ∀ u : ℝ[X], L (u.comp (-X)) = L u)
    {n : ℕ} (hn : 0 < n) (node : Fin n → ℝ) (hinj : Function.Injective node)
    (σ : Fin n ≃ Fin n) (hinvol : Function.Involutive σ)
    (hnode : ∀ i, node (σ i) = -node i) (i : Fin n) :
    weight L node (σ i) = weight L node i := by
  unfold weight
  rw [← lagrange_basis_reflection hn node hinj σ hinvol hnode i, hsym]

/-- A symmetric orthogonal grid has an actual involutive, weight-preserving
reflection permutation. Existence of reflected nodes is derived, not assumed. -/
theorem exists_orthogonal_reflection (L : ℝ[X] →ₗ[ℝ] ℝ) {n : ℕ} (hn : 0 < n)
    (hpos : ∀ u : ℝ[X], u ≠ 0 → 0 < L (u^2))
    (hsym : ∀ u : ℝ[X], L (u.comp (-X)) = L u)
    (node : Fin n → ℝ) (hinj : Function.Injective node)
    (horth : ∀ u : ℝ[X], u.degree < n →
      L (Lagrange.nodal Finset.univ node * u) = 0) :
    ∃ σ : Fin n ≃ Fin n, Function.Involutive σ ∧
      (∀ i, node (σ i) = -node i) ∧
      (∀ i, weight L node (σ i) = weight L node i) := by
  have hclosed := orthogonal_nodal_neg_closed L hpos hsym node horth
  let τ : Fin n → Fin n := fun i => Classical.choose (hclosed i)
  have htau : ∀ i, node (τ i) = -node i := fun i => Classical.choose_spec (hclosed i)
  have hinvol : Function.Involutive τ := by
    intro i
    apply hinj
    rw [htau, htau, neg_neg]
  let σ : Fin n ≃ Fin n := ⟨τ, τ, hinvol, hinvol⟩
  have hsigma : Function.Involutive σ := hinvol
  have hsigmanode : ∀ i, node (σ i) = -node i := htau
  exact ⟨σ, hsigma, hsigmanode, weight_reflection L hsym hn node hinj σ hsigma hsigmanode⟩

/-- Exactness through degree 2n-1 already implies orthogonality of the nodal
polynomial. This recovers the property from the existing Gaussian-rule API. -/
theorem nodal_orthogonal_of_exact (L : ℝ[X] →ₗ[ℝ] ℝ) {n : ℕ} (_hn : 0 < n)
    (node : Fin n → ℝ)
    (hexact : ∀ p : ℝ[X], p.natDegree < 2*n →
      L p = ∑ i, weight L node i * p.eval (node i)) :
    ∀ u : ℝ[X], u.degree < n → L (Lagrange.nodal Finset.univ node * u) = 0 := by
  intro u hu
  by_cases hu0 : u = 0
  · simp [hu0]
  have hnat : u.natDegree < n := by
    rw [Polynomial.degree_eq_natDegree hu0] at hu
    exact_mod_cast hu
  have hd : (Lagrange.nodal Finset.univ node * u).natDegree < 2*n := by
    rw [Polynomial.natDegree_mul (Lagrange.nodal_ne_zero) hu0]
    simp only [Lagrange.natDegree_nodal, Finset.card_univ, Fintype.card_fin]
    omega
  rw [hexact _ hd]
  apply Finset.sum_eq_zero
  intro i hi
  rw [Polynomial.eval_mul, Lagrange.eval_nodal_at_node (Finset.mem_univ i), zero_mul, mul_zero]

/-- Every actual Gauss--Jacobi rule supplied by the constructive exactness theorem
has the required node and weight reflection permutation. -/
theorem gaussJacobi_reflection_of_exact (a : ℝ) (ha : -1 < a) {n : ℕ} (hn : 0 < n)
    (node : Fin n → ℝ) (hinj : Function.Injective node)
    (hexact : ∀ p : ℝ[X], p.natDegree < 2*n →
      jacobiMomentGeneral a ha p =
        ∑ i, weight (jacobiMomentGeneral a ha) node i * p.eval (node i)) :
    ∃ σ : Fin n ≃ Fin n, Function.Involutive σ ∧
      (∀ i, node (σ i) = -node i) ∧
      (∀ i, weight (jacobiMomentGeneral a ha) node (σ i) =
        weight (jacobiMomentGeneral a ha) node i) := by
  have hpos : ∀ u : ℝ[X], u ≠ 0 → 0 < jacobiMomentGeneral a ha (u^2) := by
    intro u hu
    apply jacobiMomentGeneral_pos a ha _ (pow_ne_zero _ hu)
    intro x hx
    simp only [Polynomial.eval_pow]
    exact sq_nonneg _
  exact exists_orthogonal_reflection (jacobiMomentGeneral a ha) hn hpos
    (jacobiMomentGeneral_reflect a ha) node hinj
    (nodal_orthogonal_of_exact (jacobiMomentGeneral a ha) hn node hexact)

/-- Symmetry added to the actual constructed rule, with no extra symmetry
assumption on its nodes or weights. -/
theorem exists_symmetric_gaussJacobi_rule_general (a : ℝ) (ha : -1 < a)
    (n : ℕ) (hn : 0 < n) :
    ∃ (node : Fin n → ℝ) (σ : Fin n ≃ Fin n),
      Function.Injective node ∧
      (∀ i, node i ∈ Set.Ioo (-1 : ℝ) 1) ∧
      (∀ i, 0 < weight (jacobiMomentGeneral a ha) node i) ∧
      (∑ i, weight (jacobiMomentGeneral a ha) node i = 1) ∧
      (∀ p : ℝ[X], p.natDegree < 2*n → jacobiMomentGeneral a ha p =
        ∑ i, weight (jacobiMomentGeneral a ha) node i * p.eval (node i)) ∧
      Function.Involutive σ ∧
      (∀ i, node (σ i) = -node i) ∧
      (∀ i, weight (jacobiMomentGeneral a ha) node (σ i) =
        weight (jacobiMomentGeneral a ha) node i) := by
  obtain ⟨node, hinj, hinside, hpositive, hsum, hexact⟩ :=
    exists_gaussJacobi_rule_general a ha n hn
  obtain ⟨σ, hinvol, hnode, hweight⟩ :=
    gaussJacobi_reflection_of_exact a ha hn node hinj hexact
  exact ⟨node, σ, hinj, hinside, hpositive, hsum, hexact, hinvol, hnode, hweight⟩

/-- A reflection-fixed Jacobi node is precisely a zero node. Odd-order rules may
have this fixed node; antipodal sphere pairing must therefore use the full
sphere-node involution rather than assuming every scalar node has a partner. -/
theorem reflection_fixed_iff_node_zero {n : ℕ} (node : Fin n → ℝ)
    (hinj : Function.Injective node) (σ : Fin n ≃ Fin n)
    (hnode : ∀ i, node (σ i) = -node i) (i : Fin n) :
    σ i = i ↔ node i = 0 := by
  constructor
  · intro h
    have heq := hnode i
    rw [h] at heq
    linarith
  · intro hz
    apply hinj
    rw [hnode, hz, neg_zero]

/-- A symmetric node/weight rule cancels every odd sampled function, even if the
function is not a polynomial. -/
theorem sum_odd_of_reflection {n : ℕ} (node w : Fin n → ℝ)
    (σ : Fin n ≃ Fin n) (hnode : ∀ i, node (σ i) = -node i)
    (hw : ∀ i, w (σ i) = w i) (F : ℝ → ℝ) (hodd : ∀ x, F (-x) = -F x) :
    ∑ i, w i * F (node i) = 0 := by
  have hp := Equiv.sum_comp σ (fun i => w i * F (node i))
  simp only [hw, hnode, hodd, mul_neg, Finset.sum_neg_distrib] at hp
  linarith

/-- The actual symmetric Gauss--Jacobi rule cancels arbitrary odd sampled
functions; symmetry is derived from its exactness and actual moment functional. -/
theorem gaussJacobi_sum_odd (a : ℝ) (ha : -1 < a) {n : ℕ} (hn : 0 < n)
    (node : Fin n → ℝ) (hinj : Function.Injective node)
    (hexact : ∀ p : ℝ[X], p.natDegree < 2*n →
      jacobiMomentGeneral a ha p =
        ∑ i, weight (jacobiMomentGeneral a ha) node i * p.eval (node i))
    (F : ℝ → ℝ) (hodd : ∀ x, F (-x) = -F x) :
    ∑ i, weight (jacobiMomentGeneral a ha) node i * F (node i) = 0 := by
  obtain ⟨σ, _, hnode, hw⟩ := gaussJacobi_reflection_of_exact a ha hn node hinj hexact
  exact sum_odd_of_reflection node _ σ hnode hw F hodd

end QIRadon.Gaussian

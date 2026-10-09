import QIRadon.Activations.GELU
import QIRadon.Architecture.Basic

noncomputable section
open scoped BigOperators
namespace QIRadon.Activations

/-- Any affine function in any input dimension is exactly a two-neuron GELU MLP. -/
theorem gelu_affine_network {d : ℕ} (a : Fin d → ℝ) (b : ℝ) (x : Fin d → ℝ) :
    Architecture.network gelu
      (fun j : Fin 2 => if j = 0 then a else -a)
      (fun _ => 0) (fun j : Fin 2 => if j = 0 then 1 else -1) b x =
      b + ∑ k, a k * x k := by
  have h10 : (1 : Fin 2) ≠ 0 := by decide
  simp only [Architecture.network, Fin.sum_univ_two, ite_true, h10,
    ite_false, Pi.neg_apply, neg_mul, Finset.sum_neg_distrib, add_zero, one_mul]
  have h := gelu_sub_reflection (∑ k, a k * x k)
  linarith

/-- Integration polynomials of order two are exactly affine. -/
theorem anchorPolynomial_two_affine (a : ℕ → ℝ) (s : ℝ) :
    Kernel.anchorPolynomial 2 a s = a 0 + a 1 * s := by
  simp [Kernel.anchorPolynomial, Finset.sum_range_succ]

/-- All directional affine anchors combine into one affine function.
Therefore the entire Radon construction needs two extra GELU neurons total,
not two additional neurons per direction. -/
theorem directional_affine_anchors {d : ℕ} {ι : Type*} [Fintype ι]
    (α b a : ι → ℝ) (θ : ι → Fin d → ℝ) (x : Fin d → ℝ) :
    (∑ i, α i * (b i + a i * (∑ k, θ i k * x k))) =
      Architecture.network gelu
        (fun j : Fin 2 => if j = 0 then (fun k => ∑ i, α i * a i * θ i k)
          else -(fun k => ∑ i, α i * a i * θ i k))
        (fun _ => 0) (fun j : Fin 2 => if j = 0 then 1 else -1)
        (∑ i, α i * b i) x := by
  rw [gelu_affine_network]
  simp_rw [mul_add, Finset.sum_add_distrib]
  congr 1
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  simp_rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- Exact hidden-unit accounting after the global affine anchor is merged. -/
theorem gelu_total_width {ι : Type*} [Fintype ι] :
    Fintype.card (ι ⊕ Fin 2) = Fintype.card ι + 2 := by simp

end QIRadon.Activations

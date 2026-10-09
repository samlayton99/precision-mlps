import Mathlib

/-!
# The actual one-hidden-layer evaluator

This module records how the per-spoke construction becomes one ordinary MLP.
Intermediate profile polynomials do not appear in this evaluator. If an anchor
uses extra activation units, those units are explicitly included in its width.
-/

namespace QIRadon.Architecture

open scoped BigOperators

/-- One hidden layer with scalar output. The input space is explicitly ℝ^d. -/
def network {d : ℕ} {ι : Type*} [Fintype ι]
    (φ : ℝ → ℝ) (a : ι → Fin d → ℝ) (b v : ι → ℝ)
    (bias : ℝ) (x : Fin d → ℝ) : ℝ :=
  bias + ∑ i, v i * φ ((∑ k, a i k * x k) + b i)

/-- Each direction can have its own finite row of centers. -/
def spokeNetwork {d : ℕ} {M : Type*} [Fintype M] (J : M → Type*)
    [∀ m, Fintype (J m)] (φ : ℝ → ℝ) (θ : M → Fin d → ℝ)
    (α γ : M → ℝ) (c v : (m : M) → J m → ℝ)
    (bias : ℝ) (x : Fin d → ℝ) : ℝ :=
  bias + ∑ m, α m * ∑ j, v m j * φ (γ m * ((∑ k, θ m k * x k) - c m j))

/-- The construction is literally a shallow MLP after flattening the spoke and
center indices. This is an identity, not an approximation or a compilation fit. -/
theorem spokeNetwork_eq_network {d : ℕ} {M : Type*} [Fintype M]
    (J : M → Type*) [∀ m, Fintype (J m)]
    (φ : ℝ → ℝ) (θ : M → Fin d → ℝ)
    (α γ : M → ℝ) (c v : (m : M) → J m → ℝ)
    (bias : ℝ) (x : Fin d → ℝ) :
    spokeNetwork J φ θ α γ c v bias x =
      network φ (fun i : Sigma J => fun k => γ i.1 * θ i.1 k)
        (fun i => -γ i.1 * c i.1 i.2) (fun i => α i.1 * v i.1 i.2) bias x := by
  unfold spokeNetwork network
  rw [Fintype.sum_sigma]
  congr 1
  apply Finset.sum_congr rfl
  intro m _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  have harg : (∑ k, γ m * θ m k * x k) + (-γ m * c m j) =
      γ m * ((∑ k, θ m k * x k) - c m j) := by
    simp only [mul_assoc, ← Finset.mul_sum]
    ring
  rw [harg]
  ring

/-- Exact width with heterogeneous rows. -/
theorem spoke_width (M : Type*) [Fintype M] (J : M → Type*) [∀ m, Fintype (J m)] :
    Fintype.card (Sigma J) = ∑ m, Fintype.card (J m) := Fintype.card_sigma

/-- Combining a grid network and an anchor network still gives one hidden layer. -/
theorem network_sum {d : ℕ} {ι κ : Type*} [Fintype ι] [Fintype κ]
    (φ : ℝ → ℝ) (a : ι → Fin d → ℝ) (b v : ι → ℝ)
    (a' : κ → Fin d → ℝ) (b' v' : κ → ℝ) (bias bias' : ℝ) (x : Fin d → ℝ) :
    network φ a b v bias x + network φ a' b' v' bias' x =
      network φ (Sum.elim a a') (Sum.elim b b') (Sum.elim v v') (bias + bias') x := by
  simp only [network, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr]
  ring

/-- Anchor units are charged explicitly, never hidden in an external evaluator. -/
theorem total_width {ι κ : Type*} [Fintype ι] [Fintype κ] :
    Fintype.card (ι ⊕ κ) = Fintype.card ι + Fintype.card κ := Fintype.card_sum

/-- Number of scalar parameters of the ordinary dense scalar-output network:
d incoming weights, one hidden bias, and one readout per neuron, plus output bias. -/
def parameterCount (d W : ℕ) : ℕ := (d + 2) * W + 1

/-- Replacing the formal integration polynomial by its explicit anchor network
adds exactly its certified error; no exact representation is silently assumed. -/
theorem anchor_error {raw p anchor target Egrid Eanchor : ℝ}
    (hg : |raw + p - target| ≤ Egrid) (ha : |anchor - p| ≤ Eanchor) :
    |raw + anchor - target| ≤ Egrid + Eanchor := by
  have hid : raw + anchor - target = (raw + p - target) + (anchor - p) := by ring
  rw [hid]
  exact (abs_add_le _ _).trans (add_le_add hg ha)

end QIRadon.Architecture

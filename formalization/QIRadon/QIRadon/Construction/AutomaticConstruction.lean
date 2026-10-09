import QIRadon.Construction.KnownFunctionConstruction
import QIRadon.Kernel.KernelParameters
import QIRadon.Radon.AutomaticSource

/-! A complete exact-arithmetic ordinary-MLP corollary with every construction
parameter prescribed. The caller supplies the known real function, its bounded
local analytic extension, and epsilon. No profile, density, quadrature-error,
harmonic-norm, or geometry-search oracle appears among the inputs. -/

noncomputable section
open scoped Real Topology SchwartzMap
namespace QIRadon.AutomaticConstruction
open QIRadon.Radon QIRadon.Gaussian QIRadon.Activations QIRadon.KernelParameters
open QIRadon.KnownFunctionConstruction QIRadon.LocalProfileRate

def angularConstant {n : ℕ} (src : LocalAnalyticSource n) : ℝ :=
  angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src))

def scalarConstant {n : ℕ} (src : LocalAnalyticSource n) : ℝ :=
  localErrorConstant geluSpectral geluSpatial src.profileBound (scalarEllipse src) 2 2

def directionOrder {n : ℕ} (src : LocalAnalyticSource n) (ε : ℝ) : ℕ :=
  localAccuracyOrder (angularConstant src) (angularRate src) (ε/2)

def rowOrder {n : ℕ} (src : LocalAnalyticSource n) (ε : ℝ) : ℕ :=
  localAccuracyOrder (scalarConstant src) (localRate (scalarEllipse src) 2) (ε/4)

/-- Includes the exact direction grid, both halos, and two ordinary GELU anchor units. -/
abbrev NeuronIndex {n : ℕ} (src : LocalAnalyticSource n) (ε : ℝ) :=
  (SphereHalfIndex (directionOrder src ε) (n+1) ×
    localGrid (rowOrder src ε) 2 (oversampling geluSpectral) (haloScale geluSpectral geluSpatial) 2 2) ⊕ Fin 2

theorem prescribed_gelu_accuracy {n : ℕ} (src : LocalAnalyticSource n) {ε : ℝ} (hε : 0 < ε) :
    ∃ (w : NeuronIndex src ε → Fin (n+2) → ℝ)
      (b v : NeuronIndex src ε → ℝ) (bias : ℝ),
      (∀ x : EuclideanSpace ℝ (Fin (n+2)), ‖x‖ ≤ 1 →
        |Architecture.network gelu w b v bias x-src.target x| ≤ ε) ∧
      Architecture.network gelu w b v bias (0 : Fin (n+2) → ℝ)=src.target 0 := by
  obtain ⟨w,b,v,bias,he,hzero,_hw⟩ := gelu_accuracy_and_width src hε (by norm_num : (1:ℝ)<2)
    (oversampling_ge_two geluSpectral) (oversampling_separation geluSpectral)
    (haloScale_pos geluSpectral geluSpatial).le (haloScale_suppression geluSpectral geluSpatial).le
  exact ⟨w,b,v,bias,he,hzero⟩

theorem prescribed_gelu_width {n : ℕ} (src : LocalAnalyticSource n) :
    (fun ε : ℝ => (Fintype.card (NeuronIndex src ε):ℝ)) =O[𝓝[>] (0:ℝ)]
      (fun ε : ℝ => |Real.log ε|^((n+2:ℕ):ℝ)) := by
  exact gelu_total_width_bigO src (by norm_num : (1:ℝ)<2)
    (lt_of_lt_of_le (by norm_num : (0:ℝ)<2) (oversampling_ge_two geluSpectral))
    (haloScale_pos geluSpectral geluSpatial).le

/-- The original-target theorem with automatic boundary density, margins,
quadrature orders, kernel oversampling and halos. Its target assumptions are
exactly bounded local analyticity and real agreement on the specified ball. -/
theorem known_target_gelu {n : ℕ} (R σ B : ℝ)
    (hR : 1 < R) (hσ : 0 < σ) (hB : 0 ≤ B)
    (f : EuclideanSpace ℝ (Fin (n+2)) → ℝ) (F : EuclideanSpace ℂ (Fin (n+2)) → ℂ)
    (hF : AnalyticOnNhd ℂ F (realBallTube ComplexRotation.realEmbedding R σ))
    (hagree : ∀ x, ‖x‖ ≤ R → F (ComplexRotation.realEmbedding x)=(f x:ℂ))
    (hbound : ∀ (z : EuclideanSpace ℂ (Fin (n+2))),
      z ∈ realBallTube ComplexRotation.realEmbedding R σ → ‖F z‖ ≤ B) :
    let src : LocalAnalyticSource n := by
      -- Elaborate the constructor against its explicit source type before its
      -- proof fields can interact with the dependent neuron-index instances.
      apply LocalAnalyticSource.ofKnownTarget (n := n) R σ B hR hσ hB f F hF hagree
      intro z hz
      exact hbound z hz
    ((∀ ε : ℝ, 0 < ε →
      ∃ (w : NeuronIndex src ε → Fin (n+2) → ℝ)
        (b v : NeuronIndex src ε → ℝ) (bias : ℝ),
        (∀ x : EuclideanSpace ℝ (Fin (n+2)), ‖x‖ ≤ 1 →
          |Architecture.network gelu w b v bias x-f x| ≤ ε) ∧
        Architecture.network gelu w b v bias (0 : Fin (n+2) → ℝ)=f 0) ∧
      (fun ε : ℝ => (Fintype.card (NeuronIndex src ε):ℝ)) =O[𝓝[>] (0:ℝ)]
        (fun ε : ℝ => |Real.log ε|^((n+2:ℕ):ℝ))) := by
  dsimp only
  exact ⟨fun _ε hε => prescribed_gelu_accuracy _ hε,prescribed_gelu_width _⟩

end QIRadon.AutomaticConstruction

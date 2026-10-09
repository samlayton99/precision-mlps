import QIRadon.Construction.KnownFunctionConstruction
import QIRadon.Kernel.KernelParameters
import QIRadon.Activations.Tanh
import QIRadon.Radon.AutomaticSource

/-! The explicit, verified tanh corollary of the general-kernel construction.
All geometry constants are prescribed. With μ=ν=1, κ=2, hence the actual
halo-inclusive width is O(|log epsilon|^(d+1)). This is not the separate,
unproved improved rational-boundary-correction theorem. -/

noncomputable section
open scoped Real Topology
namespace QIRadon.TanhCorollary
open QIRadon.Radon QIRadon.Gaussian QIRadon.Activations QIRadon.KernelParameters
open QIRadon.KnownFunctionConstruction QIRadon.LocalProfileRate

def angularConstant {n : ℕ} (src : LocalAnalyticSource n) : ℝ :=
  angularNorm src*Real.sqrt (SharpAngular.traceConstant (n+1) (angularRate src))

def scalarConstant {n : ℕ} (src : LocalAnalyticSource n) : ℝ :=
  localErrorConstant tanhSchwartzSpectral tanhSpatial src.profileBound (scalarEllipse src) 2 1

def directionOrder {n : ℕ} (src : LocalAnalyticSource n) (ε : ℝ) : ℕ :=
  localAccuracyOrder (angularConstant src) (angularRate src) (ε/2)

def rowOrder {n : ℕ} (src : LocalAnalyticSource n) (ε : ℝ) : ℕ :=
  localAccuracyOrder (scalarConstant src) (localRate (scalarEllipse src) 2) (ε/4)

abbrev NeuronIndex {n : ℕ} (src : LocalAnalyticSource n) (ε : ℝ) :=
  SphereHalfIndex (directionOrder src ε) (n+1) ×
    localGrid (rowOrder src ε) 2 (oversampling tanhSchwartzSpectral)
      (haloScale tanhSchwartzSpectral tanhSpatial) 1 1

/-- One fixed ordinary tanh MLP uniformly approximates the original source
target and matches its origin value exactly, with no additional anchor units. -/
theorem accuracy {n : ℕ} (src : LocalAnalyticSource n) {ε : ℝ} (hε : 0 < ε) :
    ∃ (w : NeuronIndex src ε → Fin (n+2) → ℝ)
      (b v : NeuronIndex src ε → ℝ) (bias : ℝ),
      (∀ x : EuclideanSpace ℝ (Fin (n+2)), ‖x‖ ≤ 1 →
        |Architecture.network Real.tanh w b v bias x-src.target x| ≤ ε) ∧
      Architecture.network Real.tanh w b v bias (0 : Fin (n+2) → ℝ)=src.target 0 := by
  obtain ⟨w,b,v,bias,he,hzero,_hw⟩ := firstOrder_mlp_accuracy_and_width src hε
    (r := 1) (by norm_num) (tanh_contDiff 1) tanhKernel (m := 2) (by norm_num)
    tanh_derivative_kernel tanhSchwartzSpectral tanhSpatial (by norm_num : (1:ℝ)<2)
    (oversampling_ge_two tanhSchwartzSpectral) (oversampling_separation tanhSchwartzSpectral)
    (haloScale_pos tanhSchwartzSpectral tanhSpatial).le
    (haloScale_suppression tanhSchwartzSpectral tanhSpatial).le
  exact ⟨w,b,v,bias,he,hzero⟩

/-- Actual finite direction×halo-grid width: d=n+2 and exponent d+1. -/
theorem width_bigO {n : ℕ} (src : LocalAnalyticSource n) :
    (fun ε : ℝ => (Fintype.card (NeuronIndex src ε):ℝ)) =O[𝓝[>] (0:ℝ)]
      (fun ε : ℝ => |Real.log ε|^((n+3:ℕ):ℝ)) := by
  have hh := general_grid_width_bigO src tanhSchwartzSpectral tanhSpatial 1
    (by norm_num : (1:ℝ)<2)
    (lt_of_lt_of_le (by norm_num : (0:ℝ)<2) (oversampling_ge_two tanhSchwartzSpectral))
    (haloScale_pos tanhSchwartzSpectral tanhSpatial).le
  have hκ : max (1:ℝ) (1/1+1/1)=2 := by norm_num
  have hexp : ((n+1:ℕ):ℝ)+2=((n+3:ℕ):ℝ) := by push_cast; ring
  rw [hκ,hexp] at hh
  exact hh

end QIRadon.TanhCorollary

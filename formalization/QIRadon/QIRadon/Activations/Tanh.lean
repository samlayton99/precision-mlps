import QIRadon.Activations.TanhEnvelope
import QIRadon.Activations.TanhSchwartz

/-! Actual tanh activation certificate for the general kernel construction.
Its uncorrected finite-halo envelope exponents are μ=ν=1, hence κ=2.
The separate rational corrected-halo κ=1 theorem is not claimed here. -/
noncomputable section
open MeasureTheory FourierTransform
open scoped SchwartzMap
namespace QIRadon.Activations

/-- Actual μ=1 spatial envelope for the bundled Schwartz derivative kernel. -/
def tanhSpatial : QIRadon.KernelRates.SpatialEnvelope tanhKernel 1 where
  C := 2
  b := 2
  C_pos := by norm_num
  b_pos := by norm_num
  exponent_pos := by norm_num
  upper x := by
    simpa only [tanhKernel_apply, tanhKernelValue, Real.rpow_one, neg_mul] using tanhKernelValue_spatial_bound x

/-- The actual Schwartz-kernel spectrum has the proved ν=1 two-sided envelope. -/
def tanhSchwartzSpectral : QIRadon.KernelRates.SpectralEnvelope
    (𝓕 (fun x : ℝ => (tanhKernel x:ℂ))) 1 := by
  have heq : (fun x : ℝ => (tanhKernel x:ℂ))=(fun x : ℝ => (tanhKernelValue x:ℂ)) := by
    funext x
    rfl
  rw [heq]
  exact tanhSpectral

/-- The normalization used by the general derivative-kernel construction is
r=1,m=2: tanh' is twice the unit-mass kernel. -/
theorem tanh_derivative_kernel (x : ℝ) : iteratedDeriv 1 Real.tanh x = 2*tanhKernel x :=
  tanh_first_derivative x

end QIRadon.Activations

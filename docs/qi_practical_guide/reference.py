"""Small, from-scratch, sample-fitted QI ridge MLP (NumPy + SciPy only).

This is a transparent dense baseline, not the no-fit contour construction.
The returned model is an ordinary one-hidden-layer MLP in physical units.
N counts INTERVALS per direction; halo counts extra centers PER SIDE.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq
from scipy.special import roots_jacobi
from docs.qi_practical_guide.activations import Activation, get_activation, starting_lambda


def basic_tanh_lambda(tolerance=2.0**-52):
    """Root of (pi^2/lambda)/sinh(pi^2/lambda)=tolerance.

    A bandwidth prior, not a bound on the final finite fitted network's error.
    """
    if not 0 < tolerance < 1:
        raise ValueError('tolerance must lie in (0, 1)')
    def log_score(lam):
        a = math.pi**2 / lam
        if a < .01:
            return -a*a/6 + a**4/180 - a**6/2835
        return math.log(2*a) - a - math.log(-math.expm1(-2*a))
    lo, hi = 0.001, 100.0
    while log_score(hi) < math.log(tolerance):
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if log_score(mid) < math.log(tolerance):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _positive_integer(value, name):
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a positive integer') from error
    if integer != value or integer <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return integer


def directions_2d(count, phase=0.0):
    """Equal directions modulo sign, covering [phase, phase+pi)."""
    count = _positive_integer(count, 'count')
    if not np.isfinite(phase):
        raise ValueError('phase must be finite')
    theta = phase + np.arange(count) * np.pi / count
    return np.column_stack((np.cos(theta), np.sin(theta)))


def default_halo(intervals):
    """Recommended initial extra-center count PER SIDE, not a certificate."""
    intervals = _positive_integer(intervals, 'intervals')
    return max(20, math.ceil(math.sqrt(intervals)))


def sphere_directions(dimension, order):
    """Positive folded Gauss-Jacobi product rule, probability weights.

    Returns order**(dimension-1) unit directions. Folding is for an even
    reconstruction integrand (the profile and its projected argument together),
    not for arbitrary odd sphere functions. Sample-only fitting may use these
    directions without interpreting their weights as observation weights.
    """
    dimension = _positive_integer(dimension, 'dimension')
    order = _positive_integer(order, 'order')
    if dimension == 1:
        return np.ones((1,1)), np.ones(1)
    directions = directions_2d(order)
    weights = np.full(order, 1/order)
    for d in range(3, dimension+1):
        t, qweight = roots_jacobi(order, (d-3)/2, (d-3)/2)
        qweight /= qweight.sum()
        planar = np.sqrt(1-t[:,None,None]**2)*directions[None]
        last = np.broadcast_to(t[:,None,None], (order,len(directions),1))
        directions = np.concatenate((planar,last),axis=2).reshape(-1,d)
        weights = (qweight[:,None]*weights).ravel()
    return directions,weights


@dataclass
class MLP:
    weight: np.ndarray       # (width, input_dimension)
    hidden_bias: np.ndarray  # (width,)
    readout: np.ndarray      # (width, output_dimension)
    output_bias: np.ndarray  # (output_dimension,)
    scalar_output: bool = True
    activation: str | Activation = 'tanh'

    def _points(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1 and self.weight.shape[1] == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[1] != self.weight.shape[1]:
            raise ValueError('points must have shape (samples, input_dimension)')
        return x

    def __call__(self, x, chunk=512):
        chunk = _positive_integer(chunk, 'chunk')
        x = self._points(x)
        spec = get_activation(self.activation)
        y = np.empty((len(x), self.readout.shape[1]))
        for start in range(0, len(x), chunk):
            z = x[start:start+chunk] @ self.weight.T + self.hidden_bias
            y[start:start+chunk] = spec.value(z) @ self.readout + self.output_bias
        return y[:, 0] if self.scalar_output else y

    def gradient(self, x, chunk=256):
        chunk = _positive_integer(chunk, 'chunk')
        x = self._points(x)
        spec = get_activation(self.activation)
        out = np.empty((len(x), self.readout.shape[1], self.weight.shape[1]))
        for start in range(0, len(x), chunk):
            z = x[start:start+chunk] @ self.weight.T + self.hidden_bias
            out[start:start+chunk] = np.einsum('nw,wo,wd->nod', spec.first(z), self.readout, self.weight)
        return out[:, 0] if self.scalar_output else out

    def laplacian(self, x, chunk=256):
        chunk = _positive_integer(chunk, 'chunk')
        x = self._points(x)
        spec = get_activation(self.activation)
        out = np.empty((len(x), self.readout.shape[1]))
        norm2 = np.sum(self.weight**2, axis=1)
        for start in range(0, len(x), chunk):
            z = x[start:start+chunk] @ self.weight.T + self.hidden_bias
            out[start:start+chunk] = (spec.second(z)*norm2) @ self.readout
        return out[:, 0] if self.scalar_output else out

    def save(self, path):
        spec = get_activation(self.activation)
        if get_activation(spec.name) is not spec:
            raise ValueError('save requires a registered activation; preserve a custom callable separately')
        np.savez(path, weight=self.weight, hidden_bias=self.hidden_bias,
                 readout=self.readout, output_bias=self.output_bias,
                 scalar_output=np.array(self.scalar_output),activation=np.array(spec.name))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data['weight'], data['hidden_bias'], data['readout'],
                       data['output_bias'], bool(data['scalar_output']),
                       str(data['activation']) if 'activation' in data.files else 'tanh')


def fit_qi(x, y, *, directions, intervals=128, halo=None, lam=None, activation='tanh',
           origin=None, input_scale=None, projection_radius=1.0,
           rcond=1e-14, solver='gelsd', sample_weights=None, max_design_gib=2.0):
    """Fit an explicit QI dictionary by rank-revealing SVD least squares.

    Normalize physical coordinates as u=(x-origin)/input_scale. The caller
    specifies the domain: projection_radius is the half-range of each v.u,
    centered at zero. For a unit ball use 1; for a unit box use sum(abs(v)).
    Neither a training-set min/max nor a sample standard deviation certifies
    coverage of a desired domain. directions are unit vectors in u coordinates.
    sample_weights are positive least-squares weights, NOT angular quadrature.
    Fits are computed in normalized coordinates, then folded into physical
    MLP rows. Large offsets/tiny scales can lose accuracy in ordinary physical
    evaluation; diagnostics record the resulting change on fitting points.
    max_design_gib guards the design allocation only; SVD needs more memory.
    Default halo is max(20,ceil(sqrt(intervals))) per side. Default lambda is
    activation-specific, calibrated to tanh .25. GELU/SiLU include affine
    modes and emit them as exact activation pairs (two extra neurons per input
    coordinate), so the returned network has no external polynomial skip.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype=np.float64)
    scalar = y.ndim == 1
    if scalar:
        y = y[:, None]
    v = np.asarray(directions, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(y) != len(x) or v.ndim != 2 or v.shape[1] != x.shape[1]:
        raise ValueError('incompatible x, y, directions shapes')
    if len(x) == 0 or len(v) == 0 or not all(np.isfinite(a).all() for a in (x,y,v)):
        raise ValueError('nonempty finite inputs required')
    if not np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-12, rtol=1e-12):
        raise ValueError('directions must have unit norm in normalized coordinates')
    spec = get_activation(activation)
    if spec.order not in (0,1,2):
        raise ValueError('this reference implements anchor orders 0, 1 and 2')
    if spec.order==2 and (spec.affine_gain is None or not np.isfinite(spec.affine_gain) or spec.affine_gain==0):
        raise ValueError('order-two export needs a supplied exact affine-pair identity')
    if halo is None:
        halo = default_halo(intervals)
    if lam is None:
        lam = starting_lambda(activation)
    if intervals < 2 or int(intervals) != intervals or halo < 0 or int(halo) != halo or not np.isfinite(lam) or lam <= 0:
        raise ValueError('invalid intervals, halo or lambda')
    if not 0 < rcond < 1:
        raise ValueError('use an explicit singular-value cutoff in (0,1)')
    if solver not in ('gelsd','gelss','gelsy'):
        raise ValueError('solver must be a rank-revealing gelsd, gelss or gelsy driver')
    d = x.shape[1]
    origin = np.zeros(d) if origin is None else np.broadcast_to(origin, (d,)).copy()
    scale = np.ones(d) if input_scale is None else np.broadcast_to(input_scale, (d,)).copy()
    radius = np.broadcast_to(projection_radius, (len(v),)).copy()
    if not np.isfinite(origin).all() or not np.isfinite(scale).all() or not np.isfinite(radius).all() or np.any(scale <= 0) or np.any(radius <= 0):
        raise ValueError('finite origin and positive scales/radii required')
    n = int(intervals)
    r = int(halo)
    width = len(v)*(n+2*r+1)
    affine_columns = d if spec.order==2 else 0
    design_gib = len(x)*(width+affine_columns+1)*8/2**30
    if not np.isfinite(max_design_gib) or max_design_gib <= 0:
        raise ValueError('max_design_gib must be finite and positive')
    if design_gib > max_design_gib:
        raise MemoryError(f'design alone needs {design_gib:.3g} GiB; use smaller/shared-profile fits')
    h = 2*radius/n
    gamma = lam/h
    centers = -radius[:,None] + h[:,None]*np.arange(-r,n+r+1)
    normalized_w = np.repeat(gamma[:,None]*v, n+2*r+1, axis=0)
    normalized_b = (-gamma[:,None]*centers).ravel()
    w = normalized_w/scale
    b = normalized_b - w @ origin
    normalized_x = (x-origin)/scale
    if not all(np.isfinite(a).all() for a in (w,b,normalized_x)):
        raise ValueError('coordinate transform exceeds float64 range')
    design = np.empty((len(x),width+affine_columns+1),dtype=np.float64)
    for start in range(0,len(x),512):
        design[start:start+512,:width] = spec.value(normalized_x[start:start+512] @ normalized_w.T + normalized_b)
    if affine_columns:
        design[:,width:width+d] = normalized_x
    design[:,-1] = 1
    # Center/scale targets only; do not rescale dictionary columns. Centering
    # can change the selected coefficients in a rank-deficient augmented fit.
    # Scale before taking a mean to avoid overflow for large finite constants.
    amplitude = np.max(np.abs(y),axis=0)
    amplitude[amplitude == 0] = 1
    scaled_y = y/amplitude
    offset = scaled_y.mean(axis=0)
    rhs = scaled_y-offset
    if sample_weights is not None:
        sw = np.asarray(sample_weights,dtype=np.float64)
        if sw.shape != (len(x),) or not np.isfinite(sw).all() or np.any(sw<=0):
            raise ValueError('sample_weights must be finite positive and match samples')
        scaled_weights = sw/sw.max()
        factor = np.sqrt(scaled_weights/scaled_weights.mean())[:,None]
        design *= factor
        rhs = rhs*factor
    coef, _, rank, singular = lstsq(design,rhs,cond=rcond,lapack_driver=solver)
    readout = coef[:width]*amplitude
    output_bias = (coef[-1]+offset)*amplitude
    if affine_columns:
        affine_readout = coef[width:width+d]*amplitude/spec.affine_gain
        # psi(z)-psi(-z)=z for exact GELU and beta-one SiLU.
        anchor_w = np.vstack((np.eye(d),-np.eye(d)))
        normalized_w = np.vstack((normalized_w,anchor_w))
        normalized_b = np.r_[normalized_b,np.zeros(2*d)]
        anchor_physical = anchor_w/scale
        w = np.vstack((w,anchor_physical))
        b = np.r_[b,-anchor_physical@origin]
        readout = np.vstack((readout,affine_readout,-affine_readout))
    if not np.isfinite(readout).all() or not np.isfinite(output_bias).all():
        raise ValueError('computed output parameters exceed float64 range')
    model = MLP(w,b,readout,output_bias,scalar,spec)
    normalized_model = MLP(normalized_w,normalized_b,readout,output_bias,scalar,spec)
    physical_prediction = np.asarray(model(x)).reshape(y.shape)
    normalized_prediction = np.asarray(normalized_model(normalized_x)).reshape(y.shape)
    total_scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    denominator = max(float(np.linalg.norm(y/total_scale)),np.finfo(float).tiny)
    info = dict(width=len(w),geometry_width=width,anchor_neurons=2*affine_columns,
                activation=spec.name,intervals=n,halo_per_side=r,directions=len(v),
                lambda_value=lam,spacing=h.tolist(),gamma=gamma.tolist(),
                rank=int(rank),rcond=rcond,solver=solver,design_gib=design_gib,
                largest_singular_value=float(singular[0]) if singular is not None else None,
                smallest_singular_value=float(singular[-1]) if singular is not None else None,
                readout_l1=np.sum(np.abs(model.readout),axis=0).tolist(),
                training_relative_l2=float(np.linalg.norm(physical_prediction/total_scale-y/total_scale)/denominator),
                normalized_training_relative_l2=float(np.linalg.norm(normalized_prediction/total_scale-y/total_scale)/denominator),
                export_training_max_change=float(np.max(np.abs(physical_prediction-normalized_prediction))))
    return model, info

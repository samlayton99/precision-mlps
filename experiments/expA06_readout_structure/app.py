"""expA06 -- interactive readout-structure explorer.

Live Dash tool for the structure of the SVD-solved readout coefficients on the QI
geometry, toward a good readout INITIALIZATION. Every dial change re-solves
[Phi, 1][v; b] ~= y by SVD (with rcond truncation and optional ridge damping) and
redraws six panels:

  1) function + approximation        2) residual (symlog)
  3) coefficients v_k vs center c_k  4) distribution of elements (histogram)
     (+ demodulated (-1)^k overlay)
  5) coefficient spectrum |FFT(v)|   6) singular-value spectrum of [Phi, 1]
                                        (colored by alternation of the right
                                         singular vector)

Dials: N, lambda, activation (full expG01 set), target (presets + free text),
n_train, n_test, sampling, noise, rcond (10^k), ridge alpha (10^k, off by default),
halo default/override. Every panel has its own log checkbox (symlog on signed
panels, plain log otherwise).

Run:  uv run python experiments/expA06_readout_structure/run.py
Then open http://127.0.0.1:8051
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import sympy as sp
from scipy.special import erf, expit
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

from src.construction.qi_mpmath import default_halo

HALO_LAMBDA = 0.25          # halo sized at the reference lambda (expG01 convention)

# ----------------------------------------------------------------------------
# core: activation, geometry, target parsing, SVD solve
# ----------------------------------------------------------------------------
_X = sp.Symbol("x")
ACTIVATIONS = ["tanh", "gelu", "relu", "swish", "sigmoid"]
PRESETS = ["sin(2*pi*x)", "sin(8*pi*x)", "1/(1+25*x**2)",
           "sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(14*pi*x)",
           "exp(x)", "x**5-3*x**3+x", "Abs(x)**3"]
SAMPLINGS = ["equispaced", "chebyshev", "uniform"]


def apply_activation(z, name):
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "sigmoid":
        return expit(z)
    if name == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if name == "swish":
        return z * expit(z)
    return np.tanh(z)


def build_phi(x, gamma, centers, name):
    x = np.asarray(x, dtype=np.float64).ravel()
    z = gamma * (x[:, None] - centers[None, :])
    return apply_activation(z, name)


def make_fn(text):
    expr = sp.sympify(text, locals={"x": _X})
    f = sp.lambdify(_X, expr, modules=["numpy"])

    def f_vec(xv):
        xv = np.asarray(xv, dtype=np.float64)
        out = np.asarray(f(xv), dtype=np.float64)
        return np.broadcast_to(out, xv.shape).astype(np.float64)
    return f_vec


def geometry(N, lam, halo=None):
    h = 2.0 / N
    if halo is None:
        halo = default_halo(N, lambda_star=HALO_LAMBDA)
    halo = int(halo)
    centers = -1.0 + np.arange(-halo, N + halo + 1, dtype=np.float64) * h
    return centers, lam / h, halo


def sample_x(n, kind):
    if kind == "chebyshev":
        k = np.arange(1, n + 1)
        return np.cos((2 * k - 1) * np.pi / (2 * n))[::-1].copy()
    if kind == "uniform":
        return np.sort(np.random.default_rng(0).uniform(-1, 1, n))
    return np.linspace(-1.0, 1.0, n)


def next_prime(n):
    n = max(2, int(n))
    def is_p(k):
        if k < 2: return False
        if k % 2 == 0: return k == 2
        i = 3
        while i * i <= k:
            if k % i == 0: return False
            i += 2
        return True
    while not is_p(n):
        n += 1
    return n


def alternation_score(v):
    """Fraction of adjacent pairs with opposite sign (0 = smooth, 1 = perfect (-1)^k)."""
    s = np.sign(v)
    nz = s != 0
    if nz.sum() < 2:
        return 0.0
    s = s[nz]
    return float(np.mean(s[:-1] * s[1:] < 0))


def compute(N, lam, fn_text, activation, n_train, n_test, sampling,
            noise_sigma, rcond, alpha, halo=None):
    """Solve the readout by SVD and return everything the six panels need."""
    f_vec = make_fn(fn_text)
    centers, gamma, halo_n = geometry(int(N), float(lam), halo)
    W = centers.size

    x_fit = sample_x(int(n_train), sampling)
    y_fit = f_vec(x_fit)
    if noise_sigma > 0:
        y_fit = y_fit + noise_sigma * np.random.default_rng(0).standard_normal(y_fit.shape)

    Phi = build_phi(x_fit, gamma, centers, activation)
    A = np.hstack([Phi, np.ones((Phi.shape[0], 1))])

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    smax = S[0] if S.size else 1.0
    keep = S > (rcond * smax if rcond is not None else 0.0)
    if alpha and alpha > 0:
        filt = np.where(keep, S / (S ** 2 + alpha), 0.0)     # ridge damping
    else:
        filt = np.where(keep, 1.0 / np.where(S > 0, S, 1.0), 0.0)   # truncated SVD
    coeffs_modes = filt * (U.T @ y_fit)                       # per-mode amplitude
    sol = Vt.T @ coeffs_modes
    v, b = sol[:-1], sol[-1]

    # per-singular-direction alternation of the right singular vector (coeff part)
    alt_sv = np.array([alternation_score(Vt[i, :-1]) for i in range(Vt.shape[0])])

    n_te = next_prime(int(n_test))
    x_test = np.linspace(-1.0, 1.0, n_te)
    f_test = f_vec(x_test)
    f_hat = build_phi(x_test, gamma, centers, activation) @ v + b
    resid = f_hat - f_test

    # derivative-law prediction (kernel mechanism): step kernels carry f' mass,
    # ramp kernels carry f'' mass.  tanh: (h/2) f' ; sigmoid: h f' (jump 1);
    # gelu/relu/swish (ramps): (h^2/lambda) f''.
    h = 2.0 / int(N)
    eps = 1e-5
    d1 = (f_vec(centers + eps) - f_vec(centers - eps)) / (2 * eps)
    d2 = (f_vec(centers + eps) - 2 * f_vec(centers) + f_vec(centers - eps)) / eps ** 2
    if activation == "sigmoid":
        pred, pred_label = h * d1, "h * f'(c_k)"
    elif activation in ("gelu", "relu", "swish"):
        pred, pred_label = (h ** 2 / lam) * d2, "h^2/lambda * f''(c_k)"
    else:
        pred, pred_label = (h / 2.0) * d1, "h/2 * f'(c_k)"

    rel_l2 = float(np.linalg.norm(resid) / max(np.linalg.norm(f_test), 1e-300))
    return dict(
        centers=centers, v=v, b=b, W=W, halo=halo_n, gamma=gamma,
        S=S, alt_sv=alt_sv, keep=keep, mode_amp=np.abs(coeffs_modes),
        pred=pred, pred_label=pred_label,
        x_test=x_test, f_test=f_test, f_hat=f_hat, resid=resid,
        x_fit=x_fit, noise_on=noise_sigma > 0,
        norm_v=float(np.linalg.norm(v)), max_v=float(np.abs(v).max()),
        alt_rate=alternation_score(v),
        cond=float(smax / S[keep].min()) if keep.any() else np.inf,
        n_kept=int(keep.sum()), rel_l2=rel_l2, linf=float(np.abs(resid).max()),
    )


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
BLUE, RED, GREEN, ORANGE, GRID = "#1f4e8c", "#d1352b", "#2ca02c", "#e08214", "#e6e6e6"
E_MIN = -16.0


def _symlog(r):
    r = np.asarray(r, dtype=np.float64)
    a = np.abs(r)
    m = np.where(a > 0, np.log10(np.where(a > 0, a, 1.0)), E_MIN)
    return np.sign(r) * np.clip(m - E_MIN, 0.0, None)


def _symlog_axis(fig, tmax):
    S = max(2.0, tmax * 1.08)
    top = int(np.ceil(E_MIN + S))
    step = 1 if S <= 8 else 2
    tv, tt = [0.0], ["0"]
    for e in range(int(E_MIN) + step, top + 1, step):
        val = e - E_MIN
        if val > S:
            continue
        tv += [val, -val]
        tt += [f"10<sup>{e}</sup>", f"-10<sup>{e}</sup>"]
    fig.update_yaxes(tickvals=tv, ticktext=tt, range=[-S, S])


def _base(fig, title, xtitle):
    fig.update_layout(
        template="simple_white",
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=13)),
        margin=dict(l=55, r=12, t=36, b=38), height=380,
        font=dict(family="Arial, sans-serif", size=11, color="#222"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title=xtitle, showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


def fig_fit(d, ylog):
    fig = go.Figure()
    fig.add_scatter(x=d["x_test"], y=d["f_test"], mode="lines",
                    line=dict(color=BLUE, width=2.0), name="f(x)")
    fig.add_scatter(x=d["x_test"], y=d["f_hat"], mode="lines",
                    line=dict(color=RED, width=2.0), opacity=0.8, name="approx")
    fig = _base(fig, "1) target vs approximation", "x")
    if ylog:
        t = _symlog(np.concatenate([d["f_test"], d["f_hat"]]))
        fig.data = []
        fig.add_scatter(x=d["x_test"], y=_symlog(d["f_test"]), mode="lines",
                        line=dict(color=BLUE, width=2.0), name="f(x)")
        fig.add_scatter(x=d["x_test"], y=_symlog(d["f_hat"]), mode="lines",
                        line=dict(color=RED, width=2.0), opacity=0.8, name="approx")
        _symlog_axis(fig, float(np.abs(t).max()))
    return fig


def fig_residual(d, ylog):
    fig = go.Figure()
    if ylog:
        t = _symlog(d["resid"])
        fig.add_scatter(x=d["x_test"], y=t, mode="lines",
                        line=dict(color=GREEN, width=1.1), name="f - approx")
        fig.add_hline(y=0.0, line_color="#bbb", line_width=1.1)
        fig.add_hline(y=float(t.max()), line=dict(color="#7a0d0d", width=1.6, dash="dot"))
        fig.add_hline(y=float(t.min()), line=dict(color="#7a0d0d", width=1.6, dash="dot"))
        fig = _base(fig, "2) residual (symlog)", "x")
        _symlog_axis(fig, float(np.abs(t).max()))
    else:
        fig.add_scatter(x=d["x_test"], y=d["resid"], mode="lines",
                        line=dict(color=GREEN, width=1.1), name="f - approx")
        fig.add_hline(y=0.0, line_color="#bbb", line_width=1.1)
        fig = _base(fig, "2) residual", "x")
    return fig


def fig_coeffs(d, ylog):
    fig = go.Figure()
    y = _symlog(d["v"]) if ylog else d["v"]
    fig.add_scatter(x=d["centers"], y=y, mode="lines+markers",
                    line=dict(color=BLUE, width=1.0), marker=dict(size=3.5),
                    name="v_k")
    fig.add_vrect(x0=-1, x1=1, fillcolor=BLUE, opacity=0.04, line_width=0)
    fig = _base(fig, "3) readout coefficients vs center location", "center c_k")
    if ylog:
        _symlog_axis(fig, float(np.abs(_symlog(d["v"])).max()))
    return fig


def fig_hist(d, ylog):
    fig = go.Figure()
    fig.add_histogram(x=d["v"], nbinsx=81, marker=dict(color=BLUE), name="v_k")
    fig = _base(fig, "4) distribution of coefficient values", "v_k")
    fig.update_yaxes(title="count")
    if ylog:
        fig.update_yaxes(type="log")
    return fig


def fig_deriv(d, ylog):
    """Derivative-law check: solved v_k against the kernel-mechanism prediction."""
    fig = go.Figure()
    y = _symlog(d["v"]) if ylog else d["v"]
    yp = _symlog(d["pred"]) if ylog else d["pred"]
    fig.add_scatter(x=d["centers"], y=y, mode="lines+markers",
                    line=dict(color=BLUE, width=1.0), marker=dict(size=3.5),
                    name="v_k (solved)")
    fig.add_scatter(x=d["centers"], y=yp, mode="lines",
                    line=dict(color=RED, width=1.6, dash="dash"),
                    name=f"prediction {d['pred_label']}")
    fig.add_vrect(x0=-1, x1=1, fillcolor=BLUE, opacity=0.04, line_width=0)
    fig = _base(fig, "5) derivative-law check: v_k vs kernel prediction", "center c_k")
    if ylog:
        _symlog_axis(fig, float(np.abs(np.concatenate([np.atleast_1d(y), np.atleast_1d(yp)])).max()))
    else:
        # zoom to the interior signal scale so the prediction is readable
        m = np.abs(d["centers"]) <= 1.0
        lim = 3.0 * max(float(np.abs(d["pred"][m]).max()), 1e-12)
        fig.update_yaxes(range=[-lim, lim])
    return fig


def fig_svals(d, ylog):
    fig = go.Figure()
    idx = np.arange(d["S"].size)
    fig.add_scatter(
        x=idx, y=d["S"], mode="markers",
        marker=dict(size=5, color=d["alt_sv"], colorscale="RdBu_r", cmin=0, cmax=1,
                    colorbar=dict(title="alternation", thickness=10, len=0.8)),
        name="sigma_i")
    dropped = ~d["keep"]
    if dropped.any():
        fig.add_scatter(x=idx[dropped], y=d["S"][dropped], mode="markers",
                        marker=dict(size=7, symbol="x", color="#999"), name="dropped (rcond)")
    fig = _base(fig, "6) singular values of [Phi, 1]  (color = right-vector alternation)",
                "index i")
    if ylog:
        fig.update_yaxes(type="log")
    return fig


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------
_label = {"fontSize": "11px", "fontWeight": "bold", "marginTop": "6px"}
_box = {"flex": "1", "minWidth": "160px", "padding": "0 8px"}


def _logbox(pid, default_on):
    return dcc.Checklist(id=f"log-{pid}",
                         options=[{"label": " log", "value": "on"}],
                         value=(["on"] if default_on else []),
                         style={"fontSize": "10px", "textAlign": "right"})


def build_app():
    app = Dash(__name__, title="expA06 readout structure")

    controls = html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
        html.Div(style=_box, children=[
            html.Div("N (grid resolution)", style=_label),
            dcc.Slider(id="N", min=4, max=10, step=1, value=7,
                       marks={i: str(2 ** i) for i in range(4, 11)}),
            html.Div("lambda = gamma*h", style=_label),
            dcc.Slider(id="lam", min=0.05, max=1.0, step=0.05, value=0.25,
                       marks={0.05: "0.05", 0.25: "0.25", 0.5: "0.5", 1.0: "1.0"}),
        ]),
        html.Div(style=_box, children=[
            html.Div("activation", style=_label),
            dcc.Dropdown(id="act", options=[{"label": a, "value": a} for a in ACTIVATIONS],
                         value="tanh", clearable=False),
            html.Div("target f(x)", style=_label),
            dcc.Input(id="fn", value="sin(2*pi*x)", debounce=True, style={"width": "100%"}),
            dcc.Dropdown(id="fn-preset", options=[{"label": p, "value": p} for p in PRESETS],
                         placeholder="presets", clearable=True),
        ]),
        html.Div(style=_box, children=[
            html.Div("n_train", style=_label),
            dcc.Slider(id="ntr", min=8, max=12, step=0.5, value=11,
                       marks={8: "256", 10: "1024", 12: "4096"}),
            html.Div("n_test", style=_label),
            dcc.Slider(id="nte", min=9, max=13, step=1, value=12,
                       marks={9: "512", 11: "2048", 13: "8192"}),
            html.Div("sampling", style=_label),
            dcc.Dropdown(id="samp", options=[{"label": s, "value": s} for s in SAMPLINGS],
                         value="equispaced", clearable=False),
        ]),
        html.Div(style=_box, children=[
            html.Div("rcond = 10^k  (SVD truncation)", style=_label),
            dcc.Slider(id="rcond", min=-16, max=-1, step=0.5, value=-16,
                       marks={-16: "off", -8: "-8", -1: "-1"}),
            html.Div("ridge alpha = 10^k  (soft damping)", style=_label),
            dcc.Slider(id="alpha", min=-17, max=0, step=0.5, value=-17,
                       marks={-17: "off", -8: "-8", 0: "0"}),
        ]),
        html.Div(style=_box, children=[
            html.Div("noise sigma = 10^k", style=_label),
            dcc.Slider(id="noise", min=-11, max=-1, step=0.5, value=-11,
                       marks={-11: "off", -6: "-6", -1: "-1"}),
            html.Div("halo", style=_label),
            html.Div(style={"display": "flex"}, children=[
                dcc.Checklist(id="halo-default", options=[{"label": " default", "value": "on"}],
                              value=["on"], style={"fontSize": "11px"}),
                dcc.Input(id="halo", type="number", value=70, min=0, step=1, debounce=True,
                          style={"width": "70px", "marginLeft": "8px"}),
            ]),
        ]),
    ])

    def panel(pid, default_log):
        return html.Div(style={"flex": "1", "minWidth": "340px"},
                        children=[_logbox(pid, default_log), dcc.Graph(id=f"fig-{pid}")])

    app.layout = html.Div(style={"fontFamily": "Arial", "margin": "10px"}, children=[
        html.H3("expA06 -- readout structure explorer", style={"marginBottom": "2px"}),
        html.Div("SVD-solved readout on the QI geometry; every dial re-solves and redraws.",
                 style={"fontSize": "12px", "color": "#555", "marginBottom": "8px"}),
        controls,
        html.Div(id="info", style={"fontSize": "12px", "fontWeight": "bold",
                                   "margin": "8px 0", "color": "#1f4e8c"}),
        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
            panel("fit", False), panel("resid", True), panel("coef", False)]),
        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
            panel("hist", True), panel("spec", False), panel("sval", True)]),
    ])

    @app.callback(Output("fn", "value"), Input("fn-preset", "value"),
                  prevent_initial_call=True)
    def _preset(p):
        return p or "sin(2*pi*x)"

    @app.callback(
        Output("fig-fit", "figure"), Output("fig-resid", "figure"),
        Output("fig-coef", "figure"), Output("fig-hist", "figure"),
        Output("fig-spec", "figure"), Output("fig-sval", "figure"),
        Output("info", "children"),
        Input("N", "value"), Input("lam", "value"), Input("fn", "value"),
        Input("act", "value"), Input("ntr", "value"), Input("nte", "value"),
        Input("samp", "value"), Input("noise", "value"),
        Input("rcond", "value"), Input("alpha", "value"),
        Input("halo-default", "value"), Input("halo", "value"),
        Input("log-fit", "value"), Input("log-resid", "value"), Input("log-coef", "value"),
        Input("log-hist", "value"), Input("log-spec", "value"), Input("log-sval", "value"),
    )
    def update(Nexp, lam, fn, act, ntr, nte, samp, noise_k, rcond_k, alpha_k,
               halo_def, halo, lg_fit, lg_res, lg_coef, lg_hist, lg_spec, lg_sval):
        N = 2 ** int(Nexp)
        try:
            d = compute(
                N, lam, fn, act,
                n_train=int(round(2 ** ntr)), n_test=int(round(2 ** nte)),
                sampling=samp,
                noise_sigma=(10.0 ** noise_k if noise_k > -11 else 0.0),
                rcond=(10.0 ** rcond_k if rcond_k > -16 else None),
                alpha=(10.0 ** alpha_k if alpha_k > -17 else 0.0),
                halo=None if (halo_def and "on" in halo_def) else halo,
            )
        except Exception as e:
            empty = go.Figure()
            return (empty,) * 6 + (f"error: {e}",)
        info = (f"W={d['W']} (N={N} + 2*halo={d['halo']})   gamma={d['gamma']:.1f}   "
                f"cond={d['cond']:.2e}   modes kept={d['n_kept']}/{d['S'].size}   "
                f"||v||={d['norm_v']:.3e}   max|v|={d['max_v']:.3e}   "
                f"alternation rate={d['alt_rate']:.2f}   "
                f"rel L2={d['rel_l2']:.2e}   L_inf={d['linf']:.2e}")
        return (fig_fit(d, bool(lg_fit)), fig_residual(d, bool(lg_res)),
                fig_coeffs(d, bool(lg_coef)), fig_hist(d, bool(lg_hist)),
                fig_deriv(d, bool(lg_spec)), fig_svals(d, bool(lg_sval)), info)

    return app


if __name__ == "__main__":
    build_app().run(debug=False, port=8051)

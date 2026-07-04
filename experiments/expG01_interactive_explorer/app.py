"""expG01 -- interactive geometry / generalization explorer.

Live Dash tool. Set lambda, N, target function, #train/#test samples, and a
slide-able mask interval; every change re-solves the SVD min-norm readout on the
UNMASKED train points and redraws. Reuses the same halo + gamma + solver path as
the batch experiments (src/construction).

Run:  ~/.venvs/precisionMLPs/bin/python experiments/expG01_interactive_explorer/run.py
Then open http://127.0.0.1:8050
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import sympy as sp
from scipy.special import erf, expit
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, dash_table
import plotly.graph_objects as go

from src.construction.qi_mpmath import default_halo, construct_qi
from src.construction.readout import solve_readout

# Halo is sized with a fixed reference lambda (as in the batch experiments,
# e.g. expC04), NOT the slider lambda -- keeps geometry on-convention and avoids
# the halo blowing up at small lambda.
HALO_LAMBDA = 0.25

# ----------------------------------------------------------------------------
# core: parse function, build geometry, solve, evaluate
# ----------------------------------------------------------------------------
_X = sp.Symbol("x")

ACTIVATIONS = ["tanh", "gelu", "relu", "swish", "sigmoid"]
SOLVE_METHODS = [{"label": "least squares", "value": "lstsq"}, {"label": "QI", "value": "qi"}]


def apply_activation(z, name):
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "sigmoid":
        return expit(z)
    if name == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    if name == "swish":
        return z * expit(z)
    return np.tanh(z)  # default / "tanh" -- the machine-precision case


def build_phi_act(x, gamma, centers, name):
    """Phi[i,k] = act(gamma_k * (x_i - c_k)).  tanh matches src.readout.build_phi."""
    x = np.asarray(x, dtype=np.float64).ravel()
    z = gamma[None, :] * (x[:, None] - centers[None, :])
    return apply_activation(z, name)


def make_fns(text):
    """Parse a text expression in x -> (vectorized f, scalar f, scalar f') via sympy.

    The scalar f and f' feed the QI construction (which samples the target and its
    derivative on grid nodes); the vectorized f is used everywhere else.
    """
    expr = sp.sympify(text, locals={"x": _X})
    dexpr = sp.diff(expr, _X)
    f = sp.lambdify(_X, expr, modules=["numpy"])
    fp = sp.lambdify(_X, dexpr, modules=["numpy"])

    def f_vec(xv):
        xv = np.asarray(xv, dtype=np.float64)
        out = np.asarray(f(xv), dtype=np.float64)
        return np.broadcast_to(out, xv.shape).astype(np.float64)  # broadcast constants

    return f_vec, (lambda xx: float(f(float(xx)))), (lambda xx: float(fp(float(xx))))


def make_fn(text):
    return make_fns(text)[0]


def next_prime(n):
    n = max(2, int(n))

    def is_prime(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        i = 3
        while i * i <= k:
            if k % i == 0:
                return False
            i += 2
        return True

    while not is_prime(n):
        n += 1
    return n


def geometry(N, lam, halo=None):
    """Center lattice + per-center gamma. halo=None -> standard default_halo."""
    h = 2.0 / N
    if halo is None:
        halo = default_halo(N, lambda_star=HALO_LAMBDA)
    halo = int(halo)
    n_idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + n_idx.astype(np.float64) * h
    gamma = lam / h
    gamma_vec = np.full(centers.size, gamma)
    return centers, gamma_vec, halo, gamma


def _metrics(resid, f_true, idx):
    r = resid[idx]
    ft = f_true[idx]
    if r.size == 0:
        return (None, None)
    denom = np.linalg.norm(ft)
    rel_l2 = float(np.linalg.norm(r) / denom) if denom > 0 else float(np.linalg.norm(r))
    linf = float(np.abs(r).max())
    return (rel_l2, linf)


def sample_gmm(rows, seed):
    """rows: [{center, variance, count}] -> sampled training x in [-1, 1] (sorted)."""
    rng = np.random.default_rng(int(seed))
    xs = []
    for r in rows or []:
        try:
            mu, var, n = float(r["center"]), float(r["variance"]), int(r["count"])
        except (TypeError, ValueError, KeyError):
            continue
        if n > 0:
            xs.append(mu + np.sqrt(max(var, 0.0)) * rng.standard_normal(n))
    if not xs:
        return None
    return np.sort(np.concatenate(xs))   # not clipped -- samples may fall outside [-1, 1]


def build_centers_override(rows):
    """rows: [{left, right, num, lambda}] -> (centers, gamma_vec) or None.

    Each region places `num` uniform centers on [left, right]; its neurons get
    gamma = lambda / h_region (local spacing). lambda in [0, 1].
    """
    cs, gs = [], []
    for r in rows or []:
        try:
            a, b, num, lam = float(r["left"]), float(r["right"]), int(r["num"]), float(r["lambda"])
        except (TypeError, ValueError, KeyError):
            continue
        if num < 1 or b <= a or not (0.0 <= lam <= 1.0):
            continue
        pts = np.linspace(a, b, num)
        h = (b - a) / (num - 1) if num > 1 else (b - a)
        cs.append(pts)
        gs.append(np.full(num, lam / h if h > 0 else 0.0))
    if not cs:
        return None
    centers, gamma_vec = np.concatenate(cs), np.concatenate(gs)
    order = np.argsort(centers)
    return centers[order], gamma_vec[order]


def compute(N, lam, fn_text, n_train, n_test, mask_on, mlo, mhi, activation,
            halo=None, mask_centers=False, method="lstsq", noise_sigma=0.0, rcond=None,
            x_override=None, centers_override=None, mark_points=False):
    f_vec, f_scalar, fp_scalar = make_fns(fn_text)
    halo_n = int(halo) if halo is not None else default_halo(N, lambda_star=HALO_LAMBDA)

    # test grid: round #samples up to the nearest prime so it doesn't align
    # with the (evenly spaced) train grid
    n_test_prime = next_prime(int(n_test))
    x_test = np.linspace(-1.0, 1.0, n_test_prime)
    f_test = f_vec(x_test)

    if method == "qi":
        # Closed-form quasi-interpolant on the full grid (tanh only). QI needs the
        # whole grid, so the data mask / center mask / #train samples are ignored.
        qi = construct_qi(f_scalar, fp_scalar, int(N), lambda_star=float(lam),
                          halo=halo_n, precision="fp64")
        centers, gamma, v, b = qi.centers, qi.gamma, qi.a_coeffs, qi.c0
        gamma_vec = np.full(centers.size, gamma)
        f_hat = build_phi_act(x_test, gamma_vec, centers, "tanh") @ v + b
        n_fit, activation, x_fit, y_fit = centers.size, "tanh", None, None
    else:
        if centers_override is not None:              # custom piecewise-uniform centers + per-region gamma
            centers, gamma_vec = centers_override
        else:
            centers, gamma_vec, halo_n, gamma = geometry(N, lam, halo_n)
        if mask_on and mask_centers:                  # also drop neurons centered in the gap
            keep_c = (centers < mlo) | (centers > mhi)
            centers, gamma_vec = centers[keep_c], gamma_vec[keep_c]
        gamma = float(np.mean(gamma_vec)) if gamma_vec.size else 0.0   # representative bandwidth
        if x_override is not None:                     # custom (e.g. GMM-sampled) training x
            x_train = np.asarray(x_override, dtype=np.float64)
        else:
            x_train = np.linspace(-1.0, 1.0, int(n_train))
        keep = (x_train < mlo) | (x_train > mhi) if mask_on else np.ones_like(x_train, dtype=bool)
        x_fit = x_train[keep]
        y_fit = f_vec(x_fit)
        if noise_sigma > 0:                           # y-noise on the training data (fixed seed)
            y_fit = y_fit + noise_sigma * np.random.default_rng(0).standard_normal(y_fit.shape)
        Phi_fit = build_phi_act(x_fit, gamma_vec, centers, activation)
        A_fit = np.hstack([Phi_fit, np.ones((Phi_fit.shape[0], 1))])   # bias column
        sol, _ = solve_readout(A_fit, y_fit, method="svd", rcond=rcond)
        v, b = sol[:-1], sol[-1]
        f_hat = build_phi_act(x_test, gamma_vec, centers, activation) @ v + b
        n_fit = x_fit.size

    resid = f_hat - f_test
    in_mask = (x_test >= mlo) & (x_test <= mhi) if mask_on else np.zeros(x_test.size, dtype=bool)
    noise_on = noise_sigma > 0 and method != "qi"

    return dict(
        x_test=x_test, f_test=f_test, f_hat=f_hat, resid=resid,
        x_fit=x_fit, y_fit=y_fit, noise_on=noise_on,
        show_points=(mark_points and x_fit is not None),
        centers_custom=(centers_override is not None), x_custom=(x_override is not None),
        mask_on=mask_on, mlo=mlo, mhi=mhi, method=method, activation=activation,
        m_all=_metrics(resid, f_test, np.ones(x_test.size, dtype=bool)),
        m_out=_metrics(resid, f_test, ~in_mask),
        m_in=_metrics(resid, f_test, in_mask),
        W=centers.size, halo=halo_n, gamma=gamma, n_test_prime=n_test_prime,
        n_fit=n_fit,
    )


# ----------------------------------------------------------------------------
# figures + table
# ----------------------------------------------------------------------------
BLUE = "#1f4e8c"
RED = "#d1352b"
RED_DARK = "#7a0d0d"
GREEN = "#2ca02c"
GRID = "#e6e6e6"


def _add_rug(fig, xf, y_level):
    """Black baseline at y_level + red open-circle markers at the sampled x's."""
    fig.add_hline(y=y_level, line=dict(color="black", width=1))
    fig.add_scatter(x=xf, y=np.full(xf.shape, y_level), mode="markers",
                    marker=dict(color=RED, size=6, symbol="circle-open"), name="sampled x")


def _base(fig, title, ylog=False):
    fig.update_layout(
        template="simple_white",
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=14)),
        margin=dict(l=55, r=15, t=40, b=40),
        font=dict(family="Arial, sans-serif", size=12, color="#222"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=430,
    )
    fig.update_xaxes(title="x", showgrid=True, gridcolor=GRID, zeroline=False, range=[-1, 1])
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    if ylog:
        fig.update_yaxes(type="log")
    return fig


def fig_functions(d):
    fig = go.Figure()
    if d.get("noise_on"):
        # noisy training data the fit actually saw (blue), broken across the mask
        xf, yf = d["x_fit"], d["y_fit"]
        if d["mask_on"]:
            left, right = xf < d["mlo"], xf > d["mhi"]
            xf = np.concatenate([xf[left], [np.nan], xf[right]])
            yf = np.concatenate([yf[left], [np.nan], yf[right]])
        fig.add_scatter(x=xf, y=yf, mode="lines", line=dict(color=BLUE, width=1.0),
                        name="noisy data")
        fig.add_scatter(x=d["x_test"], y=d["f_test"], mode="lines",
                        line=dict(color="black", width=1.6, dash="dot"), name="f(x) true")
        fig.add_scatter(x=d["x_test"], y=d["f_hat"], mode="lines",
                        line=dict(color=RED, width=2.2), name="approx")
    else:
        fig.add_scatter(x=d["x_test"], y=d["f_test"], mode="lines",
                        line=dict(color=BLUE, width=2.2), name="f(x)")
        fig.add_scatter(x=d["x_test"], y=d["f_hat"], mode="lines",
                        line=dict(color=RED, width=2.2), opacity=0.8, name="approx")
    if d["mask_on"]:
        fig.add_vrect(x0=d["mlo"], x1=d["mhi"], fillcolor="#999", opacity=0.13, line_width=0)
    xf = d.get("x_fit")
    if d.get("show_points") and xf is not None and xf.size:
        lo, hi = float(d["f_test"].min()), float(d["f_test"].max())
        yr = lo - 0.08 * max(hi - lo, 1e-9)                     # rug just below the data
        _add_rug(fig, xf, yr)
    fig = _base(fig, "target vs approximation")
    if d.get("show_points") and xf is not None and xf.size:     # widen to show samples past [-1,1]
        fig.update_xaxes(range=[min(-1.0, float(xf.min())), max(1.0, float(xf.max()))])
    return fig


E_MIN = -16.0  # machine-eps floor exponent; maps to the centered zero line


def _symlog(r):
    """Signed symmetric-log transform: 0 at |r|<=1e-16, +/- log10 magnitude above.

    Positive residuals go up, negative go down, both log-scaled in magnitude, so
    residual=0 sits on a fixed line in the middle and the axis is mirrored.
    """
    r = np.asarray(r, dtype=np.float64)
    a = np.abs(r)
    m = np.where(a > 0, np.log10(np.where(a > 0, a, 1.0)), E_MIN)
    return np.sign(r) * np.clip(m - E_MIN, 0.0, None)


def fig_residual(d):
    t = _symlog(d["resid"])
    S = max(2.0, float(np.abs(t).max()) * 1.08)          # symmetric range -> 0 stays centered
    top_exp = int(np.ceil(E_MIN + S))
    step = 1 if S <= 8 else 2                             # denser ticks when zoomed in
    tickvals, ticktext = [0.0], ["0"]
    for e in range(int(E_MIN) + step, top_exp + 1, step):
        v = e - E_MIN
        if v > S:
            continue
        tickvals += [v, -v]
        ticktext += [f"10<sup>{e}</sup>", f"-10<sup>{e}</sup>"]
    fig = go.Figure()
    fig.add_scatter(x=d["x_test"], y=t, mode="lines",
                    line=dict(color=GREEN, width=1.1), name="f - approx")
    if d["mask_on"]:
        fig.add_vrect(x0=d["mlo"], x1=d["mhi"], fillcolor="#999", opacity=0.13, line_width=0)
    fig.add_hline(y=0.0, line_color="#bbb", line_width=1.2)
    # dark-red dotted lines at the max and min residual (the L-inf envelope)
    fig.add_hline(y=float(t.max()), line=dict(color=RED_DARK, width=2, dash="dot"))
    fig.add_hline(y=float(t.min()), line=dict(color=RED_DARK, width=2, dash="dot"))
    ymin, xf = -S, d.get("x_fit")
    if d.get("show_points") and xf is not None and xf.size:     # sampled-x rug at the bottom
        _add_rug(fig, xf, -S)
        ymin = -S - 0.10 * S
    fig = _base(fig, "residual  (f - approx),  symmetric log")
    fig.update_yaxes(tickvals=tickvals, ticktext=ticktext, range=[ymin, S], title="residual")
    if d.get("show_points") and xf is not None and xf.size:
        fig.update_xaxes(range=[min(-1.0, float(xf.min())), max(1.0, float(xf.max()))])
    return fig


def _fmt(x):
    return "—" if x is None else f"{x:.2e}"


def metrics_table(d):
    th = {"padding": "8px 14px", "borderBottom": "2px solid #bbb", "textAlign": "right",
          "fontWeight": 600, "whiteSpace": "nowrap"}
    th_l = {**th, "textAlign": "left"}
    td = {"padding": "8px 14px", "borderBottom": "1px solid #eee", "textAlign": "right",
          "fontFamily": "monospace", "whiteSpace": "nowrap"}
    td_l = {"padding": "8px 14px", "borderBottom": "1px solid #eee", "textAlign": "left",
            "whiteSpace": "nowrap"}

    def row(label, m):
        return html.Tr([html.Td(label, style=td_l),
                        html.Td(_fmt(m[0]), style=td), html.Td(_fmt(m[1]), style=td)])

    return html.Table([
        html.Thead(html.Tr([html.Th("test region", style=th_l),
                            html.Th("rel L2", style=th), html.Th("L∞", style=th)])),
        html.Tbody([
            row("entire", d["m_all"]),
            row("unmasked (data)", d["m_out"]),
            row("masked (held-out)", d["m_in"]),
        ]),
    ], style={"borderCollapse": "collapse", "width": "100%", "fontSize": "14px"})


# ----------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------
app = Dash(__name__, title="expG01 explorer")

_slider = dict(updatemode="mouseup", allow_direct_input=False)  # hide Dash 4's built-in end inputs
_num = dict(type="number", debounce=True)
_label = {"fontSize": "13px", "fontWeight": 600, "marginBottom": "4px"}
_ctl = {"display": "flex", "flexDirection": "column", "gap": "4px", "minWidth": "230px",
        "flex": "1"}
_card = {"border": "1px solid #e4e4e4", "borderRadius": "8px", "padding": "14px 16px",
         "background": "#fcfcfc", "display": "flex", "flexDirection": "column", "gap": "14px"}
_section = {"fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.06em",
            "textTransform": "uppercase", "color": "#999", "marginBottom": "2px"}
_btn = {"padding": "6px 14px", "fontSize": "13px", "border": "1px solid #ccc",
        "borderRadius": "6px", "background": "#fff", "cursor": "pointer"}
_btn_primary = {**_btn, "background": "#5b3fd1", "color": "#fff", "border": "1px solid #5b3fd1"}

PRESETS = ["sin(2*pi*x)", "sin(8*pi*x)", "1/(1+25*x**2)", "exp(x)", "x**5 - x",
           "tanh(10*x)", "sin(2*pi*x) + 0.3*sin(10*pi*x)", "Abs(x)**3"]


def _ctl_slider_num(name, slider_id, input_id, smin, smax, step, val):
    return html.Div([
        html.Div(name, style=_label),
        html.Div([
            html.Div(dcc.Slider(id=slider_id, min=smin, max=smax, step=step, value=val,
                                marks=None, **_slider),
                     style={"flex": "1"}),
            dcc.Input(id=input_id, value=val, min=smin,
                      style={"width": "78px", "flexShrink": "0"}, **_num),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
    ], style=_ctl)


app.layout = html.Div([
    html.H2("expG01 — interactive geometry / generalization explorer",
            style={"marginBottom": "2px"}),
    html.Div("Set the geometry and the data; the SVD min-norm readout is re-solved on the "
             "unmasked points on every change.",
             style={"color": "#666", "fontSize": "13px", "marginBottom": "14px"}),

    # controls -- grouped into three cards by type
    html.Div([
        html.Div([
            html.Div("geometry & data", style=_section),
            _ctl_slider_num("λ (bandwidth)", "lam-slider", "lam-input", 0.02, 1.0, 0.01, 0.25),
            _ctl_slider_num("N (grid width)", "N-slider", "N-input", 8, 512, 8, 128),
            _ctl_slider_num("# train samples", "ntr-slider", "ntr-input", 16, 4096, 16, 1024),
            _ctl_slider_num("# test samples", "nte-slider", "nte-input", 16, 8192, 16, 2048),
        ], style={**_card, "flex": "1.3", "minWidth": "320px"}),

        html.Div([
            html.Div("model", style=_section),
            html.Div([
                html.Div("solve", style=_label),
                dcc.Dropdown(id="method-input", options=SOLVE_METHODS, value="lstsq",
                             clearable=False),
            ]),
            html.Div([
                html.Div("activation", style=_label),
                dcc.Dropdown(id="act-input", options=[{"label": a, "value": a} for a in ACTIVATIONS],
                             value="tanh", clearable=False),
            ]),
            html.Div([
                html.Div("halo (ghost nodes / side)", style=_label),
                html.Div([
                    dcc.Checklist(id="halo-default", options=[{"label": " default", "value": "on"}],
                                  value=["on"], style={"marginRight": "14px"}),
                    dcc.Input(id="halo-input", type="number", value=70, min=0, step=1, debounce=True,
                              style={"width": "90px"}),
                ], style={"display": "flex", "alignItems": "center"}),
            ]),
            html.Div([
                html.Div("function  f(x)", style=_label),
                dcc.Input(id="fn-input", value="sin(2*pi*x)", debounce=True,
                          style={"width": "100%", "boxSizing": "border-box"}),
                dcc.Dropdown(id="fn-preset", options=[{"label": p, "value": p} for p in PRESETS],
                             placeholder="presets…", style={"marginTop": "6px"}),
            ]),
        ], style={**_card, "flex": "1", "minWidth": "230px"}),

        html.Div([
            html.Div("held-out & noise", style=_section),
            html.Div([
                html.Div("mask (held-out interval)", style=_label),
                html.Div([
                    dcc.Checklist(id="mask-on", options=[{"label": " on", "value": "on"}], value=[],
                                  style={"marginRight": "20px"}),
                    dcc.Checklist(id="mask-centers",
                                  options=[{"label": " drop centers too", "value": "on"}], value=[]),
                ], style={"display": "flex"}),
                html.Div(
                    dcc.RangeSlider(id="mask-range", min=-1, max=1, step=0.01, value=[-0.3, 0.3],
                                    marks={-1: "-1", 0: "0", 1: "1"},
                                    tooltip={"placement": "top", "always_visible": True}, **_slider),
                    style={"paddingTop": "34px", "paddingRight": "8px"}),
            ]),
            html.Div([
                html.Div("y-noise  σ = 10^k", style=_label),
                dcc.Checklist(id="noise-on", options=[{"label": " on", "value": "on"}], value=[]),
                html.Div(
                    dcc.Slider(id="noise-slider", min=-10, max=-1, step=0.25, value=-6,
                               marks={-10: "-10", -6: "-6", -1: "-1"},
                               tooltip={"placement": "top", "always_visible": True}, **_slider),
                    style={"paddingTop": "22px", "paddingRight": "8px"}),
            ]),
            html.Div([
                html.Div("svd solver regularization  (rcond = 10^k)", style=_label),
                html.Div(
                    dcc.Slider(id="rcond-slider", min=-13, max=-1, step=0.5, value=-13,
                               marks={-13: "-13 (off)", -7: "-7", -1: "-1"},
                               tooltip={"placement": "top", "always_visible": True}, **_slider),
                    style={"paddingTop": "42px", "paddingRight": "8px"}),
            ]),
        ], style={**_card, "flex": "1.1", "minWidth": "320px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap", "marginBottom": "16px",
              "alignItems": "flex-start"}),

    html.Hr(style={"border": "none", "borderTop": "1px solid #eee"}),

    # panels: left (functions) | middle (residual) | right (table)
    html.Div([
        html.Div(dcc.Graph(id="figL"), style={"flex": "1", "minWidth": "340px"}),
        html.Div(dcc.Graph(id="figM"), style={"flex": "1", "minWidth": "340px"}),
        html.Div([html.Div("errors over test range", style={**_label, "marginBottom": "8px"}),
                  html.Div(id="metrics")],
                 style={"flex": "0 0 330px", "paddingTop": "34px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap"}),

    html.Div(id="info-line", style={"color": "#888", "fontSize": "12px", "marginTop": "8px"}),

    # override regimes (below the graphs) -- each takes over only when enabled
    html.Div([
        html.Div([
            html.Div("x-sampling — gaussian mixture", style=_section),
            dcc.Checklist(id="gmm-enable",
                          options=[{"label": " use this (overrides # train samples)", "value": "on"}],
                          value=[], style={"marginBottom": "8px"}),
            dash_table.DataTable(
                id="gmm-table",
                columns=[{"name": "center", "id": "center", "type": "numeric"},
                         {"name": "variance", "id": "variance", "type": "numeric"},
                         {"name": "count", "id": "count", "type": "numeric"}],
                data=[{"center": -0.5, "variance": 0.02, "count": 300},
                      {"center": 0.5, "variance": 0.02, "count": 300}],
                editable=True, row_deletable=True,
                row_selectable="multi", selected_rows=[0, 1],
                style_cell={"fontFamily": "Arial", "fontSize": "13px", "padding": "4px 10px",
                            "textAlign": "center"},
                style_header={"fontWeight": 600, "backgroundColor": "#f4f4f4"}),
            html.Div([
                html.Button("+ add cluster", id="gmm-add", n_clicks=0, style=_btn),
                html.Button("sample", id="gmm-sample", n_clicks=0, style=_btn_primary),
            ], style={"display": "flex", "gap": "10px", "marginTop": "10px"}),
            dcc.Store(id="gmm-store"),
        ], style={**_card, "flex": "1", "minWidth": "340px"}),

        html.Div([
            html.Div("custom centers — piecewise-uniform, per-region λ", style=_section),
            dcc.Checklist(id="centers-enable",
                          options=[{"label": " use this (overrides grid, halo & λ)", "value": "on"}],
                          value=[], style={"marginBottom": "8px"}),
            dash_table.DataTable(
                id="centers-table",
                columns=[{"name": "left", "id": "left", "type": "numeric"},
                         {"name": "right", "id": "right", "type": "numeric"},
                         {"name": "num", "id": "num", "type": "numeric"},
                         {"name": "λ (0–1)", "id": "lambda", "type": "numeric"}],
                data=[{"left": -1.0, "right": 1.0, "num": 129, "lambda": 0.25}],
                editable=True, row_deletable=True,
                row_selectable="multi", selected_rows=[0],
                style_cell={"fontFamily": "Arial", "fontSize": "13px", "padding": "4px 10px",
                            "textAlign": "center"},
                style_header={"fontWeight": 600, "backgroundColor": "#f4f4f4"}),
            html.Button("+ add region", id="centers-add", n_clicks=0,
                        style={**_btn, "marginTop": "10px"}),
        ], style={**_card, "flex": "1", "minWidth": "340px"}),
    ], style={"display": "flex", "gap": "18px", "flexWrap": "wrap", "marginTop": "18px"}),
], style={"maxWidth": "1500px", "margin": "0 auto", "padding": "18px 24px",
          "fontFamily": "Arial, sans-serif"})


# two-way sync: slider <-> number box (per control)
def _register_sync(slider_id, input_id):
    @app.callback(Output(slider_id, "value"), Output(input_id, "value"),
                  Input(slider_id, "value"), Input(input_id, "value"))
    def _sync(s_val, i_val):
        val = s_val if ctx.triggered_id == slider_id else i_val
        return val, val


for _s, _i in [("lam-slider", "lam-input"), ("N-slider", "N-input"),
               ("ntr-slider", "ntr-input"), ("nte-slider", "nte-input")]:
    _register_sync(_s, _i)


@app.callback(Output("fn-input", "value"), Input("fn-preset", "value"), prevent_initial_call=True)
def _fill_preset(p):
    return p or "sin(2*pi*x)"


@app.callback(
    Output("halo-input", "value"), Output("halo-input", "disabled"),
    Input("halo-default", "value"), Input("N-input", "value"),
)
def _halo_default(default_val, N):
    """When 'default' is on, auto-fill the correct halo and lock the box; else free it."""
    if bool(default_val) and "on" in default_val:
        try:
            N = max(8, int(N))
        except (TypeError, ValueError):
            N = 128
        return int(default_halo(N, lambda_star=HALO_LAMBDA)), True
    return no_update, False


@app.callback(Output("gmm-table", "data"), Output("gmm-table", "selected_rows"),
              Input("gmm-add", "n_clicks"),
              State("gmm-table", "data"), State("gmm-table", "selected_rows"),
              prevent_initial_call=True)
def _gmm_add(n, data, selected):
    data = data or []
    return data + [{"center": 0.0, "variance": 0.02, "count": 200}], \
        sorted(set((selected or []) + [len(data)]))          # keep the new row checked


@app.callback(Output("centers-table", "data"), Output("centers-table", "selected_rows"),
              Input("centers-add", "n_clicks"),
              State("centers-table", "data"), State("centers-table", "selected_rows"),
              prevent_initial_call=True)
def _centers_add(n, data, selected):
    data = data or []
    return data + [{"left": -1.0, "right": 1.0, "num": 64, "lambda": 0.25}], \
        sorted(set((selected or []) + [len(data)]))


@app.callback(Output("gmm-store", "data"), Input("gmm-sample", "n_clicks"),
              State("gmm-table", "data"), State("gmm-table", "selected_rows"),
              prevent_initial_call=True)
def _gmm_sample(n, rows, selected):
    rows = rows or []
    sel = [rows[i] for i in (selected or []) if 0 <= i < len(rows)]   # only checked clusters
    x = sample_gmm(sel, seed=n)
    return None if x is None else x.tolist()


@app.callback(
    Output("figL", "figure"), Output("figM", "figure"),
    Output("metrics", "children"), Output("info-line", "children"),
    Input("lam-input", "value"), Input("N-input", "value"),
    Input("ntr-input", "value"), Input("nte-input", "value"),
    Input("fn-input", "value"), Input("act-input", "value"),
    Input("mask-on", "value"), Input("mask-range", "value"),
    Input("halo-default", "value"), Input("halo-input", "value"),
    Input("method-input", "value"), Input("mask-centers", "value"),
    Input("noise-on", "value"), Input("noise-slider", "value"),
    Input("rcond-slider", "value"),
    Input("gmm-enable", "value"), Input("gmm-store", "data"),
    Input("centers-enable", "value"), Input("centers-table", "data"),
    Input("centers-table", "selected_rows"),
)
def update(lam, N, ntr, nte, fn_text, activation, mask_val, mask_rng, halo_default, halo_val,
           method, mask_centers_val, noise_on_val, noise_k, rcond_k,
           gmm_enable, gmm_store, centers_enable, centers_table, centers_selected):
    try:
        lam = float(lam)
        N = max(8, int(N))
        ntr = max(4, int(ntr))
        nte = max(4, int(nte))
        mask_on = bool(mask_val) and "on" in mask_val
        mlo, mhi = float(min(mask_rng)), float(max(mask_rng))
        use_default = bool(halo_default) and "on" in halo_default
        halo = None if use_default else max(0, int(halo_val))
        mask_centers = bool(mask_centers_val) and "on" in mask_centers_val
        method = method or "lstsq"
        noise_sigma = (10.0 ** float(noise_k)) if (noise_on_val and "on" in noise_on_val) else 0.0
        rcond = 10.0 ** float(rcond_k)
        gmm_on = bool(gmm_enable) and "on" in gmm_enable
        x_override = np.asarray(gmm_store, dtype=np.float64) if (gmm_on and gmm_store) else None
        centers_on = bool(centers_enable) and "on" in centers_enable
        centers_rows = [centers_table[i] for i in (centers_selected or [])
                        if centers_table and 0 <= i < len(centers_table)]   # only checked regions
        centers_override = build_centers_override(centers_rows) if centers_on else None
        if method == "qi":                            # overrides are least-squares only
            x_override = centers_override = None
        d = compute(N, lam, fn_text, ntr, nte, mask_on, mlo, mhi, activation or "tanh", halo,
                    mask_centers, method, noise_sigma, rcond,
                    x_override, centers_override, mark_points=(x_override is not None))
        note = "  ·  QI: full grid (mask / #train ignored)" if method == "qi" else ""
        nnote = f"  ·  y-noise σ={noise_sigma:.0e}" if d["noise_on"] else ""
        rnote = f"  ·  rcond={rcond:.0e}" if (float(rcond_k) > -13 and method != "qi") else ""
        xnote = "  ·  GMM x-sampling" if d.get("x_custom") else ""
        geo = (f"neurons W={d['W']} (custom centers)  ·  γ~{d['gamma']:.2f}" if d.get("centers_custom")
               else f"neurons W={d['W']} (N={N}, halo={d['halo']} each side)  ·  γ={d['gamma']:.2f}")
        info = (f"solve={method}  ·  activation={d['activation']}  ·  {geo}  ·  "
                f"fit points={d['n_fit']}  ·  test samples={d['n_test_prime']} (nearest prime)"
                f"{note}{nnote}{rnote}{xnote}")
        return fig_functions(d), fig_residual(d), metrics_table(d), info
    except Exception as e:  # bad function text, etc. -- show the error, don't crash
        ef = go.Figure()
        ef.add_annotation(text=str(e), showarrow=False, font=dict(color=RED))
        ef.update_layout(template="simple_white", height=430)
        return ef, ef, html.Div(f"error: {e}", style={"color": RED, "fontSize": "13px"}), ""

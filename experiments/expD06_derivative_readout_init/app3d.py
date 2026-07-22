"""expD06 3D parameter explorer -- (center, w, readout) point cloud evolving over training.

One dot per hidden neuron in 3D: x = current effective center c_k = -b_k/w_k,
y = first-layer weight w_k, z = readout weight v_k. Dots carry a FIXED red->blue
gradient assigned from the step-0 center ordering (red = initially leftmost), so a
neuron keeps its color as it migrates. No connecting lines.

Controls: seed / width N / target function (whatever snapshot NPZs exist on disk),
play/pause + time slider + playback speed (fps), and axis scaling: global (fixed
percentile limits over the whole run), fit current frame, or manual min/max boxes.
Rotate by dragging, zoom by scrolling (plotly 3D). Camera survives frame changes.
Sidebar shows the error curves (train MSE + eval rel L2, log y) with a cursor at the
current step.

Data source: results/checkpoint_D_optimizers/expD06_derivative_readout_init/snaps/
  <seed>--<fn>--N<width>.npz     (written by run.py run_case; rerun a seed to add it)

Run:  uv run python experiments/expD06_derivative_readout_init/app3d.py   (port 8051)
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SNAP_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
            / "expD06_derivative_readout_init" / "snaps")

ACCENT = "#1f4e8c"
GRID = "#e4e7ee"
EPS_W = 1e-12                       # guard for c = -b/w when w crosses zero


# ----------------------------- data -----------------------------

def scan_cases():
    """{(seed, fn, N): path} for every snapshot NPZ on disk."""
    out = {}
    for p in sorted(SNAP_DIR.glob("*--*--N*.npz")):
        try:
            seed, fn, ntag = p.stem.split("--")
            out[(seed, fn, int(ntag[1:]))] = p
        except ValueError:
            continue
    return out


@lru_cache(maxsize=32)
def load_case(path_str):
    z = np.load(path_str)
    w, b, v = z["w"], z["b"], z["v"]                    # each (F, N)
    c = -b / np.where(np.abs(w) > EPS_W, w, EPS_W)      # current centers, (F, N)
    c0 = c[0]
    ranks = np.argsort(np.argsort(c0))                  # step-0 center rank per neuron
    t = ranks / max(len(c0) - 1, 1)
    colors = [f"rgb({int(255 * (1 - ti))},0,{int(255 * ti)})" for ti in t]  # red->blue

    def prc_lims(a):
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        pad = 0.10 * max(hi - lo, 1e-9)
        return float(lo - pad), float(hi + pad)

    return {"steps": z["steps"], "w": w, "b": b, "v": v, "c": c, "colors": colors,
            "losses": z["losses"], "rels": z["rels"],
            "glims": {"c": prc_lims(c), "w": prc_lims(w), "v": prc_lims(v)}}


# ----------------------------- figures -----------------------------

def fig3d(seed, fn, N, frame, axmode, manual, show=("axes", "faces")):
    """The 3D scatter for one frame. manual = (cmin,cmax,wmin,wmax,vmin,vmax);
    show toggles the axis lines/ticks/titles ("axes") and background planes ("faces")."""
    cases = scan_cases()
    path = cases.get((seed, fn, N))
    if path is None:
        f = go.Figure()
        f.add_annotation(text="no data for this combo yet -- hit Rescan files",
                         showarrow=False, font=dict(size=14))
        return f
    d = load_case(str(path))
    frame = int(np.clip(frame, 0, len(d["steps"]) - 1))
    c, w, v = d["c"][frame], d["w"][frame], d["v"][frame]
    step, total = int(d["steps"][frame]), int(d["steps"][-1])

    if axmode == "frame":
        def lim(a):
            lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
            pad = 0.10 * max(hi - lo, 1e-9)
            return [float(lo - pad), float(hi + pad)]
        ranges = {"c": lim(c), "w": lim(w), "v": lim(v)}
    elif axmode == "manual" and all(m is not None for m in manual):
        ranges = {"c": [manual[0], manual[1]], "w": [manual[2], manual[3]],
                  "v": [manual[4], manual[5]]}
    else:                                               # global
        ranges = {k: list(lims) for k, lims in d["glims"].items()}

    axes_on = "axes" in show
    faces_on = "faces" in show
    (x0, x1), (y0, y1), (z0, z1) = ranges["c"], ranges["w"], ranges["v"]

    # trace 0 is ALWAYS the neuron cloud (frame patches rely on this index)
    traces = [go.Scatter3d(
        x=c, y=w, z=v, mode="markers",
        marker=dict(size=3, color=d["colors"], opacity=0.9),
        hovertemplate=("c=%{x:.4f}<br>w=%{y:.4f}<br>v=%{z:.5f}<extra></extra>"))]

    if faces_on:                       # translucent coordinate planes THROUGH 0
        quads = [((0, 0, 0, 0), (y0, y1, y1, y0), (z0, z0, z1, z1)),   # x = 0
                 ((x0, x1, x1, x0), (0, 0, 0, 0), (z0, z0, z1, z1)),   # y = 0
                 ((x0, x1, x1, x0), (y0, y0, y1, y1), (0, 0, 0, 0))]   # z = 0
        for qx, qy, qz in quads:
            traces.append(go.Mesh3d(x=qx, y=qy, z=qz, i=[0, 0], j=[1, 2], k=[2, 3],
                                    color="#7d8fb0", opacity=0.10, hoverinfo="skip"))
    if axes_on:                        # axis lines through the origin, labeled at + end
        for xs, ys, zs, lbl in [((x0, x1), (0, 0), (0, 0), "c"),
                                ((0, 0), (y0, y1), (0, 0), "w"),
                                ((0, 0), (0, 0), (z0, z1), "v")]:
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines+text", text=["", lbl],
                textposition="top center", textfont=dict(size=13, color="#223"),
                line=dict(color="#445", width=4), hoverinfo="skip"))

    fig = go.Figure(traces)
    axis = dict(showgrid=False, showbackground=False, zeroline=False, showline=False,
                showticklabels=axes_on)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="center c = -b/w" if axes_on else "",
                       range=[x0, x1], **axis),
            yaxis=dict(title="first-layer w" if axes_on else "",
                       range=[y0, y1], **axis),
            zaxis=dict(title="readout v" if axes_on else "",
                       range=[z0, z1], **axis),
            aspectmode="cube",
            uirevision=f"{seed}|{fn}|{N}"),             # keep camera across rebuilds
        uirevision=f"{seed}|{fn}|{N}",
        showlegend=False,
        margin=dict(l=0, r=0, t=42, b=0),
        title=dict(text=f"{seed}   {fn}   N={N}   step {step}/{total}",
                   x=0.01, font=dict(size=14)),
        paper_bgcolor="white")
    return fig


def fig_err(seed, fn, N, frame):
    cases = scan_cases()
    path = cases.get((seed, fn, N))
    if path is None:
        return go.Figure(), ""
    d = load_case(str(path))
    frame = int(np.clip(frame, 0, len(d["steps"]) - 1))
    step = int(d["steps"][frame])
    losses, rels = d["losses"], d["rels"]
    n = len(losses)
    stride = max(1, n // 1500)                          # decimate for display only
    xs = np.arange(1, n + 1)[::stride]

    fig = go.Figure()
    fig.add_scatter(x=xs, y=losses[::stride], mode="lines", name="train MSE",
                    line=dict(color=ACCENT, width=1.4))
    fig.add_scatter(x=xs, y=rels[::stride], mode="lines", name="eval rel L2",
                    line=dict(color="#c44e52", width=1.4))
    if step >= 1:
        fig.add_vline(x=step, line_color="#444", line_width=1, line_dash="dot")
    fig.update_yaxes(type="log", showgrid=True, gridcolor=GRID)
    fig.update_xaxes(title="step", showgrid=True, gridcolor=GRID)
    fig.update_layout(
        margin=dict(l=8, r=8, t=34, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), height=270)

    i = min(step, n) - 1
    txt = ("at step 0: no update yet" if step < 1 else
           f"step {step}:  train MSE {losses[i]:.3e}   eval rel L2 {rels[i]:.3e}")
    return fig, txt


# ----------------------------- layout -----------------------------

app = Dash(__name__, title="expD06 3D explorer")


def _num(id_, ph):
    return dcc.Input(id=id_, type="number", placeholder=ph, debounce=True,
                     style={"width": "70px", "marginRight": "4px"})


def serve_layout():
    cases = scan_cases()
    seeds = sorted({k[0] for k in cases})
    return html.Div(style={"display": "flex", "fontFamily": "Helvetica, Arial",
                           "height": "100vh"}, children=[
        # ---------- sidebar ----------
        html.Div(style={"width": "330px", "minWidth": "330px", "padding": "12px",
                        "borderRight": f"1px solid {GRID}", "overflowY": "auto"},
                 children=[
            html.H3("expD06 3D parameter explorer", style={"margin": "0 0 2px"}),
            html.Div("x = center  |  y = first-layer w  |  z = readout v",
                     style={"fontSize": "11px", "color": "#555"}),
            html.Div("color: red = initially leftmost center, blue = rightmost "
                     "(fixed per neuron)", style={"fontSize": "11px", "color": "#555",
                                                  "marginBottom": "10px"}),
            html.Button("Rescan files", id="rescan", n_clicks=0,
                        style={"marginBottom": "10px"}),
            html.Label("seed", style={"fontSize": "12px", "fontWeight": "bold"}),
            dcc.Dropdown(id="seed", options=seeds, value=(seeds[0] if seeds else None),
                         clearable=False),
            html.Label("width N", style={"fontSize": "12px", "fontWeight": "bold",
                                         "marginTop": "8px", "display": "block"}),
            dcc.Dropdown(id="width", clearable=False),
            html.Label("target function", style={"fontSize": "12px",
                                                 "fontWeight": "bold",
                                                 "marginTop": "8px",
                                                 "display": "block"}),
            dcc.Dropdown(id="fn", clearable=False),
            html.Hr(),
            html.Label("playback", style={"fontSize": "12px", "fontWeight": "bold"}),
            html.Div([html.Button("Play", id="play", n_clicks=0,
                                  style={"width": "80px", "marginRight": "10px"}),
                      html.Span(id="speed-lbl", style={"fontSize": "11px"})],
                     style={"margin": "6px 0"}),
            dcc.Slider(id="speed", min=1, max=30, step=1, value=8,
                       marks={1: "1", 8: "8", 15: "15", 30: "30 fps"},
                       tooltip={"placement": "bottom"}),
            html.Hr(),
            html.Label("axis display", style={"fontSize": "12px",
                                              "fontWeight": "bold"}),
            dcc.Checklist(id="show", value=["axes", "faces"], options=[
                {"label": " axis lines / ticks / titles", "value": "axes"},
                {"label": " axis faces (background planes)", "value": "faces"}],
                style={"fontSize": "12px"}),
            html.Label("axis scaling", style={"fontSize": "12px",
                                              "fontWeight": "bold",
                                              "marginTop": "8px",
                                              "display": "block"}),
            dcc.RadioItems(id="axmode", value="global", options=[
                {"label": " global (fixed over run)", "value": "global"},
                {"label": " autoscale to current frame", "value": "frame"},
                {"label": " manual", "value": "manual"}],
                style={"fontSize": "12px"}),
            html.Div(style={"fontSize": "11px", "marginTop": "6px"}, children=[
                html.Div(["c: ", _num("cmin", "min"), _num("cmax", "max")]),
                html.Div(["w: ", _num("wmin", "min"), _num("wmax", "max")],
                         style={"marginTop": "3px"}),
                html.Div(["v: ", _num("vmin", "min"), _num("vmax", "max")],
                         style={"marginTop": "3px"})]),
            html.Hr(),
            html.Label("error over training", style={"fontSize": "12px",
                                                     "fontWeight": "bold"}),
            html.Div(id="err-txt", style={"fontSize": "11px", "margin": "4px 0",
                                          "fontFamily": "monospace"}),
            dcc.Graph(id="fig-err", config={"displayModeBar": False}),
        ]),
        # ---------- main ----------
        html.Div(style={"flex": "1", "display": "flex", "flexDirection": "column",
                        "padding": "6px 10px"}, children=[
            dcc.Graph(id="fig3d", style={"flex": "1", "minHeight": "0"},
                      config={"displayModeBar": True}),
            html.Div(dcc.Slider(id="frame", min=0, max=1, step=1, value=0,
                                updatemode="drag",
                                tooltip={"placement": "top"}),
                     style={"padding": "0 25px 8px"}),
        ]),
        dcc.Interval(id="tick", interval=125, disabled=True),
    ])


app.layout = serve_layout


# ----------------------------- callbacks -----------------------------

@app.callback(Output("seed", "options"), Output("seed", "value"),
              Input("rescan", "n_clicks"), State("seed", "value"))
def rescan(_, cur):
    seeds = sorted({k[0] for k in scan_cases()})
    return seeds, (cur if cur in seeds else (seeds[0] if seeds else None))


@app.callback(Output("width", "options"), Output("width", "value"),
              Output("fn", "options"), Output("fn", "value"),
              Input("seed", "value"), State("width", "value"), State("fn", "value"))
def update_combo(seed, cur_n, cur_fn):
    cases = scan_cases()
    ns = sorted({k[2] for k in cases if k[0] == seed})
    n = cur_n if cur_n in ns else (ns[0] if ns else None)
    fns = sorted({k[1] for k in cases if k[0] == seed and k[2] == n})
    fn = cur_fn if cur_fn in fns else (fns[0] if fns else None)
    return ns, n, fns, fn


@app.callback(Output("frame", "value"), Output("frame", "max"),
              Output("frame", "marks"),
              Input("tick", "n_intervals"),
              Input("seed", "value"), Input("width", "value"), Input("fn", "value"),
              State("frame", "value"), State("frame", "max"))
def drive_frame(_, seed, N, fn, cur, mx):
    from dash import no_update
    if ctx.triggered_id == "tick":                       # advance, wrap at end
        cur = 0 if cur is None or cur >= mx else cur + 1
        return cur, no_update, no_update
    path = scan_cases().get((seed, fn, N))               # case changed: reset slider
    if path is None:
        return 0, 1, {}
    steps = load_case(str(path))["steps"]
    F = len(steps)
    mark_at = np.unique(np.linspace(0, F - 1, 7).astype(int))
    marks = {int(i): str(int(steps[i])) for i in mark_at}
    return 0, F - 1, marks


@app.callback(Output("tick", "disabled"), Output("play", "children"),
              Input("play", "n_clicks"), State("tick", "disabled"))
def toggle_play(n, disabled):
    if not n:
        return True, "Play"
    return (not disabled), ("Pause" if disabled else "Play")


@app.callback(Output("tick", "interval"), Output("speed-lbl", "children"),
              Input("speed", "value"))
def set_speed(fps):
    return int(1000 / fps), f"{fps} frames/s"


@app.callback(Output("cmin", "value"), Output("cmax", "value"),
              Output("wmin", "value"), Output("wmax", "value"),
              Output("vmin", "value"), Output("vmax", "value"),
              Input("seed", "value"), Input("width", "value"), Input("fn", "value"))
def prefill_manual(seed, N, fn):
    path = scan_cases().get((seed, fn, N))
    if path is None:
        return [None] * 6
    g = load_case(str(path))["glims"]
    r = lambda x: float(f"{x:.4g}")
    return (r(g["c"][0]), r(g["c"][1]), r(g["w"][0]), r(g["w"][1]),
            r(g["v"][0]), r(g["v"][1]))


@app.callback(Output("fig3d", "figure"),
              Input("seed", "value"), Input("width", "value"), Input("fn", "value"),
              Input("frame", "value"), Input("axmode", "value"),
              Input("show", "value"),
              Input("cmin", "value"), Input("cmax", "value"),
              Input("wmin", "value"), Input("wmax", "value"),
              Input("vmin", "value"), Input("vmax", "value"))
def update_fig3d(seed, N, fn, frame, axmode, show, cmin, cmax, wmin, wmax, vmin, vmax):
    # frame-only change (except fit-frame mode): patch the point coordinates + title,
    # leave layout untouched -- camera, drag state, and axis labels stay put
    if ctx.triggered_id == "frame" and axmode != "frame":
        path = scan_cases().get((seed, fn, N))
        if path is not None:
            d = load_case(str(path))
            fr = int(np.clip(frame or 0, 0, len(d["steps"]) - 1))
            p = Patch()
            p["data"][0]["x"] = d["c"][fr]
            p["data"][0]["y"] = d["w"][fr]
            p["data"][0]["z"] = d["v"][fr]
            p["layout"]["title"]["text"] = (
                f"{seed}   {fn}   N={N}   step {int(d['steps'][fr])}"
                f"/{int(d['steps'][-1])}")
            return p
    return fig3d(seed, fn, N, frame or 0, axmode, (cmin, cmax, wmin, wmax, vmin, vmax),
                 tuple(show or []))


@app.callback(Output("fig-err", "figure"), Output("err-txt", "children"),
              Input("seed", "value"), Input("width", "value"), Input("fn", "value"),
              Input("frame", "value"))
def update_err(seed, N, fn, frame):
    return fig_err(seed, fn, N, frame or 0)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8051
    print(f"expD06 3D explorer -> http://127.0.0.1:{port}   (snaps: {SNAP_DIR})")
    app.run(debug=False, port=port)

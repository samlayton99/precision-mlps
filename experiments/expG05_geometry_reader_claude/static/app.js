"use strict";
// Geometry reader (Claude version) -- browser side.
// The server owns all state and numerics; this file draws views and turns gestures into commands.

const $ = (id) => document.getElementById(id);
const COL = { blue: "#1f4e8c", teal: "#138a8a", orange: "#e07b00", red: "#d1352b", green: "#2e9e44",
  grey: "#9aa0a8", ink: "#1d1f23", dim: "#6b7079", grid: "#eceef1", accent: "#6a3fb5", lsg: "#7a808a" };
const LEFT = 72, RIGHT = 14, TOP = 18, BOTTOM = 22, RAIL = 34;

let view = null;            // latest view from the server
let readoutView = "ls";     // which readout drives the fit/residual emphasis
let override = null;        // local drag state rendered on top of the latest view
let lastFrameShown = -1;
let flashTimer = null;
let presetsCatalog = [];

// ------------------------------------------------------------------ server I/O
async function api(cmd, body = {}) {
  const r = await fetch(`/api/${cmd}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!j.ok) showMessage(j.error, true);
  return j;
}

// Coalescing sender for drags: one request in flight, later gestures merge into the pending one.
const sender = {
  inflight: false, pending: null, onIdle: [],
  push(cmd, body, merge) {
    if (this.pending && this.pending.cmd === cmd && merge) this.pending.body = merge(this.pending.body, body);
    else { if (this.pending) this._queue.push(this.pending); this.pending = { cmd, body }; }
    this.flush();
  },
  _queue: [],
  async flush() {
    if (this.inflight) return;
    const next = this._queue.length ? this._queue.shift() : this.pending;
    if (!next) { const cbs = this.onIdle; this.onIdle = []; cbs.forEach((f) => f()); return; }
    if (next === this.pending) this.pending = null;
    this.inflight = true;
    try { await api(next.cmd, next.body); } finally { this.inflight = false; this.flush(); }
  },
  whenIdle(f) { if (!this.inflight && !this.pending && !this._queue.length) f(); else this.onIdle.push(f); },
};
const mergeEdit = (a, b) => {
  const out = { undo: a.undo || b.undo };
  for (const k of ["c", "g", "a"]) if (a[k] || b[k]) out[k] = Object.assign({}, a[k] || {}, b[k] || {});
  if (b.b !== undefined) out.b = b.b; else if (a.b !== undefined) out.b = a.b;
  return out;
};
const mergeScale = (a, b) => ({ op: "scale_g", factor: a.factor * b.factor, undo: a.undo || b.undo });

// ------------------------------------------------------------------ number formatting
const fmt = (v, p = 3) => (v === null || v === undefined || !isFinite(v)) ? "--" : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(p - 1) : v.toPrecision(p);
const fmtE = (v) => (v === null || v === undefined || !isFinite(v)) ? "--" : v.toExponential(2);

// ------------------------------------------------------------------ shared x view
const xview = {
  lo: -1.05, hi: 1.05, plots: [],
  set(lo, hi) { if (!(hi > lo)) return; this.lo = lo; this.hi = hi; requestDraw(); },
  zoom(xc, f) { this.set(xc - (xc - this.lo) * f, xc + (this.hi - xc) * f); },
  fitDomain() { if (!view) return; const [a, b] = view.settings.problem.domain; const p = 0.03 * (b - a); this.set(a - p, b + p); },
  fitAll() {
    if (!view) return; const c = view.geom.c; let lo = Math.min(...c, view.settings.problem.domain[0]), hi = Math.max(...c, view.settings.problem.domain[1]);
    const p = 0.03 * (hi - lo || 1); this.set(lo - p, hi + p);
  },
};

// ------------------------------------------------------------------ axis helpers
function niceStep(span, n) {
  const raw = span / Math.max(n, 1), p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}
function linTicks(lo, hi, n) { const s = niceStep(hi - lo, n), out = []; for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-12 * s; v += s) out.push(Math.abs(v) < s * 1e-9 ? 0 : v); return out; }
function tickLabel(v) { const a = Math.abs(v); if (a === 0) return "0"; if (a >= 1e4 || a < 1e-3) return v.toExponential(0).replace("e+", "e"); return String(+v.toPrecision(4)); }

// ------------------------------------------------------------------ Plot
class Plot {
  constructor(canvas, opts) {
    this.cv = canvas; this.ctx = canvas.getContext("2d"); this.o = Object.assign({ ymode: "lin", rail: 0 }, opts);
    this.ylo = 0; this.yhi = 1; this.yAuto = true; this.handles = []; this.drag = null; this.pan = null;
    xview.plots.push(this);
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e));
    canvas.addEventListener("pointercancel", (e) => this.up(e));
    canvas.addEventListener("pointerleave", () => hideTip());
    canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    canvas.addEventListener("dblclick", () => { this.yAuto = true; xview.fitDomain(); });
  }
  size() {
    const dpr = window.devicePixelRatio || 1, w = this.cv.clientWidth, h = this.cv.clientHeight;
    if (this.cv.width !== Math.round(w * dpr) || this.cv.height !== Math.round(h * dpr)) { this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr); }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = w; this.H = h; this.pw = w - LEFT - RIGHT; this.ph = h - TOP - BOTTOM - this.o.rail;
  }
  // transforms
  fy(v) { const m = this.o.ymode; if (m === "log") return Math.log10(Math.max(Math.abs(v), 1e-300)); if (m === "symlog") return Math.sign(v) * Math.log10(1 + Math.abs(v) / this.lin); return v; }
  iy(u) { const m = this.o.ymode; if (m === "log") return Math.pow(10, u); if (m === "symlog") return Math.sign(u) * this.lin * (Math.pow(10, Math.abs(u)) - 1); return u; }
  X(x) { return LEFT + (x - xview.lo) / (xview.hi - xview.lo) * this.pw; }
  Xi(px) { return xview.lo + (px - LEFT) / this.pw * (xview.hi - xview.lo); }
  Y(v) { return TOP + (1 - (this.fy(v) - this.ylo) / (this.yhi - this.ylo)) * this.ph; }
  Yi(py) { return this.iy(this.ylo + (1 - (py - TOP) / this.ph) * (this.yhi - this.ylo)); }
  autoY(vals, pad = 0.08) {
    if (!this.yAuto || this.drag) return;
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) { if (v === null || !isFinite(v)) continue; if (this.o.ymode === "log" && !(Math.abs(v) > 0)) continue; const u = this.fy(v); if (u < lo) lo = u; if (u > hi) hi = u; }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-12) { const d = this.o.ymode === "log" ? 0.5 : Math.max(Math.abs(hi) * 0.1, 1e-3); lo -= d; hi += d; }
    const p = (hi - lo) * pad; this.ylo = lo - p; this.yhi = hi + p;
  }
  frame() {
    const c = this.ctx; c.clearRect(0, 0, this.W, this.H);
    c.save(); c.font = "11px -apple-system, Helvetica, sans-serif"; c.fillStyle = COL.dim; c.strokeStyle = COL.grid; c.lineWidth = 1;
    // x ticks
    const xt = linTicks(xview.lo, xview.hi, Math.max(4, this.pw / 90));
    c.textAlign = "center"; c.textBaseline = "top";
    for (const v of xt) { const px = this.X(v); c.beginPath(); c.moveTo(px, TOP); c.lineTo(px, TOP + this.ph); c.stroke(); c.fillText(tickLabel(v), px, TOP + this.ph + this.o.rail + 5); }
    // y ticks
    c.textAlign = "right"; c.textBaseline = "middle";
    for (const [u, lab] of this.yTicks()) { const py = TOP + (1 - (u - this.ylo) / (this.yhi - this.ylo)) * this.ph; if (py < TOP - 1 || py > TOP + this.ph + 1) continue; c.beginPath(); c.moveTo(LEFT, py); c.lineTo(LEFT + this.pw, py); c.stroke(); c.fillText(lab, LEFT - 6, py); }
    c.strokeStyle = "#c3c6cc"; c.strokeRect(LEFT, TOP, this.pw, this.ph);
    c.restore();
  }
  yTicks() {
    const m = this.o.ymode, out = [];
    if (m === "log") {
      const span = this.yhi - this.ylo;
      if (span > 1.2) { const st = Math.max(1, Math.ceil(span / 7)); for (let e = Math.ceil(this.ylo); e <= this.yhi; e++) if (e % st === 0) out.push([e, `1e${e}`]); }
      else for (let e = Math.floor(this.ylo); e <= Math.ceil(this.yhi); e++) for (const k of [1, 2, 3, 5, 7]) out.push([e + Math.log10(k), tickLabel(k * Math.pow(10, e))]);
      return out;
    }
    if (m === "symres") { // signed log residual: u = sign(r)(log10|r| + 16)
      const M = Math.max(Math.abs(this.ylo), Math.abs(this.yhi)), st = Math.max(1, Math.ceil(2 * M / Math.max(this.ph / 15, 1)));
      out.push([0, "<1e-16"]);
      for (let k = st; k <= M; k += st) { out.push([k, `1e${k - 16}`]); out.push([-k, `-1e${k - 16}`]); }
      return out;
    }
    if (m === "symlog") { for (const u of linTicks(this.ylo, this.yhi, this.ph / 40)) out.push([u, tickLabel(this.iy(u))]); return out; }
    for (const v of linTicks(this.ylo, this.yhi, Math.max(3, this.ph / 38))) out.push([v, tickLabel(v)]);
    return out;
  }
  clip() { const c = this.ctx; c.save(); c.beginPath(); c.rect(LEFT, TOP - 4, this.pw, this.ph + 8); c.clip(); }
  line(xs, ys, color, width = 1.4, dash = null, useY = null) {
    const c = this.ctx; c.beginPath(); c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash || []);
    let pen = false; const Yf = useY || ((v) => this.Y(v));
    for (let i = 0; i < xs.length; i++) { const y = ys[i]; if (y === null || !isFinite(y)) { pen = false; continue; } const px = this.X(xs[i]), py = Yf(y); if (!pen) { c.moveTo(px, py); pen = true; } else c.lineTo(px, py); }
    c.stroke(); c.setLineDash([]);
  }
  dot(px, py, r, fill, stroke, lw = 1.2) { const c = this.ctx; c.beginPath(); c.arc(px, py, r, 0, 2 * Math.PI); if (fill) { c.fillStyle = fill; c.fill(); } if (stroke) { c.strokeStyle = stroke; c.lineWidth = lw; c.stroke(); } }
  // interaction
  pos(e) { const r = this.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  hit(px, py) {
    let best = null, bd = 81;
    for (const h of this.handles) { const d = (h.px - px) ** 2 + (h.py - py) ** 2; if (d < bd || (d === bd && best && h.prio > best.prio)) { bd = d; best = h; } }
    return best;
  }
  down(e) {
    const [px, py] = this.pos(e); const h = this.hit(px, py);
    this.cv.setPointerCapture(e.pointerId);
    if (h && h.drag) { this.drag = { h, first: true, px0: px, py0: py }; h.drag.start && h.drag.start(); return; }
    this.pan = { px, lo: xview.lo, hi: xview.hi, py, ylo: this.ylo, yhi: this.yhi, moved: false };
  }
  move(e) {
    const [px, py] = this.pos(e);
    if (this.drag) { const d = this.drag; d.h.drag.move(px, py, d.first); d.first = false; requestDraw(); return; }
    if (this.pan) {
      const dx = (px - this.pan.px) / this.pw * (this.pan.hi - this.pan.lo);
      xview.lo = this.pan.lo - dx; xview.hi = this.pan.hi - dx;
      if (!this.yAuto) { const dy = (py - this.pan.py) / this.ph * (this.pan.yhi - this.pan.ylo); this.ylo = this.pan.ylo + dy; this.yhi = this.pan.yhi + dy; }
      requestDraw(); return;
    }
    const h = this.hit(px, py);
    this.cv.style.cursor = h && h.drag ? (h.cursor || "ns-resize") : "crosshair";
    if (h && h.tip) showTip(e.clientX, e.clientY, h.tip()); else hideTip();
  }
  up(e) {
    if (this.drag) { const d = this.drag; this.drag = null; d.h.drag.end && d.h.drag.end(); }
    this.pan = null;
    try { this.cv.releasePointerCapture(e.pointerId); } catch (_) {}
  }
  wheel(e) {
    e.preventDefault();
    const [px, py] = this.pos(e); const f = Math.exp(Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 60) * 0.004);
    if (e.shiftKey || px < LEFT) { const u = this.ylo + (1 - (py - TOP) / this.ph) * (this.yhi - this.ylo); this.yAuto = false; this.ylo = u - (u - this.ylo) * f; this.yhi = u + (this.yhi - u) * f; requestDraw(); }
    else xview.zoom(this.Xi(px), f);
  }
}

// ------------------------------------------------------------------ tooltip / messages
function showTip(x, y, text) { const t = $("tooltip"); t.textContent = text; t.hidden = false; t.style.left = (x + 14) + "px"; t.style.top = (y + 12) + "px"; }
function hideTip() { $("tooltip").hidden = true; }
function showMessage(m, err = false) { const el = $("message"); el.textContent = m || ""; el.classList.toggle("err", err || /^error/.test(m || "")); }

// ------------------------------------------------------------------ plots
const P = {
  geom: new Plot($("c-geom"), { ymode: "log", rail: RAIL }),
  readout: new Plot($("c-readout"), { ymode: "lin" }),
  fit: new Plot($("c-fit"), { ymode: "lin" }),
  res: new Plot($("c-res"), { ymode: "symres" }),
};

// Values actually drawn: the server view with any in-progress drag applied on top.
function current() {
  const v = view; if (!v) return null;
  let c = v.geom.c, g = v.geom.g, a = v.readout.a;
  if (override) {
    if (override.c) { c = c.slice(); for (const [i, x] of Object.entries(override.c)) c[i] = x; }
    if (override.g) { g = g.slice(); for (const [i, x] of Object.entries(override.g)) g[i] = x; }
    if (override.scale) g = override.gBase.map((x) => x * override.scale);   // from drag-start values
    if (override.a) { a = a.slice(); for (const [i, x] of Object.entries(override.a)) a[i] = x; }
  }
  return { c, g, a, ag: g.map(Math.abs) };
}
function changedSet(key) { const ch = view && view.changed; return new Set(ch && ch[key] ? ch[key] : []); }

function drawGeom() {
  const p = P.geom, v = view, cur = current(); p.size();
  const { c, ag } = cur, W = c.length, ls = v.geom.lambda_star;
  p.o.ymode = $("v-glog").checked ? "log" : "lin";
  // local spacing / ideal gamma from the drawn centers (so it follows drags)
  const order = [...c.keys()].sort((i, j) => c[i] - c[j]);
  const h = new Array(W).fill(1);
  if (W > 1) order.forEach((k, r) => { const lo = order[Math.max(r - 1, 0)], hi = order[Math.min(r + 1, W - 1)]; h[k] = (c[hi] - c[lo]) / (r === 0 || r === W - 1 ? 1 : 2) || 1e-12; });
  const ideal = h.map((x) => ls / Math.max(x, 1e-300));
  const inView = [...c.keys()].filter((i) => c[i] >= xview.lo && c[i] <= xview.hi);
  const vals = inView.map((i) => ag[i]);
  if ($("v-ideal").checked && vals.length) { // ideal curve joins the scale unless near-collisions make it explode
    const lo = Math.min(...vals), hi = Math.max(...vals);
    inView.forEach((i) => { if (ideal[i] >= lo / 10 && ideal[i] <= hi * 10) vals.push(ideal[i]); });
  }
  if (v.ghost) inView.forEach((i) => vals.push(Math.abs(v.ghost.g[i])));
  p.autoY(vals.length ? vals : ag, 0.1);
  p.frame();
  const ctx = p.ctx; p.clip();
  // domain band
  const [da, db] = v.settings.problem.domain; ctx.fillStyle = "rgba(31,78,140,0.045)"; ctx.fillRect(p.X(da), TOP, p.X(db) - p.X(da), p.ph);
  if ($("v-ideal").checked) {
    p.line(order.map((k) => c[k]), order.map((k) => ideal[k]), COL.orange, 1.3, [5, 4]);
  }
  const chC = changedSet("c"), chG = changedSet("g");
  // ghosts: where the neuron was before the edit / intervention
  if (v.ghost) {
    for (const i of new Set([...chC, ...chG])) {
      const gx = p.X(v.ghost.c[i]), gy = p.Y(Math.abs(v.ghost.g[i])), nx = p.X(c[i]), ny = p.Y(ag[i]);
      ctx.strokeStyle = "rgba(224,123,0,.6)"; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(nx, ny); ctx.stroke(); ctx.setLineDash([]);
      p.dot(gx, gy, 3.5, null, COL.grey);
    }
  }
  p.handles = [];
  const r = W > 300 ? 2.6 : W > 120 ? 3.4 : 4.2;
  for (let i = 0; i < W; i++) {
    const px = p.X(c[i]), py = p.Y(ag[i]);
    const changed = chC.has(i) || chG.has(i) || (override && ((override.g && i in override.g) || (override.c && i in override.c)));
    const col = changed ? COL.orange : COL.blue;
    const neg = cur.g[i] < 0;
    if (px > LEFT - 5 && px < LEFT + p.pw + 5) p.dot(px, py, r, neg ? "#fff" : col, col);
    p.handles.push({ px, py, prio: 1, cursor: "ns-resize", tip: () => neuronTip(i, h[i]), drag: dragGamma(i) });
  }
  ctx.restore();
  // rail with center handles
  const ry = TOP + p.ph + RAIL / 2 + 2;
  ctx.strokeStyle = "#c3c6cc"; ctx.beginPath(); ctx.moveTo(LEFT, ry); ctx.lineTo(LEFT + p.pw, ry); ctx.stroke();
  let offL = 0, offR = 0;
  for (let i = 0; i < W; i++) {
    const px = p.X(c[i]);
    if (px < LEFT) { offL++; continue; } if (px > LEFT + p.pw) { offR++; continue; }
    const col = chC.has(i) || (override && override.c && i in override.c) ? COL.orange : COL.teal;
    ctx.strokeStyle = "rgba(19,138,138,.25)"; ctx.beginPath(); ctx.moveTo(px, ry - 6); ctx.lineTo(px, ry + 6); ctx.stroke();
    p.dot(px, ry, r, col, null);
    p.handles.push({ px, py: ry, prio: 2, cursor: "ew-resize", tip: () => neuronTip(i, h[i]), drag: dragCenter(i) });
  }
  ctx.fillStyle = COL.dim; ctx.font = "11px -apple-system, Helvetica, sans-serif"; ctx.textBaseline = "middle";
  if (offL) { ctx.textAlign = "left"; ctx.fillText(`< ${offL} off-screen`, LEFT + 3, ry - 12); }
  if (offR) { ctx.textAlign = "right"; ctx.fillText(`${offR} off-screen >`, LEFT + p.pw - 3, ry - 12); }
  // mean-gamma knob on the y axis (geometric mean in log mode)
  const mean = p.o.ymode === "log" ? Math.exp(ag.reduce((s, x) => s + Math.log(Math.max(x, 1e-300)), 0) / W) : ag.reduce((s, x) => s + x, 0) / W;
  const my = p.Y(mean), mx = LEFT - 1;
  ctx.fillStyle = COL.accent; ctx.beginPath(); ctx.moveTo(mx, my - 7); ctx.lineTo(mx + 7, my); ctx.lineTo(mx, my + 7); ctx.lineTo(mx - 7, my); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "rgba(106,63,181,.35)"; ctx.setLineDash([1, 3]); ctx.beginPath(); ctx.moveTo(LEFT, my); ctx.lineTo(LEFT + p.pw, my); ctx.stroke(); ctx.setLineDash([]);
  p.handles.push({ px: mx, py: my, prio: 3, cursor: "ns-resize", tip: () => `mean gamma ${fmt(mean, 4)}\nmean lambda ${fmt(view.metrics.mean_lambda, 3)}  median ${fmt(view.metrics.median_lambda, 3)}\ndrag: scale every gamma`, drag: dragMean(mean) });
}

function neuronTip(i, h) {
  const cur = current(), v = view;
  return `neuron ${i}\nc      ${cur.c[i].toPrecision(8)}\ngamma  ${cur.g[i].toPrecision(6)}\nlambda ${(Math.abs(cur.g[i]) * h).toPrecision(4)}  (h=${h.toPrecision(3)})\na      ${cur.a[i].toPrecision(6)}\na_ls   ${v.readout.a_ls[i] === null ? "--" : v.readout.a_ls[i].toPrecision(6)}`;
}

function drawReadout() {
  const p = P.readout, v = view, cur = current(); p.size();
  const { c, a } = cur, als = v.readout.a_ls, W = c.length;
  let amax = 0; for (let i = 0; i < W; i++) if (c[i] >= xview.lo && c[i] <= xview.hi) amax = Math.max(amax, Math.abs(a[i]), Math.abs(als[i] || 0));
  p.o.ymode = $("v-asym").checked ? "symlog" : "lin"; p.lin = Math.max(amax * 1e-3, 1e-300);
  // Scale to the draggable readout and f' h/2 (both O(h)); least-squares values far outside
  // that range (ill-conditioned geometry) are pinned to the edge so they cannot stretch the axis.
  const vals = [0];
  let base = 0;
  for (let i = 0; i < W; i++) if (c[i] >= xview.lo && c[i] <= xview.hi) { vals.push(a[i]); base = Math.max(base, Math.abs(a[i])); }
  v.deriv.x.forEach((x, k) => { if (x >= xview.lo && x <= xview.hi && v.deriv.y[k] !== null) { base = Math.max(base, Math.abs(v.deriv.y[k])); if ($("v-deriv").checked) vals.push(v.deriv.y[k]); } });
  const cap = 4 * (base || 1);
  for (let i = 0; i < W; i++) if (c[i] >= xview.lo && c[i] <= xview.hi && als[i] !== null && Math.abs(als[i]) <= cap) vals.push(als[i]);
  p.autoY(vals, 0.1); p.frame(); p.clip();
  const ctx = p.ctx;
  ctx.strokeStyle = "#c3c6cc"; ctx.beginPath(); ctx.moveTo(LEFT, p.Y(0)); ctx.lineTo(LEFT + p.pw, p.Y(0)); ctx.stroke();
  if ($("v-deriv").checked) p.line(v.deriv.x, v.deriv.y, "rgba(209,53,43,.75)", 1.3);
  const lsMain = readoutView === "ls", chA = changedSet("a");
  const r = W > 300 ? 2.4 : W > 120 ? 3 : 3.8;
  if (v.ghost) for (const i of chA) p.dot(p.X(c[i]), p.Y(v.ghost.a[i]), r, null, COL.grey);
  let clipped = 0;
  for (let i = 0; i < W; i++) {
    if (als[i] === null) continue;
    const py = p.Y(als[i]);
    if (py < TOP || py > TOP + p.ph) { // off-scale least-squares coefficient: triangle at the edge
      clipped++; const px = p.X(c[i]), up = py < TOP, ey = up ? TOP + 1 : TOP + p.ph - 1;
      ctx.fillStyle = lsMain ? COL.ink : COL.lsg; ctx.beginPath(); ctx.moveTo(px, ey); ctx.lineTo(px - 4, ey + (up ? 7 : -7)); ctx.lineTo(px + 4, ey + (up ? 7 : -7)); ctx.fill();
      continue;
    }
    p.dot(p.X(c[i]), py, lsMain ? r + 0.6 : r, null, lsMain ? COL.ink : COL.lsg, lsMain ? 1.5 : 1);
  }
  p.handles = [];
  for (let i = 0; i < W; i++) {
    const px = p.X(c[i]), py = p.Y(a[i]);
    const changed = chA.has(i) || (override && override.a && i in override.a);
    const col = changed ? COL.orange : COL.blue;
    ctx.globalAlpha = lsMain && !changed ? 0.45 : 1; p.dot(px, py, r, col, null); ctx.globalAlpha = 1;
    p.handles.push({ px, py, prio: 1, cursor: "ns-resize", tip: () => neuronTip(i, 1) .replace(/lambda.*\n/, ""), drag: dragReadout(i) });
  }
  ctx.restore();
  ctx.fillStyle = COL.dim; ctx.font = "11px -apple-system, Helvetica, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "top";
  ctx.fillText(`bias: Adam ${fmt(v.readout.b, 4)}   lstsq ${fmt(v.readout.b_ls, 4)}${clipped ? `   ${clipped} lstsq coefficients off-scale (triangles)` : ""}`, LEFT + p.pw - 4, TOP + 3);
}

function drawFit() {
  const p = P.fit, v = view; p.size();
  const cv = v.curves, fh = readoutView === "ls" ? cv.fhat_ls : cv.fhat;
  const vals = cv.f.filter((_, k) => cv.x[k] >= xview.lo && cv.x[k] <= xview.hi);
  p.autoY(vals.length ? vals : cv.f, 0.1); p.frame(); p.clip();
  if ($("v-train").checked) { const t = v.train; for (let k = 0; k < t.x.length; k++) p.dot(p.X(t.x[k]), p.Y(t.y[k]), 1.6, COL.grey, null); }
  p.line(cv.x, cv.f, COL.ink, 2.2);
  p.line(cv.x, fh, COL.blue, 1.5, [6, 3]);
  p.ctx.restore();
}

function drawRes() {
  const p = P.res, v = view; p.size();
  const cv = v.curves, main = readoutView === "ls" ? cv.r_ls : cv.r, other = readoutView === "ls" ? cv.r : cv.r_ls;
  const tr = (arr) => arr.map((r) => (r === null ? null : Math.sign(r) * Math.max(0, Math.log10(Math.abs(r) || 1e-300) + 16)));
  const um = tr(main), uo = tr(other);
  let M = 1; for (const u of um) if (u !== null && Math.abs(u) > M) M = Math.abs(u);
  for (const u of uo) if (u !== null && Math.abs(u) > M) M = Math.abs(u);
  if (p.yAuto && !p.drag) { p.ylo = -(M + 0.6); p.yhi = M + 0.6; }
  p.fy = (u) => u; p.frame(); p.clip();
  const ctx = p.ctx, Yu = (u) => p.Y(u);
  ctx.strokeStyle = "#c3c6cc"; ctx.beginPath(); ctx.moveTo(LEFT, Yu(0)); ctx.lineTo(LEFT + p.pw, Yu(0)); ctx.stroke();
  p.line(cv.x, uo, "rgba(122,128,138,.45)", 1);
  p.line(cv.x, um, COL.green, 1.3);
  const linf = Math.max(...main.map((r) => Math.abs(r || 0)));
  const ul = Math.max(0, Math.log10(linf || 1e-300) + 16);
  ctx.setLineDash([4, 3]); ctx.strokeStyle = "rgba(160,20,20,.8)";
  for (const s of [1, -1]) { ctx.beginPath(); ctx.moveTo(LEFT, Yu(s * ul)); ctx.lineTo(LEFT + p.pw, Yu(s * ul)); ctx.stroke(); }
  ctx.setLineDash([]); ctx.restore();
  ctx.fillStyle = COL.dim; ctx.font = "11px -apple-system, Helvetica, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "top";
  ctx.fillText(`${readoutView === "ls" ? "lstsq" : "Adam"} L∞ ${fmtE(linf)}  (grey: the other readout)`, LEFT + p.pw - 4, TOP + 3);
}

function drawHist() {
  const cv = $("c-hist"), ctx = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight; cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const H = view && view.history; if (!H || !H.step || H.step.length < 1) { ctx.fillStyle = COL.dim; ctx.font = "11px sans-serif"; ctx.fillText("eval rel L2 vs step appears here during a run", 8, h / 2); return; }
  const L = 46, R = 8, T = 6, B = 16, pw = w - L - R, ph = h - T - B;
  const s0 = H.step[0], sx = H.step.map((s) => Math.log10(s - s0 + 1));
  const xmax = Math.max(sx[sx.length - 1], 1);
  const all = H.rel_l2.concat(H.ls_rel_l2).filter((v) => v && v > 0).map(Math.log10);
  let lo = Math.floor(Math.min(...all, -1)), hi = Math.ceil(Math.max(...all, 0)); if (hi - lo < 2) lo = hi - 2;
  const X = (u) => L + u / xmax * pw, Y = (u) => T + (1 - (u - lo) / (hi - lo)) * ph;
  ctx.font = "10px -apple-system, Helvetica, sans-serif"; ctx.fillStyle = COL.dim; ctx.strokeStyle = COL.grid; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  const st = Math.max(1, Math.ceil((hi - lo) / 5));
  for (let e = lo; e <= hi; e += st) { ctx.beginPath(); ctx.moveTo(L, Y(e)); ctx.lineTo(L + pw, Y(e)); ctx.stroke(); ctx.fillText(`1e${e}`, L - 4, Y(e)); }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let e = 0; e <= xmax; e++) ctx.fillText(e === 0 ? `${s0}` : `+1e${e}`, X(e), T + ph + 3);
  const inter = new Set((view.run && view.run.interventions) || []);
  for (const i of inter) { ctx.strokeStyle = "rgba(209,53,43,.5)"; ctx.beginPath(); ctx.moveTo(X(sx[i]), T); ctx.lineTo(X(sx[i]), T + ph); ctx.stroke(); }
  const series = (arr, col, lw) => { ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath(); let pen = false; arr.forEach((v, k) => { if (!(v > 0)) { pen = false; return; } const px = X(sx[k]), py = Y(Math.log10(v)); if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py); }); ctx.stroke(); ctx.lineWidth = 1; };
  series(H.ls_rel_l2, readoutView === "ls" ? COL.ink : COL.grey, readoutView === "ls" ? 1.6 : 1);
  series(H.rel_l2, readoutView === "adam" ? COL.blue : "rgba(31,78,140,.5)", readoutView === "adam" ? 1.6 : 1);
  if (view.run) { const i = view.run.frame; ctx.strokeStyle = COL.accent; ctx.beginPath(); ctx.moveTo(X(sx[i]), T); ctx.lineTo(X(sx[i]), T + ph); ctx.stroke(); }
  ctx.fillStyle = COL.blue; ctx.textAlign = "left"; ctx.fillText("Adam", L + 4, T + 2); ctx.fillStyle = COL.ink; ctx.fillText("lstsq", L + 40, T + 2);
  cv.onclick = (e) => { if (!view.run) return; const r = cv.getBoundingClientRect(); const u = (e.clientX - r.left - L) / pw * xmax; let best = 0; sx.forEach((s, k) => { if (Math.abs(s - u) < Math.abs(sx[best] - u)) best = k; }); seek(best); };
}

// ------------------------------------------------------------------ drags
function editable() { return view && !(view.run && view.mode !== "replay" && !view.run.following); }
let dragId = 0;
function beginOverride() { dragId++; override = {}; }
function endDrag() { const id = dragId; sender.whenIdle(() => { if (id === dragId) { override = null; requestDraw(); } }); }
function stopReplayPlayback() { }

function dragGamma(i) {
  return {
    start() { stopReplayPlayback(); beginOverride(); },
    move(px, py, first) {
      if (!editable()) return;
      const p = P.geom, py2 = Math.min(Math.max(py, TOP), TOP + p.ph);
      const val = Math.max(p.Yi(py2), 1e-300) * (view.geom.g[i] < 0 ? -1 : 1);
      override.g = Object.assign(override.g || {}, { [i]: val });
      sender.push("edit", { g: { [i]: val }, undo: first }, mergeEdit);
    },
    end: endDrag,
  };
}
function dragCenter(i) {
  return {
    start() { stopReplayPlayback(); beginOverride(); },
    move(px, py, first) {
      if (!editable()) return;
      const val = P.geom.Xi(Math.min(Math.max(px, LEFT), LEFT + P.geom.pw));
      override.c = Object.assign(override.c || {}, { [i]: val });
      sender.push("edit", { c: { [i]: val }, undo: first }, mergeEdit);
    },
    end: endDrag,
  };
}
function dragReadout(i) {
  return {
    start() { stopReplayPlayback(); beginOverride(); },
    move(px, py, first) {
      if (!editable()) return;
      const p = P.readout, val = p.Yi(Math.min(Math.max(py, TOP), TOP + p.ph));
      override.a = Object.assign(override.a || {}, { [i]: val });
      sender.push("edit", { a: { [i]: val }, undo: first }, mergeEdit);
    },
    end: endDrag,
  };
}
function dragMean(mean0) {
  let sent = 1;
  return {
    start() { stopReplayPlayback(); beginOverride(); override.scale = 1; override.gBase = view.geom.g.slice(); sent = 1; },
    move(px, py, first) {
      if (!editable()) return;
      const p = P.geom, target = Math.max(p.Yi(Math.min(Math.max(py, TOP), TOP + p.ph)), 1e-300);
      const total = target / mean0;
      override.scale = total;
      sender.push("op", { op: "scale_g", factor: total / sent, undo: first }, mergeScale);
      sent = total;
    },
    end: endDrag,
  };
}

// ------------------------------------------------------------------ render loop
let drawQueued = false;
function requestDraw() { if (!drawQueued) { drawQueued = true; requestAnimationFrame(() => { drawQueued = false; drawAll(); }); } }
function drawAll() {
  if (!view || !view.geom) return;
  try { drawGeom(); drawReadout(); drawFit(); drawRes(); drawHist(); } catch (e) { console.error(e); showMessage("draw error: " + e.message, true); }
}
window.addEventListener("resize", requestDraw);

// ------------------------------------------------------------------ panel sync
function setIf(id, val) { const el = $(id); if (!el || document.activeElement === el) return; if (el.type === "checkbox") el.checked = !!val; else el.value = val; }
let firstView = true, lastResets = null;
function syncPanel() {
  const v = view, s = v.settings, pr = s.problem;
  if (v.resets !== lastResets) { lastResets = v.resets; firstView = true; }
  setIf("k-target", pr.target); setIf("k-dom-a", pr.domain[0]); setIf("k-dom-b", pr.domain[1]);
  setIf("k-ntrain", pr.n_train); setIf("k-sampling", pr.sampling); setIf("k-dseed", pr.data_seed);
  setIf("k-noise", pr.noise); setIf("k-rcond", pr.rcond > 0 ? Math.round(Math.log10(pr.rcond)) : -17);
  setIf("k-lstar", pr.lambda_star); setIf("k-neval", pr.n_eval);
  const matchT = [...$("k-target-preset").options].find((o) => o.dataset.expr === pr.target); $("k-target-preset").value = matchT ? matchT.value : "";
  const W = v.geom.c.length; setIf("g-count", W); setIf("g-count-n", W);
  setIf("g-clean-halo", s.clean_halo);
  const runCfg = v.run ? v.run : null, A = runCfg ? runCfg.adam : s.adam, S = runCfg ? runCfg.snap : s.snap;
  if (firstView || v.mode === "replay" || v.run) {
    setIf("a-lr", A.lr); setIf("a-glr", A.geom_lr_mult); setIf("a-b1", A.beta1); setIf("a-b2", A.beta2); setIf("a-eps", A.eps);
    setIf("a-batch", A.batch); setIf("a-sched", A.schedule); setIf("a-gparam", A.gamma_param);
    setIf("a-tc", A.train_c); setIf("a-tg", A.train_g); setIf("a-ta", A.train_a); setIf("a-tb", A.train_b);
    setIf("s-mode", S.mode); setIf("s-every", S.every); setIf("s-first", S.first); setIf("s-ratio", S.ratio); setIf("s-max", S.max_gap);
  }
  if (firstView) { setIf("a-steps", s.steps); setIf("a-rinit", s.readout_init.mode); setIf("a-rseed", s.readout_init.seed); setIf("t-reset-mom", s.reset_moments_on_inject); setDelayFromMs(s.delay_ms); }
  firstView = false;
  const locked = v.mode !== "idle";
  $("adam-box").querySelectorAll("input:not(#r-name), select").forEach((el) => { if (!["a-rinit", "a-rseed", "a-steps"].includes(el.id)) el.disabled = locked; });
  $("a-rapply").disabled = v.mode === "replay"; $("adam-lock").textContent = locked ? (v.mode === "replay" ? "(settings of the replayed run)" : "(locked while a run exists; Stop to change)") : "";
  // status
  const light = $("light"); light.className = "light " + (v.mode === "idle" ? "" : v.mode);
  $("mode-label").textContent = { idle: "no run (editing)", live: "training live", paused: "paused", replay: "replay" }[v.mode];
  const r = v.run;
  $("stepinfo").textContent = r ? `step ${r.step}${r.live_step !== undefined && r.live_step !== r.step ? ` (live ${r.live_step})` : ""} / ${r.total}   frame ${r.frame + 1}/${r.n_frames}` : "";
  $("t-play").textContent = (v.mode === "live" || (v.mode === "replay" && v.replay_playing)) ? "Pause" : "Play";
  $("t-stop").disabled = v.mode === "idle";
  $("t-follow").disabled = !(r && v.mode !== "replay" && !r.following);
  const sc = $("t-scrub"); sc.max = r ? r.n_frames - 1 : 0; if (!scrubbing) sc.value = r ? r.frame : 0; sc.disabled = !r;
  const ticks = $("scrub-ticks"); ticks.innerHTML = "";
  if (r && r.n_frames > 1) for (const i of r.interventions) { const t = document.createElement("i"); t.style.left = `calc(${(i / (r.n_frames - 1)) * 100}% - 1px)`; ticks.appendChild(t); }
  $("fork").textContent = v.fork ? `next Play continues '${v.fork.name}' from step ${v.fork.step} (Adam moments carried)` : (r && r.parent ? `forked from '${r.parent.name}' at step ${r.parent.step}` : "");
  // staged
  const sb = $("staged-box"); sb.classList.toggle("active", v.has_staged);
  if (v.has_staged && v.changed) { const ch = v.changed; $("staged-info").textContent = `staged: ${ch.c.length} centers, ${ch.g.length} gammas, ${ch.a.length} readout${ch.b ? ", bias" : ""} (orange; grey = run state)`; }
  else $("staged-info").textContent = v.mode === "idle" ? "edits apply directly (no run)" : v.mode === "replay" ? "editing ends the replay and continues from this frame" : "no staged edits";
  $("t-inject").disabled = !v.has_staged; $("t-discard").disabled = !v.has_staged;
  // metrics
  const m = v.metrics;
  const fill = (row, d) => { const tds = $(row).querySelectorAll("td"); tds[1].textContent = fmtE(d.train_mse); tds[2].textContent = fmtE(d.rel_l2); tds[3].textContent = fmtE(d.linf); };
  fill("m-adam", m.adam); fill("m-ls", m.ls);
  $("m-adam").classList.toggle("sel", readoutView === "adam"); $("m-ls").classList.toggle("sel", readoutView === "ls");
  $("m-extra").textContent = `W ${m.W}   rank ${m.rank}   cond ${fmtE(m.cond)}   lambda mean ${fmt(m.mean_lambda)} median ${fmt(m.median_lambda)}`;
  $("r-dir").textContent = v.recordings_dir;
  $("g-undo").disabled = !v.undo; $("g-redo").disabled = !v.redo;
  showMessage(v.message);
  snapPreview();
}

function onView(v) {
  if (v.error) { showMessage(v.error, true); return; }
  const prevKey = view ? `${view.mode}|${view.run && view.run.id}` : "";
  view = v;
  if (prevKey !== `${v.mode}|${v.run && v.run.id}`) runsDirty = true;   // run started/paused/ended: refresh the saved list
  syncPanel();
  // flash on intervention frames (live or replay)
  const r = v.run;
  if (r && v.frame_kind && r.frame !== lastFrameShown) {
    const el = $("plots"); el.classList.add("flash"); clearTimeout(flashTimer); flashTimer = setTimeout(() => el.classList.remove("flash"), 900);
  }
  lastFrameShown = r ? r.frame : -1;
  requestDraw();
  if (runsDirty) { runsDirty = false; loadRuns(); }
}

// replay playback runs on the server (steady frame timing); seeking stops it
function seek(i) { api("seek", { i }); }

// ------------------------------------------------------------------ controls
function delayMs() { const v = +$("t-delay").value; return v === 0 ? 0 : Math.round(5000 * Math.pow(v / 100, 3)); }
function setDelayFromMs(ms) { $("t-delay").value = ms <= 0 ? 0 : Math.round(100 * Math.cbrt(ms / 5000)); $("t-delay-v").textContent = `${ms} ms`; }
$("t-delay").addEventListener("input", () => { $("t-delay-v").textContent = `${delayMs()} ms`; });
$("t-delay").addEventListener("change", () => api("set", { delay_ms: delayMs() }));

function adamBody() {
  return {
    lr: +$("a-lr").value, geom_lr_mult: +$("a-glr").value, beta1: +$("a-b1").value, beta2: +$("a-b2").value, eps: +$("a-eps").value,
    batch: +$("a-batch").value, schedule: $("a-sched").value, gamma_param: $("a-gparam").value,
    train_c: $("a-tc").checked, train_g: $("a-tg").checked, train_a: $("a-ta").checked, train_b: $("a-tb").checked,
  };
}
function snapBody() { return { mode: $("s-mode").value, every: +$("s-every").value, first: +$("s-first").value, ratio: +$("s-ratio").value, max_gap: +$("s-max").value }; }
function snapPreview() {
  const S = snapBody(), steps = +$("a-steps").value; if (!(steps > 0)) return;
  let s = 0, i = 0, n = 1; while (s < steps && n < 1e6) { s += S.mode === "every" ? Math.max(1, S.every) : Math.max(1, Math.min(Math.round(S.first * Math.pow(S.ratio, i)), S.max_gap)); i++; n++; }
  $("s-preview").textContent = `${n} frames over ${steps} steps`;
}
["s-mode", "s-every", "s-first", "s-ratio", "s-max", "a-steps"].forEach((id) => $(id).addEventListener("input", snapPreview));
$("adam-box").addEventListener("change", (e) => {
  if (!view || view.mode !== "idle" || ["a-rinit", "a-rseed", "r-name"].includes(e.target.id)) return;
  api("set", { adam: adamBody(), snap: snapBody(), steps: +$("a-steps").value });
});

async function play() {
  if (!view) return;
  if (view.mode === "replay") { await api("replay_play", { on: !view.replay_playing }); return; }
  if (view.mode === "live") { await api("pause"); return; }
  const res = await api("play", { adam: adamBody(), snap: snapBody(), steps: +$("a-steps").value, name: $("r-name").value.trim() });
  if (res.needs_decision) $("modal").hidden = false;
  else if (res.ok) { $("r-name").value = ""; runsDirty = true; }
}
$("t-play").onclick = play;
$("t-stop").onclick = () => { api("stop").then(() => { runsDirty = true; }); };
$("t-follow").onclick = () => api("follow");
$("mo-inject").onclick = async () => { $("modal").hidden = true; await api("inject"); play(); };
$("mo-discard").onclick = async () => { $("modal").hidden = true; await api("discard"); play(); };
$("mo-cancel").onclick = () => { $("modal").hidden = true; };
$("t-inject").onclick = () => api("inject");
$("t-discard").onclick = () => api("discard");
$("t-inject-ls").onclick = () => api("inject_ls");
$("t-reset-mom").onchange = (e) => api("set", { reset_moments_on_inject: e.target.checked });
let scrubbing = false;
$("t-scrub").addEventListener("input", (e) => { scrubbing = true; sender.push("seek", { i: +e.target.value }, (a, b) => b); });
$("t-scrub").addEventListener("change", () => { scrubbing = false; });

// readout view toggle
function setReadoutView(v) { readoutView = v; document.querySelectorAll("#v-readout button").forEach((b) => b.classList.toggle("on", b.dataset.v === v)); if (view) { syncPanel(); requestDraw(); } }
document.querySelectorAll("#v-readout button").forEach((b) => (b.onclick = () => setReadoutView(b.dataset.v)));
["v-glog", "v-asym", "v-deriv", "v-train", "v-ideal"].forEach((id) => $(id).addEventListener("change", () => { if (id === "v-glog") P.geom.yAuto = true; if (id === "v-asym") P.readout.yAuto = true; requestDraw(); }));
$("v-fit-domain").onclick = () => xview.fitDomain();
$("v-fit-all").onclick = () => xview.fitAll();

// knobs (each change ends a live run or a replay on the server)
function problemBody() {
  const rc = +$("k-rcond").value;
  return { target: $("k-target").value, domain: [+$("k-dom-a").value, +$("k-dom-b").value], n_train: +$("k-ntrain").value,
    sampling: $("k-sampling").value, data_seed: +$("k-dseed").value, noise: +$("k-noise").value || 0,
    rcond: rc <= -17 ? 0 : Math.pow(10, rc), lambda_star: +$("k-lstar").value, n_eval: +$("k-neval").value };
}
["k-target", "k-dom-a", "k-dom-b", "k-ntrain", "k-sampling", "k-dseed", "k-noise", "k-rcond", "k-lstar", "k-neval"].forEach((id) =>
  $(id).addEventListener("change", () => { api("set_problem", problemBody()).then(() => { runsDirty = true; }); }));
$("k-target-preset").addEventListener("change", (e) => { const o = e.target.selectedOptions[0]; if (o && o.dataset.expr) { $("k-target").value = o.dataset.expr; api("set_problem", problemBody()); } });

// geometry tools
const pv = (id) => { const f = () => ($(id + "v").textContent = `${$(id).value}%`); $(id).addEventListener("input", f); f(); };
pv("g-clean-p"); pv("g-jit-c-p"); pv("g-jit-g-p");
$("g-clean").onclick = () => { api("op", { op: "clean", p: +$("g-clean-p").value / 100, what: $("g-clean-what").value }); };
$("g-clean-halo").onchange = (e) => api("set", { clean_halo: e.target.value });
$("g-jit-c").onclick = () => { api("op", { op: "jitter_c", p: +$("g-jit-c-p").value / 100, seed: +$("g-jseed").value }); };
$("g-jit-g").onclick = () => { api("op", { op: "jitter_g", p: +$("g-jit-g-p").value / 100, seed: +$("g-jseed").value }); };
$("g-undo").onclick = () => api("undo"); $("g-redo").onclick = () => api("redo");
$("g-load").onclick = () => { api("op", { op: "preset", key: $("g-preset").value, W: +$("g-count-n").value, seed: +$("g-pseed").value }).then(() => setTimeout(() => xview.fitDomain(), 50)); };
$("g-count").addEventListener("input", (e) => { $("g-count-n").value = e.target.value; });
$("g-count").addEventListener("change", (e) => { api("op", { op: "set_count", W: +e.target.value }); });
$("g-count-n").addEventListener("change", (e) => { api("op", { op: "set_count", W: +e.target.value }); });
$("a-rapply").onclick = () => api("readout_init", { mode: $("a-rinit").value, seed: +$("a-rseed").value });
$("a-rinit").onchange = () => { if (view && view.mode === "idle") api("readout_init", { mode: $("a-rinit").value, seed: +$("a-rseed").value }); };

// keyboard
document.addEventListener("keydown", (e) => {
  const tag = (document.activeElement && document.activeElement.tagName) || "";
  if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) && document.activeElement.type !== "range" && document.activeElement.type !== "checkbox") return;
  if (e.code === "Space") { e.preventDefault(); play(); }
  else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") { e.preventDefault(); api(e.shiftKey ? "redo" : "undo"); }
  else if (view && view.run && (e.key === "ArrowRight" || e.key === "ArrowLeft")) { e.preventDefault(); seek(view.run.frame + (e.key === "ArrowRight" ? 1 : -1)); }
});

// ------------------------------------------------------------------ saved runs
let runsDirty = false;
async function loadRuns() {
  const runs = await (await fetch("/api/runs")).json();
  const box = $("runs"); box.innerHTML = "";
  if (!runs.length) { box.innerHTML = '<div class="dim">no recordings yet</div>'; return; }
  const curId = view && view.run ? view.run.id : null;
  for (const r of runs) {
    const d = document.createElement("div"); d.className = "runitem" + (r.id === curId ? " cur" : "");
    const fin = r.final || {};
    d.innerHTML = `<div class="nm"></div><div class="dim small">${r.created ? r.created.replace("T", " ") : ""} &middot; W=${r.W} &middot; step ${r.step} &middot; ${r.n_frames} frames${r.n_interventions ? ` &middot; ${r.n_interventions} edits` : ""}</div>
      <div class="dim small">${r.target || ""} &middot; relL2 Adam ${fmtE(fin.rel_l2)} / lstsq ${fmtE(fin.ls_rel_l2)}${r.parent ? ` &middot; fork of '${r.parent.name}'@${r.parent.step}` : ""}</div>
      <div class="row"><button class="small b-load">replay</button><button class="small b-ren">rename</button><button class="small b-del">delete</button></div>`;
    d.querySelector(".nm").textContent = r.name;
    d.querySelector(".b-load").onclick = () => { api("load", { id: r.id }).then(() => { setTimeout(() => { xview.fitDomain(); loadRuns(); }, 60); }); };
    d.querySelector(".b-ren").onclick = () => {
      const nm = d.querySelector(".nm"); const inp = document.createElement("input"); inp.value = r.name; inp.style.width = "90%"; nm.replaceWith(inp); inp.focus();
      const done = () => api("rename", { id: r.id, name: inp.value.trim() || r.name }).then(loadRuns);
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter") done(); if (e.key === "Escape") loadRuns(); }); inp.addEventListener("blur", done);
    };
    const del = d.querySelector(".b-del");
    del.onclick = () => { if (del.dataset.armed) api("delete", { id: r.id }).then(loadRuns); else { del.dataset.armed = 1; del.textContent = "confirm delete"; del.style.color = COL.red; setTimeout(() => { delete del.dataset.armed; del.textContent = "delete"; del.style.color = ""; }, 3000); } };
    box.appendChild(d);
  }
}
$("r-refresh").onclick = loadRuns;

// full reset: two clicks, like delete
$("full-reset").onclick = () => {
  const bt = $("full-reset");
  if (!bt.dataset.armed) { bt.dataset.armed = 1; bt.textContent = "click again to reset everything"; bt.style.color = COL.red; setTimeout(() => { delete bt.dataset.armed; bt.textContent = "Full reset"; bt.style.color = ""; }, 3000); return; }
  delete bt.dataset.armed; bt.textContent = "Full reset"; bt.style.color = "";
  for (const p of Object.values(P)) p.yAuto = true;
  ["v-glog", "v-deriv", "v-ideal"].forEach((id) => ($(id).checked = true));
  ["v-asym", "v-train"].forEach((id) => ($(id).checked = false));
  $("r-name").value = ""; setReadoutView("ls");
  api("reset").then(() => { setTimeout(() => xview.fitDomain(), 50); loadRuns(); });
};

// ------------------------------------------------------------------ boot
async function boot() {
  presetsCatalog = await (await fetch("/api/presets")).json();
  const sel = $("g-preset"); let grp = null, og = null;
  for (const p of presetsCatalog) { if (p.group !== grp) { og = document.createElement("optgroup"); og.label = grp = p.group; sel.appendChild(og); } const o = document.createElement("option"); o.value = p.key; o.textContent = p.label; o.title = p.description + (p.target ? `  [sets target: ${p.target}]` : ""); og.appendChild(o); }
  sel.onchange = () => { const p = presetsCatalog.find((q) => q.key === sel.value); sel.title = p ? p.description : ""; };
  const TP = { "": "custom", sine: "sin(2*pi*x)", sqrt2_sine: "sqrt(2)*sin(2*pi*x)", cosine: "cos(2*pi*x)", sine_8pi: "sin(8*pi*x)", sine_mixture: "sin(2*pi*x) + 0.5*sin(6*pi*x) + 0.25*sin(14*pi*x)", runge: "1/(1+25*x**2)", tanh_steep: "tanh(20*x)", exp: "exp(x)", poly5: "x**5 - 3*x**3 + x", abs_cubed: "abs(x)**3", spike: "exp(-((x-0.3)/0.03)**2)" };
  for (const [k, e] of Object.entries(TP)) { const o = document.createElement("option"); o.value = k; o.textContent = k || "custom"; if (k) o.dataset.expr = e; $("k-target-preset").appendChild(o); }
  setReadoutView("ls");
  const es = new EventSource("/api/events");
  es.onmessage = (e) => onView(JSON.parse(e.data));
  es.onerror = () => showMessage("connection to the server lost; is app.py running?", true);
  loadRuns();
  setTimeout(() => xview.fitDomain(), 300);
}
boot();

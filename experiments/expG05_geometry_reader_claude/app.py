"""Geometry reader (Claude version) -- interactive geometry editor, live least squares,
recordable Adam training and replay for f(x) = sum_k a_k tanh(g_k (x - c_k)) + b.

Run on the laptop:
    ~/venv/precisionMLPs/bin/python app.py            # then open http://127.0.0.1:8790
Options: --port, --recordings DIR, --no-browser.
Needs only numpy (fp64 throughout). Recordings default to
<repo>/results/checkpoint_G_generalization/expG05_geometry_reader_claude/recordings.

Architecture: one Session holds all state behind a lock. The browser posts commands to
/api/<cmd> and receives views over a server-sent-event stream (/api/events); views are
coalesced, so a slow browser skips frames but the recording never does.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import threading
import time
import webbrowser
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import presets
from engine import (GROUPS, Adam, AdamConfig, Params, Problem, SnapConfig, clean,
                    derivative, jitter_centers, jitter_gamma, local_spacing, loss_and_grads,
                    lr_at, metrics, predict, resample, solve_ls, xavier_readout)

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
REL_RECORDINGS = Path("results/checkpoint_G_generalization/expG05_geometry_reader_claude/recordings")
FRAME_KEYS = ["step", "c", "g", "a", "b", "a_ls", "b_ls", "rank", "cond", "lr", "t",
              "train_mse", "rel_l2", "linf", "ls_train_mse", "ls_rel_l2", "ls_linf"] + \
             [f"{mv}_{k}" for mv in ("m", "v") for k in GROUPS]


def default_recordings_dir() -> Path:
    for parent in HERE.parents:
        if (parent / "src").is_dir() and (parent / "experiments").is_dir() and (parent / "results").is_dir():
            return parent / REL_RECORDINGS
    repo = Path.home() / "my-repos/research/collaborations/precisionMLPs"
    if repo.is_dir():
        return repo / REL_RECORDINGS
    return HERE / "recordings"


def clean_json(o):
    """numpy -> JSON-safe python; non-finite floats become None (JS JSON.parse rejects NaN)."""
    if isinstance(o, np.ndarray):
        if o.dtype.kind == "f" and not np.isfinite(o).all():
            return [float(v) if math.isfinite(v) else None for v in o.tolist()]
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    return o


def rnd(v, sig=7):
    """Round display-only curves to `sig` significant digits (shorter JSON). Never used on
    geometry or readout arrays, which must round-trip exactly through the browser."""
    v = np.asarray(v, dtype=np.float64)
    with np.errstate(all="ignore"):
        mag = np.where(np.isfinite(v) & (v != 0), np.floor(np.log10(np.abs(v))), 0)
        scale = 10.0 ** (sig - 1 - mag)
        out = np.round(v * scale) / scale
    return np.where(np.isfinite(out), out, v)


def changed_indices(old: Params, new: Params):
    return {"c": np.flatnonzero(old.c != new.c).tolist(),
            "g": np.flatnonzero(old.g != new.g).tolist(),
            "a": np.flatnonzero(old.a != new.a).tolist(),
            "b": bool(old.b != new.b)}


def kind_of(ch):
    geo = bool(ch["c"] or ch["g"])
    ro = bool(ch["a"] or ch["b"])
    return "geometry+readout" if geo and ro else "geometry" if geo else "readout" if ro else "none"


# ----------------------------------------------------------------------------
# recording store
# ----------------------------------------------------------------------------
class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run: "Run"):
        d = self.root / run.id
        d.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.array([fr[k] for fr in run.frames]) for k in FRAME_KEYS}
        tmp = d / "frames.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, d / "frames.npz")
        meta = run.meta()
        (d / "meta.tmp.json").write_text(json.dumps(clean_json(meta), indent=1))
        os.replace(d / "meta.tmp.json", d / "meta.json")
        run.last_save = time.time()

    def list(self):
        out = []
        for d in sorted(self.root.iterdir(), reverse=True):
            m = d / "meta.json"
            if d.is_dir() and m.exists() and not d.name.startswith("_"):
                try:
                    meta = json.loads(m.read_text())
                except json.JSONDecodeError:
                    continue
                out.append({k: meta.get(k) for k in ["id", "name", "created", "W", "step", "n_frames",
                                                      "final", "parent", "n_interventions"]}
                           | {"target": meta.get("problem", {}).get("target")})
        return out

    def load(self, rid):
        d = self.root / rid
        meta = json.loads((d / "meta.json").read_text())
        with np.load(d / "frames.npz") as z:
            arrays = {k: z[k] for k in z.files}
        return meta, arrays

    def rename(self, rid, name):
        m = self.root / rid / "meta.json"
        meta = json.loads(m.read_text())
        meta["name"] = name
        m.write_text(json.dumps(meta, indent=1))

    def delete(self, rid):
        trash = self.root / "_trash"
        trash.mkdir(exist_ok=True)
        shutil.move(str(self.root / rid), str(trash / f"{rid}-{int(time.time())}"))


# ----------------------------------------------------------------------------
# a training run
# ----------------------------------------------------------------------------
class Run:
    def __init__(self, name, prob: Problem, P: Params, cfg: AdamConfig, snap: SnapConfig, steps: int,
                 fork=None):
        now = datetime.now()
        self.id = now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"
        self.name = name or f"{now:%m-%d %H:%M:%S}  W={P.W}"
        self.created = now.isoformat(timespec="seconds")
        self.problem = prob.to_dict()
        self.cfg, self.snap = cfg, snap
        self.P = P.copy()
        self.opt = Adam(P.W)
        self.step = 0
        self.parent = None
        if fork is not None:                  # continue from a recorded state, moments included
            self.step = int(fork["step"])
            self.opt.t = int(fork["t"])
            for k in GROUPS:
                self.opt.m[k][:] = fork["m"][k]
                self.opt.v[k][:] = fork["v"][k]
            self.parent = {"run_id": fork["run_id"], "name": fork["name"], "frame": fork["frame"],
                           "step": fork["step"]}
        self.start_step = self.step
        self.total = self.step + int(steps)
        self.snap_i = 0
        self.next_snap = self.step + snap.gap(0)
        self.frames: list[dict] = []
        self.interventions: list[dict] = []
        self.last_save = 0.0

    def schedule_next(self):
        self.snap_i += 1
        self.next_snap = self.step + self.snap.gap(self.snap_i)

    def meta(self):
        last = self.frames[-1] if self.frames else {}
        return {"id": self.id, "name": self.name, "created": self.created, "W": self.P.W,
                "problem": self.problem, "adam": asdict(self.cfg), "snap": asdict(self.snap),
                "start_step": self.start_step, "total": self.total, "step": self.step,
                "n_frames": len(self.frames), "parent": self.parent,
                "interventions": self.interventions, "n_interventions": len(self.interventions),
                "final": {k: last.get(k) for k in ["rel_l2", "linf", "ls_rel_l2", "ls_linf", "train_mse"]}}


# ----------------------------------------------------------------------------
# session
# ----------------------------------------------------------------------------
class Session:
    def __init__(self, store: Store):
        self.store = store
        self.lock = threading.RLock()
        # The view signal has its own lock: waiting on the session lock would starve the
        # event stream while the trainer re-acquires it after every step.
        self.cond = threading.Condition()
        self.version = 0
        self.waiting = 0
        self.gen = 0                            # training-thread generation token
        self.replay_gen = 0                     # replay-playback generation token
        self.resets = 0                         # tells the page to reload every field
        self._defaults()

    def _defaults(self):
        """Every resettable piece of session state (start and Full reset). Locks, the view
        version and the thread generation tokens are deliberately not reset."""
        self.prob = Problem().build()
        self.adam = AdamConfig()
        self.snap = SnapConfig()
        self.steps = 5000
        self.delay_ms = 0
        self.readout_init = {"mode": "zero", "seed": 0}
        self.reset_moments_on_inject = False
        self.clean_halo = "sqrt"
        self.working, _, _ = presets.build("uniform_sqrt", 81, self.prob.domain, self.prob.lambda_star)
        self.staged: Params | None = None     # edited copy of staged_base
        self.staged_base: Params | None = None  # run state when staging began
        self.fork = None
        self.run: Run | None = None
        self.running = False
        self.thread = None
        self.view_frame = None          # None follows the newest frame
        self.replay = None              # {"meta", "arrays", "i"}
        self.replay_playing = False
        self.undo_stack, self.redo_stack = [], []
        self.jitter_counter = 0
        self.message = ""

    # -- locking: request handlers announce themselves so the trainer yields ----
    @contextmanager
    def guard(self):
        self.waiting += 1
        with self.lock:
            self.waiting -= 1
            yield

    def bump(self, msg=None):
        if msg is not None:
            self.message = msg
        with self.cond:
            self.version += 1
            self.cond.notify_all()

    @property
    def mode(self):
        if self.replay is not None:
            return "replay"
        if self.run is not None:
            return "live" if self.running else "paused"
        return "idle"

    # -- staging -------------------------------------------------------------------
    @staticmethod
    def _overlay(onto: Params, base: Params, edited: Params) -> Params:
        """Apply only the entries that differ between base and edited onto `onto`."""
        out = onto.copy()
        ch = changed_indices(base, edited)
        for k in ("c", "g", "a"):
            getattr(out, k)[ch[k]] = getattr(edited, k)[ch[k]]
        if ch["b"]:
            out.b = edited.b
        return out

    # -- frames ------------------------------------------------------------------
    def _frame(self, run: Run, kind=None, changed=None):
        P, prob = run.P, self.prob
        a_ls, b_ls, info = solve_ls(prob.x, prob.y, P.c, P.g, prob.rcond)
        m_ad, _ = metrics(P, prob)
        m_ls, _ = metrics(Params(P.c, P.g, a_ls, b_ls), prob)
        fr = {"step": run.step, "c": P.c.copy(), "g": P.g.copy(), "a": P.a.copy(), "b": P.b,
              "a_ls": a_ls, "b_ls": b_ls, "rank": info["rank"], "cond": info["cond"],
              "lr": lr_at(run.cfg, run.opt.t, run.total), "t": run.opt.t,
              **m_ad, **{f"ls_{k}": v for k, v in m_ls.items()}}
        for k in GROUPS:
            fr[f"m_{k}"] = run.opt.m[k].copy()
            fr[f"v_{k}"] = run.opt.v[k].copy()
        run.frames.append(fr)
        if kind:
            run.interventions.append({"frame": len(run.frames) - 1, "step": run.step, "kind": kind,
                                      "changed": changed})
        return fr

    def _ensure_current_frame(self, run):
        if not run.frames or run.frames[-1]["step"] != run.step:
            self._frame(run)

    # -- training thread -----------------------------------------------------------
    def _train_loop(self, gen):
        try:
            self._train_steps(gen)
        except Exception as e:  # never leave running=True with a dead thread
            with self.guard():
                if gen == self.gen:
                    self.running = False
                    self.bump(f"error: training stopped: {type(e).__name__}: {e}")

    def _train_steps(self, gen):
        while True:
            snapped = False
            with self.lock:
                run = self.run
                if gen != self.gen or not self.running or run is None or self.replay is not None:
                    return
                if run.step >= run.total:
                    self._ensure_current_frame(run)
                    self.running = False
                    self.store.save(run)
                    self.bump(f"finished at step {run.step}")
                    return
                cfg, prob = run.cfg, self.prob
                if cfg.batch and cfg.batch < prob.x.size:
                    idx = np.random.default_rng([cfg.batch_seed, run.step]).choice(prob.x.size, cfg.batch, replace=False)
                    x, y = prob.x[idx], prob.y[idx]
                else:
                    x, y = prob.x, prob.y
                loss, grads = loss_and_grads(x, y, run.P)
                if not math.isfinite(loss):
                    self._halt()
                    self._ensure_current_frame(run)
                    self.store.save(run)
                    self.bump(f"diverged at step {run.step} (non-finite loss); paused -- lower lr or undo")
                    return
                run.opt.step(run.P, grads, cfg, lr_at(cfg, run.opt.t, run.total))
                run.step += 1
                if run.step >= run.next_snap or run.step >= run.total:
                    self._frame(run)
                    run.schedule_next()
                    snapped = True
                    if time.time() - run.last_save > 30:
                        self.store.save(run)
                    self.bump()
            if snapped and self.delay_ms > 0:
                t_end = time.time() + self.delay_ms / 1000
                while time.time() < t_end and self.running and gen == self.gen:
                    time.sleep(min(0.02, max(0.0, t_end - time.time())))
            elif self.waiting:
                time.sleep(0.0005)

    def _start_thread(self):
        self.running = True
        self.gen += 1
        self.thread = threading.Thread(target=self._train_loop, args=(self.gen,), daemon=True)
        self.thread.start()

    def _halt(self):
        """Stop the thread (caller holds the lock; the thread exits at its next step)."""
        self.running = False
        self.gen += 1

    # -- transitions -----------------------------------------------------------------
    def _end_run(self, why):
        """Knob change or Stop: the run ends, its latest state becomes the editable state."""
        run = self.run
        if run is None:
            return
        self._halt()
        self._ensure_current_frame(run)
        self.store.save(run)
        fr = run.frames[-1]
        self.working = run.P.copy()
        self.fork = self._fork_from(run.id, run.name, len(run.frames) - 1, fr)
        self.run, self.staged, self.staged_base, self.view_frame = None, None, None, None
        self.undo_stack.clear(); self.redo_stack.clear()
        self.message = f"run '{run.name}' ended ({why}); saved"

    def _end_replay(self, why):
        rp = self.replay
        if rp is None:
            return
        fr = {k: rp["arrays"][k][rp["i"]] for k in FRAME_KEYS}
        self.working = Params(fr["c"].copy(), fr["g"].copy(), fr["a"].copy(), float(fr["b"]))
        self.fork = self._fork_from(rp["meta"]["id"], rp["meta"]["name"], rp["i"], fr)
        self.replay = None
        self.replay_playing = False
        self.replay_gen += 1
        self.undo_stack.clear(); self.redo_stack.clear()
        self.message = f"replay ended at frame {rp['i']} ({why}); continuing from step {int(fr['step'])}"

    @staticmethod
    def _fork_from(rid, name, i, fr):
        return {"run_id": rid, "name": name, "frame": int(i), "step": int(fr["step"]), "t": int(fr["t"]),
                "W": int(np.size(fr["c"])),
                "m": {k: np.array(fr[f"m_{k}"]) for k in GROUPS},
                "v": {k: np.array(fr[f"v_{k}"]) for k in GROUPS}}

    def _interrupt(self, why):
        """Any knob: ends a run or a replay."""
        if self.replay is not None:
            self._end_replay(why)
        if self.run is not None:
            self._end_run(why)

    # -- commands ----------------------------------------------------------------------
    def cmd_set_problem(self, fields):
        self._interrupt("data/target knob changed")
        d = self.prob.to_dict() | fields
        d["domain"] = tuple(float(v) for v in d["domain"])
        for k in ("n_train", "n_eval", "data_seed"):
            d[k] = int(d[k])
        for k in ("noise", "rcond", "lambda_star"):
            d[k] = float(d[k])
        prob = Problem(**d).build()          # raises on a bad expression, leaving state intact
        self.prob = prob

    def _edit(self, fn, undo=True, reason="edit"):
        if self.replay is not None:
            self._end_replay("geometry/readout edited")
        if self.run is not None:
            if self.view_frame is not None:
                raise ValueError("viewing an older frame: press 'Follow live' (or scrub to the end) before editing")
            prev = None if self.staged is None else (self.staged_base.copy(), self.staged.copy())
            if self.staged is None:
                self.staged_base = self._display_run_params()
                self.staged = self.staged_base.copy()
            base = self.staged
        else:
            prev = self.working.copy()
            base = self.working
        if undo:
            self.undo_stack.append(prev)
            self.redo_stack.clear()
            del self.undo_stack[:-200]
        new = fn(base.copy())
        if self.run is not None:
            self.staged = new
        else:
            self.working = new

    def _display_run_params(self):
        """The run state the user is looking at: the newest frame (== run.P when paused)."""
        fr = self.run.frames[-1]
        return Params(fr["c"].copy(), fr["g"].copy(), fr["a"].copy(), float(fr["b"]))

    def cmd_edit(self, body):
        """Sparse point edits from dragging: {c:{i:v}, g:{i:v}, a:{i:v}, b:v, undo:bool}."""
        vals = [float(v) for key in ("c", "g", "a") for v in (body.get(key) or {}).values()]
        if body.get("b") is not None:
            vals.append(float(body["b"]))
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("non-finite value in edit")

        def fn(P):
            for key in ("c", "g", "a"):
                for i, v in (body.get(key) or {}).items():
                    getattr(P, key)[int(i)] = float(v)
            if body.get("b") is not None:
                P.b = float(body["b"])
            return P
        self._edit(fn, undo=bool(body.get("undo", True)))

    def cmd_op(self, body):
        op, p = body["op"], float(body.get("p", 1.0))
        prob = self.prob
        if op == "clean":
            what = body.get("what", "both")
            self._edit(lambda P: clean(P, p, what, prob.domain, self.clean_halo, prob.lambda_star))
        elif op == "jitter_c":
            self.jitter_counter += 1
            rng = np.random.default_rng([int(body.get("seed", 0)), self.jitter_counter])
            self._edit(lambda P: jitter_centers(P, p, rng))
        elif op == "jitter_g":
            self.jitter_counter += 1
            rng = np.random.default_rng([int(body.get("seed", 0)), self.jitter_counter, 1])
            self._edit(lambda P: jitter_gamma(P, p, rng))
        elif op == "scale_g":
            f = float(body["factor"])
            def fn(P):
                P.g = P.g * f
                return P
            self._edit(fn, undo=bool(body.get("undo", True)))
        elif op == "sort":
            if self.run is not None:
                raise ValueError("sorting would misalign Adam's moments; Stop the run first")

            def fn(P):
                o = np.argsort(P.c, kind="stable")
                return Params(P.c[o], P.g[o], P.a[o], P.b)
            self._edit(fn)
        elif op == "set_count":
            self._interrupt("neuron count changed")
            self.undo_stack.append(self.working.copy())
            self.working = resample(self.working, int(body["W"]))
            if self.fork and self.fork["W"] != self.working.W:
                self.fork = None
        elif op == "preset":
            self._interrupt("preset loaded")
            P, has_readout, target = presets.build(body["key"], int(body.get("W", self.working.W)),
                                                   prob.domain, prob.lambda_star, int(body.get("seed", 0)))
            if target and body.get("set_target", True) and target != prob.target:
                self.prob = Problem(**(prob.to_dict() | {"target": target})).build()
            if not has_readout:
                P.a, P.b = self._init_readout(P)
            self.undo_stack.append(self.working.copy())
            self.working, self.fork = P, None
        else:
            raise ValueError(op)

    def _init_readout(self, P):
        mode, seed = self.readout_init["mode"], int(self.readout_init.get("seed", 0))
        if mode == "xavier":
            return xavier_readout(P.W, seed), 0.0
        if mode == "lstsq":
            a, b, _ = solve_ls(self.prob.x, self.prob.y, P.c, P.g, self.prob.rcond)
            return a, b
        return np.zeros(P.W), 0.0

    def cmd_readout_init(self, body):
        self.readout_init = {"mode": body["mode"], "seed": int(body.get("seed", 0))}
        if body.get("apply", True):
            def fn(P):
                P.a, P.b = self._init_readout(P)
                return P
            self._edit(fn)

    def cmd_undo(self, redo=False):
        src, dst = (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        if not src or self.replay is not None:
            return
        prev = src.pop()
        if self.run is not None:   # entries are (staged_base, staged) pairs or None
            dst.append(None if self.staged is None else (self.staged_base.copy(), self.staged.copy()))
            self.staged_base, self.staged = (None, None) if prev is None else prev
        else:
            dst.append(self.working.copy())
            if prev is not None:
                self.working = prev

    def cmd_inject(self, discard=False):
        run = self.run
        if run is None or self.staged is None:
            return
        if discard:
            self.staged = self.staged_base = None
            self.undo_stack.clear(); self.redo_stack.clear()
            self.message = "staged edits discarded"
            return
        self._ensure_current_frame(run)                 # the 'before' frame
        self._apply_injection(self._overlay(run.P, self.staged_base, self.staged))

    def _apply_injection(self, new: Params):
        run = self.run
        ch = changed_indices(run.P, new)
        run.P = new.copy()
        if self.reset_moments_on_inject:
            run.opt.reset()
        self._frame(run, kind=kind_of(ch), changed=ch)  # the 'after' frame, flagged
        self.staged = self.staged_base = None
        self.view_frame = None
        self.undo_stack.clear(); self.redo_stack.clear()
        self.store.save(run)
        self.message = f"injected {kind_of(ch)} edit at step {run.step}"

    def cmd_inject_ls(self):
        """Replace the Adam readout with the least-squares readout on the displayed geometry."""
        if self.replay is not None:
            self._end_replay("least-squares readout injected")
        if self.run is None:
            P = self.working
            a, b, _ = solve_ls(self.prob.x, self.prob.y, P.c, P.g, self.prob.rcond)
            self._edit(lambda Q: Params(Q.c, Q.g, a.copy(), b))
            return
        run = self.run
        self._ensure_current_frame(run)
        new = run.P.copy() if self.staged is None else self._overlay(run.P, self.staged_base, self.staged)
        new.a, new.b, _ = solve_ls(self.prob.x, self.prob.y, new.c, new.g, self.prob.rcond)
        self._apply_injection(new)

    def cmd_play(self, body):
        """Start, resume or extend. Returns a decision request when edits are staged."""
        if self.replay is not None:
            return {"replay": True}
        if self.run is not None and self.staged is not None:
            return {"needs_decision": True}
        if self.running:
            return {}
        self.adam = AdamConfig(**(asdict(self.adam) | body.get("adam", {})))
        self.snap = SnapConfig(**(asdict(self.snap) | body.get("snap", {})))
        self.steps = int(body.get("steps", self.steps))
        if self.run is None:
            fork = self.fork if (self.fork and self.fork["W"] == self.working.W) else None
            P = self.working.copy()
            if self.adam.gamma_param == "log":
                P = Params(P.c, np.abs(P.g), P.a * np.where(P.g < 0, -1.0, 1.0), P.b)
            self.run = Run(body.get("name", ""), self.prob, P, self.adam, self.snap, self.steps, fork)
            self._frame(self.run)
            self.fork = None
            self.undo_stack.clear(); self.redo_stack.clear()
            self.message = f"recording '{self.run.name}'"
        elif self.run.step >= self.run.total:
            self.run.total += self.steps
            self.message = f"extended to step {self.run.total}"
        else:
            self.message = "resumed"
        self.view_frame = None
        self._start_thread()
        return {}

    def cmd_pause(self):
        if self.run is not None and self.running:
            self._halt()
            self._ensure_current_frame(self.run)
            self.store.save(self.run)
            self.message = f"paused at step {self.run.step}"

    def cmd_stop(self):
        if self.replay is not None:
            self._end_replay("stopped")
        elif self.run is not None:
            self._end_run("stopped")

    def cmd_replay_play(self, on):
        """Server-side replay playback: frames advance at the frame delay, holding 1 s on
        frames where an edit was injected (so the flash and orange points can be seen)."""
        rp = self.replay
        self.replay_gen += 1
        self.replay_playing = bool(on) and rp is not None
        if not self.replay_playing:
            return
        if rp["i"] >= len(rp["arrays"]["step"]) - 1:
            rp["i"] = 0
        gen = self.replay_gen
        inter = {iv["frame"] for iv in rp["meta"]["interventions"]}

        def loop():
            while True:
                hold = 1.0 if self.replay is rp and rp["i"] in inter else 0.0
                time.sleep(max(self.delay_ms, 16) / 1000 + hold)
                with self.guard():
                    if gen != self.replay_gen or self.replay is not rp:
                        return
                    rp["i"] += 1
                    if rp["i"] >= len(rp["arrays"]["step"]) - 1:
                        rp["i"] = len(rp["arrays"]["step"]) - 1
                        self.replay_playing = False
                    self.bump()
                    if not self.replay_playing:
                        return

        threading.Thread(target=loop, daemon=True).start()

    def cmd_seek(self, i):
        self.replay_gen += 1
        self.replay_playing = False
        if self.replay is not None:
            n = len(self.replay["arrays"]["step"])
            self.replay["i"] = int(min(max(i, 0), n - 1))
        elif self.run is not None:
            n = len(self.run.frames)
            i = int(min(max(i, 0), n - 1))
            self.view_frame = None if i == n - 1 else i

    def cmd_load(self, rid):
        self._interrupt("replay loaded")
        meta, arrays = self.store.load(rid)
        pd = meta["problem"]
        pd["domain"] = tuple(pd["domain"])
        self.prob = Problem(**pd).build()
        self.replay = {"meta": meta, "arrays": arrays, "i": 0}
        self.staged = self.staged_base = None
        self.message = f"replay '{meta['name']}' loaded ({len(arrays['step'])} frames)"

    # -- view ------------------------------------------------------------------------------
    def build_view(self):
        """Copy what is displayed under the lock, then compute curves outside it."""
        with self.guard():
            prob = self.prob
            ghost, changed, frame_kind, hist, run_info = None, None, None, None, None
            mode = self.mode
            frame = None
            if mode == "replay":
                rp = self.replay
                A = rp["arrays"]
                i = rp["i"]
                frame = {k: A[k][i] for k in FRAME_KEYS}
                inter = {iv["frame"]: iv for iv in rp["meta"]["interventions"]}
                if i in inter and i > 0:
                    frame_kind = inter[i]["kind"]
                    changed = inter[i]["changed"]
                    ghost = {k: A[k][i - 1] for k in ("c", "g", "a")}
                hist = {k: A[k] for k in ("step", "rel_l2", "ls_rel_l2", "train_mse", "ls_train_mse")}
                meta = rp["meta"]
                run_info = {"id": meta["id"], "name": meta["name"], "frame": i, "n_frames": len(A["step"]),
                            "step": int(A["step"][i]), "total": meta["total"],
                            "interventions": [iv["frame"] for iv in meta["interventions"]],
                            "adam": meta["adam"], "snap": meta["snap"], "parent": meta.get("parent")}
            elif mode in ("live", "paused"):
                run = self.run
                i = len(run.frames) - 1 if self.view_frame is None else self.view_frame
                frame = dict(run.frames[i])
                inter = {iv["frame"]: iv for iv in run.interventions}
                if i in inter and i > 0:
                    frame_kind = inter[i]["kind"]
                    changed = inter[i]["changed"]
                    ghost = {k: run.frames[i - 1][k] for k in ("c", "g", "a")}
                hist = {k: np.array([fr[k] for fr in run.frames])
                        for k in ("step", "rel_l2", "ls_rel_l2", "train_mse", "ls_train_mse")}
                run_info = {"id": run.id, "name": run.name, "frame": i, "n_frames": len(run.frames),
                            "step": int(frame["step"]), "live_step": run.step, "total": run.total,
                            "interventions": [iv["frame"] for iv in run.interventions],
                            "adam": asdict(run.cfg), "snap": asdict(run.snap), "parent": run.parent,
                            "following": self.view_frame is None}
            staged = self.staged if (mode in ("live", "paused") and self.view_frame is None) else None
            a_ls = b_ls = info = None
            if staged is not None:
                P = self._overlay(Params(np.array(frame["c"]), np.array(frame["g"]), np.array(frame["a"]),
                                         float(frame["b"])), self.staged_base, staged)
                ghost = {"c": frame["c"], "g": frame["g"], "a": frame["a"]}
                changed = changed_indices(Params(frame["c"], frame["g"], frame["a"], float(frame["b"])), P)
                frame_kind = None
            elif frame is not None:
                P = Params(np.array(frame["c"]), np.array(frame["g"]), np.array(frame["a"]), float(frame["b"]))
                a_ls, b_ls = np.array(frame["a_ls"]), float(frame["b_ls"])
                info = {"rank": int(frame["rank"]), "cond": float(frame["cond"])}
            else:
                P = self.working.copy()
            settings = {"problem": prob.to_dict(), "adam": asdict(self.adam), "snap": asdict(self.snap),
                        "steps": self.steps, "delay_ms": self.delay_ms, "readout_init": self.readout_init,
                        "reset_moments_on_inject": self.reset_moments_on_inject,
                        "clean_halo": self.clean_halo}
            out = {"version": self.version, "mode": mode, "message": self.message,
                   "replay_playing": self.replay_playing, "resets": self.resets,
                   "has_staged": self.staged is not None, "run": run_info,
                   "fork": None if not self.fork else {k: self.fork[k] for k in ("run_id", "name", "frame", "step", "W")},
                   "settings": settings, "undo": len(self.undo_stack), "redo": len(self.redo_stack),
                   "recordings_dir": str(self.store.root)}
        # heavy part, outside the lock
        if a_ls is None:
            a_ls, b_ls, info = solve_ls(prob.x, prob.y, P.c, P.g, prob.rcond)
        L = Params(P.c, P.g, a_ls, b_ls)
        m_ad, f_ad = metrics(P, prob)
        m_ls, f_ls = metrics(L, prob)
        h = local_spacing(P.c)
        lo = min(P.c.min(), prob.domain[0]); hi = max(P.c.max(), prob.domain[1])
        xd = np.linspace(lo, hi, 1500)
        o = np.argsort(P.c)
        hd = np.interp(xd, P.c[o], h[o]) if P.W > 1 else np.full(xd.size, h[0])
        with np.errstate(all="ignore"):
            deriv = derivative(prob.f, xd) * hd / 2
        stride = max(1, prob.x.size // 1500)
        out.update({
            "geom": {"c": P.c, "g": P.g, "h": h, "lambda_star": prob.lambda_star},
            "readout": {"a": P.a, "b": P.b, "a_ls": a_ls, "b_ls": b_ls},
            "ghost": ghost, "changed": changed, "frame_kind": frame_kind,
            "curves": {"x": rnd(prob.xe, 9), "f": rnd(prob.fe), "fhat": rnd(f_ad), "fhat_ls": rnd(f_ls),
                       "r": rnd(f_ad - prob.fe, 4), "r_ls": rnd(f_ls - prob.fe, 4)},
            "deriv": {"x": rnd(xd, 6), "y": rnd(deriv, 5)},
            "train": {"x": rnd(prob.x[::stride], 6), "y": rnd(prob.y[::stride], 6), "n": int(prob.x.size)},
            "metrics": {"adam": m_ad, "ls": m_ls, **info, "W": P.W,
                        "median_lambda": float(np.median(np.abs(P.g) * h)),
                        "mean_lambda": float(np.mean(np.abs(P.g) * h))},
            "history": hist,
        })
        return clean_json(out)

    def dispatch(self, cmd, body):
        with self.guard():
            res = {}
            if cmd == "set_problem":
                self.cmd_set_problem(body)
            elif cmd == "edit":
                self.cmd_edit(body)
            elif cmd == "op":
                self.cmd_op(body)
            elif cmd == "readout_init":
                self.cmd_readout_init(body)
            elif cmd == "undo":
                self.cmd_undo()
            elif cmd == "redo":
                self.cmd_undo(redo=True)
            elif cmd == "inject":
                self.cmd_inject()
            elif cmd == "discard":
                self.cmd_inject(discard=True)
            elif cmd == "inject_ls":
                self.cmd_inject_ls()
            elif cmd == "play":
                res = self.cmd_play(body)
            elif cmd == "pause":
                self.cmd_pause()
            elif cmd == "stop":
                self.cmd_stop()
            elif cmd == "replay_play":
                self.cmd_replay_play(body.get("on", True))
            elif cmd == "seek":
                self.cmd_seek(int(body["i"]))
            elif cmd == "follow":
                self.view_frame = None
            elif cmd == "set":
                for k in ("delay_ms", "reset_moments_on_inject", "clean_halo", "steps"):
                    if k in body:
                        setattr(self, k, type(getattr(self, k))(body[k]))
                if "adam" in body and self.run is None:
                    self.adam = AdamConfig(**(asdict(self.adam) | body["adam"]))
                if "snap" in body and self.run is None:
                    self.snap = SnapConfig(**(asdict(self.snap) | body["snap"]))
            elif cmd == "reset":
                self._interrupt("full reset")
                self._halt()
                self.replay_gen += 1
                self._defaults()
                self.resets += 1
                self.message = "full reset (recordings are kept)"
            elif cmd == "load":
                self.cmd_load(body["id"])
            elif cmd == "rename":
                self.store.rename(body["id"], body["name"])
                if self.run is not None and self.run.id == body["id"]:
                    self.run.name = body["name"]
                if self.replay is not None and self.replay["meta"]["id"] == body["id"]:
                    self.replay["meta"]["name"] = body["name"]
                res = {"runs": self.store.list()}
            elif cmd == "delete":
                if self.replay is not None and self.replay["meta"]["id"] == body["id"]:
                    self._end_replay("deleted")
                if self.run is None or self.run.id != body["id"]:
                    self.store.delete(body["id"])
                res = {"runs": self.store.list()}
            else:
                raise ValueError(f"unknown command {cmd}")
            self.bump()
            return res


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
def make_handler(session: Session):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def setup(self):
            super().setup()
            # small SSE frames must not wait on Nagle + delayed ACK (~100 ms per view otherwise)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/events":
                return self._events()
            if path == "/api/runs":
                return self._send(200, json.dumps(clean_json(session.store.list())).encode())
            if path == "/api/presets":
                return self._send(200, json.dumps(presets.catalog()).encode())
            if path == "/api/view":
                return self._send(200, json.dumps(session.build_view()).encode())
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (STATIC / name).resolve()
            if STATIC in f.parents and f.is_file():
                ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}.get(f.suffix, "text/plain")
                return self._send(200, f.read_bytes(), ctype + "; charset=utf-8")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?")[0]
            if not path.startswith("/api/"):
                return self._send(404, b"{}")
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                res = session.dispatch(path[len("/api/"):], body)
                self._send(200, json.dumps(clean_json({"ok": True, **(res or {})})).encode())
            except Exception as e:  # report to the page instead of dying
                with session.guard():
                    session.bump(f"error: {type(e).__name__}: {e}")
                self._send(200, json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}).encode())

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last = -1
            t_last = 0.0
            try:
                while True:
                    with session.cond:
                        session.cond.wait_for(lambda: session.version != last, timeout=10)
                        v = session.version
                    if v == last:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    last = v
                    try:
                        payload = json.dumps(session.build_view())
                    except Exception as e:  # never kill the stream
                        payload = json.dumps({"error": f"{type(e).__name__}: {e}"})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    # coalesce to <= ~60 views/s, sleeping only when views really come that fast
                    # (an unconditional sleep gets stretched by macOS timer coalescing)
                    gap = time.monotonic() - t_last
                    t_last = time.monotonic()
                    if gap < 0.016:
                        time.sleep(0.016 - gap)
            except (BrokenPipeError, ConnectionResetError):
                return

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--recordings", type=Path, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    store = Store(args.recordings or default_recordings_dir())
    session = Session(store)
    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):  # browser tabs closing mid-stream are normal
            pass

    srv = Server(("127.0.0.1", args.port), make_handler(session))
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"geometry reader (Claude) at {url}\nrecordings: {store.root}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        with session.guard():
            if session.run is not None:
                session.cmd_pause()


if __name__ == "__main__":
    main()

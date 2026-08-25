"""expD09 shared: paths, the 9 frozen-Phi problems, metric plumbing.

The object under test is the FROZEN QI feature system

    A = [Phi, 1],   Phi_ik = tanh(gamma (x_i - c_k)),   gamma = lambda*/h,

with uniform centers + halo (repo standard, lambda* = 0.25). Nothing about
the geometry is learned here: the whole experiment is the linear readout
solve  min_a ||A a - y||_2  on a fixed A, run by block-coordinate methods.

Problem grid (Sam, 2026-07-28): 3 targets x 3 widths.
  targets  sine (low freq) | sine_8pi (high freq) | runge (peaked)
  widths   N = 64, 128, 256   ->   d = 206, 270, 462 columns
All three targets have SVD floors at 1e-13..1e-15, so a stall is the
solver's fault and not the target's.

Rows n = 4d equispaced on [-1,1] (expD07's ratio-4 cell); eval on 4001
equispaced points, misaligned with every train grid.

Two metrics are recorded at every sweep and both are needed:
  train rel L2 = ||A a - y|| / ||y||        the objective being minimized
  eval  rel L2 = ||A_ev a - y_ev|| / ||y_ev||   what we actually care about
Their GAP is the null-space drift: null(A_train) is not null for A_eval, so
a method that wanders in the train null space keeps train flat while eval
climbs. `null_norm` records that component directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEMS_DIR = REPO_ROOT / "data" / "expD09_problems"
RESULTS_DIR = (
    REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD09_2nd_order_regime"
)
FIG_DIR = RESULTS_DIR / "figures"
MANIFEST_PATH = PROBLEMS_DIR / "manifest.json"
RUNS_PATH = RESULTS_DIR / "runs.jsonl"

LAMBDA_STAR = 0.25
RCOND = 1e-15          # truncated-SVD cutoff, repo standard
N_EVAL = 4001          # prime, misaligned with any train grid
ROW_RATIO = 4          # n = ROW_RATIO * d
TARGETS = ["sine", "sine_8pi", "runge"]
WIDTHS = [64, 128, 256]

# Fixed axes for every figure in this experiment: one log decade scale so
# panels are comparable across methods, widths and targets.
YLIM = (1e-16, 1e1)


def build_phi_aug(x: np.ndarray, gamma: float, centers: np.ndarray) -> np.ndarray:
    """[tanh(gamma (x - c_k)), 1] -- same formula as src.construction.readout."""
    phi = np.tanh(gamma * (x[:, None] - centers[None, :]))
    return np.hstack([phi, np.ones((x.size, 1))])


def geometry(N: int) -> tuple[np.ndarray, float]:
    """Repo-standard uniform centers + halo and the matched bandwidth."""
    from src.construction.qi_mpmath import default_halo

    h = 2.0 / N
    halo = default_halo(N, lambda_star=LAMBDA_STAR)
    centers = -1.0 + np.arange(-halo, N + halo + 1, dtype=np.float64) * h
    return centers, LAMBDA_STAR / h


def problem_id(fn: str, N: int) -> str:
    return f"phi_{fn}_N{N}"


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())


def load_problem(entry: dict) -> dict:
    """Rebuild the frozen system from its stored grid (exact, not re-solved).

    Returns A, y (train), A_ev, y_ev (eval), and V_null -- an orthonormal
    basis of the numerical null space of A, used for the drift diagnostic.
    """
    z = np.load(PROBLEMS_DIR / entry["path"])
    gamma, centers = float(z["gamma"]), z["centers"]
    return {
        "A": build_phi_aug(z["x_tr"], gamma, centers), "y": z["y_tr"],
        "A_ev": build_phi_aug(z["x_ev"], gamma, centers), "y_ev": z["y_ev"],
        "V_null": z["V_null"], "a_svd": z["a_svd"],
    }


class Metrics:
    """Per-sweep recorder. Cheap: two matvecs and one d x (d-r) product."""

    def __init__(self, prob: dict):
        self.A, self.y = prob["A"], prob["y"]
        self.A_ev, self.y_ev = prob["A_ev"], prob["y_ev"]
        self.V_null = prob["V_null"]
        self.y_norm = float(np.linalg.norm(self.y))
        self.ye_norm = float(np.linalg.norm(self.y_ev))
        self.rows: dict[str, list[float]] = {
            k: [] for k in
            ("train_rel", "eval_rel", "eval_linf", "a_norm", "null_norm")
        }

    def record(self, a: np.ndarray) -> None:
        r_ev = self.A_ev @ a - self.y_ev
        self.rows["train_rel"].append(
            float(np.linalg.norm(self.A @ a - self.y)) / self.y_norm)
        self.rows["eval_rel"].append(float(np.linalg.norm(r_ev)) / self.ye_norm)
        self.rows["eval_linf"].append(float(np.abs(r_ev).max()))
        self.rows["a_norm"].append(float(np.linalg.norm(a)))
        self.rows["null_norm"].append(
            float(np.linalg.norm(self.V_null.T @ a)) if self.V_null.size else 0.0)


def load_runs() -> list[dict]:
    if not RUNS_PATH.exists():
        return []
    return [json.loads(ln) for ln in RUNS_PATH.read_text().splitlines() if ln.strip()]


def append_run(rec: dict) -> None:
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")

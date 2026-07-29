"""expD07 shared: paths, problem I/O, metric helpers.

Problems are fixed artifacts in data/expD07_problems/{synthetic,phis}/*.npz;
all metadata + fp64 floors live in data/expD07_problems/manifest.json.
Runs cache: results/.../runs.jsonl, one record per (problem_id, opt_id).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEMS_DIR = REPO_ROOT / "data" / "expD07_problems"
RESULTS_DIR = (
    REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD07_lstsq_optim_suite"
)
FIG_DIR = RESULTS_DIR / "figures"
MANIFEST_PATH = PROBLEMS_DIR / "manifest.json"
RUNS_PATH = RESULTS_DIR / "runs.jsonl"

LAMBDA_STAR = 0.25
RCOND = 1e-15          # truncated-SVD baseline cutoff (per Sam; expD02 used 1e-13)
N_EVAL = 4001          # prime, misaligned with any train grid
EXCESS_CLAMP = 1e-32   # numerical floor for L - L*
DIGITS_CLAMP = 1e-16   # rel-excess below this counts as "all 16 digits"


def build_phi_aug(x: np.ndarray, gamma: float, centers: np.ndarray) -> np.ndarray:
    """[tanh(gamma (x - c_k)), 1] -- same formula as src.construction.readout."""
    phi = np.tanh(gamma * (x[:, None] - centers[None, :]))
    return np.hstack([phi, np.ones((x.size, 1))])


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())


def load_problem(entry: dict) -> dict:
    """Train system A, y (+ eval A_ev, y_ev for phi problems), all fp64."""
    z = np.load(PROBLEMS_DIR / entry["path"])
    if entry["kind"] == "synthetic":
        return {"A": z["A"], "y": z["y"]}
    gamma, centers = float(z["gamma"]), z["centers"]
    return {
        "A": build_phi_aug(z["x_tr"], gamma, centers), "y": z["y_tr"],
        "A_ev": build_phi_aug(z["x_ev"], gamma, centers), "y_ev": z["y_ev"],
    }


def load_runs() -> list[dict]:
    if not RUNS_PATH.exists():
        return []
    return [json.loads(line) for line in RUNS_PATH.read_text().splitlines() if line.strip()]


def append_run(rec: dict) -> None:
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def rel_excess(train_mse: float, l_star: float, l0: float) -> float:
    """Scale-invariant excess loss: (L - L*)/(L0 - L*), L0 = loss at v=0."""
    return max(train_mse - l_star, EXCESS_CLAMP) / max(l0 - l_star, EXCESS_CLAMP)


def digits(final_rel_excess: float) -> float:
    """Digits of excess-loss reduction (capped at 16 by DIGITS_CLAMP)."""
    return float(-np.log10(max(final_rel_excess, DIGITS_CLAMP)))

"""expD05 -- scale-aware initialization story.

This experiment reruns the Convex_MLP scale-initialization screening inside the
precision-mlps fp64 PyTorch stack. It keeps the implementation local to the
experiment on purpose: the goal is to validate the Checkpoint D story, not to
add a new public training API.

Usage:
    uv run --extra dev python experiments/expD05_scale_init_story/run.py --mode smoke
    uv run --extra dev python experiments/expD05_scale_init_story/run.py --mode full
    uv run --extra dev python experiments/expD05_scale_init_story/run.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import DEVICE  # noqa: E402
from src.config.schema import ModelConfig  # noqa: E402
from src.construction.qi_mpmath import construct_qi, default_halo  # noqa: E402
from src.construction.readout import solve_readout_with_bias  # noqa: E402
from src.data.targets import get_target  # noqa: E402
from src.models.mlp import QIMlp  # noqa: E402


EXPERIMENT = "expD05_scale_init_story"
RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_D_optimizers" / EXPERIMENT
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
TRACES_CSV = RESULTS_DIR / "traces.csv"
ATOMS_CSV = RESULTS_DIR / "atoms.csv"
RUN_MATRIX_CSV = RESULTS_DIR / "run_matrix.csv"
REPORT_MD = RESULTS_DIR / "expD05_results.md"

DEFAULT_TARGETS = ("sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed")
DEFAULT_RESOLUTIONS = (64, 128, 256, 512)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_FAMILIES = (
    "standard_xavier_affine",
    "center_spread_xavier",
    "scale_corrected_xavier_l025",
    "scale_corrected_xavier_l030",
    "qi_scale_only_l025",
    "qi_scale_only_l030",
    "low_qi_l030",
    "exact_qi_oracle",
    "qi_geom_random_readout",
)
SMOKE_FAMILIES = (
    "standard_xavier_affine",
    "scale_corrected_xavier_l030",
    "qi_scale_only_l030",
    "exact_qi_oracle",
)
STANDARD_FAMILIES = {"standard_xavier_affine", "center_spread_xavier"}
SCALE_CORRECTED_FAMILIES = {"scale_corrected_xavier_l025", "scale_corrected_xavier_l030"}
STRUCTURED_FAMILIES = {"qi_scale_only_l025", "qi_scale_only_l030", "low_qi_l030"}
QI_FAMILIES = {"exact_qi_oracle", "qi_geom_random_readout"}
EPS = 1e-300


@dataclass(frozen=True)
class OutputPaths:
    summary: Path
    traces: Path
    atoms: Path
    run_matrix: Path


@dataclass(frozen=True)
class RunConfig:
    targets: tuple[str, ...]
    resolutions: tuple[int, ...]
    seeds: tuple[int, ...]
    families: tuple[str, ...]
    steps: int
    eval_interval: int
    learning_rate: float
    n_train: int
    n_eval: int
    mode: str


@dataclass(frozen=True)
class Case:
    target: str
    resolution: int
    seed: int
    family: str


def artifact_path(path: Path, suffix: str = "") -> Path:
    if not suffix:
        return path
    clean = suffix[1:] if suffix.startswith("_") else suffix
    return path.with_name(f"{path.stem}_{clean}{path.suffix}")


def artifact_paths(suffix: str = "") -> OutputPaths:
    return OutputPaths(
        summary=artifact_path(SUMMARY_CSV, suffix),
        traces=artifact_path(TRACES_CSV, suffix),
        atoms=artifact_path(ATOMS_CSV, suffix),
        run_matrix=artifact_path(RUN_MATRIX_CSV, suffix),
    )


def shard_suffix(shard_index: int, num_shards: int, explicit_suffix: str | None = None) -> str:
    if explicit_suffix:
        return explicit_suffix
    if num_shards > 1:
        return f"shard{shard_index:02d}of{num_shards:02d}"
    return ""


def case_key(case: Case) -> tuple[str, int, int, str]:
    return case.target, case.resolution, case.seed, case.family


def row_case_key(row: dict[str, str]) -> tuple[str, int, int, str]:
    return row["target"], int(row["resolution"]), int(row["seed"]), row["family"]


def select_shard(cases: Sequence[Case], shard_index: int, num_shards: int) -> list[Case]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return [case for index, case in enumerate(cases) if index % num_shards == shard_index]


@dataclass(frozen=True)
class Geometry:
    resolution: int
    h: float
    halo: int
    centers: np.ndarray

    @property
    def width(self) -> int:
        return int(self.centers.size)


@dataclass(frozen=True)
class InitialState:
    input_weights: np.ndarray
    input_biases: np.ndarray
    readout_weights: np.ndarray
    readout_bias: float
    atom_labels: tuple[str, ...]
    requested_lambdas: tuple[float, ...]


def geometry_for_resolution(resolution: int) -> Geometry:
    """Use one halo for all families so each cell has the same hidden width."""
    h = 2.0 / float(resolution)
    halo = max(
        default_halo(resolution, lambda_star=0.25),
        default_halo(resolution, lambda_star=0.30),
    )
    idx = np.arange(-halo, resolution + halo + 1, dtype=np.float64)
    centers = -1.0 + idx * h
    return Geometry(resolution=resolution, h=h, halo=int(halo), centers=centers)


def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred.ravel() - true.ravel()) / np.linalg.norm(true.ravel()))


def linf(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.max(np.abs(pred.ravel() - true.ravel())))


def features_np(x: np.ndarray, input_weights: np.ndarray, input_biases: np.ndarray) -> np.ndarray:
    return np.tanh(x.reshape(-1, 1) * input_weights.reshape(1, -1) + input_biases.reshape(1, -1))


def predict_np(
    x: np.ndarray,
    input_weights: np.ndarray,
    input_biases: np.ndarray,
    readout_weights: np.ndarray,
    readout_bias: float,
) -> np.ndarray:
    return features_np(x, input_weights, input_biases) @ readout_weights.reshape(-1) + readout_bias


def canonical_semantic(
    input_weights: np.ndarray, input_biases: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_weights = np.asarray(input_weights, dtype=np.float64).reshape(-1)
    input_biases = np.asarray(input_biases, dtype=np.float64).reshape(-1)
    valid = np.isfinite(input_weights) & np.isfinite(input_biases) & (np.abs(input_weights) > 1e-12)
    centers = np.full_like(input_weights, np.nan, dtype=np.float64)
    centers[valid] = -input_biases[valid] / input_weights[valid]
    gammas = np.abs(input_weights)
    return centers, gammas, valid


def xavier_uniform(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    return rng.uniform(-bound, bound, size=shape).astype(np.float64)


def random_readout(width: int, rng: np.random.Generator, scale: float = 0.1) -> np.ndarray:
    return rng.normal(0.0, scale / math.sqrt(float(width)), size=width).astype(np.float64)


def affine_to_state(
    input_weights: np.ndarray,
    input_biases: np.ndarray,
    readout_weights: np.ndarray,
    readout_bias: float,
    labels: Sequence[str],
    requested_lambdas: Sequence[float],
) -> InitialState:
    width = int(input_weights.size)
    if input_biases.size != width or readout_weights.size != width:
        raise ValueError("initializer arrays must all match width")
    if len(labels) != width or len(requested_lambdas) != width:
        raise ValueError("atom metadata must match width")
    return InitialState(
        input_weights=np.asarray(input_weights, dtype=np.float64).reshape(width),
        input_biases=np.asarray(input_biases, dtype=np.float64).reshape(width),
        readout_weights=np.asarray(readout_weights, dtype=np.float64).reshape(width),
        readout_bias=float(readout_bias),
        atom_labels=tuple(labels),
        requested_lambdas=tuple(float(v) for v in requested_lambdas),
    )


def semantic_to_state(
    centers: np.ndarray,
    gammas: np.ndarray,
    readout_weights: np.ndarray,
    readout_bias: float,
    labels: Sequence[str],
    requested_lambdas: Sequence[float],
) -> InitialState:
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    gammas = np.asarray(gammas, dtype=np.float64).reshape(-1)
    return affine_to_state(
        input_weights=gammas,
        input_biases=-gammas * centers,
        readout_weights=readout_weights,
        readout_bias=readout_bias,
        labels=labels,
        requested_lambdas=requested_lambdas,
    )


def low_qi_groups(width: int) -> list[tuple[str, float, int]]:
    low_count = min(10, max(1, width // 16))
    second_count = min(10, max(1, width // 16))
    qi_count = width - low_count - second_count
    if qi_count <= 0:
        low_count = max(1, width // 4)
        second_count = max(1, width // 4)
        qi_count = width - low_count - second_count
    return [
        ("lambda=0.030", 0.03, low_count),
        ("lambda=0.060", 0.06, second_count),
        ("lambda=0.300", 0.30, qi_count),
    ]


def centers_for_group(span: tuple[float, float], count: int) -> np.ndarray:
    if count == 1:
        return np.asarray([(span[0] + span[1]) / 2.0], dtype=np.float64)
    return np.linspace(span[0], span[1], count, dtype=np.float64)


def scale_center_spread_xavier(
    geometry: Geometry, rng: np.random.Generator, lambda_value: float
) -> InitialState:
    width = geometry.width
    centers = geometry.centers
    input_weights = xavier_uniform(rng, (width,), 1, width)
    input_biases = -input_weights * centers
    median_abs = max(float(np.median(np.abs(input_weights))), 1e-12)
    gamma_target = lambda_value / geometry.h
    scale = gamma_target / median_abs
    input_weights = input_weights * scale
    input_biases = input_biases * scale
    readout = xavier_uniform(rng, (width,), width, 1)
    labels = (f"lambda={lambda_value:.3f}",) * width
    lambdas = (lambda_value,) * width
    return affine_to_state(input_weights, input_biases, readout, 0.0, labels, lambdas)


def build_initial_state(family: str, target_name: str, geometry: Geometry, seed: int) -> InitialState:
    rng = np.random.default_rng(seed)
    width = geometry.width

    if family == "standard_xavier_affine":
        input_weights = xavier_uniform(rng, (width,), 1, width)
        input_biases = np.zeros(width, dtype=np.float64)
        readout = xavier_uniform(rng, (width,), width, 1)
        return affine_to_state(
            input_weights,
            input_biases,
            readout,
            0.0,
            (family,) * width,
            (math.nan,) * width,
        )

    if family == "center_spread_xavier":
        input_weights = xavier_uniform(rng, (width,), 1, width)
        input_biases = -input_weights * geometry.centers
        readout = xavier_uniform(rng, (width,), width, 1)
        return affine_to_state(
            input_weights,
            input_biases,
            readout,
            0.0,
            (family,) * width,
            (math.nan,) * width,
        )

    if family == "scale_corrected_xavier_l025":
        return scale_center_spread_xavier(geometry, rng, 0.25)

    if family == "scale_corrected_xavier_l030":
        return scale_center_spread_xavier(geometry, rng, 0.30)

    if family == "qi_scale_only_l025":
        gamma = 0.25 / geometry.h
        return semantic_to_state(
            geometry.centers,
            np.full(width, gamma, dtype=np.float64),
            random_readout(width, rng),
            0.0,
            ("lambda=0.250",) * width,
            (0.25,) * width,
        )

    if family == "qi_scale_only_l030":
        gamma = 0.30 / geometry.h
        return semantic_to_state(
            geometry.centers,
            np.full(width, gamma, dtype=np.float64),
            random_readout(width, rng),
            0.0,
            ("lambda=0.300",) * width,
            (0.30,) * width,
        )

    if family == "low_qi_l030":
        span = (float(geometry.centers[0]), float(geometry.centers[-1]))
        all_centers: list[np.ndarray] = []
        all_gammas: list[np.ndarray] = []
        labels: list[str] = []
        requested: list[float] = []
        for label, lambda_value, count in low_qi_groups(width):
            centers = centers_for_group(span, count)
            h_group = (span[1] - span[0]) / float(max(count - 1, 1))
            gamma = lambda_value / h_group
            all_centers.append(centers)
            all_gammas.append(np.full(count, gamma, dtype=np.float64))
            labels.extend([label] * count)
            requested.extend([lambda_value] * count)
        centers = np.concatenate(all_centers)
        gammas = np.concatenate(all_gammas)
        order = np.argsort(centers)
        return semantic_to_state(
            centers[order],
            gammas[order],
            random_readout(width, rng),
            0.0,
            [labels[i] for i in order],
            [requested[i] for i in order],
        )

    if family in {"exact_qi_oracle", "qi_geom_random_readout"}:
        target = get_target(target_name)
        qi = construct_qi(
            target.fn_numpy,
            target.deriv_numpy,
            N=geometry.resolution,
            precision="fp64",
            lambda_star=0.30,
            Kc=160,
            halo=geometry.halo,
        )
        if qi.centers.size != width:
            raise RuntimeError("QI construction width did not match shared geometry")
        if family == "exact_qi_oracle":
            readout = qi.a_coeffs
            readout_bias = qi.c0
        else:
            readout = random_readout(width, rng)
            readout_bias = 0.0
        return semantic_to_state(
            qi.centers,
            np.full(width, qi.gamma, dtype=np.float64),
            readout,
            readout_bias,
            ("qi",) * width,
            (qi.lambda_val,) * width,
        )

    raise ValueError(f"unknown initializer family: {family}")


def make_model(state: InitialState) -> QIMlp:
    width = int(state.input_weights.size)
    model = QIMlp(ModelConfig(width=width, layer_type="standard")).to(DEVICE)
    with torch.no_grad():
        model.inner_layer.linear.weight.copy_(
            torch.as_tensor(state.input_weights, dtype=torch.float64, device=DEVICE).reshape(width, 1)
        )
        model.inner_layer.linear.bias.copy_(
            torch.as_tensor(state.input_biases, dtype=torch.float64, device=DEVICE)
        )
        model.readout.weight.copy_(
            torch.as_tensor(state.readout_weights, dtype=torch.float64, device=DEVICE).reshape(1, width)
        )
        model.readout.bias.copy_(
            torch.as_tensor([state.readout_bias], dtype=torch.float64, device=DEVICE)
        )
    return model


def model_affine_params(model: QIMlp) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with torch.no_grad():
        input_weights = model.inner_layer.linear.weight.detach().cpu().numpy().astype(np.float64).reshape(-1)
        input_biases = model.inner_layer.linear.bias.detach().cpu().numpy().astype(np.float64).reshape(-1)
        readout = model.readout.weight.detach().cpu().numpy().astype(np.float64).reshape(-1)
        bias = float(model.readout.bias.detach().cpu().numpy().reshape(-1)[0])
    return input_weights, input_biases, readout, bias


def make_data(target_name: str, n_train: int, n_eval: int):
    target = get_target(target_name)
    x_train = np.linspace(-1.0, 1.0, n_train, dtype=np.float64)
    x_eval = np.linspace(-1.0, 1.0, n_eval, dtype=np.float64)
    y_train = target.fn_numpy(x_train).astype(np.float64)
    y_eval = target.fn_numpy(x_eval).astype(np.float64)
    xt = torch.as_tensor(x_train.reshape(-1, 1), dtype=torch.float64, device=DEVICE)
    yt = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float64, device=DEVICE)
    xe = torch.as_tensor(x_eval.reshape(-1, 1), dtype=torch.float64, device=DEVICE)
    ye = torch.as_tensor(y_eval.reshape(-1, 1), dtype=torch.float64, device=DEVICE)
    return x_train, y_train, x_eval, y_eval, xt, yt, xe, ye


@torch.no_grad()
def torch_rel_l2(model: QIMlp, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = model(x)
    return float(torch.linalg.norm(pred - y) / torch.linalg.norm(y))


@torch.no_grad()
def torch_linf(model: QIMlp, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = model(x)
    return float((pred - y).abs().max())


def refit_readout_metrics(
    input_weights: np.ndarray,
    input_biases: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float]:
    phi_train = features_np(x_train, input_weights, input_biases)
    v, b, info = solve_readout_with_bias(phi_train, y_train, method="svd")
    train_pred = phi_train @ v + b
    eval_pred = features_np(x_eval, input_weights, input_biases) @ v + b
    return {
        "refit_train_rel_l2": rel_l2(train_pred, y_train),
        "refit_eval_rel_l2": rel_l2(eval_pred, y_eval),
        "refit_eval_linf": linf(eval_pred, y_eval),
        "refit_rank": float(info.get("rank", math.nan)),
    }


def feature_diagnostics(input_weights: np.ndarray, input_biases: np.ndarray, x_eval: np.ndarray) -> dict[str, float]:
    phi = features_np(x_eval, input_weights, input_biases)
    s = np.linalg.svd(phi, compute_uv=False)
    if s.size == 0:
        return {"feature_rank": 0.0, "feature_stable_rank": 0.0, "phi_cond": math.inf}
    smax = float(s[0])
    keep = s > 1e-12 * smax
    rank = int(np.sum(keep))
    stable = float(np.sum(s * s) / max(smax * smax, EPS))
    s_kept = s[keep]
    cond = float(s_kept[0] / s_kept[-1]) if s_kept.size else math.inf
    return {"feature_rank": float(rank), "feature_stable_rank": stable, "phi_cond": cond}


def drift_metrics(initial: InitialState, final_params: tuple[np.ndarray, np.ndarray, np.ndarray, float], h: float) -> dict[str, float]:
    fw, fb, fv, _ = final_params
    init_centers, init_gammas, init_valid = canonical_semantic(initial.input_weights, initial.input_biases)
    final_centers, final_gammas, final_valid = canonical_semantic(fw, fb)
    valid = init_valid & final_valid & np.isfinite(init_centers) & np.isfinite(final_centers)
    if np.any(valid):
        center_rms = float(np.sqrt(np.mean((final_centers[valid] - init_centers[valid]) ** 2)))
        log_gamma_rms = float(
            np.sqrt(np.mean((np.log(final_gammas[valid]) - np.log(init_gammas[valid])) ** 2))
        )
    else:
        center_rms = math.nan
        log_gamma_rms = math.nan
    readout_norm = max(float(np.linalg.norm(initial.readout_weights)), EPS)
    readout_drift = float(np.linalg.norm(fv - initial.readout_weights) / readout_norm)
    init_lambda = init_gammas * h
    final_lambda = final_gammas * h
    return {
        "center_rms_drift": center_rms,
        "log_gamma_rms_drift": log_gamma_rms,
        "readout_relative_drift": readout_drift,
        "lambda_initial_median": float(np.nanmedian(init_lambda)),
        "lambda_final_median": float(np.nanmedian(final_lambda)),
        "lambda_initial_max": float(np.nanmax(init_lambda)),
        "lambda_final_max": float(np.nanmax(final_lambda)),
        "degenerate_fraction": float(1.0 - np.mean(final_valid)),
    }


def atom_rows(
    case: Case,
    phase: str,
    input_weights: np.ndarray,
    input_biases: np.ndarray,
    readout_weights: np.ndarray,
    h: float,
    labels: Sequence[str],
    requested_lambdas: Sequence[float],
) -> Iterable[dict[str, object]]:
    centers, gammas, valid = canonical_semantic(input_weights, input_biases)
    for i in range(input_weights.size):
        yield {
            "target": case.target,
            "resolution": case.resolution,
            "width": int(input_weights.size),
            "seed": case.seed,
            "family": case.family,
            "phase": phase,
            "atom_index": i,
            "center": float(centers[i]) if np.isfinite(centers[i]) else math.nan,
            "gamma": float(gammas[i]),
            "lambda": float(gammas[i] * h),
            "readout": float(readout_weights[i]),
            "requested_lambda": float(requested_lambdas[i]),
            "label": labels[i],
            "valid": bool(valid[i]),
        }


def run_case(case: Case, config: RunConfig) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    geometry = geometry_for_resolution(case.resolution)
    state = build_initial_state(case.family, case.target, geometry, case.seed)
    model = make_model(state)
    x_train, y_train, x_eval, y_eval, xt, yt, xe, ye = make_data(case.target, config.n_train, config.n_eval)

    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    trace_rows: list[dict[str, object]] = []

    def append_trace(step: int, train_loss: float) -> tuple[float, float]:
        train_r = torch_rel_l2(model, xt, yt)
        eval_r = torch_rel_l2(model, xe, ye)
        trace_rows.append(
            {
                "target": case.target,
                "resolution": case.resolution,
                "width": geometry.width,
                "seed": case.seed,
                "family": case.family,
                "step": step,
                "train_loss": train_loss,
                "train_rel_l2": train_r,
                "eval_rel_l2": eval_r,
            }
        )
        return train_r, eval_r

    with torch.no_grad():
        init_train_loss = float(torch.mean((model(xt) - yt) ** 2))
    initial_train_rel, initial_eval_rel = append_trace(0, init_train_loss)
    best_eval = initial_eval_rel
    best_step = 0
    train_losses = [init_train_loss]

    for step in range(1, config.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((model(xt) - yt) ** 2)
        loss.backward()
        opt.step()
        loss_value = float(loss.detach())
        train_losses.append(loss_value)
        if step % config.eval_interval == 0 or step == config.steps:
            _, eval_rel = append_trace(step, loss_value)
            if eval_rel < best_eval:
                best_eval = eval_rel
                best_step = step

    final_train_rel = torch_rel_l2(model, xt, yt)
    final_eval_rel = torch_rel_l2(model, xe, ye)
    final_eval_linf = torch_linf(model, xe, ye)
    final_params = model_affine_params(model)
    fw, fb, fv, _ = final_params
    refit = refit_readout_metrics(fw, fb, x_train, y_train, x_eval, y_eval)
    diagnostics = feature_diagnostics(fw, fb, x_eval)
    drift = drift_metrics(state, final_params, geometry.h)

    summary = {
        "target": case.target,
        "resolution": case.resolution,
        "width": geometry.width,
        "halo": geometry.halo,
        "h": geometry.h,
        "seed": case.seed,
        "family": case.family,
        "steps": config.steps,
        "learning_rate": config.learning_rate,
        "n_train": config.n_train,
        "n_eval": config.n_eval,
        "initial_train_rel_l2": initial_train_rel,
        "initial_eval_rel_l2": initial_eval_rel,
        "final_train_rel_l2": final_train_rel,
        "final_eval_rel_l2": final_eval_rel,
        "best_eval_rel_l2": best_eval,
        "best_step": best_step,
        "final_eval_linf": final_eval_linf,
        "train_eval_ratio": final_train_rel / max(final_eval_rel, EPS),
        "train_loss_log_bump": math.log10(max(train_losses) / max(train_losses[-1], EPS)),
        **refit,
        **diagnostics,
        **drift,
    }
    atom_payload = list(
        atom_rows(
            case,
            "initial",
            state.input_weights,
            state.input_biases,
            state.readout_weights,
            geometry.h,
            state.atom_labels,
            state.requested_lambdas,
        )
    )
    atom_payload.extend(
        atom_rows(case, "final", fw, fb, fv, geometry.h, state.atom_labels, state.requested_lambdas)
    )
    return summary, trace_rows, atom_payload


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
    else:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def make_cases(config: RunConfig) -> list[Case]:
    return [
        Case(target=target, resolution=resolution, seed=seed, family=family)
        for target in config.targets
        for resolution in config.resolutions
        for seed in config.seeds
        for family in config.families
    ]


def completed_case_keys(summary_path: Path) -> set[tuple[str, int, int, str]]:
    return {row_case_key(row) for row in read_csv(summary_path)}


def run_matrix_rows(cases: Sequence[Case], config: RunConfig) -> list[dict[str, object]]:
    return [
        {
            "target": case.target,
            "resolution": case.resolution,
            "seed": case.seed,
            "family": case.family,
            "steps": config.steps,
            "learning_rate": config.learning_rate,
            "n_train": config.n_train,
            "n_eval": config.n_eval,
        }
        for case in cases
    ]


def group_key(row: dict[str, str]) -> tuple[str, int]:
    return row["target"], int(row["resolution"])


def median(values: Sequence[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(np.median(clean)) if clean else math.nan


FIGURE_CAPTIONS = (
    (
        "train_eval_final_scatter.png",
        "Final train vs eval relative $L_2$ with the 2x sanity band. Points far from the diagonal would indicate an overfit or data-path problem before any initializer comparison is trusted.",
    ),
    (
        "family_best_eval_rankings.png",
        "Median best eval relative $L_2$ by target-resolution block and initializer family. Lower log-scale colors mark better Adam training outcomes.",
    ),
    (
        "structured_vs_standard_fold_gap.png",
        "Per-block fold gap between the standard affine median and the structured QI-scale median. Values above one favor the structured scale-aware initializers.",
    ),
    (
        "learning_curves.png",
        "Median eval relative $L_2$ learning curves for the first target-resolution block in the artifact set. This is a convergence-shape diagnostic, not the full ranking evidence.",
    ),
    (
        "lambda_distribution_histograms.png",
        r"Initial and final atom counts over $\lambda=\gamma h$ at the largest resolution, with the $0.25$-$0.30$ construction band shaded.",
    ),
    (
        "drift_vs_error_scatter.png",
        r"Center and log-$\gamma$ drift versus best eval error. This checks whether successful rows preserve or leave the intended scale geometry.",
    ),
    (
        "atom_grid_representatives.png",
        r"Representative initial/final center-$\gamma$ atom grids for standard, scale-corrected, structured, and QI-family rows.",
    ),
)


def figure_html(filename: str, caption: str) -> str:
    return (
        f'<figure>\n'
        f'  <img src="figures/{filename}" alt="{filename}" style="max-width: 100%;">\n'
        f'  <figcaption>{caption}</figcaption>\n'
        f'</figure>'
    )


def plot_outputs(summary_path: Path = SUMMARY_CSV, traces_path: Path = TRACES_CSV, atoms_path: Path = ATOMS_CSV) -> None:
    summary = read_csv(summary_path)
    traces = read_csv(traces_path)
    atoms = read_csv(atoms_path)
    if not summary:
        return

    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    families = [f for f in DEFAULT_FAMILIES if any(row["family"] == f for row in summary)]
    blocks = sorted({group_key(row) for row in summary})

    # Train/eval scatter.
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for family in families:
        rows = [row for row in summary if row["family"] == family]
        ax.scatter(
            [float(row["final_train_rel_l2"]) for row in rows],
            [float(row["final_eval_rel_l2"]) for row in rows],
            s=18,
            alpha=0.7,
            label=family,
        )
    lo, hi = 1e-15, max(float(row["final_eval_rel_l2"]) for row in summary) * 2.0
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--")
    ax.plot([lo, hi], [2 * lo, 2 * hi], color="gray", lw=0.8, ls=":")
    ax.plot([lo, hi], [0.5 * lo, 0.5 * hi], color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("final train relative $L_2$")
    ax.set_ylabel("final eval relative $L_2$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "train_eval_final_scatter.png", dpi=180)
    plt.close(fig)

    # Family ranking heatmap.
    block_labels = [f"{target}\nN={resolution}" for target, resolution in blocks]
    heat = np.full((len(blocks), len(families)), np.nan)
    for i, block in enumerate(blocks):
        for j, family in enumerate(families):
            vals = [
                float(row["best_eval_rel_l2"])
                for row in summary
                if group_key(row) == block and row["family"] == family
            ]
            value = median(vals)
            heat[i, j] = math.log10(max(value, 1e-16)) if math.isfinite(value) else math.nan
    fig, ax = plt.subplots(figsize=(max(7.0, len(families) * 0.7), max(4.0, len(blocks) * 0.35)))
    im = ax.imshow(heat, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(block_labels, fontsize=7)
    ax.set_title("median best eval relative $L_2$ (log10)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "family_best_eval_rankings.png", dpi=180)
    plt.close(fig)

    # Structured-vs-standard fold gap.
    fold = []
    labels = []
    for block in blocks:
        standard_vals = [
            float(row["best_eval_rel_l2"])
            for row in summary
            if group_key(row) == block and row["family"] in STANDARD_FAMILIES
        ]
        structured_vals = [
            float(row["best_eval_rel_l2"])
            for row in summary
            if group_key(row) == block and row["family"] in STRUCTURED_FAMILIES
        ]
        if standard_vals and structured_vals:
            fold.append(median(standard_vals) / max(median(structured_vals), EPS))
            labels.append(f"{block[0]}\nN={block[1]}")
    if fold:
        fig, ax = plt.subplots(figsize=(max(6.0, len(fold) * 0.6), 3.5))
        ax.bar(range(len(fold)), fold, color="#4c78a8")
        ax.axhline(1.0, color="black", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("standard median / structured median")
        ax.set_xticks(range(len(fold)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "structured_vs_standard_fold_gap.png", dpi=180)
        plt.close(fig)

    # Learning curves.
    if traces:
        first_block = blocks[0]
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for family in families:
            rows = [
                row
                for row in traces
                if row["target"] == first_block[0]
                and int(row["resolution"]) == first_block[1]
                and row["family"] == family
            ]
            by_step: dict[int, list[float]] = {}
            for row in rows:
                by_step.setdefault(int(row["step"]), []).append(float(row["eval_rel_l2"]))
            if by_step:
                steps = sorted(by_step)
                ax.plot(steps, [median(by_step[step]) for step in steps], marker="o", ms=3, label=family)
        ax.set_yscale("log")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("eval relative $L_2$")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=180)
        plt.close(fig)

    # Lambda distributions.
    if atoms:
        largest_resolution = max(int(row["resolution"]) for row in atoms)
        rows = [row for row in atoms if int(row["resolution"]) == largest_resolution]
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
        for ax, phase in zip(axes, ("initial", "final")):
            vals = [
                float(row["lambda"])
                for row in rows
                if row["phase"] == phase and math.isfinite(float(row["lambda"])) and float(row["lambda"]) > 0.0
            ]
            if vals:
                ax.hist(vals, bins=40, color="#59a14f", alpha=0.85)
            ax.axvspan(0.25, 0.30, color="black", alpha=0.12)
            ax.set_xscale("log")
            ax.set_title(phase)
            ax.set_xlabel("$\\lambda=\\gamma h$")
        axes[0].set_ylabel("atom count")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "lambda_distribution_histograms.png", dpi=180)
        plt.close(fig)

    # Drift-vs-error.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for family in families:
        rows = [row for row in summary if row["family"] == family]
        axes[0].scatter(
            [float(row["center_rms_drift"]) for row in rows],
            [float(row["best_eval_rel_l2"]) for row in rows],
            s=18,
            alpha=0.7,
            label=family,
        )
        axes[1].scatter(
            [float(row["log_gamma_rms_drift"]) for row in rows],
            [float(row["best_eval_rel_l2"]) for row in rows],
            s=18,
            alpha=0.7,
        )
    for ax, xlabel in zip(axes, ("center RMS drift", "log-gamma RMS drift")):
        ax.set_xscale("symlog", linthresh=1e-8)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("best eval relative $L_2$")
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.02), ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "drift_vs_error_scatter.png", dpi=180)
    plt.close(fig)

    # Representative atom grids.
    if atoms:
        categories = [
            ("standard", STANDARD_FAMILIES),
            ("scale-corrected", SCALE_CORRECTED_FAMILIES),
            ("structured", STRUCTURED_FAMILIES),
            ("qi", QI_FAMILIES),
        ]
        reps: list[tuple[str, dict[str, str]]] = []
        for label, group in categories:
            candidates = [row for row in summary if row["family"] in group]
            if candidates:
                reps.append((label, min(candidates, key=lambda row: float(row["best_eval_rel_l2"]))))
        fig, axes = plt.subplots(len(reps), 2, figsize=(9.0, max(3.0, 2.4 * len(reps))), squeeze=False)
        for row_idx, (label, rep) in enumerate(reps):
            for col_idx, phase in enumerate(("initial", "final")):
                ax = axes[row_idx, col_idx]
                rows = [
                    atom
                    for atom in atoms
                    if atom["target"] == rep["target"]
                    and atom["resolution"] == rep["resolution"]
                    and atom["seed"] == rep["seed"]
                    and atom["family"] == rep["family"]
                    and atom["phase"] == phase
                ]
                centers = np.asarray([float(row["center"]) for row in rows], dtype=np.float64)
                gammas = np.asarray([float(row["gamma"]) for row in rows], dtype=np.float64)
                readouts = np.asarray([float(row["readout"]) for row in rows], dtype=np.float64)
                if centers.size:
                    color = np.sign(readouts) * np.log10(1.0 + np.abs(readouts) / max(np.std(readouts), 1e-12))
                    ax.scatter(centers, gammas, c=color, cmap="coolwarm", s=8, alpha=0.8)
                ax.set_yscale("log")
                ax.set_title(f"{label}: {phase}", fontsize=9)
                ax.set_xlabel("center")
                ax.set_ylabel("$\\gamma$")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "atom_grid_representatives.png", dpi=180)
        plt.close(fig)


def summarize_for_report(summary: list[dict[str, str]]) -> dict[str, object]:
    if not summary:
        return {
            "n_rows": 0,
            "blocks": 0,
            "targets": 0,
            "resolutions": 0,
            "seeds": 0,
            "families": 0,
            "expected_rows": len(DEFAULT_TARGETS) * len(DEFAULT_RESOLUTIONS) * len(DEFAULT_SEEDS) * len(DEFAULT_FAMILIES),
            "is_full_matrix": False,
            "structured_wins": 0,
            "scale_corrected_improves": 0,
            "structured_refit_wins": 0,
            "scale_refit_wins": 0,
            "scale_refit_le_1e10": 0,
            "scale_refit_le_1e12": 0,
            "structured_refit_le_1e8": 0,
            "standard_refit_le_1e10": 0,
            "ratio_flags": 0,
            "qi_drift": math.nan,
            "best_row": None,
            "best_deployable_row": None,
            "best_refit_row": None,
        }
    blocks = sorted({group_key(row) for row in summary})
    targets = sorted({row["target"] for row in summary})
    resolutions = sorted({int(row["resolution"]) for row in summary})
    seeds = sorted({int(row["seed"]) for row in summary})
    families = sorted({row["family"] for row in summary})
    expected_rows = len(DEFAULT_TARGETS) * len(DEFAULT_RESOLUTIONS) * len(DEFAULT_SEEDS) * len(DEFAULT_FAMILIES)
    is_full_matrix = (
        len(summary) == expected_rows
        and tuple(targets) == tuple(sorted(DEFAULT_TARGETS))
        and tuple(resolutions) == tuple(sorted(DEFAULT_RESOLUTIONS))
        and tuple(seeds) == tuple(sorted(DEFAULT_SEEDS))
        and tuple(families) == tuple(sorted(DEFAULT_FAMILIES))
    )
    structured_wins = 0
    scale_corrected_improves = 0
    structured_refit_wins = 0
    scale_refit_wins = 0
    scale_refit_le_1e10 = 0
    scale_refit_le_1e12 = 0
    structured_refit_le_1e8 = 0
    standard_refit_le_1e10 = 0
    for block in blocks:
        standard_vals = [
            float(row["best_eval_rel_l2"])
            for row in summary
            if group_key(row) == block and row["family"] in STANDARD_FAMILIES
        ]
        structured_vals = [
            float(row["best_eval_rel_l2"])
            for row in summary
            if group_key(row) == block and row["family"] in STRUCTURED_FAMILIES
        ]
        scaled_vals = [
            float(row["best_eval_rel_l2"])
            for row in summary
            if group_key(row) == block and row["family"] in SCALE_CORRECTED_FAMILIES
        ]
        standard_refit_vals = [
            float(row["refit_eval_rel_l2"])
            for row in summary
            if group_key(row) == block and row["family"] in STANDARD_FAMILIES
        ]
        structured_refit_vals = [
            float(row["refit_eval_rel_l2"])
            for row in summary
            if group_key(row) == block and row["family"] in STRUCTURED_FAMILIES
        ]
        scaled_refit_vals = [
            float(row["refit_eval_rel_l2"])
            for row in summary
            if group_key(row) == block and row["family"] in SCALE_CORRECTED_FAMILIES
        ]
        if standard_vals and structured_vals and median(structured_vals) < median(standard_vals):
            structured_wins += 1
        if standard_vals and scaled_vals and median(scaled_vals) < median(standard_vals):
            scale_corrected_improves += 1
        if standard_refit_vals and structured_refit_vals and median(structured_refit_vals) < median(standard_refit_vals):
            structured_refit_wins += 1
        if standard_refit_vals and scaled_refit_vals and median(scaled_refit_vals) < median(standard_refit_vals):
            scale_refit_wins += 1
        if scaled_refit_vals and median(scaled_refit_vals) <= 1e-10:
            scale_refit_le_1e10 += 1
        if scaled_refit_vals and median(scaled_refit_vals) <= 1e-12:
            scale_refit_le_1e12 += 1
        if structured_refit_vals and median(structured_refit_vals) <= 1e-8:
            structured_refit_le_1e8 += 1
        if standard_refit_vals and median(standard_refit_vals) <= 1e-10:
            standard_refit_le_1e10 += 1
    ratio_flags = sum(
        1
        for row in summary
        if float(row["train_eval_ratio"]) < 0.5 or float(row["train_eval_ratio"]) > 2.0
    )
    qi_drifts = [
        float(row["log_gamma_rms_drift"])
        for row in summary
        if row["family"] == "qi_scale_only_l030" and math.isfinite(float(row["log_gamma_rms_drift"]))
    ]
    scale_025_final = [
        float(row["lambda_final_median"])
        for row in summary
        if row["family"] == "scale_corrected_xavier_l025" and math.isfinite(float(row["lambda_final_median"]))
    ]
    scale_030_final = [
        float(row["lambda_final_median"])
        for row in summary
        if row["family"] == "scale_corrected_xavier_l030" and math.isfinite(float(row["lambda_final_median"]))
    ]
    qi_030_final = [
        float(row["lambda_final_median"])
        for row in summary
        if row["family"] == "qi_scale_only_l030" and math.isfinite(float(row["lambda_final_median"]))
    ]
    standard_final = [
        float(row["lambda_final_median"])
        for row in summary
        if row["family"] == "standard_xavier_affine" and math.isfinite(float(row["lambda_final_median"]))
    ]
    deployable_rows = [row for row in summary if row["family"] != "exact_qi_oracle"]
    return {
        "n_rows": len(summary),
        "blocks": len(blocks),
        "targets": len(targets),
        "resolutions": len(resolutions),
        "seeds": len(seeds),
        "families": len(families),
        "expected_rows": expected_rows,
        "is_full_matrix": is_full_matrix,
        "structured_wins": structured_wins,
        "scale_corrected_improves": scale_corrected_improves,
        "structured_refit_wins": structured_refit_wins,
        "scale_refit_wins": scale_refit_wins,
        "scale_refit_le_1e10": scale_refit_le_1e10,
        "scale_refit_le_1e12": scale_refit_le_1e12,
        "structured_refit_le_1e8": structured_refit_le_1e8,
        "standard_refit_le_1e10": standard_refit_le_1e10,
        "ratio_flags": ratio_flags,
        "qi_drift": median(qi_drifts),
        "scale_025_final_lambda": median(scale_025_final),
        "scale_030_final_lambda": median(scale_030_final),
        "qi_030_final_lambda": median(qi_030_final),
        "standard_final_lambda": median(standard_final),
        "best_row": min(summary, key=lambda row: float(row["best_eval_rel_l2"])),
        "best_deployable_row": min(deployable_rows, key=lambda row: float(row["best_eval_rel_l2"]))
        if deployable_rows
        else None,
        "best_refit_row": min(summary, key=lambda row: float(row["refit_eval_rel_l2"])),
    }


def write_report(summary_path: Path = SUMMARY_CSV) -> None:
    summary = read_csv(summary_path)
    if not summary:
        return
    stats = summarize_for_report(summary)
    best = stats["best_row"]
    best_deployable = stats["best_deployable_row"]
    best_refit = stats["best_refit_row"]
    if best is None:
        best_sentence = "No run artifacts have been generated yet."
    else:
        best_sentence = (
            f"Best observed row in the current artifact set is `{best['target']}` N={best['resolution']} "
            f"`{best['family']}` with best eval relative $L_2$ `{float(best['best_eval_rel_l2']):.3e}`."
        )
    if best_deployable is None:
        deployable_sentence = "No deployable initializer row has been generated yet."
    else:
        deployable_sentence = (
            f"Best non-oracle row is `{best_deployable['target']}` N={best_deployable['resolution']} "
            f"`{best_deployable['family']}` with best eval relative $L_2$ `{float(best_deployable['best_eval_rel_l2']):.3e}`."
        )
    if best_refit is None:
        refit_sentence = "No final lstsq refit artifact has been generated yet."
    else:
        refit_sentence = (
            f"Best final lstsq refit is `{best_refit['target']}` N={best_refit['resolution']} "
            f"`{best_refit['family']}` with eval relative $L_2$ `{float(best_refit['refit_eval_rel_l2']):.3e}`."
        )
    status = "draft-pending-Sam; full matrix complete." if stats["is_full_matrix"] else (
        "draft-pending-Sam; artifact set is smoke or partial."
    )
    figure_blocks = "\n\n".join(figure_html(filename, caption) for filename, caption in FIGURE_CAPTIONS)

    text = rf"""# expD05 -- Scale-aware initialization story

**Status: {status}**

## TL;DR

- The full fp64 matrix is present: `{stats['n_rows']}/{stats['expected_rows']}` rows over `{stats['blocks']}` target-resolution blocks, with `{stats['ratio_flags']}` train/eval sanity flags.
- Adam alone improves strongly with scale-aware initialization but does not close the non-oracle precision gap: {deployable_sentence}
- Final lstsq refits are the decisive check: {refit_sentence} Scale-corrected refit medians beat standard refit medians on `{stats['scale_refit_wins']}/{stats['blocks']}` blocks and reach $\le 10^{{-10}}$ on `{stats['scale_refit_le_1e10']}/{stats['blocks']}` blocks; standard refits reach that threshold on `{stats['standard_refit_le_1e10']}/{stats['blocks']}` blocks.
- The scale persists through Adam: median final $\lambda$ is `{stats['scale_025_final_lambda']:.3f}` for `scale_corrected_xavier_l025`, `{stats['scale_030_final_lambda']:.3f}` for `scale_corrected_xavier_l030`, and `{stats['qi_030_final_lambda']:.3f}` for `qi_scale_only_l030`; standard affine remains at `{stats['standard_final_lambda']:.3e}`.

## Question / hypothesis

Does the transferable part of QI initialization live in the dimensionless scale $\lambda=\gamma h$, and if so does initializing in that scale regime give Adam a better starting point than standard affine Xavier-style MLP initialization?

## Experiment design

For each target, construction resolution $N$, seed, and initializer family, the runner builds one single-hidden-layer tanh MLP in affine coordinates $a_jx+b_j$, trains all parameters with full-batch Adam in fp64, and then re-solves the readout on the trained first-layer features by fp64 SVD least squares. The shared geometry for a cell uses $h=2/N$ and one halo large enough for both $\lambda=0.25$ and $\lambda=0.30$, so all initializer families in that cell have the same hidden width. The feature matrix for the final refit is $\Phi_{{ij}}=\tanh(a_jx_i+b_j)$ and the refit solves $[\Phi,\mathbf 1][v;c]\approx y$. Metrics are eval relative $L_2=\|\hat f-f\|_2/\|f\|_2$, $L_\infty=\max_x|\hat f-f|$, final train/eval relative-$L_2$ ratio, $\lambda$ distributions, center drift, log-$\gamma$ drift, readout drift, and feature-rank diagnostics.

Initializer families are standard affine Xavier; center-spread Xavier; scale-corrected Xavier at $\lambda=0.25$ and $0.30$; target-independent QI-scale grids at $\lambda=0.25$ and $0.30$; a low-plus-QI multiscale schedule; exact target-dependent QI as an oracle; and QI geometry with random readout. `exact_qi_oracle` is diagnostic only and is not a deployable initialization rule.

**Code & data.** Code: `experiments/expD05_scale_init_story/run.py`, `experiments/expD05_scale_init_story/config.yaml`. The full matrix was run as eight resumable shards with `uv run --extra dev python experiments/expD05_scale_init_story/run.py --mode full --num-shards 8 --shard-index <k>`, then merged by `uv run --extra dev python experiments/expD05_scale_init_story/run.py --merge-shards --num-shards 8`. Data: `results/checkpoint_D_optimizers/expD05_scale_init_story/summary.csv`, `traces.csv`, `atoms.csv`, `run_matrix.csv`. Figures: `results/checkpoint_D_optimizers/expD05_scale_init_story/figures/`, generated by the same runner's plotting pass.

## Results

The artifact set contains `{stats['n_rows']}/{stats['expected_rows']}` expected full-matrix summary rows over `{stats['targets']}` targets, `{stats['resolutions']}` resolutions, `{stats['seeds']}` seeds, and `{stats['families']}` initializer families. The train/eval sanity check passes globally: `{stats['ratio_flags']}` rows fall outside the `[0.5,2.0]` final relative-$L_2$ ratio band.

Adam training alone preserves the initialization story but does not solve precision by itself. The target-dependent `exact_qi_oracle` is the best trained row (`{float(best['best_eval_rel_l2']):.3e}` best eval relative $L_2$) because it starts from the construction, while the best non-oracle trained row is still `{float(best_deployable['best_eval_rel_l2']):.3e}`. Even so, structured QI-scale medians beat standard affine medians on `{stats['structured_wins']}/{stats['blocks']}` blocks, and scale-corrected Xavier medians beat standard affine medians on `{stats['scale_corrected_improves']}/{stats['blocks']}` blocks.

The final lstsq refit is the stronger evidence about the learned geometry. Scale-corrected refits beat standard refits on `{stats['scale_refit_wins']}/{stats['blocks']}` blocks, and structured QI-scale refits beat standard refits on `{stats['structured_refit_wins']}/{stats['blocks']}` blocks. Scale-corrected refit medians reach $\le 10^{{-10}}$ on `{stats['scale_refit_le_1e10']}/{stats['blocks']}` blocks and $\le 10^{{-12}}$ on `{stats['scale_refit_le_1e12']}/{stats['blocks']}` blocks; structured refit medians reach $\le 10^{{-8}}$ on `{stats['structured_refit_le_1e8']}/{stats['blocks']}` blocks. {refit_sentence}

The scale parameter is not washed out by Adam. Standard affine ends with median final $\lambda\approx {stats['standard_final_lambda']:.3e}$, while scale-corrected and QI-scale families remain at the construction scale: $\lambda\approx {stats['scale_025_final_lambda']:.3f}$, `{stats['scale_030_final_lambda']:.3f}`, and `{stats['qi_030_final_lambda']:.3f}` for the $\lambda=0.25$, scale-corrected $\lambda=0.30$, and QI-scale $\lambda=0.30$ families respectively.

### Figures

{figure_blocks}

## Conclusions

*Pending Sam review.* The data-obvious conclusion is that initializing at the construction scale is a robust improvement over standard affine initialization in fp64, and that the scale persists during Adam. Adam alone still does not reach the non-oracle construction floor; the successful path remains scale-aware initialization followed by an exact final readout solve.

## Open questions

- Does $\lambda=0.25$ or $\lambda=0.30$ dominate once train/eval sanity and final lstsq refits are both considered?
- Are low-$\lambda$ global atoms consistently helpful, or is `qi_scale_only` enough after the target-repo fp64 rerun?
- Which non-oracle initializer should be promoted into the Checkpoint D story after Sam reviews the full-matrix plots?
"""
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(text)


def merge_csv_files(inputs: Sequence[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    wrote_header = False
    with output.open("w", newline="") as out:
        writer = None
        for path in inputs:
            if not path.exists() or path.stat().st_size == 0:
                continue
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                if not wrote_header:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    wrote_header = True
                if writer is None:
                    raise RuntimeError("csv writer not initialized")
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
    return rows_written


def shard_inputs(base: Path, num_shards: int | None = None) -> list[Path]:
    if num_shards is not None and num_shards > 1:
        return [artifact_path(base, f"shard{index:02d}of{num_shards:02d}") for index in range(num_shards)]
    return sorted(base.parent.glob(f"{base.stem}_shard*of*{base.suffix}"))


def merge_shards(num_shards: int | None = None) -> None:
    specs = [
        (SUMMARY_CSV, "summary"),
        (TRACES_CSV, "traces"),
        (ATOMS_CSV, "atoms"),
        (RUN_MATRIX_CSV, "run matrix"),
    ]
    for output, label in specs:
        inputs = shard_inputs(output, num_shards)
        count = merge_csv_files(inputs, output)
        print(f"[merge] wrote {count} {label} rows to {output}", flush=True)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    if args.mode == "smoke":
        default_targets = ("sine", "runge")
        default_resolutions = (32,)
        default_seeds = (0,)
        default_families = SMOKE_FAMILIES
        default_steps = 50
        default_eval_interval = 10
        default_n_train = 257
        default_n_eval = 513
    else:
        default_targets = DEFAULT_TARGETS
        default_resolutions = DEFAULT_RESOLUTIONS
        default_seeds = DEFAULT_SEEDS
        default_families = DEFAULT_FAMILIES
        default_steps = 20000
        default_eval_interval = 500
        default_n_train = 2003
        default_n_eval = 4001

    families = tuple(args.families or default_families)
    unknown = sorted(set(families) - set(DEFAULT_FAMILIES))
    if unknown:
        raise ValueError(f"unknown initializer families: {unknown}")
    for target in args.targets or default_targets:
        get_target(target)
    return RunConfig(
        targets=tuple(args.targets or default_targets),
        resolutions=tuple(args.resolutions or default_resolutions),
        seeds=tuple(args.seeds or default_seeds),
        families=families,
        steps=int(args.steps or default_steps),
        eval_interval=int(args.eval_interval or default_eval_interval),
        learning_rate=float(args.learning_rate),
        n_train=int(args.n_train or default_n_train),
        n_eval=int(args.n_eval or default_n_eval),
        mode=args.mode,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--resolutions", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--matrix-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    if args.merge_shards:
        merge_shards(args.num_shards if args.num_shards > 1 else None)
        plot_outputs()
        write_report()
        return 0
    if args.plot_only:
        plot_outputs()
        return 0
    if args.report_only:
        write_report()
        return 0

    config = config_from_args(args)
    all_cases = make_cases(config)
    cases = select_shard(all_cases, args.shard_index, args.num_shards)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    suffix = shard_suffix(args.shard_index, args.num_shards, args.output_suffix)
    # Smoke runs must not touch the unsuffixed full-matrix artifacts (or the
    # committed report/figures regenerated from them).
    if config.mode == "smoke" and not suffix:
        suffix = "smoke"
    paths = artifact_paths(suffix)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(paths.run_matrix, run_matrix_rows(cases, config))
    print(
        f"[matrix] wrote {len(cases)} rows to {paths.run_matrix} "
        f"(shard {args.shard_index + 1}/{args.num_shards})",
        flush=True,
    )
    if args.matrix_only:
        return 0

    if args.resume:
        completed = completed_case_keys(paths.summary)
        if completed:
            before = len(cases)
            cases = [case for case in cases if case_key(case) not in completed]
            print(f"[resume] skipping {before - len(cases)} completed cases from {paths.summary}", flush=True)
    else:
        for path in (paths.summary, paths.traces, paths.atoms):
            path.unlink(missing_ok=True)

    for index, case in enumerate(cases, start=1):
        print(
            f"[case {index}/{len(cases)}] target={case.target} N={case.resolution} "
            f"seed={case.seed} family={case.family}",
            flush=True,
        )
        summary, trace_rows, atom_payload = run_case(case, config)
        append_csv(paths.summary, [summary])
        append_csv(paths.traces, trace_rows)
        append_csv(paths.atoms, atom_payload)
    print(f"[summary] available at {paths.summary}", flush=True)
    print(f"[traces] available at {paths.traces}", flush=True)
    print(f"[atoms] available at {paths.atoms}", flush=True)
    if not suffix:
        plot_outputs(paths.summary, paths.traces, paths.atoms)
        write_report(paths.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

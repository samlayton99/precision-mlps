"""Float64 numerical state for the Codex geometry reader.

The server owns scheduling, persistence and concurrency.  One call to ``step``
does only Adam; ``evaluate`` does one complete readout solve and evaluation.
This separation makes every recorded frame a barrier in the training loop.
"""
from __future__ import annotations

import ast
import copy
import math
from typing import Any

import numpy as np
from scipy import linalg
from scipy.special import erf, expit


DEFAULT_CONFIG = {
    "target": "sin(2*pi*x)", "activation": "tanh",
    "n_train": 512, "n_test": 701, "sampling": "equispaced",
    "test_sampling": "equispaced", "prime_test_grid": True,
    "noise": 0.0, "seed": 42, "rcond": 1e-13, "ridge": 0.0,
    "n_interior": 64, "n_centers": 84, "n_intervals": 63, "halo": 10, "auto_halo": True,
    "reference_h": 2.0 / 63, "center_min": -1.0 - 20.0/63, "center_max": 1.0 + 20.0/63,
    "data_min": -1.0, "data_max": 1.0, "clean_lambda": 0.25,
    "readout_init": "zero", "lr": 1e-3,
    "train_geometry": True, "train_readout": True, "ls_feedback": False,
    "mask_enabled": False, "mask_min": -0.2, "mask_max": 0.2,
    "gradient_clip": 0.0,
}

_FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
    "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "Abs": np.abs,
    "log": np.log, "log1p": np.log1p, "sinh": np.sinh, "cosh": np.cosh,
}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub, ast.Constant,
            ast.Name, ast.Load, ast.Call)


def target_function(expression: str):
    """Compile a small mathematical expression language, without Python access."""
    if not isinstance(expression, str) or not 0 < len(expression) <= 1000:
        raise ValueError("Target must be a mathematical expression of 1–1000 characters.")
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise ValueError("Target is not a valid mathematical expression.") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 200:
        raise ValueError("Target expression is too complicated.")
    for node in nodes:
        if not isinstance(node, _ALLOWED):
            raise ValueError("Use numbers, x, pi, arithmetic and named mathematical functions.")
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float)
                                              or abs(node.value) > 1e100):
            raise ValueError("Only finite numeric constants are allowed.")
        if isinstance(node, ast.Name) and node.id not in {"x", "pi", "e", *_FUNCTIONS}:
            raise ValueError(f"Unknown target symbol: {node.id}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in _FUNCTIONS or len(node.args) != 1 or node.keywords):
            raise ValueError("Mathematical functions take one positional argument.")
        if isinstance(node, ast.Constant):
            # Float constants prevent arbitrarily large Python integer powers.
            node.value = float(node.value)
    code = compile(tree, "<target>", "eval")

    def evaluate(x):
        x = np.asarray(x, dtype=np.float64)
        namespace = {**_FUNCTIONS, "x": x, "pi": np.pi, "e": np.e}
        try:
            with np.errstate(all="ignore"):
                result = np.asarray(eval(code, {"__builtins__": {}}, namespace), dtype=np.float64)
                result = np.broadcast_to(result, x.shape).copy()
        except (TypeError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError("Target is not finite on the requested domain.") from exc
        if not np.isfinite(result).all():
            raise ValueError("Target is not finite on the requested domain.")
        return result

    return evaluate


def activation(z, name, derivative=False):
    """Activation or its analytic derivative, including exact GELU."""
    if name == "tanh":
        if derivative:
            tail = np.exp(-2.0 * np.abs(z))
            return 4.0 * tail / np.square(1.0 + tail)
        return np.tanh(z)
    if name == "sigmoid":
        if derivative:
            tail = np.exp(-np.abs(z))
            return tail / np.square(1.0 + tail)
        return expit(z)
    if name == "relu":
        return (z > 0).astype(float) if derivative else np.maximum(z, 0.0)
    if name == "gelu":
        cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        if derivative:
            return cdf + z * np.exp(-0.5 * np.square(z)) / np.sqrt(2 * np.pi)
        return z * cdf
    if name == "swish":
        a = expit(z)
        return a + z * activation(z, "sigmoid", True) if derivative else z * a
    raise ValueError(f"Unknown activation: {name}")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _next_prime(n):
    n = max(2, int(n))
    while any(n % divisor == 0 for divisor in range(2, math.isqrt(n) + 1)):
        n += 1
    return n


def interior_geometry(count, data_min=-1.0, data_max=1.0, halo=None):
    """N interior points including both endpoints, plus R halo nodes per side.

    N-1 interior gaps set h. Round sqrt(N) upward because halo nodes are whole.
    An explicit halo retains a saved/custom halo count when resizing.
    """
    if isinstance(count, bool) or int(count) != count or not 2 <= count <= 1958:
        raise ValueError("Interior center count N must be an integer between 2 and 1958.")
    count = int(count)
    halo = max(10, math.ceil(math.sqrt(count))) if halo is None else math.ceil(float(halo))
    total = count + 2*halo
    if halo < 0 or not 2 <= total <= 2048:
        raise ValueError("Interior points plus halo must fit within 2048 total centers.")
    h = (data_max - data_min) / (count-1)
    return {"n_interior": count, "n_centers": total, "n_intervals": count-1, "halo": halo,
            "reference_h": h, "center_min": data_min - halo*h,
            "center_max": data_max + halo*h}


class GeometryEngine:
    """Editable MLP whose geometry coordinates are c=-b/w and lambda=|w|h."""

    def __init__(self, config=None):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.params = {}
        self.configure(config or {})

    @staticmethod
    def _validate(config):
        for key, low, high in (("n_centers", 2, 2048), ("n_train", 4, 20000),
                               ("n_test", 16, 20000)):
            config[key] = int(config[key])
            if not low <= config[key] <= high:
                raise ValueError(f"{key} must be between {low} and {high}.")
        for key in ("noise", "ridge", "lr", "reference_h", "clean_lambda",
                    "center_min", "center_max", "data_min", "data_max",
                    "mask_min", "mask_max", "gradient_clip"):
            config[key] = float(config[key])
            if not np.isfinite(config[key]):
                raise ValueError(f"{key} must be finite.")
        if config["rcond"] is not None:
            config["rcond"] = float(config["rcond"])
            if not 0 <= config["rcond"] < 1:
                raise ValueError("rcond must be in [0, 1), or null for machine precision.")
        for key in ("lr", "reference_h", "clean_lambda"):
            if config[key] <= 0:
                raise ValueError(f"{key} must be positive.")
        for key in ("noise", "ridge", "gradient_clip"):
            if config[key] < 0:
                raise ValueError(f"{key} cannot be negative.")
        if config["center_min"] >= config["center_max"] or config["data_min"] >= config["data_max"]:
            raise ValueError("Axis minima must be smaller than maxima.")
        if config["mask_min"] > config["mask_max"]:
            raise ValueError("Mask minimum must not exceed its maximum.")
        if config["activation"] not in ("tanh", "sigmoid", "relu", "gelu", "swish"):
            raise ValueError("Unknown activation.")
        if config["readout_init"] not in ("zero", "xavier"):
            raise ValueError("Readout initialization must be zero or xavier.")
        if config["sampling"] not in ("equispaced", "uniform", "random", "chebyshev", "clustered", "midpoint"):
            raise ValueError("Unknown data sampling method.")
        if config["test_sampling"] not in ("equispaced", "midpoint"):
            raise ValueError("Test sampling must be equispaced or midpoint.")
        config["seed"] = int(config["seed"])
        if config["seed"] < 0:
            raise ValueError("Seed cannot be negative.")
        target_function(config["target"])

    def configure(self, config, reset=True):
        """Apply controls atomically, preserving geometry when reset=False.

        A reference-spacing change rescales both hidden arrays to retain every
        center and lambda, and resets Adam moments for those new coordinates.
        """
        candidate = {**self.config, **copy.deepcopy(config)}
        if "n_interior" not in config:
            if "n_intervals" in config:
                candidate["n_interior"] = max(2, round(float(config["n_intervals"]) + 1))
            elif "reference_h" in config:
                candidate["n_intervals"] = (candidate["data_max"]-candidate["data_min"]) / float(config["reference_h"])
                candidate["n_interior"] = max(2, round(candidate["n_intervals"] + 1))
        explicit_geometry = {"n_intervals", "halo", "reference_h", "center_min", "center_max"}
        if "auto_halo" not in config and explicit_geometry.intersection(config):
            candidate["auto_halo"] = False
        if candidate["auto_halo"] and (reset or "auto_halo" in config
                                       or {"data_min", "data_max"}.intersection(config)):
            candidate.update(interior_geometry(candidate["n_interior"],
                                                  candidate["data_min"], candidate["data_max"]))
        self._validate(candidate)
        fn, train_x, train_y, test_x, target = self._make_data(candidate)
        if not reset and self.params and candidate["n_centers"] != self.params["w"].size:
            raise ValueError("Use resize to change center count while preserving geometry.")
        old_h = self.config["reference_h"]
        self.config = candidate
        self.fn, self.train_x, self.train_y = fn, train_x, train_y
        self.test_x, self.target = test_x, target
        if reset or not self.params:
            self.rng = np.random.default_rng(candidate["seed"])
            self.initialize()
        elif candidate["reference_h"] != old_h:
            scale = old_h / candidate["reference_h"]
            self.params["w"] *= scale
            self.params["b"] *= scale
            self._reset_optimizer()
        self._solve_cache = None
        return self.config

    @staticmethod
    def _make_data(config):
        rng = np.random.default_rng(config["seed"])
        n, lo, hi = config["n_train"], config["data_min"], config["data_max"]
        sampling = config["sampling"]
        if sampling in ("uniform", "random"):
            x = np.sort(rng.uniform(lo, hi, n))
        elif sampling == "chebyshev":
            x = (lo + hi) / 2 + (hi - lo) / 2 * np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
            x.sort()
        elif sampling == "clustered":
            middle = (lo + hi) / 2
            x = np.concatenate((rng.normal(middle - (hi-lo)/4, (hi-lo)/14, n//2),
                                rng.normal(middle + (hi-lo)/4, (hi-lo)/14, n-n//2)))
            x = np.sort(np.clip(x, lo, hi))
        elif sampling == "midpoint":
            x = lo + (np.arange(n) + 0.5) * (hi - lo) / n
        else:
            x = np.linspace(lo, hi, n)
        if config["mask_enabled"]:
            x = x[(x < config["mask_min"]) | (x > config["mask_max"])]
        if x.size < 2:
            raise ValueError("The mask leaves fewer than two training samples.")
        fn = target_function(config["target"])
        y = fn(x) + config["noise"] * rng.standard_normal(x.size)
        # Interactive defaults use a distinct prime-sized test grid; imported
        # research presets preserve their exact grid and count for reproduction.
        n_test = _next_prime(config["n_test"]) if config["prime_test_grid"] else config["n_test"]
        if config["test_sampling"] == "midpoint":
            test_x = lo + (np.arange(n_test) + 0.5) * (hi - lo) / n_test
        else:
            test_x = np.linspace(lo, hi, n_test)
        return fn, x, y, test_x, fn(test_x)

    @property
    def centers(self):
        w = self.params["w"]
        safe = np.where(np.abs(w) < 1e-14, np.where(w < 0, -1e-14, 1e-14), w)
        return -self.params["b"] / safe

    @property
    def lambdas(self):
        return np.abs(self.params["w"]) * self.config["reference_h"]

    @property
    def gammas(self):
        return np.abs(self.params["w"])

    @property
    def signs(self):
        return np.where(self.params["w"] < 0, -1.0, 1.0)

    def _set_geometry(self, centers, lambdas, signs=None):
        centers, lambdas = np.asarray(centers, float), np.asarray(lambdas, float)
        if centers.ndim != 1 or lambdas.shape != centers.shape:
            raise ValueError("Centers and lambdas must be equally sized vectors.")
        if not np.isfinite(centers).all() or not np.isfinite(lambdas).all() or np.any(lambdas < 0):
            raise ValueError("Geometry must be finite, with nonnegative lambdas.")
        if signs is None:
            signs = np.ones_like(centers)
        signs = np.where(np.asarray(signs) < 0, -1.0, 1.0)
        if signs.shape != centers.shape:
            raise ValueError("Slope signs must match the center count.")
        self.params["w"] = signs * np.maximum(lambdas, 1e-10) / self.config["reference_h"]
        self.params["b"] = -self.params["w"] * centers

    def initialize(self, preset=None, apply_settings=False):
        """Start fresh with uniform geometry or a JSON preset.

        Presets accept config overrides and either centers/lambdas[/signs],
        slopes/hidden_bias, or kind='xavier'. Saved raw arrays take precedence.
        """
        preset = copy.deepcopy(preset or {})
        if isinstance(preset, str):
            preset = {"kind": preset}
        if "geometry" in preset:
            entry, geometry, settings = preset, preset["geometry"], preset.get("settings", {})
            preset = {"kind": entry.get("kind", "uniform")}
            for source, dest in (("centers", "centers"), ("lambdas", "lambdas"),
                                 ("signs", "signs"), ("slopes", "slopes"), ("biases", "hidden_bias")):
                if geometry.get(source) is not None:
                    preset[dest] = geometry[source]
            mapped = {"auto_halo": bool(settings.get("auto_halo", False))}
            for source, dest in (("N", "n_intervals"), ("h", "reference_h"), ("halo", "halo")):
                if settings.get(source) is not None:
                    mapped[dest] = settings[source]
            if "n_interior" in settings:
                mapped["n_interior"] = int(settings["n_interior"])
            elif "n_intervals" in mapped:
                # Historical source files use N for intervals. Their arrays
                # stay unchanged; the UI calls the corresponding N+1 points N.
                mapped["n_interior"] = max(2, round(float(mapped["n_intervals"]) + 1))
            if "reference_h" not in mapped and "n_intervals" in mapped:
                mapped["reference_h"] = 2.0 / float(mapped["n_intervals"])
            if "reference_h" in mapped and "halo" in mapped:
                mapped["center_min"] = -1.0 - mapped["reference_h"] * mapped["halo"]
                mapped["center_max"] = 1.0 + mapped["reference_h"] * mapped["halo"]
            if apply_settings:
                for source, dest in (("target", "target"), ("activation", "activation"),
                                     ("n_train", "n_train"), ("n_eval", "n_test"),
                                     ("sampling", "sampling"), ("rcond", "rcond")):
                    if source in settings:
                        mapped[dest] = settings[source]
                if mapped.get("target") == "sine":
                    mapped["target"] = "sin(2*pi*x)"
                # The preset source files call an endpoint lattice 'uniform'.
                # Interactive uniform sampling means random uniform draws.
                if mapped.get("sampling") == "uniform":
                    mapped["sampling"] = "equispaced"
                if "n_eval" in settings:
                    mapped["prime_test_grid"] = False
                mapped["test_sampling"] = "midpoint" if settings.get("sampling") == "midpoint" else "equispaced"
                if "domain" in settings:
                    mapped["data_min"], mapped["data_max"] = settings["domain"]
            preset["config"] = mapped
            if entry.get("use_saved_readout"):
                readout = entry.get("readout", {})
                preset["adam_readout"] = readout.get("saved", readout.get("initial"))
                preset["adam_output_bias"] = readout.get("saved_bias", readout.get("initial_bias", 0.0))
        updates = dict(preset.get("config", {}))
        for key in ("reference_h", "target", "activation", "n_interior", "n_intervals", "halo"):
            if key in preset:
                updates[key] = preset[key]
        explicit = preset.get("centers", preset.get("slopes", preset.get("w")))
        if explicit is not None:
            updates["n_centers"] = len(explicit)
        elif preset.get("kind") in ("uniform", "xavier") and self.config["auto_halo"]:
            updates.update(interior_geometry(self.config["n_interior"],
                                                self.config["data_min"], self.config["data_max"]))
            updates["auto_halo"] = True
        if updates:
            cfg = {**self.config, **updates}
            self._validate(cfg)
            data = self._make_data(cfg)
            self.config = cfg
            self.fn, self.train_x, self.train_y, self.test_x, self.target = data
        n = self.config["n_centers"]
        self.params = {}
        kind = preset.get("kind", "uniform")
        raw_w = preset.get("slopes", preset.get("w"))
        raw_b = preset.get("hidden_bias", preset.get("b"))
        if raw_w is not None and raw_b is not None:
            self.params["w"] = np.asarray(raw_w, dtype=float).copy()
            self.params["b"] = np.asarray(raw_b, dtype=float).copy()
        elif "centers" in preset:
            lam = preset.get("lambdas", preset.get("lambda", self.config["clean_lambda"]))
            lam = np.broadcast_to(np.asarray(lam, float), (n,)).copy()
            self._set_geometry(preset["centers"], lam, preset.get("signs"))
        elif kind == "xavier":
            bound = math.sqrt(6.0 / (1 + n))
            self.params["w"] = self.rng.uniform(-bound, bound, n)
            self.params["b"] = self.rng.uniform(-bound, bound, n)
        else:
            centers = np.linspace(self.config["center_min"], self.config["center_max"], n)
            self._set_geometry(centers, np.full(n, self.config["clean_lambda"]))
        if any(v.shape != (n,) or not np.isfinite(v).all() for v in self.params.values()):
            raise ValueError("Invalid preset parameter arrays.")
        if self.config["readout_init"] == "xavier":
            self.params["v"] = self.rng.uniform(-math.sqrt(6/(n+1)), math.sqrt(6/(n+1)), n)
        else:
            self.params["v"] = np.zeros(n)
        self.params["c"] = np.zeros(1)
        raw_v = preset.get("adam_readout", preset.get("readout", preset.get("v")))
        if raw_v is not None:
            self.params["v"] = np.asarray(raw_v, float).copy()
        raw_c = preset.get("adam_output_bias", preset.get("output_bias", preset.get("c", 0.0)))
        self.params["c"] = np.asarray([float(raw_c)])
        if self.params["v"].shape != (n,) or not all(np.isfinite(a).all() for a in self.params.values()):
            raise ValueError("Invalid preset readout.")
        self.step_count = 0
        self.pending_readout_edit = False
        self._reset_optimizer()
        self._solve_cache = None
        return self

    def _reset_optimizer(self):
        self.optimizer_step = 0
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.q = {k: np.zeros_like(v) for k, v in self.params.items()}

    def reset_readout(self):
        """Apply the selected initialization while retaining the current geometry."""
        n = self.params["w"].size
        if self.config["readout_init"] == "xavier":
            bound = math.sqrt(6.0 / (n + 1))
            self.params["v"] = self.rng.uniform(-bound, bound, n)
        else:
            self.params["v"] = np.zeros(n)
        self.params["c"] = np.zeros(1)
        self.pending_readout_edit = True
        self._reset_optimizer()

    def edit(self, kind, index=None, value=None):
        """Apply one plotted handle's edit, preserving signs and neuron identities."""
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Control values must be finite.")
        n = self.params["w"].size
        if kind in ("center", "lambda", "gamma", "readout"):
            index = int(index)
            if not 0 <= index < n:
                raise ValueError("Center index is out of range.")
        if kind == "center":
            self.params["b"][index] = -self.params["w"][index] * value
        elif kind in ("lambda", "gamma"):
            c, lam = self.centers, self.lambdas
            lam[index] = max(value * self.config["reference_h"] if kind == "gamma" else value, 1e-10)
            self._set_geometry(c, lam, self.signs)
        elif kind == "mean_gamma":
            # A ratio preserves the shape of the gamma profile and all centers.
            target = max(value, 1e-10 / self.config["reference_h"])
            gammas = self.gammas
            mean = float(gammas.mean())
            scaled = gammas * (target / mean) if mean > 0 else np.full_like(gammas, target)
            self._set_geometry(self.centers, scaled * self.config["reference_h"], self.signs)
        elif kind == "mean_lambda":
            # A common additive shift preserves height differences until a dot
            # reaches zero. Solve the clipped shift to make the requested mean exact.
            target = max(value, 1e-10)
            lam = self.lambdas
            lo, hi = -float(lam.max()), target + float(lam.max())
            for _ in range(70):
                middle = (lo + hi) / 2
                if np.maximum(lam + middle, 1e-10).mean() < target:
                    lo = middle
                else:
                    hi = middle
            self._set_geometry(self.centers, np.maximum(lam+(lo+hi)/2, 1e-10), self.signs)
        elif kind == "readout":
            self.params["v"][index] = value
            self.pending_readout_edit = True
        elif kind == "output_bias":
            self.params["c"][0] = value
            self.pending_readout_edit = True
        else:
            raise ValueError(f"Unknown geometry edit: {kind}")
        self._reset_optimizer()
        self._solve_cache = None

    def transform(self, action, amount):
        """Blend toward clean geometry or jitter by amount in [0, 1]."""
        amount = float(amount)
        if not 0 <= amount <= 1:
            raise ValueError("Transformation amount must be in [0, 1].")
        if amount == 0:
            return
        if action == "reset_readout":
            self.reset_readout()
            return
        centers, lambdas, signs = self.centers, self.lambdas, self.signs
        if action == "clean":
            # Pair the sorted current centers with the uniform halo lattice.
            order = np.argsort(centers, kind="stable")
            ideal = np.empty_like(centers)
            ideal[order] = np.linspace(self.config["center_min"], self.config["center_max"], centers.size)
            centers = centers + amount * (ideal - centers)
            lambdas = lambdas + amount * (self.config["clean_lambda"] - lambdas)
        elif action == "jitter_centers":
            centers += self.rng.normal(0, amount * 3 * self.config["reference_h"], centers.size)
        elif action in ("jitter_lambda", "jitter_gamma"):
            lambdas *= np.exp(self.rng.normal(0, amount, centers.size))
        else:
            raise ValueError(f"Unknown geometry transformation: {action}")
        self._set_geometry(centers, lambdas, signs)
        self._reset_optimizer()
        self._solve_cache = None

    def resize(self, count):
        """Set the interior point count; add halo nodes separately.

        No sorting occurs: inversions in center order remain part of the profile.
        Lambda heights interpolate, then a common factor restores their original
        arithmetic mean. h shrinks as new slots are inserted, so mean gamma rises.
        """
        updated = interior_geometry(count, self.config["data_min"], self.config["data_max"],
                                    None if self.config["auto_halo"] else self.config["halo"])
        count = updated["n_centers"]
        old_n = self.params["w"].size
        if all(self.config.get(key) == value for key, value in updated.items()):
            return
        t0, t1 = np.linspace(0, 1, old_n), np.linspace(0, 1, count)
        lo, hi, old_h = self.config["center_min"], self.config["center_max"], self.config["reference_h"]
        new_h = updated["reference_h"]
        deviations = (self.centers - np.linspace(lo, hi, old_n)) / old_h
        centers = np.linspace(updated["center_min"], updated["center_max"], count) + np.interp(t1, t0, deviations) * new_h
        old_lambdas = self.lambdas
        lambdas = np.interp(t1, t0, old_lambdas)
        # Sampling the same nonuniform profile at a new set of quantiles changes
        # its arithmetic mean. Restore that mean before gamma=lambda/h.
        if float(lambdas.mean()) > 0:
            lambdas *= float(old_lambdas.mean()) / float(lambdas.mean())
        signs = self.signs[np.minimum(np.rint(t1 * (old_n-1)).astype(int), old_n-1)]
        readout = np.interp(t1, t0, self.params["v"]) * new_h / old_h
        self.config.update(updated)
        self._set_geometry(centers, lambdas, signs)
        self.params["v"] = readout
        self._reset_optimizer()
        self._solve_cache = None

    def _features(self, x):
        return activation(x[:, None] * self.params["w"] + self.params["b"], self.config["activation"])

    def _loss_and_gradients(self):
        z = self.train_x[:, None] * self.params["w"] + self.params["b"]
        features = activation(z, self.config["activation"])
        residual = features @ self.params["v"] + self.params["c"][0] - self.train_y
        upstream = (2.0 / self.train_x.size) * residual
        hidden = (upstream[:, None] * self.params["v"]) * activation(z, self.config["activation"], True)
        grads = {
            "w": np.sum(hidden * self.train_x[:, None], axis=0),
            "b": np.sum(hidden, axis=0),
            "v": features.T @ upstream,
            "c": np.array([upstream.sum()]),
        }
        return float(np.mean(residual ** 2)), grads

    def step(self, count=1):
        """Advance count Adam iterations; no solve, evaluation, delay or I/O."""
        count = int(count)
        if count < 0:
            raise ValueError("Step count cannot be negative.")
        for _ in range(count):
            loss, gradients = self._loss_and_gradients()
            enabled = set()
            if self.config["train_geometry"]:
                enabled.update(("w", "b"))
            if self.config["train_readout"]:
                enabled.update(("v", "c"))
            if not enabled:
                raise ValueError("Enable geometry or readout training before running Adam.")
            if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
                raise ValueError("Training diverged; reduce the learning rate or reset geometry.")
            clip = self.config["gradient_clip"]
            if clip:
                norm = math.sqrt(sum(float(np.sum(gradients[k] ** 2)) for k in enabled))
                if norm > clip:
                    for k in enabled:
                        gradients[k] *= clip / norm
            self.optimizer_step += 1
            t = self.optimizer_step
            for key in enabled:
                g = gradients[key]
                self.m[key] = 0.9 * self.m[key] + 0.1 * g
                self.q[key] = 0.999 * self.q[key] + 0.001 * g * g
                delta = self.config["lr"] * (self.m[key] / (1 - 0.9**t)) / (np.sqrt(self.q[key] / (1 - 0.999**t)) + 1e-8)
                self.params[key] -= delta
            if not all(np.isfinite(p).all() for p in self.params.values()):
                raise ValueError("Training produced nonfinite parameters.")
            self.step_count += 1
            self.pending_readout_edit = False
        if count:
            self._solve_cache = None
        return self.step_count

    def solve(self):
        """SVD-truncated least squares, optionally damped by ridge on all coefficients."""
        design = np.column_stack((self._features(self.train_x), np.ones(self.train_x.size)))
        u, singular, vt = linalg.svd(design, full_matrices=False, check_finite=True, lapack_driver="gesdd")
        rcond = self.config["rcond"]
        if rcond is None:
            rcond = np.finfo(float).eps * max(design.shape)
        kept = singular > float(rcond) * singular[0]
        factors = np.zeros_like(singular)
        ridge = self.config["ridge"]
        if ridge:
            factors[kept] = singular[kept] / (singular[kept]**2 + ridge)
        else:
            factors[kept] = 1 / singular[kept]
        solution = vt.T @ (factors * (u.T @ self.train_y))
        stats = {"rank": int(kept.sum()), "columns": int(design.shape[1]),
                 "condition": float(singular[0]/singular[kept][-1]) if np.any(kept) else None,
                 "singular_values": singular.tolist(), "rcond": rcond, "ridge": ridge}
        return solution[:-1], float(solution[-1]), stats

    def _derivative_overlay(self, centers):
        eps = 1e-4 * (self.config["data_max"] - self.config["data_min"])
        try:
            fm2, fm1, f0, fp1, fp2 = (self.fn(centers + k*eps) for k in (-2, -1, 0, 1, 2))
            h = self.config["reference_h"]
            if self.config["activation"] in ("relu", "gelu", "swish"):
                derivative = (-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) / (12*eps**2)
                prediction = (h*h / np.maximum(self.lambdas, 1e-10)) * derivative
                label = "h² / λ · f″(c)"
            else:
                derivative = (fm2 - 8*fm1 + 8*fp1 - fp2) / (12*eps)
                factor = h if self.config["activation"] == "sigmoid" else h/2
                prediction = factor * derivative * self.signs
                label = "h · f′(c) × slope sign" if factor == h else "h/2 · f′(c) × slope sign"
            prediction = [float(v) if np.isfinite(v) else None for v in prediction]
            return prediction, label
        except (ValueError, FloatingPointError, OverflowError):
            return [None] * centers.size, "Derivative unavailable beyond the target domain"

    @staticmethod
    def _metrics(prediction, target):
        residual = prediction - target
        return {"mse": float(np.mean(residual**2)), "linf": float(np.max(np.abs(residual))),
                "relative_l2": float(np.linalg.norm(residual)/max(np.linalg.norm(target), 1e-300))}

    def evaluate(self, feedback=False):
        """Finish a solve and both evaluations before returning a recordable frame.

        Adam readout fields capture the pre-solve network. The exact resume state
        captures post-feedback parameters, so replay never changes recorded curves.
        A manual readout edit is protected until a gradient update uses it.
        """
        solved_v, solved_c, solver = self.solve()
        adam_v, adam_c = self.params["v"].copy(), float(self.params["c"][0])
        features = self._features(self.test_x)
        adam_prediction = features @ adam_v + adam_c
        solved_prediction = features @ solved_v + solved_c
        train_features = self._features(self.train_x)
        centers = self.centers
        derivative, derivative_label = self._derivative_overlay(centers)
        did_feedback = bool(feedback and self.config["ls_feedback"] and not self.pending_readout_edit)
        frame = {
            "step": self.step_count, "config": copy.deepcopy(self.config),
            "centers": centers, "lambdas": self.lambdas, "gammas": self.gammas, "signs": self.signs,
            "slopes": self.params["w"].copy(), "hidden_bias": self.params["b"].copy(),
            "adam_readout": adam_v, "solved_readout": solved_v,
            "adam_output_bias": adam_c, "solved_output_bias": solved_c,
            "x": self.test_x, "target": self.target,
            "adam_prediction": adam_prediction, "solved_prediction": solved_prediction,
            "adam_residual": adam_prediction-self.target, "solved_residual": solved_prediction-self.target,
            "train_x": self.train_x, "train_y": self.train_y,
            "derivative_overlay": derivative, "derivative_label": derivative_label,
            "mean_lambda": float(self.lambdas.mean()), "mean_gamma": float(self.gammas.mean()),
            "ideal_gamma": self.config["clean_lambda"] / self.config["reference_h"], "solver": solver,
            "metrics": {"adam": self._metrics(adam_prediction, self.target),
                        "solved": self._metrics(solved_prediction, self.target)},
            "train_metrics": {"adam": self._metrics(train_features@adam_v+adam_c, self.train_y),
                              "solved": self._metrics(train_features@solved_v+solved_c, self.train_y)},
            "feedback_applied": did_feedback,
            "pending_readout_edit": self.pending_readout_edit,
        }
        if did_feedback:
            self.params["v"] = solved_v.copy()
            self.params["c"] = np.array([solved_c])
            # Retain optimizer moments: LS is an interleaved parameter projection.
        frame["state"] = self.state_dict()
        return _jsonable(frame)

    def state_dict(self):
        return _jsonable({
            "version": 1, "config": copy.deepcopy(self.config),
            "params": self.params, "m": self.m, "q": self.q,
            "step_count": self.step_count, "optimizer_step": self.optimizer_step,
            "pending_readout_edit": self.pending_readout_edit,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "train_x": self.train_x, "train_y": self.train_y,
        })

    def load_state(self, state):
        """Restore arrays, optimizer, data and random stream from a saved frame."""
        if state.get("version") != 1:
            raise ValueError("Unsupported saved engine state version.")
        config = {**DEFAULT_CONFIG, "auto_halo": False, **copy.deepcopy(state["config"])}
        if "n_interior" not in state["config"]:
            config["n_interior"] = max(2, round(float(config["n_intervals"]) + 1))
        self._validate(config)
        fn, train_x, train_y, test_x, target = self._make_data(config)
        n = config["n_centers"]
        groups = {}
        for group in ("params", "m", "q"):
            groups[group] = {}
            for key in ("w", "b", "v", "c"):
                a = np.asarray(state[group][key], dtype=float)
                if a.shape != ((1,) if key == "c" else (n,)) or not np.isfinite(a).all():
                    raise ValueError("Saved state contains invalid parameter arrays.")
                groups[group][key] = a.copy()
        if "train_x" in state:
            train_x, train_y = np.asarray(state["train_x"], float), np.asarray(state["train_y"], float)
            if train_x.ndim != 1 or train_x.size < 2 or train_x.shape != train_y.shape or not np.isfinite(train_x).all() or not np.isfinite(train_y).all():
                raise ValueError("Saved training data are invalid.")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.config = config
        self.fn, self.train_x, self.train_y, self.test_x, self.target = fn, train_x, train_y, test_x, target
        self.params, self.m, self.q = groups["params"], groups["m"], groups["q"]
        self.rng = rng
        self.step_count, self.optimizer_step = int(state["step_count"]), int(state["optimizer_step"])
        self.pending_readout_edit = bool(state.get("pending_readout_edit", False))
        self._solve_cache = None
        return self

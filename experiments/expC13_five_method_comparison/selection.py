"""Candidate bookkeeping and validation selection shared by the five methods (SPEC "Selection")."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import pfloat

from experiments.expC13_five_method_comparison import common as C


@dataclass
class Context:
    """Everything a method may use: the target oracle (through common.sample), the format, the
    validation inputs and meter, and the parameter budget. The reporting grid is not here."""
    target: str
    p: int
    budget: int = C.PARAM_BUDGET

    def __post_init__(self):
        self.F = C.fmt(self.p)
        pts = C.midpoint_points(C.VALIDATION_POINTS)
        self.val_x = C.inputs(pts, self.F)
        self.val_meter = _meter(self.target, "validation")


_METERS: dict = {}


def _meter(target: str, grid: str) -> C.Meter:
    key = (target, grid)
    if key not in _METERS:
        pts = C.midpoint_points(C.VALIDATION_POINTS) if grid == "validation" else C.linspace_points(C.REPORT_POINTS)
        _METERS[key] = C.Meter(target, pts)
    return _METERS[key]


def report_meter(target: str) -> C.Meter:
    return _meter(target, "report")


@dataclass
class Selection:
    method: str
    target: str
    p: int
    best_net: object = None
    best_hp: dict = None
    best_val: float = math.inf
    table: list = field(default_factory=list)   # one row per candidate: hp, size, validation error, status
    limit_key: str = "params"                    # the budgeted size: nonzero parameters (or neurons, sensitivity arm)
    limit: int = C.PARAM_BUDGET

    def offer(self, hp: dict, net, val_out: pfloat.PArray | None, meter: C.Meter, *, status: str = "ok"):
        """Record a candidate; keep it if its validation error is the smallest so far (ties keep the
        smaller network, then the earlier candidate)."""
        row = dict(hp)
        if net is None or val_out is None:
            row.update(status=status, rel_l2=math.inf)
            self.table.append(row)
            return
        err = meter.errors(val_out)
        size = net.params()
        row.update(status=status, rel_l2=err["rel_l2"], rel_linf=err["rel_linf"], params=size,
                   neurons=net.neurons(), depth=net.depth(), max_abs=net.max_abs())
        if (size if self.limit_key == "params" else net.neurons()) > self.limit:
            row["status"] = "over_budget"
            self.table.append(row)
            return
        self.table.append(row)
        e = err["rel_l2"]
        if e < self.best_val or (e == self.best_val and self.best_net is not None and size < self.best_net.params()):
            self.best_net, self.best_hp, self.best_val = net, dict(hp), e

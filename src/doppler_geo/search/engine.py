"""Budgeted coarse-to-fine search, explicit authorization and shared time alignment."""

from dataclasses import dataclass
from time import monotonic
from typing import TypedDict

import numpy as np

from doppler_geo.physics.model import closest_approach
from doppler_geo.scoring.curves import ComponentScore, curve_score


class Candidate(TypedDict):
    x_m: float
    y_m: float
    z_m: float
    offset_s: float
    loss: float
    components: list[ComponentScore]
    cpa: list[dict[str, float]]


@dataclass(frozen=True)
class SearchArea:
    """Local ENU rectangle in meters. Boundary includes its edges."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    authorized: bool = False
    max_area_km2: float = 2500.0

    def __post_init__(self):
        values = [self.xmin, self.xmax, self.ymin, self.ymax, self.max_area_km2]
        if self.authorized is not True:
            raise ValueError("Operator must explicitly authorize the test area and recording")
        if (
            not np.isfinite(values).all()
            or self.xmax <= self.xmin
            or self.ymax <= self.ymin
            or self.max_area_km2 <= 0
            or (self.xmax - self.xmin) * (self.ymax - self.ymin) > self.max_area_km2 * 1e6
        ):
            raise ValueError("Invalid or oversized research area")

    def contains(self, x, y):
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass
class SearchConfig:
    coarse_step_m: float = 500.0
    fine_step_m: float = 50.0
    top_k: int = 5
    max_evaluations: int = 10000
    max_runtime_s: float = 120.0
    max_range_m: float = 25000.0
    offsets_s: tuple = (0.0,)
    c: float = 343.0
    f0_bounds: tuple = (40.0, 1000.0)
    method: str = "gaussian"

    def __post_init__(self):
        if (
            not 0 < self.fine_step_m <= self.coarse_step_m
            or self.top_k < 1
            or self.max_evaluations < 1
            or self.max_runtime_s <= 0
            or self.max_range_m <= 0
            or not len(self.offsets_s)
            or not np.isfinite(self.offsets_s).all()
            or not np.isfinite(self.c)
            or self.c <= 0
            or not np.isfinite(
                [self.coarse_step_m, self.fine_step_m, self.max_runtime_s, self.max_range_m]
            ).all()
            or len(self.f0_bounds) != 2
            or not 0 < self.f0_bounds[0] < self.f0_bounds[1]
            or self.method not in {"gaussian", "robust"}
        ):
            raise ValueError("Invalid search configuration")


def locate(events, flights, area, config=None, elevation=None, progress=None):
    """Events/flights are explicitly paired by operator; never shifted independently.

    elevation(x,y) returns local ENU up meters (not orthometric elevation).
    """
    config = config or SearchConfig()
    if not isinstance(area, SearchArea) or not area.authorized:
        raise ValueError("An authorized SearchArea is required")
    if len(events) != len(flights) or not events:
        raise ValueError("Provide one selected flight per acoustic event")
    started = monotonic()
    evaluations = 0
    visited = set()
    rows: list[Candidate] = []
    stopped = False

    def evaluate(x, y):
        nonlocal evaluations, stopped
        if evaluations >= config.max_evaluations or monotonic() - started > config.max_runtime_s:
            stopped = True
            return
        if not area.contains(x, y) or (round(x, 6), round(y, 6)) in visited:
            return
        visited.add((round(x, 6), round(y, 6)))
        observer = np.array([x, y, elevation(x, y) if elevation else 0.0])
        # CPA is conservative geometry screening; delayed CPA is only a diagnostic,
        # since a ridge midpoint need not correspond to CPA.
        cpas = [closest_approach(f, observer, config.c) for f in flights]
        if any(p["distance_m"] > config.max_range_m for p in cpas):
            return
        best: Candidate | None = None
        for offset in config.offsets_s:
            if (
                evaluations >= config.max_evaluations
                or monotonic() - started > config.max_runtime_s
            ):
                stopped = True
                break
            components = [
                curve_score(e, f, observer, offset, config.c, config.f0_bounds, config.method)
                for e, f in zip(events, flights, strict=True)
            ]
            loss = float(np.mean([component["loss"] for component in components]))
            evaluations += 1
            if best is None or loss < best["loss"]:
                best = {
                    "x_m": float(x),
                    "y_m": float(y),
                    "z_m": float(observer[2]),
                    "offset_s": float(offset),
                    "loss": loss,
                    "components": components,
                    "cpa": cpas,
                }
        if best is not None and np.isfinite(best["loss"]):
            rows.append(best)
        if progress and evaluations % 100 == 0:
            progress(evaluations)

    # Reject an oversized coarse grid before allocating a mesh.
    nx = int(np.ceil((area.xmax - area.xmin) / config.coarse_step_m)) + 1
    ny = int(np.ceil((area.ymax - area.ymin) / config.coarse_step_m)) + 1
    if nx * ny * len(config.offsets_s) > config.max_evaluations:
        raise ValueError("Coarse grid exceeds evaluation budget; increase step or reduce area")
    for x in np.linspace(area.xmin, area.xmax, nx):
        for y in np.linspace(area.ymin, area.ymax, ny):
            evaluate(float(x), float(y))
            if stopped:
                break
        if stopped:
            break
    step = config.coarse_step_m
    while rows and step > config.fine_step_m and not stopped:
        next_step = max(config.fine_step_m, step / 4)
        seeds = sorted(rows, key=lambda row: row["loss"])[: config.top_k]
        for seed in seeds:
            for x in np.arange(seed["x_m"] - step, seed["x_m"] + step + next_step / 2, next_step):
                for y in np.arange(
                    seed["y_m"] - step, seed["y_m"] + step + next_step / 2, next_step
                ):
                    evaluate(float(x), float(y))
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
        step = next_step
    rows.sort(key=lambda row: row["loss"])
    warnings = [
        "Scores are uncalibrated; candidate spread is not a confidence region.",
        "Results depend on the selected event-flight associations and atmospheric model.",
    ]
    if len(events) == 1:
        warnings.append(
            "A single flight can have mirror ambiguity and weak along-track constraints."
        )
    if stopped:
        warnings.append("Computation limit reached; search is incomplete.")
    if rows and rows[0]["loss"] > 4:
        warnings.append("Large fitting residuals: insufficient evidence for a reliable location.")
    if not rows:
        warnings.append("No candidate has complete arrival-time coverage and acceptable range.")
    near = [r for r in rows if r["loss"] <= rows[0]["loss"] + 1] if rows else []
    spread = max(
        (np.hypot(r["x_m"] - rows[0]["x_m"], r["y_m"] - rows[0]["y_m"]) for r in near), default=0
    )
    return {
        "candidates": rows[: config.top_k],
        "evaluated_locations": rows,
        "evaluations": evaluations,
        "runtime_s": monotonic() - started,
        "near_optimal_spread_m": float(spread),
        "complete": not stopped,
        "warnings": warnings,
    }

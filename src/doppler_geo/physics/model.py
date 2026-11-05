"""SI units; static observers; homogeneous atmosphere; emission/arrival time explicit."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar


@dataclass
class Trajectory:
    times: np.ndarray
    positions: np.ndarray
    flight_id: str = "unknown"
    max_gap_s: float = 30.0

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float)
        if (
            self.times.ndim != 1
            or len(self.times) < 3
            or self.positions.shape != (len(self.times), 3)
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.positions).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("Need >=3 finite positions and strictly increasing emission times")
        if np.max(np.diff(self.times)) > self.max_gap_s:
            raise ValueError("Trajectory gap exceeds interpolation limit; segment the flight")
        self._position = PchipInterpolator(self.times, self.positions, axis=0, extrapolate=False)
        self._velocity = self._position.derivative()

    def position(self, times):
        return self._position(times)

    def velocity(self, times):
        return self._velocity(times)


def sound_speed(temperature_c=15.0):
    """Dry ideal-gas approximation; not a full atmospheric profile."""
    if not np.isfinite(temperature_c) or temperature_c <= -273.15:
        raise ValueError("Temperature must exceed absolute zero")
    return float(np.sqrt(1.4 * 287.05 * (273.15 + temperature_c)))


def propagate(flight: Trajectory, observer, emission_times=None, c=343.0):
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Sound speed must be positive")
    te = flight.times if emission_times is None else np.asarray(emission_times, dtype=float)
    r = flight.position(te) - np.asarray(observer, dtype=float)
    v = flight.velocity(te)
    d = np.linalg.norm(r, axis=-1)
    if not np.isfinite(r).all() or np.any(d <= 0):
        raise ValueError("Invalid propagation geometry or out-of-range trajectory")
    # Reject supersonic segments: this single-arrival model is not a shock-wave model.
    if np.any(np.linalg.norm(v, axis=-1) >= c):
        raise ValueError("Only subsonic trajectories are supported")
    radial_approach = -np.sum(r * v, axis=-1) / d
    ta = te + d / c
    if np.any(np.diff(ta) <= 0):
        raise ValueError("Arrival-time map is not monotonic")
    return ta, c / (c - radial_approach), d


def predict(flight, observer, arrival_times, c=343.0, step_s=0.25):
    """Return frequency multiplier at arrival times; NaN outside observed flight support."""
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    te = np.linspace(
        flight.times[0], flight.times[-1], max(3, int(np.ceil(np.ptp(flight.times) / step_s)) + 1)
    )
    ta, ratio, _ = propagate(flight, observer, te, c)
    return np.interp(arrival_times, ta, ratio, left=np.nan, right=np.nan)


def closest_approach(flight, observer, c=343.0):
    """Global sampled screening then bounded refinement of each local minimum."""
    grid = np.linspace(flight.times[0], flight.times[-1], max(3, int(np.ptp(flight.times)) + 1))
    dist = np.linalg.norm(flight.position(grid) - observer, axis=1)
    best = (float(dist[0]), float(grid[0]))
    if dist[-1] < best[0]:
        best = (float(dist[-1]), float(grid[-1]))
    for i in np.flatnonzero((dist[1:-1] <= dist[:-2]) & (dist[1:-1] <= dist[2:])) + 1:
        result = minimize_scalar(
            lambda t: np.linalg.norm(flight.position(t) - observer),
            bounds=(grid[i - 1], grid[i + 1]),
            method="bounded",
        )
        if result.fun < best[0]:
            best = (float(result.fun), float(result.x))
    d, te = best
    return {"emission_s": te, "arrival_s": te + d / c, "distance_m": d}

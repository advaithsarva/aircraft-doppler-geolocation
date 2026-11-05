"""Profile unknown source tones; preserve one global recording-time offset."""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from doppler_geo.physics.model import predict
from doppler_geo.validation import positive_integer


class ComponentScore(TypedDict):
    loss: float
    f0_hz: float | None
    rmse_hz: float | None


@dataclass
class AcousticEvent:
    times: np.ndarray
    frequencies: np.ndarray
    sigma_hz: float = 2.0
    harmonic: int = 1
    label: str = "unassigned"

    def __post_init__(self):
        positive_integer(self.harmonic, "harmonic")
        self.times = np.asarray(self.times, dtype=float)
        self.frequencies = np.asarray(self.frequencies, dtype=float)
        if (
            self.times.ndim != 1
            or len(self.times) < 4
            or self.times.shape != self.frequencies.shape
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.frequencies).all()
            or np.any(np.diff(self.times) <= 0)
            or np.any(self.frequencies <= 0)
            or not np.isfinite(self.sigma_hz)
            or self.sigma_hz <= 0
            or self.harmonic < 1
        ):
            raise ValueError("Invalid acoustic event")

    def to_dict(self):
        return {
            "times": self.times.tolist(),
            "frequencies": self.frequencies.tolist(),
            "sigma_hz": self.sigma_hz,
            "harmonic": int(self.harmonic),
            "label": self.label,
        }


def curve_score(
    event, flight, observer, offset_s=0, c=343.0, f0_bounds=(40.0, 1000.0), method="gaussian"
) -> ComponentScore:
    if (
        len(f0_bounds) != 2
        or not np.isfinite(f0_bounds).all()
        or not 0 < f0_bounds[0] < f0_bounds[1]
    ):
        raise ValueError("Invalid source-frequency bounds")
    if method not in {"gaussian", "robust"}:
        raise ValueError("Unknown curve scoring method")
    ratio = predict(flight, observer, event.times + offset_s, c) * event.harmonic
    if not np.isfinite(ratio).all():
        return {"loss": float("inf"), "f0_hz": None, "rmse_hz": None}
    f0 = float(np.clip(np.dot(ratio, event.frequencies) / np.dot(ratio, ratio), *f0_bounds))
    if method == "robust":
        # Pseudo-Huber loss is convex in f0. Solve its monotonic derivative for the
        # same objective used to rank candidates, rather than fitting outliers by LS.
        def derivative(frequency):
            r = (frequency * ratio - event.frequencies) / event.sigma_hz
            return float(np.sum(ratio * r / np.hypot(1, r)))

        lower, upper = f0_bounds
        if derivative(lower) >= 0:
            f0 = float(lower)
        elif derivative(upper) <= 0:
            f0 = float(upper)
        else:
            f0 = float(brentq(derivative, lower, upper))
    residual = (event.frequencies - f0 * ratio) / event.sigma_hz
    if method == "gaussian":
        loss = float(np.mean(residual**2))
    elif method == "robust":
        loss = float(np.mean(2 * (np.sqrt(1 + residual**2) - 1)))
    else:
        raise ValueError("Unknown curve scoring method")
    return {
        "loss": loss,
        "f0_hz": f0,
        "rmse_hz": float(np.sqrt(np.mean((residual * event.sigma_hz) ** 2))),
    }


def energy_score(spec, arrival_times, predicted_frequencies, harmonics=(1, 2, 3)):
    """Log energy above per-frame median; evidence score, never a probability."""
    whitened = np.log1p(spec.power / (np.median(spec.power, axis=0)[None, :] + 1e-15))
    sample = RegularGridInterpolator(
        (spec.frequencies, spec.times), whitened, bounds_error=False, fill_value=np.nan
    )
    values = np.array(
        [sample(np.column_stack([predicted_frequencies * h, arrival_times])) for h in harmonics]
    )
    valid = np.isfinite(values)
    if not valid.any():
        return float("nan")
    return float(np.mean(values[valid]))

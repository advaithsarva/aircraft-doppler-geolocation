"""Independent analytic sound synthesis: invert retarded time, then evaluate source phase."""

from dataclasses import dataclass

import numpy as np

from doppler_geo.physics.model import Trajectory


@dataclass
class AnalyticFlight:
    origin: np.ndarray
    velocity: np.ndarray
    f0_hz: float = 180.0
    acceleration: np.ndarray | None = None
    tone_start_s: float = 0.0
    tone_end_s: float = 120.0

    def position(self, t):
        a = np.zeros(3) if self.acceleration is None else self.acceleration
        t = np.asarray(t)[..., None]
        return self.origin + t * self.velocity + 0.5 * t * t * a

    def trajectory(self, start=-60, end=200, step=2, flight_id="synthetic"):
        t = np.arange(start, end + step / 2, step, dtype=float)
        return Trajectory(t, self.position(t), flight_id)


def synthesize(
    flights,
    observer,
    duration_s=150,
    sample_rate=4000,
    c=343.0,
    snr_db=0,
    seed=42,
    harmonics=(1.0, 0.4, 0.2),
):
    """No production Doppler predictor is used to generate ground truth.

    Newton-invert t_e + |P(t_e)-O|/c = t_a for analytic differentiable paths.
    Source phase is 2*pi*f0*t_e; optional 1/r spreading and tone envelopes.
    """
    if duration_s <= 0 or sample_rate < 1000 or c <= 0 or not flights:
        raise ValueError("Invalid simulation settings")
    if max(f.f0_hz for f in flights) * len(harmonics) * 2 >= sample_rate:
        raise ValueError("Source harmonics exceed Nyquist; increase sample rate")
    ta = np.arange(int(duration_s * sample_rate)) / sample_rate
    audio = np.zeros_like(ta)
    truth = []
    for flight in flights:
        te = ta.copy() - np.linalg.norm(flight.origin - observer) / c
        acceleration = np.zeros(3) if flight.acceleration is None else flight.acceleration
        for _ in range(20):
            r = flight.position(te) - observer
            d = np.linalg.norm(r, axis=1)
            velocity = flight.velocity + te[:, None] * acceleration
            derivative = 1 + np.sum(r * velocity, axis=1) / (c * d)
            if np.any(np.linalg.norm(velocity, axis=1) >= c) or np.any(d < 1):
                raise ValueError("Synthetic trajectory must be subsonic and separated")
            update = (te + d / c - ta) / derivative
            te -= update
            if np.max(np.abs(update)) < 1e-10:
                break
        else:
            raise ValueError("Retarded-time inversion failed")
        if np.max(flight.f0_hz * len(harmonics) / derivative) >= sample_rate / 2:
            raise ValueError("Doppler-shifted harmonics exceed Nyquist")
        active = (te >= flight.tone_start_s) & (te <= flight.tone_end_s)
        envelope = np.minimum(
            np.clip(te - flight.tone_start_s, 0, 1), np.clip(flight.tone_end_s - te, 0, 1)
        )
        tone = np.zeros_like(ta)
        for harmonic, amplitude in enumerate(harmonics, 1):
            tone += amplitude * np.sin(2 * np.pi * flight.f0_hz * harmonic * te)
        audio += envelope * tone * (1000 / np.maximum(d, 100))
        truth.append({"arrival_times": ta[active], "emission_times": te[active]})
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(audio**2))
    audio += rng.normal(0, rms * 10 ** (-snr_db / 20), len(ta))
    audio /= max(1, np.max(np.abs(audio)))
    return audio, truth


def example_scenario(held_out=False):
    """Nonparallel, overlapping flights break the straight-line mirror symmetry."""
    if held_out:
        observer = np.array([-225.0, 375.0, 0.0])
        flights = [
            AnalyticFlight(
                np.array([-7500.0, -2200.0, 1700.0]), np.array([145.0, 25.0, 1.0]), 115.0
            ),
            AnalyticFlight(
                np.array([4200.0, -9500.0, 2300.0]), np.array([-45.0, 165.0, -0.8]), 480.0
            ),
        ]
    else:
        observer = np.array([350.0, -250.0, 0.0])
        flights = [
            AnalyticFlight(np.array([-8500.0, 1400.0, 1600.0]), np.array([170.0, 0.0, 0.0]), 110.0),
            AnalyticFlight(
                np.array([-1900.0, -10000.0, 2100.0]), np.array([20.0, 180.0, 0.0]), 500.0
            ),
        ]
    return observer, flights

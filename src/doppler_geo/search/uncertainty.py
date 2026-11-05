"""Conditional perturbation analysis, explicitly not a calibrated confidence region."""

from dataclasses import replace

import numpy as np

from doppler_geo.physics.model import Trajectory
from doppler_geo.scoring.curves import AcousticEvent
from doppler_geo.search.engine import SearchConfig, locate


def perturbation_ensemble(
    events,
    flights,
    area,
    config=None,
    draws=20,
    seed=0,
    sound_speed_sigma=2.0,
    altitude_sigma_m=30.0,
):
    if not 2 <= draws <= 100:
        raise ValueError("Use between 2 and 100 bounded perturbation runs")
    if sound_speed_sigma < 0 or altitude_sigma_m < 0:
        raise ValueError("Uncertainty scales must be nonnegative")
    config = config or SearchConfig()
    rng = np.random.default_rng(seed)
    locations = []
    skipped = 0
    for _ in range(draws):
        perturbed_events = [
            AcousticEvent(
                e.times,
                e.frequencies + rng.normal(0, e.sigma_hz, len(e.times)),
                e.sigma_hz,
                e.harmonic,
                e.label,
            )
            for e in events
        ]
        perturbed_flights = [
            Trajectory(
                f.times,
                f.positions + np.array([0, 0, rng.normal(0, altitude_sigma_m)]),
                f.flight_id,
                f.max_gap_s,
            )
            for f in flights
        ]
        result = locate(
            perturbed_events,
            perturbed_flights,
            area,
            replace(config, c=float(config.c + rng.normal(0, sound_speed_sigma))),
        )
        if result["candidates"] and result["complete"]:
            best = result["candidates"][0]
            locations.append([best["x_m"], best["y_m"], best["offset_s"]])
        else:
            skipped += 1
    return {
        "samples_east_north_offset": locations,
        "incomplete_draws": skipped,
        "seed": seed,
        "draws": draws,
        "interpretation": "Conditional sensitivity to assumed independent frequency, "
        "sound-speed and altitude errors; not a calibrated confidence region.",
    }

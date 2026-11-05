"""Conditional perturbation analysis, explicitly not a calibrated confidence region."""

from dataclasses import replace
from time import monotonic

import numpy as np

from doppler_geo.physics.model import Trajectory
from doppler_geo.scoring.curves import AcousticEvent
from doppler_geo.search.engine import SearchArea, SearchConfig, locate
from doppler_geo.validation import finite_scalar, positive_integer


def perturbation_ensemble(
    events,
    flights,
    area,
    config=None,
    draws=20,
    seed=0,
    sound_speed_sigma=2.0,
    altitude_sigma_m=30.0,
    total_runtime_s=120.0,
    total_evaluations=20000,
    elevation=None,
):
    positive_integer(draws, "draws")
    if not 2 <= draws <= 100:
        raise ValueError("Use between 2 and 100 bounded perturbation runs")
    finite_scalar(sound_speed_sigma, "sound_speed_sigma", nonnegative=True)
    finite_scalar(altitude_sigma_m, "altitude_sigma_m", nonnegative=True)
    finite_scalar(total_runtime_s, "total_runtime_s", positive=True)
    positive_integer(total_evaluations, "total_evaluations")
    if not isinstance(area, SearchArea) or not events or len(events) != len(flights):
        raise ValueError("Provide an authorized area and paired events/flights")
    config = config or SearchConfig()
    per_draw_budget = min(config.max_evaluations, total_evaluations // draws)
    nx = int(np.ceil((area.xmax - area.xmin) / config.coarse_step_m)) + 1
    ny = int(np.ceil((area.ymax - area.ymin) / config.coarse_step_m)) + 1
    if per_draw_budget < nx * ny * len(config.offsets_s):
        raise ValueError("Ensemble evaluation budget cannot cover each coarse grid")
    rng = np.random.default_rng(seed)
    started = monotonic()
    locations = []
    errors = []
    evaluations = 0
    completed = 0
    attempted = 0
    for _ in range(draws):
        remaining = total_runtime_s - (monotonic() - started)
        if remaining <= 0:
            break
        attempted += 1
        sampled_c = float(config.c + rng.normal(0, sound_speed_sigma))
        try:
            sampled_config = replace(
                config,
                c=sampled_c,
                max_runtime_s=min(config.max_runtime_s, remaining),
                max_evaluations=per_draw_budget,
            )
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
                    f.segment_id,
                )
                for f in flights
            ]
            result = locate(
                perturbed_events, perturbed_flights, area, sampled_config, elevation=elevation
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        evaluations += result["evaluations"]
        if result["candidates"] and result["complete"]:
            best = result["candidates"][0]
            locations.append([best["x_m"], best["y_m"], best["offset_s"]])
            completed += 1
    return {
        "samples_east_north_offset": locations,
        "incomplete_draws": draws - completed,
        "invalid_draw_errors": errors,
        "attempted_draws": attempted,
        "evaluations": evaluations,
        "runtime_s": monotonic() - started,
        "complete": completed == draws,
        "seed": seed,
        "draws": draws,
        "interpretation": "Conditional sensitivity to assumed independent frequency, "
        "sound-speed and altitude errors; not a calibrated confidence region.",
    }

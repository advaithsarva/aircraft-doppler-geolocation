"""Measured synthetic cases; independent seeds and geometry; no external data claims."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from doppler_geo.audio.processing import load_annotations, spectrogram
from doppler_geo.detection.tracking import detect_events
from doppler_geo.experiment import run_experiment
from doppler_geo.physics.model import Trajectory, predict
from doppler_geo.scoring.curves import AcousticEvent
from doppler_geo.search.association import rank_assignments
from doppler_geo.search.engine import SearchArea, SearchConfig, locate
from doppler_geo.synthetic import example_scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/validation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for held_out, snr, seed in [
        (False, 0, 11),
        (False, -5, 42),
        (False, -15, 73),
        (True, 0, 19),
        (True, -5, 73),
        (True, -15, 91),
    ]:
        name = f"{'held-out' if held_out else 'baseline'}_{snr}_{seed}"
        result = run_experiment(output / name, seed, snr, held_out, write_plots=False)
        summary = {
            "case": name,
            **result["experiment"],
            "evaluations": result["evaluations"],
            "search_s": result["runtime_s"],
            "frequency_rmse_hz": [c["rmse_hz"] for c in result["candidates"][0]["components"]],
        }
        cases.append(summary)
        print(name, f"error={summary['localization_error_m']:.2f}m", flush=True)

    observer, analytic = example_scenario(True)
    flights = [a.trajectory(flight_id=f"flight-{i + 1}") for i, a in enumerate(analytic)]
    source = output / "held-out_-5_73" / "annotations.json"
    events = load_annotations(source)
    area = SearchArea(-1000, 1000, -1000, 1000, True)
    config = SearchConfig(coarse_step_m=500, fine_step_m=25, top_k=4)
    diagnostics = []
    for name, altered_flights, altered_events, cfg in [
        (
            "missing_samples",
            [Trajectory(f.times[::4], f.positions[::4], f.flight_id) for f in flights],
            events,
            config,
        ),
        (
            "incorrect_altitude_plus_200m",
            [Trajectory(f.times, f.positions + [0, 0, 200], f.flight_id) for f in flights],
            events,
            config,
        ),
        (
            "shared_recording_offset",
            flights,
            [AcousticEvent(e.times - 2.5, e.frequencies, e.sigma_hz) for e in events],
            SearchConfig(
                coarse_step_m=500, fine_step_m=25, top_k=4, offsets_s=(-2.5, 0.0, 2.5, 5.0)
            ),
        ),
        (
            "frequency_interference",
            flights,
            [
                AcousticEvent(
                    e.times,
                    e.frequencies + np.where(np.arange(len(e.times)) % 7 == 0, 20, 0),
                    e.sigma_hz,
                )
                for e in events
            ],
            SearchConfig(coarse_step_m=500, fine_step_m=25, top_k=4, method="robust"),
        ),
    ]:
        result = locate(altered_events, altered_flights, area, cfg)
        best = result["candidates"][0]
        diagnostics.append(
            {
                "case": name,
                "localization_error_m": float(
                    np.hypot(best["x_m"] - observer[0], best["y_m"] - observer[1])
                ),
                "offset_s": best["offset_s"],
                "loss": best["loss"],
                "runtime_s": result["runtime_s"],
                "evaluations": result["evaluations"],
            }
        )

    ranked = rank_assignments(events, flights[::-1], area, config)
    chosen = ranked["assignments"][0]["flight_ids"]
    association = {
        "selected_ids": chosen,
        "truth_ids": [f.flight_id for f in flights],
        "correct": chosen == [f.flight_id for f in flights],
        "scope": "One synthetic two-flight permutation case, not population accuracy",
    }
    false_proposals = 0
    noise_trials = 10
    for seed in range(100, 100 + noise_trials):
        noise = np.random.default_rng(seed).normal(size=4000 * 12)
        spec = spectrogram(noise, 4000, band=(50, 1200))
        false_proposals += bool(detect_events(spec))
    negatives = {
        "noise_only_trials": noise_trials,
        "trials_with_tonal_proposals": false_proposals,
        "proposal_rate": false_proposals / noise_trials,
        "scope": "White-noise tonal proposals only; not a geolocation false-positive rate",
    }
    # Delay/frequency checks against analytic independent phase are in pytest.
    # Measure frequency-model residual at ground truth on the held-out observed ridges.
    ground_truth_errors = []
    for event, flight, source_flight in zip(events, flights, analytic, strict=True):
        expected = predict(flight, observer, event.times) * source_flight.f0_hz
        ground_truth_errors.append(float(np.sqrt(np.mean((event.frequencies - expected) ** 2))))
    report = {
        "environment": {"platform": platform.platform(), "processor": platform.processor()},
        "audio_recovery_cases": cases,
        "diagnostic_cases": diagnostics,
        "association": association,
        "negative_controls": negatives,
        "held_out_ridge_rmse_hz_at_truth": ground_truth_errors,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

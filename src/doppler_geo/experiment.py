"""Reproducible audio -> STFT -> detected ridges -> ranked locations experiment."""

import json
import logging
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from doppler_geo.audio.processing import save_annotations, spectrogram
from doppler_geo.detection.tracking import track_ridge
from doppler_geo.memory import peak_rss_mib
from doppler_geo.search.engine import SearchArea, SearchConfig, locate
from doppler_geo.synthetic import example_scenario, synthesize

LOG = logging.getLogger(__name__)


def run_experiment(output, seed=42, snr_db=-5, held_out=False, write_plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    observer, analytic = example_scenario(held_out)
    fs = 8000
    audio, _ = synthesize(analytic, observer, sample_rate=fs, snr_db=snr_db, seed=seed)
    sf.write(output / "synthetic.wav", audio, fs, subtype="FLOAT")
    spec = spectrogram(audio, fs, n_fft=4096, hop=1024, band=(50, 1200))
    events = []
    quality = []
    # Frequency windows are deliberately broad known source-band priors for the synthetic demo.
    # They are not known observer positions or model-generated annotation curves.
    bands = [(60, 240), (300, 1150)]
    mask = (spec.times >= 35) & (spec.times <= 105)
    from doppler_geo.audio.processing import Spectrogram

    cropped = Spectrogram(spec.times[mask], spec.frequencies, spec.power[:, mask])
    for index, band in enumerate(bands):
        event, contrast = track_ridge(cropped, band, sigma_hz=2.0, label=f"flight-{index + 1}")
        events.append(event)
        quality.append(contrast)
    save_annotations(output / "annotations.json", events)
    tracks = [f.trajectory(flight_id=e.label) for f, e in zip(analytic, events, strict=True)]
    area = SearchArea(-1000, 1000, -1000, 1000, authorized=True)
    config = SearchConfig(coarse_step_m=500, fine_step_m=25, top_k=4)
    LOG.info("Scoring authorized synthetic test area")
    result = locate(events, tracks, area, config, progress=lambda n: LOG.info("%d scores", n))
    best = result["candidates"][0]
    error = float(np.linalg.norm(np.array([best["x_m"], best["y_m"], best["z_m"]]) - observer))
    result["experiment"] = {
        "seed": seed,
        "snr_db": snr_db,
        "held_out": held_out,
        "truth_observer_enu_m": observer.tolist(),
        "localization_error_m": error,
        "ridge_contrast": quality,
        "sample_rate": fs,
        "search_config": asdict(config),
        "end_to_end_s": perf_counter() - started,
        "peak_rss_mib": peak_rss_mib(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "flight_identification": "operator-assigned synthetic associations; not measured",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if write_plots:
        from doppler_geo.visualization.report import write_report

        write_report(output / "report.html", spec, events, tracks, result)
    return result

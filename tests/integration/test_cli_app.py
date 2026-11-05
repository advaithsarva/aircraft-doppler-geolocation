import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from doppler_geo.audio.processing import save_annotations
from doppler_geo.cli import run_config
from doppler_geo.physics.coordinates import LocalFrame
from doppler_geo.physics.model import predict
from doppler_geo.scoring.curves import AcousticEvent
from doppler_geo.synthetic import example_scenario


def test_local_config_pipeline(tmp_path):
    observer, sources = example_scenario()
    frame = LocalFrame(51.5, -0.1)
    flights = [s.trajectory(flight_id=f"F{i}") for i, s in enumerate(sources)]
    origin = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    rows = []
    for flight in flights:
        geodetic = frame.to_geodetic(flight.positions)
        for time, (lat, lon, height) in zip(flight.times, geodetic, strict=True):
            rows.append(
                {
                    "flight_id": flight.flight_id,
                    "aircraft_id": flight.flight_id,
                    "timestamp": (origin + dt.timedelta(seconds=time)).isoformat(),
                    "latitude": lat,
                    "longitude": lon,
                    "altitude_m": height,
                }
            )
    tracks_path = tmp_path / "tracks.parquet"
    pl.DataFrame(rows).write_parquet(tracks_path)
    annotations_path = tmp_path / "annotations.json"
    t = np.arange(35.0, 100.0, 2.0)
    events = [
        AcousticEvent(t, predict(f, observer, t) * s.f0_hz)
        for f, s in zip(flights, sources, strict=True)
    ]
    save_annotations(annotations_path, events)
    config = {
        "tracks": str(tracks_path),
        "annotations": str(annotations_path),
        "frame": {"latitude": 51.5, "longitude": -0.1},
        "flight_ids": ["F0", "F1"],
        "area": {"xmin": -500, "xmax": 500, "ymin": -500, "ymax": 500, "authorized": True},
        "ingestion": {
            "bbox": [-2, 50, 2, 53],
            "start_utc": "2024-12-31T23:59:00Z",
            "end_utc": "2025-01-01T00:04:00Z",
            "time_origin_utc": "2025-01-01T00:00:00Z",
            "altitude_datum": "ellipsoid",
        },
        "search": {"coarse_step_m": 250, "fine_step_m": 25},
        "output": str(tmp_path / "new-output" / "result.json"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = run_config(path)
    best = result["candidates"][0]
    assert np.hypot(best["x_m"] - observer[0], best["y_m"] - observer[1]) < 40
    assert 51 < best["latitude"] < 52
    assert len(result["curve_diagnostics"]) == 2
    assert len(result["provenance"]["config_sha256"]) == 64
    assert (tmp_path / "new-output" / "result.json").exists()
    config["area"]["authorized"] = False
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="authorize"):
        run_config(path)


def test_dashboard_synthetic_workflow():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(Path(__file__).parents[2] / "app" / "research.py"), default_timeout=30
    ).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert "m" in app.metric[0].value
    assert app.session_state["result"]["experiment"]["localization_error_m"] < 150
    app.sidebar.radio[0].set_value("Authorized local-data search").run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Audio annotation").run()
    assert not app.exception

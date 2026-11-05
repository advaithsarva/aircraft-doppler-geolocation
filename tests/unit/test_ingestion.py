import datetime as dt

import numpy as np
import polars as pl
import pytest

from doppler_geo.adsb.ingest import FlightIndex, load_tracks, parquet_window
from doppler_geo.physics.coordinates import LocalFrame


def data():
    origin = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    return pl.DataFrame(
        {
            "flight_id": ["A"] * 7,
            "aircraft_id": ["abc"] * 7,
            "timestamp": [
                (origin + dt.timedelta(seconds=i)).isoformat() for i in [0, 1, 2, 60, 61, 62, 62]
            ],
            "latitude": [51.5] * 7,
            "longitude": [-0.1 + i * 0.001 for i in [0, 1, 2, 60, 61, 62, 62]],
            "altitude_m": [1800.0] * 7,
        }
    )


def load(path, datum="ellipsoid", offset=None):
    return load_tracks(
        path,
        LocalFrame(51.5, -0.1),
        (-1, 51, 1, 52),
        "2025-01-01T00:00:00Z",
        "2025-01-01T00:02:00Z",
        altitude_datum=datum,
        altitude_offset_m=offset,
        time_origin_utc="2025-01-01T00:00:00Z",
    )


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_clean_deduplicate_segment(tmp_path, extension):
    path = tmp_path / f"tracks.{extension}"
    frame = data()
    if extension == "csv":
        frame.write_csv(path)
    else:
        frame.write_parquet(path)
    tracks, report = load(path)
    assert len(tracks) == 2
    assert report["clean_rows"] == 6
    assert tracks[0].times.tolist() == [0.0, 1.0, 2.0]
    index = FlightIndex(tracks)
    assert len(index.within_range(np.array([0, 0, 0]), 2000)) >= 1
    assert index.within_range(np.array([1e7, 0, 0]), 100) == []
    if extension == "parquet":
        assert (
            len(
                parquet_window(
                    path, (-1, 51, 1, 52), "2025-01-01T00:00:00Z", "2025-01-01T00:02:00Z"
                )
            )
            == 7
        )


def test_missing_altitude_datum_conversion(tmp_path):
    path = tmp_path / "tracks.csv"
    data().write_csv(path)
    with pytest.raises(ValueError, match="conversion"):
        load(path, "barometric")
    tracks, _ = load(path, "orthometric", 45)
    assert len(tracks) == 2


def test_conflicting_samples_excluded(tmp_path):
    path = tmp_path / "tracks.csv"
    frame = data()
    extra = frame.head(1).with_columns(pl.lit(2000.0).alias("altitude_m"))
    pl.concat([frame, extra]).write_csv(path)
    tracks, report = load(path)
    assert report["clean_rows"] == 5
    assert len(tracks) == 1


def test_timezone_and_row_budgets(tmp_path):
    path = tmp_path / "tracks.csv"
    data().with_columns(pl.col("timestamp").str.replace(r"\+00:00$", "")).write_csv(path)
    with pytest.raises(ValueError, match="UTC offset"):
        load(path)
    data().write_csv(path)
    with pytest.raises(ValueError, match="budget"):
        load_tracks(
            path,
            LocalFrame(51.5, -0.1),
            (-1, 51, 1, 52),
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:02:00Z",
            altitude_datum="ellipsoid",
            time_origin_utc="2025-01-01T00:00:00Z",
            max_rows=3,
        )


def test_segment_identifiers_and_index_generator(tmp_path):
    path = tmp_path / "segments.csv"
    data().write_csv(path)
    tracks, report = load(path)
    assert [t.segment_id for t in tracks] == ["A/segment-1", "A/segment-2"]
    assert len(report["available_segments"]) == 2
    index = FlightIndex(f for f in tracks)
    assert index.within_range(np.zeros(3), 2000)


def test_duckdb_rejects_ambiguous_time_and_invalid_bbox(tmp_path):
    path = tmp_path / "tracks.parquet"
    data().write_parquet(path)
    with pytest.raises(ValueError, match="timezone"):
        parquet_window(path, (-1, 51, 1, 52), "2025-01-01T00:00:00", "2025-01-01T00:02:00Z")
    with pytest.raises(ValueError, match="bbox"):
        parquet_window(path, (1, 51, -1, 52), "2025-01-01T00:00:00Z", "2025-01-01T00:02:00Z")

import numpy as np
import polars as pl
import pytest

from doppler_geo.adsb.opensky import normalize_export
from doppler_geo.audio.processing import spectrogram
from doppler_geo.physics.coordinates import LocalFrame
from doppler_geo.search.engine import SearchConfig
from doppler_geo.synthetic import AnalyticFlight, synthesize
from doppler_geo.terrain.dem import RasterTerrain


def test_dem_vertical_datum_and_nodata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        nodata=-9999,
        transform=from_origin(-0.15, 51.55, 0.01, 0.01),
    ) as dataset:
        dataset.write(np.full((1, 10, 10), 100.0, dtype="float32"))
    frame = LocalFrame(51.5, -0.1)
    with pytest.raises(ValueError, match="geoid"):
        RasterTerrain(path, frame, datum="orthometric")
    terrain = RasterTerrain(path, frame, datum="orthometric", geoid_offset_m=45)
    try:
        assert terrain(0, 0) == pytest.approx(145, abs=0.01)
        with pytest.raises(ValueError, match="coverage"):
            terrain(1e5, 1e5)
    finally:
        terrain.close()


def test_opensky_normalization_explicit_altitude():
    data = pl.DataFrame(
        {
            "icao24": ["abc"],
            "time": [1735689600],
            "lat": [51.5],
            "lon": [-0.1],
            "geoaltitude": [1800.0],
        }
    )
    canonical = normalize_export(data, altitude_field="geoaltitude")
    assert canonical["timestamp"][0] == "2025-01-01T00:00:00Z"
    assert canonical["aircraft_id"][0] == "abc"
    with pytest.raises(ValueError):
        normalize_export(data, altitude_field="auto")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"c": float("nan")},
        {"fine_step_m": 0},
        {"offsets_s": []},
        {"f0_bounds": (200, 100)},
        {"method": "unknown"},
    ],
)
def test_bad_search_configuration(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)


def test_doppler_aliasing_rejected():
    flight = AnalyticFlight(np.array([-6000.0, 100.0, 1000.0]), np.array([250.0, 0.0, 0.0]), 500.0)
    with pytest.raises(ValueError, match="Nyquist"):
        synthesize([flight], np.zeros(3), sample_rate=4000)


def test_short_audio_rejected():
    with pytest.raises(ValueError):
        spectrogram([1, 2, 3], 4000)

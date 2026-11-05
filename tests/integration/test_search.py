import numpy as np
import pytest

from doppler_geo.physics.model import predict
from doppler_geo.scoring.curves import AcousticEvent
from doppler_geo.search.engine import SearchArea, SearchConfig, locate
from doppler_geo.synthetic import example_scenario


def observations(offset=0):
    observer, analytic = example_scenario()
    flights = [f.trajectory() for f in analytic]
    t = np.arange(20.0, 105.0, 2.0)
    events = [
        AcousticEvent(t, predict(f, observer, t + offset) * a.f0_hz)
        for f, a in zip(flights, analytic, strict=True)
    ]
    return observer, flights, events


def test_shared_timing_and_location():
    observer, flights, events = observations(3)
    result = locate(
        events,
        flights,
        SearchArea(-500, 500, -500, 500, True),
        SearchConfig(
            coarse_step_m=250,
            fine_step_m=25,
            offsets_s=(-3.0, 0.0, 3.0, 6.0),
            max_evaluations=10000,
        ),
    )
    best = result["candidates"][0]
    assert best["offset_s"] == 3
    assert np.hypot(best["x_m"] - observer[0], best["y_m"] - observer[1]) < 40
    assert result["complete"]


def test_authorization_and_budget():
    with pytest.raises(ValueError, match="authorize"):
        SearchArea(-1, 1, -1, 1)
    with pytest.raises(ValueError, match="oversized"):
        SearchArea(-1e6, 1e6, -1e6, 1e6, True)
    _, flights, events = observations()
    with pytest.raises(ValueError, match="budget"):
        locate(
            events,
            flights,
            SearchArea(-1000, 1000, -1000, 1000, True),
            SearchConfig(coarse_step_m=100, max_evaluations=10),
        )
    result = locate(
        events,
        flights,
        SearchArea(-500, 500, -500, 500, True),
        SearchConfig(coarse_step_m=500, max_evaluations=9),
    )
    assert not result["complete"]
    assert result["evaluations"] == 9


def test_wrong_flight_timing_has_larger_residual():
    observer, flights, events = observations()
    area = SearchArea(-500, 500, -500, 500, True)
    config = SearchConfig(coarse_step_m=500, fine_step_m=100)
    valid = locate(events, flights, area, config)
    events[1].times += 12
    invalid = locate(events, flights, area, config)
    assert invalid["candidates"][0]["loss"] > valid["candidates"][0]["loss"] + 1

import numpy as np

from doppler_geo.physics.model import predict
from doppler_geo.scoring.curves import AcousticEvent
from doppler_geo.search.association import rank_assignments
from doppler_geo.search.engine import SearchArea, SearchConfig
from doppler_geo.search.uncertainty import perturbation_ensemble
from doppler_geo.synthetic import example_scenario


def test_association_and_conditional_sensitivity():
    observer, analytic = example_scenario()
    flights = [a.trajectory(flight_id=str(i)) for i, a in enumerate(analytic)]
    times = np.arange(35.0, 100.0, 2.0)
    events = [
        AcousticEvent(times, predict(f, observer, times) * a.f0_hz)
        for f, a in zip(flights, analytic, strict=True)
    ]
    area = SearchArea(-500, 500, -500, 500, True)
    config = SearchConfig(coarse_step_m=500, fine_step_m=100, top_k=2)
    ranked = rank_assignments(events, flights[::-1], area, config)
    assert ranked["assignments"][0]["flight_ids"] == ["0", "1"]
    assert ranked["exhaustive"]
    sensitivity = perturbation_ensemble(events, flights, area, config, draws=2, seed=4)
    assert len(sensitivity["samples_east_north_offset"]) == 2
    assert "not a calibrated" in sensitivity["interpretation"]


def test_ensemble_total_budget_and_input_validation(monkeypatch):
    import pytest

    observer, sources = example_scenario()
    flights = [f.trajectory() for f in sources]
    times = np.arange(35.0, 100.0, 2.0)
    events = [
        AcousticEvent(times, predict(f, observer, times) * a.f0_hz)
        for f, a in zip(flights, sources, strict=True)
    ]
    area = SearchArea(-500, 500, -500, 500, True)
    config = SearchConfig(coarse_step_m=500, fine_step_m=100, top_k=2)
    # Deadline behavior must not depend on the operating system's timer resolution.
    with monkeypatch.context() as clock_patch:
        ticks = iter([0.0, 2.0, 2.0])
        clock_patch.setattr("doppler_geo.search.uncertainty.monotonic", lambda: next(ticks))
        stopped = perturbation_ensemble(events, flights, area, config, draws=10, total_runtime_s=1)
    assert not stopped["complete"]
    assert stopped["attempted_draws"] == 0
    assert stopped["incomplete_draws"] == 10
    with pytest.raises(ValueError, match="budget"):
        perturbation_ensemble(events, flights, area, config, draws=10, total_evaluations=10)
    with pytest.raises(ValueError, match="finite"):
        perturbation_ensemble(events, flights, area, config, sound_speed_sigma=np.nan)
    ranked = rank_assignments(
        events, flights, area, SearchConfig(coarse_step_m=500, fine_step_m=100, max_evaluations=9)
    )
    assert not ranked["exhaustive"]
    assert ranked["complete_assignments"] == 0

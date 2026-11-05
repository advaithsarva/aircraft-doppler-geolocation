import numpy as np
import pytest

from doppler_geo.experiment import run_experiment
from doppler_geo.physics.model import predict
from doppler_geo.synthetic import AnalyticFlight, synthesize


@pytest.mark.parametrize("held_out,snr", [(False, -5), (True, -5), (False, -15)])
def test_audio_to_location(tmp_path, held_out, snr):
    result = run_experiment(tmp_path, seed=73, snr_db=snr, held_out=held_out, write_plots=False)
    assert result["experiment"]["localization_error_m"] < 150
    assert result["complete"]


def test_acceleration_and_climb_independent_phase_derivative():
    fs = 4000
    flight = AnalyticFlight(
        np.array([-6000.0, 1100.0, 1700.0]),
        np.array([130.0, 20.0, 2.0]),
        170.0,
        np.array([0.3, -0.1, 0.01]),
    )
    observer = np.array([200.0, -300.0, 100.0])
    _, truth = synthesize([flight], observer, duration_s=90, sample_rate=fs, snr_db=20)
    ta = truth[0]["arrival_times"][::100]
    te = truth[0]["emission_times"][::100]
    derivative = np.gradient(te, ta)
    modeled = predict(flight.trajectory(step=0.5), observer, ta)
    np.testing.assert_allclose(modeled[2:-2], derivative[2:-2], atol=1e-4)

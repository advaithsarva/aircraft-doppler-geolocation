import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from doppler_geo.physics.coordinates import LocalFrame
from doppler_geo.physics.model import Trajectory, closest_approach, predict, propagate, sound_speed


def straight(speed=180):
    t = np.arange(-80, 81, 2.0)
    return Trajectory(t, np.column_stack([speed * t, np.full_like(t, 1200), np.full_like(t, 1800)]))


def test_stationary_and_approach_recession():
    t = np.arange(0, 21, 1.0)
    stationary = Trajectory(t, np.tile([1000, 2000, 1000], (len(t), 1)))
    ta, ratio, d = propagate(stationary, [0, 0, 0])
    np.testing.assert_allclose(ratio, 1)
    np.testing.assert_allclose(ta - t, d / 343)
    _, ratio, _ = propagate(straight(), [0, 0, 0])
    assert ratio[0] > 1 and ratio[-1] < 1


@given(st.floats(min_value=50, max_value=300), st.floats(min_value=310, max_value=370))
def test_retarded_derivative_identity(speed, c):
    flight = straight(speed)
    t = np.linspace(-50, 50, 10001)
    ta, ratio, _ = propagate(flight, [0, 0, 0], t, c)
    # f_obs/f0 = dt_e/dt_a; independent differential identity.
    np.testing.assert_allclose(np.gradient(ta, t)[1:-1], 1 / ratio[1:-1], rtol=1e-5)


def test_cpa_and_delayed_cpa():
    result = closest_approach(straight(), np.zeros(3))
    assert abs(result["emission_s"]) < 1e-4
    assert result["distance_m"] == pytest.approx(np.hypot(1200, 1800))
    assert result["arrival_s"] == pytest.approx(result["distance_m"] / 343)


def test_predict_uses_arrival_times():
    flight = straight()
    arrival_cpa = np.hypot(1200, 1800) / 343
    assert predict(flight, [0, 0, 0], [arrival_cpa])[0] == pytest.approx(1, abs=1e-6)
    assert predict(flight, [0, 0, 0], [0])[0] > 1
    assert np.isnan(predict(flight, [0, 0, 0], [-1000])[0])


def test_coordinates_roundtrip():
    frame = LocalFrame(51.5, -0.1, 40)
    points = np.array([[51.51, -0.12, 1500], [51.48, -0.09, 20]])
    enu = frame.to_enu(*points.T)
    np.testing.assert_allclose(frame.to_geodetic(enu), points, atol=1e-6)
    np.testing.assert_allclose(frame.to_enu(51.5, -0.1, 40), [0, 0, 0], atol=1e-6)


def test_invalid_geometry():
    with pytest.raises(ValueError):
        propagate(straight(400), [0, 0, 0])
    with pytest.raises(ValueError):
        Trajectory([0, 1, 50], np.ones((3, 3)))
    with pytest.raises(ValueError):
        sound_speed(-274)
    assert 330 < sound_speed(15) < 350

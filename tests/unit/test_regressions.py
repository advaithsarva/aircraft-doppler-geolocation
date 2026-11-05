"""Regression cases found during post-implementation review."""

import numpy as np
import pytest
import soundfile as sf

from doppler_geo.audio.processing import Spectrogram, load_audio, spectrogram
from doppler_geo.detection.tracking import track_ridge
from doppler_geo.memory import peak_rss_mib
from doppler_geo.physics.coordinates import LocalFrame
from doppler_geo.physics.model import Trajectory, propagate
from doppler_geo.scoring.curves import AcousticEvent, curve_score
from doppler_geo.search.engine import SearchConfig


def stationary():
    return Trajectory(np.arange(0.0, 101.0), np.tile([1000.0, 0.0, 0.0], (101, 1)))


def test_robust_profile_resists_source_frequency_outlier():
    times = np.arange(10.0, 30.0)
    frequencies = np.full(len(times), 180.0)
    frequencies[-1] = 1000
    event = AcousticEvent(times, frequencies)
    gaussian = curve_score(event, stationary(), [0, 0, 0])
    robust = curve_score(event, stationary(), [0, 0, 0], method="robust")
    assert gaussian["f0_hz"] > 200
    assert robust["f0_hz"] == pytest.approx(180, abs=0.2)
    assert (
        curve_score(event, stationary(), [0, 0, 0], method="robust", f0_bounds=(190, 200))["f0_hz"]
        == 190
    )


@pytest.mark.parametrize("harmonic", [0, 1.5, True, float("nan")])
def test_invalid_harmonic(harmonic):
    with pytest.raises(ValueError, match="harmonic"):
        AcousticEvent(np.arange(4.0), np.full(4, 180.0), harmonic=harmonic)


@pytest.mark.parametrize(
    "settings", [{"top_k": 1.5}, {"max_evaluations": True}, {"f0_bounds": (1, np.inf)}]
)
def test_strict_search_budgets(settings):
    with pytest.raises(ValueError):
        SearchConfig(**settings)


def test_spectrogram_rejects_invalid_axes_and_power():
    with pytest.raises(ValueError, match="spectrogram"):
        Spectrogram([0, 0], [100, 200], np.ones((2, 2)))
    with pytest.raises(ValueError, match="spectrogram"):
        Spectrogram([0, 1], [100, 200], -np.ones((2, 2)))


def test_ridge_transition_budget_and_invalid_jump():
    spec = spectrogram(np.sin(2 * np.pi * 180 * np.arange(16000) / 4000), 4000, band=(100, 500))
    with pytest.raises(ValueError, match="budget"):
        track_ridge(spec, (100, 500), max_work_items=5)
    with pytest.raises(ValueError):
        track_ridge(spec, (100, 500), max_jump_hz=0)


def test_trajectory_inputs_are_detached_from_caller():
    times = np.arange(3.0)
    positions = np.tile([1000.0, 0.0, 0.0], (3, 1))
    track = Trajectory(times, positions)
    positions[:] = 0
    times[:] = 0
    np.testing.assert_allclose(track.positions, np.tile([1000.0, 0.0, 0.0], (3, 1)))
    assert track.times.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        track.positions[0] = 0
    with pytest.raises(ValueError, match="Observer"):
        propagate(track, [0, 0])


def test_empty_and_nonfinite_audio_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, [], 4000)
    with pytest.raises(ValueError, match="empty"):
        load_audio(path)
    sf.write(path, [float("nan")] * 4000, 4000, subtype="FLOAT")
    with pytest.raises(ValueError, match="nonfinite"):
        load_audio(path)


def test_platform_memory_and_finite_frame():
    memory = peak_rss_mib()
    assert memory is None or memory > 0
    with pytest.raises(ValueError, match="height"):
        LocalFrame(51.5, -0.1, float("nan"))

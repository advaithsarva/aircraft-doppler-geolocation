import numpy as np
import pytest
import soundfile as sf

from doppler_geo.audio.processing import (
    load_annotations,
    load_audio,
    save_annotations,
    spectrogram,
)
from doppler_geo.detection.tracking import detect_events, track_ridge
from doppler_geo.physics.model import predict
from doppler_geo.scoring.curves import AcousticEvent, curve_score, energy_score
from doppler_geo.synthetic import example_scenario


def test_tone_ridge_and_energy():
    fs = 4000
    t = np.arange(fs * 6) / fs
    samples = np.sin(2 * np.pi * 181.3 * t)
    spec = spectrogram(samples, fs, band=(100, 600))
    event, contrast = track_ridge(spec, (150, 210))
    assert np.median(event.frequencies) == pytest.approx(181.3, abs=0.2)
    assert contrast > 10
    assert detect_events(spec)
    assert energy_score(spec, event.times, np.full_like(event.times, 181.3), (1,)) > (
        energy_score(spec, event.times, np.full_like(event.times, 205.0), (1,))
    )


def test_silence_not_an_event():
    spec = spectrogram(np.zeros(16000), 4000, band=(40, 1000))
    assert detect_events(spec) == []


def test_annotation_and_audio_roundtrip(tmp_path):
    event = AcousticEvent(np.arange(10.0), np.full(10, 180.0))
    path = tmp_path / "events.json"
    save_annotations(path, [event])
    loaded = load_annotations(path)[0]
    np.testing.assert_array_equal(event.frequencies, loaded.frequencies)
    audio_path = tmp_path / "audio.flac"
    sf.write(audio_path, np.zeros(8000), 8000)
    samples, rate = load_audio(audio_path, 4000)
    assert rate == 4000 and len(samples) == 4000
    with pytest.raises(ValueError):
        load_audio(audio_path, max_duration_s=0.5)


def test_harmonic_profile_and_invalid_support():
    observer, analytic = example_scenario()
    flight = analytic[0].trajectory()
    t = np.arange(20, 100, 1.0)
    event = AcousticEvent(t, predict(flight, observer, t) * 360, harmonic=2)
    score = curve_score(event, flight, observer)
    assert score["f0_hz"] == pytest.approx(180)
    assert score["loss"] < 1e-20
    assert curve_score(event, flight, observer, offset_s=10000)["loss"] == float("inf")
    with pytest.raises(ValueError):
        AcousticEvent([0, 1, 2, 3], [100, 100, 100, 100], sigma_hz=-1)


def test_video_audio_extraction(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is optional")
    fs = 8000
    t = np.arange(fs * 2) / fs
    wav = tmp_path / "tone.wav"
    sf.write(wav, np.sin(2 * np.pi * 220 * t), fs)
    video = tmp_path / "tone.mkv"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav), "-c:a", "pcm_s16le", str(video)],
        check=True,
    )
    samples, rate = load_audio(video, target_rate=4000)
    assert rate == 4000
    spectrum = np.abs(np.fft.rfft(samples))
    frequency = np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(spectrum)]
    assert frequency == pytest.approx(220, abs=1)


def test_independent_white_noise_controls():
    for seed in range(200, 205):
        noise = np.random.default_rng(seed).normal(size=4000 * 12)
        spec = spectrogram(noise, 4000, band=(50, 1200))
        assert detect_events(spec) == []

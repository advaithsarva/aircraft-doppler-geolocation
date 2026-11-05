"""Audio I/O, frequency analysis and annotation interchange."""

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft


@dataclass
class Spectrogram:
    times: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray


def load_audio(path, target_rate=8000, max_duration_s=600):
    if target_rate < 1000 or max_duration_s <= 0:
        raise ValueError("Invalid audio limits")
    path = Path(path)
    if path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
        with TemporaryDirectory() as temp:
            output = Path(temp) / "audio.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-t",
                    str(max_duration_s),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(target_rate),
                    str(output),
                ],
                check=True,
                timeout=120,
            )
            return load_audio(output, target_rate, max_duration_s)
    with sf.SoundFile(path) as audio:
        if len(audio) / audio.samplerate > max_duration_s:
            raise ValueError("Recording exceeds configured duration limit")
        source_rate = audio.samplerate
        samples = audio.read(dtype="float64", always_2d=True).mean(axis=1)
    if source_rate != target_rate:
        ratio = Fraction(target_rate, source_rate)
        samples = resample_poly(samples, ratio.numerator, ratio.denominator)
    return samples, target_rate


def spectrogram(samples, sample_rate, n_fft=2048, hop=256, band=(40.0, 1500.0)):
    samples = np.asarray(samples, dtype=float)
    if (
        samples.ndim != 1
        or not np.isfinite(samples).all()
        or len(samples) < n_fft
        or not 0 < hop <= n_fft
        or not 0 <= band[0] < band[1] <= sample_rate / 2
    ):
        raise ValueError("Invalid STFT samples, window, hop or frequency band")
    f, t, z = stft(
        samples, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False
    )
    mask = (f >= band[0]) & (f <= band[1])
    return Spectrogram(t, f[mask], np.abs(z[mask]) ** 2)


def save_annotations(path, events):
    Path(path).write_text(json.dumps([e.to_dict() for e in events], indent=2) + "\n")


def load_annotations(path):
    from doppler_geo.scoring.curves import AcousticEvent

    return [AcousticEvent(**item) for item in json.loads(Path(path).read_text())]

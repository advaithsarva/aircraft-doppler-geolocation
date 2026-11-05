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

from doppler_geo.validation import finite_scalar, positive_integer


@dataclass
class Spectrogram:
    times: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.frequencies = np.asarray(self.frequencies, dtype=float)
        self.power = np.asarray(self.power, dtype=float)
        if (
            self.times.ndim != 1
            or self.frequencies.ndim != 1
            or not len(self.times)
            or not len(self.frequencies)
            or self.power.shape != (len(self.frequencies), len(self.times))
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.frequencies).all()
            or not np.isfinite(self.power).all()
            or np.any(self.power < 0)
            or np.any(np.diff(self.times) <= 0)
            or np.any(np.diff(self.frequencies) <= 0)
        ):
            raise ValueError("Invalid spectrogram axes or power")


def load_audio(path, target_rate=8000, max_duration_s=600):
    positive_integer(target_rate, "target_rate")
    finite_scalar(max_duration_s, "max_duration_s", positive=True)
    if target_rate < 1000:
        raise ValueError("Invalid audio limits")
    path = Path(path)
    if path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
        with TemporaryDirectory() as temp:
            output = Path(temp) / "audio.wav"
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-i",
                        str(path),
                        "-t",
                        str(max_duration_s + 1.0),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        str(target_rate),
                        str(output),
                    ],
                    check=True,
                    timeout=120,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise ValueError("FFmpeg is required to extract audio from video") from exc
            except subprocess.TimeoutExpired as exc:
                raise ValueError("FFmpeg audio extraction exceeded 120 seconds") from exc
            except subprocess.CalledProcessError as exc:
                raise ValueError(
                    f"FFmpeg audio extraction failed: {(exc.stderr or '')[:2000]}"
                ) from exc
            return load_audio(output, target_rate, max_duration_s)
    with sf.SoundFile(path) as audio:
        if len(audio) / audio.samplerate > max_duration_s:
            raise ValueError("Recording exceeds configured duration limit")
        if len(audio) == 0:
            raise ValueError("Recording is empty")
        source_rate = audio.samplerate
        samples = audio.read(dtype="float64", always_2d=True).mean(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError("Recording contains nonfinite samples")
    if source_rate != target_rate:
        ratio = Fraction(target_rate, source_rate)
        samples = resample_poly(samples, ratio.numerator, ratio.denominator)
    return samples, target_rate


def spectrogram(samples, sample_rate, n_fft=2048, hop=256, band=(40.0, 1500.0)):
    positive_integer(sample_rate, "sample_rate")
    positive_integer(n_fft, "n_fft")
    positive_integer(hop, "hop")
    samples = np.asarray(samples, dtype=float)
    if (
        samples.ndim != 1
        or not np.isfinite(samples).all()
        or n_fft < 4
        or len(samples) < n_fft
        or not np.isfinite(band).all()
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

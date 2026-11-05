"""Smooth tonal ridge proposals; proposals are not aircraft classifications."""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from doppler_geo.scoring.curves import AcousticEvent


def track_ridge(spec, band, max_jump_hz=20.0, sigma_hz=3.0, label="tonal candidate"):
    mask = (spec.frequencies >= band[0]) & (spec.frequencies <= band[1])
    f = spec.frequencies[mask]
    if len(f) < 3 or spec.power.shape[1] < 4:
        raise ValueError("Insufficient frequency/time support")
    power = spec.power[mask]
    # Whiten stationary spectral coloration; retain a small floor for silence.
    noise = median_filter(power, size=(15, 1)) + 1e-15
    evidence = np.log1p(power / noise)
    penalty = ((f[:, None] - f[None, :]) / max_jump_hz) ** 2
    penalty[np.abs(f[:, None] - f[None, :]) > max_jump_hz] = np.inf
    score = evidence[:, 0].copy()
    back = np.empty(evidence.shape, dtype=np.int32)
    back[:, 0] = 0
    for j in range(1, evidence.shape[1]):
        options = score[:, None] - penalty
        back[:, j] = np.argmax(options, axis=0)
        score = options[back[:, j], np.arange(len(f))] + evidence[:, j]
    indices = np.empty(evidence.shape[1], dtype=int)
    indices[-1] = np.argmax(score)
    for j in range(len(indices) - 1, 0, -1):
        indices[j - 1] = back[indices[j], j]
    frequencies = f[indices].copy()
    # Sub-bin parabolic interpolation of log power.
    for j, k in enumerate(indices):
        if 0 < k < len(f) - 1:
            a, b, c = np.log(power[k - 1 : k + 2, j] + 1e-20)
            denominator = a - 2 * b + c
            if denominator < -1e-12:
                frequencies[j] += np.clip(0.5 * (a - c) / denominator, -0.5, 0.5) * (f[1] - f[0])
    contrast = power[indices, np.arange(len(indices))] / (np.median(power, axis=0) + 1e-15)
    # This is a heuristic quality indicator, not calibrated aircraft confidence.
    quality = float(np.median(contrast))
    return AcousticEvent(spec.times, frequencies, sigma_hz, label=label), quality


def detect_events(spec, threshold_db=12.0, min_duration_s=2.0):
    """Temporal tonal excess detector; user must inspect proposals and assign flights."""
    excess = 10 * np.log10(
        (np.max(spec.power, axis=0) + 1e-15) / (np.median(spec.power, axis=0) + 1e-15)
    )
    active = excess >= threshold_db
    edges = np.diff(np.r_[False, active, False].astype(int))
    events = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if spec.times[end - 1] - spec.times[start] >= min_duration_s:
            profile = spec.power[:, start:end].mean(axis=1)
            peaks, _ = find_peaks(profile)
            strongest = sorted(peaks, key=lambda k: profile[k], reverse=True)[:5]
            events.append(
                {
                    "start_s": float(spec.times[start]),
                    "end_s": float(spec.times[end - 1]),
                    "tonal_peaks_hz": spec.frequencies[strongest].tolist(),
                    "quality_db": float(np.median(excess[start:end])),
                    "classification": "unverified tonal event",
                }
            )
    return events

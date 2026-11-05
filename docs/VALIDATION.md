# Validation protocol

Run `pytest -q` for physics identities, source-phase consistency, coordinate roundtrips,
CPA/arrival timing, harmonic hypotheses, sample support, archive cleaning, UTC awareness,
vertical datums, authorized boundaries, computation budgets, shared timing and independent
synthetic audio recovery. Hypothesis tests sweep subsonic source and sound speeds.

Run `python benchmarks/run_validation.py --output outputs/validation` for six seeded
end-to-end cases, two geometries, three SNR levels, and diagnostic perturbations. The
held-out geometry changes source velocities, source heights, climb/descent, observer
coordinates and tones. Noise seeds differ from the interactive default. The generator
uses analytic source phase; ground-truth observer positions are not provided to search.

Report:

- 3D localization error vs known synthetic truth, in meters.
- Per-flight frequency-fitting RMS, in Hz.
- Shared timing offset recovery (one controlled timing-error experiment).
- Correct flight permutation for one small synthetic shortlist.
- Wall runtime and process peak resident memory; RSS is Linux ru_maxrss/1024.
- Ten white-noise negative controls for tonal-proposal frequency.

RSS is the process high-water mark, so later cases in one benchmark run inherit earlier
allocation peaks. It includes audio, plotting imports if used, and Python runtime.
This is not a measured 16 GB real-archive scalability guarantee. Report hardware,
Python/library versions, settings, seed and command with results.

The benchmark separately probes missing trajectory samples, a 200 m incorrect altitude,
a shared time offset, and frequency interference. These may increase errors: do not
change test acceptance thresholds to hide them. Perturbation sensitivity is conditional
on the stated noise model. Future operational calibration requires representative
interference-only, wrong-flight, missing-ADS-B and non-aircraft recordings, repeated
across independent authorized data collections and dates.

## First development failures

The first synthetic run incorrectly used narrow frequency bands, clipping high approach
frequencies (about 1.4 km localization error). A subsequent run included pre-arrival noise
as if it were an active tone (up to about 465 m error depending on seed). The corrected
benchmark uses declared broad source-band priors and a conservative common active window,
and regression tests enforce held-out low-SNR recovery. These are explicit demo inputs;
automatic event/flight recognition on arbitrary recordings has not been validated.

DuckDB 2.0 changed Polars/Arrow export APIs; the query now obtains an Arrow table from a
relation and converts with Polars. Pin tested direct versions when reproducing results.

## Limits of the evidence

Synthetic pass results do not establish real-world accuracy. No real audio, historical
ADS-B, atmosphere, terrain coverage or license authorization was supplied. Aircraft
identification accuracy and geolocation false-positive rates are not calibrated by a
single permutation test or white-noise control. Do not describe a heuristic quality
indicator, residual score or candidate radius as a probability or coverage guarantee.

The initial 8 dB tonal-proposal threshold flagged every white-noise control (10/10).
The default was raised to a configurable 12 dB; separate seeds 200–204 form a regression
control, while the published benchmark uses seeds 100–109. This change concerns proposal
specificity only and does not turn tonal detections into aircraft detections.

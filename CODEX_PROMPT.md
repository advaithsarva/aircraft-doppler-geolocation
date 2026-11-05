# Codex Prompt — Aircraft Doppler Acoustic Geolocation Engine

You are a senior research engineer specializing in acoustic signal processing, aerospace physics, computational geometry, geospatial analysis, and high-performance Python.

Your mission is to design and implement an open-source, research-grade Aircraft Doppler Acoustic Geolocation Engine that estimates the geographical origin of an audio recording by matching aircraft Doppler signatures against historical flight trajectories.

The system must reproduce and extend the methodology described below while remaining scientifically rigorous, computationally efficient, modular, and independently testable.

## 1. Core objective

Given:

* An audio or video recording containing faint aircraft acoustic signatures.
* An approximate or exact recording date, if available.
* An optional recording time or time window.
* An optional geographical search region.
* Historical ADS-B aircraft trajectory data.
* Digital elevation models.

Build a system that:

1. Detects aircraft Doppler signatures in audio.
2. Extracts candidate acoustic events and their timing.
3. Retrieves and processes historical flight trajectories.
4. Models three-dimensional sound propagation.
5. Simulates the Doppler signatures expected at hypothetical observer locations.
6. Compares modeled signatures against recorded signatures.
7. Ranks plausible recording locations with uncertainty estimates.
8. Produces reproducible visualizations and scientific diagnostics.

The system should support both single-aircraft and multi-aircraft analysis.

Privacy boundary: Implement this as a consent-based research tool for recordings owned by or explicitly authorized by the operator. Do not implement automatic nationwide searches for the origin of arbitrary third-party recordings, covert phone-call localization, or person-tracking functionality. The initial product must use an operator-specified test area and synthetic or authorized recordings.

## 2. Research before implementation

First, inspect the existing repository.

Identify:

* Existing code, datasets, documentation, and dependencies.
* Reusable infrastructure.
* Current architectural constraints.
* Relevant tests and performance limitations.

Then investigate the mathematical and engineering requirements.

Research:

* Doppler effect for moving acoustic sources.
* Radial velocity and observer-source geometry.
* Retarded-time acoustic propagation.
* Closest point of approach (CPA).
* Delayed CPA and arrival-time modeling.
* Aircraft trajectory interpolation.
* ADS-B data quality and historical coverage.
* Spectrogram analysis and harmonic tracking.
* Digital elevation models.
* Sound-speed variation with atmospheric conditions.
* Efficient spatial indexing.
* Coarse-to-fine numerical optimization.
* Statistical uncertainty and false-positive analysis.

Document the assumptions, equations, and known limitations.

Do not claim that the original video’s implementation details are known beyond the provided description.

## 3. Proposed technology stack

Use Python 3.11+ with:

Scientific computing

* NumPy
* SciPy
* Numba
* Polars
* PyArrow
* DuckDB

Audio analysis

* librosa
* soundfile
* scipy.signal
* FFmpeg

Geospatial

* pyproj
* rasterio
* shapely
* geopandas
* scipy.spatial

Aircraft data

* traffic
* OpenSky-compatible data ingestion where legitimately accessible
* Local CSV and Parquet ingestion

Visualization

* Plotly
* Matplotlib
* Streamlit or another lightweight interface

Testing

* pytest
* hypothesis
* ruff
* mypy

Choose dependencies based on actual requirements. Avoid unnecessary frameworks.

Use CPU-first optimization. Optional GPU acceleration may be added only after profiling establishes a meaningful benefit.

## 4. Audio signal processing

Implement an audio-processing pipeline supporting WAV, FLAC, and common video formats.

Required capabilities:

* Extract audio from video.
* Resample audio.
* Compute high-resolution STFT spectrograms.
* Apply configurable frequency-band analysis.
* Detect faint tonal and harmonic structures.
* Identify candidate Doppler trajectories.
* Estimate fundamental frequencies.
* Estimate acoustic-event timing.
* Assign confidence scores to detected events.

Include both automated detection and manual spectrogram annotation.

Avoid assuming that every curved spectrogram feature originates from an aircraft.

Build synthetic signals with known Doppler trajectories to validate detection accuracy.

## 5. Aircraft trajectory processing

Create a historical ADS-B ingestion layer.

Support:

* Flight identifiers.
* Aircraft identifiers.
* Aircraft type codes.
* Latitude and longitude.
* Barometric or geometric altitude, with explicit datum handling.
* Ground speed.
* Vertical rate.
* Heading or track.
* UTC timestamps.

Implement:

* Data validation.
* Missing-value handling.
* Flight segmentation.
* Trajectory interpolation.
* Aircraft type classification where reliable.
* Duplicate removal.
* Time normalization.
* Spatial filtering.

Do not assume ADS-B records represent every aircraft.

Never silently convert missing aircraft information into a confident classification.

## 6. Three-dimensional acoustic physics

Implement a physically consistent model of aircraft-to-observer sound propagation.

For aircraft position P(t) and observer position O:

```
r(t) = P(t) - O
d(t) = |r(t)|
v_r(t) = -r(t) · V(t) / |r(t)|
f_obs(t_e) = f0 * c / (c - v_r(t_e))
t_a = t_e + d(t_e)/c
```

where t_e is the emission time. Ensure that predicted frequencies are evaluated at the correct arrival times.

Implement numerical interpolation to map emission-time samples into observer-time samples.

Account for:

* Aircraft altitude.
* Observer elevation.
* Slant distance.
* Horizontal motion.
* Vertical motion.
* Changes in heading.
* Acceleration.
* Terrain height.
* Configurable atmospheric sound speed.

Start with straight-line propagation through a homogeneous atmosphere. Treat wind, refraction, multipath, and frequency-dependent attenuation as explicitly documented later extensions.

## 7. Candidate-location search

Implement a hierarchical search restricted to a user-authorized geographical area.

Stage A — Coarse spatial screening

* Generate candidate observer locations.
* Identify aircraft passing within configurable acoustic ranges.
* Calculate CPA and delayed CPA times.
* Reject incompatible aircraft speeds and distances.

Stage B — Temporal consistency

* Match acoustic-event separations against predicted arrival times.
* Apply explicit timing uncertainty.
* Use the absence of additional detectable aircraft only as probabilistic evidence, not absolute proof.

Stage C — Doppler simulation

* Simulate expected Doppler trajectories for remaining candidates.
* Compare them with observed curves.
* Preserve the relative temporal placement of multiple aircraft events.

Stage D — Fine spatial optimization

* Refine promising candidates with smaller spatial steps.
* Use local optimization where appropriate.
* Avoid exhaustively scoring every location at maximum resolution.

Provide configurable search resolutions, candidate budgets, and computational limits.

## 8. Scoring methodology

Implement multiple scoring methods:

1. Spectrogram energy sampled along predicted Doppler trajectories.
2. Distance between annotated and simulated frequency curves.
3. Harmonic-aware curve matching.
4. Joint scoring across multiple aircraft.
5. Probabilistic scoring incorporating timing and frequency uncertainty.

Account for uncertain source fundamental frequency.

For multi-aircraft recordings, allow a shared recording-time alignment while preserving the physical relative arrival times.

Avoid independently shifting each aircraft event to artificially improve the fit.

Return:

* Ranked candidate locations.
* Component scores.
* Overall match score.
* Diagnostic plots.
* Confidence or uncertainty estimates.
* Warnings when evidence is insufficient.

Do not label an arbitrary score as a calibrated probability.

## 9. Performance engineering

The original methodology involves large historical flight datasets and many candidate observer positions.

Optimize through:

* Columnar Parquet storage.
* DuckDB queries.
* Polars lazy execution.
* Spatial bounding-box filtering.
* KD-trees or appropriate spatial indices.
* Vectorized NumPy operations.
* Numba compilation.
* Flight-pair prefiltering.
* Chunked processing.
* Intermediate caching.
* Coarse-to-fine search.

Benchmark runtime and peak memory consumption.

Avoid loading an entire country’s historical flight archive into RAM.

Build the initial version for a normal 16 GB RAM laptop without requiring a dedicated GPU.

## 10. Interactive research interface

Build a clean scientific dashboard with:

Audio analysis

* Audio waveform.
* Interactive spectrogram.
* Frequency zoom.
* Doppler-event annotation.

Flight visualization

* Aircraft trajectories.
* Time-based playback.
* Aircraft metadata.
* Candidate flight selection.

Geospatial visualization

* User-defined search boundary.
* Candidate-location heatmaps.
* Terrain elevation.
* Predicted acoustic ranges.

Model diagnostics

* Recorded versus simulated Doppler curves.
* Individual aircraft scores.
* Combined scores.
* Timing residuals.
* Frequency residuals.
* Uncertainty plots.

The interface must clearly distinguish observed data, modeled predictions, and inferred results.

## 11. Testing and scientific validation

Create a synthetic validation suite.

Generate known aircraft trajectories and observer locations.

Simulate:

* Straight-line constant-speed flights.
* Aircraft acceleration.
* Heading changes.
* Altitude changes.
* Multiple aircraft.
* Low signal-to-noise ratios.
* Missing ADS-B samples.
* Incorrect altitude assumptions.
* Recording-time uncertainty.
* Spectrogram interference.

Test whether the system recovers the known observer position under different conditions.

Measure:

* Localization error.
* Flight identification accuracy.
* Timing error.
* Frequency-fitting error.
* Runtime.
* Memory usage.
* False-positive rates.

Use independent synthetic test cases rather than optimizing and evaluating on identical scenarios.

## 12. Repository architecture

Use a modular structure similar to:

```
aircraft-doppler-geolocation/
├── src/
│   └── doppler_geo/
│       ├── audio/
│       ├── adsb/
│       ├── physics/
│       ├── terrain/
│       ├── detection/
│       ├── search/
│       ├── scoring/
│       ├── optimization/
│       ├── visualization/
│       └── utils/
├── app/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── synthetic/
├── benchmarks/
├── examples/
├── configs/
├── docs/
├── pyproject.toml
└── README.md
```

Adapt this structure if the repository already has an established architecture.

## 13. Implementation phases

Phase 1 — Scientific foundation

Derive and test the acoustic equations, coordinate transformations, CPA calculations, and retarded-time model.

Phase 2 — Synthetic Doppler engine

Generate synthetic flight audio and spectrograms, then verify the model against known trajectories.

Phase 3 — ADS-B ingestion

Implement historical flight data loading, cleaning, interpolation, and spatial indexing.

Phase 4 — Signal extraction

Implement spectrogram processing, manual annotation, and automated candidate Doppler detection.

Phase 5 — Constrained geolocation

Build a coarse-to-fine search over a bounded, authorized test area.

Phase 6 — Multi-aircraft matching

Implement physically locked temporal alignment and combined scoring.

Phase 7 — Interface and optimization

Build the dashboard, benchmark the pipeline, and optimize bottlenecks.

Phase 8 — Validation and documentation

Run reproducible synthetic experiments and authorized real-world tests.

## 14. Development rules

* Do not fabricate experimental results.
* Do not invent ADS-B records or claim missing historical data is available.
* Do not confuse barometric altitude with ground-relative altitude.
* Do not ignore sound propagation delay.
* Do not hardcode the correct observer location into the optimizer.
* Do not treat high spectrogram brightness as proof of a match.
* Do not hide failed tests.
* Do not introduce unnecessary paid APIs.
* Prefer reproducible, open-source components.
* Keep all computational assumptions configurable.
* Provide meaningful logging and progress reporting.
* Never delete existing repository work without explicit authorization.

## 15. Immediate instructions

Start by inspecting the repository and producing:

1. A technical architecture document.
2. A mathematical specification of the Doppler and propagation model.
3. A dependency and data-source plan.
4. A phased implementation checklist.
5. A minimal synthetic validation experiment.

Then implement the first working vertical slice:

Synthetic aircraft trajectory → simulated acoustic Doppler signal → spectrogram → candidate-location scoring → recovery of the known observer location.

Execute the tests and report measured results.

After completing the vertical slice, continue through the implementation phases, verifying each subsystem before moving to the next.

Prioritize correctness, reproducibility, and performance over impressive-looking but unvalidated features.

The final result should be a functioning scientific research system, not a mock dashboard or a collection of disconnected scripts.

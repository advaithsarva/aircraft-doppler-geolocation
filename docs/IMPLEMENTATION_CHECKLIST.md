# Phased implementation and validation status

This checklist separates working code from remaining research. No phase implies validation
on actual historical flight/audio data: none was supplied with authorization.

## Phase 1 — Scientific foundation
- [x] Explicit SI units, geocentric ENU transforms and roundtrip tests.
- [x] Slant range, radial velocity, Doppler sign, delayed CPA and arrival-time interpolation.
- [x] Stationary, approach/recession, derivative identity and subsonic-domain tests.
- [x] Explicit homogeneous-air and vertical-datum assumptions.

## Phase 2 — Synthetic engine and first vertical slice
- [x] Independent retarded-phase audio generation, harmonics and controlled SNR.
- [x] Audio -> STFT -> observed ridge -> location scores -> known-location recovery.
- [x] Independent held-out geometry and noise seed, climb/descent and acceleration checks.
- [x] Machine-readable runtime/RSS/errors and interactive diagnostic report.

## Phase 3 — Local ADS-B ingestion
- [x] CSV/Parquet, UTC awareness, duplicate conflicts, segmentation and interpolation.
- [x] Explicit altitude conversion, bounded lazy filter, DuckDB window queries, KD-tree.
- [x] Normalize legitimate operator-provided OpenSky exports.
- [ ] Validate on a licensed historical archive with documented provider height datum.
- [ ] Add reliable, licensed aircraft-type metadata classification if data is supplied.

## Phase 4 — Signal extraction
- [x] WAV/FLAC/video extraction, resampling, configurable STFT, tonal event proposals.
- [x] Dynamic-programming ridge tracking, frequency estimates and heuristic quality.
- [x] Interactive spectrogram, editable annotations, JSON roundtrip.
- [ ] Calibrate aircraft-vs-interference classification on labeled authorized recordings.
- [ ] Automatic multi-harmonic grouping and time-varying engine-tone models.

## Phase 5 — Constrained geolocation
- [x] Required authorization and explicit local bounds; range screening and budgets.
- [x] Shared timing grid, joint curve scores, hierarchical refinement and ranked candidates.
- [x] Optional DEM height, residual warnings, robust/energy/harmonic diagnostics.
- [ ] Atmospheric ray tracing, terrain occlusion and a formal delayed-CPA timing likelihood.

## Phase 6 — Multi-aircraft
- [x] Physically shared recording alignment; no independent per-event shifts.
- [x] Bounded local flight assignment ranking and wrong-timing regression test.
- [x] Conditional perturbation ensemble, ambiguity warnings.
- [ ] Calibrated detectability model before using missing flights as negative evidence.
- [ ] Calibrated coverage and false-positive rates on representative negative recordings.

## Phase 7 — Interface and performance
- [x] Streamlit synthetic, annotation and actual local-data search workflows.
- [x] Waveform, zoomable spectrogram, candidate scores, observed/modeled residuals.
- [x] Parquet filtering and explicit CPU/memory/computation budgets, benchmark runner.
- [ ] GPU/Numba optimization only if a measured workload justifies it.

## Phase 8 — Validation and documentation
- [x] Architecture, mathematical specification, data plan, CLI examples, CI.
- [x] Independent synthetic recovery; wrong timing, low SNR and invalid-data tests.
- [x] Benchmark suite includes model error, missing samples and interference diagnostics.
- [ ] Authorized real-world tests and external scientific replication.

The implemented system is a usable research prototype. The unchecked items are material
limits and future work, not completed functionality hidden behind UI controls.

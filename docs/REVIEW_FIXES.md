# Post-implementation review and fixes

The initial local suite passed 36 tests, but remote Python 3.12 CI failed on a NumPy
2.5 typing change in CPA array indices. The follow-up was reproduced locally with
NumPy 2.5.4 and SciPy 1.18.1. CPA indices are now explicitly integral, and CI runs
independent Linux 3.11/3.12 and Windows 3.12 jobs without matrix fail-fast cancellation.

The review fixed these substantive issues:

- Robust scoring now profiles source frequency using the same pseudo-Huber objective
  used to rank candidates. Previously outliers biased the least-squares frequency fit.
- Unix-only `resource` imports no longer prevent Windows app/CLI startup. Peak RSS has
  explicit Linux, macOS and Windows handling, with unavailable measurements reported as such.
- Video decoding rejects duration-limit overflow instead of silently truncating the
  recording. Missing FFmpeg, decode failures and timeout errors give clear messages.
- Audio rejects empty/nonfinite samples. Spectrogram axes/power and integer budget/harmonic
  parameters are validated at their public boundaries.
- Ridge transitions use a bounded predecessor band instead of dense all-frequency
  comparisons, with an explicit work budget and validated jump constraints.
- Split flight segments have stable selectable IDs and ingestion exposes available IDs.
  FlightIndex also works correctly with generator input, not just lists.
- Both ingestion engines enforce aware UTC and valid spatial bounds. Altitude offsets
  cannot be nonfinite, and interpolation arrays are detached from caller mutation.
- Search config paths resolve relative to their config file. Output paths cannot overwrite
  input configs, annotations, trajectory files or DEMs. Provenance fingerprints flight inputs.
- Coordinate transformers are cached on their frame, avoiding repeated CRS construction
  for terrain sampling while validating frame height.
- Uncertainty ensembles have total runtime/evaluation budgets, record invalid/incomplete
  draws, and reuse DEM elevation before closing the raster.
- Assignment ranking no longer calls truncated location searches exhaustive.
- Windows download/setup instructions target `C:\Projects\aircraft-doppler-geolocation`.

## Verification

Local follow-up verification: **53 tests passed in 15.68 s**; Ruff lint/format and a
non-incremental mypy run passed with NumPy 2.5.4/SciPy 1.18.1. One external anyio pytest
plugin deprecation warning remains; it does not affect results. Separate Windows CI must
confirm Windows execution; Linux tests alone are not evidence of Windows compatibility.

Re-running all six independent audio cases retained 6.25–12.5 m synthetic localization
error. The interference case improved from 31.25 m to 8.84 m with consistent robust tone
profiling. These remain controlled synthetic cases, not calibrated real-world accuracy.
See [follow-up measurements](../benchmarks/review-results.json) and
[original measurements](../benchmarks/measured-results.json). The original measured
record is preserved, rather than overwritten with new results.
